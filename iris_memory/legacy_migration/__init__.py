"""
旧版（v2.x）数据自动迁移入口

在插件启动时（组件初始化之后）执行一次性的旧数据迁移：
1. KV 标志 ``legacy:migration_done`` 存在则整体跳过（幂等）
2. 检测旧数据（chroma/ 目录、knowledge_graph.db、旧 KV 键、旧配置键）
3. 迁移前把旧数据整体复制到 ``<data_dir>/legacy_backup/``
   （只复制，永不删除原始数据；备份失败则中止迁移）
4. 各迁移器独立执行、故障隔离：L2 记忆 / L3 图谱 / 用户画像 /
   主动回复 KV / 配置键，单项失败记日志继续，不阻断插件启动
5. 完成后写标志并输出汇总日志

本包在 v4 版本将随 main.py 的 LEGACY_MIGRATION_ENABLED 开关一并移除。
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from iris_memory.core import get_logger

from .detector import LegacyDetection, detect_legacy_data

logger = get_logger("legacy_migration")

#: 迁移完成标志 KV 键
MIGRATION_DONE_KEY = "legacy:migration_done"

#: 旧数据备份目录名（相对插件数据目录）
BACKUP_DIRNAME = "legacy_backup"

#: 备份中保存旧 KV/配置快照的文件名
KV_BACKUP_FILENAME = "kv_backup.json"

__all__ = [
    "migrate_if_needed",
    "MIGRATION_DONE_KEY",
    "BACKUP_DIRNAME",
    "LegacyDetection",
    "detect_legacy_data",
]


async def migrate_if_needed(
    context: Any,
    star: Any,
    data_dir: Any,
    component_manager: Any,
) -> None:
    """旧版数据迁移唯一入口（由 main.py 在组件初始化后调用）

    Args:
        context: AstrBot Context
        star: AstrBot Star 插件实例（KV 存储）
        data_dir: 插件数据目录
        component_manager: 组件管理器（持有 L2/L3/画像适配器）

    Note:
        本函数不抛出异常；任何失败只记日志，不阻断插件启动。
    """
    try:
        await _migrate(context, star, Path(data_dir), component_manager)
    except Exception:
        logger.error("旧数据迁移流程异常（不影响插件启动）", exc_info=True)


async def _migrate(
    context: Any,
    star: Any,
    data_dir: Path,
    component_manager: Any,
) -> None:
    del context  # 当前迁移逻辑不使用 context，保留签名兼容

    # ── 1. 幂等检查 ──
    try:
        done = await star.get_kv_data(MIGRATION_DONE_KEY, None)
    except Exception as e:
        logger.warning(f"读取迁移标志失败：{e}，本次跳过迁移（下回启动重试）")
        return
    if done:
        logger.info("旧版数据迁移此前已完成，跳过")
        return

    # ── 2. 检测 ──
    raw_config = _get_raw_user_config()
    detection = await detect_legacy_data(star, data_dir, raw_config)
    if not detection.has_anything:
        logger.info("未检测到旧版（v2.x）数据，无需迁移")
        await _write_done_flag(star, {"status": "no_legacy_data"})
        return

    logger.info(f"检测到旧版数据：{detection.summary_text()}")

    # ── 3. 备份（只复制；失败则中止迁移，不写标志，下回启动重试） ──
    try:
        backup_dir = _backup_legacy_data(data_dir, detection)
        logger.info(f"旧数据已备份到 {backup_dir}（原始数据保留，未做修改）")
    except Exception as e:
        logger.error(
            f"旧数据备份失败，中止本次迁移（原始数据未改动）：{e}", exc_info=True
        )
        return

    # ── 4. 等待后台组件（L2 为后台初始化）就绪 ──
    await _wait_for_components(component_manager)

    # ── 5. 各迁移器独立执行、故障隔离 ──
    from . import config_migrator, kv_migrator, l2_migrator, l3_migrator, profile_migrator

    summary: Dict[str, Any] = {"status": "done", "started_at": datetime.now().isoformat()}
    summary["l2"] = await _run_isolated(
        "L2 记忆", l2_migrator.migrate_l2, detection, component_manager
    )
    summary["l3"] = await _run_isolated(
        "L3 图谱", l3_migrator.migrate_l3, detection, component_manager
    )
    summary["profile"] = await _run_isolated(
        "用户画像", profile_migrator.migrate_profiles, detection, component_manager
    )
    summary["kv"] = await _run_isolated(
        "主动回复KV", kv_migrator.migrate_kv, detection, star
    )
    summary["config"] = await _run_isolated(
        "配置", config_migrator.migrate_config, raw_config
    )

    # ── 6. 写标志 + 汇总 ──
    summary["finished_at"] = datetime.now().isoformat()
    summary["backup_dir"] = str(backup_dir)
    await _write_done_flag(star, summary)
    logger.info(
        "旧版数据迁移流程完成："
        f"L2={summary['l2'].get('status')}({summary['l2'].get('imported', 0)} 条)，"
        f"L3={summary['l3'].get('status')}"
        f"(节点 {summary['l3'].get('nodes_imported', 0)}/边 {summary['l3'].get('edges_imported', 0)})，"
        f"画像={summary['profile'].get('status')}({summary['profile'].get('imported', 0)} 个)，"
        f"KV={summary['kv'].get('status')}，配置={summary['config'].get('status')}。"
        f"备份位于 {backup_dir}。如需重跑，请删除 KV 键 {MIGRATION_DONE_KEY} 后重启插件"
    )


async def _run_isolated(
    name: str, func: Callable[..., Awaitable[Dict[str, Any]]], *args: Any
) -> Dict[str, Any]:
    """故障隔离地执行单个迁移器：异常只记日志，不影响其他迁移器

    同时兼容同步与异步迁移器函数。
    """
    import inspect

    try:
        result = func(*args)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, dict) else {"status": "ok"}
    except Exception as e:
        logger.error(f"{name}迁移器异常（已隔离，不影响其他迁移）：{e}", exc_info=True)
        return {"status": "error", "error": str(e)}


async def _write_done_flag(star: Any, payload: Dict[str, Any]) -> None:
    """写入迁移完成标志"""
    try:
        await star.put_kv_data(MIGRATION_DONE_KEY, payload)
    except Exception as e:
        logger.warning(f"写入迁移完成标志失败：{e}（下回启动可能重复检测）")


def _get_raw_user_config() -> Optional[dict]:
    """获取 AstrBot 用户配置原始字典（用于旧配置键检测与迁移）"""
    try:
        from iris_memory.config import get_config

        raw = getattr(get_config(), "_user_config", None)
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


async def _wait_for_components(component_manager: Any) -> None:
    """等待后台初始化组件（如 L2）就绪，超时后由迁移器自行判可用性"""
    wait = getattr(component_manager, "wait_for_background_init", None)
    if not callable(wait):
        return
    try:
        await wait(timeout=90)
    except Exception as e:
        logger.debug(f"等待后台组件初始化异常（继续迁移流程）：{e}")


def _backup_legacy_data(data_dir: Path, detection: LegacyDetection) -> Path:
    """把旧数据整体复制到 legacy_backup/（只复制，永不删除原始数据）

    Returns:
        备份目录路径

    Raises:
        Exception: 备份失败（调用方应中止迁移）
    """
    backup_dir = data_dir / BACKUP_DIRNAME
    backed_up = False

    if detection.chroma_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            detection.chroma_dir, backup_dir / detection.chroma_dir.name,
            dirs_exist_ok=True,
        )
        backed_up = True
        logger.info(f"已备份旧 ChromaDB 目录：{detection.chroma_dir}")

    if detection.kg_db_path is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        # SQLite 的 -wal/-shm 伴生文件一并复制，保证备份可独立打开
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(detection.kg_db_path) + suffix)
            if src.exists():
                shutil.copy2(src, backup_dir / src.name)
        backed_up = True
        logger.info(f"已备份旧知识图谱数据库：{detection.kg_db_path}")

    # 旧 KV 值与旧配置快照一并落盘备份
    if detection.kv_keys or detection.config_keys:
        backup_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "backed_up_at": datetime.now().isoformat(),
            "kv": detection.kv_keys,
            "config_keys": detection.config_keys,
        }
        target = backup_dir / KV_BACKUP_FILENAME
        with open(target, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
        backed_up = True
        logger.info(f"已备份旧 KV/配置快照：{target}")

    if not backed_up:
        # 理论上不会发生（has_anything 为真才进入），兜底建目录
        backup_dir.mkdir(parents=True, exist_ok=True)

    return backup_dir

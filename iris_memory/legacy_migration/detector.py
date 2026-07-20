"""
旧版（v2.x）数据检测器

检测旧版插件遗留的数据资产：
- ChromaDB 向量库目录（<data_dir>/chroma/）
- 知识图谱数据库（<data_dir>/knowledge_graph.db）
- 旧版 AstrBot KV 键（v2 persistence_service 使用的键）
- 旧版配置键（仍残留在 AstrBot 用户配置中的 v2 schema 键）

检测结果供各迁移器使用；检测本身只读，不修改任何数据。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris_memory.core import get_logger

logger = get_logger("legacy_migration")

#: 旧版 ChromaDB 数据目录名（相对插件数据目录）
LEGACY_CHROMA_DIRNAME = "chroma"

#: 旧版知识图谱 SQLite 文件名
LEGACY_KG_DB_FILENAME = "knowledge_graph.db"

#: 旧版 KV 键（见 v2 iris_memory/core/constants.py 的 KVStoreKeys）
#: 注：旧冷却管理器（cooldown/）仅内存存储（BoundedDict，重启即重置），无 KV 键。
LEGACY_KV_KEYS = (
    "sessions",
    "lifecycle_state",
    "batch_queues",
    "chat_history",
    "proactive_reply_whitelist",
    "member_identity",
    "user_personas",
    "group_activity",
    "persona_batch_queues",
)


@dataclass
class LegacyDetection:
    """旧数据检测结果

    Attributes:
        chroma_dir: 旧 ChromaDB 目录（存在且非空时设置）
        kg_db_path: 旧知识图谱数据库路径（存在且非空时设置）
        kv_keys: 命中的旧 KV 键 → 值（供迁移器复用，避免二次读取）
        config_keys: 命中的旧配置键（扁平键名，如 "persona.enabled"）
    """

    chroma_dir: Optional[Path] = None
    kg_db_path: Optional[Path] = None
    kv_keys: Dict[str, Any] = field(default_factory=dict)
    config_keys: List[str] = field(default_factory=list)

    @property
    def has_file_data(self) -> bool:
        """是否存在文件型旧数据（chroma 目录 / 图谱数据库）"""
        return self.chroma_dir is not None or self.kg_db_path is not None

    @property
    def has_anything(self) -> bool:
        """是否检测到任何旧数据"""
        return self.has_file_data or bool(self.kv_keys) or bool(self.config_keys)

    def summary_text(self) -> str:
        """人类可读的检测摘要"""
        parts: List[str] = []
        if self.chroma_dir is not None:
            parts.append(f"ChromaDB 目录 {self.chroma_dir}")
        if self.kg_db_path is not None:
            parts.append(f"知识图谱库 {self.kg_db_path}")
        if self.kv_keys:
            parts.append(f"旧 KV 键 {sorted(self.kv_keys)}")
        if self.config_keys:
            parts.append(f"旧配置键 {sorted(self.config_keys)}")
        return "；".join(parts) if parts else "无"


def detect_legacy_config_keys(raw_config: Optional[dict]) -> List[str]:
    """检测用户配置中残留的旧版配置键

    Args:
        raw_config: AstrBot 用户配置原始字典（AstrBotConfig 是 dict 子类）

    Returns:
        命中的旧配置扁平键名列表
    """
    if not isinstance(raw_config, dict):
        return []

    # 延迟导入避免循环依赖（detector ← config_migrator 的映射表）
    from .config_migrator import LEGACY_CONFIG_MAPPING

    found: List[str] = []
    for old_key in LEGACY_CONFIG_MAPPING:
        section, _, name = old_key.partition(".")
        section_value = raw_config.get(section)
        if isinstance(section_value, dict) and name in section_value:
            found.append(old_key)
    return found


async def detect_legacy_data(
    star: Any,
    data_dir: Path,
    raw_config: Optional[dict] = None,
) -> LegacyDetection:
    """检测旧版数据

    Args:
        star: AstrBot Star 插件实例（提供 get_kv_data）
        data_dir: 插件数据目录
        raw_config: AstrBot 用户配置原始字典（可选）

    Returns:
        检测结果
    """
    result = LegacyDetection()

    # ── 文件型数据 ──
    try:
        chroma_dir = data_dir / LEGACY_CHROMA_DIRNAME
        if chroma_dir.is_dir() and any(chroma_dir.iterdir()):
            result.chroma_dir = chroma_dir
    except Exception as e:
        logger.warning(f"检测旧 ChromaDB 目录失败：{e}")

    try:
        kg_db = data_dir / LEGACY_KG_DB_FILENAME
        if kg_db.is_file() and kg_db.stat().st_size > 0:
            result.kg_db_path = kg_db
    except Exception as e:
        logger.warning(f"检测旧知识图谱数据库失败：{e}")

    # ── KV 数据 ──
    for key in LEGACY_KV_KEYS:
        try:
            value = await star.get_kv_data(key, None)
        except Exception as e:
            logger.warning(f"检测旧 KV 键 {key} 失败：{e}")
            continue
        # 空容器视为无数据
        if value is None or value == {} or value == []:
            continue
        result.kv_keys[key] = value

    # ── 配置键 ──
    result.config_keys = detect_legacy_config_keys(raw_config)

    return result

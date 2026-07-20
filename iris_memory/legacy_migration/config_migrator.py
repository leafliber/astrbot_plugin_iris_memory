"""
旧版（v2.x）配置迁移器

将旧版配置键保守映射到新版配置键。规则：
- 仅映射语义明确对应的键（见 LEGACY_CONFIG_MAPPING）。
  同名键（如 error_friendly.enable、markdown_stripper.enable，
  新旧 schema 段名一致）不列入：AstrBotConfig 加载新 schema 时
  这些键天然保留，值自动沿用，无需迁移
- 仅当新键当前值仍为默认值时才写入（用户已自定义的不覆盖）
- 逐项记日志
- 优先直写 AstrBot 用户配置（AstrBotConfig 是 dict 子类，
  带 save_config() 持久化）；不可写或持久化失败时，
  退化到 hidden_config.json 记录建议（legacy_config_suggestions）
  并在日志中给出用户可见的手动修改建议

调查结论（astrbot 4.x）：插件配置由 star_manager 以 AstrBotConfig
加载，AstrBotConfig 继承 dict，可写且 save_config() 可持久化到
data/config/<插件名>_config.json。但注意 AstrBotConfig 初始化时
check_config_integrity 会删除新 schema 之外的旧键——因此真实升级
场景下旧键可能已在加载期被清除，检测不到属于正常现象。
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from iris_memory.core import get_logger

logger = get_logger("legacy_migration")

#: hidden_config.json 中记录迁移建议的键
HIDDEN_SUGGESTIONS_KEY = "legacy_config_suggestions"


# ============================================================================
# 值转换函数
# ============================================================================


def _identity(value: Any) -> Any:
    return value


def _embedding_source(value: Any) -> Optional[str]:
    """旧 embedding.source（auto/astrbot/local）→ 新 l2_memory.embedding_source（provider/local）"""
    mapping = {"local": "local", "auto": "provider", "astrbot": "provider"}
    return mapping.get(str(value).strip().lower())


def _bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


#: 旧配置键 → (新配置键, 值转换函数)
#: 仅收录语义明确对应的键；转换返回 None 表示该值不适用（跳过）。
LEGACY_CONFIG_MAPPING: Dict[str, Tuple[str, Callable[[Any], Any]]] = {
    # 嵌入设置
    "embedding.source": ("l2_memory.embedding_source", _embedding_source),
    "embedding.astrbot_provider_id": ("l2_memory.embedding_provider", _str),
    "embedding.local_model": ("l2_memory.embedding_model", _str),
    # 功能开关
    "knowledge_graph.enabled": ("l3_kg.enable", _bool),
    "persona.enabled": ("profile.enable", _bool),
    "proactive_reply.enable": ("proactive.enabled", _bool),
    # LLM Provider 指定
    "llm_providers.knowledge_graph_provider_id": ("l3_kg.extraction_provider", _str),
    "llm_providers.persona_provider_id": ("profile.analysis_provider", _str),
}


# ============================================================================
# 新版默认值解析
# ============================================================================

_new_defaults_cache: Optional[Dict[str, Any]] = None


def _load_new_schema_defaults() -> Dict[str, Any]:
    """从新版 _conf_schema.json 读取各配置键默认值

    新版 Defaults dataclass 不覆盖 proactive/error_friendly 等段，
    而 _conf_schema.json 是用户可见配置的权威来源，故直接解析 schema。

    Returns:
        扁平键名 → 默认值
    """
    global _new_defaults_cache
    if _new_defaults_cache is not None:
        return _new_defaults_cache

    defaults: Dict[str, Any] = {}
    schema_path = Path(__file__).resolve().parents[2] / "_conf_schema.json"
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        for section, body in schema.items():
            if not isinstance(body, dict):
                continue
            items = body.get("items")
            if not isinstance(items, dict):
                continue
            for key, item in items.items():
                if isinstance(item, dict) and "default" in item:
                    defaults[f"{section}.{key}"] = item["default"]
    except Exception as e:
        logger.warning(f"读取新版配置 schema 失败（{schema_path}）：{e}")

    _new_defaults_cache = defaults
    return defaults


# ============================================================================
# 迁移主逻辑
# ============================================================================


def migrate_config(raw_config: Optional[dict]) -> Dict[str, Any]:
    """迁移旧配置键到新配置键

    Args:
        raw_config: AstrBot 用户配置原始字典（AstrBotConfig 是 dict 子类）

    Returns:
        统计信息：{
            status: "ok" | "skipped_no_config" | "hidden_fallback",
            migrated: [(旧键, 新键, 新值), ...],
            skipped_non_default: [新键, ...],
            errors: [旧键, ...],
        }
    """
    stats: Dict[str, Any] = {
        "status": "ok",
        "migrated": [],
        "skipped_non_default": [],
        "errors": [],
    }

    if not isinstance(raw_config, dict):
        stats["status"] = "skipped_no_config"
        logger.info("无法访问 AstrBot 用户配置，跳过配置迁移")
        return stats

    schema_defaults = _load_new_schema_defaults()
    writes: List[Tuple[str, str, Any]] = []

    for old_key, (new_key, transform) in LEGACY_CONFIG_MAPPING.items():
        old_section, _, old_name = old_key.partition(".")
        old_section_value = raw_config.get(old_section)
        if not isinstance(old_section_value, dict) or old_name not in old_section_value:
            continue

        old_value = old_section_value[old_name]
        try:
            new_value = transform(old_value)
        except Exception as e:
            logger.warning(f"配置迁移：{old_key} 值 {old_value!r} 转换失败：{e}")
            stats["errors"].append(old_key)
            continue

        if new_value is None:
            logger.debug(f"配置迁移：{old_key} 值 {old_value!r} 无对应新值，跳过")
            continue
        # 空字符串不迁移（旧版留空 = 未指定，新版默认同为留空）
        if isinstance(new_value, str) and not new_value:
            continue

        default_value = schema_defaults.get(new_key)
        new_section, _, new_name = new_key.partition(".")
        current_section = raw_config.get(new_section)
        current_value = (
            current_section.get(new_name) if isinstance(current_section, dict) else None
        )

        # 用户已自定义新键（当前值与默认值不同）时不覆盖
        if current_value is not None and current_value != default_value:
            logger.info(
                f"配置迁移：新键 {new_key} 已被用户设置为 {current_value!r}，"
                f"不覆盖（旧值 {old_key}={old_value!r}）"
            )
            stats["skipped_non_default"].append(new_key)
            continue

        target_section = raw_config.get(new_section)
        if target_section is None:
            target_section = {}
            raw_config[new_section] = target_section
        if not isinstance(target_section, dict):
            logger.warning(f"配置迁移：新配置段 {new_section} 类型异常，跳过 {new_key}")
            stats["errors"].append(old_key)
            continue

        target_section[new_name] = new_value
        writes.append((old_key, new_key, new_value))
        logger.info(f"配置迁移：{old_key}={old_value!r} → {new_key}={new_value!r}")

    stats["migrated"] = writes

    if not writes:
        return stats

    # 持久化：优先直写 AstrBot 用户配置
    if _try_persist_user_config(raw_config):
        logger.info(f"配置迁移完成，已写入 {len(writes)} 项到 AstrBot 用户配置")
        return stats

    # 退化路径：写入 hidden_config.json 并给出用户可见建议
    stats["status"] = "hidden_fallback"
    _write_hidden_suggestions(writes)
    logger.warning(
        "AstrBot 用户配置不可写或持久化失败，配置迁移建议已写入 "
        f"hidden_config.json（键：{HIDDEN_SUGGESTIONS_KEY}）。请手动在插件配置中修改："
    )
    for old_key, new_key, new_value in writes:
        logger.warning(f"  建议将「{new_key}」设为 {new_value!r}（来自旧配置 {old_key}）")

    return stats


def _try_persist_user_config(raw_config: dict) -> bool:
    """尝试持久化 AstrBot 用户配置

    AstrBotConfig 提供 save_config()；普通 dict（测试替身）视为不可持久化。

    Returns:
        是否持久化成功
    """
    save = getattr(raw_config, "save_config", None)
    if not callable(save):
        return False
    try:
        save()
        return True
    except Exception as e:
        logger.warning(f"调用 save_config() 持久化用户配置失败：{e}")
        return False


def _write_hidden_suggestions(writes: List[Tuple[str, str, Any]]) -> None:
    """将迁移建议写入 hidden_config.json"""
    try:
        from iris_memory.config import get_config

        config = get_config()
    except Exception as e:
        logger.warning(f"配置系统不可用，无法写入隐藏配置建议：{e}")
        return

    try:
        existing = config.get(HIDDEN_SUGGESTIONS_KEY)
        suggestions: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        for old_key, new_key, new_value in writes:
            suggestions[new_key] = {"value": new_value, "from": old_key}
        config.set_hidden(HIDDEN_SUGGESTIONS_KEY, suggestions)
    except Exception as e:
        logger.warning(f"写入隐藏配置建议失败：{e}")

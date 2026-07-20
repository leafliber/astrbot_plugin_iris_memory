"""
旧版（v2.x）用户画像 KV → 新版画像存储迁移器

旧版画像存于 AstrBot KV 键 ``user_personas``（{user_id: UserPersona 序列化字典}），
无群聊/人格隔离维度，统一迁移到 ``default`` 群、``default`` 人格命名空间。

字段映射（UserPersona → UserProfile）：
    user_id               → user_id（缺省回退 KV 外层键）
    display_name          → user_name
    interests (dict 名→分) → interests（按分数降序取前 10 个名称；list 原样保留）
    emotional_baseline    → emotional_baseline
    preferred_reply_style → communication_style（brief→简洁 / detailed→详细，其余丢弃）
    topic_blacklist       → taboo_topics
    （无对应字段）         → favorability 默认 50.0
    其余字段（Big Five、work_*、habits、hourly_distribution、证据链等）
                          → 新版无对应字段，按约定丢弃

已存在的新画像不覆盖（跳过并计数），避免迁移破坏新版已生成的画像。
"""

from typing import Any, Dict, List, Optional

from iris_memory.core import get_logger
from iris_memory.profile.models import UserProfile

logger = get_logger("legacy_migration")

#: 迁移用户画像使用的群聊命名空间（旧画像无群隔离维度）
LEGACY_GROUP_ID = "default"
#: 迁移用户画像使用的人格命名空间
LEGACY_PERSONA_ID = "default"
#: 好感度默认值
DEFAULT_FAVORABILITY = 50.0
#: 兴趣字段最多保留条数
MAX_INTERESTS = 10

#: 旧 preferred_reply_style → 新 communication_style
_REPLY_STYLE_MAP = {
    "brief": "简洁",
    "detailed": "详细",
}


def _map_interests(value: Any) -> List[str]:
    """旧 interests（dict 名称→分数，或 list）→ 新 interests（List[str]）"""
    if isinstance(value, dict):
        sorted_items = sorted(
            value.items(),
            key=lambda kv: (isinstance(kv[1], (int, float)), kv[1] if isinstance(kv[1], (int, float)) else 0.0),
            reverse=True,
        )
        return [str(name) for name, _ in sorted_items[:MAX_INTERESTS] if name]
    if isinstance(value, list):
        return [str(v) for v in value if v][:MAX_INTERESTS]
    return []


def map_legacy_persona(user_id: str, data: Dict[str, Any]) -> UserProfile:
    """旧 UserPersona 序列化字典 → 新 UserProfile"""
    interests = _map_interests(data.get("interests"))

    reply_style = data.get("preferred_reply_style")
    communication_style = _REPLY_STYLE_MAP.get(str(reply_style), "") if reply_style else ""

    emotional_baseline = data.get("emotional_baseline")
    if not isinstance(emotional_baseline, str):
        emotional_baseline = ""

    taboo_topics = data.get("topic_blacklist")
    if not isinstance(taboo_topics, list):
        taboo_topics = []
    taboo_topics = [str(t) for t in taboo_topics if t]

    display_name = data.get("display_name")
    if not isinstance(display_name, str):
        display_name = ""

    return UserProfile(
        user_id=user_id,
        user_name=display_name,
        interests=interests,
        communication_style=communication_style,
        emotional_baseline=emotional_baseline,
        taboo_topics=taboo_topics,
        favorability=DEFAULT_FAVORABILITY,
    )


def _get_available_storage(component_manager: Any) -> Optional[Any]:
    try:
        storage = component_manager.get_component("profile")
    except Exception as e:
        logger.warning(f"获取画像存储组件失败：{e}")
        return None
    if storage is None or not getattr(storage, "is_available", False):
        return None
    return storage


async def migrate_profiles(detection: Any, component_manager: Any) -> Dict[str, Any]:
    """迁移旧用户画像到新版画像存储

    Args:
        detection: LegacyDetection 检测结果（kv_keys 中含 user_personas）
        component_manager: 组件管理器

    Returns:
        统计信息
    """
    stats: Dict[str, Any] = {
        "status": "ok",
        "total": 0,
        "imported": 0,
        "skipped": 0,
        "errors": 0,
    }

    personas = detection.kv_keys.get("user_personas")
    if not personas:
        stats["status"] = "skipped_no_data"
        return stats

    if not isinstance(personas, dict):
        stats["status"] = "error"
        stats["error"] = "user_personas KV 值不是字典"
        logger.warning("旧 user_personas KV 值格式异常（非字典），跳过画像迁移")
        return stats

    storage = _get_available_storage(component_manager)
    if storage is None:
        stats["status"] = "skipped_adapter_unavailable"
        logger.warning("画像存储组件不可用，跳过旧画像迁移")
        return stats

    logger.info(f"开始迁移旧用户画像：共 {len(personas)} 个用户")

    for key, data in personas.items():
        stats["total"] += 1
        try:
            if not isinstance(data, dict):
                stats["skipped"] += 1
                continue

            user_id = str(data.get("user_id") or key)
            if not user_id:
                stats["skipped"] += 1
                continue

            # 已存在的新画像不覆盖
            existing = await storage.get_user_profile(
                user_id, LEGACY_GROUP_ID, LEGACY_PERSONA_ID
            )
            if existing is not None:
                stats["skipped"] += 1
                logger.debug(f"用户 {user_id} 新版画像已存在，跳过")
                continue

            profile = map_legacy_persona(user_id, data)
            await storage.save_user_profile(
                profile,
                group_id=LEGACY_GROUP_ID,
                increment_version=False,
                persona_id=LEGACY_PERSONA_ID,
            )
            stats["imported"] += 1

        except Exception as e:
            stats["errors"] += 1
            logger.warning(f"迁移用户画像失败（{key}）：{e}")

        if stats["total"] % 100 == 0:
            logger.info(
                f"画像迁移进度：{stats['total']}/{len(personas)}"
                f"（已导入 {stats['imported']}）"
            )

    logger.info(
        f"画像迁移完成：共 {stats['total']} 个，导入 {stats['imported']} 个，"
        f"跳过 {stats['skipped']} 个，错误 {stats['errors']} 个"
    )
    return stats

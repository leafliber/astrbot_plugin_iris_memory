"""
旧版（v2.x）主动回复白名单 KV → 新版 iris_reply:* 键迁移器

映射规则：
    proactive_reply_whitelist.group_whitelist (list[str])
        → iris_reply:whitelist (list[str]，与现有值取并集)

说明：
- 旧键值形态：{"group_whitelist": [...], "group_whitelist_mode": bool}
  （v2 proactive/manager.py 的 serialize_whitelist）
- 新版 StateManager 持久化格式：iris_reply:whitelist = 群号列表
  （见 proactive/state.py 的 save_dirty/load_all）
- 旧版 group_whitelist_mode=False 表示不启用白名单（所有群都主动回复）；
  新版以白名单作为主动回复的启用开关（perception.Gatekeeper 仅放行
  白名单群），无等价语义，仅迁移名单本身并记日志提示。
- 旧冷却管理器（cooldown/）为纯内存 BoundedDict（重启即重置），
  无持久化状态，无需迁移。
- 其余旧 KV 键（sessions/lifecycle_state/batch_queues/chat_history/
  member_identity/group_activity/persona_batch_queues）为 v2 运行时
  状态，新版无对应物，只检测不迁移。
"""

from typing import Any, Dict, List

from iris_memory.core import get_logger

logger = get_logger("legacy_migration")

#: 旧主动回复白名单 KV 键
LEGACY_WHITELIST_KEY = "proactive_reply_whitelist"
#: 新版主动回复白名单 KV 键（iris_reply:* 命名空间）
NEW_WHITELIST_KEY = "iris_reply:whitelist"

#: 由本迁移器/画像迁移器处理之外的旧 KV 键 → 不迁移的原因
_NOT_MIGRATED_REASONS: Dict[str, str] = {
    "sessions": "v2 会话运行时状态，新版 L1 缓冲无对应格式",
    "lifecycle_state": "v2 生命周期状态，新版无对应物",
    "batch_queues": "v2 捕获批处理队列（瞬态），无迁移价值",
    "chat_history": "v2 聊天历史缓冲（瞬态），无迁移价值",
    "member_identity": "v2 成员身份缓存，新版无对应物",
    "group_activity": "v2 群活跃度统计，新版无对应物",
    "persona_batch_queues": "v2 画像批处理队列（瞬态），无迁移价值",
}


async def migrate_kv(detection: Any, star: Any) -> Dict[str, Any]:
    """迁移旧主动回复相关 KV 到 iris_reply:* 命名空间

    Args:
        detection: LegacyDetection 检测结果
        star: AstrBot Star 插件实例（提供 get/put_kv_data）

    Returns:
        统计信息
    """
    stats: Dict[str, Any] = {
        "status": "ok",
        "whitelist_total": 0,
        "whitelist_migrated": 0,
        "not_migrated": [],
    }

    # ── 主动回复白名单 ──
    whitelist_data = detection.kv_keys.get(LEGACY_WHITELIST_KEY)
    if isinstance(whitelist_data, dict):
        groups: List[str] = [
            str(g) for g in (whitelist_data.get("group_whitelist") or []) if g
        ]
        stats["whitelist_total"] = len(groups)
        mode = bool(whitelist_data.get("group_whitelist_mode"))

        if groups:
            try:
                existing = await star.get_kv_data(NEW_WHITELIST_KEY, None)
                existing_list = (
                    [str(g) for g in existing] if isinstance(existing, list) else []
                )
                merged = sorted(set(existing_list) | set(groups))
                if merged != sorted(existing_list):
                    await star.put_kv_data(NEW_WHITELIST_KEY, merged)
                stats["whitelist_migrated"] = len(set(groups) - set(existing_list))
                logger.info(
                    f"主动回复白名单迁移：旧 {len(groups)} 个群，"
                    f"新增 {stats['whitelist_migrated']} 个，"
                    f"合并后共 {len(merged)} 个"
                )
                if not mode:
                    logger.info(
                        "旧版未启用白名单模式（所有群均可主动回复）；"
                        "新版以白名单作为启用开关，已迁移名单中的群，"
                        "其余群请在 Web UI「主动回复 → 白名单」中按需添加"
                    )
            except Exception as e:
                stats["status"] = "error"
                stats["error"] = str(e)
                logger.error(f"迁移主动回复白名单失败：{e}", exc_info=True)
    elif whitelist_data is not None:
        logger.warning("旧主动回复白名单 KV 值格式异常（非字典），跳过")

    # ── 其余旧 KV 键：只记录不迁移 ──
    handled = {LEGACY_WHITELIST_KEY, "user_personas"}
    for key in detection.kv_keys:
        if key in handled:
            continue
        reason = _NOT_MIGRATED_REASONS.get(key, "新版无对应物")
        stats["not_migrated"].append(key)
        logger.info(f"旧 KV 键 {key} 不迁移：{reason}（原始数据保留，未做修改）")

    return stats

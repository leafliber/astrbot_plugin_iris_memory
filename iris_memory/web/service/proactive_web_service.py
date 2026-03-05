"""Web 主动回复管理服务

封装面向 Web 的主动回复管理功能。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("proactive_web_service")


class ProactiveWebService:
    """Web 端主动回复管理服务"""

    def __init__(self, memory_service: Any) -> None:
        self._memory_service = memory_service

    def _get_proactive_manager(self) -> Optional[Any]:
        """获取主动回复管理器"""
        if not self._memory_service:
            return None
        return getattr(self._memory_service, "proactive_manager", None)

    async def get_status(self) -> Dict[str, Any]:
        """获取主动回复模块状态

        Returns:
            {
                enabled: bool,
                whitelist_mode: bool,
                whitelist: list,
                stats: dict,
                config: dict
            }
        """
        manager = self._get_proactive_manager()
        if not manager:
            return {
                "enabled": False,
                "whitelist_mode": False,
                "whitelist": [],
                "stats": {},
                "config": {},
                "error": "主动回复模块未初始化"
            }

        return {
            "enabled": manager.enabled,
            "whitelist_mode": manager.group_whitelist_mode,
            "whitelist": manager.get_whitelist(),
            "stats": {},
            "config": {}
        }

    async def list_whitelist(self) -> List[str]:
        """获取群聊白名单列表

        Returns:
            群聊 ID 列表
        """
        manager = self._get_proactive_manager()
        if not manager:
            return []
        return manager.get_whitelist()

    async def add_to_whitelist(self, group_id: str) -> Dict[str, Any]:
        """将群聊添加到白名单

        Args:
            group_id: 群聊 ID

        Returns:
            {success: bool, message: str}
        """
        manager = self._get_proactive_manager()
        if not manager:
            return {"success": False, "message": "主动回复模块未初始化"}

        if not manager.group_whitelist_mode:
            return {"success": False, "message": "群聊白名单模式未开启"}

        if not group_id or not group_id.strip():
            return {"success": False, "message": "群聊 ID 不能为空"}

        group_id = str(group_id).strip()
        result = manager.add_group_to_whitelist(group_id)

        if result:
            logger.info(f"Added group {group_id} to proactive reply whitelist via Web UI")
            return {"success": True, "message": f"已添加群聊 {group_id} 到白名单"}
        else:
            return {"success": False, "message": f"群聊 {group_id} 已在白名单中"}

    async def remove_from_whitelist(self, group_id: str) -> Dict[str, Any]:
        """从白名单移除群聊

        Args:
            group_id: 群聊 ID

        Returns:
            {success: bool, message: str}
        """
        manager = self._get_proactive_manager()
        if not manager:
            return {"success": False, "message": "主动回复模块未初始化"}

        if not manager.group_whitelist_mode:
            return {"success": False, "message": "群聊白名单模式未开启"}

        group_id = str(group_id).strip()
        result = manager.remove_group_from_whitelist(group_id)

        if result:
            logger.info(f"Removed group {group_id} from proactive reply whitelist via Web UI")
            return {"success": True, "message": f"已从白名单移除群聊 {group_id}"}
        else:
            return {"success": False, "message": f"群聊 {group_id} 不在白名单中"}

    async def check_whitelist(self, group_id: str) -> Dict[str, Any]:
        """检查群聊是否在白名单中

        Args:
            group_id: 群聊 ID

        Returns:
            {in_whitelist: bool, group_id: str}
        """
        manager = self._get_proactive_manager()
        if not manager:
            return {"in_whitelist": False, "group_id": group_id}

        return {
            "in_whitelist": manager.is_group_in_whitelist(group_id),
            "group_id": group_id
        }

    async def get_stats(self) -> Dict[str, Any]:
        """获取主动回复统计信息

        Returns:
            统计数据
        """
        manager = self._get_proactive_manager()
        if not manager:
            return {
                "replies_sent": 0,
                "replies_skipped": 0,
                "replies_failed": 0,
                "pending_tasks": 0,
                "last_reply_times": 0,
                "daily_counts": {}
            }
        return await manager.get_stats()

    async def get_followup_status(self) -> Dict[str, Any]:
        """获取 FollowUp 跟进状态

        Returns:
            活跃期待列表和统计信息
        """
        manager = self._get_proactive_manager()
        if not manager:
            return {"error": "主动回复模块未初始化", "active_expectations": [], "total": 0}

        planner = getattr(manager, "_followup_planner", None)
        if not planner:
            return {"error": "FollowUp planner 未初始化", "active_expectations": [], "total": 0}

        store = getattr(planner, "_store", None)
        if not store:
            return {"error": "ExpectationStore 未初始化", "active_expectations": [], "total": 0}

        expectations = store.get_all()

        return {
            "active_expectations": [
                {
                    "group_id": e.group_id,
                    "trigger_user": e.trigger_user_id,
                    "followup_count": e.followup_count,
                    "messages_count": len(e.aggregated_messages),
                    "window_end": e.followup_window_end.isoformat(),
                    "short_window_end": (
                        e.short_window_end.isoformat() if e.short_window_end else None
                    ),
                    "created_at": e.created_at.isoformat(),
                }
                for e in expectations
            ],
            "total": len(expectations),
        }

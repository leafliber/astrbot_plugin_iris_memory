"""主动回复管理路由蓝图"""

from __future__ import annotations

from typing import Any

from iris_memory.web.api.response import error_response, success_response
from iris_memory.utils.logger import get_logger

logger = get_logger("route_proactive")


def register_proactive_routes(app: Any, proactive_service: Any) -> None:
    """注册主动回复管理路由

    Args:
        app: Quart 应用实例
        proactive_service: ProactiveWebService 实例
    """
    from quart import request

    @app.route("/api/proactive/status", methods=["GET"])
    async def proactive_status():
        """获取主动回复模块状态"""
        try:
            status = await proactive_service.get_status()
            return success_response(status)
        except Exception as e:
            logger.error(f"Proactive status error: {e}")
            return error_response("获取状态失败", 500)

    @app.route("/api/proactive/whitelist", methods=["GET"])
    async def proactive_whitelist_list():
        """获取群聊白名单列表"""
        try:
            whitelist = await proactive_service.list_whitelist()
            return success_response({"items": whitelist, "total": len(whitelist)})
        except Exception as e:
            logger.error(f"Proactive whitelist list error: {e}")
            return error_response("获取白名单失败", 500)

    @app.route("/api/proactive/whitelist", methods=["POST"])
    async def proactive_whitelist_add():
        """添加群聊到白名单"""
        try:
            data = await request.get_json()
            group_id = data.get("group_id", "")
            if not group_id or not str(group_id).strip():
                return error_response("群聊 ID 不能为空")

            result = await proactive_service.add_to_whitelist(group_id)
            if result["success"]:
                return success_response(result)
            return error_response(result["message"])
        except Exception as e:
            logger.error(f"Proactive whitelist add error: {e}")
            return error_response("添加失败", 500)

    @app.route("/api/proactive/whitelist/<group_id>", methods=["DELETE"])
    async def proactive_whitelist_remove(group_id: str):
        """从白名单移除群聊"""
        try:
            result = await proactive_service.remove_from_whitelist(group_id)
            if result["success"]:
                return success_response(result)
            return error_response(result["message"])
        except Exception as e:
            logger.error(f"Proactive whitelist remove error: {e}")
            return error_response("移除失败", 500)

    @app.route("/api/proactive/whitelist/check", methods=["GET"])
    async def proactive_whitelist_check():
        """检查群聊是否在白名单中"""
        try:
            group_id = request.args.get("group_id", "")
            if not group_id:
                return error_response("缺少 group_id 参数")

            result = await proactive_service.check_whitelist(group_id)
            return success_response(result)
        except Exception as e:
            logger.error(f"Proactive whitelist check error: {e}")
            return error_response("检查失败", 500)

    @app.route("/api/proactive/stats", methods=["GET"])
    async def proactive_stats():
        """获取主动回复统计信息"""
        try:
            stats = await proactive_service.get_stats()
            return success_response(stats)
        except Exception as e:
            logger.error(f"Proactive stats error: {e}")
            return error_response("获取统计失败", 500)

"""用户画像与情感状态路由蓝图"""

from __future__ import annotations

from typing import Any

from iris_memory.web.api.response import error_response, success_response
from iris_memory.web.utils.helpers import safe_int
from iris_memory.utils.logger import get_logger

logger = get_logger("route_personas")


def register_persona_routes(app: Any, persona_service: Any) -> None:
    """注册用户画像与情感状态路由

    Args:
        app: Quart 应用实例
        persona_service: PersonaWebService 实例
    """
    from quart import request

    @app.route("/api/personas", methods=["GET"])
    async def personas_list():
        try:
            personas = await persona_service.list_personas(
                query=request.args.get("q", ""),
                page=safe_int(request.args.get("page"), 1, 1, 10000),
                page_size=safe_int(request.args.get("page_size"), 20, 1, 100),
            )
            return success_response(personas)
        except Exception as e:
            logger.error(f"Personas list error: {e}")
            return error_response("查询失败", 500)

    @app.route("/api/personas/detail", methods=["GET"])
    async def persona_detail():
        try:
            user_id = request.args.get("user_id")
            if not user_id:
                return error_response("缺少 user_id")

            detail = await persona_service.get_persona_detail(user_id)
            if detail:
                return success_response(detail)
            return error_response("用户画像不存在", 404)
        except Exception as e:
            logger.error(f"Persona detail error: {e}")
            return error_response("获取详情失败", 500)

    @app.route("/api/emotions", methods=["GET"])
    async def emotion_state():
        try:
            user_id = request.args.get("user_id")
            group_id = request.args.get("group_id")

            state = await persona_service.get_emotion_state(user_id, group_id)
            return success_response(state)
        except Exception as e:
            logger.error(f"Emotion state error: {e}")
            return error_response("获取情感状态失败", 500)

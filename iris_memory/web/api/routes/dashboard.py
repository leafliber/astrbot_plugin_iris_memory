"""Dashboard 路由蓝图"""

from __future__ import annotations

from typing import Any

from iris_memory.web.api.response import error_response, success_response
from iris_memory.web.utils.helpers import safe_int
from iris_memory.utils.logger import get_logger

logger = get_logger("route_dashboard")


def register_dashboard_routes(app: Any, dashboard_service: Any) -> None:
    """注册仪表盘路由

    Args:
        app: Quart 应用实例
        dashboard_service: DashboardService 实例
    """
    from quart import request

    @app.route("/api/dashboard", methods=["GET"])
    async def dashboard():
        try:
            stats = await dashboard_service.get_dashboard_stats()
            return success_response(stats)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return error_response("获取统计失败", 500)

    @app.route("/api/dashboard/trend", methods=["GET"])
    async def dashboard_trend():
        try:
            days = safe_int(request.args.get("days"), 30, 1, 365)
            trend = await dashboard_service.get_memory_trend(days)
            return success_response(trend)
        except Exception as e:
            logger.error(f"Trend error: {e}")
            return error_response("获取趋势失败", 500)

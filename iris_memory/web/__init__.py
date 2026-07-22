"""
Web 模块 - 通过 AstrBot Plugin Pages 提供管理界面

架构：
- 后端：通过 context.register_web_api() 注册 API
- 前端：Vue.js 3 SPA（构建到 pages/iris/ 目录）
- 认证：由 AstrBot Dashboard 统一处理

使用方式：
    from iris_memory.web import register_all_routes

    register_all_routes(context)
"""

from iris_memory.core import get_logger

logger = get_logger("web")

__all__ = [
    "register_all_routes",
]


def register_all_routes(context) -> None:
    from .routes.memory import register_memory_routes
    from .routes.profile import register_profile_routes
    from .routes.stats import register_stats_routes
    from .routes.data_routes import register_data_routes
    from .routes.manage_routes import register_manage_routes
    from .routes.hidden_config_routes import register_hidden_config_routes
    from .routes.ui_preferences_routes import register_ui_preferences_routes
    from .routes.run_log_routes import register_run_log_routes

    register_memory_routes(context)
    register_profile_routes(context)
    register_stats_routes(context)
    register_data_routes(context)
    register_manage_routes(context)
    register_hidden_config_routes(context)
    register_ui_preferences_routes(context)
    register_run_log_routes(context)

    logger.info("所有 Web API 路由已注册")

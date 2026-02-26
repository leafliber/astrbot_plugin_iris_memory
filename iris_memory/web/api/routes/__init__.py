"""路由蓝图 - 按领域拆分的 API 路由注册函数"""

from __future__ import annotations

from iris_memory.web.api.routes.dashboard import register_dashboard_routes
from iris_memory.web.api.routes.io import register_io_routes
from iris_memory.web.api.routes.kg import register_kg_routes
from iris_memory.web.api.routes.memories import register_memory_routes
from iris_memory.web.api.routes.personas import register_persona_routes

__all__ = [
    "register_dashboard_routes",
    "register_io_routes",
    "register_kg_routes",
    "register_memory_routes",
    "register_persona_routes",
]

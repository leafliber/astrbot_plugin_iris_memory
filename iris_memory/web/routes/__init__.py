from .memory import register_memory_routes
from .profile import register_profile_routes
from .stats import register_stats_routes
from .data_routes import register_data_routes
from .manage_routes import register_manage_routes
from .hidden_config_routes import register_hidden_config_routes

__all__ = [
    "register_memory_routes",
    "register_profile_routes",
    "register_stats_routes",
    "register_data_routes",
    "register_manage_routes",
    "register_hidden_config_routes",
]

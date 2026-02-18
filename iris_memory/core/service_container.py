"""
轻量级依赖注入容器

用于替代模块级全局变量 (``_config_manager``, ``_identity_service`` 等)，
提供线程安全的服务注册与获取，支持热更新时的清理。

使用方式::

    from iris_memory.core.service_container import ServiceContainer

    container = ServiceContainer.instance()
    container.register("config_manager", my_config_mgr)
    mgr = container.get("config_manager")

    # 热更新时
    container.clear()
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class ServiceContainer:
    """轻量级依赖注入容器

    采用单例模式，线程安全。所有服务按名称注册。
    """

    _instance: Optional[ServiceContainer] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._service_lock = threading.Lock()

    # ── 单例 ──

    @classmethod
    def instance(cls) -> ServiceContainer:
        """获取全局唯一容器实例（线程安全、双重检查锁）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（仅用于测试）"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._services.clear()
            cls._instance = None

    # ── 服务注册与获取 ──

    def register(self, name: str, service: Any) -> None:
        """注册一个服务实例

        如果同名服务已存在，会被替换（支持热更新场景）。

        Args:
            name: 服务名称，如 ``"config_manager"``
            service: 服务实例
        """
        with self._service_lock:
            self._services[name] = service

    def get(self, name: str, default: Any = None) -> Any:
        """获取已注册的服务

        Args:
            name: 服务名称
            default: 未找到时的默认值

        Returns:
            服务实例或 *default*
        """
        return self._services.get(name, default)

    def has(self, name: str) -> bool:
        """检查服务是否已注册"""
        return name in self._services

    def unregister(self, name: str) -> Optional[Any]:
        """注销一个服务

        Args:
            name: 服务名称

        Returns:
            被注销的服务实例，如果不存在则返回 ``None``
        """
        with self._service_lock:
            return self._services.pop(name, None)

    def clear(self) -> None:
        """清除所有已注册服务

        在热更新或插件卸载时应调用此方法，避免旧实例残留。
        """
        with self._service_lock:
            self._services.clear()

    @property
    def registered_services(self) -> tuple:
        """列出所有已注册的服务名称（仅用于调试）"""
        return tuple(self._services.keys())

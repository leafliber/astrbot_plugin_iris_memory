"""
Iris Chat Memory - 存储适配器

提供统一的 KV 存储接口，解耦组件与 AstrBot 的直接依赖。
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KVStorage(Protocol):
    """KV 存储接口协议

    定义 KV 存储的基本操作接口。
    AstrBot 的 Star 类实现了这些方法。

    Methods:
        get_kv_data: 获取数据
        put_kv_data: 保存数据
        delete_kv_data: 删除数据
    """

    async def get_kv_data(self, key: str, default: Any) -> Any:
        """获取数据

        Args:
            key: 键名
            default: 默认值（必需）

        Returns:
            存储的值，不存在则返回默认值
        """
        ...

    async def put_kv_data(self, key: str, value: Any) -> None:
        """保存数据

        Args:
            key: 键名
            value: 值
        """
        ...

    async def delete_kv_data(self, key: str) -> None:
        """删除数据

        Args:
            key: 键名
        """
        ...

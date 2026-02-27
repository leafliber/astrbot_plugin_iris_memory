"""
有界字典 - 防止内存无限增长

提供 LRU 淘汰策略的字典封装，适用于 per-user/per-session 状态跟踪。
支持可选的淘汰回调，以便上层服务在条目被移除前执行清理（如持久化）。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar, Optional, Iterator, Callable

KT = TypeVar("KT")
VT = TypeVar("VT")


class BoundedDict(OrderedDict[KT, VT]):
    """带最大容量限制的 OrderedDict（LRU 淘汰策略）。

    当条目数超过 ``max_size`` 时，自动移除最早插入（最少使用）的条目。
    读取和写入操作都会将对应 key 移到末尾（标记为最近使用）。

    用法::

        d = BoundedDict(max_size=1000, on_evict=lambda k, v: save(v))
        d["user_123"] = some_state

    Args:
        max_size: 最大条目数，默认 1000
        on_evict: 淘汰回调函数，签名 ``(key, value) -> None``，在条目被移除时调用
    """

    def __init__(self, max_size: int = 1000, on_evict: Optional[Callable[[KT, VT], None]] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_size = max(1, max_size)
        self._on_evict = on_evict

    @property
    def max_size(self) -> int:
        return self._max_size

    def __setitem__(self, key: KT, value: VT) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        self._evict()

    def __getitem__(self, key: KT) -> VT:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key: KT, default: Optional[VT] = None) -> Optional[VT]:  # type: ignore[override]
        if key in self:
            return self[key]
        return default

    def _evict(self) -> None:
        """移除超出容量的最旧条目，触发淘汰回调。"""
        while len(self) > self._max_size:
            key, value = self.popitem(last=False)
            if self._on_evict is not None:
                try:
                    self._on_evict(key, value)
                except Exception:
                    pass  # 回调异常不应影响字典操作

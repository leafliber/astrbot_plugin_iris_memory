"""
Iris Chat Memory - 隐藏配置管理器

管理隐藏配置的存储、修改和持久化，提供：
- 内存缓存 + 文件持久化
- 线程安全(RLock)
- 观察者模式支持
- 热修改支持
"""

import json
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional
from dataclasses import asdict

from iris_memory.core import get_logger

from .defaults import HiddenConfig


logger = get_logger("config")


class HiddenConfigManager:
    """隐藏配置管理器

    特性：
    - 内存缓存：避免频繁文件读取
    - 文件持久化：运行时自动保存到 data/iris_memory/hidden_config.json
    - 线程安全：使用 RLock 保护读写操作
    - 观察者模式：配置变更时通知订阅者
    - 热修改支持：运行时通过 set() 方法修改配置

    风险说明：
    - 隐藏配置项不在 _conf_schema.json 中定义，不会与用户配置冲突
    - 热修改后立即生效，但可能导致其他模块的缓存不一致，需通过观察者模式处理
    """

    _path: Path
    _defaults: HiddenConfig
    _cache: Dict[str, object]
    _lock: threading.RLock
    _observers: List[Callable[[str, object, object], None]]
    _dirty: bool

    def __init__(self, config_path: Path, defaults: HiddenConfig):
        """初始化隐藏配置管理器

        Args:
            config_path: 配置文件路径(data/iris_memory/hidden_config.json)
            defaults: 默认配置对象
        """
        self._path = config_path
        self._defaults = defaults
        self._cache = {}
        self._lock = threading.RLock()
        self._observers = []
        self._dirty = False

        # 确保目录存在
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self._load()

    def _load(self) -> None:
        """从文件加载隐藏配置到缓存"""
        with self._lock:
            if not self._path.exists():
                # 文件不存在，使用默认值
                logger.debug(f"隐藏配置文件不存在，使用默认值: {self._path}")
                return

            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    self._cache = data
                    logger.info(f"成功加载隐藏配置，共 {len(self._cache)} 项")
                else:
                    logger.warning("隐藏配置文件格式错误，使用默认值")

            except json.JSONDecodeError as e:
                logger.warning(f"隐藏配置文件损坏，使用默认值: {e}")
            except Exception as e:
                logger.error(f"加载隐藏配置失败: {e}")

    def _persist(self) -> None:
        """持久化隐藏配置到文件"""
        try:
            from iris_memory.utils import atomic_write_json

            atomic_write_json(self._path, self._cache, ensure_ascii=False, indent=2)
            logger.debug(f"隐藏配置已持久化到 {self._path}")
        except Exception as e:
            logger.error(f"持久化隐藏配置失败: {e}")

    def _persist_if_dirty(self) -> None:
        """仅在数据脏时持久化，锁外执行 I/O"""
        with self._lock:
            if not self._dirty:
                return
            data = dict(self._cache)
            self._dirty = False

        try:
            from iris_memory.utils import atomic_write_json

            atomic_write_json(self._path, data, ensure_ascii=False, indent=2)
            logger.debug(f"隐藏配置已持久化到 {self._path}")
        except Exception as e:
            logger.error(f"持久化隐藏配置失败: {e}")

    def get(self, key: str) -> Optional[object]:
        """获取隐藏配置值

        Args:
            key: 配置键名(如 "debug_mode")

        Returns:
            配置值，找不到返回 None

        优先级：
            1. 缓存中的值
            2. 默认值
        """
        with self._lock:
            # 优先从缓存获取
            if key in self._cache:
                return self._cache[key]

            # 返回默认值
            return getattr(self._defaults, key, None)

    def set(self, key: str, value: object) -> None:
        """设置隐藏配置(热修改)

        Args:
            key: 配置键名
            value: 配置值

        特性：
            - 线程安全
            - 自动持久化
            - 通知观察者
            - 记录日志

        风险说明：
            - 如果用户在 WebUI 中也设置了同名配置，优先使用用户配置
            - 热修改可能导致其他模块的缓存不一致，需通过观察者模式处理
        """
        with self._lock:
            old_value = self._cache.get(key)

            self._cache[key] = value
            self._dirty = True

            logger.info(f"隐藏配置已修改: {key} = {value} (原值: {old_value})")

            for observer in self._observers:
                try:
                    observer(key, old_value, value)
                except Exception as e:
                    logger.error(f"观察者回调执行失败: {e}")

        self._persist_if_dirty()

    def update(self, updates: Dict[str, object]) -> None:
        """批量更新隐藏配置

        Args:
            updates: 配置字典 {key: value}

        特性：
            - 批量更新时只持久化一次
            - 每个配置项都会触发观察者回调
        """
        with self._lock:
            for key, value in updates.items():
                old_value = self._cache.get(key)
                self._cache[key] = value

                for observer in self._observers:
                    try:
                        observer(key, old_value, value)
                    except Exception as e:
                        logger.error(f"[iris-memory:config] 观察者回调执行失败: {e}")

            self._dirty = True

        self._persist_if_dirty()
        logger.info(f"批量更新隐藏配置，共 {len(updates)} 项")

    def delete(self, key: str) -> bool:
        """删除隐藏配置项

        Args:
            key: 配置键名

        Returns:
            是否删除成功(键不存在返回 False)
        """
        with self._lock:
            if key not in self._cache:
                return False

            old_value = self._cache.pop(key)
            self._dirty = True

            logger.info(f"已删除隐藏配置: {key} (原值: {old_value})")

            for observer in self._observers:
                try:
                    observer(key, old_value, None)
                except Exception as e:
                    logger.error(f"观察者回调执行失败: {e}")

        self._persist_if_dirty()
        return True

    def get_all(self) -> Dict[str, object]:
        """获取所有隐藏配置(缓存 + 默认值)

        Returns:
            配置字典
        """
        with self._lock:
            # 从默认值开始
            result = asdict(self._defaults)
            # 用缓存覆盖
            result.update(self._cache)
            return result

    def add_observer(self, callback: Callable[[str, object, object], None]) -> None:
        """添加配置变更观察者

        Args:
            callback: 回调函数，签名为 (key, old_value, new_value) -> None

        Examples:
            >>> def on_config_change(key: str, old_value: Any, new_value: Any):
            ...     logger.info(f"配置 {key} 已修改: {old_value} -> {new_value}")
            >>> manager.add_observer(on_config_change)
        """
        with self._lock:
            self._observers.append(callback)
            logger.debug(f"已添加配置变更观察者，当前共 {len(self._observers)} 个")

    def remove_observer(self, callback: Callable[[str, object, object], None]) -> bool:
        """移除配置变更观察者

        Args:
            callback: 回调函数

        Returns:
            是否移除成功
        """
        with self._lock:
            try:
                self._observers.remove(callback)
                logger.debug(f"已移除配置变更观察者，当前共 {len(self._observers)} 个")
                return True
            except ValueError:
                return False

    def clear_cache(self) -> None:
        """清空缓存(保留持久化文件)"""
        with self._lock:
            self._cache.clear()
            logger.info("已清空隐藏配置缓存")

    def reset_to_defaults(self) -> None:
        """重置为默认值"""
        with self._lock:
            self._cache.clear()
            self._dirty = True

        self._persist_if_dirty()

        # 通知所有观察者每个键被重置为新值（默认值）。
        # 此前 reset 不触发观察者，下游组件（如 LLM manager 读取
        # temperature）不会收到变更通知，继续用旧缓存值。
        for key, value in self._cache.items():
            for observer in self._observers:
                try:
                    observer(key, None, value)
                except Exception as e:
                    logger.warning(f"配置观察者通知失败（reset {key}）：{e}")

        logger.info("已重置隐藏配置为默认值")

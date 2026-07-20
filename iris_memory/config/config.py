"""
Iris Chat Memory - 配置管理主类

提供统一的配置访问接口，支持：
- 用户配置(AstrBotConfig)
- 隐藏配置(HiddenConfigManager)
- 默认值(Defaults)
- 三层优先级查找
- 扁平化键名访问
- 观察者模式
"""

import threading
from pathlib import Path
from typing import Callable, Dict, Optional

from astrbot.api import AstrBotConfig

from iris_memory.core import get_logger

from .defaults import Defaults
from .hidden_config import HiddenConfigManager

_UNSET = object()


logger = get_logger("config")


class Config:
    """统一配置管理类

    配置优先级：
        - 用户配置项(在 _conf_schema.json 中定义)：用户配置 > 默认值
        - 隐藏配置项(不在 _conf_schema.json 中定义)：隐藏配置 > 默认值
        - 注意：用户配置和隐藏配置不会冲突，因为它们的配置项完全不同

    键名格式：
        - 用户配置：扁平化键名 "l1_buffer.enable"(内部转换为嵌套访问)
        - 隐藏配置：单层键名 "debug_mode"

    风险说明(记录在日志中)：
        - 当用户配置与隐藏配置同时存在时，优先使用用户配置
        - 隐藏配置的热修改不会覆盖用户配置
        - 需要显式清除用户配置才能使用隐藏配置的值
        - 热修改可能导致其他模块的缓存不一致，需通过观察者模式处理

    Attributes:
        data_dir: 插件数据目录路径

    Examples:
        >>> config = get_config()
        >>>
        >>> # 获取配置
        >>> enable_l1 = config.get("l1_buffer.enable")
        >>> debug_mode = config.get("debug_mode")
        >>>
        >>> # 获取数据目录
        >>> data_dir = config.data_dir
        >>> faiss_dir = data_dir / "faiss"
        >>>
        >>> # 热修改隐藏配置
        >>> config.set_hidden("debug_mode", True)
        >>>
        >>> # 订阅配置变更
        >>> config.on_config_change(lambda k, old, new: logger.info(f"{k}: {old} -> {new}"))
    """

    _user_config: AstrBotConfig
    _hidden: HiddenConfigManager
    _defaults: Defaults
    _lock: threading.RLock
    _data_dir: Path

    def __init__(
        self,
        astrbot_config: AstrBotConfig,
        hidden_manager: HiddenConfigManager,
        defaults: Defaults,
        data_dir: Path,
    ):
        """初始化配置管理器

        Args:
            astrbot_config: AstrBot 用户配置对象
            hidden_manager: 隐藏配置管理器
            defaults: 默认配置对象
            data_dir: 插件数据目录
        """
        self._user_config = astrbot_config
        self._hidden = hidden_manager
        self._defaults = defaults
        self._lock = threading.RLock()
        self._data_dir = data_dir

    @property
    def data_dir(self) -> Path:
        """获取插件数据目录

        Returns:
            数据目录路径
        """
        return self._data_dir

    def get(self, flat_key: str, default: object = None) -> object:
        """获取配置值(统一入口)

        Args:
            flat_key: 扁平化键名，支持两种格式：
                - "l1_buffer.enable" (用户配置)
                - "debug_mode" (隐藏配置)
            default: 默认值(当三层都找不到时返回)

        Returns:
            配置值

        优先级：
            - 用户配置项（如 "l1_buffer.enable"）：用户配置 > 默认值
            - 隐藏配置项（如 "debug_mode"）：隐藏配置 > 默认值

        Examples:
            >>> config.get("l1_buffer.enable")      # 用户配置
            True
            >>> config.get("debug_mode")             # 隐藏配置
            False
            >>> config.get("not_exist", "default")   # 不存在的配置
            "default"
        """
        parts = flat_key.split(".")

        if len(parts) >= 2:
            user_value = self._get_user_config(flat_key)
            if user_value is not _UNSET:
                return user_value

        if len(parts) == 1:
            hidden_value = self._hidden.get(parts[0])
            if hidden_value is not None:
                return hidden_value

        default_value = self._defaults.get_by_flat_key(flat_key)
        if default_value is not None:
            return default_value

        return default

    def _get_user_config(self, flat_key: str) -> Optional[object]:
        """从 AstrBotConfig 获取用户配置(支持多层嵌套访问)

        Args:
            flat_key: 扁平化键名，如 "l1_buffer.enable" 或 "l1_buffer.image_parsing.enable"

        Returns:
            配置值，找不到返回 _UNSET
        """
        try:
            parts = flat_key.split(".")
            if len(parts) < 2:
                return _UNSET

            current = self._user_config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return _UNSET
            return current
        except Exception as e:
            logger.warning(f"读取用户配置失败 {flat_key}: {e}")
            return _UNSET

    def set_hidden(self, key: str, value: object) -> None:
        """热修改隐藏配置

        Args:
            key: 配置键名(如 "debug_mode")
            value: 配置值

        特性：
            - 自动持久化
            - 通知观察者
            - 线程安全

        风险说明：
            - 隐藏配置项不在 _conf_schema.json 中定义，不会与用户配置冲突
            - 隐藏配置支持热修改，修改后立即生效并持久化
        """
        self._hidden.set(key, value)

    def update_hidden(self, updates: Dict[str, object]) -> None:
        """批量更新隐藏配置

        Args:
            updates: 配置字典 {key: value}
        """
        self._hidden.update(updates)

    def delete_hidden(self, key: str) -> bool:
        """删除隐藏配置项

        Args:
            key: 配置键名

        Returns:
            是否删除成功
        """
        return self._hidden.delete(key)

    def on_config_change(self, callback: Callable[[str, object, object], None]) -> None:
        """订阅配置变更事件

        Args:
            callback: 回调函数，签名为 (key, old_value, new_value) -> None

        Examples:
            >>> def on_change(key: str, old_value: Any, new_value: Any):
            ...     logger.info(f"配置 {key} 已修改: {old_value} -> {new_value}")
            >>> config.on_config_change(on_change)
        """
        self._hidden.add_observer(callback)

    def remove_config_change_observer(
        self, callback: Callable[[str, object, object], None]
    ) -> bool:
        """移除配置变更观察者

        Args:
            callback: 回调函数

        Returns:
            是否移除成功
        """
        return self._hidden.remove_observer(callback)

    def get_section(self, section: str) -> Dict[str, object]:
        """获取指定配置分组的所有配置值

        Args:
            section: 配置分组名，如 "l1_buffer"

        Returns:
            配置字典(合并用户配置和默认值)

        Examples:
            >>> l1_config = config.get_section("l1_buffer")
            >>> logger.info(l1_config["enable"])
            True
        """
        # 从默认值开始
        result = self._defaults.get_section_defaults(section)

        # 用用户配置覆盖
        if section in self._user_config:
            user_section = self._user_config[section]
            if isinstance(user_section, dict):
                result.update(user_section)

        return result

    def has(self, flat_key: str) -> bool:
        """检查配置项是否存在

        Args:
            flat_key: 扁平化键名

        Returns:
            是否存在
        """
        parts = flat_key.split(".")

        # 检查用户配置
        if len(parts) >= 2:
            if self._get_user_config(flat_key) is not _UNSET:
                return True

        # 检查隐藏配置
        if len(parts) == 1:
            if self._hidden.get(parts[0]) is not None:
                return True

        # 检查默认值
        return self._defaults.get_by_flat_key(flat_key) is not None

    def get_all_hidden(self) -> Dict[str, object]:
        """获取所有隐藏配置

        Returns:
            隐藏配置字典
        """
        return self._hidden.get_all()

    def reset_hidden_to_defaults(self) -> None:
        """重置隐藏配置为默认值"""
        self._hidden.reset_to_defaults()
        logger.info("已重置隐藏配置为默认值")


# ============================================================================
# 全局单例访问函数
# ============================================================================

_config_instance: Optional[Config] = None


def init_config(astrbot_config: AstrBotConfig, data_dir: Path) -> Config:
    """初始化全局配置实例(仅插件启动时调用一次)

    Args:
        astrbot_config: AstrBot 用户配置对象(从插件构造函数获取)
        data_dir: 插件数据目录(从 context.get_data_dir() 获取)

    Returns:
        Config 实例

    Examples:
        >>> from pathlib import Path
        >>> from astrbot.api import Star, StarContext
        >>>
        >>> class MyPlugin(Star):
        ...     def __init__(self, context: StarContext, config: AstrBotConfig):
        ...         super().__init__(context)
        ...         data_dir = Path(context.get_data_dir())
        ...         self.config = init_config(config, data_dir)
    """
    global _config_instance

    if _config_instance is not None:
        logger.warning("配置实例已存在，将重新初始化")

    # 创建默认配置
    defaults = Defaults()

    # 创建隐藏配置管理器
    hidden_path = data_dir / "hidden_config.json"
    hidden_manager = HiddenConfigManager(hidden_path, defaults.hidden)

    # 创建配置实例
    _config_instance = Config(astrbot_config, hidden_manager, defaults, data_dir)

    logger.info("配置系统初始化完成")

    return _config_instance


def get_config() -> Config:
    """获取全局配置实例(其他模块使用此函数)

    Returns:
        Config 实例

    Raises:
        RuntimeError: 配置未初始化

    Examples:
        >>> from iris_memory.config import get_config
        >>>
        >>> config = get_config()
        >>> enable_l1 = config.get("l1_buffer.enable")
    """
    if _config_instance is None:
        raise RuntimeError(
            "Config not initialized. Call init_config() first in plugin __init__."
        )
    return _config_instance


def reset_config() -> None:
    """重置全局配置实例(仅用于测试)

    注意：此函数会清空全局配置实例，仅应在测试环境中使用。
    """
    global _config_instance
    _config_instance = None
    logger.debug("已重置全局配置实例")

"""
ConfigStore — 核心配置存储与访问

提供统一的扁平化 API：
- ``store.get("module.key")``
- ``store.get("module.key", default)``
- 属性访问（向后兼容）
- 热更新 + Pub/Sub 事件广播

线程安全：对内部数据的所有操作均通过锁保护。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from iris_memory.config.backup import ConfigBackup
from iris_memory.config.events import ConfigEventEmitter, config_events
from iris_memory.config.loader import PLUGIN_DATA_FILENAME, ConfigLoader
from iris_memory.config.schema import (
    ALIAS_MAP,
    SCHEMA,
    AccessLevel,
    ConfigField,
)
from iris_memory.config.validators import validate_field

logger = logging.getLogger(__name__)


class ConfigStore:
    """配置存储中心

    持有完整的配置快照（扁平 ``{key: value}``），对外暴露极简 API。

    特性：
    - 两级优先级合并（Level 1 只读 > Level 2 可写 > Schema 默认）
    - 运行时类型校验
    - TTL 缓存（与旧 ConfigManager 兼容）
    - 事件驱动热更新（Pub/Sub）
    - 写入前自动备份
    """

    DEFAULT_CACHE_TTL: float = 10.0

    def __init__(
        self,
        user_config: Any = None,
        plugin_data_path: Optional[Path] = None,
        *,
        cache_ttl: Optional[float] = None,
        events: Optional[ConfigEventEmitter] = None,
    ):
        self._lock = threading.Lock()
        self._user_config = user_config
        self._plugin_data_path = plugin_data_path
        self._cache_ttl = cache_ttl if cache_ttl is not None else self.DEFAULT_CACHE_TTL
        self._events = events or config_events

        # 加载并合并配置
        self._loader = ConfigLoader(user_config, plugin_data_path)
        self._data: Dict[str, Any] = self._loader.load()

        # Debug 输出：记录初始化后所有载入的最终配置
        logger.debug("ConfigStore initialized with final config: %s", json.dumps(self._data, ensure_ascii=False, indent=2))

        # TTL 缓存（兼容旧 ConfigManager 行为）
        self._cache: Dict[str, Tuple[Any, float]] = {}

        # 备份管理器
        self._backup: Optional[ConfigBackup] = None
        if plugin_data_path is not None:
            self._backup = ConfigBackup(
                plugin_data_path / PLUGIN_DATA_FILENAME
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  核心读取 API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（线程安全）

        优先从缓存读取（TTL 内），过期后从 _data 刷新。

        Args:
            key: 配置键，如 ``"basic.enable_memory"``
            default: 显式默认值（覆盖 Schema 默认值）

        Returns:
            配置值
        """
        now = time.monotonic()
        with self._lock:
            # TTL 缓存
            if key in self._cache:
                cached, expire = self._cache[key]
                if now < expire:
                    return cached

            value = self._data.get(key)
            if value is not None:
                self._cache[key] = (value, now + self._cache_ttl)
                return value

            # 从 Schema 获取默认值
            field = SCHEMA.get(key)
            if field is not None:
                val = field.default if default is None else default
                self._cache[key] = (val, now + self._cache_ttl)
                return val

            return default

    def get_typed(self, key: str, tp: type, default: Any = None) -> Any:
        """获取配置值并确保类型"""
        value = self.get(key, default)
        if value is not None and not isinstance(value, tp):
            try:
                return tp(value)
            except (ValueError, TypeError):
                return default
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data or key in SCHEMA

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  属性访问（向后兼容）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __getattr__(self, name: str) -> Any:
        """通过属性名访问配置（向后兼容 ConfigManager 的属性访问模式）"""
        if name.startswith("_"):
            raise AttributeError(name)

        if name == "llm_enhanced_enabled":
            modes = [
                self.sensitivity_mode,
                self.trigger_mode,
                self.emotion_mode,
                self.conflict_mode,
                self.retrieval_mode,
            ]
            return any(mode in ("llm", "hybrid") for mode in modes)

        if name == "persona_llm_provider":
            from iris_memory.core.provider_utils import normalize_provider_id
            provider_id = normalize_provider_id(
                self.get("llm_providers.persona_provider_id", "")
            )
            return provider_id or "default"

        if name == "proactive_mode":
            return self.get("proactive_reply.proactive_mode", "rule")

        if name == "default_persona_id":
            return self.get("persona_isolation.default_persona_id", "default")

        if name == "persona_id_max_length":
            return self.get("persona_isolation.persona_id_max_length", 64)

        if name == "enable_activity_adaptive":
            return self.get("activity_adaptive.enable", True)

        key = ALIAS_MAP.get(name)
        if key is not None:
            value = self.get(key)
            field = SCHEMA.get(key)
            if field and field.normalize_provider:
                from iris_memory.core.provider_utils import normalize_provider_id
                return normalize_provider_id(value)
            return value

        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  写入 API（仅 Level 2 可写配置）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def set(self, key: str, value: Any) -> None:
        """设置配置值（仅限 WRITABLE 级别）

        修改后自动：
        1. 运行时校验
        2. 备份旧配置
        3. 持久化
        4. 广播变更事件

        Raises:
            PermissionError: 尝试修改只读或内部配置
            ValueError: 值校验失败
        """
        field = SCHEMA.get(key)
        if field is None:
            raise KeyError(f"未知配置键: {key}")
        if field.access != AccessLevel.WRITABLE:
            raise PermissionError(
                f"配置 {key} 访问级别为 {field.access.value}，不可修改"
            )

        validated = validate_field(field, value)

        with self._lock:
            old_value = self._data.get(key, field.default)
            if old_value == validated:
                return  # 值未改变
            self._data[key] = validated
            self._cache.pop(key, None)

        # 持久化 + 备份
        self._persist()

        # 广播事件
        self._events.emit(key, old_value, validated)

    def set_batch(self, updates: Dict[str, Any]) -> Dict[str, str]:
        """批量设置配置值

        Returns:
            错误信息字典 ``{key: error_msg}``，空表示全部成功
        """
        changes: Dict[str, Tuple[Any, Any]] = {}
        errors: Dict[str, str] = {}

        with self._lock:
            for key, value in updates.items():
                field = SCHEMA.get(key)
                if field is None:
                    errors[key] = f"未知配置键: {key}"
                    continue
                if field.access != AccessLevel.WRITABLE:
                    errors[key] = f"配置 {key} 不可修改 ({field.access.value})"
                    continue
                try:
                    validated = validate_field(field, value)
                    old = self._data.get(key, field.default)
                    if old != validated:
                        self._data[key] = validated
                        self._cache.pop(key, None)
                        changes[key] = (old, validated)
                except (ValueError, TypeError) as exc:
                    errors[key] = str(exc)

        if changes:
            self._persist()
            self._events.emit_batch(changes)

        return errors

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  WebUI / 外部 API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_all_for_webui(self) -> List[Dict[str, Any]]:
        """返回全量配置列表（附带只读/可写状态标记）

        用于 WebUI 展示。

        Returns:
            ``[{key, value, default, type, description, access, section}, ...]``
        """
        result = []
        for key, field in SCHEMA.items():
            if field.access == AccessLevel.INTERNAL:
                continue
            result.append({
                "key": key,
                "value": self.get(key),
                "default": field.default,
                "type": field.value_type.__name__,
                "description": field.description,
                "access": field.access.value,
                "section": field.section,
                "choices": list(field.choices) if field.choices else None,
                "min_val": field.min_val,
                "max_val": field.max_val,
            })
        return result

    def get_writable_keys(self) -> Set[str]:
        """返回所有可写配置键"""
        return {
            key for key, field in SCHEMA.items()
            if field.access == AccessLevel.WRITABLE
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  缓存 & 热更新
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """清除缓存"""
        with self._lock:
            if key is None:
                self._cache.clear()
            else:
                self._cache.pop(key, None)

    def reload(self) -> None:
        """重新加载所有配置源"""
        new_data = self._loader.load()
        with self._lock:
            self._data = new_data
            self._cache.clear()

    def set_user_config(self, config: Any) -> None:
        """替换用户配置对象并重新加载（用于热更新）"""
        self._user_config = config
        self._loader = ConfigLoader(config, self._plugin_data_path)
        self.reload()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  事件订阅快捷方法
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def on(self, key: str, handler: Any) -> Any:
        """订阅特定配置键变更"""
        return self._events.on(key, handler)

    def on_section(self, section: str, handler: Any) -> Any:
        """订阅 section 变更"""
        return self._events.on_section(section, handler)

    def on_any(self, handler: Any) -> Any:
        """订阅所有变更"""
        return self._events.on_any(handler)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  快照 & 诊断
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def snapshot(self) -> Dict[str, Any]:
        """返回当前配置的完整快照（深拷贝）"""
        import copy
        with self._lock:
            return copy.deepcopy(self._data)

    def diff_from_defaults(self) -> Dict[str, Tuple[Any, Any]]:
        """返回与默认值不同的配置项 ``{key: (current, default)}``"""
        result = {}
        for key, field in SCHEMA.items():
            current = self._data.get(key, field.default)
            if current != field.default:
                result[key] = (current, field.default)
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  人格隔离相关
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_persona_id_for_storage(self, event_persona_id: Optional[str]) -> str:
        """获取用于存储的人格 ID"""
        if event_persona_id and event_persona_id.strip():
            normalized = event_persona_id.strip()
            max_len = self.persona_id_max_length
            if len(normalized) > max_len:
                normalized = normalized[:max_len]
            return normalized
        return self.default_persona_id

    def get_persona_id_for_query(
        self, event_persona_id: Optional[str], module: str = "memory"
    ) -> Optional[str]:
        """获取用于查询的人格 ID"""
        if module == "memory" and not self.memory_query_by_persona:
            return None
        if module == "knowledge_graph" and not self.kg_query_by_persona:
            return None
        if event_persona_id and event_persona_id.strip():
            normalized = event_persona_id.strip()
            max_len = self.persona_id_max_length
            if len(normalized) > max_len:
                return normalized[:max_len]
            return normalized
        return self.default_persona_id

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  内部
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _persist(self) -> None:
        """将 Level 2 配置持久化到 JSON 文件

        仅保存 WRITABLE 级别且与默认值不同的配置项。
        写入前自动备份。
        """
        if self._plugin_data_path is None:
            return

        fp = self._plugin_data_path / PLUGIN_DATA_FILENAME

        # 自动备份
        if self._backup:
            self._backup.backup_before_write()

        # 收集需要持久化的值
        persist_data: Dict[str, Any] = {}
        for key, field in SCHEMA.items():
            if field.access != AccessLevel.WRITABLE:
                continue
            current = self._data.get(key)
            if current is not None and current != field.default:
                persist_data[key] = current

        # 写入
        try:
            self._plugin_data_path.mkdir(parents=True, exist_ok=True)
            fp.write_text(
                json.dumps(persist_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug("配置已持久化: %s (%d 项)", fp, len(persist_data))
        except OSError:
            logger.exception("配置持久化失败: %s", fp)

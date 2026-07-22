"""
Iris Chat Memory - 运行日志管理器

记录插件运行过程中的关键事件，用于排查使用问题：
- llm_call: 每次 LLM 调用（含主动回复决策调用之外的内部调用）
- injection: 每次 LLM 请求的对话注入（L1/L2/L3/画像，含截断详情）
- proactive: 主动回复统一决策与发送结果

特性：
- 每类日志独立环形缓冲，容量由隐藏配置 run_log_max_entries 控制（默认 10）
- 线程安全（RLock），记录失败永不影响主流程
- 长文本字段按 run_log_content_max_chars 截断并保留原始长度
- 可通过隐藏配置 run_log_enabled 整体关闭
"""

import threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from .logger import get_logger

logger = get_logger("run_log")

LOG_TYPES = ("llm_call", "injection", "proactive")

TYPE_LABELS = {
    "llm_call": "LLM 调用",
    "injection": "上下文注入",
    "proactive": "主动回复",
}

_DEFAULT_MAX_ENTRIES = 10
_DEFAULT_CONTENT_MAX_CHARS = 2000


def _read_settings() -> tuple[bool, int, int]:
    """读取运行日志相关隐藏配置，未初始化或异常时返回默认值"""
    try:
        from iris_memory.config import get_config

        config = get_config()
        enabled = bool(config.get("run_log_enabled", True))
        max_entries = config.get("run_log_max_entries", _DEFAULT_MAX_ENTRIES)
        content_max_chars = config.get(
            "run_log_content_max_chars", _DEFAULT_CONTENT_MAX_CHARS
        )
        if not isinstance(max_entries, int) or isinstance(max_entries, bool):
            max_entries = _DEFAULT_MAX_ENTRIES
        if not isinstance(content_max_chars, int) or isinstance(content_max_chars, bool):
            content_max_chars = _DEFAULT_CONTENT_MAX_CHARS
        return enabled, max(1, max_entries), max(100, content_max_chars)
    except Exception:
        return True, _DEFAULT_MAX_ENTRIES, _DEFAULT_CONTENT_MAX_CHARS


def _truncate_value(value: Any, max_chars: int) -> Any:
    """递归截断 detail 中的超长字符串，并记录原始长度"""
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + f"... [截断，原始 {len(value)} 字符]"
        return value
    if isinstance(value, dict):
        return {k: _truncate_value(v, max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate_value(v, max_chars) for v in value]
    return value


class RunLogManager:
    """运行日志管理器（单例）"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, deque] = {t: deque() for t in LOG_TYPES}
        self._next_id = 0

    def record(
        self,
        log_type: str,
        title: str,
        success: bool = True,
        **detail: Any,
    ) -> None:
        """记录一条运行日志

        Args:
            log_type: 日志类型（llm_call / injection / proactive）
            title: 列表展示标题
            success: 是否成功
            **detail: 类型相关的详细字段
        """
        try:
            enabled, max_entries, content_max_chars = _read_settings()
            if not enabled:
                return
            if log_type not in LOG_TYPES:
                logger.warning(f"未知运行日志类型：{log_type}")
                return

            entry = {
                "type": log_type,
                "type_label": TYPE_LABELS[log_type],
                "title": title,
                "success": success,
                "detail": _truncate_value(detail, content_max_chars),
            }

            with self._lock:
                self._next_id += 1
                entry["id"] = self._next_id
                entry["timestamp"] = datetime.now().isoformat(timespec="milliseconds")

                buf = self._entries[log_type]
                if buf.maxlen != max_entries:
                    buf = deque(buf, maxlen=max_entries)
                    self._entries[log_type] = buf
                buf.append(entry)
        except Exception as e:
            logger.debug(f"运行日志记录失败（已忽略）：{e}")

    def get_entries(
        self,
        log_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """获取日志条目（新的在前）

        Args:
            log_type: 按类型过滤，None 表示全部
            limit: 最大返回条数，None 表示不过滤
        """
        with self._lock:
            if log_type and log_type in self._entries:
                entries = list(self._entries[log_type])
            else:
                entries = [e for buf in self._entries.values() for e in buf]
        entries.sort(key=lambda e: e["id"], reverse=True)
        if limit is not None and limit > 0:
            entries = entries[:limit]
        return entries

    def get_counts(self) -> Dict[str, int]:
        """获取各类型当前缓存条数"""
        with self._lock:
            return {t: len(buf) for t, buf in self._entries.items()}

    def clear(self, log_type: Optional[str] = None) -> int:
        """清空日志，返回清除条数"""
        with self._lock:
            if log_type and log_type in self._entries:
                n = len(self._entries[log_type])
                self._entries[log_type].clear()
                return n
            n = sum(len(buf) for buf in self._entries.values())
            for buf in self._entries.values():
                buf.clear()
            return n


_manager: Optional[RunLogManager] = None
_manager_lock = threading.RLock()


def get_run_log_manager() -> RunLogManager:
    """获取运行日志管理器单例"""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = RunLogManager()
        return _manager


def reset_run_log_manager() -> None:
    """重置单例（仅测试使用）"""
    global _manager
    with _manager_lock:
        _manager = None

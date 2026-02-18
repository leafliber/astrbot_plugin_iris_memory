"""
用户画像结构化日志器

为画像全生命周期（更新、持久化、注入、恢复）提供统一的 DEBUG 日志。
所有日志事件均以 ``PERSONA.`` 前缀标识，可通过 ``log_level=DEBUG`` 激活。

用法示例::

    from iris_memory.analysis.persona.logger import persona_log
    persona_log.update_start(user_id, memory_id)
    persona_log.update_applied(user_id, changes)
    persona_log.persist(user_id)
"""

from typing import Any, Dict, List, Optional

from iris_memory.utils.logger import get_logger

_logger = get_logger("persona")


class PersonaLogger:
    """画像结构化日志器 — 统一 DEBUG 输出"""

    # ------------------------------------------------------------------
    # 更新 lifecycle
    # ------------------------------------------------------------------
    def update_start(self, user_id: str, memory_id: Optional[str] = None) -> None:
        _logger.debug(
            f"PERSONA.UPDATE.START user={user_id} mem={memory_id}"
        )

    def update_applied(
        self, user_id: str, changes: List[Dict[str, Any]]
    ) -> None:
        if not changes:
            _logger.debug(f"PERSONA.UPDATE.NOOP user={user_id} (no changes)")
            return
        for c in changes:
            _logger.debug(
                f"PERSONA.UPDATE.CHANGE user={user_id} "
                f"field={c.get('field', c.get('field_name', '?'))} "
                f"old={_trunc(c.get('old', c.get('old_value')))} "
                f"new={_trunc(c.get('new', c.get('new_value')))} "
                f"rule={c.get('rule', c.get('rule_id', ''))} "
                f"conf={c.get('conf', c.get('confidence', ''))}"
            )

    def update_skipped(self, user_id: str, reason: str) -> None:
        _logger.debug(
            f"PERSONA.UPDATE.SKIP user={user_id} reason={reason}"
        )

    def update_error(self, user_id: str, error: Exception) -> None:
        _logger.warning(
            f"PERSONA.UPDATE.ERROR user={user_id} error={error}"
        )

    # ------------------------------------------------------------------
    # 持久化 lifecycle
    # ------------------------------------------------------------------
    def persist_start(self, user_id: str) -> None:
        _logger.debug(f"PERSONA.PERSIST.START user={user_id}")

    def persist_ok(self, user_id: str, update_count: int) -> None:
        _logger.debug(
            f"PERSONA.PERSIST.OK user={user_id} updates={update_count}"
        )

    def persist_error(self, user_id: str, error: Exception) -> None:
        _logger.warning(
            f"PERSONA.PERSIST.ERROR user={user_id} error={error}"
        )

    # ------------------------------------------------------------------
    # 恢复 lifecycle
    # ------------------------------------------------------------------
    def restore_start(self, count: int) -> None:
        _logger.debug(f"PERSONA.RESTORE.START count={count}")

    def restore_ok(self, user_id: str) -> None:
        _logger.debug(f"PERSONA.RESTORE.OK user={user_id}")

    def restore_error(self, user_id: str, error: Exception) -> None:
        _logger.warning(
            f"PERSONA.RESTORE.ERROR user={user_id} error={error}"
        )

    # ------------------------------------------------------------------
    # 注入 lifecycle
    # ------------------------------------------------------------------
    def inject_view(self, user_id: str, view: Dict[str, Any]) -> None:
        _logger.debug(
            f"PERSONA.INJECT user={user_id} keys={list(view.keys())}"
        )

    def inject_skip(self, user_id: str, reason: str) -> None:
        _logger.debug(
            f"PERSONA.INJECT.SKIP user={user_id} reason={reason}"
        )

    # ------------------------------------------------------------------
    # 画像快照（数据级）
    # ------------------------------------------------------------------
    def snapshot(self, user_id: str, persona_dict: Dict[str, Any]) -> None:
        """输出一次画像完整快照（仅在 DEBUG 级别）"""
        _logger.debug(
            f"PERSONA.SNAPSHOT user={user_id} "
            f"update_count={persona_dict.get('update_count', 0)} "
            f"interests={list(persona_dict.get('interests', {}).keys())[:5]} "
            f"emotional_baseline={persona_dict.get('emotional_baseline', 'neutral')} "
            f"trust={persona_dict.get('trust_level', '?')} "
            f"proactive_pref={persona_dict.get('proactive_reply_preference', '?')}"
        )


# ------------------------------------------------------------------
# 内部工具
# ------------------------------------------------------------------

def _trunc(value: Any, max_len: int = 60) -> str:
    s = str(value)
    return s if len(s) <= max_len else s[:max_len] + "..."


# 全局单例
persona_log = PersonaLogger()

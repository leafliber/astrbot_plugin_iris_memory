from __future__ import annotations

from .prompts import display_level, resolve_level
from .state import StateManager


class AdminCommands:
    def __init__(self, state: StateManager) -> None:
        self._state = state

    def get_status(self, group_id: str) -> str:
        return self._state.get_status_text(group_id)

    def reset_group(self, group_id: str) -> str:
        self._state.reset_group(group_id)
        return f"群 {group_id} 状态已重置"

    def set_cooldown(self, group_id: str, minutes: int) -> str:
        actual = self._state.set_cooldown(group_id, minutes)
        return f"群 {group_id} 冷却已设置为 {actual} 分钟"

    def set_willingness(self, group_id: str, raw_level: str) -> str:
        level = resolve_level(raw_level)
        if not level:
            return f"无效的回复意愿等级: {raw_level}，可选: 低/中/高 (low/medium/high)"
        self._state.set_willingness(group_id, level)
        return f"群 {group_id} 回复意愿已设置为: {display_level(level)}"

    def get_willingness(self, group_id: str) -> str:
        level = self._state.get_willingness(group_id)
        return display_level(level)

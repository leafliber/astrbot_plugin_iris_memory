"""
人格 ID 提取工具

从 AstrBot 事件中安全提取 persona_id。
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


def get_event_persona_id(event: "AstrMessageEvent") -> Optional[str]:
    """从事件中提取 persona_id

    兼容 AstrBot v4.5.7+ 的 event.persona 属性。
    如果事件没有 persona 属性或 persona 为 None，返回 None。

    Args:
        event: AstrBot 消息事件

    Returns:
        persona_id 字符串，或 None（表示未指定人格）
    """
    persona = getattr(event, "persona", None)
    if persona is None:
        return None
    persona_id = getattr(persona, "id", None)
    if persona_id and isinstance(persona_id, str) and persona_id.strip():
        return persona_id.strip()
    return None

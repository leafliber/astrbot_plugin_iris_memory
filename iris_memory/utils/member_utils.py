"""
Member identity helpers.
"""

from typing import Optional


def short_member_id(user_id: Optional[str], length: int = 6) -> str:
    """Build a short, stable suffix for member IDs."""
    if not user_id:
        return ""

    text = "".join(ch for ch in str(user_id) if ch.isalnum())
    if not text:
        text = str(user_id)

    if length <= 0:
        return ""

    return text[-length:]


def format_member_tag(sender_name: Optional[str], user_id: Optional[str]) -> str:
    """Format a human-friendly member tag with a stable ID suffix."""
    short_id = short_member_id(user_id)
    name = (sender_name or "").strip()

    if not name and not short_id:
        return ""

    if not name:
        name = "成员"

    if short_id:
        return f"{name}#{short_id}"
    return name

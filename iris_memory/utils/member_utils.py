"""
Member identity helpers.

提供成员标识的基础工具函数。

当 MemberIdentityService 已注册时，``format_member_tag`` 会委托给
服务的 ``resolve_tag_sync``，从而实现名称变更追踪和活跃度统计。
未注册时退回到纯函数逻辑，保持向后兼容。
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from iris_memory.utils.member_identity_service import MemberIdentityService

# 全局服务引用（由 MemoryService 初始化时注入）
_identity_service: Optional["MemberIdentityService"] = None


def set_identity_service(service: Optional["MemberIdentityService"]) -> None:
    """注册全局 MemberIdentityService 实例

    由 MemoryService.initialize() 调用。
    """
    global _identity_service
    _identity_service = service


def get_identity_service() -> Optional["MemberIdentityService"]:
    """获取已注册的 MemberIdentityService 实例"""
    return _identity_service


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


def format_member_tag(
    sender_name: Optional[str],
    user_id: Optional[str],
    group_id: Optional[str] = None,
) -> str:
    """Format a human-friendly member tag with a stable ID suffix.

    如果 MemberIdentityService 已注册，自动委托给它以实现
    名称变更追踪和群成员登记。否则退回到纯函数逻辑。

    Args:
        sender_name: 发送者显示名称
        user_id: 用户唯一ID
        group_id: 群组ID（可选）

    Returns:
        str: 格式为 ``名称#短ID`` 的标签
    """
    if _identity_service is not None and user_id:
        return _identity_service.resolve_tag_sync(user_id, sender_name, group_id)

    # 退回到纯函数逻辑
    short_id = short_member_id(user_id)
    name = (sender_name or "").strip()

    if not name and not short_id:
        return ""

    if not name:
        name = "成员"

    if short_id:
        return f"{name}#{short_id}"
    return name

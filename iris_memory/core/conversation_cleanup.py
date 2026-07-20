"""
对话清理模块

管理 AstrBot 内置对话历史的清理策略，确保上下文完全由本插件的
L1/L2/L3 记忆系统控制，避免内置对话历史与插件记忆系统冲突。

两种清理策略（由隐藏参数 enable_legacy_cleanup 控制）：

1. 默认策略（enable_legacy_cleanup=False，推荐）：
   - 在 on_llm_request 钩子中清空 req.contexts
   - 保留对话 ID，使 AstrBot 主动回复（active_reply）能正常检测到对话存在
   - Agent 完成后对话历史由 AstrBot 正常保存，下次请求时再次清空

2. 旧版策略（enable_legacy_cleanup=True）：
   - 在 on_agent_done 钩子中调用 conversationManager.delete_conversation
   - 直接删除整个对话，下次消息时 AstrBot 自动创建新对话
   - 会导致主动回复因 get_curr_conversation_id 返回 None 而失败

两种策略均受 context_control.enable_conversation_cleanup 配置控制。
"""

from typing import TYPE_CHECKING

from iris_memory.core import get_logger

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import LLMResponse, ProviderRequest
    from astrbot.api.star import Context
    from iris_memory.core.components import ComponentManager

logger = get_logger("conversation_cleanup")


async def handle_pre_request_cleanup(
    event: "AstrMessageEvent",
    req: "ProviderRequest",
    context: "Context",
    component_manager: "ComponentManager",
) -> None:
    """LLM 请求前的对话上下文清理（默认策略）

    在 LLM 请求前清空 req.contexts，确保 Agent 不会使用 AstrBot 内置
    对话历史，上下文完全由本插件的 L1/L2/L3 系统通过
    extra_user_content_parts 注入。

    保留对话 ID 不删除，使 AstrBot 主动回复机制能通过
    get_curr_conversation_id 检测到对话存在。

    仅当 enable_conversation_cleanup=True 且 enable_legacy_cleanup=False 时执行。

    Args:
        event: AstrBot 消息事件对象
        req: LLM 提供者请求对象
        context: AstrBot 插件上下文
        component_manager: 组件管理器实例
    """
    from iris_memory.config import get_config

    config = get_config()

    if not config.get("context_control.enable_conversation_cleanup"):
        return

    if config.get("enable_legacy_cleanup", False):
        return

    try:
        ctx_count = len(req.contexts) if req.contexts else 0
        if ctx_count > 0:
            req.contexts = []
            logger.debug(
                f"已清空会话 {event.unified_msg_origin} 的内置上下文 "
                f"({ctx_count} 条历史消息)，改由 L1/L2/L3 记忆系统提供上下文"
            )
    except Exception as e:
        logger.warning(f"清空内置上下文失败: {e}")


async def handle_agent_done(
    event: "AstrMessageEvent",
    resp: "LLMResponse",
    context: "Context",
    component_manager: "ComponentManager",
) -> None:
    """Agent 运行完成后的对话清理（旧版策略）

    在 Agent 运行完成后，删除 AstrBot 内置对话管理器中的当前对话。
    这是旧版清理策略，会导致主动回复失效，仅作为隐藏参数保留。

    仅当 enable_conversation_cleanup=True 且 enable_legacy_cleanup=True 时执行。

    Args:
        event: AstrBot 消息事件对象
        resp: LLM 响应对象
        context: AstrBot 插件上下文
        component_manager: 组件管理器实例
    """
    from iris_memory.config import get_config

    config = get_config()

    if not config.get("context_control.enable_conversation_cleanup"):
        return

    if not config.get("enable_legacy_cleanup", False):
        return

    conv_mgr = getattr(context, "conversation_manager", None)
    if conv_mgr is None:
        logger.debug("对话管理器不可用，跳过对话清理")
        return

    umo = event.unified_msg_origin
    if not umo:
        logger.debug("无法获取 unified_msg_origin，跳过对话清理")
        return

    try:
        curr_cid = await conv_mgr.get_curr_conversation_id(umo)
        if not curr_cid:
            logger.debug("当前无活跃对话，跳过对话清理")
            return

        await conv_mgr.delete_conversation(umo, curr_cid)
        logger.debug(f"已清理会话 {umo} 的内置对话历史 (cid={curr_cid})")
    except Exception as e:
        logger.warning(f"清理内置对话历史失败: {e}")

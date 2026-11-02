"""
LLM 响应钩子处理模块

负责处理 LLM 响应相关的钩子逻辑。
"""

from typing import TYPE_CHECKING, cast

from ..core import get_logger

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import LLMResponse
    from ..core.components import ComponentManager
    from ..l1_buffer import L1Buffer

logger = get_logger("llm_response_hook")


PENDING_ASSISTANT_EXTRA = "iris_pending_assistant"
# 能给回执的平台：学习模块的对话对采集也等到确认发出后再做（没发出去的回复不学）
PENDING_LEARNING_EXTRA = "iris_pending_learning"


async def handle_llm_response(
    event: "AstrMessageEvent",
    resp: "LLMResponse",
    component_manager: "ComponentManager",
    *,
    defer_assistant: bool = False,
) -> None:
    """处理 LLM 响应钩子

    只添加助手响应，用户消息已在 on_all_message 中添加。

    Args:
        event: AstrBot 消息事件对象
        resp: LLM 响应对象
        component_manager: 组件管理器实例
        defer_assistant: True 时不立即写 L1，而是暂存到 event extra
            ``iris_pending_assistant``，由 after_message_sent 在拿到平台发送回执
            后调用 :func:`commit_pending_assistant` 写入（没发出去的话不进 L1）。
    """
    from ..platform import get_adapter
    from ..core.persona import resolve_persona

    # 提取助手响应内容
    assistant_msg = resp.completion_text

    if not assistant_msg:
        logger.debug("LLM 响应内容为空，跳过添加")
        return

    # 直接操作 L1Buffer，避免调用其他模块
    buffer = component_manager.get_available_component("l1_buffer")
    if not buffer:
        logger.debug("L1 Buffer 组件不可用，跳过响应添加")
        return

    l1_buffer = cast("L1Buffer", buffer)

    adapter = get_adapter(event)
    # L1 队列键使用会话 ID（私聊为 private:{user_id}），与写入侧保持一致
    session_id = adapter.get_session_id(event)

    # 解析 persona_id：助手响应必须携带正确人格归属，
    # 否则 buffer.py 用 messages[-1].persona_id 决定画像与 L2 摘要归属时，
    # default 占位会污染人格命名空间。
    persona_id = await resolve_persona(component_manager, event)

    if defer_assistant:
        event.set_extra(
            PENDING_ASSISTANT_EXTRA,
            {
                "session_id": session_id,
                "content": assistant_msg,
                "persona_id": persona_id,
            },
        )
        logger.debug(f"助手响应已暂存，等待发送回执后写入会话 {session_id} 的 L1")
    else:
        await l1_buffer.add_message(
            group_id=session_id,
            role="assistant",
            content=assistant_msg,
            source="assistant",
            persona_id=persona_id,
        )
        logger.debug(f"已添加助手响应到会话 {session_id} 的 L1 Buffer")

    # 学习模块旁路采集：对话对配对落库 + 表达模式提取，异常不影响主流程。
    # 延迟写 L1 的平台同样延迟到拿到发送回执（commit_pending_learning）。
    if defer_assistant:
        event.set_extra(PENDING_LEARNING_EXTRA, resp)
        return
    await _learning_on_response(event, resp, component_manager)


async def _learning_on_response(event, resp, component_manager) -> None:
    learning = component_manager.get_available_component("learning")
    if learning:
        try:
            await learning.on_response(event, resp)
        except Exception as e:
            logger.error(f"learning on_response 失败，已隔离：{e}", exc_info=True)


async def commit_pending_learning(
    event: "AstrMessageEvent",
    component_manager: "ComponentManager",
    sent: bool,
) -> bool:
    """读取并清除暂存的学习采集；``sent`` 为真（有发送回执）才真正采集。"""
    try:
        resp = event.get_extra(PENDING_LEARNING_EXTRA)
    except Exception:
        return False
    if resp is None:
        return False
    event.set_extra(PENDING_LEARNING_EXTRA, None)
    if not sent or component_manager is None:
        return False
    await _learning_on_response(event, resp, component_manager)
    return True


async def commit_pending_assistant(
    event: "AstrMessageEvent",
    component_manager: "ComponentManager",
    message_ids: "list[int]",
    turn: str = "",
) -> bool:
    """把暂存的助手响应写入 L1（仅在拿到发送回执后调用）。

    读取并清除 event extra ``iris_pending_assistant``；无暂存、回执为空或
    L1 不可用时返回 False（不写入）。metadata 带 ``message_ids`` 与 ``turn``。
    """
    pending = event.get_extra(PENDING_ASSISTANT_EXTRA)
    if not pending:
        return False
    event.set_extra(PENDING_ASSISTANT_EXTRA, None)
    if not message_ids:
        return False
    buffer = component_manager.get_available_component("l1_buffer")
    if not buffer:
        return False
    l1_buffer = cast("L1Buffer", buffer)
    metadata = {"message_ids": [int(i) for i in message_ids]}
    if turn:
        metadata["turn"] = turn
    return bool(
        await l1_buffer.add_message(
            group_id=pending.get("session_id", ""),
            role="assistant",
            content=pending.get("content", ""),
            source="assistant",
            metadata=metadata,
            persona_id=pending.get("persona_id") or "default",
        )
    )

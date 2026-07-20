"""
LLM 响应钩子处理模块

负责处理 LLM 响应相关的钩子逻辑。
"""

from typing import TYPE_CHECKING, cast

from iris_memory.core import get_logger

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import LLMResponse
    from iris_memory.core.components import ComponentManager
    from iris_memory.l1_buffer import L1Buffer

logger = get_logger("llm_response_hook")


async def handle_llm_response(
    event: "AstrMessageEvent",
    resp: "LLMResponse",
    component_manager: "ComponentManager",
) -> None:
    """处理 LLM 响应钩子

    只添加助手响应，用户消息已在 on_all_message 中添加。

    Args:
        event: AstrBot 消息事件对象
        resp: LLM 响应对象
        component_manager: 组件管理器实例
    """
    from iris_memory.platform import get_adapter
    from iris_memory.core.persona import resolve_persona

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

    await l1_buffer.add_message(
        group_id=session_id,
        role="assistant",
        content=assistant_msg,
        source="assistant",
        persona_id=persona_id,
    )

    logger.debug(f"已添加助手响应到会话 {session_id} 的 L1 Buffer")

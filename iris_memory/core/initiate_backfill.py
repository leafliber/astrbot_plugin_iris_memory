"""
initiate 直发消息的 L1 回填模块

主动回复引擎（proactive）的 initiate 通路经 context.send_message 直发，
不触发 AstrBot 事件钩子（on_llm_response / after_message_sent 均不执行），
bot 的这类发起消息因此不会进入 L1 缓冲。本模块在直发成功后把消息
按 bot 回复（handle_llm_response）的同一条路径补写入 L1，保持上下文连续。
"""

from typing import TYPE_CHECKING, cast

from iris_memory.core import get_logger

if TYPE_CHECKING:
    from iris_memory.core.components import ComponentManager
    from iris_memory.l1_buffer import L1Buffer

logger = get_logger("initiate_backfill")


async def handle_initiate_backfill(
    group_id: str,
    text: str,
    component_manager: "ComponentManager",
) -> None:
    """把 initiate 直发的 bot 消息回填到该群的 L1 缓冲

    复用 handle_llm_response 的写入路径与格式：
    role="assistant"、source="assistant"、内容原文写入，不加任何前缀
    （渲染层统一显示为 "Bot: {content}"，见
    llm_request_hook._collect_l1_context 对 assistant 消息的处理）。

    会话键约定：群聊的 L1 队列键即群号
    （见 PlatformAdapter.get_session_id；私聊才使用 private:{user_id}，
    initiate 只发生在群聊，故直接使用 group_id）。

    persona_id 继承该群 L1 队列最后一条消息：直发通路无事件对象，
    resolve_persona 不可用；队列尾部消息的人格即该群当前活跃人格，
    与正常消息流中 resolve_persona 的语义一致（见 buffer.py 总结归属逻辑）。
    若硬编码 "default"，人格隔离启用时该消息会成为 segment_2 尾部，
    导致整批 L1→L2 摘要与画像更新被错误归入 default 命名空间。
    人格隔离未启用时所有消息 persona_id 恒为 "default"，行为不变。

    L1 组件未就绪或该群尚无缓冲内容时静默跳过。

    Args:
        group_id: 群号（即 L1 会话键）
        text: bot 直发的消息原文
        component_manager: 组件管理器实例
    """
    if not group_id or not text:
        return

    buffer = component_manager.get_available_component("l1_buffer")
    if not buffer:
        logger.debug("L1 Buffer 组件不可用，跳过 initiate 回填")
        return

    l1_buffer = cast("L1Buffer", buffer)

    messages = l1_buffer.get_context(group_id)
    if not messages:
        logger.debug(f"群 {group_id} 尚无 L1 缓冲内容，跳过 initiate 回填")
        return

    persona_id = messages[-1].persona_id or "default"

    await l1_buffer.add_message(
        group_id=group_id,
        role="assistant",
        content=text,
        source="assistant",
        persona_id=persona_id,
    )

    logger.debug(
        f"已回填 initiate 消息到群 {group_id} 的 L1 Buffer（persona={persona_id}）"
    )

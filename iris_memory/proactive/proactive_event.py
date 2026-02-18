"""
主动回复合成事件

仿照 AstrBot 的 CronMessageEvent，为主动回复创建合成事件。
合成事件注入事件队列后，经历完整 Pipeline：
  WakingCheck → Whitelist → Session → RateLimit → ContentSafety
  → Process(on_all_messages 检测标记 → yield event.request_llm())
  → ResultDecorate → Respond

关键特性：
- send() 通过 Context.send_message() 实现，适配所有平台
- _extras["iris_proactive"] 标记防止死循环
- 保留原始 user_id / group_id / umo 供记忆检索和会话定位
"""
import time
import uuid
from typing import Any, Dict, List, Optional

from astrbot.core.message.components import Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import (
    AstrBotMessage,
    Group,
    MessageMember,
)
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata

from iris_memory.utils.logger import get_logger

logger = get_logger("proactive_event")


class ProactiveMessageEvent(AstrMessageEvent):
    """主动回复合成事件

    与 CronMessageEvent 相似，但专为插件主动回复设计。
    通过 Context.send_message(session, message) 发送消息，
    适配所有平台适配器。
    """

    def __init__(
        self,
        *,
        context,
        umo: str,
        trigger_prompt: str,
        user_id: str,
        sender_name: str = "",
        group_id: Optional[str] = None,
        proactive_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            context: AstrBot 的 Context 对象
            umo: 原始 unified_msg_origin（如 "aiocqhttp:GroupMessage:12345"）
            trigger_prompt: 注入 LLM 的触发提示（描述为什么主动回复）
            user_id: 目标用户 ID
            sender_name: 用户昵称（用于记忆检索中的身份标注）
            group_id: 群聊 ID（私聊时为 None）
            proactive_context: 主动回复的上下文（检测原因、情感、消息摘要等）
        """
        # 从 UMO 解析 session
        try:
            session = MessageSession.from_str(umo)
        except Exception:
            # 手动构造 session
            if group_id:
                session = MessageSession(
                    platform_name=umo.split(":")[0] if ":" in umo else "unknown",
                    message_type=MessageType.GROUP_MESSAGE,
                    session_id=group_id,
                )
            else:
                session = MessageSession(
                    platform_name=umo.split(":")[0] if ":" in umo else "unknown",
                    message_type=MessageType.FRIEND_MESSAGE,
                    session_id=user_id,
                )

        # 构造平台元数据（使用原始平台 ID，确保路由正确）
        platform_meta = PlatformMetadata(
            name=session.platform_name,
            description="ProactiveReply",
            id=session.platform_id,
        )

        # 构造消息对象
        msg_obj = AstrBotMessage()
        msg_obj.type = session.message_type
        msg_obj.self_id = "bot"
        msg_obj.session_id = session.session_id
        msg_obj.message_id = uuid.uuid4().hex
        msg_obj.sender = MessageMember(
            user_id=user_id,
            nickname=sender_name or user_id,
        )
        msg_obj.message = [Plain(trigger_prompt)]
        msg_obj.message_str = trigger_prompt
        msg_obj.raw_message = trigger_prompt
        msg_obj.timestamp = int(time.time())

        # 群聊时设置 group 对象
        if group_id:
            msg_obj.group = Group(group_id=group_id)

        # 调用父类初始化
        super().__init__(
            trigger_prompt,
            msg_obj,
            platform_meta,
            session.session_id,
        )

        # 覆写 session 以确保 UMO 与原始一致
        self.session = session

        # 存储 context 引用（用于 send）
        self.context_obj = context

        # 标记为主动回复事件（防循环核心）
        self._extras["iris_proactive"] = True

        # 存储主动回复上下文
        if proactive_context:
            self._extras["iris_proactive_context"] = proactive_context

        # 设置唤醒标记（确保通过 WakingCheckStage 后 handler 能激活）
        # 注意：is_wake 和 is_at_or_wake_command 会被 WakingCheckStage 重新计算，
        # 但我们的 on_all_messages handler 使用 EventMessageType.ALL 过滤器，
        # 无论 wake 状态如何都会被激活
        self.is_at_or_wake_command = False
        self.is_wake = False

        logger.debug(
            f"ProactiveMessageEvent created: umo={umo}, "
            f"user_id={user_id}, group_id={group_id}"
        )

    async def send(self, message: MessageChain) -> None:
        """通过 Context.send_message 发送消息"""
        if message is None:
            return
        try:
            await self.context_obj.send_message(self.session, message)
            await super().send(message)
        except Exception as e:
            logger.error(f"ProactiveMessageEvent send failed: {e}")

    async def send_streaming(self, generator, use_fallback: bool = False) -> None:
        """流式发送"""
        async for chain in generator:
            await self.send(chain)


__all__ = ["ProactiveMessageEvent"]

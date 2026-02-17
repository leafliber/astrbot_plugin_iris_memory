"""
主动回复模块

提供批量处理中的主动回复能力：
- 检测用户是否需要回复
- 构造合成事件注入 AstrBot 事件队列
- 经由完整 Pipeline 流程（人格、插件钩子、装饰、发送）执行主动回复
"""

from .proactive_reply_detector import (
    ProactiveReplyDetector,
    ProactiveReplyDecision,
    ReplyUrgency
)
from .reply_generator import (
    ProactiveReplyGenerator,
    GeneratedReply
)
from .message_sender import MessageSender, SendResult
from .proactive_manager import ProactiveReplyManager, ProactiveReplyTask
from .proactive_event import ProactiveMessageEvent

__all__ = [
    'ProactiveReplyDetector',
    'ProactiveReplyDecision',
    'ReplyUrgency',
    'ProactiveReplyGenerator',
    'GeneratedReply',
    'MessageSender',
    'SendResult',
    'ProactiveReplyManager',
    'ProactiveReplyTask',
    'ProactiveMessageEvent'
]

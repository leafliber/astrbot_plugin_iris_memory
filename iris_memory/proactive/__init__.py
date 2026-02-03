"""
主动回复模块

提供批量处理中的主动回复能力：
- 检测用户是否需要回复
- 生成个性化回复内容
- 主动发送消息
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

__all__ = [
    'ProactiveReplyDetector',
    'ProactiveReplyDecision',
    'ReplyUrgency',
    'ProactiveReplyGenerator',
    'GeneratedReply',
    'MessageSender',
    'SendResult',
    'ProactiveReplyManager',
    'ProactiveReplyTask'
]

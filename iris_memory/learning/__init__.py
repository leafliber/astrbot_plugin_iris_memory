"""
Iris Chat Memory - 学习子模块

从群聊对话中学习表达风格、对话样例与群黑话，
注入 LLM 上下文让回复更贴合群氛围。
"""

from .component import LearningComponent
from .storage import LearningStorage

__all__ = [
    "LearningComponent",
    "LearningStorage",
]

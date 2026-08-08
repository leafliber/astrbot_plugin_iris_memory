"""
Iris Chat Memory - 学习采集与配对

用户消息入口（on_message）与 LLM 响应落库时（on_response）的采集逻辑：
- on_message：self_id 过滤、剥离图片占位符、空文本跳过，更新黑话词频；
- on_response：把触发消息与 bot 回复配成 few_shot 对话对落库（pending_review），
  按规则提取表达模式候选，并把对话对放入审查器待审队列。
"""

import re
from typing import Any, Optional, TYPE_CHECKING

from iris_memory.core import get_logger
from iris_memory.platform import get_adapter
from iris_memory.utils.token_counter import count_tokens
from . import expression
from .jargon import JargonLearner
from .reviewer import LearningReviewer
from .storage import LearningStorage

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import LLMResponse

logger = get_logger("learning.collector")

# 图片占位符：[图:...] / [IMG:...]
_IMAGE_PLACEHOLDER = re.compile(r"\[(?:图:[^\]]*|IMG:[^\]]*)\]")

# bot 回复超过该 token 数不入 few_shot（与 L1 单条消息上限一致）
_MAX_REPLY_TOKENS = 500


def clean_text(text: str) -> str:
    """剥离图片占位符并修剪空白"""
    return _IMAGE_PLACEHOLDER.sub("", text or "").strip()


class LearningCollector:
    """学习数据采集器

    持有 storage / jargon / reviewer 引用，完成
    消息侧词频统计与响应侧对话对配对落库。
    """

    def __init__(
        self,
        storage: LearningStorage,
        jargon: JargonLearner,
        reviewer: LearningReviewer,
    ):
        self._storage = storage
        self._jargon = jargon
        self._reviewer = reviewer

    def on_message(
        self, event: "AstrMessageEvent", session_id: str, user_id: str, text: str
    ) -> None:
        """用户消息采集入口

        self_id 过滤、图片占位符剥离、空文本跳过后，
        更新黑话词频计数。

        Args:
            event: AstrBot 消息事件
            session_id: 会话 ID
            user_id: 用户 ID
            text: 消息文本（未清洗）
        """
        # 过滤机器人自己发出的消息
        try:
            self_id = event.get_self_id()
            if self_id and user_id and str(user_id) == str(self_id):
                return
        except Exception:
            pass

        cleaned = clean_text(text)
        if not cleaned:
            return

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event) or session_id
        self._jargon.update_counts(group_id, cleaned)

    def on_response(self, event: "AstrMessageEvent", resp: "LLMResponse") -> Optional[int]:
        """LLM 响应采集入口：对话对配对落库 + 表达模式提取

        跳过：无法取到触发消息文本、回复为空、回复超 500 token。

        Args:
            event: 触发消息事件
            resp: LLM 响应对象

        Returns:
            新插入的 few_shot 行 id；跳过返回 None
        """
        user_text = clean_text(getattr(event, "message_str", "") or "")
        if not user_text:
            return None

        bot_text = (getattr(resp, "completion_text", "") or "").strip()
        if not bot_text or count_tokens(bot_text) > _MAX_REPLY_TOKENS:
            return None

        adapter = get_adapter(event)
        session_id = adapter.get_session_id(event)
        group_id = adapter.get_group_id(event) or session_id
        user_id = adapter.get_user_id(event)

        # OneBot 平台可取消息 ID，其他平台防御性置空
        message_id: Optional[str] = None
        try:
            message_obj = getattr(event, "message_obj", None)
            raw_id = getattr(message_obj, "message_id", None) if message_obj else None
            if raw_id is not None:
                message_id = str(raw_id)
        except Exception:
            message_id = None

        pair_id = self._storage.insert_pair(
            group_id=group_id,
            user_id=user_id,
            user_text=user_text,
            bot_text=bot_text,
            message_id=message_id,
        )
        self._reviewer.enqueue(pair_id)

        # 规则提取表达模式候选（零 LLM）
        scene = expression.classify_scene(user_text)
        for expr in expression.extract_expressions(bot_text):
            pattern_id = self._storage.insert_pattern(
                group_id=group_id,
                scene=scene,
                expression=expr,
                source_pair_id=pair_id,
            )
            self._reviewer.enqueue_pattern(pattern_id)

        return pair_id

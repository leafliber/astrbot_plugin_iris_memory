"""
L1 规则检测器

快速过滤明显不需要回复的场景，高确信度场景直接回复。
零 I/O 开销，纯内存计算。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from iris_memory.proactive.core.models import (
    PersonalityConfig,
    ProactiveContext,
    ReplyType,
    RuleResult,
    UrgencyLevel,
    get_personality_config,
)
from iris_memory.proactive.detectors.base import BaseDetector
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.detector.rule")

# ========== 信号关键词表 ==========

# 疑问词（中文+英文常见）
QUESTION_KEYWORDS: List[str] = [
    "吗", "呢", "什么", "怎么", "为什么", "如何", "哪里", "哪个",
    "几个", "几点", "多少", "能不能", "可以吗", "是不是", "有没有",
    "谁", "怎样", "何时", "how", "what", "why", "where", "when",
]

# @Bot / 提及
MENTION_PATTERNS: List[str] = [
    "你说", "你怎么看", "你觉得", "你认为",
    "帮我", "帮忙", "求助",
]

# 情感词
EMOTION_POSITIVE: List[str] = [
    "开心", "高兴", "太好了", "成功了", "庆祝", "激动", "兴奋",
    "棒", "厉害", "牛", "绝了", "爽",
]
EMOTION_NEGATIVE: List[str] = [
    "难过", "伤心", "烦", "累", "焦虑", "压力", "失眠", "崩溃",
    "无聊", "不爽", "郁闷", "痛苦", "绝望", "迷茫",
]

# 寻求关注
ATTENTION_KEYWORDS: List[str] = [
    "有人吗", "在吗", "出来聊天", "好无聊", "陪我", "说说话",
]

# 简短确认（直接过滤）
SHORT_CONFIRM_PATTERNS: List[str] = [
    "嗯", "哦", "好的", "好吧", "行", "ok", "OK", "Ok",
    "收到", "了解", "知道了", "明白",
]

# 纯表情/表情包正则
# 使用 alternation 分开匹配 Unicode emoji 字符与 [表情] 标记
EMOJI_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"[\s"
    r"\U0001F600-\U0001F64F"  # Emoticons
    r"\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
    r"\U0001F680-\U0001F6FF"  # Transport & Map
    r"\U0001F1E0-\U0001F1FF"  # Flags
    r"\U0001F900-\U0001F9FF"  # Supplemental Symbols
    r"\U0001FA00-\U0001FA6F"  # Chess Symbols
    r"\U0001FA70-\U0001FAFF"  # Symbols Extended-A
    r"\U00002702-\U000027B0"  # Dingbats
    r"\U00002600-\U000026FF"  # Misc Symbols
    r"\U0000FE00-\U0000FE0F"  # Variation Selectors
    r"\U0000200D"             # ZWJ
    r"\U000023E9-\U000023F3"  # Misc Technical
    r"\U000023F8-\U000023FA"
    r"]"
    r"|\[\w+\]"              # [表情] 标记
    r")+$"
)


class RuleDetector(BaseDetector):
    """L1 规则检测器

    基于关键词和模式匹配进行快速检测：
    - score >= direct_reply_threshold → 直接回复（HIGH 紧急度）
    - score <= fast_reject_threshold → 直接忽略
    - 中间区域 → 传递给 L2 向量检测
    """

    def __init__(
        self,
        personality: str = "balanced",
    ) -> None:
        super().__init__(name="rule")
        self._personality = personality

    async def detect(self, context: ProactiveContext) -> RuleResult:
        """执行规则检测

        Args:
            context: 主动回复上下文

        Returns:
            RuleResult 检测结果
        """
        # 获取人格化阈值
        p_config = get_personality_config(
            self._personality, context.session_type
        )

        # 提取最新消息文本
        text = self._get_latest_text(context)
        if not text:
            return RuleResult(
                score=0.0,
                should_reply=False,
                confidence=1.0,
                matched_rules=["empty_message"],
            )

        # 计算各信号
        signals: Dict[str, float] = {}
        matched_rules: List[str] = []
        reply_type = ReplyType.CHAT

        # === 负向信号（先检测，一旦匹配直接返回） ===
        if self._is_short_confirm(text):
            return RuleResult(
                score=-1.0,
                signals={"short_confirm": -1.0},
                should_reply=False,
                confidence=1.0,
                matched_rules=["short_confirm"],
            )

        if self._is_emoji_only(text):
            return RuleResult(
                score=-1.0,
                signals={"emoji_only": -1.0},
                should_reply=False,
                confidence=1.0,
                matched_rules=["emoji_only"],
            )

        # === 正向信号 ===
        q_score = self._detect_question(text)
        if q_score > 0:
            signals["question"] = q_score
            matched_rules.append("question")
            reply_type = ReplyType.QUESTION

        mention_score = self._detect_mention(text)
        if mention_score > 0:
            signals["mention_bot"] = mention_score
            matched_rules.append("mention_bot")
            reply_type = ReplyType.QUESTION

        emo_score, emo_type = self._detect_emotion(text)
        if emo_score > 0:
            signals["emotion"] = emo_score
            matched_rules.append(f"emotion_{emo_type}")
            if emo_type == "negative" and emo_score > q_score:
                reply_type = ReplyType.EMOTION

        attention_score = self._detect_attention(text)
        if attention_score > 0:
            signals["attention"] = attention_score
            matched_rules.append("attention")

        # 综合得分
        total_score = sum(signals.values())
        total_score = max(0.0, min(1.0, total_score))

        # 决策
        direct_reply_threshold = p_config.rule_direct_reply
        fast_reject_threshold = (
            0.15 if context.session_type == "private" else 0.2
        )

        if total_score >= direct_reply_threshold:
            return RuleResult(
                score=total_score,
                signals=signals,
                should_reply=True,
                urgency=UrgencyLevel.HIGH,
                confidence=min(1.0, total_score),
                matched_rules=matched_rules,
                reply_type=reply_type,
            )
        elif total_score <= fast_reject_threshold:
            return RuleResult(
                score=total_score,
                signals=signals,
                should_reply=False,
                urgency=UrgencyLevel.LOW,
                confidence=1.0 - total_score,
                matched_rules=matched_rules,
                reply_type=reply_type,
            )
        else:
            # 中间区域，传给 L2
            return RuleResult(
                score=total_score,
                signals=signals,
                should_reply=False,  # 等待 L2 判断
                urgency=UrgencyLevel.LOW,
                confidence=total_score,
                matched_rules=matched_rules,
                reply_type=reply_type,
            )

    # ========== 信号检测 ==========

    @staticmethod
    def _get_latest_text(context: ProactiveContext) -> str:
        """获取最新一条用户消息文本"""
        msgs = context.conversation.recent_messages
        if not msgs:
            return ""
        # 最后一条消息
        last = msgs[-1]
        return last.get("text", "") if isinstance(last, dict) else ""

    @staticmethod
    def _is_short_confirm(text: str) -> bool:
        stripped = text.strip()
        return stripped in SHORT_CONFIRM_PATTERNS or len(stripped) <= 2

    @staticmethod
    def _is_emoji_only(text: str) -> bool:
        return bool(EMOJI_ONLY_PATTERN.match(text.strip())) and len(text.strip()) > 0

    @staticmethod
    def _detect_question(text: str) -> float:
        """检测疑问信号 → +0.3"""
        count = sum(1 for kw in QUESTION_KEYWORDS if kw in text)
        if text.rstrip().endswith("?") or text.rstrip().endswith("？"):
            count += 1
        return 0.3 if count > 0 else 0.0

    @staticmethod
    def _detect_mention(text: str) -> float:
        """检测 @Bot / 提及信号 → +0.4"""
        for pattern in MENTION_PATTERNS:
            if pattern in text:
                return 0.4
        return 0.0

    @staticmethod
    def _detect_emotion(text: str) -> tuple[float, str]:
        """检测情感信号 → +0.25

        Returns:
            (score, emotion_type: "positive" / "negative" / "")
        """
        pos_count = sum(1 for kw in EMOTION_POSITIVE if kw in text)
        neg_count = sum(1 for kw in EMOTION_NEGATIVE if kw in text)
        if neg_count > pos_count:
            return (0.25, "negative")
        elif pos_count > 0:
            return (0.25, "positive")
        return (0.0, "")

    @staticmethod
    def _detect_attention(text: str) -> float:
        """检测寻求关注信号 → +0.2"""
        for kw in ATTENTION_KEYWORDS:
            if kw in text:
                return 0.2
        return 0.0

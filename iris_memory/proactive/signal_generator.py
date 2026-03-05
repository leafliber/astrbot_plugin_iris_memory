"""
信号生成器

改造自 v2 RuleDetector，保留规则匹配逻辑，但输出改为 Signal 对象。
零 I/O 开销，纯内存计算。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from iris_memory.config import get_store
from iris_memory.proactive.models import Signal, SignalType
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.signal_generator")


# ========== 信号关键词表 ==========

# 疑问词（中文 + 英文常见 + 口语变体）
QUESTION_KEYWORDS: List[str] = [
    "吗", "呢", "什么", "怎么", "为什么", "如何", "哪里", "哪个",
    "几个", "几点", "多少", "能不能", "可以吗", "是不是", "有没有",
    "谁", "怎样", "何时", "咱", "啊", "哪样",
    "how", "what", "why", "where", "when",
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
    "绝绝子", "yyds", "赞", "欧耶", "嘴角上扬",
    "开花", "小确幸", "笑死",
]
EMOTION_NEGATIVE: List[str] = [
    "难过", "伤心", "烦", "累", "焦虑", "压力", "失眠", "崩溃",
    "无聊", "不爽", "郁闷", "痛苦", "绝望", "迷茫",
    "破防", "emo", "裂开", "麻了", "不想努力了",
    "不开心", "心累", "自闭",
]

# 寻求关注
ATTENTION_KEYWORDS: List[str] = [
    "有人吗", "在吗", "出来聊天", "好无聊", "陪我", "说说话",
    "有没有人", "谁在", "好寂寞", "一个人好无聊",
]

# 简短确认（直接过滤）
SHORT_CONFIRM_PATTERNS: List[str] = [
    "嗯", "哦", "好的", "好吧", "行", "ok", "OK", "Ok",
    "收到", "了解", "知道了", "明白",
]

# 纯表情正则
EMOJI_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"[\s"
    r"\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002702-\U000027B0"
    r"\U00002600-\U000026FF"
    r"\U0000FE00-\U0000FE0F"
    r"\U0000200D"
    r"\U000023E9-\U000023F3"
    r"\U000023F8-\U000023FA"
    r"]"
    r"|\[\w+\]"
    r")+$"
)


class SignalGenerator:
    """信号生成器

    基于规则快速检测消息是否包含值得主动回复的信号。
    只判断信号权重，不做最终决策（决策交给 GroupScheduler）。

    检测流程：
    1. 负向检测（短确认、纯表情）→ 不生成信号
    2. 规则匹配检测 → 生成 rule_match 信号
    3. 情感强度检测 → 生成 emotion_high 信号
    """

    def __init__(self) -> None:
        self._cfg = get_store()

    def generate(
        self,
        text: str,
        user_id: str,
        group_id: str,
        session_key: str,
        emotion_intensity: float = 0.0,
    ) -> List[Signal]:
        """从消息中生成信号

        Args:
            text: 消息文本
            user_id: 用户 ID
            group_id: 群组 ID
            session_key: 会话标识
            emotion_intensity: 情感强度（0.0 - 1.0）

        Returns:
            生成的信号列表（可能为空）
        """
        signals: List[Signal] = []

        if not text or not text.strip():
            return signals

        text = text.strip()

        # 负向检测：短确认和纯表情直接跳过
        if self._is_short_confirm(text) or self._is_emoji_only(text):
            return signals

        # 规则匹配检测
        rule_signal = self._detect_rule_match(
            text, user_id, group_id, session_key
        )
        if rule_signal:
            signals.append(rule_signal)

        # 情感强度检测
        emotion_signal = self._detect_emotion_high(
            text, user_id, group_id, session_key, emotion_intensity
        )
        if emotion_signal:
            signals.append(emotion_signal)

        return signals

    def _detect_rule_match(
        self,
        text: str,
        user_id: str,
        group_id: str,
        session_key: str,
    ) -> Optional[Signal]:
        """规则匹配检测

        Returns:
            匹配的 Signal，或 None
        """
        score = 0.0
        matched: List[str] = []

        # 疑问检测
        q_score = self._detect_question(text)
        if q_score > 0:
            score += q_score
            matched.append("question")

        # 提及检测
        mention_score = self._detect_mention(text)
        if mention_score > 0:
            score += mention_score
            matched.append("mention")

        # 寻求关注检测
        attention_score = self._detect_attention(text)
        if attention_score > 0:
            score += attention_score
            matched.append("attention")

        # 情感词检测（规则层面，与情感强度独立）
        emo_score, emo_type = self._detect_emotion_keywords(text)
        if emo_score > 0:
            score += emo_score
            matched.append(f"emotion_{emo_type}")

        score = max(0.0, min(1.0, score))

        if score < 0.2:  # 低于最低阈值，不生成信号
            return None

        ttl = self._cfg.get("proactive_reply.signal_ttl_rule_match", 300)
        return Signal(
            signal_type=SignalType.RULE_MATCH,
            session_key=session_key,
            group_id=group_id,
            user_id=user_id,
            weight=score,
            expires_at=datetime.now() + timedelta(seconds=ttl),
            metadata={"matched_rules": matched, "text_preview": text[:50]},
        )

    def _detect_emotion_high(
        self,
        text: str,
        user_id: str,
        group_id: str,
        session_key: str,
        emotion_intensity: float,
    ) -> Optional[Signal]:
        """高情感信号检测

        当外部提供的情感强度 > 0.7 时生成 emotion_high 信号。

        Returns:
            Signal 或 None
        """
        if emotion_intensity < 0.7:
            return None

        ttl = self._cfg.get("proactive_reply.signal_ttl_emotion_high", 180)
        weight = min(1.0, 0.7 + (emotion_intensity - 0.7) * 1.0)

        return Signal(
            signal_type=SignalType.EMOTION_HIGH,
            session_key=session_key,
            group_id=group_id,
            user_id=user_id,
            weight=weight,
            expires_at=datetime.now() + timedelta(seconds=ttl),
            metadata={
                "emotion_intensity": emotion_intensity,
                "text_preview": text[:50],
            },
        )

    # ========== 关键词检测 ==========

    @staticmethod
    def _detect_question(text: str) -> float:
        """疑问检测，返回 0.0 - 0.3"""
        count = sum(1 for kw in QUESTION_KEYWORDS if kw in text)
        if count == 0:
            return 0.0
        if text.rstrip().endswith("?") or text.rstrip().endswith("？"):
            return 0.3
        return min(0.3, count * 0.15)

    @staticmethod
    def _detect_mention(text: str) -> float:
        """提及检测，返回 0.0 - 0.4"""
        for pattern in MENTION_PATTERNS:
            if pattern in text:
                return 0.4
        return 0.0

    @staticmethod
    def _detect_attention(text: str) -> float:
        """寻求关注检测，返回 0.0 - 0.2"""
        for kw in ATTENTION_KEYWORDS:
            if kw in text:
                return 0.2
        return 0.0

    @staticmethod
    def _detect_emotion_keywords(text: str) -> Tuple[float, str]:
        """情感关键词检测

        Returns:
            (score, type) 分数和类型（positive/negative）
        """
        neg_count = sum(1 for kw in EMOTION_NEGATIVE if kw in text)
        pos_count = sum(1 for kw in EMOTION_POSITIVE if kw in text)

        if neg_count > pos_count:
            return min(0.25, neg_count * 0.1), "negative"
        elif pos_count > 0:
            return min(0.15, pos_count * 0.08), "positive"
        return 0.0, ""

    @staticmethod
    def _is_short_confirm(text: str) -> bool:
        """是否为简短确认"""
        stripped = text.strip().rstrip("。.!！~")
        return stripped in SHORT_CONFIRM_PATTERNS

    @staticmethod
    def _is_emoji_only(text: str) -> bool:
        """是否为纯表情"""
        if not text.strip():
            return True
        return EMOJI_ONLY_PATTERN.match(text.strip()) is not None

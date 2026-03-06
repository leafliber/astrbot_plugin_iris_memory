"""
情感差异化衰减模型

基于心理学"情感适应理论"（Hedonic Adaptation），
正面情感记忆衰减慢以保留美好回忆，负面情感记忆衰减快以实现"情感愈合"。
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from iris_memory.models.memory import Memory


@dataclass
class EmotionDecayProfile:
    """情感衰减配置表 - 基于心理学研究的差异化参数

    衰减公式: S(t) = e^(-λ * t)
    其中 λ 为衰减率，t 为自上次访问以来的天数。
    """

    # 正面情感：慢衰减，保留美好陪伴记忆
    POSITIVE_DECAY_RATES: ClassVar[Dict[str, float]] = {
        "joy": 0.012,          # ~60天半衰期
        "grateful": 0.012,     # ~60天
        "love": 0.015,         # ~45天
        "excitement": 0.018,   # ~38天
        "calm": 0.020,         # ~35天
        "contentment": 0.015,  # ~45天
        "amusement": 0.023,    # ~30天（默认）
    }

    # 负面情感：快衰减，加速情感愈合
    NEGATIVE_DECAY_RATES: ClassVar[Dict[str, float]] = {
        "sadness": 0.099,      # ~7天半衰期
        "anger": 0.099,        # ~7天
        "anxiety": 0.099,      # ~7天
        "anxious": 0.099,      # ~7天（别名）
        "fear": 0.077,         # ~9天
        "disgust": 0.077,      # ~9天
        "frustration": 0.077,  # ~9天
    }

    # 中性情感：标准衰减
    NEUTRAL_DECAY_RATE: ClassVar[float] = 0.023  # ~30天

    @classmethod
    def get_decay_rate(cls, emotion_subtype: Optional[str]) -> float:
        """获取特定情感的衰减率"""
        if not emotion_subtype:
            return cls.NEUTRAL_DECAY_RATE
        subtype = emotion_subtype.lower()
        if subtype in cls.POSITIVE_DECAY_RATES:
            return cls.POSITIVE_DECAY_RATES[subtype]
        if subtype in cls.NEGATIVE_DECAY_RATES:
            return cls.NEGATIVE_DECAY_RATES[subtype]
        return cls.NEUTRAL_DECAY_RATE

    @classmethod
    def get_valence(cls, emotion_subtype: Optional[str]) -> str:
        """获取情感效价: 'positive' / 'negative' / 'neutral'"""
        if not emotion_subtype:
            return "neutral"
        subtype = emotion_subtype.lower()
        if subtype in cls.POSITIVE_DECAY_RATES:
            return "positive"
        if subtype in cls.NEGATIVE_DECAY_RATES:
            return "negative"
        return "neutral"

    @classmethod
    def calculate_emotion_time_score(
        cls,
        memory: "Memory",
        reference_time: Optional[datetime] = None,
    ) -> float:
        """计算情感记忆的时间得分

        差异化衰减: score = e^(-λ_emotion * t)

        Args:
            memory: 情感记忆对象
            reference_time: 参考时间（默认当前时间）

        Returns:
            时间得分 (0.0-1.0)
        """
        ref = reference_time or datetime.now()
        days_since_access = (ref - memory.last_access_time).total_seconds() / 86400
        if days_since_access < 0:
            days_since_access = 0

        # 优先使用记忆自身的衰减率，否则按子类型查表
        decay_rate = memory.emotion_decay_rate if (hasattr(memory, "emotion_decay_rate") and memory.emotion_decay_rate) else cls.get_decay_rate(memory.subtype)

        score = math.exp(-decay_rate * days_since_access)

        # 保护标记加成
        if hasattr(memory, "has_protection"):
            from iris_memory.models.protection import ProtectionFlag
            if memory.has_protection(ProtectionFlag.HIGH_EMOTION):
                score = min(1.0, score * 1.3)

        return min(1.0, max(0.0, score))

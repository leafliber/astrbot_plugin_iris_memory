"""
核心数据类型定义
根据companion-memory框架文档定义所有枚举和基础类型
"""

from enum import Enum
from typing import Optional
from datetime import datetime


class MemoryType(str, Enum):
    """记忆类型分类"""
    FACT = "fact"                    # 事实类：可验证的信息
    EMOTION = "emotion"              # 情感类：情感状态和反应
    RELATIONSHIP = "relationship"    # 关系类：互动关系和边界
    INTERACTION = "interaction"      # 互动类：对话模式和反馈
    INFERRED = "inferred"            # 推断类：系统推断的信息


class ModalityType(str, Enum):
    """模态类型"""
    TEXT = "text"                    # 文本
    AUDIO = "audio"                  # 语音
    IMAGE = "image"                  # 图像
    VIDEO = "video"                  # 视频


class QualityLevel(int, Enum):
    """质量等级 - 置信度分级"""
    UNCERTAIN = 1                    # 高度不确定，置信度0.0-0.3
    LOW_CONFIDENCE = 2               # 推测或间接获取，置信度0.3-0.5
    MODERATE = 3                     # 提及过但未验证，置信度0.5-0.75
    HIGH_CONFIDENCE = 4              # 多次提及且一致，置信度0.75-0.9
    CONFIRMED = 5                    # 用户明确确认的信息，置信度0.9-1.0


class SensitivityLevel(int, Enum):
    """敏感度等级"""
    PUBLIC = 0                       # 公开信息，无隐私风险
    PERSONAL = 1                     # 个人偏好，轻微隐私
    PRIVATE = 2                      # 私人信息，需要保护
    SENSITIVE = 3                    # 敏感信息，严格保护
    CRITICAL = 4                     # 极度敏感，最高保护


class StorageLayer(str, Enum):
    """存储层"""
    WORKING = "working"              # 工作记忆：会话内临时存储
    EPISODIC = "episodic"            # 情景记忆：基于RIF评分动态管理
    SEMANTIC = "semantic"            # 语义记忆：永久保存用户画像


class EmotionType(str, Enum):
    """情感类型"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"
    CALM = "calm"


class VerificationMethod(str, Enum):
    """验证方法"""
    USER_EXPLICIT = "user_explicit"          # 用户明确确认
    MULTIPLE_MENTIONS = "multiple_mentions"  # 多次提及
    CROSS_VALIDATION = "cross_validation"    # 跨验证
    SYSTEM_INFERRED = "system_inferred"      # 系统推断
    UNVERIFIED = "unverified"                # 未验证


class DecayRate:
    """记忆衰减率配置
    
    根据companion-memory框架，不同类型记忆有不同的衰减半衰期：
    - 兴趣：30天半衰期
    - 习惯：90天半衰期
    - 人格：365天半衰期
    - 价值观：730天半衰期
    """
    
    # 基础衰减率（λ值，用于指数衰减）
    INTEREST = 0.023          # 30天半衰期: ln(0.5)/30 ≈ 0.023
    HABIT = 0.008             # 90天半衰期: ln(0.5)/90 ≈ 0.008
    PERSONALITY = 0.002       # 365天半衰期: ln(0.5)/365 ≈ 0.002
    VALUES = 0.001            # 730天半衰期: ln(0.5)/730 ≈ 0.001
    
    @classmethod
    def get_decay_rate(cls, memory_type: MemoryType) -> float:
        """根据记忆类型获取衰减率"""
        mapping = {
            MemoryType.FACT: cls.HABIT,            # 事实类：90天半衰期
            MemoryType.EMOTION: cls.INTEREST,      # 情感类：30天半衰期
            MemoryType.RELATIONSHIP: cls.PERSONALITY,  # 关系类：365天半衰期
            MemoryType.INTERACTION: cls.INTEREST,  # 互动类：30天半衰期
            MemoryType.INFERRED: cls.HABIT,        # 推断类：90天半衰期
        }
        return mapping.get(memory_type, cls.HABIT)


class RetrievalStrategy(str, Enum):
    """检索策略"""
    VECTOR_ONLY = "vector_only"                    # 纯向量检索
    GRAPH_ONLY = "graph_only"                      # 纯图检索
    TIME_AWARE = "time_aware"                      # 时间感知检索
    EMOTION_AWARE = "emotion_aware"                # 情感感知检索
    HYBRID = "hybrid"                              # 混合检索


class TriggerType(str, Enum):
    """触发器类型"""
    EXPLICIT = "explicit"                          # 显式触发器：记住、重要
    PREFERENCE = "preference"                      # 偏好触发器：喜欢、讨厌
    EMOTION = "emotion"                            # 情感触发器：觉得、感到
    RELATIONSHIP = "relationship"                  # 关系触发器：我们是、你对我来说
    FACT = "fact"                                  # 事实触发器：我是、我有
    BOUNDARY = "boundary"                          # 边界触发器：不要、不想

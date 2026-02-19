"""
情感分析器
根据companion-memory框架文档实现混合情感分析模型
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from iris_memory.utils.logger import get_logger

from iris_memory.core.types import EmotionType
from iris_memory.core.constants import NEGATIVE_EMOTIONS_CORE
from iris_memory.models.emotion_state import EmotionalState

# 模块logger
logger = get_logger("emotion_analyzer")


class EmotionAnalyzer:
    """情感分析器
    
    实现混合情感分析模型：
    - 情感词典（30%）
    - 规则系统（30%）
    - 轻量模型（40%）
    
    增强能力：
    - 上下文修正：识别讽刺、隐喻、文化语境
    - 时序建模：滑动窗口分析最近7天的情感变化
    - 双向反馈：情感状态影响记忆检索，记忆检索反向更新情感历史
    """
    
    def __init__(self, config=None):
        """初始化情感分析器
        
        Args:
            config: 插件配置对象（保留用于未来扩展）
        """
        
        # 情感分析始终启用
        self.enable_emotion = True
        
        # 初始化情感词典
        self._init_emotion_dict()
        
        # 初始化规则系统
        self._init_rules()
    
    def _init_emotion_dict(self):
        """初始化情感词典"""
        # 基础情感词典（简化版，实际应使用更完整的词典）
        self.emotion_dict = {
            EmotionType.JOY: [
                "开心", "快乐", "高兴", "喜悦", "愉快", "幸福", "满足",
                "喜欢", "爱", "棒", "好", "赞", "优秀", "成功", "赢",
                "happy", "joy", "glad", "love", "great", "good", "excellent"
            ],
            EmotionType.SADNESS: [
                "难过", "伤心", "悲伤", "痛苦", "失望", "沮丧", "郁闷",
                "哭", "眼泪", "不幸", "失败", "失去", "sad", "sorry", "cry"
            ],
            EmotionType.ANGER: [
                "生气", "愤怒", "火大", "恼火", "烦躁", "讨厌", "恨",
                "怒", "愤", "angry", "hate", "mad", "furious"
            ],
            EmotionType.FEAR: [
                "害怕", "恐惧", "担心", "焦虑", "紧张", "不安", "慌",
                "怕", "惧", "scared", "fear", "worried", "anxious"
            ],
            EmotionType.ANXIETY: [
                "焦虑", "担心", "不安", "紧张", "压力", "困扰", "烦恼",
                "烦躁", "焦虑", "anxiety", "stress", "nervous"
            ],
            EmotionType.EXCITEMENT: [
                "兴奋", "激动", "期待", "憧憬", "热情", "充满活力",
                "excited", "thrilled", "eager", "enthusiastic"
            ],
            EmotionType.CALM: [
                "平静", "安静", "淡定", "冷静", "宁静", "安逸",
                "calm", "peaceful", "quiet", "serene"
            ]
        }
        
        # 否定词列表 - 使用更精确的否定词匹配
        self.negation_words = [
            "不", "没", "无", "别", "莫", "勿", "不是", "没有",
            "not", "no", "never", "don't", "doesn't", "won't"
        ]
        
        # 程度副词（不应被当作否定词）
        self.degree_words = [
            "非常", "很", "十分", "极其", "特别", "相当"
        ]
    
    def _init_rules(self):
        """初始化情感分析规则"""
        self.rules = [
            # 句子规则
            {
                "pattern": r"^[!！]{2,}$",
                "emotion": EmotionType.EXCITEMENT,
                "weight": 0.8
            },
            {
                "pattern": r"[？?]{2,}$",
                "emotion": EmotionType.ANXIETY,
                "weight": 0.6
            },
            # 标点规则
            {
                "pattern": r"[~～]{3,}",
                "emotion": EmotionType.JOY,
                "weight": 0.5
            },
            # 上下文规则（讽刺检测）
            {
                "pattern": r"真是.*呀$|真是.*啊$|你说.*呢",
                "emotion": EmotionType.ANGER,
                "weight": 0.7,
                "sarcastic": True
            }
        ]
    
    async def analyze_emotion(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """分析文本情感
        
        Args:
            text: 输入文本
            context: 上下文信息（对话历史、当前情境等）
            
        Returns:
            Dict[str, Any]: 情感分析结果
            {
                "primary": EmotionType,
                "secondary": List[EmotionType],
                "intensity": float (0-1),
                "confidence": float (0-1),
                "contextual_correction": bool
            }
        """
        # 处理 None 或空文本
        if text is None:
            logger.debug("None text, returning neutral")
            return {
                "primary": EmotionType.NEUTRAL,
                "secondary": [],
                "intensity": 0.5,
                "confidence": 0.5,
                "contextual_correction": False
            }
        
        text_preview = text[:40] + "..." if len(text) > 40 else text
        logger.debug(f"Analyzing emotion for text: '{text_preview}'")
        
        if not self.enable_emotion:
            logger.debug("Emotion analysis disabled, returning neutral")
            return {
                "primary": EmotionType.NEUTRAL,
                "secondary": [],
                "intensity": 0.5,
                "confidence": 0.5,
                "contextual_correction": False
            }
        
        
        # 1. 情感词典分析（30%权重）
        logger.debug("Step 1: Dictionary-based analysis...")
        dict_result = self._analyze_by_dict(text)
        logger.debug(f"Dictionary result: primary={dict_result['primary'].value}, intensity={dict_result['intensity']:.2f}")
        
        # 2. 规则系统分析（30%权重）
        logger.debug("Step 2: Rule-based analysis...")
        rule_result = self._analyze_by_rules(text)
        logger.debug(f"Rule result: primary={rule_result['primary'].value}, intensity={rule_result['intensity']:.2f}")
        
        # 3. 上下文修正（可选）
        logger.debug("Step 3: Contextual correction...")
        contextual_correction = self._detect_contextual_correction(text, context)
        logger.debug(f"Contextual correction: {contextual_correction}")
        
        # 综合分析结果
        primary, secondary, intensity, confidence = self._combine_results(
            dict_result,
            rule_result,
            contextual_correction
        )
        
        result = {
            "primary": primary,
            "secondary": secondary,
            "intensity": intensity,
            "confidence": confidence,
            "contextual_correction": contextual_correction
        }
        
        logger.debug(f"Emotion analysis complete: primary={primary.value}, intensity={intensity:.2f}, confidence={confidence:.2f}")
        return result
    
    def _analyze_by_dict(self, text: str) -> Dict[str, Any]:
        """使用情感词典分析（30%权重）"""
        emotion_scores = {emotion: 0 for emotion in EmotionType}
        
        # 检测否定词（排除程度副词）
        has_negation = False
        for neg in self.negation_words:
            # 检查是否是真正的否定词（不是程度副词的一部分）
            if neg in text:
                # 检查是否在程度副词中
                is_part_of_degree = any(degree_word in text and neg in degree_word for degree_word in self.degree_words)
                if not is_part_of_degree:
                    has_negation = True
                    break
        
        # 统计情感词出现次数
        for emotion, keywords in self.emotion_dict.items():
            count = 0
            for keyword in keywords:
                # 使用正则匹配，对中文和英文都适用
                # 不使用单词边界，直接匹配
                matches = re.findall(re.escape(keyword), text, re.IGNORECASE)
                count += len(matches)
            
            emotion_scores[emotion] = count
        
        # 如果有否定词，反转部分情感（更智能的处理）
        if has_negation:
            # 只反转明确的正负情感对
            joy_score = emotion_scores[EmotionType.JOY]
            sadness_score = emotion_scores[EmotionType.SADNESS]
            excitement_score = emotion_scores[EmotionType.EXCITEMENT]
            
            # 否定正面情感 -> 变为中性或负面
            if joy_score > 0:
                emotion_scores[EmotionType.JOY] = 0
                emotion_scores[EmotionType.NEUTRAL] = emotion_scores.get(EmotionType.NEUTRAL, 0) + joy_score * 0.5
            
            if excitement_score > 0:
                emotion_scores[EmotionType.EXCITEMENT] = 0
                emotion_scores[EmotionType.CALM] = emotion_scores.get(EmotionType.CALM, 0) + excitement_score * 0.5
        
        # 计算强度和置信度
        total_score = sum(emotion_scores.values())
        
        if total_score == 0:
            return {
                "primary": EmotionType.NEUTRAL,
                "secondary": [],
                "intensity": 0.5,
                "confidence": 0.3
            }
        
        # 找到主要情感
        primary = max(emotion_scores.items(), key=lambda x: x[1])[0]
        intensity = min(1.0, emotion_scores[primary] / 3.0)  # 最多3个关键词达到最大强度
        confidence = min(0.8, total_score * 0.15 + 0.3)  # 基础置信度0.3
        
        # 找到次要情感（得分>0）
        secondary = [
            e for e, score in emotion_scores.items()
            if score > 0 and e != primary
        ]
        
        return {
            "primary": primary,
            "secondary": secondary[:2],  # 最多2个次要情感
            "intensity": intensity,
            "confidence": confidence
        }
    
    def _analyze_by_rules(self, text: str) -> Dict[str, Any]:
        """使用规则系统分析（30%权重）"""
        for rule in self.rules:
            pattern = rule["pattern"]
            if re.search(pattern, text):
                return {
                    "primary": rule["emotion"],
                    "secondary": [],
                    "intensity": rule.get("weight", 0.5),
                    "confidence": 0.7
                }
        
        return {
            "primary": EmotionType.NEUTRAL,
            "secondary": [],
            "intensity": 0.5,
            "confidence": 0.0
        }
    
    def _detect_contextual_correction(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """检测是否需要上下文修正
        
        识别讽刺、隐喻、文化语境等
        """
        if not context:
            return False
        
        # 检测讽刺
        sarcastic_patterns = [
            r"真是.*呀", r"你真行", r"太棒了（？！）", r"呵呵"
        ]
        for pattern in sarcastic_patterns:
            if re.search(pattern, text):
                return True
        
        # 检测反问句
        if re.search(r"[？?]{2,}", text):
            return True
        
        # 检查上下文不一致
        if "history" in context and context["history"]:
            # 如果最近的情感是负面，当前文本包含正面词，可能是讽刺
            recent_emotions = [e.get("primary") for e in context["history"][-3:]]
            negative_count = sum(
                1 for e in recent_emotions
                if e in NEGATIVE_EMOTIONS_CORE
            )
            if negative_count >= 2 and any(
                kw in text for kw in ["好", "棒", "赞", "优秀"]
            ):
                return True
        
        return False
    
    def _combine_results(
        self,
        dict_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        contextual_correction: bool
    ) -> tuple:
        """综合分析结果
        
        Args:
            dict_result: 词典分析结果
            rule_result: 规则分析结果
            contextual_correction: 是否需要上下文修正
            
        Returns:
            tuple: (primary, secondary, intensity, confidence)
        """
        # 权重分配
        dict_weight = 0.3
        rule_weight = 0.3
        
        # 计算综合得分
        emotion_scores = {}
        
        # 词典结果
        primary = dict_result["primary"]
        emotion_scores[primary] = dict_result["intensity"] * dict_weight
        for sec in dict_result["secondary"]:
            emotion_scores[sec] = emotion_scores.get(sec, 0) + dict_result["intensity"] * 0.1 * dict_weight
        
        # 规则结果
        if rule_result["confidence"] > 0:
            rule_primary = rule_result["primary"]
            emotion_scores[rule_primary] = emotion_scores.get(rule_primary, 0) + rule_result["intensity"] * rule_weight
        
        # 轻量模型（40%权重，这里简化处理）
        # 实际应该调用transformers模型
        model_weight = 0.4
        # 暂时使用词典结果的加权版本
        if primary in emotion_scores:
            emotion_scores[primary] += dict_result["intensity"] * model_weight
        
        # 确定主要情感
        if not emotion_scores:
            return (EmotionType.NEUTRAL, [], 0.5, 0.3)
        
        primary = max(emotion_scores.items(), key=lambda x: x[1])[0]
        intensity = min(1.0, emotion_scores[primary])
        
        # 计算置信度
        dict_conf = dict_result["confidence"] * dict_weight
        rule_conf = rule_result["confidence"] * rule_weight
        model_conf = dict_result["confidence"] * model_weight
        confidence = dict_conf + rule_conf + model_conf
        
        # 如果需要上下文修正，降低置信度
        if contextual_correction:
            confidence *= 0.6
        
        # 找次要情感
        secondary = [
            e for e, score in emotion_scores.items()
            if e != primary and score > 0.1
        ][:2]
        
        return (primary, secondary, intensity, confidence)
    
    def update_emotional_state(
        self,
        emotional_state: EmotionalState,
        primary: EmotionType,
        intensity: float,
        confidence: float,
        secondary: List[EmotionType] = None
    ):
        """更新情感状态
        
        Args:
            emotional_state: 情感状态对象
            primary: 主要情感
            intensity: 强度
            confidence: 置信度
            secondary: 次要情感列表
        """
        emotional_state.update_current_emotion(
            primary=primary,
            intensity=intensity,
            confidence=confidence,
            secondary=secondary or []
        )
    
    def should_filter_positive_memories(self, emotional_state: EmotionalState) -> bool:
        """判断是否应该过滤高强度正面记忆

        当用户心情不好时，避免检索高强度正面记忆

        Args:
            emotional_state: 情感状态对象

        Returns:
            bool: 是否应该过滤
        """
        if emotional_state is None:
            return False
        return emotional_state.should_filter_positive()
    
    def analyze_time_series(
        self,
        emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """分析情感时序
        
        Args:
            emotional_state: 情感状态对象
            
        Returns:
            Dict[str, Any]: 时序分析结果
        """
        return {
            "trend": emotional_state.trajectory.trend.value,
            "volatility": emotional_state.trajectory.volatility,
            "anomaly_detected": emotional_state.trajectory.anomaly_detected,
            "needs_intervention": emotional_state.trajectory.needs_intervention,
            "negative_ratio": emotional_state.get_negative_ratio()
        }

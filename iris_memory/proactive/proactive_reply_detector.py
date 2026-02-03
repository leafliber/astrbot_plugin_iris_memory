"""
主动回复检测器
判断批量处理的消息是否需要主动回复
"""
import re
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from iris_memory.utils.logger import get_logger
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer

logger = get_logger("proactive_reply")


class ReplyUrgency(Enum):
    """回复紧急度"""
    IGNORE = "ignore"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProactiveReplyDecision:
    """主动回复决策"""
    should_reply: bool
    urgency: ReplyUrgency
    reason: str
    suggested_delay: int
    reply_context: Dict[str, Any]


class ProactiveReplyDetector:
    """主动回复检测器"""
    
    def __init__(
        self,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        config: Optional[Dict] = None
    ):
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self.config = config or {}
        
        # 配置阈值
        self.high_emotion_threshold = self.config.get("high_emotion_threshold", 0.7)
        self.question_threshold = self.config.get("question_threshold", 0.8)
        self.mention_threshold = self.config.get("mention_threshold", 0.9)
        
        # 需要回复的关键词（适配群聊场景）
        self.reply_triggers = {
            "question": [
                r"[吗嘛呢吧？?]$",
                r"^(什么|怎么|为什么|如何|哪里|谁|多少|啥)",
                r"^(能|可以|会).*吗[?？]?$",
                r"^(是不是|对不对|行不行)",
                r".*?(呢|吧|啊)[?？]$",
            ],
            "emotional_support": [
                r"(难过|伤心|痛苦|哭|难受|烦|郁闷|孤独|emo|破防)",
                r"(开心|高兴|兴奋|惊喜|喜欢|感谢|爽|牛|棒)",
                r"(压力|累|疲惫|焦虑|担心|害怕|慌|怂)",
                r"(笑死|哈哈哈|呜呜呜|啊这)",
            ],
            "seeking_attention": [
                r"(在吗|有人吗|喂|哈喽|hello|在么|在不在)",
                r"(出来|冒泡|潜水|水群)",
            ],
            "mention_bot": [
                r"(你说|你觉得|你怎么看|你的意见|你的想法)",
                r"(@bot|机器人|AI|助手|千酱)",
            ],
            "expect_response": [
                r"(等你|期待|希望|想听|想问问)",
                r"(对吧|是吧|好吗|行吗|可以吗|对吧)",
                r"(求|有没有|谁知道)",
            ],
            "chat_topics": [
                r"(今天|昨天|最近|刚才).*?(怎么样|如何|发生了)",
                r"(大家觉得|你们认为|有没有人不)",
                r"(分享|推荐|安利|避雷)",
            ]
        }
        
        # 无需回复的模式
        self.ignore_patterns = [
            r"^(嗯|哦|啊|好|行|可以|OK|ok)$",
            r"^(哈哈|呵呵|嘻嘻|嘿嘿)$",
            r"^(谢谢|感谢|谢了)$",
            r"^[0-9\s]+$",
        ]
    
    async def analyze(
        self,
        messages: List[str],
        user_id: str,
        group_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> ProactiveReplyDecision:
        """分析消息列表，判断是否需要主动回复"""
        if not messages:
            return self._no_reply_decision("empty_messages")
        
        combined_text = "\n".join(messages)
        last_message = messages[-1]
        
        # 分析情感状态
        emotion_result = await self.emotion_analyzer.analyze_emotion(
            last_message, context
        )
        
        # 检测信号
        signals = self._detect_reply_signals(last_message, combined_text)
        
        # 综合判断
        decision = self._make_decision(
            signals=signals,
            emotion=emotion_result,
            message_count=len(messages),
            time_span=context.get("time_span", 0) if context else 0,
            user_persona=context.get("user_persona", {}) if context else {}
        )
        
        logger.info(f"Proactive reply decision for {user_id}: "
                   f"{decision.urgency.value}, reason: {decision.reason}")
        
        return decision
    
    def _detect_reply_signals(
        self,
        last_message: str,
        combined_text: str
    ) -> Dict[str, float]:
        """检测回复信号"""
        signals = {}
        
        signals["question"] = self._match_patterns(
            last_message, self.reply_triggers["question"]
        )
        
        signals["emotional_support"] = self._match_patterns(
            combined_text, self.reply_triggers["emotional_support"]
        )
        
        signals["seeking_attention"] = self._match_patterns(
            last_message, self.reply_triggers["seeking_attention"]
        )
        
        signals["mention_bot"] = self._match_patterns(
            combined_text, self.reply_triggers["mention_bot"]
        )
        
        signals["expect_response"] = self._match_patterns(
            last_message, self.reply_triggers["expect_response"]
        )
        
        signals["length"] = min(len(last_message) / 100, 1.0)
        
        if any(re.match(p, last_message.strip()) for p in self.ignore_patterns):
            signals["ignore"] = 1.0
        else:
            signals["ignore"] = 0.0
        
        return signals
    
    def _match_patterns(self, text: str, patterns: List[str]) -> float:
        """匹配模式并返回强度"""
        if not text or not patterns:
            return 0.0
        
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        return min(matches / max(len(patterns) * 0.5, 1), 1.0)
    
    def _make_decision(
        self,
        signals: Dict[str, float],
        emotion: Dict[str, Any],
        message_count: int,
        time_span: float,
        user_persona: Dict
    ) -> ProactiveReplyDecision:
        """做出回复决策"""
        
        if signals.get("ignore", 0) > 0.5:
            return ProactiveReplyDecision(
                should_reply=False,
                urgency=ReplyUrgency.IGNORE,
                reason="message_should_be_ignored",
                suggested_delay=0,
                reply_context={"signals": signals, "emotion": emotion}
            )
        
        reply_score = 0.0
        reasons = []
        
        if signals["question"] > 0.5:
            reply_score += 0.4 * signals["question"]
            reasons.append(f"question({signals['question']:.2f})")
        
        if signals["emotional_support"] > 0.3:
            reply_score += 0.3 * signals["emotional_support"]
            reasons.append(f"emotion({signals['emotional_support']:.2f})")
        
        if signals["seeking_attention"] > 0.5:
            reply_score += 0.3 * signals["seeking_attention"]
            reasons.append(f"attention({signals['seeking_attention']:.2f})")
        
        if signals["mention_bot"] > 0.3:
            reply_score += 0.5 * signals["mention_bot"]
            reasons.append(f"mention({signals['mention_bot']:.2f})")
        
        if signals["expect_response"] > 0.3:
            reply_score += 0.35 * signals["expect_response"]
            reasons.append(f"expect({signals['expect_response']:.2f})")
        
        if signals.get("chat_topics", 0) > 0.3:
            reply_score += 0.25 * signals["chat_topics"]
            reasons.append(f"topic({signals['chat_topics']:.2f})")
        
        emotion_intensity = emotion.get("intensity", 0)
        emotion_type = emotion.get("primary", "neutral")
        
        # 降低情感阈值，让更多情感能被触发
        if emotion_intensity > self.high_emotion_threshold:
            reply_score += 0.2
            reasons.append(f"high_emotion({emotion_intensity:.2f})")
        elif emotion_intensity > 0.4 and emotion_type != "neutral":
            # 非中性情感且有一定强度
            reply_score += 0.1
            reasons.append(f"emotion({emotion_type},{emotion_intensity:.2f})")
        elif emotion_type in ["joy", "excitement"] and emotion_intensity > 0.3:
            # 积极情感降低阈值（适合群聊氛围）
            reply_score += 0.15
            reasons.append(f"positive({emotion_intensity:.2f})")
        
        # 用户个性化
        user_preference = user_persona.get("proactive_reply_preference", 0.5)
        reply_score *= (0.8 + 0.4 * user_preference)
        
        # 根据分数决定（群聊场景优化，降低阈值）
        if reply_score >= 0.7:
            urgency = ReplyUrgency.CRITICAL
            should_reply = True
            delay = 0
        elif reply_score >= 0.5:
            urgency = ReplyUrgency.HIGH
            should_reply = True
            delay = 5
        elif reply_score >= 0.3:
            urgency = ReplyUrgency.MEDIUM
            should_reply = True
            delay = 30
        elif reply_score >= 0.15:
            urgency = ReplyUrgency.LOW
            should_reply = False
            delay = 120
        else:
            urgency = ReplyUrgency.IGNORE
            should_reply = False
            delay = 0
        
        return ProactiveReplyDecision(
            should_reply=should_reply,
            urgency=urgency,
            reason=" + ".join(reasons) if reasons else "low_score",
            suggested_delay=delay,
            reply_context={
                "signals": signals,
                "emotion": emotion,
                "reply_score": reply_score,
                "message_count": message_count,
                "time_span": time_span
            }
        )
    
    def _no_reply_decision(self, reason: str) -> ProactiveReplyDecision:
        """创建无需回复的决策"""
        return ProactiveReplyDecision(
            should_reply=False,
            urgency=ReplyUrgency.IGNORE,
            reason=reason,
            suggested_delay=0,
            reply_context={}
        )

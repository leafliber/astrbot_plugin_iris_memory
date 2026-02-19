"""
LLM增强主动回复检测器
使用LLM进行社交语境理解，判断是否需要主动回复

基于 LLMEnhancedDetector 模板方法模式
修复了 sync/async 阻抗不匹配问题
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDetector,
    ProactiveReplyDecision,
    ReplyUrgency,
)
from iris_memory.core.detection.base_result import BaseDetectionResult
from iris_memory.core.detection.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedDetector,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("llm_proactive_reply_detector")


PROACTIVE_REPLY_PROMPT = """分析以下对话是否需要AI主动回复。

## 判断标准
需要主动回复的情况：
1. **明确提问**：用户提出了问题，期待回答
2. **情感支持**：用户表达了负面情绪，需要安慰
3. **寻求关注**：用户在寻找陪伴或回应
4. **提及AI**：用户直接提到或询问AI
5. **期待回应**：用户表达了期待回复的意图

不需要主动回复的情况：
1. **已解决的求助**：用户自己找到了答案
2. **简单确认**：用户只是简单确认或回应
3. **自言自语**：用户在自说自话，不需要回应
4. **群聊闲聊**：群内正常聊天，不需要AI介入
5. **指令消息**：用户在发送指令，不是在对话

## 对话内容
{messages}

## 输出格式
请以JSON格式返回：
```json
{{
  "should_reply": true/false,
  "urgency": "critical|high|medium|low|ignore",
  "confidence": 0.0-1.0,
  "reason": "判断理由",
  "reply_type": "answer|comfort|chat|acknowledge|null",
  "suggested_delay": 建议延迟秒数(0-120)
}}
```

仅返回JSON，不要有其他文字。"""


# 紧急度枚举映射
_URGENCY_MAP = {
    "critical": ReplyUrgency.CRITICAL,
    "high": ReplyUrgency.HIGH,
    "medium": ReplyUrgency.MEDIUM,
    "low": ReplyUrgency.LOW,
    "ignore": ReplyUrgency.IGNORE,
}

# 微妙回复信号词
_SUBTLE_REPLY_SIGNALS = [
    "有人", "在吗", "有人吗", "有人吗？",
    "怎么办", "求助", "帮帮我", "救命",
    "好烦", "好累", "好难过", "emo"
]


@dataclass
class LLMReplyDecision(BaseDetectionResult):
    """LLM主动回复决策"""
    should_reply: bool = False
    urgency: ReplyUrgency = ReplyUrgency.IGNORE
    reply_type: str = "null"
    suggested_delay: int = 0


class LLMProactiveReplyDetector(LLMEnhancedDetector[LLMReplyDecision]):
    """LLM增强主动回复检测器
    
    支持三种模式：
    - rule: 仅使用规则检测（快速）
    - llm: 仅使用LLM检测（社交理解）
    - hybrid: 规则预筛 + LLM确认（推荐）
    """
    
    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        mode: DetectionMode = DetectionMode.HYBRID,
        daily_limit: int = 100,
    ):
        super().__init__(
            astrbot_context=astrbot_context,
            provider_id=provider_id,
            mode=mode,
            daily_limit=daily_limit,
            max_tokens=200,
        )
        self._rule_detector = ProactiveReplyDetector()
    
    def _should_skip_input(self, *args, **kwargs) -> bool:
        """空消息时跳过"""
        messages = args[0] if args else kwargs.get("messages")
        if not messages:
            return True
        if all(not m or not m.strip() for m in messages):
            return True
        return False
    
    async def analyze(
        self,
        messages: List[str],
        user_id: str = "",
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMReplyDecision:
        """分析消息，判断是否需要主动回复（detect的别名，保持接口兼容）"""
        return await self.detect(messages, user_id, group_id, context)
    
    def _get_empty_result(self) -> LLMReplyDecision:
        """空输入默认结果"""
        return LLMReplyDecision(
            confidence=1.0,
            source="rule",
            reason="空消息列表",
        )
    
    async def _rule_detect_async(
        self,
        messages: List[str],
        user_id: str,
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMReplyDecision:
        """规则检测（异步版本）"""
        try:
            decision = await self._rule_detector.analyze(
                messages, user_id, group_id, context
            )
        except Exception as e:
            logger.debug(f"Async rule detection failed, falling back to sync: {e}")
            decision = self._sync_rule_detect(messages, user_id, group_id, context)
        
        return LLMReplyDecision(
            should_reply=decision.should_reply,
            urgency=decision.urgency,
            confidence=0.7,
            reason=decision.reason,
            reply_type="chat",
            suggested_delay=decision.suggested_delay,
            source="rule",
        )
    
    def _rule_detect(
        self,
        messages: List[str],
        user_id: str = "",
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMReplyDecision:
        """规则检测（同步回退版本）"""
        decision = self._sync_rule_detect(messages, user_id, group_id, context)
        return LLMReplyDecision(
            should_reply=decision.should_reply,
            urgency=decision.urgency,
            confidence=0.7,
            reason=decision.reason,
            reply_type="chat",
            suggested_delay=decision.suggested_delay,
            source="rule",
        )
    
    def _sync_rule_detect(
        self,
        messages: List[str],
        user_id: str = "",
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ProactiveReplyDecision:
        """同步规则检测"""
        last_message = messages[-1] if messages else ""
        signals = self._rule_detector._detect_reply_signals(
            last_message, "\n".join(messages)
        )
        
        if signals.get("ignore", 0) > 0.5:
            return ProactiveReplyDecision(
                should_reply=False,
                urgency=ReplyUrgency.IGNORE,
                reason="message_should_be_ignored",
                suggested_delay=0,
                reply_context={"signals": signals}
            )
        
        reply_score = 0.0
        reasons = []
        
        if signals["question"] > 0.5:
            reply_score += 0.4 * signals["question"]
            reasons.append("question")
        if signals["emotional_support"] > 0.3:
            reply_score += 0.3 * signals["emotional_support"]
            reasons.append("emotion")
        if signals["seeking_attention"] > 0.5:
            reply_score += 0.3 * signals["seeking_attention"]
            reasons.append("attention")
        if signals["mention_bot"] > 0.3:
            reply_score += 0.5 * signals["mention_bot"]
            reasons.append("mention")
        
        if reply_score >= 0.5:
            return ProactiveReplyDecision(
                should_reply=True,
                urgency=(
                    ReplyUrgency.HIGH if reply_score >= 0.7
                    else ReplyUrgency.MEDIUM
                ),
                reason=" + ".join(reasons),
                suggested_delay=5 if reply_score >= 0.7 else 30,
                reply_context={
                    "signals": signals, "reply_score": reply_score
                }
            )
        
        return ProactiveReplyDecision(
            should_reply=False,
            urgency=ReplyUrgency.IGNORE,
            reason="low_score",
            suggested_delay=0,
            reply_context={"signals": signals, "reply_score": reply_score}
        )
    
    def _build_prompt(
        self,
        messages: List[str],
        user_id: str = "",
        group_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """构建LLM提示词"""
        messages_text = "\n".join(
            [f"用户: {m}" for m in messages[-5:]]
        )
        return PROACTIVE_REPLY_PROMPT.format(messages=messages_text)
    
    async def _llm_detect(
        self,
        messages: List[str],
        user_id: str = "",
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMReplyDecision:
        """LLM检测（覆写以使用异步规则回退）"""
        prompt = self._build_prompt(messages, user_id, group_id)
        result = await self._call_llm(prompt)
        
        if not result.success or not result.parsed_json:
            logger.debug(
                "LLM proactive reply detection failed, falling back to rule"
            )
            return await self._rule_detect_async(
                messages, user_id, group_id, context
            )
        
        return self._parse_llm_result(result.parsed_json)
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> LLMReplyDecision:
        """解析LLM结果"""
        suggested_delay = data.get("suggested_delay", 30)
        if isinstance(suggested_delay, (int, float)):
            suggested_delay = int(
                BaseDetectionResult.clamp(float(suggested_delay), 0, 120)
            )
        else:
            suggested_delay = 30
        
        return LLMReplyDecision(
            should_reply=data.get("should_reply", False),
            urgency=BaseDetectionResult.map_enum(
                data.get("urgency", "ignore"), _URGENCY_MAP,
                ReplyUrgency.IGNORE
            ),
            confidence=BaseDetectionResult.parse_confidence(data),
            reason=data.get("reason", "LLM判断"),
            reply_type=data.get("reply_type", "chat"),
            suggested_delay=suggested_delay,
            source="llm",
        )
    
    async def _hybrid_detect(
        self,
        messages: List[str],
        user_id: str = "",
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMReplyDecision:
        """混合检测：规则预筛 + LLM确认"""
        rule_result = await self._rule_detect_async(
            messages, user_id, group_id, context
        )
        
        # 紧急 → 直接返回
        if rule_result.urgency == ReplyUrgency.CRITICAL:
            return rule_result
        
        # 忽略+高置信度 → 检查微妙信号
        if (rule_result.urgency == ReplyUrgency.IGNORE
                and rule_result.confidence >= 0.8):
            last_message = messages[-1] if messages else ""
            if self._might_need_reply_anyway(last_message):
                llm_result = await self._llm_detect(
                    messages, user_id, group_id, context
                )
                if llm_result.should_reply:
                    llm_result.source = "hybrid"
                    return llm_result
            return rule_result
        
        # 应回复但低置信度 → LLM确认
        if rule_result.should_reply and rule_result.confidence < 0.8:
            llm_result = await self._llm_detect(
                messages, user_id, group_id, context
            )
            if llm_result.confidence >= 0.6:
                llm_result.source = "hybrid"
                return llm_result
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _might_need_reply_anyway(self, text: str) -> bool:
        """检查是否可能仍需要回复"""
        return any(signal in text for signal in _SUBTLE_REPLY_SIGNALS)

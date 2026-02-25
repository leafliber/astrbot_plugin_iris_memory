"""
LLM增强冲突解决器
使用LLM进行语义层面的记忆冲突检测和解决

基于 LLMEnhancedDetector 模板方法模式
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from iris_memory.capture.conflict.conflict_resolver import ConflictResolver
from iris_memory.models.memory import Memory
from iris_memory.core.detection.base_result import BaseDetectionResult
from iris_memory.core.detection.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedDetector,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("llm_conflict_resolver")


CONFLICT_DETECTION_PROMPT = """分析以下两条记忆是否存在语义冲突。

## 记忆1
{memory1}

## 记忆2
{memory2}

## 冲突类型
1. **直接矛盾**：内容完全相反（如"我喜欢苹果" vs "我讨厌苹果"）
2. **条件差异**：有条件限制的矛盾（如"我讨厌雨天，但今天的雨让我很开心"）
3. **时间变化**：偏好随时间改变（如"以前喜欢咖啡，现在改喝茶了"）
4. **无冲突**：内容不矛盾或互补

## 输出格式
请以JSON格式返回：
```json
{{
  "has_conflict": true/false,
  "conflict_type": "direct|conditional|temporal|none",
  "confidence": 0.0-1.0,
  "reason": "判断理由",
  "resolution_suggestion": "merge|keep_both|keep_newer|keep_older|need_context"
}}
```

仅返回JSON，不要有其他文字。"""


# 允许的冲突类型
_ALLOWED_CONFLICT_TYPES = ["direct", "conditional", "temporal", "none"]
# 允许的解决策略
_ALLOWED_RESOLUTIONS = [
    "merge", "keep_both", "keep_newer", "keep_older", "need_context"
]
# 微妙冲突指示词对（扩展多维度检测）
_SUBTLE_CONFLICT_INDICATORS = [
    # 偏好矛盾
    ("喜欢", "不"), ("爱", "不"), ("讨厌", "但"),
    ("恨", "但"), ("想", "不"), ("要", "不"),
    ("喜欢", "讨厌"), ("爱", "恨"), ("想要", "不想"),
    # 情感矛盾
    ("开心", "难过"), ("高兴", "伤心"), ("快乐", "痛苦"),
    ("满意", "不满"), ("期待", "害怕"), ("兴奋", "失望"),
    ("乐观", "悲观"), ("自信", "自卑"),
    # 状态矛盾
    ("是", "不是"), ("有", "没有"), ("会", "不会"),
    ("能", "不能"), ("在", "不在"), ("懂", "不懂"),
    # 时间变化信号
    ("以前", "现在"), ("之前", "后来"), ("原来", "现在"),
    ("过去", "如今"), ("曾经", "目前"), ("小时候", "长大后"),
    # 频率矛盾
    ("经常", "很少"), ("总是", "从不"), ("每天", "偶尔"),
    ("常常", "几乎不"), ("一直", "不再"),
    # 英文
    ("like", "dislike"), ("love", "hate"), ("always", "never"),
    ("before", "now"), ("used to", "no longer"),
]


@dataclass
class ConflictDetectionResult(BaseDetectionResult):
    """冲突检测结果"""
    has_conflict: bool = False
    conflict_type: str = "none"
    resolution_suggestion: str = "keep_both"


class LLMConflictResolver(LLMEnhancedDetector[ConflictDetectionResult]):
    """LLM增强冲突解决器
    
    支持三种模式：
    - rule: 仅使用规则检测（快速）
    - llm: 仅使用LLM检测（语义判断）
    - hybrid: 规则预筛 + LLM确认（推荐）
    """
    
    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        mode: DetectionMode = DetectionMode.HYBRID,
        daily_limit: int = 0,
        similarity_calculator=None,
    ):
        super().__init__(
            astrbot_context=astrbot_context,
            provider_id=provider_id,
            mode=mode,
            daily_limit=daily_limit,
            max_tokens=200,
        )
        self._rule_resolver = ConflictResolver(similarity_calculator)
    
    def _should_skip_input(self, *args, **kwargs) -> bool:
        """冲突检测不跳过输入"""
        return False
    
    def _get_empty_result(self) -> ConflictDetectionResult:
        return ConflictDetectionResult(
            confidence=1.0,
            source="rule",
            reason="无输入",
        )
    
    def _rule_detect(
        self,
        memory1: Memory,
        memory2: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictDetectionResult:
        """规则检测"""
        is_opposite = self._rule_resolver.is_opposite(
            memory1.content, memory2.content
        )
        
        if is_opposite:
            return ConflictDetectionResult(
                has_conflict=True,
                conflict_type="direct",
                confidence=0.7,
                reason="规则检测到相反内容",
                resolution_suggestion="keep_newer",
                source="rule",
            )
        
        return ConflictDetectionResult(
            confidence=0.6,
            reason="规则未检测到冲突",
            source="rule",
        )
    
    def _build_prompt(
        self,
        memory1: Memory,
        memory2: Memory,
        **kwargs
    ) -> str:
        """构建LLM提示词"""
        return CONFLICT_DETECTION_PROMPT.format(
            memory1=(
                f"内容: {memory1.content}\n"
                f"时间: {memory1.created_time.isoformat() if memory1.created_time else '未知'}"
            ),
            memory2=(
                f"内容: {memory2.content}\n"
                f"时间: {memory2.created_time.isoformat() if memory2.created_time else '未知'}"
            ),
        )
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> ConflictDetectionResult:
        """解析LLM结果"""
        return ConflictDetectionResult(
            has_conflict=data.get("has_conflict", False),
            conflict_type=BaseDetectionResult.validate_string(
                data.get("conflict_type", "none"),
                _ALLOWED_CONFLICT_TYPES,
                "none",
            ),
            confidence=BaseDetectionResult.parse_confidence(data),
            reason=data.get("reason", "LLM判断"),
            resolution_suggestion=BaseDetectionResult.validate_string(
                data.get("resolution_suggestion", "keep_both"),
                _ALLOWED_RESOLUTIONS,
                "keep_both",
            ),
            source="llm",
        )
    
    async def _hybrid_detect(
        self,
        memory1: Memory,
        memory2: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictDetectionResult:
        """混合检测：规则预筛 + LLM确认"""
        rule_result = self._rule_detect(memory1, memory2, context)
        
        # 无冲突 → 检查微妙冲突
        if not rule_result.has_conflict:
            if self._might_have_subtle_conflict(
                memory1.content, memory2.content
            ):
                llm_result = await self._llm_detect(
                    memory1, memory2, context
                )
                if llm_result.has_conflict:
                    llm_result.source = "hybrid"
                    return llm_result
            rule_result.source = "hybrid"
            return rule_result
        
        # 有冲突 → LLM确认
        llm_result = await self._llm_detect(memory1, memory2, context)
        if llm_result.confidence >= 0.6:
            llm_result.source = "hybrid"
            return llm_result
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _might_have_subtle_conflict(self, text1: str, text2: str) -> bool:
        """检查是否有细微冲突
        
        多维度检测策略：
        1. 词对指示器（偏好/情感/状态/时间/频率矛盾）
        2. 数值差异检测（相同描述框架下的不同数字）
        3. 时间矛盾检测（相同主题不同时间线）
        """
        # 1. 词对事指示器
        for word1, word2 in _SUBTLE_CONFLICT_INDICATORS:
            if ((word1 in text1 and word2 in text2)
                    or (word1 in text2 and word2 in text1)):
                return True
        
        # 2. 数值差异检测
        nums1 = re.findall(r'\d+', text1)
        nums2 = re.findall(r'\d+', text2)
        if nums1 and nums2 and set(nums1) != set(nums2):
            # 检查非数字部分是否相似（相同描述框架）
            non_num1 = re.sub(r'\d+', '', text1).strip()
            non_num2 = re.sub(r'\d+', '', text2).strip()
            if non_num1 and non_num2:
                # 简单字符重叠检查
                chars1 = set(non_num1.replace(' ', ''))
                chars2 = set(non_num2.replace(' ', ''))
                overlap = len(chars1 & chars2) / max(len(chars1), len(chars2), 1)
                if overlap > 0.5:
                    return True
        
        # 3. 时间矛盾检测：相同主题不同时间词
        time_markers = [
            "昨天", "今天", "明天", "上周", "上月", "去年",
            "今年", "明年", "昨晚", "今晚", "上午", "下午",
        ]
        has_time1 = any(t in text1 for t in time_markers)
        has_time2 = any(t in text2 for t in time_markers)
        if has_time1 and has_time2:
            # 两条都有时间词且时间不同，可能是时间矛盾
            times1 = [t for t in time_markers if t in text1]
            times2 = [t for t in time_markers if t in text2]
            if set(times1) != set(times2):
                return True
        
        return False
    
    async def resolve(
        self,
        new_memory: Memory,
        conflicting_memories: List[Memory],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """解决冲突"""
        if not conflicting_memories:
            return "no_conflict"
        
        resolutions = []
        for old_memory in conflicting_memories:
            result = await self.detect(new_memory, old_memory, context)
            
            if result.has_conflict:
                if result.conflict_type == "conditional":
                    resolutions.append("keep_both")
                elif result.conflict_type == "temporal":
                    resolutions.append("keep_newer")
                elif result.resolution_suggestion == "merge":
                    resolutions.append("merge")
                else:
                    resolutions.append("keep_newer")
            else:
                resolutions.append("keep_both")
        
        if all(r == "keep_both" for r in resolutions):
            return "keep_both"
        if all(r == "merge" for r in resolutions):
            return "merge"
        if "keep_newer" in resolutions:
            return "keep_newer"
        
        return "need_context"
    

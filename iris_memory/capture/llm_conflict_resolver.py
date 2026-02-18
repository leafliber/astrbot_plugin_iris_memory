"""
LLM增强冲突解决器
使用LLM进行语义层面的记忆冲突检测和解决
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from iris_memory.capture.conflict_resolver import ConflictResolver
from iris_memory.models.memory import Memory
from iris_memory.processing.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedBase,
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


@dataclass
class ConflictDetectionResult:
    """冲突检测结果"""
    has_conflict: bool
    conflict_type: str  # "direct" | "conditional" | "temporal" | "none"
    confidence: float
    reason: str
    resolution_suggestion: str  # "merge" | "keep_both" | "keep_newer" | "keep_older" | "need_context"
    source: str


class LLMConflictResolver(LLMEnhancedBase):
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
    
    async def detect(
        self,
        memory1: Memory,
        memory2: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictDetectionResult:
        """检测两条记忆是否存在冲突
        
        Args:
            memory1: 记忆1
            memory2: 记忆2
            context: 上下文信息
            
        Returns:
            ConflictDetectionResult: 检测结果
        """
        if self._mode == DetectionMode.RULE:
            return self._rule_detect(memory1, memory2, context)
        elif self._mode == DetectionMode.LLM:
            return await self._llm_detect(memory1, memory2, context)
        else:
            return await self._hybrid_detect(memory1, memory2, context)
    
    def _rule_detect(
        self,
        memory1: Memory,
        memory2: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictDetectionResult:
        """规则检测"""
        is_opposite = self._rule_resolver.is_opposite(memory1.content, memory2.content)
        
        if is_opposite:
            return ConflictDetectionResult(
                has_conflict=True,
                conflict_type="direct",
                confidence=0.7,
                reason="规则检测到相反内容",
                resolution_suggestion="keep_newer",
                source="rule"
            )
        
        return ConflictDetectionResult(
            has_conflict=False,
            conflict_type="none",
            confidence=0.6,
            reason="规则未检测到冲突",
            resolution_suggestion="keep_both",
            source="rule"
        )
    
    async def _llm_detect(
        self,
        memory1: Memory,
        memory2: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictDetectionResult:
        """LLM检测"""
        prompt = CONFLICT_DETECTION_PROMPT.format(
            memory1=f"内容: {memory1.content}\n时间: {memory1.created_time.isoformat() if memory1.created_time else '未知'}",
            memory2=f"内容: {memory2.content}\n时间: {memory2.created_time.isoformat() if memory2.created_time else '未知'}"
        )
        result = await self._call_llm(prompt)
        
        if not result.success or not result.parsed_json:
            logger.debug(f"LLM conflict detection failed, falling back to rule")
            return self._rule_detect(memory1, memory2, context)
        
        return self._parse_llm_result(result.parsed_json)
    
    async def _hybrid_detect(
        self,
        memory1: Memory,
        memory2: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictDetectionResult:
        """混合检测：规则预筛 + LLM确认"""
        rule_result = self._rule_detect(memory1, memory2, context)
        
        if not rule_result.has_conflict:
            if self._might_have_subtle_conflict(memory1.content, memory2.content):
                llm_result = await self._llm_detect(memory1, memory2, context)
                if llm_result.has_conflict:
                    llm_result.source = "hybrid"
                    return llm_result
            rule_result.source = "hybrid"
            return rule_result
        
        llm_result = await self._llm_detect(memory1, memory2, context)
        if llm_result.confidence >= 0.6:
            llm_result.source = "hybrid"
            return llm_result
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _might_have_subtle_conflict(self, text1: str, text2: str) -> bool:
        """检查是否有细微冲突"""
        subtle_conflict_indicators = [
            ("喜欢", "不"),
            ("爱", "不"),
            ("讨厌", "但"),
            ("恨", "但"),
            ("想", "不"),
            ("要", "不"),
        ]
        for word1, word2 in subtle_conflict_indicators:
            if (word1 in text1 and word2 in text2) or (word1 in text2 and word2 in text1):
                return True
        return False
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> ConflictDetectionResult:
        """解析LLM结果"""
        has_conflict = data.get("has_conflict", False)
        
        conflict_type = data.get("conflict_type", "none").lower()
        if conflict_type not in ["direct", "conditional", "temporal", "none"]:
            conflict_type = "none"
        
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        
        resolution = data.get("resolution_suggestion", "keep_both").lower()
        if resolution not in ["merge", "keep_both", "keep_newer", "keep_older", "need_context"]:
            resolution = "keep_both"
        
        return ConflictDetectionResult(
            has_conflict=has_conflict,
            conflict_type=conflict_type,
            confidence=confidence,
            reason=data.get("reason", "LLM判断"),
            resolution_suggestion=resolution,
            source="llm"
        )
    
    async def resolve(
        self,
        new_memory: Memory,
        conflicting_memories: List[Memory],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """解决冲突
        
        Args:
            new_memory: 新记忆
            conflicting_memories: 冲突的记忆列表
            context: 上下文信息
            
        Returns:
            str: 解决策略
        """
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
    
    def is_opposite(self, text1: str, text2: str) -> bool:
        """判断两个文本是否相反（兼容原接口）"""
        return self._rule_resolver.is_opposite(text1, text2)
    
    async def check_duplicate_by_vector(
        self,
        memory: Memory,
        user_id: str,
        group_id: Optional[str],
        chroma_manager,
        similarity_threshold: float = 0.95
    ) -> Optional[Memory]:
        """检查重复记忆（兼容原接口）"""
        return await self._rule_resolver.check_duplicate_by_vector(
            memory, user_id, group_id, chroma_manager, similarity_threshold
        )
    
    async def check_conflicts_by_vector(
        self,
        memory: Memory,
        user_id: str,
        group_id: Optional[str],
        chroma_manager
    ) -> List[Memory]:
        """检查记忆冲突（兼容原接口）"""
        return await self._rule_resolver.check_conflicts_by_vector(
            memory, user_id, group_id, chroma_manager
        )

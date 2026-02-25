"""
LLM增强检索路由器
使用LLM进行查询意图理解，选择最优检索策略

基于 LLMEnhancedDetector 模板方法模式
"""
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from iris_memory.core.types import RetrievalStrategy
from iris_memory.retrieval.retrieval_router import RetrievalRouter
from iris_memory.core.detection.base_result import BaseDetectionResult
from iris_memory.core.detection.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedDetector,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("llm_retrieval_router")


RETRIEVAL_ROUTING_PROMPT = """分析以下查询，选择最优的检索策略。

## 查询内容
{query}

## 检索策略说明
1. **VECTOR_ONLY**：纯向量检索
   - 适用于：简单查询、单关键词、短文本
   - 示例："我喜欢什么"、"我的爱好"

2. **TIME_AWARE**：时间感知检索
   - 适用于：包含时间线索的查询
   - 示例："我昨天说了什么"、"最近有什么重要的事"

3. **EMOTION_AWARE**：情感感知检索
   - 适用于：情感相关的查询或用户当前情感状态需要关注
   - 示例："我什么时候最开心"、"我最近心情怎么样"

4. **GRAPH_ONLY**：图遍历检索（关系查询）
   - 适用于：涉及实体关系的查询
   - 示例："谁是我的上司"、"我和XXX是什么关系"

5. **HYBRID**：混合检索
   - 适用于：复杂查询、多维度约束
   - 示例："我上周在公司和同事讨论了什么项目"

## 输出格式
请以JSON格式返回：
```json
{{
  "strategy": "vector_only|time_aware|emotion_aware|graph_only|hybrid",
  "confidence": 0.0-1.0,
  "reason": "选择理由",
  "query_type": "simple|time|emotion|relation|complex",
  "key_entities": ["提取的关键实体"]
}}
```

仅返回JSON，不要有其他文字。"""


# 检索策略枚举映射
_STRATEGY_MAP = {
    "vector_only": RetrievalStrategy.VECTOR_ONLY,
    "time_aware": RetrievalStrategy.TIME_AWARE,
    "emotion_aware": RetrievalStrategy.EMOTION_AWARE,
    "graph_only": RetrievalStrategy.GRAPH_ONLY,
    "hybrid": RetrievalStrategy.HYBRID,
}

# 复杂查询指示词（扩展多维度语义检测）
_COMPLEX_INDICATORS = [
    # 连接词（多实体/多约束）
    "和", "与", "跟", "一起", "同时", "以及", "还有",
    # 时间约束
    "之前", "之后", "期间", "时候", "那时", "当时",
    # 因果/推理
    "为什么", "怎么", "如何", "原因", "因为", "所以", "导致",
    # 条件/比较
    "如果", "假如", "相比", "比较", "不同", "区别",
    # 多跳推理信号
    "关于", "有关", "涉及", "相关",
    # 英文
    "why", "how", "because", "compared", "between", "about",
]

# 高价值查询模式（语义复杂度更高，用正则匹配）
_HIGH_VALUE_QUERY_PATTERNS = [
    r"(?:什么|哪些).*(?:和|与|跟)",   # "什么和什么"
    r"(?:谁|哪个).*(?:之间|关系)",    # "谁和谁之间"
    r"(?:怎么|如何).*(?:变化|改变)",  # "怎么变化的"
    r"(?:为什么).*(?:不|没|却)",         # "为什么不..."
    r"(?:从.*到).*(?:变|成)",            # "从...到...变化"
    r"(?:what|which).*(?:and|between|with)",    # English multi-entity
    r"(?:how|why).*(?:change|differ|affect)",   # English causation
]


@dataclass
class RoutingDetectionResult(BaseDetectionResult):
    """路由检测结果"""
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR_ONLY
    query_type: str = "simple"
    key_entities: List[str] = field(default_factory=list)


class LLMRetrievalRouter(LLMEnhancedDetector[RoutingDetectionResult]):
    """LLM增强检索路由器
    
    支持三种模式：
    - rule: 仅使用规则路由（快速）
    - llm: 仅使用LLM路由（意图理解）
    - hybrid: 混合（推荐）
    """
    
    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        mode: DetectionMode = DetectionMode.RULE,
        daily_limit: int = 0,
    ):
        super().__init__(
            astrbot_context=astrbot_context,
            provider_id=provider_id,
            mode=mode,
            daily_limit=daily_limit,
            max_tokens=200,
        )
        self._rule_router = RetrievalRouter()
    
    def _should_skip_input(self, query: str = "", **kwargs) -> bool:
        """空查询时跳过"""
        return not query
    
    def _get_empty_result(self) -> RoutingDetectionResult:
        """空输入默认结果"""
        return RoutingDetectionResult(
            confidence=1.0,
            source="rule",
            reason="空查询",
        )
    
    def _rule_detect(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDetectionResult:
        """规则路由"""
        strategy = self._rule_router.route(query, context)
        analysis = self._rule_router.analyze_query_complexity(query)
        
        return RoutingDetectionResult(
            strategy=strategy,
            confidence=0.7,
            reason=f"规则路由: {analysis['complexity']}",
            query_type=analysis["complexity"],
            source="rule",
        )
    
    def _build_prompt(self, query: str, **kwargs) -> str:
        """构建LLM提示词"""
        return RETRIEVAL_ROUTING_PROMPT.format(query=query[:300])
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> RoutingDetectionResult:
        """解析LLM结果"""
        return RoutingDetectionResult(
            strategy=BaseDetectionResult.map_enum(
                data.get("strategy", "vector_only"),
                _STRATEGY_MAP,
                RetrievalStrategy.VECTOR_ONLY,
            ),
            confidence=BaseDetectionResult.parse_confidence(data),
            reason=data.get("reason", "LLM判断"),
            query_type=data.get("query_type", "simple"),
            key_entities=BaseDetectionResult.ensure_list(
                data.get("key_entities", [])
            ),
            source="llm",
        )
    
    async def _hybrid_detect(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDetectionResult:
        """混合路由：规则快速判断 + LLM复杂确认"""
        rule_result = self._rule_detect(query, context)
        
        # VECTOR_ONLY 且可能复杂 → LLM确认
        if rule_result.strategy == RetrievalStrategy.VECTOR_ONLY:
            if len(query) > 20 or self._might_be_complex(query):
                llm_result = await self._llm_detect(query, context=context)
                if llm_result.confidence >= 0.6:
                    llm_result.source = "hybrid"
                    return llm_result
        
        # HYBRID策略 → LLM确认
        if rule_result.strategy == RetrievalStrategy.HYBRID:
            llm_result = await self._llm_detect(query, context=context)
            if llm_result.confidence >= 0.6:
                llm_result.source = "hybrid"
                return llm_result
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _might_be_complex(self, query: str) -> bool:
        """检查是否可能是复杂查询
        
        多层检测：
        1. 复杂指示词计数（≥2 个立即判定）
        2. 高价值模式正则匹配
        3. 实体数量估计（多个专有名词）
        4. 单个指示词 + 较长文本
        """
        # 1. 复杂指示词计数
        indicator_count = sum(1 for ind in _COMPLEX_INDICATORS if ind in query)
        if indicator_count >= 2:
            return True
        
        # 2. 高价值模式匹配
        for pattern in _HIGH_VALUE_QUERY_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # 3. 实体数量估计（引号内容或专有名词）
        quoted_entities = re.findall(r'[「」“”\'\"]+', query)
        if len(quoted_entities) >= 2:
            return True
        
        # 4. 单个指示词 + 较长文本
        if indicator_count >= 1 and len(query) > 15:
            return True
        
        return False
    

"""
LLM升级评估器
使用LLM来判断记忆是否应该升级存储层
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum

from datetime import datetime

from iris_memory.utils.logger import get_logger
from iris_memory.core.types import StorageLayer, QualityLevel
from iris_memory.models.memory import Memory

logger = get_logger("upgrade_evaluator")


def _get_storage_layer_value(storage_layer) -> str:
    """安全获取 storage_layer 的字符串值
    
    Args:
        storage_layer: 可能是 StorageLayer 枚举、字符串或整数
        
    Returns:
        str: storage_layer 的字符串值
    """
    if isinstance(storage_layer, StorageLayer):
        return storage_layer.value
    elif isinstance(storage_layer, str):
        return storage_layer
    else:
        return str(storage_layer)


def _get_quality_level_value(quality_level) -> int:
    """安全获取 quality_level 的整数值
    
    Args:
        quality_level: 可能是 QualityLevel 枚举、整数或字符串
        
    Returns:
        int: quality_level 的整数值
    """
    if isinstance(quality_level, QualityLevel):
        return quality_level.value
    elif isinstance(quality_level, int):
        return quality_level
    elif isinstance(quality_level, str):
        try:
            return int(quality_level)
        except ValueError:
            try:
                return QualityLevel[quality_level].value
            except KeyError:
                return QualityLevel.MODERATE.value
    else:
        return QualityLevel.MODERATE.value


class UpgradeMode(str, Enum):
    """升级判断模式"""
    RULE = "rule"      # 仅使用规则判断
    LLM = "llm"        # 仅使用LLM判断
    HYBRID = "hybrid"  # 混合模式：三层评估策略


class EvaluationTier(str, Enum):
    """评估层级"""
    HIGH_CONFIDENCE_PASS = "high_confidence_pass"  # 高置信度通过，无需LLM
    BOUNDARY = "boundary"                          # 边界情况，需要LLM确认
    CLEAR_REJECT = "clear_reject"                  # 明确拒绝，无需LLM


class UpgradeEvaluator:
    """LLM升级评估器
    
    使用LLM来评估记忆是否应该升级到更高的存储层
    
    优化特性：
    - 全局并发信号量限制LLM请求数
    - 三层评估策略减少不必要的LLM调用
    - 请求间隔防止密集调用
    - 评估去重机制避免重复评估
    """
    
    _llm_semaphore: Optional[asyncio.Semaphore] = None
    _max_concurrent_llm: int = 3
    _evaluating_memories: Set[str] = set()
    _request_interval: float = 0.5
    
    WORKING_TO_EPISODIC_PROMPT = """你是一个记忆管理系统的评估助手。请判断以下工作记忆是否值得长期保存。

## 评估标准
工作记忆应该升级到情景记忆（长期保存），如果满足以下任一条件：
1. **个人重要信息**：包含用户的姓名、生日、职业、爱好、偏好等个人特征
2. **关系信息**：涉及用户的家人、朋友、同事等人际关系
3. **情感意义**：记录了重要的情感体验或事件
4. **反复提及**：用户多次提到或强调的内容（访问次数多）
5. **明确请求**：用户明确表示希望记住的内容

## 待评估的记忆
{memories_json}

## 输出格式
请以JSON格式返回评估结果，每条记忆一个结果：
```json
{{
  "results": [
    {{
      "memory_id": "记忆ID",
      "should_upgrade": true/false,
      "confidence": 0.0-1.0,
      "reason": "简短的升级/不升级理由"
    }}
  ]
}}
```

请仅返回JSON，不要有其他文字。"""

    # EPISODIC → SEMANTIC 评估提示词
    EPISODIC_TO_SEMANTIC_PROMPT = """你是一个记忆管理系统的评估助手。请判断以下情景记忆是否应该升级为永久核心记忆。

## 评估标准
情景记忆应该升级到语义记忆（永久保存），如果满足以下任一条件：
1. **核心身份信息**：用户的基本身份特征，如姓名、生日、性别等
2. **稳定偏好**：用户长期稳定的喜好、习惯、价值观
3. **重要关系**：密切的家庭成员、挚友等核心社交关系
4. **频繁访问**：被频繁检索和使用的记忆（访问次数>=5次）
5. **较高置信度**：多次验证、高度可信的事实性记忆（置信度>0.65）
6. **高重要性**：重要性评分>=0.8且被访问>=3次的记忆
7. **时间验证**：存在超过7天且被访问>=3次且置信度>0.6的记忆

## 待评估的记忆
{memories_json}

## 输出格式
请以JSON格式返回评估结果：
```json
{{
  "results": [
    {{
      "memory_id": "记忆ID",
      "should_upgrade": true/false,
      "confidence": 0.0-1.0,
      "reason": "简短的升级/不升级理由"
    }}
  ]
}}
```

请仅返回JSON，不要有其他文字。"""

    def __init__(
        self,
        llm_provider=None,
        mode: UpgradeMode = UpgradeMode.HYBRID,
        batch_size: int = 10,
        confidence_threshold: float = 0.7,
        max_concurrent_llm: int = 3,
        request_interval: float = 0.5
    ):
        """初始化升级评估器
        
        Args:
            llm_provider: LLM提供者（需要支持llm_request方法）
            mode: 升级判断模式
            batch_size: 批量评估大小（默认10，从5增加）
            confidence_threshold: 置信度阈值
            max_concurrent_llm: 最大并发LLM请求数
            request_interval: LLM请求间隔（秒）
        """
        self.llm_provider = llm_provider
        self.mode = mode
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        UpgradeEvaluator._max_concurrent_llm = max_concurrent_llm
        UpgradeEvaluator._request_interval = request_interval
        
        if UpgradeEvaluator._llm_semaphore is None:
            UpgradeEvaluator._llm_semaphore = asyncio.Semaphore(max_concurrent_llm)
    
    def set_llm_provider(self, llm_provider):
        """设置LLM提供者
        
        Args:
            llm_provider: LLM提供者
        """
        self.llm_provider = llm_provider
        logger.debug("LLM provider set for UpgradeEvaluator")
    
    async def evaluate_working_to_episodic(
        self,
        memories: List[Memory]
    ) -> Dict[str, Tuple[bool, float, str]]:
        """评估工作记忆是否应升级到情景记忆
        
        Args:
            memories: 待评估的工作记忆列表
            
        Returns:
            Dict[str, Tuple[bool, float, str]]: {memory_id: (should_upgrade, confidence, reason)}
        """
        if self.mode == UpgradeMode.RULE:
            # 仅规则判断
            return self._rule_based_evaluation(memories, "working_to_episodic")
        
        elif self.mode == UpgradeMode.LLM:
            # 仅LLM判断
            return await self._llm_evaluation(memories, "working_to_episodic")
        
        else:  # HYBRID
            # 混合模式：规则预筛 + LLM确认
            return await self._hybrid_evaluation(memories, "working_to_episodic")
    
    async def evaluate_episodic_to_semantic(
        self,
        memories: List[Memory]
    ) -> Dict[str, Tuple[bool, float, str]]:
        """评估情景记忆是否应升级到语义记忆
        
        Args:
            memories: 待评估的情景记忆列表
            
        Returns:
            Dict[str, Tuple[bool, float, str]]: {memory_id: (should_upgrade, confidence, reason)}
        """
        if self.mode == UpgradeMode.RULE:
            return self._rule_based_evaluation(memories, "episodic_to_semantic")
        
        elif self.mode == UpgradeMode.LLM:
            return await self._llm_evaluation(memories, "episodic_to_semantic")
        
        else:  # HYBRID
            return await self._hybrid_evaluation(memories, "episodic_to_semantic")
    
    def _classify_evaluation_tier(self, memory: Memory, upgrade_type: str) -> EvaluationTier:
        """将记忆分类到评估层级
        
        三层策略：
        - 高置信度通过：指标明显超出阈值，直接升级
        - 边界情况：指标接近阈值，需要LLM确认
        - 明确拒绝：指标明显不足，直接拒绝
        
        Args:
            memory: 记忆对象
            upgrade_type: 升级类型
            
        Returns:
            EvaluationTier: 评估层级
        """
        if upgrade_type == "working_to_episodic":
            if (
                memory.is_user_requested or
                (memory.access_count >= 5 and memory.importance_score > 0.6) or
                memory.emotional_weight > 0.8 or
                memory.confidence >= 0.85 or
                (memory.rif_score > 0.7 and memory.access_count >= 3) or
                _get_quality_level_value(memory.quality_level) >= QualityLevel.CONFIRMED.value
            ):
                return EvaluationTier.HIGH_CONFIDENCE_PASS
            
            if (
                memory.access_count < 2 and
                memory.importance_score <= 0.3 and
                memory.emotional_weight <= 0.4 and
                memory.confidence < 0.5 and
                memory.rif_score <= 0.3 and
                not memory.is_user_requested and
                _get_quality_level_value(memory.quality_level) < QualityLevel.HIGH_CONFIDENCE.value
            ):
                return EvaluationTier.CLEAR_REJECT
            
            return EvaluationTier.BOUNDARY
        
        else:  # episodic_to_semantic
            if (
                _get_quality_level_value(memory.quality_level) == QualityLevel.CONFIRMED.value or
                (memory.access_count >= 10 and memory.confidence > 0.7) or
                (memory.importance_score >= 0.9 and memory.access_count >= 5) or
                (memory.access_count >= 8 and memory.confidence > 0.8)
            ):
                return EvaluationTier.HIGH_CONFIDENCE_PASS
            
            if (
                memory.access_count < 2 and
                memory.confidence <= 0.4 and
                memory.importance_score < 0.5
            ):
                return EvaluationTier.CLEAR_REJECT
            
            return EvaluationTier.BOUNDARY
    
    def _rule_based_evaluation(
        self,
        memories: List[Memory],
        upgrade_type: str
    ) -> Dict[str, Tuple[bool, float, str]]:
        """基于规则的评估
        
        Args:
            memories: 待评估的记忆列表
            upgrade_type: 升级类型
            
        Returns:
            评估结果字典
        """
        results = {}
        
        for memory in memories:
            if upgrade_type == "working_to_episodic":
                should_upgrade = memory.should_upgrade_to_episodic()
                reason = self._get_working_upgrade_reason(memory, should_upgrade)
                confidence = self._calculate_rule_confidence(memory, upgrade_type) if should_upgrade else 0.2
            else:
                should_upgrade = memory.should_upgrade_to_semantic()
                reason = self._get_episodic_upgrade_reason(memory, should_upgrade)
                confidence = self._calculate_rule_confidence(memory, upgrade_type) if should_upgrade else 0.2
            
            results[memory.id] = (should_upgrade, confidence, reason)
        
        return results
    
    def _calculate_rule_confidence(self, memory: Memory, upgrade_type: str) -> float:
        """根据满足条件的数量和强度动态计算规则判断的置信度
        
        满足的条件越多、指标值越高，置信度越高。
        
        Args:
            memory: 记忆对象
            upgrade_type: 升级类型
            
        Returns:
            float: 置信度（0.5-0.95）
        """
        base_confidence = 0.5
        bonus = 0.0
        
        if upgrade_type == "working_to_episodic":
            if memory.is_user_requested:
                bonus += 0.3
            if _get_quality_level_value(memory.quality_level) >= QualityLevel.HIGH_CONFIDENCE.value:
                bonus += 0.2
            # 高置信度
            if memory.confidence >= 0.7:
                bonus += 0.15
            elif memory.confidence >= 0.5:
                bonus += 0.05
            # 多次访问
            if memory.access_count >= 5:
                bonus += 0.15
            elif memory.access_count >= 3:
                bonus += 0.08
            # 情感权重
            if memory.emotional_weight > 0.6:
                bonus += 0.1
            # RIF评分
            if memory.rif_score > 0.7:
                bonus += 0.1
            elif memory.rif_score > 0.5:
                bonus += 0.05
        else:  # episodic_to_semantic
            # CONFIRMED 质量
            if memory.quality_level == QualityLevel.CONFIRMED:
                bonus += 0.3
            # 频繁访问
            if memory.access_count >= 10:
                bonus += 0.2
            elif memory.access_count >= 5:
                bonus += 0.15
            elif memory.access_count >= 3:
                bonus += 0.08
            # 高置信度
            if memory.confidence > 0.8:
                bonus += 0.15
            elif memory.confidence > 0.65:
                bonus += 0.1
            # 高重要性
            if memory.importance_score >= 0.8:
                bonus += 0.1
            # 时间验证（存在超过7天）
            days_since_creation = (datetime.now() - memory.created_time).days
            if days_since_creation >= 30:
                bonus += 0.1
            elif days_since_creation >= 7:
                bonus += 0.05
        
        return min(0.95, base_confidence + bonus)
    
    def _get_working_upgrade_reason(self, memory: Memory, should_upgrade: bool) -> str:
        """获取工作记忆升级原因
        
        条件与 Memory.should_upgrade_to_episodic() 保持完全一致
        """
        if should_upgrade:
            if memory.is_user_requested:
                return "用户主动请求保存"
            elif memory.access_count >= 3 and memory.importance_score > 0.5:
                return f"访问{memory.access_count}次且重要性{memory.importance_score:.2f}"
            elif memory.emotional_weight > 0.6:
                return f"情感权重高({memory.emotional_weight:.2f})"
            elif memory.confidence >= 0.7:
                return f"置信度较高({memory.confidence:.2f})"
            elif memory.rif_score > 0.5 and memory.access_count >= 2:
                return f"RIF评分较高({memory.rif_score:.2f})且已被访问{memory.access_count}次"
            elif _get_quality_level_value(memory.quality_level) >= QualityLevel.HIGH_CONFIDENCE.value:
                ql_val = _get_quality_level_value(memory.quality_level)
                ql_name = QualityLevel(ql_val).name if ql_val <= QualityLevel.CONFIRMED.value else str(ql_val)
                return f"质量等级较高({ql_name})"
            else:
                return "满足升级条件"
        else:
            return "未达到升级条件"
    
    def _get_episodic_upgrade_reason(self, memory: Memory, should_upgrade: bool) -> str:
        """获取情景记忆升级原因
        
        条件与 Memory.should_upgrade_to_semantic() 保持完全一致
        """
        if should_upgrade:
            if memory.access_count >= 5 and memory.confidence > 0.65:
                return f"频繁访问({memory.access_count}次)且置信度较高({memory.confidence:.2f})"
            elif _get_quality_level_value(memory.quality_level) == QualityLevel.CONFIRMED.value:
                return "质量已确认"
            elif memory.importance_score >= 0.8 and memory.access_count >= 3:
                return f"高重要性({memory.importance_score:.2f})且已被访问{memory.access_count}次"
            else:
                days_since_creation = (datetime.now() - memory.created_time).days
                if days_since_creation >= 7 and memory.access_count >= 3 and memory.confidence > 0.6:
                    return f"存在{days_since_creation}天且访问{memory.access_count}次，置信度{memory.confidence:.2f}"
                return "满足升级条件"
        else:
            return "未达到升级条件"
    
    async def _llm_evaluation(
        self,
        memories: List[Memory],
        upgrade_type: str
    ) -> Dict[str, Tuple[bool, float, str]]:
        """基于LLM的评估（带并发控制和去重）
        
        Args:
            memories: 待评估的记忆列表
            upgrade_type: 升级类型
            
        Returns:
            评估结果字典
        """
        if not self.llm_provider:
            logger.warning("No LLM provider available, falling back to rule-based")
            return self._rule_based_evaluation(memories, upgrade_type)
        
        to_evaluate = []
        for memory in memories:
            if memory.id not in self._evaluating_memories:
                self._evaluating_memories.add(memory.id)
                to_evaluate.append(memory)
            else:
                logger.debug(f"Memory {memory.id} is already being evaluated, skipping")
        
        if not to_evaluate:
            return {}
        
        results = {}
        
        try:
            for i in range(0, len(to_evaluate), self.batch_size):
                batch = to_evaluate[i:i + self.batch_size]
                batch_results = await self._evaluate_batch_with_llm(batch, upgrade_type)
                results.update(batch_results)
                
                if i + self.batch_size < len(to_evaluate):
                    await asyncio.sleep(self._request_interval)
        finally:
            for memory in to_evaluate:
                self._evaluating_memories.discard(memory.id)
        
        return results
    
    async def _evaluate_batch_with_llm(
        self,
        memories: List[Memory],
        upgrade_type: str
    ) -> Dict[str, Tuple[bool, float, str]]:
        """使用LLM评估一批记忆（带信号量控制）
        
        Args:
            memories: 记忆批次
            upgrade_type: 升级类型
            
        Returns:
            评估结果字典
        """
        # 准备记忆数据
        memories_data = []
        for memory in memories:
            memories_data.append({
                "id": memory.id,
                "content": memory.content,
                "type": memory.type.value,
                "importance_score": memory.importance_score,
                "emotional_weight": memory.emotional_weight,
                "access_count": memory.access_count,
                "confidence": memory.confidence,
                "created_time": memory.created_time.isoformat()
            })
        
        memories_json = json.dumps(memories_data, ensure_ascii=False, indent=2)
        
        # 选择提示词
        if upgrade_type == "working_to_episodic":
            prompt = self.WORKING_TO_EPISODIC_PROMPT.format(memories_json=memories_json)
        else:
            prompt = self.EPISODIC_TO_SEMANTIC_PROMPT.format(memories_json=memories_json)
        
        try:
            async with self._llm_semaphore:
                response = await self._call_llm(prompt)
                if not response or not response.strip():
                    logger.warning("LLM returned empty response (rate limit or error), falling back to rule-based evaluation")
                    return self._rule_based_evaluation(memories, upgrade_type)
                return self._parse_llm_response(response, memories)
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return self._rule_based_evaluation(memories, upgrade_type)
    
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM
        
        Args:
            prompt: 提示词
            
        Returns:
            LLM响应文本
        """
        if not self.llm_provider:
            raise ValueError("No LLM provider available")
        
        # 使用LLMMessageProcessor的_call_llm方法
        if hasattr(self.llm_provider, '_call_llm'):
            response = await self.llm_provider._call_llm(prompt, max_tokens=500)
            return response or ""
        
        # 直接调用text_chat
        if hasattr(self.llm_provider, 'text_chat'):
            response = await self.llm_provider.text_chat(prompt)
            if isinstance(response, dict):
                return response.get("text", "") or response.get("content", "")
            return str(response) if response else ""
        
        raise ValueError("LLM provider does not support any known LLM method")
    
    def _parse_llm_response(
        self,
        response: str,
        memories: List[Memory]
    ) -> Dict[str, Tuple[bool, float, str]]:
        """解析LLM响应
        
        Args:
            response: LLM响应文本
            memories: 原始记忆列表
            
        Returns:
            评估结果字典
        """
        results = {}
        
        try:
            # 尝试提取JSON
            response = response.strip()
            
            # 处理可能的markdown代码块
            if response.startswith("```"):
                lines = response.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response = "\n".join(json_lines)
            
            data = json.loads(response)
            
            # 处理结果
            for result in data.get("results", []):
                memory_id = result.get("memory_id")
                should_upgrade = result.get("should_upgrade", False)
                confidence = result.get("confidence", 0.5)
                reason = result.get("reason", "LLM判断")
                
                # 应用置信度阈值
                if should_upgrade and confidence < self.confidence_threshold:
                    should_upgrade = False
                    reason = f"置信度{confidence:.2f}低于阈值{self.confidence_threshold}"
                
                results[memory_id] = (should_upgrade, confidence, reason)
            
            # 确保所有记忆都有结果
            for memory in memories:
                if memory.id not in results:
                    logger.warning(f"Memory {memory.id} not in LLM response")
                    layer_val = _get_storage_layer_value(memory.storage_layer)
                    should_upgrade = memory.should_upgrade_to_episodic() if layer_val == "working" else memory.should_upgrade_to_semantic()
                    results[memory.id] = (should_upgrade, 0.5, "LLM未返回结果，使用规则判断")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response}")
            for memory in memories:
                layer_val = _get_storage_layer_value(memory.storage_layer)
                should_upgrade = memory.should_upgrade_to_episodic() if layer_val == "working" else memory.should_upgrade_to_semantic()
                results[memory.id] = (should_upgrade, 0.5, "JSON解析失败，使用规则判断")
        
        return results
    
    async def _hybrid_evaluation(
        self,
        memories: List[Memory],
        upgrade_type: str
    ) -> Dict[str, Tuple[bool, float, str]]:
        """优化后的混合模式评估
        
        三层策略：
        1. 高置信度通过 → 直接升级（不调用LLM）
        2. 边界情况 → LLM确认
        3. 明确拒绝 → 直接拒绝（不调用LLM）
        
        Args:
            memories: 待评估的记忆列表
            upgrade_type: 升级类型
            
        Returns:
            评估结果字典
        """
        results = {}
        high_confidence_pass = []
        boundary_cases = []
        clear_reject = []
        
        for memory in memories:
            tier = self._classify_evaluation_tier(memory, upgrade_type)
            
            if tier == EvaluationTier.HIGH_CONFIDENCE_PASS:
                confidence = self._calculate_rule_confidence(memory, upgrade_type)
                reason = self._get_working_upgrade_reason(memory, True) if upgrade_type == "working_to_episodic" else self._get_episodic_upgrade_reason(memory, True)
                results[memory.id] = (True, confidence, reason)
                high_confidence_pass.append(memory.id)
                
            elif tier == EvaluationTier.CLEAR_REJECT:
                results[memory.id] = (False, 0.1, "未达到升级条件")
                clear_reject.append(memory.id)
                
            else:  # BOUNDARY
                boundary_cases.append(memory)
        
        if boundary_cases and self.llm_provider:
            llm_results = await self._llm_evaluation(boundary_cases, upgrade_type)
            results.update(llm_results)
        elif boundary_cases:
            rule_results = self._rule_based_evaluation(boundary_cases, upgrade_type)
            results.update(rule_results)
        
        if len(memories) > 0:
            total = len(memories)
            pass_count = len(high_confidence_pass)
            boundary_count = len(boundary_cases)
            reject_count = len(clear_reject)
            llm_saved = pass_count + reject_count
            
            logger.debug(
                f"Hybrid evaluation ({upgrade_type}): "
                f"{pass_count} high-confidence pass ({pass_count*100//total}%), "
                f"{boundary_count} boundary (LLM) ({boundary_count*100//total}%), "
                f"{reject_count} clear reject ({reject_count*100//total}%), "
                f"LLM calls saved: {llm_saved}/{total} ({llm_saved*100//total}%)"
            )
        
        return results

"""
语义提取器（通道 B 主控）

编排 聚类 → LLM 提取 → 置信度计算 → 存储 的完整流程。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer, QualityLevel
from iris_memory.capture.semantic.semantic_clustering import (
    SemanticClustering,
    MemoryCluster,
)
from iris_memory.capture.semantic.semantic_confidence import (
    SemanticConfidenceCalculator,
    ConfidenceResult,
)
from iris_memory.utils.logger import get_logger
from iris_memory.utils.llm_helper import call_llm, parse_llm_json, resolve_llm_provider

if TYPE_CHECKING:
    from iris_memory.storage.chroma_manager import ChromaManager

logger = get_logger("semantic_extractor")


# ── LLM 提取 Prompt ──

SEMANTIC_EXTRACTION_PROMPT = """你是一个记忆管理系统的语义提取助手。请从以下情景记忆聚类中提取核心语义信息。

## 任务
将多条具体的情景记忆抽象为一条简洁的语义记忆（去除时间戳，保留核心含义）。

## 用户信息
用户ID：{user_id}

## 聚类主题
主题/实体：{cluster_key}

## 情景记忆列表
{memories_text}

## 输出格式
请以JSON格式返回提取结果：
```json
{{
  "content": "抽象化的语义内容（简洁、无时间戳、表达核心含义）",
  "type": "fact/emotion/relationship",
  "subtype": "preference/habit/relationship/trait/interest/value",
  "contradiction_ids": ["如有矛盾记忆的ID，列在此处，无矛盾则为空数组"]
}}
```

## 注意事项
1. content 应该是抽象的、概括性的描述，不包含具体时间，应以用户视角表达
2. 如果记忆之间有矛盾（如"喜欢X"和"不喜欢X"），请在 contradiction_ids 中标注矛盾记忆的ID
3. type 选择最匹配的记忆类型
4. subtype 描述语义记忆的子分类

请仅返回JSON，不要有其他文字。"""


# ── 提取结果 ──

class ExtractionResult:
    """单个聚类的语义提取结果"""

    def __init__(
        self,
        cluster: MemoryCluster,
        semantic_memory: Optional[Memory],
        confidence_result: ConfidenceResult,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        self.cluster = cluster
        self.semantic_memory = semantic_memory
        self.confidence_result = confidence_result
        self.success = success
        self.error = error


class SemanticExtractor:
    """语义提取器

    编排完整的通道 B 流程：
    1. 聚类 EPISODIC 记忆
    2. 对每个聚类调用 LLM 提取核心语义
    3. 计算置信度
    4. 创建 SEMANTIC 记忆
    5. 标记已提取的源记忆
    """

    def __init__(
        self,
        chroma_manager: Optional[ChromaManager] = None,
        astrbot_context: Any = None,
        provider_id: str = "",
        clustering: Optional[SemanticClustering] = None,
        confidence_calculator: Optional[SemanticConfidenceCalculator] = None,
        source_expiry_days: int = 0,
        llm_max_tokens: int = 500,
        use_vector_clustering: bool = False,
    ) -> None:
        self.chroma_manager = chroma_manager
        self.astrbot_context = astrbot_context
        self._provider_id = provider_id
        self._llm_provider = None
        self._llm_resolved_id: Optional[str] = None
        self.clustering = clustering or SemanticClustering()
        self.confidence_calculator = confidence_calculator or SemanticConfidenceCalculator()
        self.source_expiry_days = source_expiry_days
        self.llm_max_tokens = llm_max_tokens
        self.use_vector_clustering = use_vector_clustering

    def set_llm_provider(self, provider: Any, provider_id: str = "") -> None:
        """显式设置 LLM 提供者"""
        self._llm_provider = provider
        self._llm_resolved_id = provider_id

    def set_chroma_manager(self, chroma_manager: ChromaManager) -> None:
        """延迟注入 ChromaManager"""
        self.chroma_manager = chroma_manager

    # ── 主入口 ──

    async def run(self) -> List[ExtractionResult]:
        """执行一轮完整的语义提取

        Returns:
            提取结果列表
        """
        if not self.chroma_manager:
            logger.warning("SemanticExtractor: chroma_manager not available, skipping")
            return []

        # 1. 获取所有 EPISODIC 记忆
        episodic_memories = await self.chroma_manager.get_memories_by_storage_layer(
            StorageLayer.EPISODIC
        )
        if not episodic_memories:
            logger.debug("No episodic memories found for semantic extraction")
            return []

        logger.debug(f"Semantic extraction: processing {len(episodic_memories)} episodic memories")

        # 2. 聚类
        if self.use_vector_clustering:
            # 填充缺失的嵌入向量
            await self._ensure_embeddings(episodic_memories)
            clusters = self.clustering.cluster_with_vectors(episodic_memories)
        else:
            clusters = self.clustering.cluster(episodic_memories)

        if not clusters:
            logger.debug("No clusters produced from episodic memories")
            return []

        logger.debug(f"Produced {len(clusters)} clusters for extraction")

        # 3. 对每个聚类执行 LLM 提取
        results: List[ExtractionResult] = []
        for cluster in clusters:
            result = await self._extract_cluster(cluster)
            results.append(result)

        # 4. 持久化结果
        success_count = 0
        for result in results:
            if result.success and result.semantic_memory:
                persisted = await self._persist_result(result)
                if persisted:
                    success_count += 1

        logger.info(
            f"Semantic extraction completed: "
            f"{len(clusters)} clusters → {success_count} semantic memories created"
        )
        return results

    async def extract_from_clusters(
        self,
        clusters: List[MemoryCluster],
    ) -> List[ExtractionResult]:
        """从给定聚类列表提取（可用于冲突解决后重新评估）

        Args:
            clusters: 预构建的聚类列表

        Returns:
            提取结果列表
        """
        results: List[ExtractionResult] = []
        for cluster in clusters:
            result = await self._extract_cluster(cluster)
            results.append(result)
        return results

    # ── 单个聚类提取 ──

    async def _extract_cluster(self, cluster: MemoryCluster) -> ExtractionResult:
        """对单个聚类执行 LLM 语义提取"""
        try:
            # 解析 LLM provider
            self._ensure_llm_provider()

            # 构建 prompt
            memories_text = self._format_memories_for_prompt(cluster.memories)
            ref_memory = cluster.memories[0]
            prompt = SEMANTIC_EXTRACTION_PROMPT.format(
                user_id=cluster.user_id or ref_memory.user_id,
                cluster_key=cluster.cluster_key,
                memories_text=memories_text,
            )

            # 调用 LLM
            llm_result = await call_llm(
                self.astrbot_context,
                self._llm_provider,
                self._llm_resolved_id,
                prompt,
                parse_json=True,
            )

            if not llm_result.success or not llm_result.parsed_json:
                # LLM 失败，尝试从 content 再解析一次
                if llm_result.content:
                    parsed = parse_llm_json(llm_result.content)
                    if parsed:
                        llm_result.parsed_json = parsed

                if not llm_result.parsed_json:
                    logger.warning(
                        f"LLM extraction failed for cluster {cluster.cluster_id}: "
                        f"{llm_result.error or 'empty response'}"
                    )
                    return ExtractionResult(
                        cluster=cluster,
                        semantic_memory=None,
                        confidence_result=self.confidence_calculator.calculate(0),
                        success=False,
                        error=llm_result.error or "LLM returned empty/unparseable response",
                    )

            # 解析 LLM 响应
            data = llm_result.parsed_json
            content = data.get("content", "")
            memory_type_str = data.get("type", "fact")
            subtype = data.get("subtype", "")
            contradiction_ids = data.get("contradiction_ids", [])

            if not content:
                return ExtractionResult(
                    cluster=cluster,
                    semantic_memory=None,
                    confidence_result=self.confidence_calculator.calculate(0),
                    success=False,
                    error="LLM returned empty content",
                )

            # 计算置信度
            conf_result = self.confidence_calculator.calculate(
                evidence_count=cluster.size,
                contradiction_count=len(contradiction_ids) if contradiction_ids else 0,
            )

            # 映射记忆类型
            try:
                memory_type = MemoryType(memory_type_str)
            except ValueError:
                memory_type = MemoryType.FACT

            # 创建语义记忆（ref_memory 已在上方构建 prompt 时取得）
            # 低置信度设置为待审核状态
            review_status: Optional[str] = None
            if conf_result.needs_human_review:
                review_status = "pending_review"
            else:
                review_status = "approved"

            # 创建语义记忆
            semantic_memory = Memory(
                id=str(uuid.uuid4()),
                user_id=ref_memory.user_id,
                group_id=ref_memory.group_id,
                persona_id=ref_memory.persona_id,
                scope=ref_memory.scope,
                type=memory_type,
                subtype=subtype or None,
                content=content,
                confidence=conf_result.confidence,
                storage_layer=StorageLayer.SEMANTIC,
                quality_level=QualityLevel.MODERATE,
                source_type="semantic_extraction",
                evidence_ids=cluster.memory_ids,
                evidence_count=cluster.size,
                last_validated=datetime.now(),
                created_time=datetime.now(),
                importance_score=min(0.9, 0.5 + cluster.size * 0.05),
                review_status=review_status,
            )

            return ExtractionResult(
                cluster=cluster,
                semantic_memory=semantic_memory,
                confidence_result=conf_result,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error extracting cluster {cluster.cluster_id}: {e}", exc_info=True)
            return ExtractionResult(
                cluster=cluster,
                semantic_memory=None,
                confidence_result=self.confidence_calculator.calculate(0),
                success=False,
                error=str(e),
            )

    # ── 持久化 ──

    async def _persist_result(self, result: ExtractionResult) -> bool:
        """持久化提取结果

        1. 添加新的 SEMANTIC 记忆到 Chroma
        2. 标记源 EPISODIC 记忆为 summarized
        3. 可选：设置源记忆的 expires_at
        """
        if not self.chroma_manager or not result.semantic_memory:
            return False

        try:
            # 添加语义记忆
            added = await self.chroma_manager.add_memory(result.semantic_memory)
            if not added:
                logger.error(f"Failed to add semantic memory {result.semantic_memory.id}")
                return False

            # 标记源记忆
            for memory in result.cluster.memories:
                memory.summarized = True
                memory.semantic_memory_id = result.semantic_memory.id

                if self.source_expiry_days > 0:
                    memory.expires_at = datetime.now() + timedelta(days=self.source_expiry_days)

                try:
                    await self.chroma_manager.update_memory(memory)
                except Exception as e:
                    logger.warning(
                        f"Failed to mark source memory {memory.id} as summarized: {e}"
                    )

            logger.debug(
                f"Persisted semantic memory {result.semantic_memory.id} "
                f"from {result.cluster.size} source memories "
                f"(confidence={result.confidence_result.confidence:.2f})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to persist extraction result: {e}", exc_info=True)
            return False

    # ── 工具方法 ──

    async def _ensure_embeddings(self, memories: List[Memory]) -> None:
        """为缺少嵌入向量的记忆生成嵌入

        向量聚类需要嵌入向量。从 Chroma 获取的记忆通常已包含嵌入，
        但某些旧记忆可能缺失。此方法尝试通过 ChromaManager 的
        embedding_manager 重新生成。

        Args:
            memories: 记忆列表（就地更新 embedding 字段）
        """
        if not self.chroma_manager:
            return

        embedding_manager = getattr(self.chroma_manager, 'embedding_manager', None)
        if not embedding_manager or not getattr(embedding_manager, 'is_ready', False):
            return

        missing = [m for m in memories if m.embedding is None and m.content]
        if not missing:
            return

        logger.debug(f"Generating embeddings for {len(missing)} memories without vectors")
        for memory in missing:
            try:
                embedding = await embedding_manager.embed(memory.content)
                if embedding is not None:
                    memory.embedding = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
            except Exception as e:
                logger.debug(f"Failed to generate embedding for {memory.id}: {e}")

    def _ensure_llm_provider(self) -> None:
        """确保 LLM provider 可用（延迟解析）"""
        if self._llm_provider is not None:
            return

        provider, resolved_id = resolve_llm_provider(
            self.astrbot_context,
            self._provider_id,
            label="SemanticExtractor",
        )
        self._llm_provider = provider
        self._llm_resolved_id = resolved_id

    @staticmethod
    def _format_memories_for_prompt(memories: List[Memory]) -> str:
        """格式化记忆列表为 LLM prompt 文本"""
        lines = []
        for i, memory in enumerate(memories, 1):
            date_str = memory.created_time.strftime("%Y-%m-%d")
            lines.append(
                f"- 记忆{i} (ID={memory.id}): "
                f"'{memory.content}' ({date_str}, "
                f"type={memory.type.value}, confidence={memory.confidence:.2f})"
            )
        return "\n".join(lines)

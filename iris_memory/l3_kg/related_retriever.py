"""
Iris Chat Memory - 相关记忆检索器

用于 L3 知识图谱提取任务，检索与目标记忆相关的其他记忆。
支持三种检索方式的组合：
- 语义相似记忆
- 同群聊记忆
- 同用户记忆
"""

from typing import List, Optional, TYPE_CHECKING, cast

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.l2_memory import MemoryEntry, MemorySearchResult
from iris_memory.l2_memory.adapter import L2MemoryAdapter

if TYPE_CHECKING:
    from iris_memory.core import ComponentManager
    from iris_memory.l2_memory import L2MemoryAdapter

logger = get_logger("l3_kg.related_retriever")


class RelatedMemoryRetriever:
    """相关记忆检索器

    检索与目标记忆相关的其他记忆，用于 L3 知识图谱提取。

    检索策略：
    1. 语义相似：基于向量相似度检索
    2. 同群聊：同一群聊内的最近记忆
    3. 同用户：同一用户的最近记忆

    三种结果按权重合并，返回 top_k 条相关记忆。

    Attributes:
        _component_manager: 组件管理器引用
        _adapter: L2 记忆适配器
    """

    def __init__(self, component_manager: "ComponentManager"):
        """初始化检索器

        Args:
            component_manager: 组件管理器实例
        """
        self._component_manager = component_manager
        self._adapter: Optional["L2MemoryAdapter"] = None

    def _get_adapter(self) -> Optional["L2MemoryAdapter"]:
        """获取 L2 适配器

        Returns:
            L2MemoryAdapter 实例，不可用时返回 None
        """
        if self._adapter is None:
            adapter = self._component_manager.get_component(
                "l2_memory", L2MemoryAdapter
            )
            if adapter and adapter.is_available:
                self._adapter = adapter
        return self._adapter

    async def retrieve_related(
        self, memory: MemoryEntry, top_k: int = 5
    ) -> List[MemoryEntry]:
        """检索与目标记忆相关的其他记忆

        Args:
            memory: 目标记忆
            top_k: 最大返回数量

        Returns:
            相关记忆列表（不包含目标记忆本身）
        """
        adapter = self._get_adapter()
        if not adapter:
            logger.debug("L2 记忆库不可用，返回空列表")
            return []

        config = get_config()

        semantic_weight = cast(float, config.get("kg_extraction_semantic_weight"))
        same_group_weight = cast(float, config.get("kg_extraction_same_group_weight"))
        same_user_weight = cast(float, config.get("kg_extraction_same_user_weight"))

        # 归一化权重，防止用户改参数后权重和≠1.0 导致评分偏移
        weight_sum = semantic_weight + same_group_weight + same_user_weight
        if weight_sum > 0:
            semantic_weight /= weight_sum
            same_group_weight /= weight_sum
            same_user_weight /= weight_sum

        related_by_score: dict[str, float] = {}

        semantic_results: List[tuple[MemoryEntry, float]] = []
        group_results: List[tuple[MemoryEntry, float]] = []
        user_results: List[tuple[MemoryEntry, float]] = []

        if semantic_weight > 0:
            semantic_results = await self._retrieve_semantic(adapter, memory, top_k * 2)
            for entry, score in semantic_results:
                if entry.id != memory.id:
                    related_by_score[entry.id] = (
                        related_by_score.get(entry.id, 0) + score * semantic_weight
                    )

        if same_group_weight > 0 and memory.group_id:
            group_results = await self._retrieve_same_group(adapter, memory, top_k)
            for entry, score in group_results:
                if entry.id != memory.id:
                    related_by_score[entry.id] = (
                        related_by_score.get(entry.id, 0) + score * same_group_weight
                    )

        if same_user_weight > 0:
            user_id = memory.metadata.get("user_id")
            if user_id:
                user_results = await self._retrieve_same_user(
                    adapter, memory, user_id, top_k
                )
                for entry, score in user_results:
                    if entry.id != memory.id:
                        related_by_score[entry.id] = (
                            related_by_score.get(entry.id, 0) + score * same_user_weight
                        )

        sorted_ids = sorted(
            related_by_score.keys(), key=lambda x: related_by_score[x], reverse=True
        )

        all_entries: dict[str, MemoryEntry] = {}
        for entry, _ in semantic_results:
            all_entries[entry.id] = entry
        for entry, _ in group_results:
            all_entries[entry.id] = entry
        for entry, _ in user_results:
            all_entries[entry.id] = entry

        result = []
        for memory_id in sorted_ids[:top_k]:
            if memory_id in all_entries:
                result.append(all_entries[memory_id])

        logger.debug(f"检索到 {len(result)} 条相关记忆")
        return result

    async def _retrieve_semantic(
        self, adapter: "L2MemoryAdapter", memory: MemoryEntry, top_k: int
    ) -> List[tuple[MemoryEntry, float]]:
        """语义相似检索

        Args:
            adapter: L2 适配器
            memory: 目标记忆
            top_k: 返回数量

        Returns:
            (记忆条目, 分数) 列表
        """
        try:
            results: List[MemorySearchResult] = await adapter.retrieve(
                query=memory.content, top_k=top_k
            )
            return [(r.entry, r.score) for r in results]
        except Exception as e:
            logger.warning(f"语义相似检索失败: {e}")
            return []

    async def _retrieve_same_group(
        self, adapter: "L2MemoryAdapter", memory: MemoryEntry, top_k: int
    ) -> List[tuple[MemoryEntry, float]]:
        """同群聊记忆检索

        Args:
            adapter: L2 适配器
            memory: 目标记忆
            top_k: 返回数量

        Returns:
            (记忆条目, 分数) 列表
        """
        if not memory.group_id:
            return []

        try:
            group_entries = await adapter.get_entries_by_group(memory.group_id)

            group_entries = [e for e in group_entries if e.id != memory.id]

            group_entries.sort(key=lambda e: e.timestamp or "", reverse=True)

            return [(e, 1.0) for e in group_entries[:top_k]]
        except Exception as e:
            logger.warning(f"同群聊检索失败: {e}")
            return []

    async def _retrieve_same_user(
        self, adapter: "L2MemoryAdapter", memory: MemoryEntry, user_id: str, top_k: int
    ) -> List[tuple[MemoryEntry, float]]:
        """同用户记忆检索

        Args:
            adapter: L2 适配器
            memory: 目标记忆
            user_id: 用户ID
            top_k: 返回数量

        Returns:
            (记忆条目, 分数) 列表
        """
        try:
            user_entries = await adapter.get_entries_by_user(user_id)

            user_entries = [e for e in user_entries if e.id != memory.id]

            user_entries.sort(key=lambda e: e.timestamp or "", reverse=True)

            return [(e, 1.0) for e in user_entries[:top_k]]
        except Exception as e:
            logger.warning(f"同用户检索失败: {e}")
            return []

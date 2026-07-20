"""
Iris Chat Memory - L2 记忆检索器

提供记忆检索、写入和访问更新的高级接口。
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING

from iris_memory.core import get_logger, ComponentManager
from iris_memory.config import get_config
from iris_memory.utils import count_tokens
from .models import MemorySearchResult
from .adapter import L2MemoryAdapter

if TYPE_CHECKING:
    from iris_memory.llm.manager import LLMManager

logger = get_logger("l2_memory.retriever")


class MemoryRetriever:
    """记忆检索器

    提供记忆的高级检索和管理接口。

    Features:
        - 记忆检索（支持群聊隔离）
        - 从总结写入记忆
        - 访问频率更新
        - Token 预算控制

    Examples:
        >>> retriever = MemoryRetriever(component_manager)
        >>> results = await retriever.retrieve("喜欢吃什么", group_id="group_123")
    """

    def __init__(
        self,
        component_manager: ComponentManager,
        llm_manager: Optional["LLMManager"] = None,
    ):
        """初始化检索器

        Args:
            component_manager: 组件管理器实例
            llm_manager: LLM 调用管理器实例（可选）
        """
        self._manager = component_manager
        self._llm_manager = llm_manager
        self._adapter: Optional[L2MemoryAdapter] = None

    def _get_adapter(self) -> Optional[L2MemoryAdapter]:
        """获取 L2 适配器

        Returns:
            L2MemoryAdapter 实例，不可用时返回 None
        """
        if self._adapter is None:
            adapter = self._manager.get_component("l2_memory", L2MemoryAdapter)
            if adapter and adapter.is_available:
                self._adapter = adapter
        return self._adapter

    async def retrieve(
        self,
        query: str,
        group_id: Optional[str] = None,
        top_k: Optional[int] = None,
        persona_id: str = "default",
    ) -> List[MemorySearchResult]:
        """检索记忆

        根据查询文本检索相似记忆。

        Args:
            query: 查询文本
            group_id: 群聊 ID（可选，用于隔离检索）
            top_k: 返回数量，默认从配置读取
            persona_id: 人格ID（用于隔离检索）

        Returns:
            检索结果列表

        Examples:
            >>> results = await retriever.retrieve("用户喜欢什么")
            >>> len(results)
            10
        """
        config = get_config()

        adapter = self._get_adapter()
        if not adapter:
            logger.debug("L2 记忆库不可用，返回空结果")
            return []

        if top_k is None:
            top_k = config.get("l2_memory.top_k")

        enable_group_isolation = config.get(
            "isolation_config.enable_group_memory_isolation"
        )
        if not enable_group_isolation:
            group_id = None

        results = await adapter.retrieve(query, group_id, top_k, persona_id)

        relevance_threshold = config.get("l2_memory.relevance_threshold", 0.3)
        if relevance_threshold > 0:
            filtered = [r for r in results if r.score >= relevance_threshold]
            if len(filtered) < len(results):
                logger.debug(
                    f"相似度阈值过滤：{len(results)} -> {len(filtered)} 条 "
                    f"(阈值={relevance_threshold}, "
                    f"最高分={max(r.score for r in results):.4f}, "
                    f"最低分={min(r.score for r in results):.4f})"
                )
            results = filtered

        if results:
            memory_ids = [r.entry.id for r in results]
            await adapter.batch_update_access(memory_ids)

        logger.debug(f"检索到 {len(results)} 条记忆")
        return results

    async def add_from_summary(
        self,
        summary_content: str,
        metadata: Optional[Dict[str, Any]] = None,
        persona_id: str = "default",
    ) -> Optional[str]:
        """从总结写入记忆

        将总结内容添加到记忆库。

        Args:
            summary_content: 总结内容
            metadata: 元数据（group_id、user_id 等）
            persona_id: 人格ID

        Returns:
            记忆 ID，失败时返回 None
        """
        adapter = self._get_adapter()
        if not adapter:
            logger.warning("L2 记忆库不可用，跳过写入记忆")
            return None

        memory_id = await adapter.add_memory(
            summary_content, metadata, persona_id=persona_id
        )

        if memory_id:
            logger.info(f"已从总结写入记忆：{memory_id}")
        else:
            logger.warning("从总结写入记忆失败")

        return memory_id

    async def update_access(self, memory_id: str) -> bool:
        """更新记忆的访问信息

        增加访问次数并更新最近访问时间。

        Args:
            memory_id: 记忆 ID

        Returns:
            是否更新成功
        """
        adapter = self._get_adapter()
        if not adapter:
            return False

        return await adapter.update_access(memory_id)

    async def retrieve_for_context(
        self, query: str, group_id: Optional[str] = None, max_tokens: int = 2000
    ) -> str:
        """检索记忆并格式化为上下文文本

        检索记忆并格式化为适用于 LLM 上下文的文本，带 Token 预算控制。

        Args:
            query: 查询文本
            group_id: 群聊 ID
            max_tokens: 最大 Token 数

        Returns:
            格式化的上下文文本
        """
        results = await self.retrieve(query, group_id)

        if not results:
            return ""

        trimmed_results = self.trim_by_token_budget(results, max_tokens)

        context_lines = ["## 相关记忆"]
        for i, result in enumerate(trimmed_results, 1):
            context_lines.append(f"{i}. {result.entry.content}")

        return "\n".join(context_lines)

    @staticmethod
    def trim_by_token_budget(
        memories: List[MemorySearchResult], max_tokens: int
    ) -> List[MemorySearchResult]:
        """按 Token 预算裁剪记忆列表

        使用字符估算（平均 2 字符/token）从前往后逐条添加，
        直到超出预算。

        Args:
            memories: 记忆检索结果列表
            max_tokens: 最大 Token 预算

        Returns:
            裁剪后的记忆列表
        """
        trimmed: List[MemorySearchResult] = []
        total_tokens = 0

        for memory in memories:
            content = memory.entry.content
            memory_tokens = count_tokens(content) if content else 0

            if total_tokens + memory_tokens > max_tokens:
                if not trimmed:
                    trimmed.append(memory)
                break

            trimmed.append(memory)
            total_tokens += memory_tokens

        return trimmed

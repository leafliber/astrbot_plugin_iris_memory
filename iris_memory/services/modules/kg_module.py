"""
知识图谱模块 — 封装 KGStorage / KGExtractor / KGReasoning / KGContextFormatter

作为第 7 个 Feature Module 集成到 MemoryService 中。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.knowledge_graph.kg_storage import KGStorage
    from iris_memory.knowledge_graph.kg_extractor import KGExtractor
    from iris_memory.knowledge_graph.kg_reasoning import KGReasoning
    from iris_memory.knowledge_graph.kg_context import KGContextFormatter
    from iris_memory.knowledge_graph.kg_maintenance import KGMaintenanceManager
    from iris_memory.knowledge_graph.kg_consistency import KGConsistencyDetector
    from iris_memory.knowledge_graph.kg_quality import KGQualityReporter
    from iris_memory.knowledge_graph.kg_models import KGTriple, KGEdge, KGNode
    from iris_memory.knowledge_graph.kg_reasoning import ReasoningResult
    from iris_memory.knowledge_graph.kg_maintenance import MaintenanceReport
    from iris_memory.knowledge_graph.kg_consistency import ConsistencyReport
    from iris_memory.knowledge_graph.kg_quality import QualityReport

logger = get_logger("module.kg")


class KnowledgeGraphModule:
    """知识图谱模块

    职责：
    1. 管理 KGStorage / KGExtractor / KGReasoning 的生命周期
    2. 提供统一接口供 MemoryService 调用
    3. 协同 CaptureModule（提取三元组）和 RetrievalModule（图遍历检索）
    """

    def __init__(self) -> None:
        self._storage: Optional[KGStorage] = None
        self._extractor: Optional[KGExtractor] = None
        self._reasoning: Optional[KGReasoning] = None
        self._formatter: Optional[KGContextFormatter] = None
        self._maintenance: Optional[KGMaintenanceManager] = None
        self._consistency: Optional[KGConsistencyDetector] = None
        self._quality: Optional[KGQualityReporter] = None
        self._scheduler: Optional[KGScheduler] = None
        self._enabled: bool = True

    # ── 属性 ──

    @property
    def storage(self) -> Optional["KGStorage"]:
        return self._storage

    @property
    def extractor(self) -> Optional["KGExtractor"]:
        return self._extractor

    @property
    def reasoning(self) -> Optional["KGReasoning"]:
        return self._reasoning

    @property
    def formatter(self) -> Optional["KGContextFormatter"]:
        return self._formatter

    @property
    def maintenance(self) -> Optional["KGMaintenanceManager"]:
        return self._maintenance

    @property
    def consistency(self) -> Optional["KGConsistencyDetector"]:
        return self._consistency

    @property
    def quality(self) -> Optional["KGQualityReporter"]:
        return self._quality

    @property
    def is_initialized(self) -> bool:
        return self._storage is not None

    @property
    def enabled(self) -> bool:
        return self._enabled and self.is_initialized

    # ── 初始化 ──

    async def initialize(
        self,
        plugin_data_path: Path,
        astrbot_context: Any = None,
        provider_id: Optional[str] = None,
        kg_mode: str = "rule",
        max_depth: int = 3,
        max_nodes_per_hop: int = 10,
        max_facts: int = 8,
        enabled: bool = True,
        auto_maintenance: bool = True,
        maintenance_interval: int = 86400,
        auto_cleanup_orphans: bool = True,
        auto_cleanup_low_confidence: bool = True,
        low_confidence_threshold: float = 0.2,
        staleness_days: int = 30,
    ) -> None:
        """初始化知识图谱模块

        Args:
            plugin_data_path: 数据目录
            astrbot_context: AstrBot 上下文（用于 LLM）
            provider_id: LLM provider ID
            kg_mode: 提取模式 ("rule" / "llm" / "hybrid")
            max_depth: BFS 最大跳数
            max_nodes_per_hop: 每跳最大节点数
            max_facts: 注入 LLM 的最大事实数
            enabled: 是否启用
            auto_maintenance: 是否启用自动维护
            maintenance_interval: 维护任务执行间隔（秒）
            auto_cleanup_orphans: 是否自动清理孤立节点
            auto_cleanup_low_confidence: 是否自动清理低置信度边
            low_confidence_threshold: 低置信度阈值
            staleness_days: 过期天数
        """
        self._enabled = enabled
        if not enabled:
            logger.debug("KnowledgeGraphModule disabled by config")
            return

        from iris_memory.knowledge_graph.kg_storage import KGStorage
        from iris_memory.knowledge_graph.kg_extractor import KGExtractor
        from iris_memory.knowledge_graph.kg_reasoning import KGReasoning
        from iris_memory.knowledge_graph.kg_context import KGContextFormatter
        from iris_memory.knowledge_graph.kg_maintenance import KGMaintenanceManager
        from iris_memory.knowledge_graph.kg_consistency import KGConsistencyDetector
        from iris_memory.knowledge_graph.kg_quality import KGQualityReporter

        # 初始化存储
        db_path = plugin_data_path / "knowledge_graph.db"
        self._storage = KGStorage(db_path)
        await self._storage.initialize(db_path)

        # 初始化提取器
        self._extractor = KGExtractor(
            storage=self._storage,
            mode=kg_mode,
            astrbot_context=astrbot_context,
            provider_id=provider_id,
        )

        # 初始化推理引擎
        self._reasoning = KGReasoning(
            storage=self._storage,
            max_depth=max_depth,
            max_nodes_per_hop=max_nodes_per_hop,
        )

        # 初始化格式化器
        self._formatter = KGContextFormatter(
            max_facts=max_facts,
        )

        # 初始化维护组件
        self._maintenance = KGMaintenanceManager(self._storage)
        self._consistency = KGConsistencyDetector(self._storage)
        self._quality = KGQualityReporter(self._storage)

        # 初始化定时调度器
        if auto_maintenance:
            self._scheduler = KGScheduler(
                kg_module=self,
                interval=maintenance_interval,
                auto_cleanup_orphans=auto_cleanup_orphans,
                auto_cleanup_low_confidence=auto_cleanup_low_confidence,
                low_confidence_threshold=low_confidence_threshold,
                staleness_days=staleness_days,
            )
            await self._scheduler.start()

        logger.debug(
            f"KnowledgeGraphModule initialized: mode={kg_mode}, "
            f"max_depth={max_depth}, db={db_path}, "
            f"auto_maintenance={auto_maintenance}"
        )

    async def close(self) -> None:
        """关闭资源"""
        if self._scheduler:
            await self._scheduler.stop()
            self._scheduler = None
        if self._storage:
            await self._storage.close()
            self._storage = None

    # ── 捕获阶段接口 ──

    async def process_memory(
        self,
        memory: Any,
        persona_id: Optional[str] = None,
    ) -> List["KGTriple"]:
        """从记忆中提取三元组并存入图谱

        在 capture_and_store_memory() 之后调用。

        Args:
            memory: Memory 对象
            persona_id: 人格 ID（始终写入节点/边）

        Returns:
            提取到的三元组列表
        """
        if not self.enabled or not self._extractor:
            return []

        try:
            _raw = persona_id or getattr(memory, "persona_id", None)
            _persona = _raw if isinstance(_raw, str) else "default"
            triples = await self._extractor.extract_and_store(
                text=memory.content,
                user_id=memory.user_id,
                group_id=memory.group_id,
                memory_id=memory.id,
                sender_name=memory.sender_name,
                existing_entities=getattr(memory, "detected_entities", None),
                persona_id=_persona,
            )

            # 更新 Memory 的 graph_nodes / graph_edges（可选）
            if triples and hasattr(memory, "graph_nodes"):
                nodes_set = set(memory.graph_nodes or [])
                edges_set = set(memory.graph_edges or [])
                for triple in triples:
                    nodes_set.add(triple.subject)
                    nodes_set.add(triple.object)
                    edges_set.add(f"{triple.subject}->{triple.predicate}->{triple.object}")
                memory.graph_nodes = list(nodes_set)
                memory.graph_edges = list(edges_set)

            return triples

        except Exception as e:
            logger.warning(f"Failed to process memory for KG: {e}")
            return []

    # ── 检索阶段接口 ──

    async def graph_retrieve(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        max_results: int = 10,
        persona_id: Optional[str] = None,
    ) -> "ReasoningResult":
        """执行图遍历检索 + 多跳推理

        Args:
            query: 查询文本
            user_id: 用户 ID
            group_id: 群组 ID
            max_depth: 最大跳数
            max_results: 最大路径数
            persona_id: 人格 ID（非 None 时启用 persona 过滤）

        Returns:
            ReasoningResult
        """
        if not self.enabled or not self._reasoning:
            from iris_memory.knowledge_graph.kg_reasoning import ReasoningResult
            return ReasoningResult()

        return await self._reasoning.reason(
            query=query,
            user_id=user_id,
            group_id=group_id,
            max_depth=max_depth,
            max_results=max_results,
            persona_id=persona_id,
        )

    async def format_graph_context(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> str:
        """执行图检索并格式化为 LLM 上下文

        一站式接口：推理 + 格式化。

        Args:
            query: 查询文本
            user_id: 用户 ID
            group_id: 群组 ID
            persona_id: 人格 ID（非 None 时启用 persona 过滤）

        Returns:
            格式化后的知识关联文本（可能为空字符串）
        """
        if not self.enabled or not self._reasoning or not self._formatter:
            return ""

        try:
            result = await self._reasoning.reason(
                query=query,
                user_id=user_id,
                group_id=group_id,
                persona_id=persona_id,
            )

            return self._formatter.format_reasoning_result(result, group_id)

        except Exception as e:
            logger.warning(f"Failed to format graph context: {e}")
            return ""

    # ── 统计 / 管理 ──

    async def get_stats(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """获取统计"""
        if not self._storage:
            return {"nodes": 0, "edges": 0}
        return await self._storage.get_stats(user_id, group_id)

    async def delete_user_data(
        self,
        user_id: str,
        group_id: Optional[str] = None,
    ) -> int:
        """删除用户图谱数据"""
        if not self._storage:
            return 0
        return await self._storage.delete_user_data(user_id, group_id)

    async def delete_all(self) -> int:
        """删除所有图谱数据"""
        if not self._storage:
            return 0
        return await self._storage.delete_all()

    # ── 维护 / 一致性 / 质量 ──

    async def run_maintenance(
        self,
        confidence_threshold: float = 0.2,
        staleness_days: int = 30,
    ) -> "MaintenanceReport":
        """执行完整图谱维护清理

        Args:
            confidence_threshold: 低置信度阈值
            staleness_days: 过期天数

        Returns:
            维护报告
        """
        if not self._maintenance:
            from iris_memory.knowledge_graph.kg_maintenance import MaintenanceReport
            return MaintenanceReport()

        return await self._maintenance.run_full_cleanup(
            confidence_threshold=confidence_threshold,
            staleness_days=staleness_days,
        )

    async def check_consistency(
        self,
        max_cycle_length: int = 3,
    ) -> "ConsistencyReport":
        """执行完整一致性检查

        Args:
            max_cycle_length: 循环检测最大长度

        Returns:
            一致性报告
        """
        if not self._consistency:
            from iris_memory.knowledge_graph.kg_consistency import ConsistencyReport
            return ConsistencyReport()

        return await self._consistency.run_all_checks(max_cycle_length)

    async def generate_quality_report(
        self,
        low_confidence_threshold: float = 0.3,
    ) -> "QualityReport":
        """生成图谱质量报告

        Args:
            low_confidence_threshold: 低置信度阈值

        Returns:
            质量报告
        """
        if not self._quality:
            from iris_memory.knowledge_graph.kg_quality import QualityReport
            return QualityReport()

        return await self._quality.generate_report(low_confidence_threshold)


class KGScheduler:
    """知识图谱定时维护调度器

    按周期自动执行：
    1. 完整维护清理（孤立节点、低置信度边、悬空边）
    2. 一致性检测（用于日志记录，不自动修复）

    遵循项目中已有的 asyncio.create_task 后台循环模式
    （参考 SessionLifecycleManager、BatchProcessor）。
    """

    def __init__(
        self,
        kg_module: KnowledgeGraphModule,
        interval: int = 86400,
        auto_cleanup_orphans: bool = True,
        auto_cleanup_low_confidence: bool = True,
        low_confidence_threshold: float = 0.2,
        staleness_days: int = 30,
    ) -> None:
        self._kg = kg_module
        self._interval = interval
        self._auto_cleanup_orphans = auto_cleanup_orphans
        self._auto_cleanup_low_confidence = auto_cleanup_low_confidence
        self._low_confidence_threshold = low_confidence_threshold
        self._staleness_days = staleness_days
        self._task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self) -> None:
        """启动定时维护任务"""
        if self._is_running:
            return
        self._is_running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"KGScheduler started: interval={self._interval}s, "
            f"cleanup_orphans={self._auto_cleanup_orphans}, "
            f"cleanup_low_conf={self._auto_cleanup_low_confidence}"
        )

    async def stop(self) -> None:
        """停止定时维护任务（热更新友好）"""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"KGScheduler stop error: {e}")
            self._task = None
        logger.debug("KGScheduler stopped")

    async def _run_loop(self) -> None:
        """维护循环：等待 interval 后执行一次维护"""
        while self._is_running:
            try:
                await asyncio.sleep(self._interval)
                if not self._is_running:
                    break
                await self._execute_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"KGScheduler loop error: {e}")

    async def _execute_maintenance(self) -> None:
        """执行一次完整维护"""
        try:
            # 1. 维护清理
            report = await self._kg.run_maintenance(
                confidence_threshold=self._low_confidence_threshold,
                staleness_days=self._staleness_days,
            )
            logger.info(f"KGScheduler maintenance: {report.summary()}")

            # 2. 一致性检测（仅记录日志，不自动修复）
            consistency = await self._kg.check_consistency()
            if not consistency.is_consistent:
                logger.warning(f"KGScheduler consistency: {consistency.summary()}")
            else:
                logger.debug("KGScheduler consistency: no issues found")

        except Exception as e:
            logger.error(f"KGScheduler maintenance failed: {e}")

    async def run_once(self) -> None:
        """手动触发一次维护（供 Web API 调用）"""
        await self._execute_maintenance()

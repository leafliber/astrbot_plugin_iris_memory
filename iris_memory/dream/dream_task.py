"""
Iris Chat Memory - 梦境任务主编排器

记忆的离线深度加工，6 阶段流水线：
1. 合并重复项 — 归拢同一话题的碎片记忆
2. 时间锚定 — 将相对时间表达转换为绝对日期
3. 矛盾消解 — 检测并解决逻辑冲突
4. 模式挖掘 — 发现隐含的行为规律
5. 知识提取 — L2→L3 结构化
6. 遗忘清洗 — 淘汰低价值记忆

Features:
    - 6 阶段可独立开关
    - 阶段间数据流：前阶段输出影响后阶段
    - 统一报告输出
    - 单阶段失败不阻塞后续阶段
    - L2 entries 单次加载 + 按需重载
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l2_memory.models import MemoryEntry
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager

if TYPE_CHECKING:
    from iris_memory.core import ComponentManager

logger = get_logger("dream")


@dataclass
class DreamPhaseReport:
    phase: str
    enabled: bool
    success: bool
    duration_ms: int
    details: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DreamReport:
    started_at: str = ""
    finished_at: str = ""
    total_duration_ms: int = 0
    phases: List[DreamPhaseReport] = field(default_factory=list)

    @property
    def summary(self) -> str:
        enabled = [p for p in self.phases if p.enabled]
        succeeded = [p for p in enabled if p.success]
        failed = [p for p in enabled if not p.success]
        skipped = [p for p in self.phases if not p.enabled]
        parts = [f"{len(succeeded)} 阶段成功"]
        if failed:
            parts.append(f"{len(failed)} 阶段失败")
        if skipped:
            parts.append(f"{len(skipped)} 阶段跳过")
        return f"梦境完成：{', '.join(parts)}，耗时 {self.total_duration_ms}ms"


_PHASE_CONFIG_KEYS = {
    "consolidation": "scheduled_tasks.dream_enable_consolidation",
    "temporal_anchor": "scheduled_tasks.dream_enable_temporal_anchor",
    "contradiction": "scheduled_tasks.dream_enable_contradiction",
    "pattern_discovery": "scheduled_tasks.dream_enable_pattern_discovery",
    "knowledge_extract": "scheduled_tasks.dream_enable_knowledge_extract",
    "pruning": "scheduled_tasks.dream_enable_pruning",
}

_PHASES_THAT_MUTATE_ENTRIES = {"consolidation", "contradiction", "temporal_anchor"}


class DreamTask:
    """梦境任务 - 记忆离线深度加工

    6 阶段流水线，每个阶段可独立开关。
    阶段间有数据流：前一阶段的输出影响后一阶段的输入。
    单阶段失败不阻塞后续阶段执行。

    L2 entries 在首次需要时加载一次，传递给各阶段复用；
    在会修改条目集的阶段（合并、矛盾消解）执行后自动重载。
    """

    def __init__(self, component_manager: "ComponentManager"):
        self._component_manager = component_manager
        self._cached_entries: Optional[List[MemoryEntry]] = None
        self._cached_persona: Optional[str] = None

    async def _get_entries(
        self, l2: L2MemoryAdapter, persona_id: str
    ) -> List[MemoryEntry]:
        # 缓存按 persona 区分；persona 变化时重载
        if self._cached_entries is None or self._cached_persona != persona_id:
            self._cached_entries = await l2.get_all_entries(persona_id=persona_id)
            self._cached_persona = persona_id
        return self._cached_entries

    async def _invalidate_entries(self) -> None:
        self._cached_entries = None

    async def execute(self) -> DreamReport:
        config = get_config()

        if not config.get("scheduled_tasks.enable_dream"):
            logger.debug("梦境任务未启用，跳过")
            return DreamReport()

        started_at = datetime.now()
        report = DreamReport(started_at=started_at.isoformat())

        l2 = self._get_l2()
        l3 = self._get_l3()
        llm = self._get_llm()

        if not l2:
            logger.warning("L2 记忆库不可用，无法执行梦境")
            return report

        # 按人格隔离加工：每个 persona 独立跑一遍流水线，避免跨人格合并
        persona_ids = await l2.get_all_persona_ids() or ["default"]
        logger.info(f"🌙 梦境开始，待加工 persona：{persona_ids}")

        for persona_id in persona_ids:
            await self._run_pipeline_for_persona(persona_id, l2, l3, llm, report)
            await self._invalidate_entries()

        finished_at = datetime.now()
        report.finished_at = finished_at.isoformat()
        report.total_duration_ms = int(
            (finished_at - started_at).total_seconds() * 1000
        )

        logger.info(f"🌙 {report.summary}")
        return report

    async def _run_pipeline_for_persona(
        self,
        persona_id: str,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        llm: Optional["LLMManager"],
        report: DreamReport,
    ) -> None:
        """对单个 persona 执行完整 6 阶段流水线"""
        config = get_config()
        logger.info(f"🌙 persona [{persona_id}] 开始加工...")

        phase_order = [
            ("consolidation", self._run_consolidation),
            ("temporal_anchor", self._run_temporal_anchor),
            ("contradiction", self._run_contradiction),
            ("pattern_discovery", self._run_pattern_discovery),
            ("knowledge_extract", self._run_knowledge_extract),
            ("pruning", self._run_pruning),
        ]

        for phase_name, phase_func in phase_order:
            config_key = _PHASE_CONFIG_KEYS[phase_name]
            enabled = bool(config.get(config_key))

            needs_entries = phase_name != "knowledge_extract"
            entries = (
                await self._get_entries(l2, persona_id)
                if (enabled and needs_entries)
                else None
            )

            phase_report = await self._run_phase(
                phase_name, enabled, phase_func, l2, l3, llm, entries, persona_id
            )
            report.phases.append(phase_report)

            if enabled and phase_name in _PHASES_THAT_MUTATE_ENTRIES:
                await self._invalidate_entries()

    async def _run_phase(
        self,
        phase_name: str,
        enabled: bool,
        phase_func,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        llm: Optional["LLMManager"],
        entries: Optional[List["MemoryEntry"]] = None,
        persona_id: str = "default",
    ) -> DreamPhaseReport:
        if not enabled:
            logger.debug(f"阶段 [{phase_name}] 已禁用，跳过")
            return DreamPhaseReport(
                phase=phase_name, enabled=False, success=True, duration_ms=0
            )

        phase_start = datetime.now()
        try:
            details = await phase_func(l2, l3, llm, entries, persona_id)
            duration_ms = int((datetime.now() - phase_start).total_seconds() * 1000)
            logger.info(
                f"阶段 [{phase_name}] (persona {persona_id}) 完成，耗时 {duration_ms}ms"
            )
            return DreamPhaseReport(
                phase=phase_name,
                enabled=True,
                success=True,
                duration_ms=duration_ms,
                details=details or {},
            )
        except Exception as e:
            duration_ms = int((datetime.now() - phase_start).total_seconds() * 1000)
            logger.error(
                f"阶段 [{phase_name}] (persona {persona_id}) 失败：{e}", exc_info=True
            )
            return DreamPhaseReport(
                phase=phase_name,
                enabled=True,
                success=False,
                duration_ms=duration_ms,
                error=str(e),
            )

    async def _run_consolidation(self, l2, l3, llm, entries=None, persona_id="default"):
        from .consolidation import ConsolidationPhase

        phase = ConsolidationPhase()
        return await phase.execute(l2, l3, llm, entries=entries, persona_id=persona_id)

    async def _run_temporal_anchor(
        self, l2, l3, llm, entries=None, persona_id="default"
    ):
        from .temporal_anchor import TemporalAnchorPhase

        phase = TemporalAnchorPhase()
        return await phase.execute(l2, l3, llm, entries=entries, persona_id=persona_id)

    async def _run_contradiction(self, l2, l3, llm, entries=None, persona_id="default"):
        from .contradiction import ContradictionPhase

        phase = ContradictionPhase()
        return await phase.execute(l2, l3, llm, entries=entries, persona_id=persona_id)

    async def _run_pattern_discovery(
        self, l2, l3, llm, entries=None, persona_id="default"
    ):
        from .pattern_discovery import PatternDiscoveryPhase

        phase = PatternDiscoveryPhase()
        return await phase.execute(l2, l3, llm, entries=entries, persona_id=persona_id)

    async def _run_knowledge_extract(
        self, l2, l3, llm, entries=None, persona_id="default"
    ):
        from .knowledge_extract import KnowledgeExtractPhase

        phase = KnowledgeExtractPhase()
        return await phase.execute(l2, l3, llm, persona_id=persona_id)

    async def _run_pruning(self, l2, l3, llm, entries=None, persona_id="default"):
        from .pruning import PruningPhase

        phase = PruningPhase()
        return await phase.execute(l2, l3, llm, entries=entries, persona_id=persona_id)

    def _get_l2(self) -> Optional["L2MemoryAdapter"]:
        adapter = self._component_manager.get_component("l2_memory", L2MemoryAdapter)
        if adapter and adapter.is_available:
            return adapter
        return None

    def _get_l3(self) -> Optional["L3KGAdapter"]:
        adapter = self._component_manager.get_component("l3_kg", L3KGAdapter)
        if adapter and adapter.is_available:
            return adapter
        return None

    def _get_llm(self) -> Optional["LLMManager"]:
        manager = self._component_manager.get_component("llm_manager", LLMManager)
        if manager and manager.is_available:
            return manager
        return None

"""
Iris Chat Memory - 梦境任务主编排器

记忆的离线深度加工，收敛为 4 个 persona 阶段 + 1 个全局维护阶段：
1. 时间锚定 — 将相对时间表达转换为绝对日期（零 LLM）
2. 记忆协调 — 一次近邻扫描完成重复合并与矛盾消解
3. 知识归纳 — 增量模式发现 + L2→L3 结构化
4. L2 遗忘清洗 — 淘汰低价值记忆
5. 全局 L3 维护 — 每轮仅执行一次图谱去重与淘汰

Features:
    - 5 个执行阶段独立开关
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
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_requests: int = 0
    embedded_texts: int = 0


@dataclass
class DreamReport:
    started_at: str = ""
    finished_at: str = ""
    total_duration_ms: int = 0
    phases: List[DreamPhaseReport] = field(default_factory=list)

    @property
    def cost(self) -> dict:
        return {
            "llm_calls": sum(p.llm_calls for p in self.phases),
            "input_tokens": sum(p.input_tokens for p in self.phases),
            "output_tokens": sum(p.output_tokens for p in self.phases),
            "embedding_requests": sum(p.embedding_requests for p in self.phases),
            "embedded_texts": sum(p.embedded_texts for p in self.phases),
        }

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
        cost = self.cost
        return (
            f"梦境完成：{', '.join(parts)}，耗时 {self.total_duration_ms}ms，"
            f"LLM {cost['llm_calls']} 次，Embedding {cost['embedding_requests']} 次"
        )


_PHASE_CONFIG_KEYS = {
    "temporal_anchor": "scheduled_tasks.dream_stage_temporal_anchor_enabled",
    "reconciliation": "scheduled_tasks.dream_stage_reconciliation_enabled",
    "knowledge_induction": "scheduled_tasks.dream_stage_knowledge_induction_enabled",
    "pruning": "scheduled_tasks.dream_stage_l2_pruning_enabled",
}
_GLOBAL_L3_CONFIG_KEY = "scheduled_tasks.dream_stage_l3_maintenance_enabled"

_PHASES_THAT_MUTATE_ENTRIES = {
    "reconciliation",
    "temporal_anchor",
    "knowledge_induction",
}


class DreamTask:
    """梦境任务 - 记忆离线深度加工

    4 个 persona 阶段 + 1 个全局维护阶段。
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

        # L3 是共享适配器；全图去重/孤儿清理/淘汰每轮只执行一次，避免随
        # persona 数量线性重复。L2 清洗仍在各 persona 流水线内独立执行。
        if l3 and config.get(_GLOBAL_L3_CONFIG_KEY):
            global_report = await self._run_phase(
                "pruning_l3_global",
                True,
                self._run_global_pruning,
                l2,
                l3,
                llm,
                [],
                "*",
            )
            report.phases.append(global_report)

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
        """对单个 persona 执行优化后的流水线"""
        config = get_config()
        logger.info(f"🌙 persona [{persona_id}] 开始加工...")

        phase_order = [
            ("temporal_anchor", self._run_temporal_anchor),
            ("reconciliation", self._run_reconciliation),
            ("knowledge_induction", self._run_knowledge_induction),
            ("pruning", self._run_pruning),
        ]

        for phase_name, phase_func in phase_order:
            config_key = _PHASE_CONFIG_KEYS[phase_name]
            enabled = bool(config.get(config_key))

            needs_entries = True
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
        before_cost = await self._cost_snapshot(l2, llm)
        try:
            details = await phase_func(l2, l3, llm, entries, persona_id)
            after_cost = await self._cost_snapshot(l2, llm)
            cost = self._cost_delta(before_cost, after_cost)
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
                **cost,
            )
        except Exception as e:
            after_cost = await self._cost_snapshot(l2, llm)
            cost = self._cost_delta(before_cost, after_cost)
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
                **cost,
            )

    @staticmethod
    async def _cost_snapshot(l2, llm) -> dict:
        llm_stats = (
            await llm.get_token_stats("global")
            if llm and hasattr(llm, "get_token_stats")
            else {}
        )
        embedding_stats = (
            l2.get_embedding_stats() if hasattr(l2, "get_embedding_stats") else {}
        )
        return {
            "llm_calls": int(llm_stats.get("total_calls", 0)),
            "input_tokens": int(llm_stats.get("total_input_tokens", 0)),
            "output_tokens": int(llm_stats.get("total_output_tokens", 0)),
            "embedding_requests": int(embedding_stats.get("requests", 0)),
            "embedded_texts": int(embedding_stats.get("texts", 0)),
        }

    @staticmethod
    def _cost_delta(before: dict, after: dict) -> dict:
        return {
            key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
            for key in before
        }

    async def _run_reconciliation(
        self, l2, l3, llm, entries=None, persona_id="default"
    ):
        from .reconciliation import ReconciliationPhase

        return await ReconciliationPhase().execute(
            l2, l3, llm, entries=entries, persona_id=persona_id
        )

    async def _run_temporal_anchor(
        self, l2, l3, llm, entries=None, persona_id="default"
    ):
        from .temporal_anchor import TemporalAnchorPhase

        phase = TemporalAnchorPhase()
        return await phase.execute(l2, l3, llm, entries=entries, persona_id=persona_id)

    async def _run_knowledge_induction(
        self, l2, l3, llm, entries=None, persona_id="default"
    ):
        from .knowledge_induction import KnowledgeInductionPhase

        return await KnowledgeInductionPhase().execute(
            l2,
            l3,
            llm,
            entries=entries,
            persona_id=persona_id,
            component_manager=self._component_manager,
        )

    async def _run_pruning(self, l2, l3, llm, entries=None, persona_id="default"):
        from .pruning import PruningPhase

        phase = PruningPhase()
        return await phase.execute(
            l2,
            None,
            llm,
            entries=entries,
            persona_id=persona_id,
            process_l2=True,
            process_l3=False,
        )

    async def _run_global_pruning(
        self, l2, l3, llm, entries=None, persona_id="*"
    ):
        from .pruning import PruningPhase

        return await PruningPhase().execute(
            l2,
            l3,
            llm,
            entries=[],
            persona_id=persona_id,
            process_l2=False,
            process_l3=True,
        )

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

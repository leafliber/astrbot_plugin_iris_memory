"""记忆协调阶段：一次近邻扫描同时服务重复合并与矛盾消解。"""

import random
from typing import Dict, Optional, cast

from iris_memory.config import get_config
from iris_memory.core import get_logger
from iris_memory.l2_memory.models import MemoryEntry

from .consolidation import ConsolidationPhase, UnionFind
from .contradiction import ContradictionPhase

logger = get_logger("dream.reconciliation")


class ReconciliationPhase:
    """共享候选图的重复合并 + 矛盾消解。"""

    async def execute(
        self,
        l2,
        l3,
        llm,
        entries: Optional[list[MemoryEntry]] = None,
        persona_id: str = "default",
    ) -> dict:
        config = get_config()
        if not llm:
            return {}
        if entries is None:
            entries = await l2.get_all_entries(persona_id=persona_id)
        if len(entries) < 2:
            return {}

        consolidation = ConsolidationPhase()
        consolidation._similarity_threshold = cast(
            float, config.get("dream_consolidation_similarity_threshold")
        )
        consolidation._batch_size = max(
            1, cast(int, config.get("dream_consolidation_batch_size"))
        )
        consolidation._scan_budget = max(
            1, cast(int, config.get("dream_consolidation_scan_budget"))
        )
        consolidation._query_batch_size = max(
            1, cast(int, config.get("dream_consolidation_query_batch_size"))
        )
        consolidation._max_group_size = max(
            2, cast(int, config.get("dream_consolidation_max_group_size"))
        )
        consolidation._query_top_k = max(
            1, cast(int, config.get("dream_consolidation_query_top_k"))
        )

        contradiction = ContradictionPhase()
        contradiction._similarity_floor = cast(
            float, config.get("dream_contradiction_similarity_floor")
        )
        contradiction._similarity_ceiling = cast(
            float, config.get("dream_contradiction_similarity_ceiling")
        )
        contradiction._max_groups = max(
            1, cast(int, config.get("dream_contradiction_max_groups"))
        )
        contradiction._scan_budget = max(
            1, cast(int, config.get("dream_contradiction_scan_budget"))
        )
        contradiction._query_batch_size = max(
            1, cast(int, config.get("dream_contradiction_query_batch_size"))
        )
        contradiction._llm_batch_size = max(
            1, cast(int, config.get("dream_contradiction_llm_batch_size", 5))
        )

        scan_budget = max(
            consolidation._scan_budget,
            contradiction._scan_budget,
        )
        scan_entries = (
            random.sample(entries, scan_budget) if len(entries) > scan_budget else entries
        )
        entry_index: Dict[str, MemoryEntry] = {entry.id: entry for entry in entries}
        enable_group_isolation = bool(
            config.get("isolation_config.enable_group_memory_isolation")
        )
        grouped: dict[Optional[str], list[MemoryEntry]] = {}
        if enable_group_isolation:
            for entry in scan_entries:
                grouped.setdefault(entry.group_id, []).append(entry)
        else:
            grouped[None] = scan_entries

        high_uf = UnionFind()
        mid_edges: list[tuple[float, tuple[str, str]]] = []
        seen_pairs: set[tuple[str, str]] = set()
        scanned = 0
        query_batch_size = max(
            consolidation._query_batch_size,
            contradiction._query_batch_size,
        )
        top_k = max(consolidation._query_top_k, contradiction._query_top_k)

        for group_id, group_entries in grouped.items():
            for offset in range(0, len(group_entries), query_batch_size):
                batch = group_entries[offset : offset + query_batch_size]
                results_batch = await l2.batch_retrieve_by_ids(
                    [entry.id for entry in batch],
                    group_id=group_id,
                    top_k=top_k,
                    persona_id=persona_id,
                )
                scanned += len(batch)
                for query_entry, results in zip(batch, results_batch):
                    for result in results:
                        hit = result.entry
                        if hit.id == query_entry.id or hit.id not in entry_index:
                            continue
                        if enable_group_isolation and hit.group_id != query_entry.group_id:
                            continue
                        pair = tuple(sorted((query_entry.id, hit.id)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        if (
                            result.score >= consolidation._similarity_threshold
                        ):
                            high_uf.union(*pair)
                        elif (
                            contradiction._similarity_floor <= result.score
                            < contradiction._similarity_ceiling
                        ):
                            mid_edges.append((result.score, pair))

        high_groups = list(high_uf.groups().values())
        selected_high = high_groups[: consolidation._batch_size]
        merged = 0
        deleted = 0
        consumed_ids: set[str] = set()
        for group_ids in selected_high:
            group_ids = group_ids[: consolidation._max_group_size]
            group = [entry_index[item] for item in group_ids if item in entry_index]
            if len(group) < 2:
                continue
            merged_delta, deleted_delta = await consolidation._merge_group(
                group, l2, llm, persona_id
            )
            if merged_delta:
                consumed_ids.update(item.id for item in group)
            merged += merged_delta
            deleted += deleted_delta

        contradiction_groups = []
        used_contradiction_ids: set[str] = set()
        for _score, pair in sorted(mid_edges, reverse=True):
            left_id, right_id = pair
            if (
                left_id in consumed_ids
                or right_id in consumed_ids
                or left_id in used_contradiction_ids
                or right_id in used_contradiction_ids
            ):
                continue
            left = entry_index.get(left_id)
            right = entry_index.get(right_id)
            if left is None or right is None:
                continue
            contradiction_groups.append([left, right])
            used_contradiction_ids.update(pair)
        contradiction_groups = contradiction_groups[: contradiction._max_groups]

        contradictions_found = 0
        resolved = 0
        for offset in range(0, len(contradiction_groups), contradiction._llm_batch_size):
            batch = contradiction_groups[
                offset : offset + contradiction._llm_batch_size
            ]
            results = await contradiction._check_and_resolve_batch(batch, l2, llm)
            for result in results:
                if result is not None:
                    contradictions_found += 1
                    if result:
                        resolved += 1

        logger.info(
            f"协调完成：扫描 {scanned}，合并 {merged} 组，"
            f"矛盾 {contradictions_found}/{len(contradiction_groups)}，解决 {resolved}"
        )
        return {
            "scanned": scanned,
            "consolidation": {
                "candidate_groups": len(high_groups),
                "merged_groups": merged,
                "deleted_entries": deleted,
            },
            "contradiction": {
                "groups_checked": len(contradiction_groups),
                "contradictions_found": contradictions_found,
                "resolved": resolved,
            },
        }

"""
Iris Chat Memory - 梦境阶段1：合并重复项

归拢同一话题的碎片记忆，生成完整的话题摘要。

Features:
    - 基于已存向量的批量检索（零 embedding 调用）
    - 并查集连通分量（传递性合并）
    - 采样扫描预算（大数据量优化）
    - LLM 智能合并（话题级归拢）
"""

import random
from datetime import datetime
from typing import Dict, List, Optional, cast

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager
from iris_memory.l2_memory.models import MemoryEntry

logger = get_logger("dream.consolidation")


class UnionFind:
    """并查集数据结构

    用于构建相似记忆的连通分量，支持传递性合并：
    若 A~B 且 B~C，则 A、B、C 属于同一合并组。
    """

    __slots__ = ("_parent", "_rank")

    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            next_x = self._parent[x]
            self._parent[x] = root
            x = next_x
        return root

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def groups(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for x in self._parent:
            root = self.find(x)
            if root not in result:
                result[root] = []
            result[root].append(x)
        return {k: v for k, v in result.items() if len(v) > 1}


class ConsolidationPhase:
    """合并重复项阶段

    归拢同一话题的碎片记忆，生成完整的话题摘要。
    """

    def __init__(self):
        self._similarity_threshold = 0.85
        self._batch_size = 10
        self._scan_budget = 500
        self._query_batch_size = 50
        self._max_group_size = 5
        self._query_top_k = 5

    async def execute(
        self,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        llm: Optional["LLMManager"],
        entries: Optional[List["MemoryEntry"]] = None,
        persona_id: str = "default",
    ) -> dict:
        config = get_config()
        self._similarity_threshold = cast(
            float, config.get("dream_consolidation_similarity_threshold")
        )
        self._batch_size = cast(int, config.get("dream_consolidation_batch_size"))
        self._scan_budget = cast(int, config.get("dream_consolidation_scan_budget"))
        self._query_batch_size = cast(
            int, config.get("dream_consolidation_query_batch_size")
        )
        self._max_group_size = cast(
            int, config.get("dream_consolidation_max_group_size")
        )
        self._query_top_k = cast(int, config.get("dream_consolidation_query_top_k"))

        if not llm:
            logger.warning("LLMManager 不可用，跳过合并重复项")
            return {"merged_groups": 0, "deleted_entries": 0}

        try:
            if entries is None:
                entries = await l2.get_all_entries(persona_id=persona_id)

            if len(entries) < 2:
                logger.debug("记忆数量不足，无需合并")
                return {"merged_groups": 0, "deleted_entries": 0}

            entry_index: Dict[str, "MemoryEntry"] = {e.id: e for e in entries}

            logger.info(f"开始分析 {len(entries)} 条记忆的相似度...")

            merge_groups = await self._find_merge_groups(
                entries, entry_index, l2, persona_id
            )

            if not merge_groups:
                logger.debug("未发现相似记忆，无需合并")
                return {"merged_groups": 0, "deleted_entries": 0}

            logger.info(f"发现 {len(merge_groups)} 组相似记忆")

            merged_count = 0
            deleted_count = 0

            for group_ids in merge_groups[: self._batch_size]:
                group_entries = [
                    entry_index[eid] for eid in group_ids if eid in entry_index
                ]
                if len(group_entries) < 2:
                    continue

                try:
                    m, d = await self._merge_group(group_entries, l2, llm, persona_id)
                    merged_count += m
                    deleted_count += d
                except Exception as e:
                    logger.error(f"合并记忆组失败：{e}", exc_info=True)

            logger.info(
                f"合并重复项完成，共合并 {merged_count} 组，删除 {deleted_count} 条旧记忆"
            )
            return {"merged_groups": merged_count, "deleted_entries": deleted_count}

        except Exception as e:
            logger.error(f"合并重复项失败：{e}", exc_info=True)
            return {"merged_groups": 0, "deleted_entries": 0, "error": str(e)}

    async def _find_merge_groups(
        self,
        entries: List["MemoryEntry"],
        entry_index: Dict[str, "MemoryEntry"],
        adapter: "L2MemoryAdapter",
        persona_id: str = "default",
    ) -> List[List[str]]:
        config = get_config()
        enable_group_isolation = bool(
            config.get("isolation_config.enable_group_memory_isolation")
        )

        if len(entries) > self._scan_budget:
            scan_entries = random.sample(entries, self._scan_budget)
            logger.info(
                f"记忆数量 {len(entries)} 超过扫描预算 {self._scan_budget}，"
                f"随机采样 {self._scan_budget} 条"
            )
        else:
            scan_entries = entries

        if enable_group_isolation:
            groups_by_gid: Dict[Optional[str], List["MemoryEntry"]] = {}
            for e in scan_entries:
                gid = e.group_id
                groups_by_gid.setdefault(gid, []).append(e)
        else:
            groups_by_gid = {None: scan_entries}

        uf = UnionFind()
        total_queries = 0

        for gid, group_entries in groups_by_gid.items():
            for i in range(0, len(group_entries), self._query_batch_size):
                batch = group_entries[i : i + self._query_batch_size]
                query_ids = [e.id for e in batch]

                try:
                    # 查询对象本身已在 L2 库中，直接复用索引中已存的向量，
                    # 无需对文本重新计算 embedding
                    results_batch = await adapter.batch_retrieve_by_ids(
                        memory_ids=query_ids,
                        group_id=gid,
                        top_k=self._query_top_k,
                        persona_id=persona_id,
                    )

                    for query_id, results in zip(query_ids, results_batch):
                        query_entry = entry_index.get(query_id)
                        if not query_entry:
                            continue
                        query_gid = query_entry.group_id

                        for r in results:
                            if r.entry.id == query_id:
                                continue
                            if r.score < self._similarity_threshold:
                                continue
                            if enable_group_isolation:
                                if r.entry.group_id != query_gid:
                                    continue
                            uf.union(query_id, r.entry.id)

                    total_queries += len(batch)
                    if (
                        total_queries % max(100, self._query_batch_size)
                        < self._query_batch_size
                    ):
                        logger.info(
                            f"已扫描 {total_queries}/{len(scan_entries)} 条记忆..."
                        )

                except Exception as e:
                    logger.warning(f"批量检索失败：{e}")

        raw_groups = list(uf.groups().values())

        result: List[List[str]] = []
        for group in raw_groups:
            if len(group) > self._max_group_size:
                result.append(group[: self._max_group_size])
            else:
                result.append(group)

        logger.info(f"扫描 {total_queries} 条记忆，发现 {len(result)} 组相似记忆")
        return result

    async def _merge_group(
        self,
        entries: List["MemoryEntry"],
        l2_adapter: "L2MemoryAdapter",
        llm_manager: "LLMManager",
        persona_id: str = "default",
    ) -> tuple:
        ids_to_delete = [e.id for e in entries]

        sorted_entries = sorted(
            entries,
            key=lambda e: (e.metadata.get("confidence", 0.5), len(e.content)),
            reverse=True,
        )

        current_content = sorted_entries[0].content
        best_metadata = sorted_entries[0].metadata

        for i in range(1, len(sorted_entries)):
            merged = await self._merge_memories(
                current_content, sorted_entries[i].content, llm_manager
            )
            if merged:
                current_content = merged
            else:
                if len(sorted_entries[i].content) > len(current_content):
                    current_content = sorted_entries[i].content

        group_id = best_metadata.get("group_id")
        max_confidence = max(e.metadata.get("confidence", 0.5) for e in entries)
        merged_from = ",".join(e.id for e in entries)

        # 先写入合并后的新记忆，确认成功后再删除旧记忆，
        # 避免 add_memory 失败时原始记忆已被删除导致数据丢失。
        new_id = await l2_adapter.add_memory(
            current_content,
            metadata={
                "group_id": group_id,
                "confidence": max_confidence,
                "timestamp": datetime.now().isoformat(),
                "merged_from": merged_from,
            },
            skip_dedup=True,
            persona_id=persona_id,
        )

        if new_id:
            await l2_adapter.delete_entries(ids_to_delete)
            deleted_count = len(ids_to_delete)
            logger.info(f"已合并 {len(entries)} 条记忆 -> {new_id}")
            return 1, deleted_count
        else:
            logger.warning("合并记忆存储失败，原始记忆未删除，保留原样")
            return 0, 0

    async def _merge_memories(
        self, content1: str, content2: str, llm_manager: "LLMManager"
    ) -> Optional[str]:
        try:
            prompt = f"""将以下两条关于同一话题的记忆合并为一条更完整的摘要。

记忆1：{content1}
记忆2：{content2}

要求：合并重复信息，保留独特细节，时间冲突保留更近期的。仅输出合并后的内容。

合并后："""

            merged = await llm_manager.generate_direct(
                prompt=prompt, module="dream_consolidation"
            )

            if not merged or not merged.strip():
                logger.warning(
                    f"LLM 合并记忆返回空结果，"
                    f"content1={content1[:50]}..., content2={content2[:50]}..."
                )
                return None

            return merged.strip()

        except Exception as e:
            logger.error(f"LLM 合并记忆失败：{e}")
            return None

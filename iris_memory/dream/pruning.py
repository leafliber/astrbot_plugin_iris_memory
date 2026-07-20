"""
Iris Chat Memory - 梦境阶段6：遗忘清洗

淘汰低价值记忆，放在最后执行以确保前5个阶段已修复/提升有价值记忆的评分。

Features:
    - L2 记忆库遗忘清洗
    - L3 知识图谱节点淘汰
    - L3 重复节点合并
    - LLM 最终兜底确认（可选）
    - 低置信度数据标记
    - 批量处理优化
"""

from datetime import datetime
from typing import List, Optional, cast

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager
from iris_memory.utils.forgetting import (
    calculate_forgetting_score,
    should_evict,
    calculate_kg_forgetting_score,
    should_evict_kg_node,
)

logger = get_logger("dream.pruning")

_L3_FULL_SCAN_LIMIT = 999999


class PruningPhase:
    """遗忘清洗阶段

    淘汰低价值记忆。放在梦境最后执行，
    确保前5个阶段已修复/提升有价值记忆的评分。
    """

    def __init__(self):
        self._batch_size = 100

    async def execute(
        self,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        llm: Optional["LLMManager"],
        entries: Optional[list] = None,
        persona_id: str = "default",
    ) -> dict:
        config = get_config()
        self._batch_size = cast(int, config.get("eviction_batch_size"))

        if entries is None:
            if l2.is_available:
                entries = await l2.get_all_entries(persona_id=persona_id)
            else:
                entries = []

        result = {
            "l2_evicted": 0,
            "l3_merged": 0,
            "l3_evicted": 0,
            "l2_low_confidence_marked": 0,
            "l3_low_confidence_marked": 0,
            "l3_orphaned_removed": 0,
        }

        l2_marked = await self._mark_low_confidence_l2(l2, entries)
        result["l2_low_confidence_marked"] = l2_marked
        l2_evicted = await self._evict_l2_memories(l2, llm, entries)
        result["l2_evicted"] = l2_evicted

        if l3 and l3.is_available:
            merged, deleted = await self._merge_l3_duplicates(l3)
            result["l3_merged"] = merged
            l3_marked = await self._mark_low_confidence_l3(l3)
            result["l3_low_confidence_marked"] = l3_marked
            # 无主节点清理：在遗忘淘汰之前，先定向清除无 Person 关联的
            # 主体绑定类型节点（Preference/Trait/Belief/Goal/Skill）
            orphaned_removed = await self._cleanup_orphaned_subject_nodes(l3)
            result["l3_orphaned_removed"] = orphaned_removed
            l3_evicted = await self._evict_l3_nodes(l3, llm)
            result["l3_evicted"] = l3_evicted

        logger.info(
            f"遗忘清洗完成：L2 淘汰 {result['l2_evicted']}，"
            f"L3 合并 {result['l3_merged']}，L3 淘汰 {result['l3_evicted']}，"
            f"L3 无主清理 {result['l3_orphaned_removed']}"
        )
        return result

    async def _mark_low_confidence_l2(
        self, l2: "L2MemoryAdapter", entries: list
    ) -> int:
        try:
            if not entries:
                return 0

            config = get_config()
            confidence_threshold = cast(float, config.get("node_confidence_threshold"))

            marked_count = 0
            for entry in entries:
                if entry.confidence < confidence_threshold:
                    if not entry.metadata.get("low_confidence"):
                        entry.metadata["low_confidence"] = True
                        await l2.update_metadata(entry.id, entry.metadata)
                        marked_count += 1

            if marked_count > 0:
                logger.info(f"L2 低置信度标记完成：{marked_count} 条记忆被标记")

            return marked_count

        except Exception as e:
            logger.error(f"L2 低置信度标记失败：{e}", exc_info=True)
            return 0

    async def _evict_l2_memories(
        self, l2: "L2MemoryAdapter", llm: Optional["LLMManager"], entries: list
    ) -> int:
        try:
            if not entries:
                logger.debug("L2 记忆库为空，无需清洗")
                return 0

            logger.info(f"开始评估 {len(entries)} 条 L2 记忆...")

            config = get_config()
            retention_days = cast(int, config.get("l2_retention_days", 30))

            to_evict_with_score = []

            for entry in entries:
                if should_evict(entry, retention_days=retention_days):
                    score = calculate_forgetting_score(entry)
                    to_evict_with_score.append((entry.id, entry.content, score))

            if not to_evict_with_score:
                logger.debug("L2 无需淘汰的记忆")
                return 0

            confirmed_ids = await self._llm_confirm_eviction(
                to_evict_with_score, llm, source="l2"
            )

            evicted_count = 0
            batch = []
            for entry_id in confirmed_ids:
                batch.append(entry_id)
                if len(batch) >= self._batch_size:
                    await l2.evict_memories(batch)
                    evicted_count += len(batch)
                    batch = []

            if batch:
                await l2.evict_memories(batch)
                evicted_count += len(batch)

            logger.info(f"L2 遗忘清洗完成，共淘汰 {evicted_count} 条记忆")
            return evicted_count

        except Exception as e:
            logger.error(f"L2 遗忘清洗失败：{e}", exc_info=True)
            return 0

    async def _merge_l3_duplicates(self, l3: "L3KGAdapter") -> tuple:
        try:
            merged, deleted = await l3.merge_duplicate_nodes()
            if merged > 0:
                logger.info(
                    f"L3 去重合并完成：合并 {merged} 组，删除 {deleted} 个重复节点"
                )
            return merged, deleted
        except Exception as e:
            logger.error(f"L3 去重合并失败：{e}", exc_info=True)
            return 0, 0

    async def _cleanup_orphaned_subject_nodes(self, l3: "L3KGAdapter") -> int:
        """清理无 Person 关联的主体绑定类型节点（无主节点）

        Preference/Trait/Belief/Goal/Skill 类型的节点如果没有连接到
        Person 节点的边，则无法确定主体是谁（如"有特定角色偏好"但
        不知道是谁的偏好），这类节点应当被删除。

        在遗忘淘汰之前执行，因为这些节点的遗忘评分可能不足以触发淘汰
        （新建节点置信度尚可、时间戳较新），但它们对用户没有任何价值。
        """
        try:
            orphaned = await l3.find_orphaned_subject_nodes()
            if not orphaned:
                return 0

            orphaned_ids = [n["id"] for n in orphaned]
            removed = await l3.evict_nodes(orphaned_ids)

            logger.info(
                f"无主节点清理：删除 {removed} 个无 Person 关联的节点："
                f"{[n['name'] for n in orphaned]}"
            )
            return removed
        except Exception as e:
            logger.error(f"无主节点清理失败：{e}", exc_info=True)
            return 0

    async def _mark_low_confidence_l3(self, l3: "L3KGAdapter") -> int:
        try:
            nodes = await l3.get_all_nodes(limit=_L3_FULL_SCAN_LIMIT)
            if not nodes:
                return 0

            config = get_config()
            confidence_threshold = cast(float, config.get("node_confidence_threshold"))

            marked_count = 0
            for node_dict in nodes:
                confidence = node_dict.get("confidence", 1.0)
                properties = node_dict.get("properties", {})

                if confidence < confidence_threshold:
                    if not properties.get("low_confidence"):
                        try:
                            node_id = node_dict["id"]
                            properties["low_confidence"] = True
                            success = await l3.update_node_properties(
                                node_id, properties
                            )
                            if success:
                                marked_count += 1
                        except Exception as e:
                            logger.debug(f"标记节点 {node_dict.get('id')} 失败：{e}")

            if marked_count > 0:
                logger.info(f"L3 低置信度标记完成：{marked_count} 个节点被标记")

            return marked_count

        except Exception as e:
            logger.error(f"L3 低置信度标记失败：{e}", exc_info=True)
            return 0

    async def _evict_l3_nodes(
        self, l3: "L3KGAdapter", llm: Optional["LLMManager"]
    ) -> int:
        try:
            nodes = await l3.get_all_nodes(limit=_L3_FULL_SCAN_LIMIT)

            if not nodes:
                logger.debug("L3 知识图谱为空，无需淘汰")
                return 0

            logger.info(f"开始评估 {len(nodes)} 个 L3 节点...")

            connection_counts = await l3.get_node_connection_counts()

            config = get_config()
            threshold_kg = cast(float, config.get("forgetting_threshold_kg", 0.2))
            retention_days = cast(int, config.get("kg_retention_days", 30))

            to_evict_with_score = []

            for node_dict in nodes:
                node_id = node_dict["id"]
                confidence = node_dict.get("confidence", 1.0)
                last_access_time = node_dict.get("last_access_time")
                access_count = node_dict.get("access_count", 0)
                properties = node_dict.get("properties", {})
                connected_count = connection_counts.get(node_id, 0)

                source_memory_ids_str = properties.get("source_memory_ids", "")
                source_memory_count = len(
                    [x for x in source_memory_ids_str.split(",") if x.strip()]
                )

                last_access_str = None
                if last_access_time:
                    if isinstance(last_access_time, datetime):
                        last_access_str = last_access_time.isoformat()
                    else:
                        last_access_str = str(last_access_time)

                if should_evict_kg_node(
                    last_access_time=last_access_str,
                    access_count=access_count,
                    confidence=confidence,
                    connected_count=connected_count,
                    source_memory_count=source_memory_count,
                    threshold=threshold_kg,
                    retention_days=retention_days,
                ):
                    score = calculate_kg_forgetting_score(
                        last_access_time=last_access_str,
                        access_count=access_count,
                        confidence=confidence,
                        connected_count=connected_count,
                        source_memory_count=source_memory_count,
                    )
                    to_evict_with_score.append(
                        (node_id, node_dict.get("content", ""), score)
                    )

            if not to_evict_with_score:
                logger.debug("L3 无需淘汰的节点")
                return 0

            confirmed_ids = await self._llm_confirm_eviction(
                to_evict_with_score, llm, source="l3"
            )

            evicted_count = 0
            batch = []
            for node_id in confirmed_ids:
                batch.append(node_id)
                if len(batch) >= self._batch_size:
                    await l3.evict_nodes(batch)
                    evicted_count += len(batch)
                    batch = []

            if batch:
                await l3.evict_nodes(batch)
                evicted_count += len(batch)

            logger.info(f"L3 图谱淘汰完成，共淘汰 {evicted_count} 个节点")
            return evicted_count

        except Exception as e:
            logger.error(f"L3 图谱淘汰失败：{e}", exc_info=True)
            return 0

    async def _llm_confirm_eviction(
        self,
        entries: List[tuple],
        llm: Optional["LLMManager"],
        source: str = "l2",
    ) -> List[str]:
        config = get_config()
        if not config.get("forgetting_llm_confirm_enable"):
            return [e[0] for e in entries]

        if not llm:
            logger.warning("LLM Manager 不可用，跳过兜底确认，默认保留")
            return []

        confirm_threshold = cast(float, config.get("forgetting_llm_confirm_threshold"))
        provider = cast(Optional[str], config.get("forgetting_llm_confirm_provider"))

        confirmed = []
        for entry_id, content, score in entries:
            if score >= confirm_threshold:
                confirmed.append(entry_id)
                continue

            try:
                prompt = (
                    "以下是一条记忆内容，系统评估其重要性极低，建议遗忘。\n"
                    "请判断该记忆是否确实没有保留价值。\n"
                    '回复 "FORGET" 表示确认遗忘，回复 "KEEP" 表示应保留。\n\n'
                    f"记忆内容：{content[:500]}\n\n"
                    "请只回复 FORGET 或 KEEP："
                )

                response = await llm.generate_direct(
                    prompt=prompt,
                    module="dream_pruning_confirm",
                    provider_id=provider,
                )

                decision = response.strip().upper() if response else "KEEP"

                if "KEEP" in decision:
                    logger.info(f"LLM 兜底确认保留记忆：{entry_id}（评分 {score:.3f}）")
                else:
                    confirmed.append(entry_id)

            except Exception as e:
                logger.warning(f"LLM 兜底确认失败：{e}，默认保留 {entry_id}")

        return confirmed

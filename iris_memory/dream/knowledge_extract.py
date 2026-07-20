"""
Iris Chat Memory - 梦境阶段5：知识提取

从 L2 未处理记忆中提取实体和关系，写入 L3 知识图谱。

Features:
    - 按群聊/用户分组批量聚合提取
    - 空提取结果不标记为已处理
    - 批量处理优化
"""

from collections import defaultdict
from typing import List, Optional, cast

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager

logger = get_logger("dream.knowledge_extract")


class KnowledgeExtractPhase:
    """知识提取阶段

    从 L2 未处理记忆中提取实体和关系，写入 L3 知识图谱。
    """

    async def execute(
        self,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        llm: Optional["LLMManager"],
        persona_id: str = "default",
    ) -> dict:
        config = get_config()

        if not l3 or not l3.is_available:
            logger.debug("L3 知识图谱不可用，跳过知识提取")
            return {
                "memories_processed": 0,
                "nodes_extracted": 0,
                "edges_extracted": 0,
            }

        if not llm:
            logger.warning("LLMManager 不可用，跳过知识提取")
            return {
                "memories_processed": 0,
                "nodes_extracted": 0,
                "edges_extracted": 0,
            }

        min_unprocessed = cast(
            int, config.get("dream_knowledge_extract_min_unprocessed")
        )
        unprocessed_count = await l2.get_unprocessed_count(persona_id=persona_id)

        if unprocessed_count < min_unprocessed:
            logger.debug(
                f"未处理记忆数量 {unprocessed_count} < {min_unprocessed}，跳过提取"
            )
            return {
                "memories_processed": 0,
                "nodes_extracted": 0,
                "edges_extracted": 0,
            }

        logger.info(f"开始知识提取，未处理记忆数：{unprocessed_count}")

        batch_size = cast(int, config.get("dream_knowledge_extract_batch_size"))

        unprocessed_memories = await l2.get_unprocessed_memories(
            limit=batch_size, persona_id=persona_id
        )

        if not unprocessed_memories:
            logger.debug("没有未处理的记忆")
            return {
                "memories_processed": 0,
                "nodes_extracted": 0,
                "edges_extracted": 0,
            }

        groups = self._group_memories(unprocessed_memories)

        logger.info(
            f"按群聊分组：{len(groups)} 个组，共 {len(unprocessed_memories)} 条记忆"
        )

        from iris_memory.l3_kg import EntityExtractor

        extractor = EntityExtractor(llm)

        all_processed_ids: List[str] = []
        total_nodes = 0
        total_edges = 0

        for group_key, memories in groups.items():
            try:
                context = {"group_id": memories[0].group_id}

                result = await extractor.extract_from_memories(memories, context)

                if result.nodes or result.edges:
                    node_count = 0
                    for node in result.nodes:
                        success = await l3.add_node(node)
                        if success:
                            node_count += 1

                    edge_count = 0
                    for edge in result.edges:
                        success = await l3.add_edge(edge)
                        if success:
                            edge_count += 1

                    total_nodes += node_count
                    total_edges += edge_count

                    logger.info(
                        f"群组 [{group_key}] 提取完成："
                        f"{node_count}/{len(result.nodes)} 个节点，"
                        f"{edge_count}/{len(result.edges)} 条边"
                    )

                    # 仅当至少一条写入成功时才标记为已处理，
                    # 否则全失败的提取应可重试，不应永久跳过。
                    if node_count > 0 or edge_count > 0:
                        for mem in memories:
                            all_processed_ids.append(mem.id)
                    else:
                        logger.warning(
                            f"群组 [{group_key}] 提取结果非空但全部写入失败，"
                            f"不标记为已处理以便重试"
                        )
                else:
                    logger.debug(f"群组 [{group_key}] 提取结果为空，不标记为已处理")

            except Exception as e:
                logger.error(f"处理群组 [{group_key}] 失败：{e}", exc_info=True)

        if all_processed_ids:
            await l2.mark_memories_processed(all_processed_ids)

        logger.info(
            f"知识提取完成：处理 {len(all_processed_ids)} 条记忆，"
            f"提取 {total_nodes} 个节点，{total_edges} 条边"
        )
        return {
            "memories_processed": len(all_processed_ids),
            "nodes_extracted": total_nodes,
            "edges_extracted": total_edges,
        }

    def _group_memories(self, memories: list) -> dict[str, list]:
        groups: dict[str, list] = defaultdict(list)

        for mem in memories:
            group_key = mem.group_id or "_no_group"
            groups[group_key].append(mem)

        return dict(groups)

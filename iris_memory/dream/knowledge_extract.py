"""
Iris Chat Memory - 梦境阶段5：知识提取

从 L2 未处理记忆中提取实体和关系，写入 L3 知识图谱。

Features:
    - 按群聊/用户分组批量聚合提取
    - 相同内容连续两次空提取后终态处理，避免永久重试
    - 批量处理优化
"""

from collections import defaultdict
import hashlib
from typing import List, Optional, cast

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager
from iris_memory.llm_modules import DREAM_KNOWLEDGE_INDUCTION

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
        component_manager=None,
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

        extractor = EntityExtractor(llm, module=DREAM_KNOWLEDGE_INDUCTION)

        all_processed_ids: List[str] = []
        total_nodes = 0
        total_edges = 0
        empty_finalized = 0

        for group_key, memories in groups.items():
            try:
                context = {
                    "group_id": memories[0].group_id,
                    "persona_id": persona_id,
                }

                user_aliases = await self._build_user_aliases(
                    memories, persona_id, component_manager
                )
                if user_aliases:
                    context["user_aliases"] = user_aliases

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
                    finalized = await self._record_empty_result(memories, l2)
                    empty_finalized += finalized
                    logger.debug(
                        f"群组 [{group_key}] 提取结果为空，"
                        f"本轮终态标记 {finalized} 条"
                    )

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
            "empty_finalized": empty_finalized,
        }

    def _group_memories(self, memories: list) -> dict[str, list]:
        groups: dict[str, list] = defaultdict(list)

        for mem in memories:
            group_key = mem.group_id or "_no_group"
            groups[group_key].append(mem)

        return dict(groups)

    async def _record_empty_result(
        self, memories: list, l2: "L2MemoryAdapter"
    ) -> int:
        """记录空抽取；相同内容连续两次为空后终态处理，避免永久重试。"""
        finalized = 0
        for memory in memories:
            content_hash = hashlib.sha256(memory.content.encode("utf-8")).hexdigest()
            same_content = memory.metadata.get("kg_attempt_content_hash") == content_hash
            attempts = (
                int(memory.metadata.get("kg_empty_attempts", 0))
                if same_content
                else 0
            )
            attempts += 1
            memory.metadata["kg_attempt_content_hash"] = content_hash
            memory.metadata["kg_empty_attempts"] = attempts
            memory.metadata["kg_last_result"] = "empty"
            if attempts >= 2:
                memory.metadata["kg_processed"] = True
                finalized += 1
            await l2.update_metadata(memory.id, memory.metadata)
        return finalized

    async def _build_user_aliases(
        self,
        memories: list,
        persona_id: str,
        component_manager=None,
    ) -> dict[str, list[str]]:
        """构建 {user_id: [昵称...]} 别名映射，供 L3 提取归一化 Person 节点

        数据来源：
        1. L2 记忆 metadata 中的 user_id + user_name（L1 落盘）；
        2. 用户画像的 user_name + historical_names（需 component_manager，
           未传入或不可用时退化为仅用 metadata）。
        """
        user_ids: set[str] = set()
        aliases: dict[str, set[str]] = {}

        for mem in memories:
            user_id = mem.metadata.get("user_id")
            if not user_id:
                continue
            user_ids.add(user_id)
            user_name = mem.metadata.get("user_name")
            if user_name:
                aliases.setdefault(user_id, set()).add(user_name)

        if component_manager is not None and user_ids:
            try:
                from iris_memory.profile.storage import ProfileStorage

                profile_storage = component_manager.get_component("profile")
                if profile_storage and getattr(
                    profile_storage, "is_available", False
                ):
                    assert isinstance(profile_storage, ProfileStorage)
                    config = get_config()
                    group_id = memories[0].group_id if memories else ""
                    effective_group_id = (
                        group_id
                        if config.get("isolation_config.enable_group_isolation")
                        else "default"
                    )
                    for user_id in user_ids:
                        profile = await profile_storage.get_user_profile(
                            user_id, effective_group_id, persona_id
                        )
                        if not profile:
                            continue
                        names = aliases.setdefault(user_id, set())
                        if profile.user_name:
                            names.add(profile.user_name)
                        for hist in profile.historical_names:
                            if hist:
                                names.add(hist)
            except Exception as e:
                logger.warning(f"从画像构建用户别名失败，退化为仅 metadata：{e}")

        return {
            user_id: sorted(names)
            for user_id, names in aliases.items()
            if names
        }

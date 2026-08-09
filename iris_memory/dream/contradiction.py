"""
Iris Chat Memory - 梦境阶段3：矛盾消解

检测记忆间的逻辑冲突，保留更新、更可靠的那一条。

Features:
    - 基于已存向量的近邻矛盾检测（零 embedding 调用）
    - LLM 矛盾判断
    - 多维度冲突解决策略（时间 > 置信度 > 访问频率）
    - 采样扫描预算（大数据量优化）
    - 批量向量检索（batch query）
"""

import random
import json
from datetime import datetime
from typing import Dict, List, Optional, cast

from iris_memory.core import get_logger
from iris_memory.llm_modules import DREAM_CONTRADICTION
from iris_memory.config import get_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager
from iris_memory.l2_memory.models import MemoryEntry

logger = get_logger("dream.contradiction")


class ContradictionPhase:
    """矛盾消解阶段

    检测记忆间的逻辑冲突，保留更新、更可靠的那一条。
    """

    def __init__(self):
        self._similarity_floor = 0.55
        self._similarity_ceiling = 0.85
        self._max_groups = 20
        self._scan_budget = 200
        self._query_batch_size = 50
        self._query_top_k = 5
        self._llm_batch_size = 5

    async def execute(
        self,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        llm: Optional["LLMManager"],
        entries: Optional[List["MemoryEntry"]] = None,
        persona_id: str = "default",
    ) -> dict:
        config = get_config()
        self._similarity_floor = cast(
            float, config.get("dream_contradiction_similarity_floor")
        )
        self._similarity_ceiling = cast(
            float, config.get("dream_contradiction_similarity_ceiling")
        )
        self._max_groups = cast(int, config.get("dream_contradiction_max_groups"))
        self._scan_budget = cast(
            int, config.get("dream_contradiction_scan_budget")
        )
        self._query_batch_size = cast(
            int, config.get("dream_contradiction_query_batch_size")
        )
        self._llm_batch_size = cast(
            int, config.get("dream_contradiction_llm_batch_size", 5)
        )

        if not llm:
            logger.warning("LLMManager 不可用，跳过矛盾消解")
            return {"groups_checked": 0, "contradictions_found": 0, "resolved": 0}

        try:
            if entries is None:
                entries = await l2.get_all_entries(persona_id=persona_id)

            if len(entries) < 2:
                logger.debug("记忆数量不足，无需矛盾检测")
                return {"groups_checked": 0, "contradictions_found": 0, "resolved": 0}

            logger.info(f"开始检测 {len(entries)} 条记忆的逻辑矛盾...")

            candidate_groups = await self._find_contradiction_candidates(
                entries, l2, persona_id
            )

            if not candidate_groups:
                logger.debug("未发现矛盾候选")
                return {"groups_checked": 0, "contradictions_found": 0, "resolved": 0}

            logger.info(f"发现 {len(candidate_groups)} 组矛盾候选")

            groups_checked = 0
            contradictions_found = 0
            resolved = 0

            selected_groups = candidate_groups[: self._max_groups]
            for offset in range(0, len(selected_groups), self._llm_batch_size):
                batch = selected_groups[offset : offset + self._llm_batch_size]
                try:
                    batch_results = await self._check_and_resolve_batch(batch, l2, llm)
                    groups_checked += len(batch)
                    for result in batch_results:
                        if result is not None:
                            contradictions_found += 1
                            if result:
                                resolved += 1
                except Exception as e:
                    groups_checked += len(batch)
                    logger.error(f"批量矛盾消解失败：{e}", exc_info=True)

            logger.info(
                f"矛盾消解完成：检查 {groups_checked} 组，"
                f"发现 {contradictions_found} 个矛盾，解决 {resolved} 个"
            )
            return {
                "groups_checked": groups_checked,
                "contradictions_found": contradictions_found,
                "resolved": resolved,
            }

        except Exception as e:
            logger.error(f"矛盾消解失败：{e}", exc_info=True)
            return {
                "groups_checked": 0,
                "contradictions_found": 0,
                "resolved": 0,
                "error": str(e),
            }

    async def _find_contradiction_candidates(
        self,
        entries: List["MemoryEntry"],
        l2: "L2MemoryAdapter",
        persona_id: str = "default",
    ) -> List[List["MemoryEntry"]]:
        config = get_config()
        enable_group_isolation = bool(
            config.get("isolation_config.enable_group_memory_isolation")
        )

        # 扫描预算：记忆量超过上限时随机采样，控制单轮检索规模
        if len(entries) > self._scan_budget:
            scan_entries = random.sample(entries, self._scan_budget)
            logger.info(
                f"记忆数量 {len(entries)} 超过扫描预算 {self._scan_budget}，"
                f"随机采样 {self._scan_budget} 条"
            )
        else:
            scan_entries = entries

        entry_index: Dict[str, "MemoryEntry"] = {e.id: e for e in entries}

        if enable_group_isolation:
            groups_by_gid: Dict[Optional[str], List["MemoryEntry"]] = {}
            for e in scan_entries:
                groups_by_gid.setdefault(e.group_id, []).append(e)
        else:
            groups_by_gid = {None: scan_entries}

        candidate_edges: list[tuple[float, tuple[str, str]]] = []
        seen_pairs: set = set()
        total_queries = 0

        for gid, group_entries in groups_by_gid.items():
            for i in range(0, len(group_entries), self._query_batch_size):
                batch = group_entries[i : i + self._query_batch_size]

                try:
                    # 查询对象本身已在 L2 库中，直接复用索引中已存的向量，
                    # 无需对文本重新计算 embedding
                    results_batch = await l2.batch_retrieve_by_ids(
                        memory_ids=[e.id for e in batch],
                        group_id=gid,
                        top_k=self._query_top_k,
                        persona_id=persona_id,
                    )
                except Exception as e:
                    logger.warning(f"批量检索矛盾候选失败：{e}")
                    continue

                for query_entry, results in zip(batch, results_batch):
                    for result in results:
                        if result.entry.id == query_entry.id:
                            continue
                        if result.score < self._similarity_floor:
                            continue
                        if result.score >= self._similarity_ceiling:
                            continue
                        if enable_group_isolation:
                            if result.entry.group_id != query_entry.group_id:
                                continue

                        pair_key = tuple(sorted([query_entry.id, result.entry.id]))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        candidate_edges.append((result.score, pair_key))

                total_queries += len(batch)

        # 每条记忆每轮最多进入一个二元候选。这样既避免重复 LLM 判断/陈旧
        # 删除，也不会把 A-B、B-C 的相似链误当成 A/B/C 必须三选一。
        candidates: List[List["MemoryEntry"]] = []
        used_ids: set[str] = set()
        for _score, pair in sorted(candidate_edges, reverse=True):
            left_id, right_id = pair
            if left_id in used_ids or right_id in used_ids:
                continue
            left = entry_index.get(left_id)
            right = entry_index.get(right_id)
            if left is None or right is None:
                continue
            candidates.append([left, right])
            used_ids.update(pair)

        logger.info(
            f"矛盾检测扫描 {total_queries} 条记忆，"
            f"发现 {len(candidates)} 组矛盾候选"
        )
        return candidates

    async def _check_and_resolve_batch(
        self,
        groups: List[List["MemoryEntry"]],
        l2: "L2MemoryAdapter",
        llm: "LLMManager",
    ) -> List[Optional[bool]]:
        """一次调用判断多个互不重叠候选组，并安全应用结果。"""
        if not groups:
            return []

        sections = []
        for group_index, group in enumerate(groups, 1):
            lines = []
            for memory_index, entry in enumerate(group, 1):
                ts = entry.metadata.get("timestamp", "未知时间")
                lines.append(f"[{memory_index}] {ts}: {entry.content}")
            sections.append(f"GROUP {group_index}\n" + "\n".join(lines))

        prompt = f"""判断下列多组记忆是否各自存在逻辑矛盾。各组相互独立。

{chr(10).join(sections)}

严格输出 JSON 数组，每组一个对象：
[
  {{"group": 1, "conflict": true, "keep": 2, "merged": "保留主体的正确记忆"}},
  {{"group": 2, "conflict": false, "keep": null, "merged": ""}}
]
仅在确有冲突时 conflict=true；keep 是该组中应保留的 1-based 编号。"""
        try:
            response = await llm.generate_direct(
                prompt=prompt, module=DREAM_CONTRADICTION
            )
            decisions = self._parse_batch_decisions(response)
        except Exception as e:
            logger.error(f"矛盾消解 LLM 批量调用失败：{e}")
            return [None] * len(groups)

        results: List[Optional[bool]] = [None] * len(groups)
        for decision in decisions:
            try:
                group_idx = int(decision.get("group", 0)) - 1
                if group_idx < 0 or group_idx >= len(groups):
                    continue
                if decision.get("conflict") is not True:
                    continue
                group = groups[group_idx]
                keep_idx = int(decision.get("keep", 0)) - 1
                merged = str(decision.get("merged", "")).strip()
                if keep_idx < 0 or keep_idx >= len(group) or not merged:
                    continue
                results[group_idx] = await self._apply_resolution(
                    group, keep_idx, merged, l2
                )
            except (TypeError, ValueError):
                continue
        return results

    @staticmethod
    def _parse_batch_decisions(response: Optional[str]) -> List[dict]:
        if not response:
            return []
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            data = json.loads(text.strip())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    async def _apply_resolution(
        self,
        group: List["MemoryEntry"],
        resolved_idx: int,
        merged_content: str,
        l2: "L2MemoryAdapter",
    ) -> bool:
        keep_entry = group[resolved_idx]
        delete_ids = [e.id for e in group if e.id != keep_entry.id]
        success = await l2.update_content(keep_entry.id, merged_content)
        if not success:
            logger.warning("矛盾合并内容更新失败，保留全部原始记忆")
            return False

        keep_entry.metadata["confidence"] = min(
            1.0, keep_entry.metadata.get("confidence", 0.5) + 0.1
        )
        keep_entry.metadata["updated_at"] = datetime.now().isoformat()
        keep_entry.metadata["kg_processed"] = False
        await l2.update_metadata(keep_entry.id, keep_entry.metadata)
        if delete_ids:
            await l2.delete_entries(delete_ids)
        logger.info(f"矛盾消解：保留 [{resolved_idx + 1}]，删除 {len(delete_ids)} 条")
        return True

    async def _check_and_resolve(
        self,
        group: List["MemoryEntry"],
        l2: "L2MemoryAdapter",
        llm: "LLMManager",
    ) -> Optional[bool]:
        if len(group) < 2:
            return None

        memory_texts = []
        for i, entry in enumerate(group):
            ts = entry.metadata.get("timestamp", "未知时间")
            memory_texts.append(f"[{i + 1}] {ts}: {entry.content}")

        prompt = f"""以下记忆可能存在逻辑矛盾，请分析：

{chr(10).join(memory_texts)}

如果存在矛盾：
1. 指出哪条记忆更新/更可靠
2. 输出合并后的正确记忆（保留正确信息，去除过时或错误信息）
3. 合并后的记忆必须保留主体（是谁的偏好/行为），禁止写成无主体的表述
4. 格式：
RESOLVED: <保留的记忆编号>
MERGED: <合并后的正确记忆内容>

如果不矛盾，回复 NO_CONFLICT。"""

        try:
            response = await llm.generate_direct(
                prompt=prompt, module=DREAM_CONTRADICTION
            )

            if not response or not response.strip():
                return None

            response = response.strip()

            if response.upper().startswith("NO_CONFLICT"):
                return None

            if "RESOLVED:" not in response or "MERGED:" not in response:
                logger.debug(f"LLM 矛盾判断格式不符：{response[:100]}")
                return None

            resolved_idx = self._parse_resolved_index(response, group_size=len(group))
            merged_content = self._parse_merged_content(response)

            if resolved_idx is None or not merged_content:
                return None

            return await self._apply_resolution(
                group, resolved_idx, merged_content, l2
            )

        except Exception as e:
            logger.error(f"矛盾消解 LLM 调用失败：{e}")
            return None

    def _parse_resolved_index(
        self, response: str, group_size: int = 0
    ) -> Optional[int]:
        try:
            for line in response.split("\n"):
                if line.strip().upper().startswith("RESOLVED:"):
                    val = line.split(":", 1)[1].strip()
                    idx = int(val) - 1
                    if idx < 0 or (group_size > 0 and idx >= group_size):
                        logger.warning(
                            f"RESOLVED 索引越界：{val}，group_size={group_size}"
                        )
                        return None
                    return idx
        except (ValueError, IndexError):
            pass
        return None

    def _parse_merged_content(self, response: str) -> Optional[str]:
        try:
            in_merged = False
            lines = []
            for line in response.split("\n"):
                if line.strip().upper().startswith("MERGED:"):
                    in_merged = True
                    remainder = line.split(":", 1)[1].strip()
                    if remainder:
                        lines.append(remainder)
                    continue
                if in_merged:
                    if (
                        line.strip()
                        .upper()
                        .startswith(
                            ("RESOLVED:", "PATTERN:", "EVIDENCE:", "CONFIDENCE:")
                        )
                    ):
                        break
                    lines.append(line)
            if lines:
                return "\n".join(lines).strip()
        except (ValueError, IndexError):
            pass
        return None

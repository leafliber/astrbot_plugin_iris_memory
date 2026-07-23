"""
Iris Chat Memory - 梦境阶段4：模式挖掘

跨记忆发现隐含的行为规律、偏好模式、因果关联，
归类到已有节点类型（Trait/Preference/Belief/Goal/Skill）
并建立与 Person 的关系边。

Features:
    - 按群聊/用户分组采样
    - LLM 模式提取（含类型归类和人物关联）
    - 写入 L2 + L3（具体类型节点 + 关系边）
    - 向量检索去重
"""

import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, cast

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager
from iris_memory.l3_kg.models import GraphNode, GraphEdge

logger = get_logger("dream.pattern_discovery")

# 置信度级别排序，用于按 dream_pattern_min_confidence 过滤低置信度模式
_CONFIDENCE_LEVEL = {"low": 0, "medium": 1, "high": 2}

# 模式允许映射到的节点类型 → 对应关系边
_TYPE_TO_RELATION = {
    "Trait": "HAS_TRAIT",
    "Preference": "HAS_PREFERENCE",
    "Belief": "HAS_BELIEF",
    "Goal": "HAS_GOAL",
    "Skill": "HAS_SKILL",
}

_ALLOWED_TYPES = set(_TYPE_TO_RELATION.keys())


class PatternDiscoveryPhase:
    """模式挖掘阶段

    跨记忆发现隐含的行为规律、偏好模式、因果关联，
    归类为 Trait/Preference/Belief/Goal/Skill 并关联到 Person。
    """

    def __init__(self):
        self._sample_size = 30
        self._min_confidence = "medium"

    async def execute(
        self,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        llm: Optional["LLMManager"],
        entries: Optional[list] = None,
        persona_id: str = "default",
    ) -> dict:
        config = get_config()
        self._sample_size = cast(int, config.get("dream_pattern_sample_size"))
        raw_confidence = cast(str, config.get("dream_pattern_min_confidence", "medium"))
        self._min_confidence = raw_confidence.lower() if raw_confidence else "medium"
        if self._min_confidence not in _CONFIDENCE_LEVEL:
            logger.warning(
                f"dream_pattern_min_confidence 值 '{raw_confidence}' 无效，"
                f"回退为 'medium'（可选: low/medium/high）"
            )
            self._min_confidence = "medium"

        if not llm:
            logger.warning("LLMManager 不可用，跳过模式挖掘")
            return {"groups_analyzed": 0, "patterns_found": 0, "patterns_written": 0}

        try:
            if entries is None:
                entries = await l2.get_all_entries(persona_id=persona_id)

            if not entries:
                logger.debug("L2 记忆库为空，跳过模式挖掘")
                return {
                    "groups_analyzed": 0,
                    "patterns_found": 0,
                    "patterns_written": 0,
                }

            groups = self._group_entries(entries)

            logger.info(f"开始模式挖掘：{len(groups)} 个分组，共 {len(entries)} 条记忆")

            groups_analyzed = 0
            patterns_found = 0
            patterns_written = 0

            for group_key, group_entries in groups.items():
                if len(group_entries) < 3:
                    continue

                groups_analyzed += 1

                sample = (
                    random.sample(group_entries, self._sample_size)
                    if len(group_entries) > self._sample_size
                    else group_entries
                )

                try:
                    patterns = await self._extract_patterns(sample, llm)
                    patterns_found += len(patterns)

                    min_level = _CONFIDENCE_LEVEL.get(self._min_confidence, 1)
                    for pattern in patterns:
                        conf_level = _CONFIDENCE_LEVEL.get(
                            pattern.get("confidence", "medium"), 1
                        )
                        if conf_level < min_level:
                            continue

                        is_dup = await self._check_duplicate(
                            pattern["description"], l2, persona_id
                        )
                        if is_dup:
                            logger.debug(
                                f"模式已存在，跳过：{pattern['description'][:50]}"
                            )
                            continue

                        written = await self._write_pattern(
                            pattern, group_key, l2, l3, persona_id
                        )
                        if written:
                            patterns_written += 1

                except Exception as e:
                    logger.error(f"分组 [{group_key}] 模式挖掘失败：{e}", exc_info=True)

            logger.info(
                f"模式挖掘完成：分析 {groups_analyzed} 组，"
                f"发现 {patterns_found} 个模式，写入 {patterns_written} 个"
            )
            return {
                "groups_analyzed": groups_analyzed,
                "patterns_found": patterns_found,
                "patterns_written": patterns_written,
            }

        except Exception as e:
            logger.error(f"模式挖掘失败：{e}", exc_info=True)
            return {
                "groups_analyzed": 0,
                "patterns_found": 0,
                "patterns_written": 0,
                "error": str(e),
            }

    def _group_entries(self, entries: list) -> Dict[str, list]:
        config = get_config()
        enable_group_isolation = bool(
            config.get("isolation_config.enable_group_memory_isolation")
        )

        groups: Dict[str, list] = defaultdict(list)

        if enable_group_isolation:
            for entry in entries:
                gid = entry.group_id or "_no_group"
                groups[gid].append(entry)
        else:
            groups["_all"] = entries

        return dict(groups)

    async def _extract_patterns(self, entries: list, llm: "LLMManager") -> List[dict]:
        memory_texts = []
        for i, entry in enumerate(entries, 1):
            user_id = entry.metadata.get("user_id", "")
            user_prefix = f"[用户:{user_id}] " if user_id else ""
            memory_texts.append(f"{i}. {user_prefix}{entry.content}")

        prompt = f"""以下是同一用户/群聊的若干记忆片段，请挖掘其中隐含的行为模式、偏好规律或因果关联。
只输出你确信发现的模式，不要猜测。

记忆片段：
{chr(10).join(memory_texts)}

输出格式（每行一个模式）：
TYPE: <Trait/Preference/Belief/Goal/Skill 之一>
PERSON: <用户标识，必须来自记忆片段中的[用户:xxx]标记；无法确定则不输出该模式>
DESCRIPTION: <具体描述，必须包含主体，如"张三喜欢使用孙权"而非"有角色偏好">
EVIDENCE: <支撑记忆编号，逗号分隔>
CONFIDENCE: <high/medium/low>

类型：Trait=性格行为 / Preference=喜好倾向 / Belief=观点价值观 / Goal=计划意图 / Skill=能力

PERSON 必须填写，无法确定归属用户时不要输出该模式。没有可靠模式则输出 NONE。"""

        try:
            response = await llm.generate_direct(
                prompt=prompt, module="dream_pattern_discovery"
            )

            if not response or not response.strip():
                return []

            if "NONE" in response.strip().upper():
                return []

            return self._parse_patterns(response)

        except Exception as e:
            logger.error(f"LLM 模式提取失败：{e}")
            return []

    def _parse_patterns(self, response: str) -> List[dict]:
        patterns = []
        current: dict = {}

        for line in response.strip().split("\n"):
            line = line.strip()
            upper = line.upper()

            if upper.startswith("TYPE:"):
                if current.get("description"):
                    patterns.append(current)
                raw_type = line.split(":", 1)[1].strip()
                current = {
                    "type": raw_type if raw_type in _ALLOWED_TYPES else "Trait",
                }
            elif upper.startswith("PERSON:") and "type" in current:
                current["person"] = line.split(":", 1)[1].strip()
            elif upper.startswith("DESCRIPTION:") and "type" in current:
                current["description"] = line.split(":", 1)[1].strip()
            elif upper.startswith("EVIDENCE:") and "type" in current:
                current["evidence"] = line.split(":", 1)[1].strip()
            elif upper.startswith("CONFIDENCE:") and "type" in current:
                conf = line.split(":", 1)[1].strip().lower()
                current["confidence"] = (
                    conf if conf in ("high", "medium", "low") else "low"
                )

        if current.get("description"):
            patterns.append(current)

        return patterns

    async def _check_duplicate(
        self, description: str, l2: "L2MemoryAdapter", persona_id: str = "default"
    ) -> bool:
        try:
            results = await l2.retrieve(description, top_k=3, persona_id=persona_id)
            for result in results:
                if (
                    result.score > 0.9
                    and result.entry.metadata.get("source") == "dream_pattern"
                ):
                    return True
        except Exception:
            pass
        return False

    async def _write_pattern(
        self,
        pattern: dict,
        group_key: str,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        persona_id: str = "default",
    ) -> bool:
        description = pattern["description"]
        node_type = pattern.get("type", "Trait")
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
        confidence = confidence_map.get(pattern.get("confidence", "low"), 0.4)

        # 主体校验：PERSON 为空时不写入 L2。
        # 无主体的模式记忆（如"有特定角色偏好"不知道是谁的偏好）
        # 没有检索价值，且会流入下游 L3 产生孤儿节点。
        person_id = pattern.get("person", "").strip()
        if not person_id:
            logger.info(
                f"模式挖掘跳过无主体记忆：'{description[:50]}' "
                f"（PERSON 字段为空，无法关联到用户）"
            )
            return False

        new_id = await l2.add_memory(
            description,
            metadata={
                "source": "dream_pattern",
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "group_id": group_key if group_key != "_all" else None,
                "evidence": pattern.get("evidence", ""),
                "pattern_type": node_type,
                "user_id": person_id,
            },
            persona_id=persona_id,
        )

        if not new_id:
            return False

        if l3 and l3.is_available:
            try:
                # 创建具体类型节点
                node = GraphNode(
                    id="",
                    label=node_type,
                    name=description[:50],
                    content=description,
                    confidence=confidence,
                    group_id=group_key if group_key != "_all" else None,
                    properties={"source": "dream_pattern"},
                )
                node.id = node.generate_id()
                node_added = await l3.add_node(node)

                # 建立与 Person 的关系边
                if node_added:
                    await self._link_to_person(
                        l3,
                        person_id,
                        node.id,
                        node_type,
                        group_key,
                        confidence,
                    )
            except Exception as e:
                logger.debug(f"写入 L3 节点失败：{e}")

        return True

    async def _link_to_person(
        self,
        l3: "L3KGAdapter",
        person_id_str: str,
        target_node_id: str,
        node_type: str,
        group_key: str,
        confidence: float,
    ) -> None:
        """查找或创建 Person 节点，建立关系边"""
        relation_type = _TYPE_TO_RELATION.get(node_type)
        if not relation_type:
            return

        try:
            # 搜索是否已有该 Person 节点
            existing = await l3.search_nodes(person_id_str, limit=5)
            person_node_id = None
            for n in existing:
                if n.get("label") == "Person" and n.get("name", "") == person_id_str:
                    person_node_id = n["id"]
                    break

            # 不存在则创建
            if not person_node_id:
                person = GraphNode(
                    id="",
                    label="Person",
                    name=person_id_str,
                    content=f"用户 {person_id_str}",
                    confidence=0.5,
                    group_id=group_key if group_key != "_all" else None,
                    properties={"source": "dream_pattern"},
                )
                person.id = person.generate_id()
                added = await l3.add_node(person)
                if not added:
                    return
                person_node_id = person.id

            # 创建关系边
            edge = GraphEdge(
                source_id=person_node_id,
                target_id=target_node_id,
                relation_type=relation_type,
                weight=confidence,
                confidence=confidence,
            )
            await l3.add_edge(edge)
            logger.debug(
                f"建立关系边：{person_node_id} --[{relation_type}]--> {target_node_id}"
            )
        except Exception as e:
            logger.debug(f"建立关系边失败：{e}")

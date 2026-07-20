"""
PatternDiscoveryPhase 模式挖掘测试

测试核心功能：
- 记忆分组采样
- LLM 模式提取（含类型归类和人物关联）
- 模式解析（TYPE/PERSON/DESCRIPTION/EVIDENCE/CONFIDENCE）
- 去重写入
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from iris_memory.dream.pattern_discovery import PatternDiscoveryPhase


def _mock_config():
    mock = Mock()
    mock.get = Mock(
        side_effect=lambda key, default=None: {
            "dream_pattern_sample_size": 30,
            "dream_pattern_min_confidence": "medium",
            "isolation_config.enable_group_memory_isolation": False,
        }.get(key, default)
    )
    return mock


class TestPatternDiscoveryPhase:
    @pytest.fixture
    def phase(self):
        return PatternDiscoveryPhase()

    @pytest.mark.asyncio
    async def test_execute_no_llm(self, phase):
        l2 = Mock()
        l2.is_available = True
        l3 = None
        llm = None

        with patch(
            "iris_memory.dream.pattern_discovery.get_config",
            return_value=_mock_config(),
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["patterns_found"] == 0
        assert result["patterns_written"] == 0

    @pytest.mark.asyncio
    async def test_execute_no_entries(self, phase):
        l2 = Mock()
        l2.is_available = True
        l2.get_all_entries = AsyncMock(return_value=[])
        l3 = None
        llm = Mock()

        with patch(
            "iris_memory.dream.pattern_discovery.get_config",
            return_value=_mock_config(),
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["patterns_found"] == 0
        assert result["patterns_written"] == 0

    def test_parse_patterns_valid(self, phase):
        response = """TYPE: Preference
PERSON: user123
DESCRIPTION: 用户偏好使用Python进行开发
EVIDENCE: 1,3,5
CONFIDENCE: high

TYPE: Trait
PERSON: user456
DESCRIPTION: 用户习惯在晚上讨论技术问题
EVIDENCE: 2,4
CONFIDENCE: medium"""

        patterns = phase._parse_patterns(response)

        assert len(patterns) == 2
        assert patterns[0]["type"] == "Preference"
        assert patterns[0]["person"] == "user123"
        assert patterns[0]["description"] == "用户偏好使用Python进行开发"
        assert patterns[0]["confidence"] == "high"
        assert patterns[1]["type"] == "Trait"
        assert patterns[1]["confidence"] == "medium"

    def test_parse_patterns_empty(self, phase):
        patterns = phase._parse_patterns("")
        assert len(patterns) == 0

    def test_parse_patterns_no_confidence(self, phase):
        response = "TYPE: Belief\nPERSON: \nDESCRIPTION: 测试模式\nEVIDENCE: 1"
        patterns = phase._parse_patterns(response)
        assert len(patterns) == 1
        assert patterns[0]["type"] == "Belief"
        assert patterns[0]["description"] == "测试模式"
        assert "confidence" not in patterns[0]

    def test_parse_patterns_unknown_type_defaults_to_trait(self, phase):
        response = (
            "TYPE: UnknownType\nDESCRIPTION: 某个模式\nEVIDENCE: 1\nCONFIDENCE: high"
        )
        patterns = phase._parse_patterns(response)
        assert len(patterns) == 1
        assert patterns[0]["type"] == "Trait"

    def test_parse_patterns_no_person(self, phase):
        response = (
            "TYPE: Goal\nDESCRIPTION: 想学习Rust\nEVIDENCE: 3,7\nCONFIDENCE: medium"
        )
        patterns = phase._parse_patterns(response)
        assert len(patterns) == 1
        assert patterns[0].get("person", "") == ""

    def test_type_to_relation_mapping(self):
        from iris_memory.dream.pattern_discovery import _TYPE_TO_RELATION

        assert _TYPE_TO_RELATION["Trait"] == "HAS_TRAIT"
        assert _TYPE_TO_RELATION["Preference"] == "HAS_PREFERENCE"
        assert _TYPE_TO_RELATION["Belief"] == "HAS_BELIEF"
        assert _TYPE_TO_RELATION["Goal"] == "HAS_GOAL"
        assert _TYPE_TO_RELATION["Skill"] == "HAS_SKILL"

    @pytest.mark.asyncio
    async def test_link_to_person_uses_exact_name_match(self, phase):
        """回归：_link_to_person 应使用精确名称匹配，而非子串匹配

        历史 bug：使用 ``person_id_str in n.get("name", "")`` 子串匹配，
        导致搜索 "alice" 时会错误命中 "malice"（"alice" in "malice" 为 True）。
        修复后使用 ``n.get("name", "") == person_id_str`` 精确匹配。

        将 "malice" 放在搜索结果首位以暴露 bug：子串匹配会先命中 "malice"，
        而精确匹配只会命中 "alice"。
        """
        l3 = Mock()
        l3.is_available = True
        l3.search_nodes = AsyncMock(
            return_value=[
                {"id": "malice_id", "label": "Person", "name": "malice"},
                {"id": "alice_id", "label": "Person", "name": "alice"},
            ]
        )
        l3.add_node = AsyncMock(return_value=True)
        l3.add_edge = AsyncMock(return_value=True)

        await phase._link_to_person(
            l3,
            person_id_str="alice",
            target_node_id="target_node_id",
            node_type="Trait",
            group_key="_all",
            confidence=0.7,
        )

        # 应精确匹配到 "alice"，而非子串命中的 "malice"
        l3.add_edge.assert_called_once()
        edge = l3.add_edge.call_args.args[0]
        assert edge.source_id == "alice_id"
        assert edge.target_id == "target_node_id"

        # alice 已存在，不应创建新的 Person 节点
        l3.add_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_pattern_skips_subjectless_node(self, phase):
        """回归：_write_pattern 在 person 字段为空时跳过创建 L3 节点

        此前 person_id 为空仍创建节点，导致无主节点（如"有特定角色偏好"
        不知道是谁的偏好）无法关联到用户，成为图谱中的孤儿。修复后当
        PERSON 字段为空时直接返回，不调用 l3.add_node。
        """
        l2 = Mock()
        l2.add_memory = AsyncMock(return_value="mem_new_id")
        l3 = Mock()
        l3.is_available = True
        l3.add_node = AsyncMock(return_value=True)
        l3.add_edge = AsyncMock(return_value=True)

        pattern = {
            "type": "Preference",
            "person": "",
            "description": "有特定角色偏好",
            "evidence": "1,3",
            "confidence": "high",
        }

        written = await phase._write_pattern(pattern, "_all", l2, l3)

        # L2 仍然写入
        l2.add_memory.assert_called_once()
        # 返回 True（L2 写入成功），但 L3 不应创建节点
        assert written is True
        l3.add_node.assert_not_called()
        l3.add_edge.assert_not_called()

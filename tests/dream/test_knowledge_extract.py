"""
KnowledgeExtractPhase 知识提取测试

测试核心功能：
- 未处理记忆筛选
- 记忆分组
- 实体关系提取
- L3 写入
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from iris_memory.dream.knowledge_extract import KnowledgeExtractPhase
from iris_memory.l2_memory.models import MemoryEntry


def _mock_config():
    mock = Mock()
    mock.get = Mock(
        side_effect=lambda key, default=None: {
            "dream_knowledge_extract_min_unprocessed": 10,
            "dream_knowledge_extract_batch_size": 20,
            "isolation_config.enable_group_memory_isolation": False,
        }.get(key, default)
    )
    return mock


class TestKnowledgeExtractPhase:
    @pytest.fixture
    def phase(self):
        return KnowledgeExtractPhase()

    @pytest.mark.asyncio
    async def test_execute_l3_unavailable(self, phase):
        l2 = Mock()
        l2.is_available = True
        l3 = Mock()
        l3.is_available = False
        llm = Mock()

        with patch(
            "iris_memory.dream.knowledge_extract.get_config",
            return_value=_mock_config(),
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["memories_processed"] == 0
        assert result["nodes_extracted"] == 0

    @pytest.mark.asyncio
    async def test_execute_no_llm(self, phase):
        l2 = Mock()
        l2.is_available = True
        l3 = Mock()
        l3.is_available = True
        llm = None

        with patch(
            "iris_memory.dream.knowledge_extract.get_config",
            return_value=_mock_config(),
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["memories_processed"] == 0
        assert result["nodes_extracted"] == 0

    @pytest.mark.asyncio
    async def test_execute_no_unprocessed(self, phase):
        l2 = Mock()
        l2.is_available = True
        l2.get_unprocessed_count = AsyncMock(return_value=3)
        l3 = Mock()
        l3.is_available = True
        llm = Mock()

        with patch(
            "iris_memory.dream.knowledge_extract.get_config",
            return_value=_mock_config(),
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["memories_processed"] == 0

    @pytest.mark.asyncio
    async def test_all_writes_fail_does_not_mark_processed(self, phase):
        """回归：提取结果非空但全部写入失败时，不标记记忆为已处理

        此前只要 result.nodes 或 result.edges 非空就标记已处理，
        导致 L3 写入全失败的记忆被永久跳过无法重试。
        修复后仅当至少一条节点/边写入成功时才标记。
        """
        from iris_memory.l2_memory.models import MemoryEntry
        from iris_memory.l3_kg.models import GraphNode, GraphEdge, ExtractionResult

        # 构造一条未处理记忆
        mem = MemoryEntry(
            id="mem_1",
            content="Alice 喜欢编程",
            metadata={"group_id": "group_123", "user_id": "u1"},
        )

        l2 = Mock()
        l2.is_available = True
        l2.get_unprocessed_count = AsyncMock(return_value=10)
        l2.get_unprocessed_memories = AsyncMock(return_value=[mem])
        l2.mark_memories_processed = AsyncMock()

        l3 = Mock()
        l3.is_available = True
        l3.add_node = AsyncMock(return_value=False)
        l3.add_edge = AsyncMock(return_value=False)

        llm = Mock()

        # 构造非空提取结果（有节点也有边）
        node = GraphNode(id="", label="Person", name="Alice", content="软件工程师")
        node.id = node.generate_id()
        edge = GraphEdge(
            source_id="src_id", target_id="tgt_id", relation_type="KNOWS"
        )
        fake_result = ExtractionResult(nodes=[node], edges=[edge])

        with patch(
            "iris_memory.dream.knowledge_extract.get_config",
            return_value=_mock_config(),
        ):
            with patch("iris_memory.l3_kg.EntityExtractor") as MockExtractor:
                MockExtractor.return_value.extract_from_memories = AsyncMock(
                    return_value=fake_result
                )
                result = await phase.execute(l2, l3, llm)

        # 全部写入失败，不应标记为已处理
        l2.mark_memories_processed.assert_not_called()
        assert result["memories_processed"] == 0
        assert result["nodes_extracted"] == 0
        assert result["edges_extracted"] == 0

    # ------------------------------------------------------------------
    # _build_user_aliases（user_id→昵称 别名映射）回归测试
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_build_user_aliases_from_metadata(self, phase):
        """从 L2 metadata 的 user_id + user_name 构建别名映射"""
        from iris_memory.l2_memory.models import MemoryEntry

        memories = [
            MemoryEntry(
                id="m1", content="x", metadata={"user_id": "u1", "user_name": "庭"}
            ),
            MemoryEntry(
                id="m2", content="y", metadata={"user_id": "u1", "user_name": "小庭"}
            ),
            MemoryEntry(id="m3", content="z", metadata={"user_id": "u2"}),
        ]

        aliases = await phase._build_user_aliases(
            memories, "default", component_manager=None
        )

        # u1 聚合两个昵称；u2 无昵称来源，仍以空别名列表纳入，
        # 保证 name=user_id 的 Person 节点能命中归一化打标
        assert set(aliases.keys()) == {"u1", "u2"}
        assert set(aliases["u1"]) == {"庭", "小庭"}
        assert aliases["u2"] == []

    @pytest.mark.asyncio
    async def test_build_user_aliases_enriches_from_profile(self, phase):
        """有 component_manager 时，用画像 user_name + historical_names 补充"""
        from iris_memory.l2_memory.models import MemoryEntry
        from iris_memory.profile.storage import ProfileStorage

        memories = [
            MemoryEntry(
                id="m1", content="x", metadata={"user_id": "u1", "user_name": "庭"}
            ),
        ]

        profile = Mock()
        profile.user_name = "阿庭"
        profile.historical_names = ["旧昵称"]

        profile_storage = Mock(spec=ProfileStorage)
        profile_storage.is_available = True
        profile_storage.get_user_profile = AsyncMock(return_value=profile)

        cm = Mock()
        cm.get_component = Mock(return_value=profile_storage)

        with patch(
            "iris_memory.dream.knowledge_extract.get_config",
            return_value=_mock_config(),
        ):
            aliases = await phase._build_user_aliases(
                memories, "default", component_manager=cm
            )

        assert set(aliases["u1"]) == {"庭", "阿庭", "旧昵称"}

    @pytest.mark.asyncio
    async def test_build_user_aliases_degrades_without_component_manager(self, phase):
        """未传 component_manager 时退化为仅 metadata，不报错"""
        from iris_memory.l2_memory.models import MemoryEntry

        memories = [
            MemoryEntry(
                id="m1", content="x", metadata={"user_id": "u1", "user_name": "庭"}
            ),
        ]

        aliases = await phase._build_user_aliases(
            memories, "default", component_manager=None
        )

        assert aliases == {"u1": ["庭"]}

    @pytest.mark.asyncio
    async def test_build_user_aliases_degrades_on_profile_error(self, phase):
        """画像读取异常时退化为仅 metadata，不中断"""
        from iris_memory.l2_memory.models import MemoryEntry

        memories = [
            MemoryEntry(
                id="m1", content="x", metadata={"user_id": "u1", "user_name": "庭"}
            ),
        ]

        cm = Mock()
        cm.get_component = Mock(side_effect=RuntimeError("boom"))

        with patch(
            "iris_memory.dream.knowledge_extract.get_config",
            return_value=_mock_config(),
        ):
            aliases = await phase._build_user_aliases(
                memories, "default", component_manager=cm
            )

        assert aliases == {"u1": ["庭"]}

    @pytest.mark.asyncio
    async def test_execute_injects_user_aliases(self, phase):
        """execute 应将构建的 user_aliases 注入提取 context"""
        from iris_memory.l2_memory.models import MemoryEntry
        from iris_memory.l3_kg.models import ExtractionResult

        mem = MemoryEntry(
            id="m1",
            content="庭喜欢编程",
            metadata={"group_id": "group_A", "user_id": "u1", "user_name": "庭"},
        )

        l2 = Mock()
        l2.is_available = True
        l2.get_unprocessed_count = AsyncMock(return_value=10)
        l2.get_unprocessed_memories = AsyncMock(return_value=[mem])
        l2.mark_memories_processed = AsyncMock()

        l3 = Mock()
        l3.is_available = True
        l3.add_node = AsyncMock(return_value=True)
        l3.add_edge = AsyncMock(return_value=True)

        llm = Mock()

        captured = {}

        async def fake_extract(memories, context):
            captured.update(context)
            return ExtractionResult()

        with patch(
            "iris_memory.dream.knowledge_extract.get_config",
            return_value=_mock_config(),
        ):
            with patch("iris_memory.l3_kg.EntityExtractor") as MockExtractor:
                MockExtractor.return_value.extract_from_memories = AsyncMock(
                    side_effect=fake_extract
                )
                await phase.execute(l2, l3, llm)

        assert captured.get("user_aliases") == {"u1": ["庭"]}

    @pytest.mark.asyncio
    async def test_empty_extraction_is_finalized_after_two_identical_attempts(
        self, phase
    ):
        memory = MemoryEntry(id="m1", content="一次性寒暄", metadata={})
        l2 = Mock()
        l2.update_metadata = AsyncMock(return_value=True)

        first = await phase._record_empty_result([memory], l2)
        second = await phase._record_empty_result([memory], l2)

        assert first == 0
        assert second == 1
        assert memory.metadata["kg_empty_attempts"] == 2
        assert memory.metadata["kg_processed"] is True

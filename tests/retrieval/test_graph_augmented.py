"""图增强混合检索策略测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from iris_memory.retrieval.strategies.graph_augmented import GraphAugmentedHybridStrategy
from iris_memory.retrieval.strategies.base import StrategyParams
from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer


@pytest.fixture
def sample_memories():
    return [
        Memory(
            id="m1", type=MemoryType.FACT, content="用户喜欢编程",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            metadata={}, graph_nodes=["node_1"],
        ),
        Memory(
            id="m2", type=MemoryType.FACT, content="用户住在北京",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            metadata={},
        ),
    ]


@pytest.fixture
def graph_expanded_memory():
    return Memory(
        id="m3", type=MemoryType.RELATIONSHIP, content="用户和北京有关联",
        user_id="u1", storage_layer=StorageLayer.EPISODIC,
        metadata={},
    )


@pytest.fixture
def mock_deps(sample_memories, graph_expanded_memory):
    chroma = AsyncMock()
    chroma.query_memories = AsyncMock(return_value=sample_memories)
    chroma.get_memory = AsyncMock(return_value=graph_expanded_memory)

    kg = AsyncMock()
    neighbor = MagicMock()
    neighbor.memory_id = "m3"
    kg.get_neighbors = AsyncMock(return_value=[neighbor])

    update_access = AsyncMock()
    emotion_filter = MagicMock(side_effect=lambda mems, *a, **kw: mems)
    rerank = MagicMock(side_effect=lambda mems, *a, **kw: mems)
    get_working = AsyncMock(return_value=[])
    merge = MagicMock(side_effect=lambda a, b, **kw: a + b)
    session_mgr = MagicMock()

    return {
        "chroma": chroma,
        "kg": kg,
        "update_access": update_access,
        "emotion_filter": emotion_filter,
        "rerank": rerank,
        "get_working": get_working,
        "merge": merge,
        "session_mgr": session_mgr,
    }


@pytest.fixture
def strategy(mock_deps):
    return GraphAugmentedHybridStrategy(
        chroma_manager=mock_deps["chroma"],
        update_access_fn=mock_deps["update_access"],
        emotion_filter_fn=mock_deps["emotion_filter"],
        rerank_fn=mock_deps["rerank"],
        get_working_memories_fn=mock_deps["get_working"],
        merge_memories_fn=mock_deps["merge"],
        session_manager=mock_deps["session_mgr"],
        kg_storage=mock_deps["kg"],
    )


@pytest.fixture
def base_params():
    return StrategyParams(
        query="编程 北京",
        user_id="u1",
        group_id=None,
        top_k=5,
        storage_layer=None,
        persona_id=None,
        emotional_state=None,
    )


class TestGraphAugmentedExecution:
    @pytest.mark.asyncio
    async def test_returns_results(self, strategy, base_params):
        results = await strategy.execute(base_params)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_graph_expansion_adds_candidates(self, strategy, base_params, mock_deps):
        results = await strategy.execute(base_params)
        # m3 should be added via graph expansion
        ids = [m.id for m in results]
        assert "m3" in ids

    @pytest.mark.asyncio
    async def test_dedup_graph_expansion(self, strategy, base_params, mock_deps):
        """图扩展不应重复添加已有记忆"""
        # Make graph return same id as vector result
        neighbor = MagicMock()
        neighbor.memory_id = "m1"
        mock_deps["kg"].get_neighbors = AsyncMock(return_value=[neighbor])
        results = await strategy.execute(base_params)
        id_counts = {}
        for m in results:
            id_counts[m.id] = id_counts.get(m.id, 0) + 1
        for count in id_counts.values():
            assert count == 1

    @pytest.mark.asyncio
    async def test_pending_review_filtered(self, strategy, base_params, mock_deps):
        """pending_review 状态的记忆应被过滤"""
        pending_mem = Memory(
            id="m_pending", type=MemoryType.FACT, content="test",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            metadata={}, review_status="pending_review",
        )
        mock_deps["chroma"].query_memories = AsyncMock(return_value=[pending_mem])
        mock_deps["kg"].get_neighbors = AsyncMock(return_value=[])
        results = await strategy.execute(base_params)
        assert all(getattr(m, "review_status", None) not in ("pending_review", "rejected") for m in results)


class TestNoKnowledgeGraph:
    @pytest.mark.asyncio
    async def test_works_without_kg(self, mock_deps, base_params):
        strategy = GraphAugmentedHybridStrategy(
            chroma_manager=mock_deps["chroma"],
            update_access_fn=mock_deps["update_access"],
            emotion_filter_fn=mock_deps["emotion_filter"],
            rerank_fn=mock_deps["rerank"],
            get_working_memories_fn=mock_deps["get_working"],
            merge_memories_fn=mock_deps["merge"],
            session_manager=mock_deps["session_mgr"],
            kg_storage=None,
        )
        results = await strategy.execute(base_params)
        assert len(results) > 0


class TestEmptyResults:
    @pytest.mark.asyncio
    async def test_returns_empty_on_no_candidates(self, mock_deps, base_params):
        mock_deps["chroma"].query_memories = AsyncMock(return_value=[])
        strategy = GraphAugmentedHybridStrategy(
            chroma_manager=mock_deps["chroma"],
            update_access_fn=mock_deps["update_access"],
            emotion_filter_fn=mock_deps["emotion_filter"],
            rerank_fn=mock_deps["rerank"],
            get_working_memories_fn=mock_deps["get_working"],
            merge_memories_fn=mock_deps["merge"],
            kg_storage=None,
        )
        results = await strategy.execute(base_params)
        assert results == []

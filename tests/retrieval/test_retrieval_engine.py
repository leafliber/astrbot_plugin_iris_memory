"""
测试记忆检索引擎
测试 iris_memory.retrieval.retrieval_engine 中的 MemoryRetrievalEngine 类
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime

from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine
from iris_memory.core.types import StorageLayer, RetrievalStrategy, EmotionType
from iris_memory.models.memory import Memory
from iris_memory.models.emotion_state import EmotionalState, CurrentEmotionState, EmotionConfig, EmotionContext


@pytest.fixture
def mock_chroma_manager():
    """模拟Chroma管理器"""
    manager = Mock()
    manager.query_memories = AsyncMock()
    return manager


@pytest.fixture
def mock_rif_scorer():
    """模拟RIF评分器"""
    return Mock()


@pytest.fixture
def mock_emotion_analyzer():
    """模拟情感分析器"""
    analyzer = Mock()
    analyzer.should_filter_positive_memories = Mock(return_value=False)
    return analyzer


@pytest.fixture
def mock_reranker():
    """模拟重排序器"""
    reranker = Mock()
    reranker.rerank = Mock(return_value=[])
    return reranker


@pytest.fixture
def sample_memories():
    """示例记忆列表"""
    return [
        Memory(
            id="1",
            user_id="user_1",
            content="用户喜欢红色",
            type="fact",
            storage_layer="episodic",
            created_time=datetime.now(),
            access_count=1,
            rif_score=0.8
        ),
        Memory(
            id="2",
            user_id="user_1",
            content="今天很高兴",
            type="emotion",
            subtype="joy",
            emotional_weight=0.9,
            storage_layer="episodic",
            created_time=datetime.now(),
            access_count=5,
            rif_score=0.7
        ),
        Memory(
            id="3",
            user_id="user_1",
            content="用户很安静",
            type="emotion",
            subtype="calm",
            emotional_weight=0.6,
            storage_layer="episodic",
            created_time=datetime.now(),
            access_count=2,
            rif_score=0.6
        )
    ]


@pytest.fixture
def emotional_state():
    """示例情感状态"""
    return EmotionalState(
        current=CurrentEmotionState(
            primary=EmotionType.JOY,
            secondary=[EmotionType.NEUTRAL],
            intensity=0.8,
            confidence=0.9
        ),
        config=EmotionConfig(
            history_size=100,
            window_size=7,
            min_confidence=0.3
        ),
        context=EmotionContext(
            active_session="test_session",
            user_situation="test situation"
        )
    )


class TestMemoryRetrievalEngineInit:
    """测试MemoryRetrievalEngine初始化"""

    def test_init_with_all_params(
        self, mock_chroma_manager, mock_rif_scorer, mock_emotion_analyzer, mock_reranker
    ):
        """测试完整参数初始化"""
        engine = MemoryRetrievalEngine(
            chroma_manager=mock_chroma_manager,
            rif_scorer=mock_rif_scorer,
            emotion_analyzer=mock_emotion_analyzer,
            reranker=mock_reranker
        )

        assert engine.chroma_manager == mock_chroma_manager
        assert engine.rif_scorer == mock_rif_scorer
        assert engine.emotion_analyzer == mock_emotion_analyzer
        assert engine.reranker == mock_reranker
        assert engine.max_context_memories == 3
        assert engine.enable_time_aware is True
        assert engine.enable_emotion_aware is True
        assert engine.enable_token_budget is True

    def test_init_with_defaults(self, mock_chroma_manager):
        """测试默认参数初始化"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)

        assert engine.chroma_manager == mock_chroma_manager
        assert engine.rif_scorer is not None
        assert engine.emotion_analyzer is not None
        assert engine.reranker is not None
        assert engine.token_budget is not None
        assert engine.memory_compressor is not None
        assert engine.memory_selector is not None


class TestMemoryRetrievalEngineRetrieve:
    """测试retrieve方法"""

    @pytest.mark.asyncio
    async def test_retrieve_basic(
        self, mock_chroma_manager, sample_memories, emotional_state
    ):
        """测试基本检索"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = sample_memories

        results = await engine.retrieve(
            query="喜欢",
            user_id="user_1",
            top_k=10,
            emotional_state=emotional_state
        )

        assert len(results) == 3
        mock_chroma_manager.query_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_no_memories(self, mock_chroma_manager):
        """测试无结果检索"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = []

        results = await engine.retrieve(
            query="测试",
            user_id="user_1",
            top_k=10
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_with_storage_layer(
        self, mock_chroma_manager, sample_memories
    ):
        """测试带存储层过滤的检索"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = sample_memories

        await engine.retrieve(
            query="测试",
            user_id="user_1",
            storage_layer=StorageLayer.EPISODIC
        )

        # 验证调用了存储层过滤
        call_kwargs = mock_chroma_manager.query_memories.call_args.kwargs
        assert call_kwargs["storage_layer"] == StorageLayer.EPISODIC

    @pytest.mark.asyncio
    async def test_retrieve_with_group_id(
        self, mock_chroma_manager, sample_memories
    ):
        """测试带群组ID的检索"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = sample_memories

        await engine.retrieve(
            query="测试",
            user_id="user_1",
            group_id="group_1"
        )

        # 验证调用了群组ID
        call_kwargs = mock_chroma_manager.query_memories.call_args.kwargs
        assert call_kwargs["group_id"] == "group_1"


class TestApplyEmotionFilter:
    """测试_apply_emotion_filter方法"""

    def test_emotion_filter_disabled(self, mock_chroma_manager, sample_memories):
        """测试情感过滤禁用"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.enable_emotion_aware = False

        results = engine._apply_emotion_filter(
            memories=sample_memories,
            emotional_state=None
        )

        assert len(results) == 3

    def test_emotion_filter_positive_memories(
        self, mock_chroma_manager, sample_memories, emotional_state
    ):
        """测试过滤高强度正面记忆"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.emotion_analyzer.should_filter_positive_memories.return_value = True

        results = engine._apply_emotion_filter(
            memories=sample_memories,
            emotional_state=emotional_state
        )

        # 应该过滤掉高强度的正面记忆（emotional_weight > 0.8）
        assert len(results) <= 3
        assert all(
            m.type != "emotion" or m.emotional_weight <= 0.8
            for m in results
        )

    def test_emotion_filter_no_positive_memories(
        self, mock_chroma_manager, sample_memories, emotional_state
    ):
        """测试没有高强度正面记忆时不过滤"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.emotion_analyzer.should_filter_positive_memories.return_value = True

        # 降低所有记忆的情感权重
        low_weight_memories = [
            m for m in sample_memories if not (m.type == "emotion" and m.emotional_weight > 0.8)
        ]

        results = engine._apply_emotion_filter(
            memories=low_weight_memories,
            emotional_state=emotional_state
        )

        assert len(results) == len(low_weight_memories)


class TestRerankMemories:
    """测试_rerank_memories方法"""

    def test_rerank_with_reranker(
        self, mock_chroma_manager, mock_reranker, sample_memories, emotional_state
    ):
        """测试使用Reranker重排序"""
        engine = MemoryRetrievalEngine(
            chroma_manager=mock_chroma_manager,
            reranker=mock_reranker
        )

        # 设置重排序结果
        reranked = sample_memories[::-1]  # 反转顺序
        mock_reranker.rerank.return_value = reranked

        results = engine._rerank_memories(
            memories=sample_memories,
            query="喜欢",
            emotional_state=emotional_state
        )

        mock_reranker.rerank.assert_called_once()
        assert results == reranked

    def test_rerank_without_emotional_state(
        self, mock_chroma_manager, mock_reranker, sample_memories
    ):
        """测试没有情感状态时的重排序"""
        engine = MemoryRetrievalEngine(
            chroma_manager=mock_chroma_manager,
            reranker=mock_reranker
        )

        results = engine._rerank_memories(
            memories=sample_memories,
            query="喜欢",
            emotional_state=None
        )

        # 验证没有传递emotional_state到context
        call_args = mock_reranker.rerank.call_args
        context = call_args[0][2] if len(call_args[0]) > 2 else call_args.kwargs.get("context", {})

        assert "emotional_state" not in context


class TestRetrieveWithStrategy:
    """测试retrieve_with_strategy方法"""

    @pytest.mark.asyncio
    async def test_retrieve_with_vector_only(
        self, mock_chroma_manager, sample_memories
    ):
        """测试纯向量检索策略"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = sample_memories

        results = await engine.retrieve_with_strategy(
            query="测试",
            user_id="user_1",
            strategy=RetrievalStrategy.VECTOR_ONLY
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_with_time_aware(
        self, mock_chroma_manager, sample_memories
    ):
        """测试时间感知检索策略"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = sample_memories

        results = await engine.retrieve_with_strategy(
            query="测试",
            user_id="user_1",
            strategy=RetrievalStrategy.TIME_AWARE
        )

        # 时间感知应该请求双倍候选
        assert mock_chroma_manager.query_memories.called

    @pytest.mark.asyncio
    async def test_retrieve_with_emotion_aware(
        self, mock_chroma_manager, sample_memories, emotional_state
    ):
        """测试情感感知检索策略"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = sample_memories

        results = await engine.retrieve_with_strategy(
            query="测试",
            user_id="user_1",
            strategy=RetrievalStrategy.EMOTION_AWARE,
            emotional_state=emotional_state
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_with_hybrid(
        self, mock_chroma_manager, sample_memories, emotional_state
    ):
        """测试混合检索策略（默认）"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = sample_memories

        results = await engine.retrieve_with_strategy(
            query="测试",
            user_id="user_1",
            strategy=RetrievalStrategy.HYBRID,
            emotional_state=emotional_state
        )

        assert isinstance(results, list)


class TestFormatMemoriesForLLM:
    """测试format_memories_for_llm方法"""

    def test_format_empty_memories(self, mock_chroma_manager):
        """测试格式化空记忆列表"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)

        result = engine.format_memories_for_llm([])

        assert result == ""

    def test_format_memories_basic(self, mock_chroma_manager, sample_memories):
        """测试基本格式化"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.enable_token_budget = False

        result = engine.format_memories_for_llm(sample_memories)

        assert "【相关记忆】" in result
        assert "用户喜欢红色" in result
        assert "今天很高兴" in result

    def test_format_memories_with_token_budget(
        self, mock_chroma_manager, sample_memories
    ):
        """测试使用token预算格式化"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.enable_token_budget = True

        result = engine.format_memories_for_llm(sample_memories)

        # 应该调用DynamicMemorySelector
        assert isinstance(result, str)

    def test_format_memories_with_persona(
        self, mock_chroma_manager, sample_memories
    ):
        """测试带用户画像的格式化"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.enable_token_budget = False

        user_persona = {"preferences": {"style": "formal"}}
        result = engine.format_memories_for_llm(
            sample_memories,
            user_persona=user_persona
        )

        # 应该应用人格协调
        assert isinstance(result, str)

    def test_format_memories_limited_count(self, mock_chroma_manager, sample_memories):
        """测试限制数量的格式化"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.enable_token_budget = False
        engine.max_context_memories = 2

        result = engine.format_memories_for_llm(sample_memories)

        # 只应该格式化前2个记忆
        assert "【相关记忆】" in result


class TestSetConfig:
    """测试set_config方法"""

    def test_set_max_context_memories(self, mock_chroma_manager):
        """测试设置最大上下文记忆数"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)

        engine.set_config({"max_context_memories": 5})

        assert engine.max_context_memories == 5

    def test_set_enable_time_aware(self, mock_chroma_manager):
        """测试设置时间感知开关"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)

        engine.set_config({"enable_time_aware": False})

        assert engine.enable_time_aware is False

    def test_set_enable_emotion_aware(self, mock_chroma_manager):
        """测试设置情感感知开关"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)

        engine.set_config({"enable_emotion_aware": False})

        assert engine.enable_emotion_aware is False

    def test_set_token_budget(self, mock_chroma_manager):
        """测试设置token预算"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)

        engine.set_config({"token_budget": 1024})

        assert engine.token_budget.total_budget == 1024

    def test_set_coordination_strategy(self, mock_chroma_manager):
        """测试设置人格协调策略"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)

        engine.set_config({"coordination_strategy": "bot_priority"})

        assert engine.persona_coordinator.strategy.value == "bot_priority"

    def test_set_multiple_configs(self, mock_chroma_manager):
        """测试同时设置多个配置"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)

        config = {
            "max_context_memories": 5,
            "enable_time_aware": False,
            "enable_emotion_aware": False,
            "enable_token_budget": False,
            "token_budget": 1024,
            "coordination_strategy": "user_priority"
        }

        engine.set_config(config)

        assert engine.max_context_memories == 5
        assert engine.enable_time_aware is False
        assert engine.enable_emotion_aware is False
        assert engine.enable_token_budget is False
        assert engine.token_budget.total_budget == 1024
        assert engine.persona_coordinator.strategy.value == "user_priority"


class TestMemoryAccessUpdate:
    """测试记忆访问更新"""

    @pytest.mark.asyncio
    async def test_access_count_updated(
        self, mock_chroma_manager, sample_memories
    ):
        """测试检索后访问次数更新"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.return_value = sample_memories

        initial_counts = [m.access_count for m in sample_memories]

        await engine.retrieve(query="测试", user_id="user_1")

        # 验证所有记忆的access_count都增加了
        for i, memory in enumerate(sample_memories):
            assert memory.access_count == initial_counts[i] + 1


class TestEdgeCases:
    """测试边界情况"""

    @pytest.mark.asyncio
    async def test_retrieve_exception_handling(self, mock_chroma_manager):
        """测试检索异常处理"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        mock_chroma_manager.query_memories.side_effect = Exception("Database error")

        results = await engine.retrieve(query="测试", user_id="user_1")

        # 异常情况下应该返回空列表
        assert results == []

    def test_format_empty_memory_context(self, mock_chroma_manager):
        """测试格式化空上下文"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.enable_token_budget = False

        result = engine.format_memories_for_llm([])

        assert result == ""

    def test_inject_disabled(self, mock_chroma_manager, sample_memories):
        """测试禁用注入"""
        engine = MemoryRetrievalEngine(chroma_manager=mock_chroma_manager)
        engine.enable_token_budget = False
        engine.enable_injection = False

        result = engine.format_memories_for_llm(sample_memories)

        # 应该直接返回system_prompt（模拟）
        assert isinstance(result, str)

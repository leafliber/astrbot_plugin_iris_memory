"""
Reranker单元测试
测试Reranker模块，验证权重合并后的正确性
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from iris_memory.retrieval.reranker import Reranker
from iris_memory.models.memory import Memory
from iris_memory.core.types import QualityLevel, EmotionType, MemoryType
from iris_memory.models.emotion_state import EmotionalState


class TestReranker:
    """Reranker单元测试"""

    @pytest.fixture
    def reranker(self):
        """创建Reranker实例"""
        return Reranker(enable_vector_score=True)

    @pytest.fixture
    def sample_memories(self):
        """创建测试记忆列表"""
        now = datetime.now()

        memories = [
            # 高质量、高RIF、新记忆
            Memory(
                id="mem1",
                content="我喜欢吃披萨",
                user_id="user1",
                group_id="group1",
                type=MemoryType.FACT,
                quality_level=QualityLevel.CONFIRMED,
                rif_score=0.9,
                access_count=15,
                last_access_time=now,
                created_time=now - timedelta(days=1)
            ),
            # 中等质量、中等RIF、旧记忆
            Memory(
                id="mem2",
                content="我有两只猫",
                user_id="user1",
                group_id="group1",
                type=MemoryType.FACT,
                quality_level=QualityLevel.MODERATE,
                rif_score=0.6,
                access_count=5,
                last_access_time=now - timedelta(days=30),
                created_time=now - timedelta(days=35)
            ),
            # 低质量、低RIF、很旧的记忆
            Memory(
                id="mem3",
                content="我喜欢蓝色",
                user_id="user1",
                group_id="group1",
                type=MemoryType.FACT,
                quality_level=QualityLevel.UNCERTAIN,
                rif_score=0.3,
                access_count=1,
                last_access_time=now - timedelta(days=100),
                created_time=now - timedelta(days=110)
            ),
            # 高质量、情感类记忆
            Memory(
                id="mem4",
                content="我感到很开心",
                user_id="user1",
                group_id="group1",
                type=MemoryType.EMOTION,
                subtype="joy",
                quality_level=QualityLevel.CONFIRMED,
                rif_score=0.85,
                emotional_weight=0.9,
                access_count=8,
                last_access_time=now,
                created_time=now - timedelta(days=2)
            ),
        ]

        # 添加vector_similarity属性（如果需要测试）
        for mem in memories:
            mem.vector_similarity = mem.rif_score

        return memories

    def test_reranker_initialization(self, reranker):
        """测试Reranker初始化"""
        assert reranker is not None
        assert reranker.enable_vector_score is True
        assert QualityLevel.CONFIRMED in reranker.quality_weights
        assert reranker.quality_weights[QualityLevel.CONFIRMED] == 1.5

    def test_reranker_empty_list(self, reranker):
        """测试空列表"""
        result = reranker.rerank([])
        assert result == []

    def test_reranker_basic(self, reranker, sample_memories):
        """测试基本重排序"""
        result = reranker.rerank(sample_memories)

        assert len(result) == 4
        assert result[0].id == "mem1"  # 应该是最高的（高质量+高RIF+新记忆）
        assert result[-1].id == "mem3"  # 应该是最低的（低质量+低RIF+旧记忆）

    def test_weight_distribution(self, reranker, sample_memories):
        """测试权重分配是否正确"""
        # 使用第一个记忆（fact类型）
        memory = sample_memories[0]

        # 计算各个分数
        quality_score = reranker.quality_weights.get(memory.quality_level, 1.0)
        rif_score = memory.rif_score
        time_score = reranker._calculate_time_score(memory)
        access_score = min(1.0, memory.access_count / 10.0)

        # 计算情感得分（fact类型，默认0.5）
        emotion_score = reranker._calculate_emotion_score(memory, {})

        # 计算发送者匹配得分（无context，默认0.5）
        sender_score = reranker._calculate_sender_score(memory, {})

        # 计算活跃度得分（无service，默认0.5）
        activity_score = reranker._calculate_activity_score(memory, {})

        # 计算期望的综合得分（新权重分配：RIF已含时近性，time_score仅微调）
        expected_score = (
            0.25 * quality_score +  # 质量等级
            0.25 * rif_score +     # RIF评分（内含时近性40%+相关性30%+频率30%）
            0.05 * time_score +    # 时间衰减（微调补充）
            0.10 * sender_score +  # 发送者匹配
            0.05 * activity_score + # 活跃度
            0.10 * access_score +   # 访问频率
            0.05 * emotion_score   # 情感一致性
        )

        # 加上向量相似度
        expected_score += 0.15 * memory.vector_similarity

        # 计算实际得分
        actual_score = reranker._calculate_rerank_score(memory, "query", {})

        # 验证得分一致（考虑浮点数误差）
        assert abs(actual_score - expected_score) < 0.01

    def test_quality_weights(self, reranker):
        """测试质量等级权重"""
        weights = reranker.quality_weights
        assert weights[QualityLevel.CONFIRMED] == 1.5
        assert weights[QualityLevel.HIGH_CONFIDENCE] == 1.3
        assert weights[QualityLevel.MODERATE] == 1.0
        assert weights[QualityLevel.LOW_CONFIDENCE] == 0.7
        assert weights[QualityLevel.UNCERTAIN] == 0.4

    def test_time_score(self, reranker):
        """测试时间得分计算"""
        now = datetime.now()

        # 测试新记忆（1天内）— calculate_time_score 返回 1.0
        new_mem = Mock()
        new_mem.last_access_time = now
        new_mem.calculate_time_score = Mock(return_value=1.0)
        score = reranker._calculate_time_score(new_mem)
        assert score == 1.0

        # 测试旧记忆（100天前）
        old_mem = Mock()
        old_mem.last_access_time = now - timedelta(days=100)
        old_mem.calculate_time_score = Mock(return_value=0.5)
        score = reranker._calculate_time_score(old_mem)
        assert abs(score - 0.5) < 0.01

    def test_emotion_score_no_context(self, reranker, sample_memories):
        """测试无情感上下文时的得分"""
        memory = sample_memories[3]  # 情感类记忆
        score = reranker._calculate_emotion_score(memory, {})
        assert score == 0.5  # 默认中等得分

    def test_emotion_score_consistent(self, reranker, sample_memories):
        """测试情感一致时的得分"""
        memory = sample_memories[3]  # joy情感

        # 创建情感上下文
        emotional_state = Mock()
        emotional_state.current = Mock()
        # 直接使用枚举类型，它的value属性会自动返回"joy"
        emotional_state.current.primary = EmotionType.JOY

        context = {"emotional_state": emotional_state}

        score = reranker._calculate_emotion_score(memory, context)
        assert score == 1.0  # 完全一致

    def test_emotion_score_similar(self, reranker, sample_memories):
        """测试情感相似时的得分"""
        # 创建excitement情感记忆
        memory = Memory(
            id="mem5",
            content="我很兴奋",
            user_id="user1",
            group_id="group1",
            type=MemoryType.EMOTION,
            subtype="excitement",
            quality_level=QualityLevel.HIGH_CONFIDENCE,
            rif_score=0.7
        )

        # 创建calm情感上下文
        emotional_state = Mock()
        emotional_state.current = Mock()
        # 直接使用枚举类型
        emotional_state.current.primary = EmotionType.CALM

        context = {"emotional_state": emotional_state}

        score = reranker._calculate_emotion_score(memory, context)
        assert score == 0.7  # 相似情感（joy/excitement/calm在同一组）

    def test_emotion_score_negative_context(self, reranker, sample_memories):
        """测试负面情感时高强度正面记忆的得分"""
        memory = sample_memories[3]  # joy情感，高权重(0.9)

        # 创建负面情感上下文（悲伤）
        emotional_state = Mock()
        emotional_state.current = Mock()
        emotional_state.current.primary = EmotionType.SADNESS

        context = {"emotional_state": emotional_state}

        score = reranker._calculate_emotion_score(memory, context)
        assert score == 0.0  # 负面情感时，高强度正面记忆相关性为0

    def test_filter_by_quality(self, reranker, sample_memories):
        """测试按质量过滤"""
        result = reranker.filter_by_quality(sample_memories, QualityLevel.MODERATE)

        # 应该包含MODERATE及以上（MODERATE=3, CONFIRMED=5）
        assert len(result) == 3  # mem1(CONFIRMED), mem2(MODERATE), mem4(CONFIRMED)
        assert all(m.quality_level.value >= QualityLevel.MODERATE.value for m in result)

    def test_filter_by_storage_layer(self, reranker, sample_memories):
        """测试按存储层过滤"""
        # 设置存储层
        sample_memories[0].storage_layer = "working"
        sample_memories[1].storage_layer = "episodic"
        sample_memories[2].storage_layer = "episodic"
        sample_memories[3].storage_layer = "semantic"

        result = reranker.filter_by_storage_layer(sample_memories, "episodic")

        assert len(result) == 2
        assert all(m.storage_layer == "episodic" for m in result)

    def test_group_by_type(self, reranker, sample_memories):
        """测试按类型分组"""
        grouped = reranker.group_by_type(sample_memories)

        assert "fact" in grouped
        assert "emotion" in grouped
        assert len(grouped["fact"]) == 3
        assert len(grouped["emotion"]) == 1

    def test_deduplicate(self, reranker):
        """测试去重"""
        memories = [
            Memory(
                id="mem1",
                content="我喜欢吃披萨",
                user_id="user1",
                group_id="group1",
                type=MemoryType.FACT,
                quality_level=QualityLevel.CONFIRMED,
                rif_score=0.9,
                access_count=10,
                last_access_time=datetime.now()
            ),
            Memory(
                id="mem2",
                content="我喜欢吃披萨",  # 相同内容
                user_id="user1",
                group_id="group1",
                type=MemoryType.FACT,
                quality_level=QualityLevel.MODERATE,
                rif_score=0.8,
                access_count=5,
                last_access_time=datetime.now()
            ),
            Memory(
                id="mem3",
                content="我不喜欢吃披萨",  # 相似但不同
                user_id="user1",
                group_id="group1",
                type=MemoryType.FACT,
                quality_level=QualityLevel.MODERATE,
                rif_score=0.7,
                access_count=5,
                last_access_time=datetime.now()
            ),
        ]

        result = reranker.deduplicate(memories, similarity_threshold=0.8)

        # 应该去重，保留第一个
        assert len(result) == 2
        assert result[0].id == "mem1"
        assert result[1].id == "mem3"

    def test_calculate_similarity(self, reranker):
        """测试文本相似度计算"""
        # 完全相同
        sim = reranker._calculate_similarity("test text", "test text")
        assert sim == 1.0

        # 部分相同
        sim = reranker._calculate_similarity("hello world", "hello there")
        assert 0 < sim < 1

        # 完全不同
        sim = reranker._calculate_similarity("hello", "goodbye")
        assert sim < 0.5

    def test_vector_score_disabled(self, sample_memories):
        """测试禁用向量相似度的情况"""
        reranker = Reranker(enable_vector_score=False)

        # 移除vector_similarity属性
        for mem in sample_memories:
            if hasattr(mem, 'vector_similarity'):
                delattr(mem, 'vector_similarity')

        # 应该能正常工作，不报错
        result = reranker.rerank(sample_memories)
        assert len(result) == 4

    def test_integration_with_retrieval_engine(self):
        """测试与RetrievalEngine的集成"""
        from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine

        # 创建mock chroma_manager
        chroma_manager = Mock()

        # 创建检索引擎（使用默认reranker）
        engine = MemoryRetrievalEngine(chroma_manager)

        # 验证reranker已初始化
        assert engine.reranker is not None
        assert isinstance(engine.reranker, Reranker)
        assert engine.reranker.enable_vector_score is True

    def test_custom_reranker_in_retrieval_engine(self):
        """测试在RetrievalEngine中使用自定义Reranker"""
        from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine

        chroma_manager = Mock()
        custom_reranker = Reranker(enable_vector_score=False)

        # 创建检索引擎（使用自定义reranker）
        engine = MemoryRetrievalEngine(
            chroma_manager=chroma_manager,
            reranker=custom_reranker
        )

        # 验证使用了自定义reranker
        assert engine.reranker is custom_reranker
        assert engine.reranker.enable_vector_score is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

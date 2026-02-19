"""RIFScorer测试
测试RIF（Recency, Relevance, Frequency）评分系统的核心功能
"""

import pytest
from datetime import datetime, timedelta
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer


@pytest.fixture
def rif_scorer():
    """RIFScorer实例"""
    return RIFScorer()


@pytest.fixture
def sample_memory():
    """创建示例记忆对象"""
    return Memory(
        id="mem_001",
        type=MemoryType.FACT,
        content="用户喜欢编程",
        user_id="user_123",
        group_id=None,
        quality_level=0.8,
        storage_layer=StorageLayer.EPISODIC,
        created_time=datetime.now() - timedelta(days=1),
        last_access_time=datetime.now()
    )


class TestRIFScorerInit:
    """测试初始化功能"""
    
    def test_init_basic(self, rif_scorer):
        """测试基本初始化"""
        assert rif_scorer.recency_weight == 0.4
        assert rif_scorer.relevance_weight == 0.3
        assert rif_scorer.frequency_weight == 0.3
        assert 'new' in rif_scorer.time_weights
        assert 'medium' in rif_scorer.time_weights
        assert 'old' in rif_scorer.time_weights
        assert 'very_old' in rif_scorer.time_weights


class TestRIFScorerRecencyScore:
    """测试时近性评分"""
    
    def test_calculate_recency_recent(self, rif_scorer, sample_memory):
        """测试近期记忆的时近性评分"""
        sample_memory.last_access_time = datetime.now() - timedelta(hours=1)
        score = rif_scorer._calculate_recency(sample_memory)
        assert 0.0 <= score <= 1.0


class TestRIFScorerCalculateRIF:
    """测试RIF综合评分"""
    
    def test_calculate_rif_basic(self, rif_scorer, sample_memory):
        """测试基本RIF评分"""
        sample_memory.consistency_score = 0.5
        sample_memory.access_frequency = 0.3
        sample_memory.importance_score = 0.5
        score = rif_scorer.calculate_rif(sample_memory)
        assert 0.0 <= score <= 1.0
        assert hasattr(sample_memory, 'rif_score')


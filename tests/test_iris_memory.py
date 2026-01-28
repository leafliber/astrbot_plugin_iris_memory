"""
Iris Memory Plugin 单元测试
测试所有核心模块的功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import numpy as np

# 导入所有需要测试的模块
from iris_memory.models.memory import Memory
from iris_memory.models.emotion_state import CurrentEmotionState, EmotionalState
from iris_memory.core.types import (
    MemoryType,
    ModalityType,
    QualityLevel,
    SensitivityLevel,
    StorageLayer,
    VerificationMethod,
    DecayRate,
    EmotionType
)
from iris_memory.analysis.entity_extractor import EntityExtractor, EntityType, Entity
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.storage.cache import (
    LRUCache,
    LFUCache,
    EmbeddingCache,
    WorkingMemoryCache,
    MemoryCompressor,
    CacheStrategy
)
from iris_memory.storage.session_manager import SessionManager


# ========== Fixtures ==========

@pytest.fixture
def sample_config():
    """提供测试用的配置对象"""
    class MockConfig:
        def __init__(self):
            self.config = {
                'embedding_cache': {
                    'max_size': 100,
                    'strategy': 'lru'
                },
                'working_cache': {
                    'max_sessions': 10,
                    'max_memories_per_session': 20,
                    'ttl': 3600
                },
                'compression': {
                    'max_length': 200
                }
                }
        
        def get(self, key, default=None):
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value.get(k, default) if isinstance(value, dict) else default
            return value if value is not None else default
    
    return MockConfig()


@pytest.fixture
def sample_memory():
    """提供测试用的Memory对象"""
    return Memory(
        id="test_memory_001",
        user_id="user123",
        group_id="group456",
        type=MemoryType.FACT,
        modality=ModalityType.TEXT,
        content="用户说喜欢吃苹果",
        quality_level=QualityLevel.MODERATE,
        sensitivity_level=SensitivityLevel.PUBLIC,
        storage_layer=StorageLayer.EPISODIC,
        importance_score=0.7,
        rif_score=0.6
    )


@pytest.fixture
def sample_emotion_state():
    """提供测试用的CurrentEmotionState对象"""
    return CurrentEmotionState(
        primary=EmotionType.JOY,
        intensity=0.8,
        confidence=0.6
    )


# ========== Memory模型测试 ==========

def test_memory_creation(sample_memory):
    """测试Memory对象创建"""
    assert sample_memory.id == "test_memory_001"
    assert sample_memory.user_id == "user123"
    assert sample_memory.type == MemoryType.FACT
    assert sample_memory.content == "用户说喜欢吃苹果"
    assert sample_memory.importance_score == 0.7


def test_memory_serialization(sample_memory):
    """测试Memory对象的序列化和反序列化"""
    # 序列化
    data = sample_memory.to_dict()
    
    # 验证序列化结果
    assert 'id' in data
    assert 'user_id' in data
    assert 'content' in data
    assert 'created_time' in data
    
    # 反序列化
    restored = Memory.from_dict(data)
    
    # 验证反序列化结果
    assert restored.id == sample_memory.id
    assert restored.user_id == sample_memory.user_id
    assert restored.content == sample_memory.content
    assert restored.type == sample_memory.type


def test_memory_access_update(sample_memory):
    """测试访问统计更新"""
    initial_count = sample_memory.access_count
    initial_time = sample_memory.last_access_time
    
    # 稍等一会儿确保时间戳不同
    import time
    time.sleep(0.01)
    
    # 更新访问
    sample_memory.update_access()
    
    # 验证
    assert sample_memory.access_count == initial_count + 1
    assert sample_memory.last_access_time > initial_time


def test_memory_upgrade_criteria():
    """测试记忆升级条件判断"""
    # 测试工作记忆升级到情景记忆
    working_memory = Memory(
        storage_layer=StorageLayer.WORKING,
        access_count=3,
        importance_score=0.7
    )
    assert working_memory.should_upgrade_to_episodic() is True
    
    # 测试不满足升级条件
    working_memory2 = Memory(
        storage_layer=StorageLayer.WORKING,
        access_count=2,
        importance_score=0.5
    )
    assert working_memory2.should_upgrade_to_episodic() is False
    
    # 测试情景记忆升级到语义记忆
    episodic_memory = Memory(
        storage_layer=StorageLayer.EPISODIC,
        access_count=10,
        confidence=0.85
    )
    assert episodic_memory.should_upgrade_to_semantic() is True


def test_memory_time_weight():
    """测试时间权重计算"""
    # 创建不同时间的记忆
    recent_memory = Memory()
    old_memory = Memory()
    old_memory.last_access_time = datetime.now() - timedelta(days=100)
    
    # 计算时间权重
    recent_weight = recent_memory.calculate_time_weight()
    old_weight = old_memory.calculate_time_weight()
    
    # 验证近期记忆权重更高
    assert recent_weight > old_weight


# ========== Emotion模型测试 ==========

def test_emotion_state_creation(sample_emotion_state):
    """测试CurrentEmotionState对象创建"""
    assert sample_emotion_state.primary == EmotionType.JOY
    assert sample_emotion_state.intensity == 0.8
    assert sample_emotion_state.confidence == 0.6
    assert 0 <= sample_emotion_state.intensity <= 1
    assert 0 <= sample_emotion_state.confidence <= 1


def test_emotional_state_creation():
    """测试EmotionalState对象创建"""
    emotion_state = EmotionalState()
    
    # 验证初始状态
    assert emotion_state.current.primary == EmotionType.NEUTRAL
    assert len(emotion_state.history) == 0
    assert isinstance(emotion_state.trajectory, object)
    assert isinstance(emotion_state.patterns, dict)


def test_emotional_state_update():
    """测试EmotionalState更新"""
    emotion_state = EmotionalState()
    
    # 更新情感状态
    emotion_state.update_current_emotion(
        primary=EmotionType.JOY,
        intensity=0.9,
        confidence=0.8
    )
    
    # 验证更新
    assert emotion_state.current.primary == EmotionType.JOY
    assert emotion_state.current.intensity == 0.9
    assert len(emotion_state.history) == 1  # 之前的状态被保存到历史


# ========== 实体提取器测试 ==========

def test_entity_extractor_initialization():
    """测试实体提取器初始化"""
    extractor = EntityExtractor()
    assert extractor is not None
    assert extractor.reference_date is not None


def test_extract_person_entities():
    """测试提取人名实体"""
    extractor = EntityExtractor()
    text = "明天我要和小王、张三去上海开会，还会见到Alice Johnson"
    
    entities = extractor.extract_entities(text)
    person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
    
    # 验证提取到人名
    assert len(person_entities) > 0
    person_texts = [e.text for e in person_entities]
    assert any('小王' in text for text in person_texts) or any('张三' in text for text in person_texts)


def test_extract_location_entities():
    """测试提取地点实体"""
    extractor = EntityExtractor()
    text = "下周三在北京的清华大学图书馆见面"
    
    entities = extractor.extract_entities(text)
    location_entities = [e for e in entities if e.entity_type == EntityType.LOCATION]
    
    # 验证提取到地点
    assert len(location_entities) > 0
    location_texts = [e.text for e in location_entities]
    assert any('北京' in text for text in location_texts)


def test_extract_time_entities():
    """测试提取时间实体"""
    extractor = EntityExtractor()
    text = "明天下午3点在上海见面，后天早上开会"
    
    entities = extractor.extract_entities(text)
    time_entities = [e for e in entities if e.entity_type == EntityType.TIME]
    
    # 验证提取到时间
    assert len(time_entities) > 0


def test_extract_email_entities():
    """测试提取邮箱实体"""
    extractor = EntityExtractor()
    text = "我的邮箱是test@example.com，备用邮箱是backup@test.org"
    
    entities = extractor.extract_entities(text)
    email_entities = [e for e in entities if e.entity_type == EntityType.EMAIL]
    
    # 验证提取到邮箱
    assert len(email_entities) > 0
    email_texts = [e.text for e in email_entities]
    assert 'test@example.com' in email_texts or 'backup@test.org' in email_texts


def test_extract_url_entities():
    """测试提取URL实体"""
    extractor = EntityExtractor()
    text = "这是我的网站：https://www.example.com，还有这个http://test.org"
    
    entities = extractor.extract_entities(text)
    url_entities = [e for e in entities if e.entity_type == EntityType.URL]
    
    # 验证提取到URL
    assert len(url_entities) > 0
    url_texts = [e.text for e in url_entities]
    assert 'https://www.example.com' in url_texts or 'http://test.org' in url_texts


def test_entity_normalization():
    """测试实体标准化"""
    extractor = EntityExtractor(reference_date=datetime(2025, 1, 27))
    text = "明天下午3点"
    
    entities = extractor.extract_entities(text)
    time_entities = [e for e in entities if e.entity_type == EntityType.TIME]
    
    if time_entities:
        entity = time_entities[0]
        # 验证标准化后的时间
        assert 'normalized_time' in entity.metadata


def test_entity_deduplication():
    """测试实体去重"""
    extractor = EntityExtractor()
    text = "明天下午3点，下午3点开会"
    
    entities = extractor.extract_entities(text)
    
    # 验证没有重复实体
    entity_positions = [(e.start_pos, e.end_pos) for e in entities]
    assert len(entity_positions) == len(set(entity_positions))


def test_get_entities_by_type():
    """测试按类型提取实体"""
    extractor = EntityExtractor()
    text = "明天下午3点和Alice在北京见面"
    
    person_entities = extractor.get_entities_by_type(text, EntityType.PERSON)
    location_entities = extractor.get_entities_by_type(text, EntityType.LOCATION)
    
    # 验证只返回指定类型的实体
    assert all(e.entity_type == EntityType.PERSON for e in person_entities)
    assert all(e.entity_type == EntityType.LOCATION for e in location_entities)


def test_get_entity_summary():
    """测试获取实体摘要"""
    extractor = EntityExtractor()
    text = "明天下午3点和Alice在北京见面，邮箱是test@example.com"
    
    summary = extractor.get_entity_summary(text)
    
    # 验证摘要结构
    assert isinstance(summary, dict)
    # 验证有实体类型分组
    assert len(summary) > 0


# ========== 缓存模块测试 ==========

def test_lru_cache_basic():
    """测试LRU缓存基本功能"""
    cache = LRUCache(max_size=3)
    
    # 测试设置和获取
    assert cache.set('key1', 'value1') is True
    assert cache.get('key1') == 'value1'
    assert cache.get_size() == 1
    
    # 测试未命中
    assert cache.get('nonexistent') is None
    
    # 测试LRU淘汰
    cache.set('key2', 'value2')
    cache.set('key3', 'value3')
    cache.set('key4', 'value4')  # 应该淘汰key1
    
    assert cache.get('key1') is None
    assert cache.get('key2') == 'value2'
    assert cache.get('key3') == 'value3'
    assert cache.get('key4') == 'value4'


def test_lru_cache_stats():
    """测试LRU缓存统计"""
    cache = LRUCache(max_size=3)
    
    cache.set('key1', 'value1')
    cache.set('key2', 'value2')
    
    # 测试命中
    cache.get('key1')
    cache.get('key1')
    cache.get('key2')
    
    # 测试未命中
    cache.get('nonexistent')
    
    stats = cache.get_stats()
    
    # 验证统计
    assert stats.hits == 3
    assert stats.misses == 1
    assert stats.hit_rate == 0.75


def test_lfu_cache_basic():
    """测试LFU缓存基本功能"""
    cache = LFUCache(max_size=3)
    
    cache.set('key1', 'value1')
    cache.set('key2', 'value2')
    
    # 频繁访问key1
    for _ in range(5):
        cache.get('key1')
    
    # 访问key2一次
    cache.get('key2')
    
    # 添加key3
    cache.set('key3', 'value3')
    
    # 添加key4，应该淘汰key2（访问次数最少）
    cache.set('key4', 'value4')
    
    assert cache.get('key1') == 'value1'  # 访问次数最多，保留
    assert cache.get('key2') is None  # 被淘汰
    assert cache.get('key3') == 'value3'
    assert cache.get('key4') == 'value4'


def test_lfu_cache_stats():
    """测试LFU缓存统计"""
    cache = LFUCache(max_size=5)
    
    cache.set('key1', 'value1')
    cache.set('key2', 'value2')
    
    # 访问key1多次
    for _ in range(3):
        cache.get('key1')
    
    # 访问key2一次
    cache.get('key2')
    
    stats = cache.get_stats()
    
    # 验证统计
    assert stats.hits == 4
    assert stats.size == 2


def test_cache_ttl():
    """测试缓存TTL"""
    cache = LRUCache(max_size=10, ttl=1)  # 1秒过期
    
    cache.set('key1', 'value1', ttl=1)
    
    # 立即获取应该成功
    assert cache.get('key1') == 'value1'
    
    # 等待过期
    import time
    time.sleep(1.1)
    
    # 过期后应该返回None
    assert cache.get('key1') is None


def test_embedding_cache():
    """测试嵌入向量缓存"""
    cache = EmbeddingCache(max_size=100, strategy=CacheStrategy.LRU)
    
    text1 = "这是一个测试文本"
    embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # 测试设置和获取
    assert cache.set(text1, embedding1) is True
    assert cache.get(text1) == embedding1
    
    # 测试未命中
    assert cache.get("不存在的文本") is None
    
    # 测试不同文本有不同的哈希
    text2 = "这是另一个测试文本"
    embedding2 = [0.6, 0.7, 0.8, 0.9, 1.0]
    cache.set(text2, embedding2)
    
    assert cache.get(text1) == embedding1
    assert cache.get(text2) == embedding2
    
    # 测试相同文本使用相同缓存
    cache.set(text1, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert cache.get(text1) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_embedding_cache_stats():
    """测试嵌入缓存统计"""
    cache = EmbeddingCache(max_size=50)
    
    text1 = "测试文本1"
    embedding1 = [0.1, 0.2, 0.3]
    
    cache.set(text1, embedding1)
    cache.get(text1)
    cache.get(text1)
    cache.get("不存在的文本")
    
    stats = cache.get_stats()
    
    # 验证统计
    assert stats.hits == 2
    assert stats.misses == 1


@pytest.mark.asyncio
async def test_working_memory_cache():
    """测试工作记忆缓存"""
    cache = WorkingMemoryCache(
        max_sessions=5,
        max_memories_per_session=10,
        ttl=3600
    )
    
    user_id = "user123"
    group_id = "group456"
    
    # 添加记忆
    memory1 = {"id": "mem1", "content": "记忆1"}
    memory2 = {"id": "mem2", "content": "记忆2"}
    
    assert await cache.add_memory(user_id, group_id, "mem1", memory1) is True
    assert await cache.add_memory(user_id, group_id, "mem2", memory2) is True
    
    # 获取记忆
    retrieved = await cache.get_memory(user_id, group_id, "mem1")
    assert retrieved == memory1
    
    # 获取最近记忆
    recent = await cache.get_recent_memories(user_id, group_id, limit=10)
    assert len(recent) == 2


@pytest.mark.asyncio
async def test_working_memory_cache_isolation():
    """测试工作记忆会话隔离"""
    cache = WorkingMemoryCache(max_sessions=10, max_memories_per_session=20)
    
    # 两个不同会话
    user1 = "user1"
    user2 = "user2"
    group = "group1"
    
    memory1 = {"id": "mem1", "content": "用户1的记忆"}
    memory2 = {"id": "mem2", "content": "用户2的记忆"}
    
    # 添加到不同会话
    await cache.add_memory(user1, group, "mem1", memory1)
    await cache.add_memory(user2, group, "mem1", memory2)
    
    # 验证会话隔离
    assert await cache.get_memory(user1, group, "mem1") == memory1
    assert await cache.get_memory(user2, group, "mem1") == memory2


@pytest.mark.asyncio
async def test_working_memory_cache_clear():
    """测试工作记忆缓存清理"""
    cache = WorkingMemoryCache(max_sessions=10, max_memories_per_session=20)
    
    user_id = "user123"
    group_id = "group456"
    
    # 添加记忆
    memory1 = {"id": "mem1", "content": "记忆1"}
    await cache.add_memory(user_id, group_id, "mem1", memory1)
    
    # 验证记忆存在
    assert await cache.get_memory(user_id, group_id, "mem1") == memory1
    
    # 清空会话
    assert await cache.clear_session(user_id, group_id) is True
    
    # 验证记忆已被清除
    assert await cache.get_memory(user_id, group_id, "mem1") is None


@pytest.mark.asyncio
async def test_working_memory_cache_stats():
    """测试工作记忆缓存统计"""
    cache = WorkingMemoryCache(max_sessions=5, max_memories_per_session=10)
    
    # 添加一些记忆
    await cache.add_memory("user1", "group1", "mem1", {"id": "mem1"})
    await cache.add_memory("user1", "group2", "mem2", {"id": "mem2"})
    await cache.add_memory("user2", "group1", "mem3", {"id": "mem3"})
    
    stats = await cache.get_stats()
    
    # 验证统计
    assert stats['total_sessions'] == 3
    assert stats['total_memories'] == 3


def test_memory_compressor():
    """测试记忆压缩器"""
    compressor = MemoryCompressor(max_length=50)

    short_text = "短文本"
    long_text = "这是一个非常长的文本，应该被压缩。这个文本的长度超过了50个字符的限制，所以应该被截断并添加省略号以示压缩成功。"

    # 测试短文本（不需要压缩）
    compressed_short = compressor.compress_memory(short_text)
    assert compressed_short == short_text

    # 测试长文本（需要压缩）
    compressed_long = compressor.compress_memory(long_text)
    assert len(compressed_long) <= 53  # 50 + 3 for "..."
    assert compressed_long.endswith("...")


def test_memory_compressor_keywords():
    """测试关键词提取"""
    compressor = MemoryCompressor()
    
    text = "苹果是一种水果，橙子也是一种水果。水果对健康有益，苹果富含维生素。"
    
    keywords = compressor.extract_keywords(text, top_k=3)
    
    # 验证提取到关键词
    assert len(keywords) <= 3
    assert all(isinstance(kw, str) for kw in keywords)
    # 验证关键词相关
    assert any('苹果' in kw or '橙子' in kw or '水果' in kw for kw in keywords)


# ========== Session管理测试 ==========

def test_session_manager_creation():
    """测试SessionManager创建"""
    manager = SessionManager()
    assert manager is not None
    assert manager.get_session_count() == 0


def test_session_manager_create_session():
    """测试创建会话"""
    manager = SessionManager()
    
    # 创建会话
    key1 = manager.create_session("user1", "group1")
    key2 = manager.create_session("user2", "group2")
    
    # 验证会话键不同
    assert key1 != key2
    
    # 验证会话计数
    assert manager.get_session_count() == 2


def test_session_manager_get_session():
    """测试获取会话"""
    manager = SessionManager()

    # 创建会话
    session_data = {"user": "user1", "group": "group1", "data": "测试数据"}
    key = manager.create_session("user1", "group1", initial_data=session_data)

    # 获取会话
    retrieved = manager.get_session(key)

    # 验证
    assert retrieved is not None
    assert retrieved['data'] == "测试数据"


def test_session_manager_delete_session():
    """测试删除会话"""
    manager = SessionManager()

    # 创建会话
    key1 = manager.create_session("user1", "group1")
    key2 = manager.create_session("user2", "group2")

    # 删除一个会话
    assert manager.delete_session(key1) is True

    # 验证会话计数
    assert manager.get_session_count() == 1

    # 验证会话不存在
    assert manager.get_session(key1) is None
    assert manager.get_session(key2) is not None


# ========== 集成测试 ==========

@pytest.mark.asyncio
async def test_entity_extraction_and_caching():
    """测试实体提取和缓存集成"""
    extractor = EntityExtractor()
    cache = EmbeddingCache(max_size=50)
    
    text = "明天下午3点和Alice在北京见面"
    
    # 提取实体
    entities = extractor.extract_entities(text)
    
    # 缓存实体（模拟）
    cache.set(text, entities)
    
    # 从缓存获取
    cached_entities = cache.get(text)
    
    # 验证
    assert cached_entities == entities


@pytest.mark.asyncio
async def test_complete_workflow():
    """测试完整工作流：创建记忆 -> 添加缓存 -> 检索"""
    # 创建记忆
    memory = Memory(
        user_id="user123",
        group_id="group456",
        content="用户说喜欢吃苹果",
        type=MemoryType.FACT
    )
    
    # 添加到工作记忆缓存
    cache = WorkingMemoryCache(max_sessions=10, max_memories_per_session=20)
    await cache.add_memory(memory.user_id, memory.group_id, memory.id, memory)
    
    # 从缓存检索
    retrieved = await cache.get_memory(memory.user_id, memory.group_id, memory.id)
    
    # 验证
    assert retrieved is not None
    assert retrieved.id == memory.id
    assert retrieved.content == memory.content


# 运行测试的主函数
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

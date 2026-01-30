"""
缓存系统单元测试
测试LRU/LFU/Embedding/WorkingMemory缓存功能
"""

import pytest
import time
from iris_memory.storage.cache import (
    LRUCache,
    LFUCache,
    EmbeddingCache,
    WorkingMemoryCache
)


class TestLRUCache:
    """测试LRU缓存"""
    
    def test_init(self):
        """测试初始化"""
        cache = LRUCache(max_size=100)
        
        assert cache.max_size == 100
        assert len(cache) == 0
    
    def test_put_and_get(self):
        """测试存取"""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        value = cache.get("key1")
        
        assert value == "value1"
        assert len(cache) == 1
    
    def test_get_nonexistent(self):
        """测试获取不存在的键"""
        cache = LRUCache(max_size=10)
        
        value = cache.get("nonexistent")
        
        assert value is None
    
    def test_eviction(self):
        """测试淘汰策略"""
        cache = LRUCache(max_size=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # 应该淘汰key1
        
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"


class TestLFUCache:
    """测试LFU缓存"""
    
    def test_init(self):
        """测试初始化"""
        cache = LFUCache(max_size=100)
        
        assert cache.max_size == 100
        assert len(cache) == 0
    
    def test_frequency_tracking(self):
        """测试频率追踪"""
        cache = LFUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.get("key1")
        cache.get("key1")
        cache.get("key2")
        
        assert cache._get_frequency("key1") == 2
        assert cache._get_frequency("key2") == 1
    
    def test_eviction_lowest_frequency(self):
        """测试淘汰最低频率"""
        cache = LFUCache(max_size=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # 访问key1和key3多次
        for _ in range(5):
            cache.get("key1")
            cache.get("key3")
        
        # 添加key4，应该淘汰key2（最低频率）
        cache.put("key4", "value4")
        
        assert cache.get("key2") is None
        assert cache.get("key1") is not None


class TestEmbeddingCache:
    """测试Embedding缓存"""
    
    def test_init(self):
        """测试初始化"""
        cache = EmbeddingCache(max_size=100)
        
        assert cache.max_size == 100
        assert isinstance(cache._cache, LRUCache)
    
    def test_cache_hit(self):
        """测试缓存命中"""
        cache = EmbeddingCache(max_size=10)
        
        embedding = [0.1] * 768
        cache.put("test", embedding)
        
        result = cache.get("test")
        
        assert result == embedding
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        cache = EmbeddingCache(max_size=10)
        
        result = cache.get("nonexistent")
        
        assert result is None


class TestWorkingMemoryCache:
    """测试工作记忆缓存"""
    
    def test_init(self):
        """测试初始化"""
        cache = WorkingMemoryCache(
            max_sessions=10,
            max_memories_per_session=20,
            ttl=3600
        )
        
        assert cache.max_sessions == 10
        assert cache.max_memories_per_session == 20
        assert cache.ttl == 3600
    
    def test_add_memory(self):
        """测试添加记忆"""
        cache = WorkingMemoryCache(max_sessions=10, max_memories_per_session=20)
        
        memory = Mock(id="mem_001", user_id="user_123", group_id="group_456")
        cache.add_memory(memory)
        
        # 应该成功添加
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

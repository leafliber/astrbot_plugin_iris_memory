"""配置管理器测试 - cache_ttl 可配置"""

import time
import pytest
from iris_memory.core.config_manager import ConfigManager


class TestConfigManagerCacheTTL:
    """ConfigManager cache_ttl 参数化测试"""

    def test_default_cache_ttl(self):
        """默认 TTL 为 30 秒"""
        mgr = ConfigManager()
        assert mgr._cache_ttl == ConfigManager.DEFAULT_CACHE_TTL

    def test_custom_cache_ttl(self):
        """可通过构造参数自定义 TTL"""
        mgr = ConfigManager(cache_ttl=60.0)
        assert mgr._cache_ttl == 60.0

    def test_zero_cache_ttl_always_refetch(self):
        """TTL=0 时每次都重新获取"""
        mgr = ConfigManager(cache_ttl=0.0)
        # 设一个获取值
        val = mgr.get("nonexistent.key", "default_val")
        assert val == "default_val"

    def test_cache_invalidation(self):
        """手动失效缓存"""
        mgr = ConfigManager(cache_ttl=3600.0)
        mgr.get("test.key", "val1")
        assert "test.key" in mgr._cache
        mgr.invalidate_cache("test.key")
        assert "test.key" not in mgr._cache

    def test_cache_invalidation_all(self):
        """失效所有缓存"""
        mgr = ConfigManager(cache_ttl=3600.0)
        mgr.get("key1", "v1")
        mgr.get("key2", "v2")
        mgr.invalidate_cache()
        assert len(mgr._cache) == 0

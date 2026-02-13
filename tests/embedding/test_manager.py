"""
嵌入管理器测试
"""

import asyncio
import pytest
from pathlib import Path

from iris_memory.embedding.manager import EmbeddingManager, EmbeddingStrategy
from iris_memory.embedding.base import EmbeddingRequest


class MockConfig:
    """模拟配置对象"""
    def __init__(self):
        self._data = {
            "embedding": {
                "embedding_strategy": "auto",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dimension": 1024,
                "auto_detect_dimension": True
            }
        }

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'MockConfig' object has no attribute '{name}'")


@pytest.mark.asyncio
async def test_embedding_manager_initialization():
    """测试嵌入管理器初始化"""
    config = MockConfig()
    manager = EmbeddingManager(config)
    
    result = await manager.initialize()
    
    # 应该至少有一个提供者可用（至少降级）
    assert result is True
    assert manager.current_provider is not None
    assert manager.get_dimension() > 0


@pytest.mark.asyncio
async def test_fallback_provider():
    """测试降级提供者"""
    config = MockConfig()
    manager = EmbeddingManager(config)
    
    # 切换到降级策略
    await manager.switch_strategy(EmbeddingStrategy.FALLBACK)
    
    # 生成嵌入
    text = "测试文本"
    embedding = await manager.embed(text)
    
    assert len(embedding) > 0
    assert all(isinstance(x, (int, float)) for x in embedding)
    # 同样的文本应该生成相同的嵌入
    embedding2 = await manager.embed(text)
    assert embedding == embedding2


@pytest.mark.asyncio
async def test_embedding_dimension_adaptation():
    """测试维度适配"""
    config = MockConfig()
    manager = EmbeddingManager(config)
    
    await manager.initialize()
    
    # 测试不同维度
    text = "测试文本"
    embedding_1024 = await manager.embed(text, dimension=1024)
    assert len(embedding_1024) == 1024
    
    embedding_512 = await manager.embed(text, dimension=512)
    assert len(embedding_512) == 512


@pytest.mark.asyncio
async def test_health_check():
    """测试健康检查"""
    config = MockConfig()
    manager = EmbeddingManager(config)
    
    await manager.initialize()
    
    health = await manager.health_check()
    
    assert "strategy" in health
    assert "current_provider" in health
    assert "providers" in health
    assert "stats" in health


@pytest.mark.asyncio
async def test_auto_strategy_fallback():
    """测试自动策略降级"""
    config = MockConfig()
    manager = EmbeddingManager(config)
    
    # 自动策略应该初始化多个提供者
    await manager.initialize()
    
    # 应该至少有一个提供者
    assert len(manager.providers) >= 1
    
    # 测试生成嵌入（应该自动选择最佳提供者）
    text = "测试自动降级"
    embedding = await manager.embed(text)
    
    assert len(embedding) > 0
    assert manager.current_provider is not None


if __name__ == "__main__":
    asyncio.run(test_embedding_manager_initialization())
    asyncio.run(test_fallback_provider())
    asyncio.run(test_embedding_dimension_adaptation())
    asyncio.run(test_health_check())
    asyncio.run(test_auto_strategy_fallback())
    print("All tests passed!")

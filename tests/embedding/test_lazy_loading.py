"""
延迟加载机制测试

测试 LocalProvider 的后台加载、is_ready 属性和 EmbeddingManager 的代理功能。
"""

import asyncio
import threading
import time
from unittest.mock import Mock, patch
import pytest

from iris_memory.embedding.base import EmbeddingProvider, EmbeddingRequest
from iris_memory.embedding.manager import EmbeddingManager
from iris_memory.embedding.local_provider import LocalProvider
from iris_memory.embedding.fallback_provider import FallbackProvider
from iris_memory.core.config_manager import init_config_manager, reset_config_manager


class MockConfig:
    """模拟配置对象"""
    def __init__(self, embedding_strategy="auto", enable_local_provider=True):
        self._data = {
            "embedding": {
                "embedding_strategy": embedding_strategy,
                "embedding_model": "BAAI/bge-small-zh-v1.5",
                "embedding_dimension": 512,
                "auto_detect_dimension": True,
                "enable_local_provider": enable_local_provider,
            }
        }
        self._plugin_context = None

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'MockConfig' object has no attribute '{name}'")


@pytest.fixture(autouse=True)
def _reset_global_config_manager():
    """每个测试用例前后重置全局配置管理器，避免状态污染。"""
    reset_config_manager()
    init_config_manager(MockConfig())
    yield
    reset_config_manager()


# ==================== 基类 is_ready 属性测试 ====================

class TestBaseProviderIsReady:
    """测试 EmbeddingProvider 基类的 is_ready 属性"""

    def test_is_ready_returns_false_when_dimension_not_set(self):
        """_dimension 未设置时 is_ready 应返回 False"""
        # 创建一个具体实现来测试基类行为
        class ConcreteProvider(EmbeddingProvider):
            async def initialize(self) -> bool:
                return True

            async def embed(self, request: EmbeddingRequest):
                pass

            async def embed_batch(self, requests):
                return []

        provider = ConcreteProvider(Mock())
        # _dimension 未设置
        assert provider._dimension is None
        assert provider.is_ready is False

    def test_is_ready_returns_true_when_dimension_set(self):
        """_dimension 已设置时 is_ready 应返回 True"""
        class ConcreteProvider(EmbeddingProvider):
            async def initialize(self) -> bool:
                self._dimension = 512
                return True

            async def embed(self, request: EmbeddingRequest):
                pass

            async def embed_batch(self, requests):
                return []

        provider = ConcreteProvider(Mock())
        provider._dimension = 512
        assert provider.is_ready is True


# ==================== LocalProvider is_ready 测试 ====================

class TestLocalProviderIsReady:
    """测试 LocalProvider 的 is_ready 属性"""

    def test_is_ready_returns_false_initially(self):
        """初始化后模型未加载时 is_ready 应返回 False"""
        provider = LocalProvider(MockConfig())
        # 未初始化
        assert provider.is_ready is False

    @pytest.mark.asyncio
    async def test_is_ready_returns_false_after_initialize_starts(self):
        """initialize() 启动后台加载后，模型加载完成前 is_ready 为 False"""
        provider = LocalProvider(MockConfig())

        # 模拟依赖检查通过但模型加载需要时间
        with patch.object(provider, '_start_background_load'):
            # 只调用父类的初始化逻辑，不实际加载模型
            result = await provider.initialize()

            # 后台加载已启动但未完成
            assert provider._load_complete.is_set() is False
            assert provider.is_ready is False

    @pytest.mark.asyncio
    async def test_is_ready_returns_true_after_load_complete(self):
        """模型加载完成后 is_ready 应返回 True"""
        provider = LocalProvider(MockConfig())

        # 模拟加载成功
        provider._load_complete.set()
        provider._dimension = 512

        assert provider.is_ready is True

    @pytest.mark.asyncio
    async def test_is_ready_returns_false_on_load_error(self):
        """模型加载失败时 is_ready 应返回 False"""
        provider = LocalProvider(MockConfig())

        # 模拟加载失败
        provider._load_complete.set()
        provider._load_error = RuntimeError("Test error")

        assert provider.is_ready is False


# ==================== LocalProvider 后台加载测试 ====================

class TestLocalProviderBackgroundLoading:
    """测试 LocalProvider 的后台加载机制"""

    @pytest.mark.asyncio
    async def test_initialize_starts_background_thread(self):
        """initialize() 应启动后台线程"""
        provider = LocalProvider(MockConfig())
        init_config_manager(MockConfig())

        with patch.object(provider, '_start_background_load') as mock_start:
            with patch('torch.cuda.is_available', return_value=False):
                await provider.initialize()

            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_returns_false_when_disabled(self):
        """配置禁用时 initialize() 应返回 False"""
        init_config_manager(MockConfig(enable_local_provider=False))
        provider = LocalProvider(MockConfig(enable_local_provider=False))

        result = await provider.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_returns_false_without_dependencies(self):
        """缺少依赖时 initialize() 应返回 False"""
        provider = LocalProvider(MockConfig())
        init_config_manager(MockConfig())

        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("sentence_transformers"):
                raise ImportError("No module")
            return real_import(name, globals, locals, fromlist, level)

        with patch('builtins.__import__', side_effect=fake_import):
            result = await provider.initialize()

        assert result is False

    def test_background_load_updates_dimension_on_success(self):
        """后台加载成功时应更新维度"""
        provider = LocalProvider(MockConfig())
        init_config_manager(MockConfig())

        # 模拟模型实例
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            provider._start_background_load("test-model")
            assert provider._load_complete.wait(timeout=2)

        assert provider._dimension == 768
        assert provider.is_ready is True

    def test_background_load_sets_error_on_failure(self):
        """后台加载失败时应设置错误"""
        provider = LocalProvider(MockConfig())

        # 模拟加载失败
        provider._load_error = RuntimeError("Load failed")
        provider._load_complete.set()

        assert provider.is_ready is False

    def test_wait_for_model_raises_on_timeout(self):
        """_wait_for_model 超时应抛出异常"""
        provider = LocalProvider(MockConfig())
        # 加载未完成

        with pytest.raises(RuntimeError, match="loading timed out"):
            provider._wait_for_model(timeout=0.1)

    def test_wait_for_model_raises_on_error(self):
        """_wait_for_model 加载错误时应抛出异常"""
        provider = LocalProvider(MockConfig())
        provider._load_error = RuntimeError("Model failed")
        provider._load_complete.set()

        with pytest.raises(RuntimeError, match="failed to load"):
            provider._wait_for_model()


# ==================== LocalProvider embed 等待测试 ====================

class TestLocalProviderEmbedWait:
    """测试 LocalProvider 的 embed 等待机制"""

    @pytest.mark.asyncio
    async def test_embed_waits_for_model_load(self):
        """embed 应等待模型加载完成"""
        provider = LocalProvider(MockConfig())
        provider._model = "test-model"

        # 模拟模型
        mock_model = Mock()
        mock_model.encode.return_value = [0.1] * 512

        # 在后台延迟设置完成状态
        def delayed_load():
            time.sleep(0.1)
            provider._model_instance = mock_model
            provider._dimension = 512
            provider._load_complete.set()

        thread = threading.Thread(target=delayed_load, daemon=True)
        thread.start()

        # 初始状态
        assert provider.is_ready is False

        # embed 应等待并成功
        request = EmbeddingRequest(text="test")
        response = await provider.embed(request)

        assert response is not None
        assert provider.is_ready is True

    @pytest.mark.asyncio
    async def test_embed_raises_on_load_error(self):
        """模型加载失败时 embed 应抛出异常"""
        provider = LocalProvider(MockConfig())
        provider._load_error = RuntimeError("Load failed")
        provider._load_complete.set()

        request = EmbeddingRequest(text="test")

        with pytest.raises(RuntimeError, match="failed to load"):
            await provider.embed(request)


# ==================== EmbeddingManager is_ready 测试 ====================

class TestEmbeddingManagerIsReady:
    """测试 EmbeddingManager 的 is_ready 属性代理"""

    @pytest.mark.asyncio
    async def test_is_ready_returns_false_without_provider(self):
        """无提供者时 is_ready 应返回 False"""
        manager = EmbeddingManager(MockConfig())
        manager.current_provider = None

        assert manager.is_ready is False

    @pytest.mark.asyncio
    async def test_is_ready_proxies_to_provider(self):
        """is_ready 应代理到当前提供者"""
        manager = EmbeddingManager(MockConfig())

        # 创建 Mock 提供者
        mock_provider = Mock()
        mock_provider.is_ready = True
        manager.current_provider = mock_provider

        assert manager.is_ready is True

        mock_provider.is_ready = False
        assert manager.is_ready is False

    @pytest.mark.asyncio
    async def test_is_ready_with_fallback_provider(self):
        """FallbackProvider 应立即就绪"""
        manager = EmbeddingManager(MockConfig())
        fallback = FallbackProvider(MockConfig())
        await fallback.initialize()
        manager.current_provider = fallback

        # FallbackProvider 立即就绪
        assert manager.is_ready is True

    @pytest.mark.asyncio
    async def test_initialization_logs_loading_status(self):
        """初始化日志应正确显示 loading 状态"""
        init_config_manager(MockConfig(enable_local_provider=False))
        manager = EmbeddingManager(MockConfig(embedding_strategy="fallback", enable_local_provider=False))

        with patch('iris_memory.embedding.manager.logger') as mock_logger:
            await manager.initialize()

            # 检查日志包含提供者信息
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Selected" in str(call) or "Initialized" in str(call) for call in log_calls)

    @pytest.mark.asyncio
    async def test_embed_fallback_when_local_load_failed(self):
        """当前 provider 为 local 且加载失败时，应自动降级到 fallback"""
        manager = EmbeddingManager(MockConfig())

        local_provider = LocalProvider(MockConfig())
        local_provider._load_error = RuntimeError("mock load failed")
        local_provider._load_complete.set()

        fallback_provider = FallbackProvider(MockConfig())
        await fallback_provider.initialize()

        manager.providers = {
            "local": local_provider,
            "fallback": fallback_provider,
        }
        manager.current_provider = local_provider

        embedding = await manager.embed("fallback-case", dimension=128)

        assert len(embedding) == 128
        assert manager.current_provider is fallback_provider


# ==================== 配置禁用测试 ====================

class TestLocalProviderDisableConfig:
    """测试配置禁用本地提供者"""

    @pytest.mark.asyncio
    async def test_disabled_provider_not_initialized(self):
        """配置禁用时不应初始化"""
        init_config_manager(MockConfig(enable_local_provider=False))
        provider = LocalProvider(MockConfig(enable_local_provider=False))

        result = await provider.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_auto_strategy_skips_disabled_local(self):
        """AUTO 策略应跳过被禁用的 LocalProvider"""
        init_config_manager(MockConfig(enable_local_provider=False))
        config = MockConfig(enable_local_provider=False)
        manager = EmbeddingManager(config)

        await manager.initialize()

        # 应该没有 local 提供者
        assert "local" not in manager.providers
        # 应该有 fallback 作为后备
        assert "fallback" in manager.providers


# ==================== 状态转换测试 ====================

class TestStateTransitions:
    """测试状态转换"""

    @pytest.mark.asyncio
    async def test_state_transitions_correctly(self):
        """测试状态正确转换：未初始化 -> 加载中 -> 就绪"""
        provider = LocalProvider(MockConfig())

        # 初始状态
        assert provider.is_ready is False

        # 模拟后台加载完成
        provider._model_instance = Mock()
        provider._dimension = 512
        provider._load_complete.set()

        assert provider.is_ready is True

    @pytest.mark.asyncio
    async def test_state_transitions_to_error(self):
        """测试状态正确转换到错误状态"""
        provider = LocalProvider(MockConfig())

        # 模拟加载失败
        provider._load_error = RuntimeError("Test error")
        provider._load_complete.set()

        assert provider.is_ready is False


# ==================== health_check 测试 ====================

class TestHealthCheck:
    """测试健康检查"""

    @pytest.mark.asyncio
    async def test_health_check_shows_loading_status(self):
        """健康检查应显示加载状态"""
        provider = LocalProvider(MockConfig())
        provider._model = "test-model"
        provider._dimension = 512

        # 未加载完成
        health = await provider.health_check()

        assert health["status"] == "loading"
        assert health["loading"] is True
        assert health["is_ready"] is False

    @pytest.mark.asyncio
    async def test_health_check_shows_ready_status(self):
        """健康检查应显示就绪状态"""
        provider = LocalProvider(MockConfig())
        provider._model_instance = Mock()
        provider._dimension = 512
        provider._load_complete.set()

        # Mock embed 方法
        async def mock_embed(*args):
            from iris_memory.embedding.base import EmbeddingResponse
            import numpy as np
            return EmbeddingResponse(
                embedding=np.array([0.1] * 512),
                model="test",
                dimension=512
            )

        provider.embed = mock_embed

        health = await provider.health_check()

        assert health["status"] == "ok"
        assert health["is_ready"] is True

    @pytest.mark.asyncio
    async def test_health_check_shows_error_status(self):
        """健康检查应显示错误状态"""
        provider = LocalProvider(MockConfig())
        provider._load_error = RuntimeError("Test error")
        provider._load_complete.set()

        health = await provider.health_check()

        assert health["status"] == "error"
        assert health["load_error"] is not None


# ==================== 并发安全测试 ====================

class TestConcurrencySafety:
    """测试并发安全性"""

    @pytest.mark.asyncio
    async def test_concurrent_is_ready_calls(self):
        """并发调用 is_ready 应安全"""
        provider = LocalProvider(MockConfig())

        results = []

        def check_ready():
            results.append(provider.is_ready)

        # 创建多个线程同时检查状态
        threads = [threading.Thread(target=check_ready) for _ in range(10)]

        for t in threads:
            t.start()

        # 模拟加载完成
        provider._dimension = 512
        provider._load_complete.set()

        for t in threads:
            t.join()

        # 所有调用都应成功（无异常）
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_concurrent_embed_waits_properly(self):
        """并发 embed 调用应正确等待"""
        provider = LocalProvider(MockConfig())
        provider._model = "test-model"

        # 模拟模型
        mock_model = Mock()
        mock_model.encode.return_value = [0.1] * 512

        call_count = 0

        def delayed_set():
            nonlocal call_count
            time.sleep(0.1)
            provider._model_instance = mock_model
            provider._dimension = 512
            provider._load_complete.set()

        threading.Thread(target=delayed_set, daemon=True).start()

        # 并发调用 embed
        async def call_embed():
            request = EmbeddingRequest(text="test")
            return await provider.embed(request)

        # 多个并发调用
        results = await asyncio.gather(*[call_embed() for _ in range(3)])

        # 所有调用都应成功
        assert all(r is not None for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

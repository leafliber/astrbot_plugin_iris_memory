"""生命周期集成测试

验证完整初始化流程：
- ComponentManager 初始化顺序
- 依赖注入机制
- 组件间协作
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from iris_memory.core.components import ComponentManager
from iris_memory.core.lifecycle import initialize_components, _inject_component_manager
from iris_memory.l1_buffer import L1Buffer
from iris_memory.llm.manager import LLMManager
from iris_memory.config import init_config


class TestLifecycleIntegration:
    """生命周期集成测试"""

    @pytest.fixture
    def mock_context(self):
        """创建 mock Context"""
        context = MagicMock()

        # Mock llm_generate
        mock_response = MagicMock()
        mock_response.completion_text = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.input_other = 80
        mock_response.usage.input_cached = 20
        mock_response.usage.output = 50
        context.llm_generate = AsyncMock(return_value=mock_response)

        # Mock KV storage
        context.get_kv_data = MagicMock(return_value={})
        context.put_kv_data = MagicMock()

        return context

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        storage.get_kv_data = AsyncMock(return_value={})
        storage.put_kv_data = AsyncMock()
        return storage

    @pytest.fixture
    def mock_config(self, tmp_path: Path):
        """模拟配置"""
        astrbot_config = Mock()
        astrbot_config.__getitem__ = Mock(
            return_value={
                "enable": True,
                "summary_provider": "",
                "inject_queue_length": 20,
                "max_queue_tokens": 4000,
            }
        )
        astrbot_config.__contains__ = Mock(return_value=True)

        return init_config(astrbot_config, tmp_path)

    @pytest.mark.asyncio
    async def test_component_initialization_order(
        self, mock_context, mock_storage, mock_config
    ):
        """测试组件初始化顺序

        验证：LLMManager 应该最先初始化，然后是其他组件
        """
        with patch("iris_memory.llm.manager.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config

            # 创建组件实例
            llm_manager = LLMManager(mock_context, mock_storage)
            l1_buffer = L1Buffer()

            # 跟踪初始化顺序
            init_order = []

            original_llm_init = llm_manager.initialize

            async def tracked_llm_init():
                init_order.append("llm_manager")
                return await original_llm_init()

            original_buffer_init = l1_buffer.initialize

            async def tracked_buffer_init():
                init_order.append("l1_buffer")
                return await original_buffer_init()

            llm_manager.initialize = tracked_llm_init
            l1_buffer.initialize = tracked_buffer_init

            # 创建 ComponentManager
            component_manager = ComponentManager([llm_manager, l1_buffer])

            # 初始化所有组件
            await component_manager.initialize_all()

            # 验证初始化顺序
            assert len(init_order) == 2
            assert init_order[0] == "llm_manager"
            assert init_order[1] == "l1_buffer"

    @pytest.mark.asyncio
    async def test_component_manager_injection(
        self, mock_context, mock_storage, mock_config
    ):
        """测试 ComponentManager 注入机制

        验证修复：初始化完成后应该注入 ComponentManager 到 L1Buffer
        """
        with patch("iris_memory.llm.manager.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config

            # 创建组件
            llm_manager = LLMManager(mock_context, mock_storage)
            l1_buffer = L1Buffer()

            # 创建 ComponentManager
            component_manager = ComponentManager([llm_manager, l1_buffer])

            # 初始化所有组件
            await component_manager.initialize_all()

            # 执行注入
            _inject_component_manager(component_manager)

            # 验证 L1Buffer 已获取 ComponentManager 引用
            assert l1_buffer._component_manager == component_manager

    @pytest.mark.asyncio
    async def test_full_initialization_flow(
        self, mock_context, mock_storage, mock_config
    ):
        """测试完整初始化流程

        验证修复后的完整流程：
        1. 初始化所有组件
        2. 注入 ComponentManager
        3. L1Buffer 可以获取 LLMManager
        """
        with patch("iris_memory.llm.manager.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config

            # 创建组件
            llm_manager = LLMManager(mock_context, mock_storage)
            l1_buffer = L1Buffer()

            # 创建 ComponentManager
            component_manager = ComponentManager([llm_manager, l1_buffer])

            # 执行完整初始化
            success = await initialize_components(component_manager)

            # 验证初始化成功
            assert success is True

            # 验证组件已初始化
            assert llm_manager.is_available
            assert l1_buffer.is_available

            # 验证 L1Buffer 已获取 ComponentManager
            assert l1_buffer._component_manager == component_manager

            # 验证 L1Buffer 可以获取 LLMManager
            summarizer = l1_buffer._get_or_create_summarizer()
            assert summarizer is not None
            assert summarizer.llm_manager == llm_manager

    @pytest.mark.asyncio
    async def test_injection_idempotent(self, mock_context, mock_storage, mock_config):
        """测试注入是幂等的

        多次注入应该不会出错
        """
        with patch("iris_memory.llm.manager.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config

            l1_buffer = L1Buffer()
            component_manager = ComponentManager([l1_buffer])

            await component_manager.initialize_all()

            # 多次注入
            _inject_component_manager(component_manager)
            _inject_component_manager(component_manager)
            _inject_component_manager(component_manager)

            # 应该仍然是同一个引用
            assert l1_buffer._component_manager == component_manager

    @pytest.mark.asyncio
    async def test_injection_with_missing_component(self, mock_config):
        """测试组件缺失时的注入

        验证：如果组件不存在，注入不会出错
        """
        component_manager = ComponentManager([])

        # L1Buffer 不存在
        assert component_manager.get_component("l1_buffer") is None

        # 注入不应该出错
        _inject_component_manager(component_manager)

    @pytest.mark.asyncio
    async def test_component_dependencies(
        self, mock_context, mock_storage, mock_config
    ):
        """测试组件依赖关系

        验证：L1Buffer 依赖 LLMManager，但初始化时 LLMManager 已经可用
        """
        with patch("iris_memory.llm.manager.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config

            llm_manager = LLMManager(mock_context, mock_storage)
            l1_buffer = L1Buffer()

            component_manager = ComponentManager([llm_manager, l1_buffer])

            # 初始化并注入
            await initialize_components(component_manager)

            # L1Buffer 应该能成功创建 Summarizer
            summarizer = l1_buffer._get_or_create_summarizer()
            assert summarizer is not None

            # 验证 Summarizer 使用了正确的 LLMManager
            assert summarizer.llm_manager == llm_manager

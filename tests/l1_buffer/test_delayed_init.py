"""L1Buffer 延迟初始化测试

验证修复的问题：
- L1Buffer 延迟获取 LLMManager
- 通过 ComponentManager 注入实现依赖
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from iris_memory.l1_buffer import L1Buffer
from iris_memory.core.components import ComponentManager
from iris_memory.llm.manager import LLMManager
from iris_memory.config import init_config


class TestL1BufferDelayedInit:
    """L1Buffer 延迟初始化测试"""

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
                "max_single_message_tokens": 500,
            }
        )
        astrbot_config.__contains__ = Mock(return_value=True)

        return init_config(astrbot_config, tmp_path)

    @pytest.fixture
    def mock_llm_manager(self):
        manager = MagicMock(spec=LLMManager)
        manager.is_available = True
        manager.name = "llm_manager"
        manager.generate = AsyncMock(return_value="这是一个总结")
        return manager

    @pytest.mark.asyncio
    async def test_summarizer_not_created_on_init(self, mock_config):
        """测试初始化时不创建 Summarizer

        验证修复：Summarizer 应该延迟创建，而不是在 initialize() 中创建
        """
        buffer = L1Buffer()
        await buffer.initialize()

        # 初始化后 Summarizer 应该为 None
        assert buffer._summarizer is None
        assert buffer._component_manager is None
        assert buffer._provider == ""  # 从配置读取

    @pytest.mark.asyncio
    async def test_set_component_manager(self, mock_config):
        """测试设置 ComponentManager"""
        buffer = L1Buffer()
        await buffer.initialize()

        # 创建 ComponentManager
        component_manager = ComponentManager([])

        # 设置 ComponentManager 引用
        buffer.set_component_manager(component_manager)

        # 验证引用已设置
        assert buffer._component_manager == component_manager

    @pytest.mark.asyncio
    async def test_delayed_summarizer_creation(self, mock_config, mock_llm_manager):
        """测试延迟创建 Summarizer

        验证修复：Summarizer 应该在需要时才创建
        """
        buffer = L1Buffer()
        await buffer.initialize()

        # 创建 ComponentManager 并注入 LLMManager
        component_manager = ComponentManager([mock_llm_manager])

        # 设置 ComponentManager 引用
        buffer.set_component_manager(component_manager)

        # 调用内部方法获取 Summarizer
        summarizer = buffer._get_or_create_summarizer()

        # 验证 Summarizer 已创建
        assert summarizer is not None
        assert buffer._summarizer == summarizer

        # 再次调用应该返回缓存的实例
        summarizer2 = buffer._get_or_create_summarizer()
        assert summarizer2 == summarizer  # 同一个实例

    @pytest.mark.asyncio
    async def test_summarizer_creation_without_component_manager(self, mock_config):
        """测试没有 ComponentManager 时创建 Summarizer

        应该返回 None，记录警告日志
        """
        buffer = L1Buffer()
        await buffer.initialize()

        # 不设置 ComponentManager
        assert buffer._component_manager is None

        # 尝试获取 Summarizer
        summarizer = buffer._get_or_create_summarizer()

        # 应该返回 None
        assert summarizer is None
        assert buffer._summarizer is None

    @pytest.mark.asyncio
    async def test_summarizer_creation_with_unavailable_llm(self, mock_config):
        """测试 LLMManager 不可用时创建 Summarizer"""
        buffer = L1Buffer()
        await buffer.initialize()

        # 创建不可用的 LLMManager
        unavailable_llm = MagicMock()
        unavailable_llm.is_available = False
        unavailable_llm.name = "llm_manager"

        component_manager = ComponentManager([unavailable_llm])

        buffer.set_component_manager(component_manager)

        # 尝试获取 Summarizer
        summarizer = buffer._get_or_create_summarizer()

        # 应该返回 None
        assert summarizer is None

    @pytest.mark.asyncio
    async def test_provider_config_saved_on_init(self, mock_config):
        """测试 Provider 配置在初始化时保存"""
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.enable": True,
                    "l1_buffer.summary_provider": "gpt-4o",
                }.get(key, default)
            )

            buffer = L1Buffer()
            await buffer.initialize()

            # 验证 Provider 已保存
            assert buffer._provider == "gpt-4o"

    @pytest.mark.asyncio
    async def test_full_initialization_flow(self, mock_config, mock_llm_manager):
        """测试完整初始化流程

        验证修复后的流程：
        1. L1Buffer.initialize() - 不创建 Summarizer
        2. ComponentManager 注入
        3. 首次调用 _get_or_create_summarizer() - 创建 Summarizer
        """
        buffer = L1Buffer()
        await buffer.initialize()

        # 阶段1：初始化完成，Summarizer 为 None
        assert buffer._summarizer is None

        # 阶段2：注入 ComponentManager
        component_manager = ComponentManager([mock_llm_manager])
        buffer.set_component_manager(component_manager)

        # 阶段3：首次获取 Summarizer
        summarizer = buffer._get_or_create_summarizer()

        # 验证 Summarizer 已创建并缓存
        assert summarizer is not None
        assert buffer._summarizer is summarizer

        # 验证 Summarizer 使用了正确的 LLMManager
        assert summarizer.llm_manager == mock_llm_manager

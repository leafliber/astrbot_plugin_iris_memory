"""端到端测试：消息处理到总结流程

验证完整业务流程：
1. 消息添加到 L1 缓冲
2. 触发总结条件
3. LLM 生成总结
4. Token 统计记录
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from iris_memory.l1_buffer import L1Buffer
from iris_memory.llm.manager import LLMManager
from iris_memory.core.components import ComponentManager
from iris_memory.core.lifecycle import initialize_components
from iris_memory.config import init_config


class TestMessageToSummaryFlow:
    """消息到总结的端到端流程测试"""

    @pytest.fixture
    def mock_context(self):
        """创建 mock Context"""
        context = MagicMock()

        # Mock llm_generate
        mock_response = MagicMock()
        mock_response.completion_text = (
            "这是一段对话总结：用户询问了关于项目的问题，助手提供了相关建议。"
        )
        mock_response.usage = MagicMock()
        mock_response.usage.input_other = 160
        mock_response.usage.input_cached = 40
        mock_response.usage.output = 40
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
                "summary_provider": "gpt-4o-mini",
                "inject_queue_length": 10,
                "max_queue_tokens": 2000,
                "max_single_message_tokens": 500,
            }
        )
        astrbot_config.__contains__ = Mock(return_value=True)

        return init_config(astrbot_config, tmp_path)

    @pytest.mark.asyncio
    async def test_full_message_to_summary_flow(
        self, mock_context, mock_storage, mock_config
    ):
        """测试完整的消息到总结流程

        端到端验证：
        1. 初始化所有组件
        2. 添加消息到 L1 缓冲
        3. 检查总结条件
        4. 生成总结（模拟）
        5. 验证 Token 统计
        """
        with (
            patch("iris_memory.llm.manager.get_config") as mock_get_config,
            patch("iris_memory.l1_buffer.buffer.get_config") as mock_buffer_config,
        ):
            mock_get_config.return_value = mock_config
            mock_buffer_config.return_value = mock_config

            # ===== 阶段 1：初始化组件 =====
            llm_manager = LLMManager(mock_context, mock_storage)
            l1_buffer = L1Buffer()

            component_manager = ComponentManager([llm_manager, l1_buffer])

            success = await initialize_components(component_manager)
            assert success is True

            # ===== 阶段 2：添加消息 =====
            # 模拟用户对话
            messages = [
                ("user", "你好，我想了解这个项目"),
                ("assistant", "你好！这是一个记忆管理插件"),
                ("user", "它有什么功能？"),
                ("assistant", "支持多层级记忆存储和智能总结"),
                ("user", "如何配置？"),
                ("assistant", "可以在配置文件中设置参数"),
            ]

            for role, content in messages:
                success = await l1_buffer.add_message(
                    group_id="group_123", role=role, content=content, source="user_456"
                )
                assert success is True

            # 验证消息已添加
            stats = l1_buffer.get_queue_stats("group_123")
            assert stats is not None
            assert stats["message_count"] == 6

            # ===== 阶段 3：获取上下文 =====
            context = l1_buffer.get_context("group_123", max_length=10)
            assert len(context) == 6

            # ===== 阶段 4：手动调用 LLM（模拟总结） =====
            summary = await llm_manager.generate(
                prompt="请总结以下对话：\n"
                + "\n".join([f"{msg.role}: {msg.content}" for msg in context]),
                module="l1_summarizer",
                provider_id="gpt-4o-mini",
            )

            assert summary is not None
            assert "总结" in summary

            # ===== 阶段 5：验证 Token 统计 =====
            stats = await llm_manager.get_token_stats("l1_summarizer")
            assert stats["total_input_tokens"] == 200
            assert stats["total_output_tokens"] == 40
            assert stats["total_calls"] == 1

            # ===== 阶段 6：验证调用日志 =====
            logs = llm_manager.get_recent_call_logs()
            assert len(logs) == 1
            assert logs[0]["success"] is True
            assert logs[0]["module"] == "l1_summarizer"

    @pytest.mark.asyncio
    async def test_multi_group_message_isolation(
        self, mock_context, mock_storage, mock_config
    ):
        """测试多群聊消息隔离

        端到端验证：
        1. 多个群聊同时添加消息
        2. 每个群聊独立管理
        3. 总结互不干扰
        """
        with (
            patch("iris_memory.llm.manager.get_config") as mock_get_config,
            patch("iris_memory.l1_buffer.buffer.get_config") as mock_buffer_config,
        ):
            mock_get_config.return_value = mock_config
            mock_buffer_config.return_value = mock_config

            # 初始化
            llm_manager = LLMManager(mock_context, mock_storage)
            l1_buffer = L1Buffer()

            component_manager = ComponentManager([llm_manager, l1_buffer])

            await initialize_components(component_manager)

            # 群聊 1
            await l1_buffer.add_message("group_111", "user", "群聊1的消息", "user_1")
            await l1_buffer.add_message("group_111", "assistant", "群聊1的回复", "bot")

            # 群聊 2
            await l1_buffer.add_message("group_222", "user", "群聊2的消息", "user_2")
            await l1_buffer.add_message("group_222", "assistant", "群聊2的回复", "bot")

            # 验证隔离
            context1 = l1_buffer.get_context("group_111")
            context2 = l1_buffer.get_context("group_222")

            assert len(context1) == 2
            assert len(context2) == 2
            assert context1[0].content == "群聊1的消息"
            assert context2[0].content == "群聊2的消息"

    @pytest.mark.asyncio
    async def test_large_message_handling(
        self, mock_context, mock_storage, mock_config
    ):
        """测试大消息处理

        端到端验证：
        1. 超大消息被拒绝
        2. 正常消息正常处理
        """
        with (
            patch("iris_memory.llm.manager.get_config") as mock_get_config,
            patch("iris_memory.l1_buffer.buffer.get_config") as mock_buffer_config,
        ):
            # 配置较小的消息限制
            def config_side_effect(key, default=None):
                config_map = {
                    "l1_buffer.enable": True,
                    "l1_max_single_message_tokens": 100,
                    "l1_buffer.inject_queue_length": 10,
                    "l1_max_queue_tokens": 2000,
                }
                return config_map.get(key, default)

            mock_get_config.return_value = mock_config
            mock_buffer_config.return_value.get = Mock(side_effect=config_side_effect)

            llm_manager = LLMManager(mock_context, mock_storage)
            l1_buffer = L1Buffer()

            component_manager = ComponentManager([llm_manager, l1_buffer])

            await initialize_components(component_manager)

            # 大消息（超过 100 tokens）
            large_content = "这是一条很长的消息" * 100

            success = await l1_buffer.add_message(
                group_id="group_123",
                role="user",
                content=large_content,
                source="user_456",
            )

            # 应该被拒绝
            assert success is False

            # 正常消息
            success = await l1_buffer.add_message(
                group_id="group_123", role="user", content="正常消息", source="user_456"
            )

            assert success is True

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_config):
        """测试优雅降级

        端到端验证：
        1. LLMManager 不可用时
        2. L1Buffer 仍然可以正常工作
        3. 总结功能优雅降级
        """
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_buffer_config:
            mock_buffer_config.return_value = mock_config

            # 不创建 LLMManager
            l1_buffer = L1Buffer()

            component_manager = ComponentManager([l1_buffer])

            await component_manager.initialize_all()

            # L1Buffer 应该可以正常工作
            assert l1_buffer.is_available

            # 添加消息应该成功
            success = await l1_buffer.add_message(
                group_id="group_123", role="user", content="测试消息", source="user_456"
            )

            assert success is True

            # 获取上下文应该正常
            context = l1_buffer.get_context("group_123")
            assert len(context) == 1

            # 尝试获取 Summarizer 应该返回 None（优雅降级）
            summarizer = l1_buffer._get_or_create_summarizer()
            assert summarizer is None

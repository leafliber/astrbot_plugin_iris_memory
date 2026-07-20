"""图片解析集成测试

测试消息钩子与图片解析的完整集成流程。
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from iris_memory.image import ImageInfo, ParseResult
from iris_memory.core.message_hook import handle_user_message


class TestImageParsingIntegration:
    """图片解析集成测试"""

    @pytest.fixture
    def mock_event(self):
        """模拟 AstrBot 消息事件"""
        event = Mock()
        event.message_str = "看看这张图片"
        return event

    @pytest.fixture
    def mock_adapter(self):
        """模拟平台适配器"""
        adapter = Mock()
        adapter.get_group_id = Mock(return_value="test_group")
        adapter.get_user_id = Mock(return_value="test_user")
        adapter.get_user_name = Mock(return_value="测试用户")
        adapter.get_images = Mock(return_value=[])
        return adapter

    @pytest.fixture
    def mock_l1_buffer(self):
        """模拟 L1 Buffer"""
        buffer = Mock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()
        buffer._queued_images = []
        buffer._image_queues = {}

        def add_image(group_id, item):
            buffer._queued_images.append(item)

        def get_images(group_id, limit=5, only_pending=False):
            return buffer._queued_images[:limit]

        buffer.add_image = Mock(side_effect=add_image)
        buffer.get_images = Mock(side_effect=get_images)
        buffer.mark_image_parsed = Mock()
        return buffer

    @pytest.fixture
    def mock_quota_manager(self):
        """模拟配额管理器"""
        manager = Mock()
        manager.is_available = True
        manager.check_quota = AsyncMock(return_value=True)
        manager.use_quota = AsyncMock(return_value=True)
        manager.release_quota = AsyncMock(return_value=0)
        return manager

    @pytest.fixture
    def mock_llm_manager(self):
        """模拟 LLM Manager"""
        manager = Mock()
        manager.is_available = True
        manager.generate_with_images = AsyncMock(return_value="这是一张风景图片")
        return manager

    @pytest.fixture
    def mock_component_manager(
        self, mock_l1_buffer, mock_quota_manager, mock_llm_manager
    ):
        """模拟组件管理器"""
        manager = Mock()

        def get_component(name: str, expected_type=None):
            if name == "l1_buffer":
                return mock_l1_buffer
            elif name == "image_quota":
                return mock_quota_manager
            elif name == "llm_manager":
                return mock_llm_manager
            return None

        def get_available_component(name: str):
            comp = get_component(name)
            if comp and comp.is_available:
                return comp
            return None

        manager.get_component = Mock(side_effect=get_component)
        manager.get_available_component = Mock(side_effect=get_available_component)
        return manager

    @pytest.mark.asyncio
    async def test_all_mode_parse_images(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_l1_buffer,
        mock_quota_manager,
        mock_llm_manager,
    ):
        """测试 all 模式下解析图片"""
        # 准备测试数据
        images = [
            ImageInfo(url="https://example.com/image1.jpg"),
            ImageInfo(url="https://example.com/image2.jpg"),
        ]
        mock_adapter.get_images = Mock(return_value=images)

        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
            patch(
                "iris_memory.image.ImageParser._fetch_image_data_url",
                new_callable=AsyncMock,
                return_value="data:image/jpeg;base64,bW9jaw==",
            ),
        ):
            # 配置 mock
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "all",
                    "l1_buffer.image_parsing.provider": "",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            # 执行测试
            await handle_user_message(mock_event, mock_component_manager)

            # 验证配额检查
            mock_quota_manager.check_quota.assert_called_once()
            mock_quota_manager.use_quota.assert_called_once_with(2)

            # 验证 LLM 调用
            assert mock_llm_manager.generate_with_images.call_count == 2

            # 验证 L1 Buffer 入队
            assert mock_l1_buffer.add_message.call_count >= 1
            # 检查图片已标记为解析成功
            assert mock_l1_buffer.mark_image_parsed.call_count >= 2

    @pytest.mark.asyncio
    async def test_related_mode_skip_parsing(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_quota_manager,
    ):
        """测试 related 模式下不解析图片"""
        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
        ):
            # 配置 mock
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "related",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            # 执行测试
            await handle_user_message(mock_event, mock_component_manager)

            # 验证配额管理器未被调用
            mock_quota_manager.check_quota.assert_not_called()
            mock_quota_manager.use_quota.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_parsing(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_quota_manager,
    ):
        """测试禁用图片解析"""
        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
        ):
            # 配置 mock
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": False,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            # 执行测试
            await handle_user_message(mock_event, mock_component_manager)

            # 验证配额管理器未被调用
            mock_quota_manager.check_quota.assert_not_called()

    @pytest.mark.asyncio
    async def test_quota_exhausted(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_quota_manager,
        mock_llm_manager,
    ):
        """测试配额耗尽时跳过解析"""
        # 配置配额耗尽
        mock_quota_manager.check_quota = AsyncMock(return_value=False)

        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
        ):
            # 配置 mock
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "all",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            # 执行测试
            await handle_user_message(mock_event, mock_component_manager)

            # 验证 LLM 未被调用
            mock_llm_manager.generate_with_images.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_manager_unavailable(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_quota_manager,
        mock_llm_manager,
        mock_l1_buffer,
    ):
        """测试 LLM Manager 不可用时跳过解析"""
        mock_llm_manager.is_available = False

        images = [ImageInfo(url="https://example.com/image.jpg")]
        mock_adapter.get_images = Mock(return_value=images)

        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
        ):
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "all",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            await handle_user_message(mock_event, mock_component_manager)

            mock_llm_manager.generate_with_images.assert_not_called()

    @pytest.mark.asyncio
    async def test_l1_buffer_unavailable(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_l1_buffer,
        mock_quota_manager,
        mock_llm_manager,
    ):
        """测试 L1 Buffer 不可用时跳过所有处理"""
        images = [ImageInfo(url="https://example.com/image.jpg")]
        mock_adapter.get_images = Mock(return_value=images)

        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
        ):
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "all",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            mock_l1_buffer.is_available = False

            await handle_user_message(mock_event, mock_component_manager)

            mock_llm_manager.generate_with_images.assert_not_called()

    @pytest.mark.asyncio
    async def test_parse_failure_handling(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_l1_buffer,
        mock_quota_manager,
        mock_llm_manager,
    ):
        """测试解析失败时的错误处理"""
        # 配置图片和 LLM 失败
        images = [
            ImageInfo(url="https://example.com/image1.jpg"),
            ImageInfo(url="https://example.com/image2.jpg"),
        ]
        mock_adapter.get_images = Mock(return_value=images)

        # 第一张图片成功，第二张失败
        mock_llm_manager.generate_with_images = AsyncMock(
            side_effect=[
                "这是图片1的内容",
                Exception("LLM 调用失败"),
            ]
        )

        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
            patch(
                "iris_memory.image.ImageParser._fetch_image_data_url",
                new_callable=AsyncMock,
                return_value="data:image/jpeg;base64,bW9jaw==",
            ),
        ):
            # 配置 mock
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "all",
                    "l1_buffer.image_parsing.provider": "",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            # 执行测试
            await handle_user_message(mock_event, mock_component_manager)

            # 验证 LLM 调用了两次
            assert mock_llm_manager.generate_with_images.call_count == 2

            # 验证只有一张图片的解析结果标记成功
            assert mock_l1_buffer.mark_image_parsed.call_count >= 1

    @pytest.mark.asyncio
    async def test_no_images_in_message(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_quota_manager,
        mock_llm_manager,
    ):
        """测试消息中无图片时跳过解析"""
        # 配置无图片
        mock_adapter.get_images = Mock(return_value=[])

        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
        ):
            # 配置 mock
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "all",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            # 执行测试
            await handle_user_message(mock_event, mock_component_manager)

            # 验证配额未使用
            mock_quota_manager.use_quota.assert_not_called()

            # 验证 LLM 未调用
            mock_llm_manager.generate_with_images.assert_not_called()

    @pytest.mark.asyncio
    async def test_quota_manager_unavailable(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_quota_manager,
        mock_llm_manager,
    ):
        """测试配额管理器不可用时仍可解析（跳过配额检查）"""
        mock_quota_manager.is_available = False

        images = [ImageInfo(url="https://example.com/image.jpg")]
        mock_adapter.get_images = Mock(return_value=images)

        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
            patch(
                "iris_memory.image.ImageParser._fetch_image_data_url",
                new_callable=AsyncMock,
                return_value="data:image/jpeg;base64,bW9jaw==",
            ),
        ):
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "all",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            await handle_user_message(mock_event, mock_component_manager)

            mock_llm_manager.generate_with_images.assert_called()

    @pytest.mark.asyncio
    async def test_provider_config_passed_to_parser(
        self,
        mock_event,
        mock_adapter,
        mock_component_manager,
        mock_l1_buffer,
        mock_quota_manager,
        mock_llm_manager,
    ):
        """测试 Provider 配置正确传递给解析器"""
        images = [ImageInfo(url="https://example.com/image.jpg")]
        mock_adapter.get_images = Mock(return_value=images)

        with (
            patch("iris_memory.config.get_config") as mock_get_config,
            patch("iris_memory.platform.get_adapter") as mock_get_adapter,
            patch("iris_memory.image.ImageParser") as MockImageParser,
        ):
            # 配置 mock
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.mode": "all",
                    "l1_buffer.image_parsing.provider": "custom_vision_provider",
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config
            mock_get_adapter.return_value = mock_adapter

            # Mock ImageParser
            mock_parser_instance = Mock()
            mock_parser_instance.parse_batch = AsyncMock(
                return_value=[
                    ParseResult(
                        image_info=images[0],
                        content="图片内容",
                        success=True,
                    )
                ]
            )
            MockImageParser.return_value = mock_parser_instance

            # 执行测试
            await handle_user_message(mock_event, mock_component_manager)

            # 验证 ImageParser 使用正确的 provider 创建
            MockImageParser.assert_called_once_with(
                mock_llm_manager,
                "custom_vision_provider",
                recorder_bridge=None,
            )

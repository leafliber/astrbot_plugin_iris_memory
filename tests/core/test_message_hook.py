"""测试消息钩子处理模块"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import ExitStack
from datetime import datetime

from iris_memory.core.message_hook import (
    handle_user_message,
    update_l1_buffer,
    _backfill_reply_from_buffer,
    _parse_images_if_enabled,
    _queue_images_to_l1_buffer,
)
from iris_memory.image import ImageParseStatus
from iris_memory.image.models import ImageInfo, ImageQueueItem, ParseResult
from iris_memory.l1_buffer.models import ContextMessage
from iris_memory.platform.base import ReplyInfo


def _patch_handle_deps(adapter=None):
    patches = [
        patch("iris_memory.utils.sanitize_input", side_effect=lambda x, **kw: x),
        patch(
            "iris_memory.core.message_hook._update_profile_names",
            new_callable=AsyncMock,
        ),
        patch(
            "iris_memory.core.message_hook._queue_images_to_l1_buffer",
            new_callable=AsyncMock,
        ),
        patch(
            "iris_memory.core.message_hook._parse_images_if_enabled",
            new_callable=AsyncMock,
        ),
    ]
    if adapter:
        patches.append(patch("iris_memory.platform.get_adapter", return_value=adapter))
    return patches


class TestHandleUserMessage:
    """测试用户消息处理主函数"""

    @pytest.mark.asyncio
    async def test_handle_with_available_buffer(self):
        """测试 L1 Buffer 可用时的消息处理"""
        event = MagicMock()
        event.message_str = "你好"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_session_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "测试用户"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "999"}
        adapter.get_reply_info.return_value = ReplyInfo()

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["group_id"] == "group123"
        assert call_kwargs["role"] == "user"
        assert call_kwargs["content"] == "你好"
        assert call_kwargs["source"] == "user456"
        assert call_kwargs["metadata"]["user_name"] == "测试用户"
        assert call_kwargs["metadata"]["message_id"] == "999"
        assert "reply_message_id" not in call_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_handle_with_reply_info(self):
        """测试带回复信息的消息处理"""
        event = MagicMock()
        event.message_str = "我也觉得"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "李四"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "1000"}
        adapter.get_reply_info.return_value = ReplyInfo(
            message_id="6283", user_id="1234567", user_name="张三", content="你好啊"
        )

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["metadata"]["message_id"] == "1000"
        assert call_kwargs["metadata"]["reply_message_id"] == "6283"
        assert call_kwargs["metadata"]["reply_user_id"] == "1234567"
        assert call_kwargs["metadata"]["reply_user_name"] == "张三"
        assert call_kwargs["metadata"]["reply_content"] == "你好啊"

    @pytest.mark.asyncio
    async def test_handle_with_reply_no_content(self):
        """测试回复信息无内容时尝试 L1 Buffer 回填和 API 回填"""
        event = MagicMock()
        event.message_str = "是的"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()
        buffer.get_context.return_value = []

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = ""
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "1001"}
        adapter.get_reply_info.return_value = ReplyInfo(message_id="6283")
        adapter.get_msg_by_id = AsyncMock(return_value=ReplyInfo())

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["metadata"]["message_id"] == "1001"
        assert call_kwargs["metadata"]["reply_message_id"] == "6283"
        assert "reply_user_id" not in call_kwargs["metadata"]
        assert "reply_user_name" not in call_kwargs["metadata"]
        assert "reply_content" not in call_kwargs["metadata"]
        adapter.get_msg_by_id.assert_called_once_with(event, "6283")

    @pytest.mark.asyncio
    async def test_handle_with_unavailable_buffer(self):
        """测试 L1 Buffer 不可用时的消息处理"""
        event = MagicMock()
        event.message_str = "你好"

        buffer = MagicMock()
        buffer.is_available = False
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = None

        with ExitStack() as stack:
            for p in _patch_handle_deps():
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_with_empty_content(self):
        """测试消息内容为空时的处理"""
        event = MagicMock()
        event.message_str = ""

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        with ExitStack() as stack:
            for p in _patch_handle_deps():
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_with_none_content(self):
        """测试消息内容为 None 时的处理"""
        event = MagicMock()
        event.message_str = None

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        with ExitStack() as stack:
            for p in _patch_handle_deps():
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_not_called()


class TestUpdateL1Buffer:
    """测试 L1 Buffer 更新函数"""

    @pytest.mark.asyncio
    async def test_update_with_user_message(self):
        """测试添加用户消息"""
        event = MagicMock()

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_session_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"

        with patch("iris_memory.platform.get_adapter", return_value=adapter):
            await update_l1_buffer(event, component_manager, "user", "你好")

        buffer.add_message.assert_called_once_with(
            group_id="group123",
            role="user",
            content="你好",
            source="user456",
            persona_id="default",
        )

    @pytest.mark.asyncio
    async def test_update_with_assistant_message(self):
        """测试添加助手消息"""
        event = MagicMock()

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_session_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"

        with patch("iris_memory.platform.get_adapter", return_value=adapter):
            await update_l1_buffer(event, component_manager, "assistant", "你好！")

        buffer.add_message.assert_called_once_with(
            group_id="group123",
            role="assistant",
            content="你好！",
            source="assistant",
            persona_id="default",
        )

    @pytest.mark.asyncio
    async def test_update_with_unavailable_buffer(self):
        """测试 L1 Buffer 不可用时不添加消息"""
        event = MagicMock()

        buffer = MagicMock()
        buffer.is_available = False
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = None

        await update_l1_buffer(event, component_manager, "user", "你好")

        buffer.add_message.assert_not_called()


class TestBackfillReplyFromBuffer:
    """测试从 L1 Buffer 回填回复信息"""

    def test_backfill_found_in_buffer(self):
        """测试从 L1 Buffer 中找到被回复消息并回填"""
        original_msg = ContextMessage(
            role="user",
            content="你好啊",
            timestamp=datetime.now(),
            token_count=3,
            source="1234567",
            metadata={"message_id": "6283", "user_name": "张三"},
        )

        buffer = MagicMock()
        buffer.get_context.return_value = [original_msg]

        metadata = {}
        _backfill_reply_from_buffer(buffer, "group123", "6283", metadata)

        assert metadata["reply_content"] == "你好啊"
        assert metadata["reply_user_name"] == "张三"

    def test_backfill_not_found_in_buffer(self):
        """测试 L1 Buffer 中找不到被回复消息时不回填"""
        other_msg = ContextMessage(
            role="user",
            content="其他消息",
            timestamp=datetime.now(),
            token_count=2,
            source="other",
            metadata={"message_id": "9999"},
        )

        buffer = MagicMock()
        buffer.get_context.return_value = [other_msg]

        metadata = {}
        _backfill_reply_from_buffer(buffer, "group123", "6283", metadata)

        assert "reply_content" not in metadata
        assert "reply_user_name" not in metadata

    def test_backfill_skips_already_filled(self):
        """测试已有 reply_content 时不覆盖"""
        original_msg = ContextMessage(
            role="user",
            content="你好啊",
            timestamp=datetime.now(),
            token_count=3,
            source="1234567",
            metadata={"message_id": "6283", "user_name": "张三"},
        )

        buffer = MagicMock()
        buffer.get_context.return_value = [original_msg]

        metadata = {"reply_content": "已有内容", "reply_user_name": "已有名字"}
        _backfill_reply_from_buffer(buffer, "group123", "6283", metadata)

        assert metadata["reply_content"] == "已有内容"
        assert metadata["reply_user_name"] == "已有名字"

    def test_backfill_no_user_name_in_original(self):
        """测试被回复消息没有 user_name 时不回填 reply_user_name"""
        original_msg = ContextMessage(
            role="user",
            content="你好啊",
            timestamp=datetime.now(),
            token_count=3,
            source="1234567",
            metadata={"message_id": "6283"},
        )

        buffer = MagicMock()
        buffer.get_context.return_value = [original_msg]

        metadata = {}
        _backfill_reply_from_buffer(buffer, "group123", "6283", metadata)

        assert metadata["reply_content"] == "你好啊"
        assert "reply_user_name" not in metadata

    def test_backfill_includes_assistant_messages(self):
        """测试匹配 assistant 消息的 message_id（用户可能回复 bot）"""
        assistant_msg = ContextMessage(
            role="assistant",
            content="你好！",
            timestamp=datetime.now(),
            token_count=3,
            source="assistant",
            metadata={"message_id": "6283"},
        )

        buffer = MagicMock()
        buffer.get_context.return_value = [assistant_msg]

        metadata = {}
        _backfill_reply_from_buffer(buffer, "group123", "6283", metadata)

        assert metadata["reply_content"] == "你好！"

    def test_backfill_exception_handled(self):
        """测试 get_context 抛异常时不崩溃"""
        buffer = MagicMock()
        buffer.get_context.side_effect = Exception("buffer error")

        metadata = {}
        _backfill_reply_from_buffer(buffer, "group123", "6283", metadata)

        assert "reply_content" not in metadata


class TestHandleWithReplyBackfill:
    """测试回复消息入队时自动回填"""

    @pytest.mark.asyncio
    async def test_handle_reply_backfill_from_buffer(self):
        """测试平台未提供 reply_content 时从 L1 Buffer 回填"""
        event = MagicMock()
        event.message_str = "我也觉得"

        original_msg = ContextMessage(
            role="user",
            content="你好啊",
            timestamp=datetime.now(),
            token_count=3,
            source="1234567",
            metadata={"message_id": "6283", "user_name": "张三"},
        )

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()
        buffer.get_context.return_value = [original_msg]

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "李四"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "1000"}
        adapter.get_reply_info.return_value = ReplyInfo(message_id="6283")
        adapter.get_msg_by_id = AsyncMock(return_value=ReplyInfo())

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["metadata"]["reply_message_id"] == "6283"
        assert call_kwargs["metadata"]["reply_content"] == "你好啊"
        assert call_kwargs["metadata"]["reply_user_name"] == "张三"
        adapter.get_msg_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_reply_no_backfill_when_content_provided(self):
        """测试平台已提供 reply_content 时不触发回填"""
        event = MagicMock()
        event.message_str = "我也觉得"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()
        buffer.get_context.return_value = []

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "李四"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "1000"}
        adapter.get_reply_info.return_value = ReplyInfo(
            message_id="6283", content="平台提供的内容"
        )

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["metadata"]["reply_content"] == "平台提供的内容"
        buffer.get_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_id_not_stored_when_empty(self):
        """测试 message_id 为空时不存入 metadata"""
        event = MagicMock()
        event.message_str = "你好"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "测试用户"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {}
        adapter.get_reply_info.return_value = ReplyInfo()

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert "message_id" not in call_kwargs["metadata"]


class TestHandleWithApiBackfill:
    """测试 L1 Buffer 回填失败后通过平台 API 回填"""

    @pytest.mark.asyncio
    async def test_api_backfill_success(self):
        """测试 L1 Buffer 无结果但 API 回填成功"""
        event = MagicMock()
        event.message_str = "是的"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()
        buffer.get_context.return_value = []

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "李四"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "1001"}
        adapter.get_reply_info.return_value = ReplyInfo(message_id="6283")
        adapter.get_msg_by_id = AsyncMock(
            return_value=ReplyInfo(
                message_id="6283",
                user_id="1234567",
                user_name="张三",
                content="你好啊",
            )
        )

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["metadata"]["reply_message_id"] == "6283"
        assert call_kwargs["metadata"]["reply_content"] == "你好啊"
        assert call_kwargs["metadata"]["reply_user_name"] == "张三"
        assert call_kwargs["metadata"]["reply_user_id"] == "1234567"
        adapter.get_msg_by_id.assert_called_once_with(event, "6283")

    @pytest.mark.asyncio
    async def test_api_backfill_partial(self):
        """测试 API 只返回部分信息时正确回填"""
        event = MagicMock()
        event.message_str = "是的"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()
        buffer.get_context.return_value = []

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "李四"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "1001"}
        adapter.get_reply_info.return_value = ReplyInfo(message_id="6283")
        adapter.get_msg_by_id = AsyncMock(
            return_value=ReplyInfo(
                message_id="6283",
                content="你好啊",
            )
        )

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["metadata"]["reply_content"] == "你好啊"
        assert "reply_user_name" not in call_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_api_backfill_failure_keeps_degraded(self):
        """测试 API 也失败时 metadata 中只有 reply_message_id"""
        event = MagicMock()
        event.message_str = "是的"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()
        buffer.get_context.return_value = []

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = ""
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "1001"}
        adapter.get_reply_info.return_value = ReplyInfo(message_id="6283")
        adapter.get_msg_by_id = AsyncMock(return_value=ReplyInfo())

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["metadata"]["reply_message_id"] == "6283"
        assert "reply_content" not in call_kwargs["metadata"]
        assert "reply_user_name" not in call_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_api_backfill_does_not_override_existing(self):
        """测试 API 回填不覆盖已有字段"""
        event = MagicMock()
        event.message_str = "是的"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()
        buffer.get_context.return_value = []

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "李四"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "1001"}
        adapter.get_reply_info.return_value = ReplyInfo(
            message_id="6283",
            user_name="平台提供的名字",
        )
        adapter.get_msg_by_id = AsyncMock(
            return_value=ReplyInfo(
                message_id="6283",
                user_name="API返回的名字",
                content="API返回的内容",
            )
        )

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["metadata"]["reply_user_name"] == "平台提供的名字"
        assert call_kwargs["metadata"]["reply_content"] == "API返回的内容"


def _make_image_item(
    image_hash: str, url: str = "", file_path: str = ""
) -> ImageQueueItem:
    """构造一个 PENDING 状态的图片队列项"""
    info = ImageInfo(url=url or None, file_path=file_path or None)
    return ImageQueueItem(
        image_hash=image_hash,
        image_url=url,
        image_info=info,
        group_id="group_test",
    )


def _build_all_mode_deps(pending_images, parse_results, *, with_cache=True):
    """构造 _parse_images_if_enabled 所需的全部 mock 依赖。

    返回 (component_manager, l1_buffer, cache_manager, quota_manager)，
    parse_results 用于 stub ImageParser.parse_batch。
    """
    config = MagicMock()
    config.get = lambda key, default=None: {
        "l1_buffer.image_parsing.enable": True,
        "l1_buffer.image_parsing.mode": "all",
        "image_max_parse_per_request": 10,
        "l1_buffer.image_parsing.provider": "",
    }.get(key, default)

    adapter = MagicMock()
    adapter.get_group_id = MagicMock(return_value="group_test")

    l1_buffer = MagicMock()
    l1_buffer.get_images = MagicMock(return_value=list(pending_images))
    l1_buffer.mark_image_parsed = MagicMock(return_value=True)
    l1_buffer.replace_image_placeholder = MagicMock(return_value=True)

    cache_manager = None
    if with_cache:
        cache_manager = MagicMock()
        cache_manager.is_available = True
        cache_manager.get_cache = AsyncMock(return_value=None)
        cache_manager.set_cache = AsyncMock(return_value=None)

    quota_manager = MagicMock()
    quota_manager.is_available = True
    quota_manager.check_quota = AsyncMock(return_value=True)
    quota_manager.use_quota = AsyncMock(return_value=True)
    quota_manager.release_quota = AsyncMock(return_value=None)

    llm_manager = MagicMock()
    llm_manager.is_available = True

    component_manager = MagicMock()
    component_manager.get_available_component = lambda name: {
        "l1_buffer": l1_buffer,
        "image_cache": cache_manager,
        "image_quota": quota_manager,
        "llm_manager": llm_manager,
    }.get(name)

    return config, adapter, component_manager, l1_buffer, cache_manager, quota_manager


class TestParseImagesAllModeIndexAlignment:
    """回归：all 模式图片解析结果下标与 images_to_parse 对齐

    历史 bug：image_infos 按 has_url 过滤（更短），parse_batch 按它返回结果，
    但回写用 parse_results[i] 索引未过滤的 images_to_parse[i]，存在仅 file_path
    无 url 的图片时两列表错位，解析结果归属错误 img_item，缓存写入（以 image_hash
    为键）与 mark_image_parsed 全部错位，被过滤项还永久残留 [IMG:...] 占位符。
    """

    @pytest.mark.asyncio
    async def test_mixed_url_and_no_url_alignment(self):
        """仅 file_path 无 url 的图片在前，有 url 的在后：结果不得错位"""
        # img_nourl: 仅 file_path，无 url —— 会被过滤出 image_infos
        img_nourl = _make_image_item("hash_nourl", file_path="/tmp/a.jpg")
        # img_url: 有 url —— 唯一可解析项
        img_url = _make_image_item("hash_url", url="https://x.com/b.jpg")

        # parse_batch 仅收到 img_url 的 info，返回一条成功结果
        parse_results = [
            ParseResult(content="这是 URL 图片的描述", success=True),
        ]

        (
            config,
            adapter,
            component_manager,
            l1_buffer,
            cache_manager,
            quota_manager,
        ) = _build_all_mode_deps([img_nourl, img_url], parse_results)

        with (
            patch("iris_memory.config.get_config", return_value=config),
            patch("iris_memory.platform.get_adapter", return_value=adapter),
            patch("iris_memory.image.ImageParser") as MockParser,
            patch(
                "iris_memory.image.recorder_bridge.get_recorder_bridge",
                return_value=None,
            ),
        ):
            MockParser.return_value.parse_batch = AsyncMock(return_value=parse_results)
            await _parse_images_if_enabled(MagicMock(), component_manager)

        # 关键断言：img_url 的解析结果写入 img_url 的缓存，不是 img_nourl 的
        cache_manager.set_cache.assert_awaited_once()
        cache_entry = cache_manager.set_cache.call_args[0][0]
        assert cache_entry.image_hash == "hash_url", (
            "解析结果应归属有 URL 的图片，不得错位到无 URL 图片"
        )
        assert cache_entry.content == "这是 URL 图片的描述"

        # img_nourl 被标记 FAILED 并清占位符（不留残留）
        mark_calls = l1_buffer.mark_image_parsed.call_args_list
        marked = {c[0][1]: c[0][2] for c in mark_calls}
        assert marked.get("hash_nourl") == ImageParseStatus.FAILED
        assert marked.get("hash_url") == ImageParseStatus.SUCCESS

        # img_nourl 的占位符被清空
        replace_calls = l1_buffer.replace_image_placeholder.call_args_list
        replaced = {c[0][1]: c[0][2] for c in replace_calls}
        assert replaced.get("[IMG:hash_nourl]") == "", "无 URL 项占位符应清空"
        assert replaced.get("[IMG:hash_url]") == "[图:这是 URL 图片的描述]", (
            "有 URL 项占位符应替换为解析结果"
        )

    @pytest.mark.asyncio
    async def test_all_have_url_no_misalignment(self):
        """对照：全部有 URL 时不应错位（无过滤发生）"""
        img1 = _make_image_item("hash_1", url="https://x.com/1.jpg")
        img2 = _make_image_item("hash_2", url="https://x.com/2.jpg")

        parse_results = [
            ParseResult(content="描述一", success=True),
            ParseResult(content="描述二", success=True),
        ]

        (
            config,
            adapter,
            component_manager,
            l1_buffer,
            cache_manager,
            quota_manager,
        ) = _build_all_mode_deps([img1, img2], parse_results)

        with (
            patch("iris_memory.config.get_config", return_value=config),
            patch("iris_memory.platform.get_adapter", return_value=adapter),
            patch("iris_memory.image.ImageParser") as MockParser,
            patch(
                "iris_memory.image.recorder_bridge.get_recorder_bridge",
                return_value=None,
            ),
        ):
            MockParser.return_value.parse_batch = AsyncMock(return_value=parse_results)
            await _parse_images_if_enabled(MagicMock(), component_manager)

        cache_calls = cache_manager.set_cache.call_args_list
        hashes = [c[0][0].image_hash for c in cache_calls]
        assert hashes == ["hash_1", "hash_2"], "全有 URL 时顺序应保持"

    @pytest.mark.asyncio
    async def test_all_no_url_skips_parse_and_releases_quota(self):
        """全部无 URL：不调 parse_batch，标记 FAILED，退还配额"""
        img1 = _make_image_item("hash_1", file_path="/tmp/1.jpg")
        img2 = _make_image_item("hash_2", file_path="/tmp/2.jpg")

        (
            config,
            adapter,
            component_manager,
            l1_buffer,
            cache_manager,
            quota_manager,
        ) = _build_all_mode_deps([img1, img2], parse_results=[])

        with (
            patch("iris_memory.config.get_config", return_value=config),
            patch("iris_memory.platform.get_adapter", return_value=adapter),
            patch("iris_memory.image.ImageParser") as MockParser,
            patch(
                "iris_memory.image.recorder_bridge.get_recorder_bridge",
                return_value=None,
            ),
        ):
            MockParser.return_value.parse_batch = AsyncMock(return_value=[])
            await _parse_images_if_enabled(MagicMock(), component_manager)

        MockParser.return_value.parse_batch.assert_not_awaited()
        # 两项均标记 FAILED
        mark_calls = l1_buffer.mark_image_parsed.call_args_list
        marked = {c[0][1]: c[0][2] for c in mark_calls}
        assert marked.get("hash_1") == ImageParseStatus.FAILED
        assert marked.get("hash_2") == ImageParseStatus.FAILED
        # 退还预扣配额
        quota_manager.release_quota.assert_awaited_once_with(2)


class TestQueueImagesPersonaId:
    """回归：占位消息（prepend 失败时新建）必须携带 persona_id

    历史 bug：_queue_images_to_l1_buffer 在 prepend 失败时新建占位消息
    add_message 不传 persona_id（默认 default），且从不调 resolve_persona。
    该占位常是最后入队消息，buffer.py 用 messages[-1].persona_id 决定
    画像与 L2 摘要归属，default 占位会污染人格命名空间。
    """

    @pytest.mark.asyncio
    async def test_placeholder_message_carries_persona_id(self):
        """prepend 失败时新建的占位消息必须透传 persona_id"""
        from pathlib import Path

        config = MagicMock()
        config.get = lambda key, default=None: {
            "l1_buffer.image_parsing.enable": True,
            "image_phash_enable": False,
            "image_filter_enable": False,
        }.get(key, default)
        config.data_dir = Path("/tmp/iris_test")

        adapter = MagicMock()
        adapter.get_images = MagicMock(
            return_value=[ImageInfo(url="https://x.com/img.jpg")]
        )
        adapter.get_group_id = MagicMock(return_value="group_test")
        adapter.get_user_id = MagicMock(return_value="user_1")
        adapter.get_raw_message = MagicMock(return_value={"message_id": "mid_1"})
        adapter.get_user_name = MagicMock(return_value="测试用户")

        l1_buffer = MagicMock()
        l1_buffer.get_all_phash_hashes = MagicMock(return_value=[])
        l1_buffer.add_image = MagicMock()
        # prepend 失败 → 触发 add_message 占位路径
        l1_buffer.prepend_to_last_message = MagicMock(return_value=False)
        l1_buffer.add_message = AsyncMock(return_value=True)

        cache_manager = MagicMock()
        cache_manager.is_available = True
        cache_manager.get_cache = AsyncMock(return_value=None)

        component_manager = MagicMock()
        component_manager.get_available_component = lambda name: {
            "l1_buffer": l1_buffer,
            "image_cache": cache_manager,
        }.get(name)

        with (
            patch("iris_memory.config.get_config", return_value=config),
            patch("iris_memory.platform.get_adapter", return_value=adapter),
            patch(
                "iris_memory.core.persona.resolve_persona",
                new=AsyncMock(return_value="yuki"),
            ),
            patch(
                "iris_memory.image.image_utils.compute_image_hash",
                new=AsyncMock(return_value="testhash123456"),
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            # 避免 httpx 真实下载
            MockClient.side_effect = RuntimeError("mock: no network")
            await _queue_images_to_l1_buffer(MagicMock(), component_manager)

        # 核心断言：占位消息 add_message 携带 persona_id="yuki"
        l1_buffer.add_message.assert_awaited_once()
        call_kwargs = l1_buffer.add_message.call_args[1]
        assert call_kwargs["persona_id"] == "yuki", (
            "占位消息必须透传 persona_id，不得默认 default 污染人格命名空间"
        )
        # 占位内容是图片占位符
        assert "[IMG:" in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_prepend_success_no_add_message(self):
        """对照：prepend 成功时不新建占位消息"""
        from pathlib import Path

        config = MagicMock()
        config.get = lambda key, default=None: {
            "l1_buffer.image_parsing.enable": True,
            "image_phash_enable": False,
            "image_filter_enable": False,
        }.get(key, default)
        config.data_dir = Path("/tmp/iris_test")

        adapter = MagicMock()
        adapter.get_images = MagicMock(
            return_value=[ImageInfo(url="https://x.com/img.jpg")]
        )
        adapter.get_group_id = MagicMock(return_value="group_test")
        adapter.get_user_id = MagicMock(return_value="user_1")
        adapter.get_raw_message = MagicMock(return_value={"message_id": "mid_1"})
        adapter.get_user_name = MagicMock(return_value="测试用户")

        l1_buffer = MagicMock()
        l1_buffer.get_all_phash_hashes = MagicMock(return_value=[])
        l1_buffer.add_image = MagicMock()
        # prepend 成功 → 不触发 add_message
        l1_buffer.prepend_to_last_message = MagicMock(return_value=True)
        l1_buffer.add_message = AsyncMock(return_value=True)

        cache_manager = MagicMock()
        cache_manager.is_available = True
        cache_manager.get_cache = AsyncMock(return_value=None)

        component_manager = MagicMock()
        component_manager.get_available_component = lambda name: {
            "l1_buffer": l1_buffer,
            "image_cache": cache_manager,
        }.get(name)

        with (
            patch("iris_memory.config.get_config", return_value=config),
            patch("iris_memory.platform.get_adapter", return_value=adapter),
            patch(
                "iris_memory.core.persona.resolve_persona",
                new=AsyncMock(return_value="yuki"),
            ),
            patch(
                "iris_memory.image.image_utils.compute_image_hash",
                new=AsyncMock(return_value="testhash123456"),
            ),
            patch("httpx.AsyncClient") as MockClient,
        ):
            MockClient.side_effect = RuntimeError("mock: no network")
            await _queue_images_to_l1_buffer(MagicMock(), component_manager)

        l1_buffer.add_message.assert_not_awaited()


class TestPrivateChatSessionKey:
    """私聊会话键端到端（私聊 L1 队列隔离）

    私聊事件 group_id 为空字符串，L1 写入必须使用
    private:{user_id} 会话键，而非空字符串。
    """

    @pytest.mark.asyncio
    async def test_private_message_uses_private_session_key(self):
        """私聊消息经 handle_user_message 写入 private:{user_id} 队列"""
        event = MagicMock()
        event.message_str = "你好"

        buffer = MagicMock()
        buffer.is_available = True
        buffer.add_message = AsyncMock()

        component_manager = MagicMock()
        component_manager.get_component.return_value = buffer
        component_manager.get_available_component.return_value = buffer

        adapter = MagicMock()
        adapter.get_group_id.return_value = ""  # 私聊：群 ID 为空字符串
        adapter.get_session_id.return_value = "private:user456"
        adapter.get_user_id.return_value = "user456"
        adapter.get_user_name.return_value = "私聊用户"
        adapter.get_group_name.return_value = ""
        adapter.get_raw_message.return_value = {"message_id": "888"}
        adapter.get_reply_info.return_value = ReplyInfo()

        with ExitStack() as stack:
            for p in _patch_handle_deps(adapter=adapter):
                stack.enter_context(p)
            await handle_user_message(event, component_manager)

        buffer.add_message.assert_called_once()
        call_kwargs = buffer.add_message.call_args[1]
        assert call_kwargs["group_id"] == "private:user456"
        assert call_kwargs["role"] == "user"
        assert call_kwargs["source"] == "user456"

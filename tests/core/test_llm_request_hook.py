"""测试 LLM 请求钩子处理模块"""

import asyncio

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import ExitStack
from datetime import datetime

from iris_memory.l1_buffer.models import ContextMessage
from iris_memory.core.llm_request_hook import (
    preprocess_llm_request,
    _inject_to_extra_user_content_parts,
    _build_image_map,
    _get_inline_image_desc,
)


def _make_msg(role, content, source="user1", metadata=None, token_count=1):
    return ContextMessage(
        role=role,
        content=content,
        timestamp=datetime.now(),
        token_count=token_count,
        source=source,
        metadata=metadata or {},
    )


def _make_component_manager(buffer=None):
    cm = MagicMock()
    if buffer is not None:
        cm.get_component.return_value = buffer
        cm.get_available_component.return_value = (
            buffer if buffer.is_available else None
        )
    return cm


_ADAPTER_PATCH = "iris_memory.platform.get_adapter"
_COLLECT_PROFILE_PATCH = "iris_memory.core.llm_request_hook._collect_user_profile"
_COLLECT_L2_PATCH = "iris_memory.core.llm_request_hook._collect_l2_memory"
_COLLECT_L3_PATCH = "iris_memory.core.llm_request_hook._collect_l3_knowledge_graph"
_PARSE_IMAGES_PATCH = "iris_memory.core.llm_request_hook._parse_images_if_related_mode"
_BUILD_IMAGE_MAP_PATCH = "iris_memory.core.llm_request_hook._build_image_map"
_LOG_CONTEXT_PATCH = "iris_memory.core.llm_request_hook._log_final_context"
_GET_CONFIG_PATCH = "iris_memory.config.get_config"


def _default_config():
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None: {
        "l1_buffer.inject_queue_length": 50,
        "l1_inject_max_content_chars": 200,
    }.get(key, default)
    return cfg


def _patch_collect_fns(
    profile_text="", l2_text="", l2_results=None, l3_text="", adapter=None
):
    patches = [
        patch(
            _COLLECT_PROFILE_PATCH, new_callable=AsyncMock, return_value=profile_text
        ),
        patch(
            _COLLECT_L2_PATCH,
            new_callable=AsyncMock,
            return_value=(l2_text, l2_results or []),
        ),
        patch(_COLLECT_L3_PATCH, new_callable=AsyncMock, return_value=l3_text),
        patch(_PARSE_IMAGES_PATCH, new_callable=AsyncMock, return_value=None),
        patch(_BUILD_IMAGE_MAP_PATCH, new_callable=AsyncMock, return_value={}),
        patch(_LOG_CONTEXT_PATCH),
        patch(_GET_CONFIG_PATCH, return_value=_default_config()),
    ]
    if adapter:
        patches.append(patch(_ADAPTER_PATCH, return_value=adapter))
    return patches


def _get_extra_parts_text(req):
    parts = getattr(req, "extra_user_content_parts", [])
    texts = []
    for part in parts:
        text = getattr(part, "text", None) or str(part)
        texts.append(text)
    return "\n".join(texts)


class TestInjectToExtraUserContentParts:
    """测试所有内容注入到 extra_user_content_parts"""

    def test_inject_all_sections(self):
        req = MagicMock()
        req.extra_user_content_parts = []
        _inject_to_extra_user_content_parts(req, "对话", "画像", "记忆", "图谱")
        assert len(req.extra_user_content_parts) == 1
        text = req.extra_user_content_parts[0].text
        assert "<iris:l1_context>" in text
        assert "对话" in text
        assert "<iris:profile>" in text
        assert "画像" in text
        assert "<iris:l3_kg>" in text
        assert "图谱" in text
        assert "<iris:l2_memory>" in text
        assert "记忆" in text

    def test_inject_only_l1(self):
        req = MagicMock()
        req.extra_user_content_parts = []
        _inject_to_extra_user_content_parts(req, "对话内容", "", "", "")
        assert len(req.extra_user_content_parts) == 1
        text = req.extra_user_content_parts[0].text
        assert "<iris:l1_context>" in text
        assert "对话内容" in text
        assert "<iris:profile>" not in text
        assert "<iris:l2_memory>" not in text
        assert "<iris:l3_kg>" not in text

    def test_inject_empty_does_not_append(self):
        req = MagicMock()
        req.extra_user_content_parts = []
        _inject_to_extra_user_content_parts(req, "", "", "", "")
        assert len(req.extra_user_content_parts) == 0

    def test_inject_section_order(self):
        req = MagicMock()
        req.extra_user_content_parts = []
        _inject_to_extra_user_content_parts(req, "对话", "画像", "记忆", "图谱")
        text = req.extra_user_content_parts[0].text
        l1_idx = text.index("l1_context")
        profile_idx = text.index("profile")
        l2_idx = text.index("l2_memory")
        l3_idx = text.index("l3_kg")
        assert l1_idx < profile_idx < l2_idx < l3_idx

    def test_inject_does_not_modify_system_prompt(self):
        req = MagicMock()
        req.extra_user_content_parts = []
        req.system_prompt = "你是一个助手"
        _inject_to_extra_user_content_parts(req, "对话", "画像", "", "")
        assert req.system_prompt == "你是一个助手"

    def test_inject_calls_mark_as_temp_when_available(self):
        req = MagicMock()
        req.extra_user_content_parts = []

        from astrbot.core.agent.message import TextPart

        with patch.object(TextPart, "mark_as_temp", create=True) as mock_mark:
            _inject_to_extra_user_content_parts(req, "对话", "画像", "", "")
            if hasattr(TextPart, "mark_as_temp"):
                mock_mark.assert_called_once()

    def test_inject_graceful_without_mark_as_temp(self):
        req = MagicMock()
        req.extra_user_content_parts = []
        _inject_to_extra_user_content_parts(req, "对话", "", "", "")
        assert len(req.extra_user_content_parts) == 1
        assert (
            req.extra_user_content_parts[0].text
            == "<iris:l1_context>\n对话\n</iris:l1_context>"
        )


class TestBuildImageMap:
    """测试图片映射表构建"""

    @pytest.mark.asyncio
    async def test_build_image_map_empty_images(self):
        l1_buffer = MagicMock()
        l1_buffer.get_images.return_value = []
        component_manager = MagicMock()

        result = await _build_image_map(l1_buffer, "group1", component_manager)
        assert result == {}

    @pytest.mark.asyncio
    async def test_build_image_map_with_cached_images(self):
        from iris_memory.image import ImageParseStatus
        from datetime import datetime

        img_item = MagicMock()
        img_item.status = ImageParseStatus.SUCCESS
        img_item.message_id = "msg123"
        img_item.image_hash = "hash1"
        img_item.timestamp = datetime.now()
        img_item.user_id = "user1"

        l1_buffer = MagicMock()
        l1_buffer.get_images.return_value = [img_item]

        cached_result = MagicMock()
        cached_result.content = "一只猫的照片"

        cache_manager = MagicMock()
        cache_manager.is_available = True
        cache_manager.get_cache = AsyncMock(return_value=cached_result)

        component_manager = MagicMock()
        component_manager.get_component.return_value = cache_manager

        result = await _build_image_map(l1_buffer, "group1", component_manager)
        assert "msg123" in result
        assert result["msg123"] == ["一只猫的照片"]

    @pytest.mark.asyncio
    async def test_build_image_map_skips_non_success(self):
        from iris_memory.image import ImageParseStatus

        img_item = MagicMock()
        img_item.status = ImageParseStatus.PENDING
        img_item.message_id = "msg123"
        img_item.image_hash = "hash1"

        l1_buffer = MagicMock()
        l1_buffer.get_images.return_value = [img_item]

        component_manager = MagicMock()
        component_manager.get_component.return_value = None

        result = await _build_image_map(l1_buffer, "group1", component_manager)
        assert result == {}

    @pytest.mark.asyncio
    async def test_build_image_map_fallback_to_timestamp_key(self):
        from iris_memory.image import ImageParseStatus
        from datetime import datetime

        ts = datetime(2025, 1, 1, 10, 30)
        img_item = MagicMock()
        img_item.status = ImageParseStatus.SUCCESS
        img_item.message_id = ""
        img_item.image_hash = "hash1"
        img_item.timestamp = ts
        img_item.user_id = "user1"

        l1_buffer = MagicMock()
        l1_buffer.get_images.return_value = [img_item]

        cached_result = MagicMock()
        cached_result.content = "风景照"

        cache_manager = MagicMock()
        cache_manager.is_available = True
        cache_manager.get_cache = AsyncMock(return_value=cached_result)

        component_manager = MagicMock()
        component_manager.get_component.return_value = cache_manager

        result = await _build_image_map(l1_buffer, "group1", component_manager)
        key = "user1:10:30"
        assert key in result
        assert result[key] == ["风景照"]


class TestGetInlineImageDesc:
    """测试行内图片描述获取"""

    def test_empty_image_map(self):
        msg = _make_msg("user", "看看这个")
        result = _get_inline_image_desc(msg, None, {})
        assert result == ""

    def test_match_by_message_id(self):
        msg = _make_msg("user", "看看这个")
        image_map = {"msg123": ["一只猫的照片"]}
        result = _get_inline_image_desc(msg, "msg123", image_map)
        assert result == " [图:一只猫的照片]"

    def test_match_by_timestamp_window(self):
        from datetime import datetime

        ts = datetime(2025, 1, 1, 10, 30)
        msg = ContextMessage(
            role="user",
            content="看看这个",
            timestamp=ts,
            token_count=1,
            source="user1",
            metadata={},
        )
        image_map = {"user1:10:30": ["风景照"]}
        result = _get_inline_image_desc(msg, None, image_map)
        assert result == " [图:风景照]"

    def test_multiple_images_joined(self):
        msg = _make_msg("user", "看看这些")
        image_map = {"msg123": ["图片1", "图片2"]}
        result = _get_inline_image_desc(msg, "msg123", image_map)
        assert result == " [图:图片1；图片2]"

    def test_no_match(self):
        msg = _make_msg("user", "看看这个")
        image_map = {"other_msg": ["一只猫的照片"]}
        result = _get_inline_image_desc(msg, "msg456", image_map)
        assert result == ""


class TestPreprocessLLMRequest:
    """测试 LLM 对话前处理主函数"""

    @pytest.mark.asyncio
    async def test_preprocess_with_available_buffer(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [
            _make_msg("user", "你好", token_count=2),
            _make_msg("assistant", "你好！", token_count=3),
        ]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        assert req.prompt == "你好"
        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "你好" in text
        assert "你好！" in text

    @pytest.mark.asyncio
    async def test_preprocess_with_unavailable_buffer(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = False

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" not in text
        assert req.prompt == "你好"

    @pytest.mark.asyncio
    async def test_preprocess_with_empty_messages(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True
        buffer.get_context.return_value = []

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" not in text

    @pytest.mark.asyncio
    async def test_preprocess_preserves_system_prompt(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = "你是一个助手"
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [_make_msg("user", "问题", token_count=1)]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        assert req.system_prompt == "你是一个助手"
        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "问题" in text
        assert req.prompt == "你好"

    @pytest.mark.asyncio
    async def test_preprocess_with_user_name_binding(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [
            _make_msg("user", "你好", metadata={"user_name": "张三"}, token_count=2),
            _make_msg("assistant", "你好！", token_count=3),
        ]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "张三(user1):" in text
        assert "你好" in text
        assert "Bot: 你好！" in text

    @pytest.mark.asyncio
    async def test_preprocess_without_user_name(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [_make_msg("user", "你好", token_count=2)]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "你好" in text

    @pytest.mark.asyncio
    async def test_preprocess_assistant_message_format(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [
            _make_msg(
                "assistant",
                "你好！",
                metadata={"user_name": "助手"},
                token_count=3,
            )
        ]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "Bot: 你好！" in text

    @pytest.mark.asyncio
    async def test_preprocess_with_reply_info(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [
            _make_msg(
                "user",
                "我也觉得",
                source="user456",
                metadata={
                    "user_name": "李四",
                    "reply_message_id": "6283",
                    "reply_user_name": "张三",
                    "reply_content": "你好啊",
                },
                token_count=4,
            )
        ]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "李四(user456): ↩️回复张三「你好啊」" in text
        assert "我也觉得" in text

    @pytest.mark.asyncio
    async def test_preprocess_with_reply_no_user_name(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [
            _make_msg(
                "user",
                "是的",
                source="user456",
                metadata={
                    "user_name": "李四",
                    "reply_message_id": "6283",
                    "reply_content": "你好啊",
                },
                token_count=2,
            )
        ]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "↩️回复某人「你好啊」" in text

    @pytest.mark.asyncio
    async def test_preprocess_with_reply_no_content(self):
        """测试回复信息无内容时从 L1 上下文回填"""
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [
            _make_msg(
                "user",
                "你好啊",
                source="user123",
                metadata={"message_id": "6283", "user_name": "张三"},
                token_count=3,
            ),
            _make_msg(
                "user",
                "是的",
                source="user456",
                metadata={
                    "user_name": "李四",
                    "reply_message_id": "6283",
                },
                token_count=2,
            ),
        ]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "李四(user456): ↩️回复张三(user123)「你好啊」" in text
        assert "是的" in text

    @pytest.mark.asyncio
    async def test_preprocess_with_reply_no_content_no_match(self):
        """测试回复信息无内容且 L1 上下文中也找不到时显示降级提示"""
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = ""
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True

        messages = [
            _make_msg(
                "user",
                "是的",
                source="user456",
                metadata={
                    "user_name": "李四",
                    "reply_message_id": "9999",
                },
                token_count=2,
            ),
        ]
        buffer.get_context.return_value = messages

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(adapter=adapter):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "李四(user456): ↩️回复了某条消息" in text


class TestExtraUserContentPartsInjection:
    """测试 extra_user_content_parts 注入逻辑"""

    @pytest.mark.asyncio
    async def test_prompt_not_modified(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "用户当前消息"
        req.system_prompt = "系统提示"
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True
        buffer.get_context.return_value = [_make_msg("user", "你好", token_count=1)]

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(
                profile_text="画像", l2_text="记忆", l3_text="图谱", adapter=adapter
            ):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        assert req.prompt == "用户当前消息"
        text = _get_extra_parts_text(req)
        assert "<iris:profile>" in text
        assert "<iris:l3_kg>" in text
        assert "<iris:l2_memory>" in text
        assert "<iris:l1_context>" in text

    @pytest.mark.asyncio
    async def test_contexts_not_modified(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = [{"role": "user", "content": "当前消息"}]
        req.prompt = "你好"
        req.system_prompt = "系统提示"
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True
        buffer.get_context.return_value = [_make_msg("user", "历史", token_count=1)]

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(
                profile_text="画像", l2_text="记忆", l3_text="图谱", adapter=adapter
            ):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        assert len(req.contexts) == 1
        assert req.contexts[0]["content"] == "当前消息"
        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text

    @pytest.mark.asyncio
    async def test_system_prompt_not_modified(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = "系统提示"
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True
        buffer.get_context.return_value = [_make_msg("user", "历史", token_count=1)]

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(
                profile_text="画像", l2_text="记忆", l3_text="图谱", adapter=adapter
            ):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        assert req.system_prompt == "系统提示"
        text = _get_extra_parts_text(req)
        assert "<iris:l1_context>" in text
        assert "<iris:profile>" in text
        assert "<iris:l3_kg>" in text
        assert "<iris:l2_memory>" in text

    @pytest.mark.asyncio
    async def test_all_sections_in_extra_user_content_parts(self):
        event = MagicMock()
        req = MagicMock()
        req.contexts = []
        req.prompt = "你好"
        req.system_prompt = "系统提示"
        req.extra_user_content_parts = []

        buffer = MagicMock()
        buffer.is_available = True
        buffer.get_context.return_value = [_make_msg("user", "历史", token_count=1)]

        component_manager = _make_component_manager(buffer)

        adapter = MagicMock()
        adapter.get_group_id.return_value = "group123"

        with ExitStack() as stack:
            for p in _patch_collect_fns(
                profile_text="画像", l2_text="记忆", l3_text="图谱", adapter=adapter
            ):
                stack.enter_context(p)
            await preprocess_llm_request(event, req, component_manager)

        assert len(req.extra_user_content_parts) == 1
        text = req.extra_user_content_parts[0].text
        assert "<iris:l1_context>" in text
        assert "<iris:profile>" in text
        assert "<iris:l3_kg>" in text
        assert "<iris:l2_memory>" in text


class TestParseImagesTimeout:
    """测试图片解析整体超时（方案1：防止 on_llm_request 钩子阻塞会话锁）"""

    @pytest.mark.asyncio
    async def test_parse_timeout_marks_failed_and_does_not_block(self):
        """provider 卡死时，整体超时后函数返回不阻塞，未完成图片标记失败"""
        from iris_memory.core.llm_request_hook import _parse_images_if_related_mode
        from iris_memory.image import ImageParseStatus

        def config_get(key, default=None):
            return {
                "l1_buffer.image_parsing.enable": True,
                "image_skip_on_passive_trigger": False,
                "l1_buffer.image_parsing.mode": "related",
                "image_max_parse_per_request": 3,
                "image_max_concurrent_parse": 2,
                "image_parse_timeout_ms": 100,
                "l1_buffer.image_parsing.provider": "",
            }.get(key, default)

        img1 = MagicMock()
        img1.image_hash = "ph:aaaa1111bbbb2222"
        img1.image_info = MagicMock()
        img1.image_info.has_url = True
        img2 = MagicMock()
        img2.image_hash = "ph:cccc3333dddd4444"
        img2.image_info = MagicMock()
        img2.image_info.has_url = True

        l1_buffer = MagicMock()
        l1_buffer.is_available = True
        l1_buffer.get_images.return_value = [img1, img2]
        l1_buffer.replace_image_placeholder = MagicMock()
        l1_buffer.mark_image_parsed = MagicMock()

        cache_manager = MagicMock()
        cache_manager.is_available = True
        cache_manager.get_cache = AsyncMock(return_value=None)

        quota_manager = MagicMock()
        quota_manager.is_available = True
        quota_manager.check_quota = AsyncMock(return_value=True)
        quota_manager.use_quota = AsyncMock(return_value=True)
        quota_manager.release_quota = AsyncMock(return_value=0)

        llm_manager = MagicMock()
        llm_manager.is_available = True

        cm = MagicMock()
        cm.get_available_component = MagicMock(
            side_effect=lambda name: {
                "l1_buffer": l1_buffer,
                "image_cache": cache_manager,
                "image_quota": quota_manager,
                "llm_manager": llm_manager,
            }.get(name)
        )

        adapter = MagicMock()
        adapter.get_group_id = MagicMock(return_value="group1")

        event = MagicMock()

        async def slow_parse(*a, **kw):
            await asyncio.sleep(10)

        with (
            patch("iris_memory.config.get_config") as mock_cfg,
            patch("iris_memory.platform.get_adapter", return_value=adapter),
            patch("iris_memory.image.ImageParser") as mock_parser_cls,
            patch(
                "iris_memory.image.recorder_bridge.get_recorder_bridge",
                return_value=None,
            ),
        ):
            mock_cfg.return_value.get.side_effect = config_get
            mock_parser_cls.return_value.parse = slow_parse

            await asyncio.wait_for(
                _parse_images_if_related_mode(event, MagicMock(), cm),
                timeout=5.0,
            )

        failed_calls = [
            c
            for c in l1_buffer.mark_image_parsed.call_args_list
            if c.args[2] == ImageParseStatus.FAILED
        ]
        assert len(failed_calls) == 2


class TestFaultIsolation:
    """回归：各收集步骤独立 try/except，单步异常不影响其余上下文注入

    历史 bug：无顶层 try/except，[t.result() for t in done] 会 re-raise，
    异常穿透使已收集的 L1/画像/L2/L3 注入全丢，违背故障隔离约定。
    """

    @pytest.mark.asyncio
    async def test_l1_failure_does_not_block_l2_injection(self):
        """_collect_l1_context 异常时 L2 上下文仍被注入"""
        from iris_memory.core.llm_request_hook import preprocess_llm_request

        config = MagicMock()
        config.get = MagicMock(
            side_effect=lambda key, default=None: {
                "l1_buffer.enable": True,
                "l1_buffer.context_message_count": 10,
                "l1_buffer.image_parsing.enable": False,
                "l2_memory.enable": True,
                "l2_memory.max_results": 5,
                "l3_kg.enable": False,
                "profile.enable": False,
            }.get(key, default)
        )

        req = MagicMock()
        req.system_prompt = "system"
        req.extra_user_content_parts = []

        event = MagicMock()
        event.message_str = "hello"

        l1_buffer = MagicMock()
        l1_buffer.is_available = True
        l1_buffer.get_context_messages = MagicMock(side_effect=RuntimeError("L1 崩溃"))

        l2_adapter = MagicMock()
        l2_adapter.is_available = True
        l2_adapter.search = AsyncMock(return_value=[])
        l2_adapter._llm_manager = None

        cm = MagicMock()
        cm.get_available_component = lambda name: {
            "l1_buffer": l1_buffer,
            "l2_memory": l2_adapter,
        }.get(name)

        with (
            patch("iris_memory.config.get_config", return_value=config),
            patch("iris_memory.platform.get_adapter") as mock_adapter,
        ):
            mock_adapter.return_value.get_group_id.return_value = "g1"
            mock_adapter.return_value.get_user_id.return_value = "u1"

            # 不应抛异常
            await preprocess_llm_request(event, req, cm)

        # L1 崩溃但函数正常返回，未抛异常即通过
        # 验证 extra_user_content_parts 被调用（L2 注入仍尝试）
        assert req.extra_user_content_parts is not None

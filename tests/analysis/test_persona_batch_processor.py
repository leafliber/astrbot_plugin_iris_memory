"""
PersonaBatchProcessor 单元测试

测试覆盖:
- 消息入队与队列管理
- 数量触发器
- 时间触发器（flush loop）
- 批量提取与结果应用
- 消息合并策略
- 容量保护
- 序列化/反序列化
- 生命周期管理
- 统计数据
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.analysis.persona.persona_batch_processor import (
    PersonaBatchProcessor,
    PersonaQueuedMessage,
    PersonaBatchStats,
)
from iris_memory.analysis.persona.keyword_maps import ExtractionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_extractor():
    """创建模拟的 PersonaExtractor"""
    extractor = AsyncMock()
    extractor.extract = AsyncMock(return_value=ExtractionResult(
        interests={"编程": 0.8},
        social_style="外向",
        confidence=0.7,
        source="rule",
    ))
    return extractor


@pytest.fixture
def mock_empty_extractor():
    """创建返回空结果的模拟 PersonaExtractor"""
    extractor = AsyncMock()
    extractor.extract = AsyncMock(return_value=ExtractionResult(
        confidence=0.0,
        source="rule",
    ))
    return extractor


@pytest.fixture
def apply_callback():
    """创建结果应用回调"""
    callback = MagicMock()
    return callback


@pytest.fixture
def processor(mock_extractor, apply_callback):
    """创建标准测试处理器（阈值=3，刷新间隔=300s）"""
    return PersonaBatchProcessor(
        persona_extractor=mock_extractor,
        batch_threshold=3,
        flush_interval=300,
        batch_max_size=5,
        apply_result_callback=apply_callback,
    )


@pytest.fixture
def low_threshold_processor(mock_extractor, apply_callback):
    """创建低阈值处理器（阈值=2）"""
    return PersonaBatchProcessor(
        persona_extractor=mock_extractor,
        batch_threshold=2,
        flush_interval=300,
        batch_max_size=5,
        apply_result_callback=apply_callback,
    )


# ---------------------------------------------------------------------------
# PersonaQueuedMessage 测试
# ---------------------------------------------------------------------------

class TestPersonaQueuedMessage:
    """队列消息测试"""

    def test_basic_creation(self):
        msg = PersonaQueuedMessage(
            content="我喜欢编程",
            user_id="user1",
            memory_type="fact",
            confidence=0.8,
        )
        assert msg.content == "我喜欢编程"
        assert msg.user_id == "user1"
        assert msg.memory_type == "fact"
        assert msg.confidence == 0.8
        assert msg.group_id is None
        assert msg.enqueue_time > 0

    def test_content_truncation(self):
        long_content = "a" * 1000
        msg = PersonaQueuedMessage(content=long_content, user_id="user1")
        assert len(msg.content) == PersonaQueuedMessage.MAX_CONTENT_LENGTH

    def test_serialization_roundtrip(self):
        msg = PersonaQueuedMessage(
            content="测试内容",
            summary="摘要",
            memory_type="fact",
            confidence=0.9,
            memory_id="mem-123",
            user_id="user1",
            group_id="group1",
        )
        d = msg.to_dict()
        restored = PersonaQueuedMessage.from_dict(d)

        assert restored.content == msg.content
        assert restored.summary == msg.summary
        assert restored.memory_type == msg.memory_type
        assert restored.confidence == msg.confidence
        assert restored.memory_id == msg.memory_id
        assert restored.user_id == msg.user_id
        assert restored.group_id == msg.group_id

    def test_from_dict_with_defaults(self):
        msg = PersonaQueuedMessage.from_dict({"content": "hi"})
        assert msg.content == "hi"
        assert msg.user_id == ""
        assert msg.confidence == 0.5
        assert msg.memory_id is None


# ---------------------------------------------------------------------------
# PersonaBatchStats 测试
# ---------------------------------------------------------------------------

class TestPersonaBatchStats:
    """统计数据测试"""

    def test_default_values(self):
        stats = PersonaBatchStats()
        assert stats.messages_enqueued == 0
        assert stats.messages_processed == 0
        assert stats.batches_processed == 0
        assert stats.llm_calls == 0

    def test_serialization_roundtrip(self):
        stats = PersonaBatchStats(
            messages_enqueued=10,
            messages_processed=8,
            batches_processed=3,
            llm_calls=3,
            extraction_errors=1,
        )
        d = stats.to_dict()
        restored = PersonaBatchStats.from_dict(d)

        assert restored.messages_enqueued == 10
        assert restored.messages_processed == 8
        assert restored.extraction_errors == 1


# ---------------------------------------------------------------------------
# PersonaBatchProcessor 测试
# ---------------------------------------------------------------------------

class TestAddMessage:
    """消息入队测试"""

    @pytest.mark.asyncio
    async def test_add_single_message(self, processor):
        result = await processor.add_message(
            content="我喜欢看书",
            user_id="user1",
            memory_type="fact",
        )
        assert result is False  # 未达到阈值
        assert processor.pending_count == 1
        assert processor._stats.messages_enqueued == 1

    @pytest.mark.asyncio
    async def test_add_message_with_group(self, processor):
        await processor.add_message(
            content="群聊消息",
            user_id="user1",
            group_id="group1",
            memory_type="interaction",
        )
        assert processor.pending_count == 1

    @pytest.mark.asyncio
    async def test_session_isolation(self, processor):
        """私聊和群聊消息应在不同队列"""
        await processor.add_message(
            content="私聊消息", user_id="user1", memory_type="fact"
        )
        await processor.add_message(
            content="群聊消息", user_id="user1", group_id="group1", memory_type="fact"
        )
        assert processor.pending_count == 2
        assert len(processor._queues) == 2


class TestCountTrigger:
    """数量触发器测试"""

    @pytest.mark.asyncio
    async def test_trigger_at_threshold(self, processor, mock_extractor, apply_callback):
        """达到阈值时应触发批量处理"""
        await processor.add_message(content="msg1", user_id="u1", memory_type="fact")
        await processor.add_message(content="msg2", user_id="u1", memory_type="fact")
        result = await processor.add_message(content="msg3", user_id="u1", memory_type="fact")

        assert result is True  # 阈值=3，第3条触发
        assert processor.pending_count == 0  # 处理完队列清空
        mock_extractor.extract.assert_called_once()
        assert apply_callback.call_count == 3  # 每条消息回调一次
        assert processor._stats.batches_processed == 1
        assert processor._stats.messages_processed == 3

    @pytest.mark.asyncio
    async def test_no_trigger_below_threshold(self, processor, mock_extractor):
        """未达到阈值不应触发"""
        await processor.add_message(content="msg1", user_id="u1", memory_type="fact")
        await processor.add_message(content="msg2", user_id="u1", memory_type="fact")

        assert processor.pending_count == 2
        mock_extractor.extract.assert_not_called()


class TestBatchExtraction:
    """批量提取测试"""

    @pytest.mark.asyncio
    async def test_extraction_with_merged_content(self, mock_extractor, apply_callback):
        """多条消息应合并为格式化文本"""
        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
            batch_max_size=10,
            apply_result_callback=apply_callback,
        )

        await proc.add_message(content="我喜欢编程", user_id="u1", memory_type="fact")
        await proc.add_message(content="最近在学Python", user_id="u1", memory_type="fact")

        # 验证传给 extractor 的是合并后的文本
        call_args = mock_extractor.extract.call_args
        content_arg = call_args.kwargs.get("content") or call_args[1].get("content", call_args[0][0] if call_args[0] else "")
        assert "[1]" in content_arg
        assert "[2]" in content_arg
        assert "我喜欢编程" in content_arg
        assert "最近在学Python" in content_arg

    @pytest.mark.asyncio
    async def test_single_message_not_numbered(self, mock_extractor, apply_callback):
        """单条消息不加序号"""
        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=1,
            flush_interval=300,
            apply_result_callback=apply_callback,
        )

        await proc.add_message(content="单条消息", user_id="u1", memory_type="fact")

        call_args = mock_extractor.extract.call_args
        content_arg = call_args.kwargs.get("content") or call_args[0][0]
        assert content_arg == "单条消息"
        assert "[1]" not in content_arg

    @pytest.mark.asyncio
    async def test_extraction_failure_handled(self, apply_callback):
        """提取失败不影响后续处理"""
        failing_extractor = AsyncMock()
        failing_extractor.extract = AsyncMock(side_effect=RuntimeError("LLM down"))

        proc = PersonaBatchProcessor(
            persona_extractor=failing_extractor,
            batch_threshold=2,
            flush_interval=300,
            apply_result_callback=apply_callback,
        )

        await proc.add_message(content="msg1", user_id="u1", memory_type="fact")
        await proc.add_message(content="msg2", user_id="u1", memory_type="fact")

        assert proc.pending_count == 0  # 队列已清空
        assert proc._stats.extraction_errors == 1
        apply_callback.assert_not_called()  # 提取失败，不应调用回调

    @pytest.mark.asyncio
    async def test_empty_result_skips_callback(self, mock_empty_extractor, apply_callback):
        """空提取结果不应调用回调"""
        proc = PersonaBatchProcessor(
            persona_extractor=mock_empty_extractor,
            batch_threshold=2,
            flush_interval=300,
            apply_result_callback=apply_callback,
        )

        await proc.add_message(content="随便说", user_id="u1", memory_type="fact")
        await proc.add_message(content="无意义", user_id="u1", memory_type="fact")

        apply_callback.assert_not_called()


class TestBatchMaxSize:
    """批次大小限制测试"""

    @pytest.mark.asyncio
    async def test_exceeds_max_size_splits(self, mock_extractor, apply_callback):
        """超出单次最大批次大小时分批"""
        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=8,
            flush_interval=300,
            batch_max_size=3,
            apply_result_callback=apply_callback,
        )

        # 添加8条消息触发处理
        for i in range(8):
            await proc.add_message(
                content=f"msg{i}", user_id="u1", memory_type="fact"
            )

        # 第一次处理最多3条，剩余5条留在队列
        assert mock_extractor.extract.call_count == 1
        assert proc.pending_count == 5  # 8-3=5


class TestQueueCapacityProtection:
    """队列容量保护测试"""

    @pytest.mark.asyncio
    async def test_force_flush_at_max_capacity(self, mock_extractor, apply_callback):
        """队列满时强制刷新"""
        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=100,  # 设置很高不会自然触发
            flush_interval=300,
            batch_max_size=50,
            apply_result_callback=apply_callback,
        )

        # 填满队列 + 1 条消息触发容量保护
        for i in range(proc.MAX_QUEUE_SIZE + 1):
            await proc.add_message(
                content=f"msg{i}", user_id="u1", memory_type="fact"
            )

        # 应该已经强制刷新过（容量保护在第51条消息入队时触发）
        assert mock_extractor.extract.call_count >= 1


class TestFlushLoop:
    """时间触发器测试"""

    @pytest.mark.asyncio
    async def test_flush_all_queues(self, processor, mock_extractor, apply_callback):
        """手动刷新所有队列"""
        await processor.add_message(content="msg1", user_id="u1", memory_type="fact")
        await processor.add_message(content="msg2", user_id="u2", memory_type="fact")

        assert processor.pending_count == 2

        await processor._flush_all_queues()

        assert processor.pending_count == 0
        assert mock_extractor.extract.call_count == 2  # 两个会话各处理一次


class TestLifecycle:
    """生命周期测试"""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, processor):
        await processor.start()
        assert processor.is_running is True
        assert processor._flush_task is not None

        await processor.stop()
        assert processor.is_running is False
        assert processor._flush_task is None

    @pytest.mark.asyncio
    async def test_stop_flushes_remaining(self, processor, mock_extractor):
        """停止时应处理剩余队列"""
        await processor.add_message(content="msg1", user_id="u1", memory_type="fact")

        await processor.start()
        await processor.stop()

        mock_extractor.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_start_idempotent(self, processor):
        await processor.start()
        await processor.start()  # 不应报错
        assert processor.is_running is True
        await processor.stop()

    @pytest.mark.asyncio
    async def test_double_stop_idempotent(self, processor):
        await processor.start()
        await processor.stop()
        await processor.stop()  # 不应报错


class TestSerialization:
    """序列化测试"""

    @pytest.mark.asyncio
    async def test_serialize_empty(self, processor):
        data = await processor.serialize_queues()
        assert data["queues"] == {}
        assert "stats" in data

    @pytest.mark.asyncio
    async def test_serialize_with_messages(self, processor):
        await processor.add_message(
            content="test", user_id="u1", memory_type="fact",
            memory_id="mem-1",
        )
        data = await processor.serialize_queues()
        assert len(data["queues"]) == 1

        # 检验序列化内容
        queue_key = list(data["queues"].keys())[0]
        msgs = data["queues"][queue_key]
        assert len(msgs) == 1
        assert msgs[0]["content"] == "test"
        assert msgs[0]["memory_id"] == "mem-1"

    @pytest.mark.asyncio
    async def test_deserialize_and_clear(self, processor):
        """反序列化后应清空队列，保留统计"""
        data = {
            "queues": {
                "u1:private": [
                    {"content": "old msg", "user_id": "u1", "memory_type": "fact"},
                ],
            },
            "stats": {
                "messages_enqueued": 10,
                "messages_processed": 8,
                "batches_processed": 3,
                "llm_calls": 3,
                "extraction_errors": 0,
                "messages_discarded": 0,
            },
        }
        await processor.deserialize_and_clear(data)

        # 队列应被清空
        assert processor.pending_count == 0
        # 统计应恢复
        assert processor._stats.messages_enqueued == 10
        assert processor._stats.messages_processed == 8
        # 应记录被丢弃的消息
        assert processor._stats.messages_discarded == 1


class TestMergeStrategies:
    """消息合并策略测试"""

    def test_merge_single_message(self):
        msgs = [PersonaQueuedMessage(content="单条", user_id="u1")]
        result = PersonaBatchProcessor._merge_messages(msgs)
        assert result == "单条"

    def test_merge_multiple_messages(self):
        msgs = [
            PersonaQueuedMessage(content="第一条", user_id="u1"),
            PersonaQueuedMessage(content="第二条", user_id="u1"),
            PersonaQueuedMessage(content="第三条", user_id="u1"),
        ]
        result = PersonaBatchProcessor._merge_messages(msgs)
        assert "[1] 第一条" in result
        assert "[2] 第二条" in result
        assert "[3] 第三条" in result

    def test_merge_summaries_none(self):
        msgs = [PersonaQueuedMessage(content="a", user_id="u1")]
        result = PersonaBatchProcessor._merge_summaries(msgs)
        assert result is None

    def test_merge_summaries_single(self):
        msgs = [PersonaQueuedMessage(content="a", summary="摘要1", user_id="u1")]
        result = PersonaBatchProcessor._merge_summaries(msgs)
        assert result == "摘要1"

    def test_merge_summaries_multiple(self):
        msgs = [
            PersonaQueuedMessage(content="a", summary="摘要1", user_id="u1"),
            PersonaQueuedMessage(content="b", summary="摘要2", user_id="u1"),
        ]
        result = PersonaBatchProcessor._merge_summaries(msgs)
        assert "摘要1" in result
        assert "摘要2" in result
        assert "|" in result


class TestGetStats:
    """统计信息测试"""

    @pytest.mark.asyncio
    async def test_get_stats_initial(self, processor):
        stats = processor.get_stats()
        assert stats["messages_enqueued"] == 0
        assert stats["pending_queues"] == 0
        assert stats["total_pending"] == 0
        assert stats["is_running"] is False

    @pytest.mark.asyncio
    async def test_get_stats_after_messages(self, processor):
        await processor.add_message(content="msg1", user_id="u1", memory_type="fact")
        await processor.add_message(content="msg2", user_id="u2", memory_type="fact")

        stats = processor.get_stats()
        assert stats["messages_enqueued"] == 2
        assert stats["pending_queues"] == 2
        assert stats["total_pending"] == 2


class TestAsyncCallback:
    """异步回调测试"""

    @pytest.mark.asyncio
    async def test_async_callback_supported(self, mock_extractor):
        """支持异步回调函数"""
        callback = AsyncMock()

        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=1,
            flush_interval=300,
            apply_result_callback=callback,
        )

        await proc.add_message(content="msg1", user_id="u1", memory_type="fact")

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_crash(self, mock_extractor):
        """回调失败不影响处理器"""
        failing_callback = MagicMock(side_effect=RuntimeError("callback error"))

        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
            apply_result_callback=failing_callback,
        )

        await proc.add_message(content="msg1", user_id="u1", memory_type="fact")
        await proc.add_message(content="msg2", user_id="u1", memory_type="fact")

        # 处理器不应崩溃
        assert proc.pending_count == 0
        assert proc._stats.batches_processed == 1


class TestNoCallback:
    """无回调测试"""

    @pytest.mark.asyncio
    async def test_works_without_callback(self, mock_extractor):
        """不设置回调也能正常工作"""
        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
        )

        await proc.add_message(content="msg1", user_id="u1", memory_type="fact")
        await proc.add_message(content="msg2", user_id="u1", memory_type="fact")

        assert proc.pending_count == 0
        assert proc._stats.batches_processed == 1


class TestWorkingMemoryCallback:
    """工作记忆查询回调测试"""

    @pytest.mark.asyncio
    async def test_working_memory_callback_sync(self, mock_extractor):
        """测试同步工作记忆回调"""
        # 模拟工作记忆
        mock_memories = [
            MagicMock(
                content="工作记忆内容1",
                summary="摘要1",
                type=MagicMock(value="fact"),
                created_time=1000,
            ),
            MagicMock(
                content="工作记忆内容2",
                summary=None,
                type="emotion",
                created_time=2000,
            ),
        ]

        wm_callback = MagicMock(return_value=mock_memories)

        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
            working_memory_callback=wm_callback,
        )

        # 查询工作记忆
        memories = await proc.get_working_memory_for_session("user1", "group1")

        assert len(memories) == 2
        wm_callback.assert_called_once_with("user1", "group1")

    @pytest.mark.asyncio
    async def test_working_memory_callback_async(self, mock_extractor):
        """测试异步工作记忆回调"""
        mock_memories = [MagicMock(content="async memory", summary=None, type="fact")]
        wm_callback = AsyncMock(return_value=mock_memories)

        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
            working_memory_callback=wm_callback,
        )

        memories = await proc.get_working_memory_for_session("user1", None)

        assert len(memories) == 1
        wm_callback.assert_called_once_with("user1", None)

    @pytest.mark.asyncio
    async def test_working_memory_callback_none(self, mock_extractor):
        """测试未设置回调时返回空列表"""
        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
        )

        memories = await proc.get_working_memory_for_session("user1", None)

        assert memories == []

    @pytest.mark.asyncio
    async def test_working_memory_callback_error(self, mock_extractor):
        """测试回调异常时返回空列表"""
        wm_callback = MagicMock(side_effect=RuntimeError("查询失败"))

        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
            working_memory_callback=wm_callback,
        )

        memories = await proc.get_working_memory_for_session("user1", None)

        assert memories == []

    @pytest.mark.asyncio
    async def test_get_working_memory_context(self, mock_extractor):
        """测试获取工作记忆上下文文本"""
        mock_memories = [
            MagicMock(
                content="这是很长的内容" * 10,
                summary="简短摘要",
                type=MagicMock(value="fact"),
                created_time=2000,
            ),
            MagicMock(
                content="普通内容",
                summary=None,
                type="emotion",
                created_time=1000,
            ),
        ]
        wm_callback = MagicMock(return_value=mock_memories)

        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
            working_memory_callback=wm_callback,
        )

        context = await proc.get_working_memory_context("user1", "group1", max_memories=2)

        # 验证上下文格式
        assert "[1] [fact] 简短摘要" in context
        assert "[2] [emotion] 普通内容" in context
        # 验证内容被截断
        assert len(context) < 500

    @pytest.mark.asyncio
    async def test_get_working_memory_context_empty(self, mock_extractor):
        """测试无工作记忆时返回空字符串"""
        wm_callback = MagicMock(return_value=[])

        proc = PersonaBatchProcessor(
            persona_extractor=mock_extractor,
            batch_threshold=2,
            flush_interval=300,
            working_memory_callback=wm_callback,
        )

        context = await proc.get_working_memory_context("user1", None)

        assert context == ""

"""总结器测试"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from iris_memory.l1_buffer import Summarizer, SegmentedMessageQueue, ContextMessage
from iris_memory.l1_buffer.summarizer import (
    parse_summary_response,
    confidence_to_float,
)


@pytest.fixture
def mock_llm_manager():
    manager = AsyncMock()
    manager.generate_direct = AsyncMock(return_value="这是一个总结")
    return manager


@pytest.fixture
def mock_messages():
    messages = []
    for i in range(5):
        msg = ContextMessage(
            role="user" if i % 2 == 0 else "assistant",
            content=f"消息{i}",
            timestamp=datetime.now(),
            token_count=10,
            source="user_456",
        )
        messages.append(msg)
    return messages


@pytest.fixture
def mock_queue():
    queue = SegmentedMessageQueue(
        group_id="group_123",
        segment_1_length=2,
        segment_3_length=2,
        total_length=8,
    )
    for i in range(8):
        queue.add_message(
            ContextMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"消息{i}",
                timestamp=datetime.now(),
                token_count=10,
                source="user_456",
            )
        )
    return queue


class TestSummarizer:
    def test_create_summarizer(self, mock_llm_manager):
        summarizer = Summarizer(llm_manager=mock_llm_manager)

        assert summarizer.llm_manager == mock_llm_manager
        assert summarizer.provider == ""

    def test_create_summarizer_with_provider(self, mock_llm_manager):
        summarizer = Summarizer(llm_manager=mock_llm_manager, provider="gpt-4o-mini")

        assert summarizer.provider == "gpt-4o-mini"

    def test_should_summarize_when_full(self, mock_queue):
        with patch("iris_memory.l1_buffer.summarizer.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_max_queue_tokens": 10000,
                }.get(key)
            )
            mock_get_config.return_value = mock_config

            summarizer = Summarizer(llm_manager=Mock())

            assert summarizer.should_summarize(mock_queue)

    def test_should_summarize_by_tokens(self, mock_queue):
        with patch("iris_memory.l1_buffer.summarizer.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_max_queue_tokens": 40,
                }.get(key)
            )
            mock_get_config.return_value = mock_config

            summarizer = Summarizer(llm_manager=Mock())

            assert summarizer.should_summarize(mock_queue)

    def test_should_not_summarize(self):
        queue = SegmentedMessageQueue(
            group_id="g1",
            segment_1_length=5,
            segment_3_length=3,
            total_length=20,
        )
        for i in range(3):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=10,
                    source="user",
                )
            )

        with patch("iris_memory.l1_buffer.summarizer.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_max_queue_tokens": 10000,
                }.get(key)
            )
            mock_get_config.return_value = mock_config

            summarizer = Summarizer(llm_manager=Mock())

            assert not summarizer.should_summarize(queue)

    @pytest.mark.asyncio
    async def test_summarize_messages(self, mock_llm_manager, mock_messages):
        summarizer = Summarizer(llm_manager=mock_llm_manager)

        summary = await summarizer.summarize(
            context_messages=mock_messages, target_messages=mock_messages
        )

        assert summary == "这是一个总结"
        assert mock_llm_manager.generate_direct.called

    @pytest.mark.asyncio
    async def test_summarize_empty_target(self, mock_llm_manager, mock_messages):
        summarizer = Summarizer(llm_manager=mock_llm_manager)

        summary = await summarizer.summarize(
            context_messages=mock_messages, target_messages=[]
        )

        assert summary is None

    def test_build_summary_prompt(self, mock_llm_manager):
        summarizer = Summarizer(llm_manager=mock_llm_manager)

        context_messages = [
            ContextMessage(
                role="user",
                content="你好",
                timestamp=datetime.now(),
                token_count=2,
                source="user_001",
                metadata={"user_name": "张三"},
            ),
            ContextMessage(
                role="assistant",
                content="你好！",
                timestamp=datetime.now(),
                token_count=3,
                source="bot",
            ),
        ]

        target_messages = context_messages

        prompt = summarizer._build_summary_prompt(context_messages, target_messages)

        assert "[张三]: 你好" in prompt
        assert "[助手]: 你好！" in prompt
        assert "保守" in prompt
        assert "memories" in prompt
        assert "完整对话上下文" in prompt
        assert "需要总结的对话片段" in prompt
        assert "confidence" in prompt

    def test_build_summary_prompt_format(self, mock_llm_manager):
        summarizer = Summarizer(llm_manager=mock_llm_manager)

        messages = [
            ContextMessage(
                role="user",
                content="我喜欢吃苹果",
                timestamp=datetime.now(),
                token_count=5,
                source="user_001",
                metadata={"user_name": "张三"},
            ),
            ContextMessage(
                role="assistant",
                content="好的，我记住了",
                timestamp=datetime.now(),
                token_count=5,
                source="bot",
            ),
        ]

        prompt = summarizer._build_summary_prompt(messages, messages)

        assert "信息价值" in prompt
        assert "独立完整" in prompt
        assert "非即时性" in prompt
        assert "确定性" in prompt
        assert "JSON" in prompt
        assert "宁缺毋滥" in prompt
        assert "high" in prompt
        assert "medium" in prompt
        assert "low" in prompt

    def test_build_summary_prompt_with_user_names(self, mock_llm_manager):
        summarizer = Summarizer(llm_manager=mock_llm_manager)

        messages = [
            ContextMessage(
                role="user",
                content="我喜欢吃苹果",
                timestamp=datetime.now(),
                token_count=5,
                source="user_001",
                metadata={"user_name": "张三"},
            ),
            ContextMessage(
                role="user",
                content="我喜欢编程",
                timestamp=datetime.now(),
                token_count=5,
                source="user_002",
                metadata={"user_name": "李四"},
            ),
        ]

        prompt = summarizer._build_summary_prompt(messages, messages)

        assert "[张三]: 我喜欢吃苹果" in prompt
        assert "[李四]: 我喜欢编程" in prompt

    def test_build_summary_prompt_without_user_name(self, mock_llm_manager):
        summarizer = Summarizer(llm_manager=mock_llm_manager)

        messages = [
            ContextMessage(
                role="user",
                content="你好",
                timestamp=datetime.now(),
                token_count=2,
                source="user_001",
            )
        ]

        prompt = summarizer._build_summary_prompt(messages, messages)

        assert "[用户]: 你好" in prompt

    def test_build_summary_prompt_different_context_and_target(self, mock_llm_manager):
        summarizer = Summarizer(llm_manager=mock_llm_manager)

        context_messages = [
            ContextMessage(
                role="user",
                content="旧消息",
                timestamp=datetime.now(),
                token_count=2,
                source="user_001",
            ),
            ContextMessage(
                role="user",
                content="目标消息",
                timestamp=datetime.now(),
                token_count=2,
                source="user_001",
            ),
            ContextMessage(
                role="user",
                content="新消息",
                timestamp=datetime.now(),
                token_count=2,
                source="user_001",
            ),
        ]

        target_messages = [context_messages[1]]

        prompt = summarizer._build_summary_prompt(context_messages, target_messages)

        assert "旧消息" in prompt
        assert "目标消息" in prompt
        assert "新消息" in prompt

    def test_format_messages(self, mock_llm_manager):
        _ = Summarizer(llm_manager=mock_llm_manager)

        messages = [
            ContextMessage(
                role="user",
                content="你好",
                timestamp=datetime.now(),
                token_count=2,
                source="user_001",
                metadata={"user_name": "张三"},
            ),
            ContextMessage(
                role="assistant",
                content="你好！",
                timestamp=datetime.now(),
                token_count=3,
                source="bot",
            ),
        ]

        result = Summarizer._format_messages(messages)

        assert "[张三]: 你好" in result
        assert "[助手]: 你好！" in result


class TestSegmentedQueueSummarization:
    def test_queue_full_triggers_summarize(self):
        queue = SegmentedMessageQueue(
            group_id="g1",
            segment_1_length=2,
            segment_3_length=2,
            total_length=6,
        )

        for i in range(8):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=10,
                    source="user",
                )
            )

        assert queue.is_full()

    def test_target_messages_are_segment_2(self):
        queue = SegmentedMessageQueue(
            group_id="g1",
            segment_1_length=2,
            segment_3_length=2,
            total_length=6,
        )

        for i in range(8):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=10,
                    source="user",
                )
            )

        target = list(queue.segment_2)
        context = queue.all_messages

        assert len(target) == 6
        assert len(context) == 8

    def test_after_rotate_segments_shift(self):
        queue = SegmentedMessageQueue(
            group_id="g1",
            segment_1_length=2,
            segment_3_length=2,
            total_length=6,
        )

        for i in range(8):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=10,
                    source="user",
                )
            )

        old_seg1 = list(queue.segment_1)
        old_seg2 = list(queue.segment_2)

        queue.rotate_after_summary()

        assert list(queue.segment_2) == []
        assert len(queue.segment_3) == 2
        assert list(queue.segment_3) == old_seg2[-2:]
        assert list(queue.segment_1) == old_seg1


class TestParseSummaryResponse:
    def test_parse_new_format_with_confidence(self):
        response = json.dumps(
            {
                "memories": [
                    {"content": "张三是程序员", "confidence": "high"},
                    {"content": "李四可能喜欢摄影", "confidence": "medium"},
                    {"content": "王五好像在忙", "confidence": "low"},
                ]
            }
        )

        result = parse_summary_response(response)

        assert len(result["memories"]) == 3
        assert result["memories"][0] == {
            "content": "张三是程序员",
            "confidence": "high",
        }
        assert result["memories"][1] == {
            "content": "李四可能喜欢摄影",
            "confidence": "medium",
        }
        assert result["memories"][2] == {
            "content": "王五好像在忙",
            "confidence": "low",
        }

    def test_parse_old_format_string_array(self):
        response = json.dumps(
            {
                "memories": [
                    "- 张三是程序员",
                    "李四喜欢摄影",
                ]
            }
        )

        result = parse_summary_response(response)

        assert len(result["memories"]) == 2
        assert result["memories"][0] == {
            "content": "张三是程序员",
            "confidence": "medium",
        }
        assert result["memories"][1] == {
            "content": "李四喜欢摄影",
            "confidence": "medium",
        }

    def test_parse_empty_memories(self):
        response = json.dumps({"memories": []})

        result = parse_summary_response(response)

        assert result["memories"] == []

    def test_parse_fenced_empty_memories(self):
        """小模型常用 ```json 围栏包裹输出，空 memories 必须正常解析为 json_parsed=True。

        回归测试：此前围栏内的 {"memories": []} 虽能解析，但 buffer 侧会误触发行式回退，
        把 ```json 和 "memories": [] 当作记忆导入。
        """
        response = '```json\n{"memories": []}\n```'

        result = parse_summary_response(response)

        assert result["json_parsed"] is True
        assert result["memories"] == []

    def test_parse_fenced_json_with_content(self):
        response = (
            "```json\n"
            '{"memories": [{"content": "张三是程序员", "confidence": "high"}]}\n'
            "```"
        )

        result = parse_summary_response(response)

        assert result["json_parsed"] is True
        assert len(result["memories"]) == 1
        assert result["memories"][0]["content"] == "张三是程序员"
        assert result["memories"][0]["confidence"] == "high"

    def test_parse_fenced_json_uppercase_tag(self):
        response = (
            '```JSON\n{"memories": [{"content": "测试", "confidence": "medium"}]}\n```'
        )

        result = parse_summary_response(response)

        assert result["json_parsed"] is True
        assert len(result["memories"]) == 1

    def test_parse_plain_fenced_no_lang(self):
        response = '```\n{"memories": []}\n```'

        result = parse_summary_response(response)

        assert result["json_parsed"] is True
        assert result["memories"] == []

    def test_parse_fenced_empty_memories_no_garbage(self):
        """围栏空 JSON 不应产生任何记忆（确保剥离围栏后直接解析）。"""
        response = '```json\n{\n  "memories": []\n}\n```'

        result = parse_summary_response(response)

        assert result["json_parsed"] is True
        assert result["memories"] == []
        # 即便误入文本回退，也不应提取到任何行
        assert all("```" not in m.get("content", "") for m in result["memories"])

    def test_parse_empty_response(self):
        result = parse_summary_response("")

        assert result["memories"] == []

    def test_parse_fallback_dash_format(self):
        response = "- 张三是程序员\n- 李四喜欢摄影"

        result = parse_summary_response(response)

        assert len(result["memories"]) == 2
        assert result["memories"][0]["content"] == "张三是程序员"
        assert result["memories"][0]["confidence"] == "medium"

    def test_parse_invalid_confidence_normalized(self):
        response = json.dumps(
            {
                "memories": [
                    {"content": "测试", "confidence": "invalid_value"},
                ]
            }
        )

        result = parse_summary_response(response)

        assert result["memories"][0]["confidence"] == "medium"

    def test_parse_dict_without_content_skipped(self):
        response = json.dumps(
            {
                "memories": [
                    {"content": "", "confidence": "high"},
                    {"content": "有效内容", "confidence": "high"},
                ]
            }
        )

        result = parse_summary_response(response)

        assert len(result["memories"]) == 1
        assert result["memories"][0]["content"] == "有效内容"

    def test_parse_mixed_format(self):
        response = json.dumps(
            {
                "memories": [
                    {"content": "新格式记忆", "confidence": "high"},
                    "旧格式记忆",
                ]
            }
        )

        result = parse_summary_response(response)

        assert len(result["memories"]) == 2
        assert result["memories"][0]["confidence"] == "high"
        assert result["memories"][1]["confidence"] == "medium"


class TestConfidenceToFloat:
    def test_high_confidence(self):
        assert confidence_to_float("high") == 0.85

    def test_medium_confidence(self):
        assert confidence_to_float("medium") == 0.6

    def test_low_confidence(self):
        assert confidence_to_float("low") == 0.35

    def test_unknown_confidence(self):
        assert confidence_to_float("unknown") == 0.5

    def test_empty_confidence(self):
        assert confidence_to_float("") == 0.5

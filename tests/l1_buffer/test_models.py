"""L1 数据模型测试"""

from datetime import datetime
from iris_memory.l1_buffer.models import (
    ContextMessage,
    SegmentedMessageQueue,
    MessageQueue,
)


class TestContextMessage:
    """ContextMessage 测试"""

    def test_create_message(self):
        msg = ContextMessage(
            role="user",
            content="你好",
            timestamp=datetime.now(),
            token_count=2,
            source="group_123",
        )

        assert msg.role == "user"
        assert msg.content == "你好"
        assert msg.token_count == 2
        assert msg.source == "group_123"
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        metadata = {"user_id": "user_456", "nickname": "测试用户"}
        msg = ContextMessage(
            role="user",
            content="测试",
            timestamp=datetime.now(),
            token_count=1,
            source="group_123",
            metadata=metadata,
        )

        assert msg.metadata == metadata

    def test_to_dict(self):
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        msg = ContextMessage(
            role="user",
            content="测试",
            timestamp=timestamp,
            token_count=1,
            source="group_123",
        )

        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "测试"
        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["token_count"] == 1
        assert data["source"] == "group_123"

    def test_from_dict(self):
        data = {
            "role": "assistant",
            "content": "回复",
            "timestamp": "2024-01-01T12:00:00",
            "token_count": 3,
            "source": "assistant",
            "metadata": {"test": "value"},
        }

        msg = ContextMessage.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "回复"
        assert msg.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert msg.token_count == 3
        assert msg.metadata == {"test": "value"}

    def test_persona_id_roundtrip(self):
        """persona_id 序列化往返"""
        msg = ContextMessage(
            role="user",
            content="测试",
            timestamp=datetime(2024, 1, 1),
            token_count=1,
            source="u1",
            persona_id="yuki",
        )
        data = msg.to_dict()
        assert data["persona_id"] == "yuki"

        restored = ContextMessage.from_dict(data)
        assert restored.persona_id == "yuki"

    def test_from_dict_without_persona_defaults(self):
        """旧数据无 persona_id 字段时回退 default"""
        data = {
            "role": "user",
            "content": "x",
            "timestamp": "2024-01-01T12:00:00",
            "token_count": 1,
            "source": "u1",
        }
        msg = ContextMessage.from_dict(data)
        assert msg.persona_id == "default"


class TestSegmentedMessageQueue:
    """SegmentedMessageQueue 三段式队列测试

    L1-1 和 L1-2 组成 FIFO，L1-3 不参与 FIFO 流动。
    """

    def test_create_queue(self):
        queue = SegmentedMessageQueue(group_id="group_123")

        assert queue.group_id == "group_123"
        assert len(queue) == 0
        assert queue.total_tokens == 0
        assert queue.segment_1_length == 10
        assert queue.segment_3_length == 5
        assert queue.total_length == 30
        assert queue.segment_2_length == 20

    def test_create_queue_custom_lengths(self):
        queue = SegmentedMessageQueue(
            group_id="g1",
            segment_1_length=5,
            segment_3_length=3,
            total_length=20,
        )

        assert queue.segment_1_length == 5
        assert queue.segment_3_length == 3
        assert queue.total_length == 20
        assert queue.segment_2_length == 15

    def test_segment_2_length_minimum(self):
        queue = SegmentedMessageQueue(
            group_id="g1",
            segment_1_length=29,
            segment_3_length=5,
            total_length=30,
        )

        assert queue.segment_2_length == 1

    def test_add_message_to_segment_1(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=5, segment_3_length=2, total_length=15
        )
        msg = ContextMessage(
            role="user",
            content="测试",
            timestamp=datetime.now(),
            token_count=5,
            source="user_456",
        )

        queue.add_message(msg)

        assert len(queue.segment_1) == 1
        assert len(queue.segment_2) == 0
        assert len(queue.segment_3) == 0
        assert queue.total_tokens == 5

    def test_overflow_from_seg1_to_seg2(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=3, segment_3_length=2, total_length=10
        )

        for i in range(5):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        assert len(queue.segment_1) == 3
        assert len(queue.segment_2) == 2
        assert len(queue.segment_3) == 0

    def test_no_overflow_to_seg3(self):
        """L1-3 不参与 FIFO 流动，消息不会从 L1-2 溢出到 L1-3"""
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=6
        )

        for i in range(10):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        assert len(queue.segment_1) == 2
        assert len(queue.segment_2) == 8
        assert len(queue.segment_3) == 0

    def test_is_full_when_seg2_full(self):
        """L1-2 满时触发 is_full"""
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=6
        )

        for i in range(8):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        assert len(queue.segment_2) >= queue.segment_2_length
        assert queue.is_full()

    def test_is_not_full(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=8
        )

        for i in range(5):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        assert not queue.is_full()

    def test_all_messages_order(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=8
        )

        for i in range(5):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        all_msgs = queue.all_messages

        assert len(all_msgs) == 5
        assert all_msgs[0].content == "消息0"
        assert all_msgs[-1].content == "消息4"

    def test_inject_messages_excludes_seg3(self):
        """inject_messages 不包含 L1-3 的消息"""
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=8
        )

        for i in range(5):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        queue.rotate_after_summary()

        for i in range(5, 8):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        inject = queue.inject_messages

        seg3_contents = {m.content for m in queue.segment_3}
        inject_contents = {m.content for m in inject}
        assert seg3_contents.isdisjoint(inject_contents)
        assert len(inject) == len(queue.segment_1) + len(queue.segment_2)

    def test_rotate_after_summary(self):
        """总结后：L1-2 最新部分→L1-3，L1-2 清空，L1-1 不动"""
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=6
        )

        for i in range(8):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        assert len(queue.segment_1) == 2
        assert len(queue.segment_2) == 6
        assert len(queue.segment_3) == 0

        queue.rotate_after_summary()

        assert len(queue.segment_1) == 2
        assert len(queue.segment_2) == 0
        assert len(queue.segment_3) == 2

        assert queue.segment_3[0].content == "消息4"
        assert queue.segment_3[1].content == "消息5"

        assert queue.segment_1[0].content == "消息6"
        assert queue.segment_1[1].content == "消息7"

    def test_rotate_after_summary_token_update(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=6
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

        assert queue.total_tokens == 80

        queue.rotate_after_summary()

        assert len(queue.segment_1) == 2
        assert len(queue.segment_3) == 2
        assert queue.total_tokens == 40

    def test_rotate_when_seg2_less_than_seg3_capacity(self):
        """L1-2 消息数少于 L1-3 容量时，全部转入 L1-3"""
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=5, total_length=6
        )

        for i in range(4):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        assert len(queue.segment_2) == 2

        queue.rotate_after_summary()

        assert len(queue.segment_3) == 2
        assert len(queue.segment_2) == 0
        assert len(queue.segment_1) == 2

    def test_clear(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=8
        )

        for i in range(5):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        queue.clear()

        assert len(queue) == 0
        assert queue.total_tokens == 0
        assert len(queue.segment_1) == 0
        assert len(queue.segment_2) == 0
        assert len(queue.segment_3) == 0

    def test_is_empty(self):
        queue = SegmentedMessageQueue(group_id="g1")

        assert queue.is_empty()

        queue.add_message(
            ContextMessage(
                role="user",
                content="测试",
                timestamp=datetime.now(),
                token_count=1,
                source="user",
            )
        )

        assert not queue.is_empty()

    def test_to_message_list(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=1, total_length=6
        )

        queue.add_message(
            ContextMessage(
                role="user",
                content="你好",
                timestamp=datetime.now(),
                token_count=2,
                source="user_456",
            )
        )
        queue.add_message(
            ContextMessage(
                role="assistant",
                content="你好！",
                timestamp=datetime.now(),
                token_count=3,
                source="assistant",
            )
        )

        message_list = queue.to_message_list()

        assert len(message_list) == 2
        assert message_list[0] == {"role": "user", "content": "你好"}
        assert message_list[1] == {"role": "assistant", "content": "你好！"}

    def test_remove_user_messages(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=8
        )

        for i in range(6):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user_A" if i % 2 == 0 else "user_B",
                )
            )

        removed = queue.remove_user_messages("user_A")

        assert removed == 3
        assert all(msg.source != "user_A" for msg in queue.all_messages)

    def test_remove_messages(self):
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=8
        )

        for i in range(5):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        all_msgs = queue.all_messages
        to_remove = all_msgs[:2]

        queue.remove_messages(to_remove)

        assert len(queue) == 3

    def test_message_queue_alias(self):
        assert MessageQueue is SegmentedMessageQueue

    def test_full_cycle_add_summarize_rotate(self):
        """完整周期：添加→满→总结→rotate→继续添加"""
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=6
        )

        for i in range(8):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        assert queue.is_full()

        queue.rotate_after_summary()

        assert len(queue.segment_1) == 2
        assert len(queue.segment_2) == 0
        assert len(queue.segment_3) == 2
        assert queue.total_tokens == 4

        for i in range(8, 12):
            queue.add_message(
                ContextMessage(
                    role="user",
                    content=f"消息{i}",
                    timestamp=datetime.now(),
                    token_count=1,
                    source="user",
                )
            )

        assert len(queue.segment_1) == 2
        assert len(queue.segment_2) == 4
        assert len(queue.segment_3) == 2

        inject = queue.inject_messages
        assert len(inject) == 6
        seg3_contents = {m.content for m in queue.segment_3}
        inject_contents = {m.content for m in inject}
        assert seg3_contents.isdisjoint(inject_contents)

    def test_multiple_rotate_cycles(self):
        """多次 rotate 周期，验证 token 不会出现负数"""
        queue = SegmentedMessageQueue(
            group_id="g1", segment_1_length=2, segment_3_length=2, total_length=6
        )

        for cycle in range(5):
            for i in range(8):
                queue.add_message(
                    ContextMessage(
                        role="user",
                        content=f"周期{cycle}_消息{i}",
                        timestamp=datetime.now(),
                        token_count=10,
                        source="user",
                    )
                )

            assert queue.is_full()
            queue.rotate_after_summary()
            assert queue.total_tokens >= 0

        assert queue.total_tokens > 0

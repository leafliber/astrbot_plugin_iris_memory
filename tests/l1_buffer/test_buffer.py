"""L1 缓冲组件测试"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from iris_memory.l1_buffer import L1Buffer, ContextMessage
from iris_memory.l1_buffer.models import SegmentedMessageQueue
from iris_memory.config import init_config


@pytest.fixture
def mock_config(tmp_path: Path):
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


class TestL1Buffer:
    """L1 缓冲组件测试"""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config):
        """测试初始化成功"""
        buffer = L1Buffer()

        await buffer.initialize()

        assert buffer.is_available
        assert buffer.name == "l1_buffer"

    @pytest.mark.asyncio
    async def test_initialize_disabled(self, mock_config):
        """测试禁用状态初始化"""
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {"l1_buffer.enable": False}.get(
                    key, default
                )
            )

            buffer = L1Buffer()
            await buffer.initialize()

            assert not buffer.is_available

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_config):
        """测试关闭"""
        buffer = L1Buffer()
        await buffer.initialize()

        # 添加一些消息
        await buffer.add_message("group_123", "user", "测试", "user_456")

        await buffer.shutdown()

        assert not buffer.is_available
        assert len(buffer._queues) == 0

    @pytest.mark.asyncio
    async def test_add_message_success(self, mock_config):
        """测试添加消息成功"""
        buffer = L1Buffer()
        await buffer.initialize()

        success = await buffer.add_message(
            group_id="group_123", role="user", content="你好", source="user_456"
        )

        assert success

        context = buffer.get_context("group_123")
        assert len(context) == 1
        assert context[0].content == "你好"

    @pytest.mark.asyncio
    async def test_add_message_too_large(self, mock_config):
        """测试添加超大消息"""
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.enable": True,
                    "l1_max_single_message_tokens": 10,
                }.get(key, default)
            )

            buffer = L1Buffer()
            await buffer.initialize()

            # 创建一个超过限制的消息
            large_content = "这是一条很长的消息" * 100

            success = await buffer.add_message(
                group_id="group_123",
                role="user",
                content=large_content,
                source="user_456",
            )

            assert not success

            context = buffer.get_context("group_123")
            assert len(context) == 0

    @pytest.mark.asyncio
    async def test_add_message_disabled(self, mock_config):
        """测试禁用时添加消息"""
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {"l1_buffer.enable": False}.get(
                    key, default
                )
            )

            buffer = L1Buffer()
            await buffer.initialize()

            success = await buffer.add_message(
                group_id="group_123", role="user", content="测试", source="user_456"
            )

            assert not success

    @pytest.mark.asyncio
    async def test_get_context_with_limit(self, mock_config):
        """测试获取有限制的上下文"""
        buffer = L1Buffer()
        await buffer.initialize()

        # 添加 10 条消息
        for i in range(10):
            await buffer.add_message(
                group_id="group_123", role="user", content=f"消息{i}", source="user_456"
            )

        # 获取最近 5 条
        context = buffer.get_context("group_123", max_length=5)

        assert len(context) == 5
        assert context[0].content == "消息5"
        assert context[4].content == "消息9"

    @pytest.mark.asyncio
    async def test_clear_context(self, mock_config):
        """测试清空指定队列"""
        buffer = L1Buffer()
        await buffer.initialize()

        # 添加消息
        await buffer.add_message("group_123", "user", "测试", "user_456")

        buffer.clear_context("group_123")

        context = buffer.get_context("group_123")
        assert len(context) == 0

    @pytest.mark.asyncio
    async def test_clear_all(self, mock_config):
        """测试清空所有队列"""
        buffer = L1Buffer()
        await buffer.initialize()

        # 添加消息到多个队列
        await buffer.add_message("group_123", "user", "测试1", "user_456")
        await buffer.add_message("group_456", "user", "测试2", "user_789")

        buffer.clear_all()

        assert len(buffer._queues) == 0

    @pytest.mark.asyncio
    async def test_group_isolation_always_enabled(self, mock_config):
        """测试 L1 缓冲始终按群隔离

        L1 不受 enable_group_memory_isolation 配置影响，始终分群存储。
        该配置仅控制 L2/L3 的查询是否带群 ID 条件。
        """
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.enable": True,
                }.get(key, default)
            )

            buffer = L1Buffer()
            await buffer.initialize()

            await buffer.add_message("group_123", "user", "测试1", "user_456")
            await buffer.add_message("group_456", "user", "测试2", "user_789")

            assert len(buffer._queues) == 2

            context1 = buffer.get_context("group_123")
            context2 = buffer.get_context("group_456")

            assert len(context1) == 1
            assert len(context2) == 1

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, mock_config):
        """测试获取队列统计"""
        buffer = L1Buffer()
        await buffer.initialize()

        # 添加消息
        await buffer.add_message("group_123", "user", "测试", "user_456")

        stats = buffer.get_queue_stats("group_123")

        assert stats is not None
        assert stats["message_count"] == 1
        assert stats["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_get_queue_stats_nonexistent(self, mock_config):
        """测试获取不存在的队列统计"""
        buffer = L1Buffer()
        await buffer.initialize()

        stats = buffer.get_queue_stats("nonexistent_group")

        assert stats is None


def _make_msg(content: str = "测试消息", token_count: int = 10) -> ContextMessage:
    return ContextMessage(
        role="user",
        content=content,
        timestamp=datetime.now(),
        token_count=token_count,
        source="group_test",
    )


class TestEmptySummarySegmentPreservation:
    """回归：空总结首次失败不得 rotate，segment_2 必须保留以供重试。

    历史 bug：else 分支仅 fail_count>=2 时 return，fail_count==1 时
    落到 rotate_after_summary() 清空 segment_2，重试阈值 2 形同虚设。
    """

    def _build_buffer_with_queue(self, mock_config, seg2_count: int = 3):
        """构造一个 segment_2 有内容的 buffer，summarizer 返回空。"""
        buffer = L1Buffer()
        buffer._is_available = True
        buffer._component_manager = None  # 避免 _get_or_create_summarizer 触达

        queue = SegmentedMessageQueue(group_id="group_test")
        for _ in range(seg2_count):
            queue.segment_2.append(_make_msg(token_count=10))
        queue.total_tokens = seg2_count * 10
        buffer._queues["group_test"] = queue

        # summarizer.should_summarize -> True；summarize -> ""（空总结）
        fake_summarizer = Mock()
        fake_summarizer.should_summarize = Mock(return_value=True)
        fake_summarizer.summarize = AsyncMock(return_value="")
        buffer._summarizer = fake_summarizer

        # 副作用桩，避免触碰 L2/profile
        buffer._write_summary_to_l2 = AsyncMock(return_value=None)
        buffer._update_profile_after_summary = AsyncMock(return_value=None)
        buffer._clear_images_for_summarized_messages = Mock()

        return buffer, queue

    @pytest.mark.asyncio
    async def test_empty_summary_first_failure_preserves_segment_2(self, mock_config):
        """首次空总结失败：segment_2 保留，不 rotate"""
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.enable": True,
                }.get(key, default)
            )
            buffer, queue = self._build_buffer_with_queue(mock_config, seg2_count=3)

            await buffer._check_and_summarize("group_test")

            # 关键断言：segment_2 内容未被 rotate 清空
            assert len(queue.segment_2) == 3, (
                "空总结首次失败不应 rotate，segment_2 必须保留"
            )
            assert len(queue.segment_3) == 0
            # 失败计数递增到 1
            assert buffer._summary_fail_counts["group_test"] == 1
            # 未写入 L2
            buffer._write_summary_to_l2.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_summary_second_failure_clears_segment_2(self, mock_config):
        """第二次空总结失败（达阈值）：清除 segment_2 并重置计数"""
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.enable": True,
                }.get(key, default)
            )
            buffer, queue = self._build_buffer_with_queue(mock_config, seg2_count=3)
            # 预置已失败一次，本次为第二次
            buffer._summary_fail_counts["group_test"] = 1

            await buffer._check_and_summarize("group_test")

            assert len(queue.segment_2) == 0, "达阈值时应 clear_segment_2"
            assert buffer._summary_fail_counts["group_test"] == 0  # 重置

    @pytest.mark.asyncio
    async def test_successful_summary_rotates(self, mock_config):
        """对照：成功总结后正常 rotate，segment_2 清空"""
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.enable": True,
                }.get(key, default)
            )
            buffer, queue = self._build_buffer_with_queue(mock_config, seg2_count=3)
            buffer._summarizer.summarize = AsyncMock(return_value="有内容的总结")

            await buffer._check_and_summarize("group_test")

            assert len(queue.segment_2) == 0, "成功总结应 rotate 清空 segment_2"
            assert buffer._summary_fail_counts["group_test"] == 0
            buffer._write_summary_to_l2.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rotate_preserves_new_messages_during_await(self, mock_config):
        """回归：rotate 基于快照精确移除已总结消息，保留 await 期间新添加的消息

        历史 bug：快照 target_messages 后 await summarize 让出循环，其间
        add_message 可使新消息进 segment_2；随后 rotate 基于"当前"segment_2
        取尾入 L1-3 并清空。已写入 L2 的旧消息被保留为 L1-3，下轮再次参与
        总结，产生重复 L2 记忆；新消息也被误移除。
        """
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.enable": True,
                }.get(key, default)
            )
            buffer, queue = self._build_buffer_with_queue(mock_config, seg2_count=3)

            # 在 summarize 的 await 期间，模拟新消息入队 segment_2
            new_msg = _make_msg(content="await 期间新消息", token_count=10)

            async def summarize_side_effect(**kwargs):
                # 模拟 await 期间新消息到达
                queue.segment_2.append(new_msg)
                return "有内容的总结"

            buffer._summarizer.summarize = AsyncMock(side_effect=summarize_side_effect)

            await buffer._check_and_summarize("group_test")

            # 关键断言：新消息保留在 segment_2（不被 rotate 误移除）
            assert len(queue.segment_2) == 1, "await 期间新添加的消息应保留在 segment_2"
            assert queue.segment_2[0] is new_msg
            # segment_3 仅含快照尾部（已总结消息），不含新消息
            assert new_msg not in list(queue.segment_3)
            # L2 仅被写入一次（旧消息不重复总结）
            buffer._write_summary_to_l2.assert_awaited_once()


class TestRotateAfterSummarySnapshot:
    """rotate_after_summary 快照模式单元测试（不依赖 buffer 初始化）"""

    def test_snapshot_mode_preserves_new_messages(self):
        """快照模式：仅移除快照中的消息，保留新消息"""
        from iris_memory.l1_buffer.models import SegmentedMessageQueue

        queue = SegmentedMessageQueue(group_id="g1")
        old_msgs = [_make_msg(content=f"旧消息{i}", token_count=5) for i in range(3)]
        new_msg = _make_msg(content="新消息", token_count=5)
        for m in old_msgs:
            queue.segment_2.append(m)
        queue.total_tokens = 15

        # 快照 = 旧消息（不含新消息）
        snapshot = list(old_msgs)
        # 模拟 await 期间新消息入队
        queue.segment_2.append(new_msg)
        queue.total_tokens += 5

        queue.rotate_after_summary(summarized_messages=snapshot)

        # 新消息保留在 segment_2
        assert len(queue.segment_2) == 1
        assert queue.segment_2[0] is new_msg
        # segment_3 含快照尾部
        assert len(queue.segment_3) > 0
        assert new_msg not in list(queue.segment_3)

    def test_legacy_mode_clears_all(self):
        """旧模式（不传快照）：清空整个 segment_2（向后兼容）"""
        from iris_memory.l1_buffer.models import SegmentedMessageQueue

        queue = SegmentedMessageQueue(group_id="g1")
        for i in range(3):
            queue.segment_2.append(_make_msg(content=f"消息{i}", token_count=5))
        queue.total_tokens = 15

        queue.rotate_after_summary()

        assert len(queue.segment_2) == 0
        assert len(queue.segment_3) > 0


class TestUserIdentification:
    """测试用户识别逻辑"""

    def test_build_name_to_id_map(self):
        """测试构建用户名到用户ID的映射"""
        buffer = L1Buffer()

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
            ContextMessage(
                role="user",
                content="我还喜欢吃香蕉",
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

        name_to_id = buffer._build_name_to_id_map(messages)

        assert len(name_to_id) == 2
        assert name_to_id["张三"] == "user_001"
        assert name_to_id["李四"] == "user_002"

    def test_build_name_to_id_map_empty(self):
        """测试空消息列表"""
        buffer = L1Buffer()

        name_to_id = buffer._build_name_to_id_map([])

        assert len(name_to_id) == 0

    def test_build_name_to_id_map_no_metadata(self):
        """测试没有 metadata 的消息"""
        buffer = L1Buffer()

        messages = [
            ContextMessage(
                role="user",
                content="我喜欢吃苹果",
                timestamp=datetime.now(),
                token_count=5,
                source="user_001",
            ),
        ]

        name_to_id = buffer._build_name_to_id_map(messages)

        assert len(name_to_id) == 0

    def test_extract_user_from_item(self):
        """测试从总结条目提取用户ID"""
        buffer = L1Buffer()

        name_to_id = {"张三": "user_001", "李四": "user_002"}

        user_id = buffer._extract_user_from_item("张三提到喜欢吃苹果", name_to_id)

        assert user_id == "user_001"

        user_id = buffer._extract_user_from_item("李四表示喜欢编程", name_to_id)

        assert user_id == "user_002"

    def test_extract_user_no_match(self):
        """测试无法匹配时返回 None"""
        buffer = L1Buffer()

        name_to_id = {"张三": "user_001", "李四": "user_002"}

        user_id = buffer._extract_user_from_item("王五提到今天天气很好", name_to_id)

        assert user_id is None

    def test_extract_user_empty_map(self):
        """测试空用户映射"""
        buffer = L1Buffer()

        user_id = buffer._extract_user_from_item("任何内容", {})

        assert user_id is None


class TestParseSummaryItems:
    """测试分条总结解析"""

    def test_parse_with_dash_prefix(self):
        """测试解析带 "- " 前缀的条目"""
        buffer = L1Buffer()

        summary = """- 用户提到喜欢吃苹果
- 用户询问了项目的配置方法
- 用户表示今天工作压力很大"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 3
        assert items[0] == "用户提到喜欢吃苹果"
        assert items[1] == "用户询问了项目的配置方法"
        assert items[2] == "用户表示今天工作压力很大"

    def test_parse_with_bullet_prefix(self):
        """测试解析带 "• " 前缀的条目"""
        buffer = L1Buffer()

        summary = """• 用户提到喜欢吃苹果
• 用户询问了项目的配置方法"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 2
        assert items[0] == "用户提到喜欢吃苹果"
        assert items[1] == "用户询问了项目的配置方法"

    def test_parse_mixed_format(self):
        """测试解析混合格式的条目"""
        buffer = L1Buffer()

        summary = """- 用户提到喜欢吃苹果
1. 用户询问了项目的配置方法
• 用户表示今天工作压力很大"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 3

    def test_parse_empty_lines_ignored(self):
        """测试空行被忽略"""
        buffer = L1Buffer()

        summary = """- 用户提到喜欢吃苹果

- 用户询问了项目的配置方法

"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 2

    def test_parse_short_items_filtered(self):
        """测试短条目被过滤"""
        buffer = L1Buffer()

        summary = """- 用户提到喜欢吃苹果
- 短
- 用户询问了项目的配置方法
- abc
- 用户表示今天工作压力很大"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 3
        assert "短" not in items
        assert "abc" not in items

    def test_parse_min_length_parameter(self):
        """测试最小长度参数"""
        buffer = L1Buffer()

        summary = """- 用户提到喜欢吃苹果和橙子
- 短条目
- 用户询问了项目的配置方法"""

        items = buffer._parse_summary_items(summary, min_length=10)

        assert len(items) == 2
        assert "短条目" not in items

    def test_parse_plain_lines(self):
        """测试解析无前缀的普通行"""
        buffer = L1Buffer()

        summary = """用户提到喜欢吃苹果
用户询问了项目的配置方法
用户表示今天工作压力很大"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 3

    def test_parse_empty_summary(self):
        """测试空总结"""
        buffer = L1Buffer()

        items = buffer._parse_summary_items("")

        assert len(items) == 0

    def test_parse_whitespace_only(self):
        """测试仅包含空白字符的总结"""
        buffer = L1Buffer()

        summary = """   
   
"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 0

    def test_parse_chinese_numbered_prefix(self):
        """测试中文数字前缀"""
        buffer = L1Buffer()

        summary = """1、用户提到喜欢吃苹果
2、用户询问了项目的配置方法"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 2
        assert items[0] == "用户提到喜欢吃苹果"
        assert items[1] == "用户询问了项目的配置方法"

    def test_parse_parenthesis_prefix(self):
        """测试括号前缀"""
        buffer = L1Buffer()

        summary = """1) 用户提到喜欢吃苹果
2) 用户询问了项目的配置方法"""

        items = buffer._parse_summary_items(summary)

        assert len(items) == 2
        assert items[0] == "用户提到喜欢吃苹果"
        assert items[1] == "用户询问了项目的配置方法"

    def test_parse_skips_markdown_fences(self):
        """回归测试：行式回退不应把 Markdown 代码块标记当作记忆。"""
        buffer = L1Buffer()

        summary = """```json
{
  "memories": []
}
```"""

        items = buffer._parse_summary_items(summary)

        assert items == []
        assert not any("```" in i for i in items)

    def test_parse_skips_json_structural_lines(self):
        """回归测试：行式回退不应把 JSON 骨架行（括号、键名行）当作记忆。"""
        buffer = L1Buffer()

        summary = """{
  "memories": [
    {"content": "有效记忆条目内容", "confidence": "high"}
  ]
}"""

        items = buffer._parse_summary_items(summary)

        # 只应保留真正的记忆文本，不含 JSON 结构行
        assert all(not i.startswith("{") for i in items)
        assert all(not i.startswith("}") for i in items)
        assert all('"memories"' not in i for i in items)
        assert all('"content"' not in i for i in items)

    def test_parse_fenced_empty_json_no_garbage(self):
        """回归测试：用户报告的 bug——```json + "memories": [] 被误导入。"""
        buffer = L1Buffer()

        summary = '```json\n{\n  "memories": []\n}\n```'

        items = buffer._parse_summary_items(summary)

        assert items == []
        assert "```json" not in items
        assert '"memories": []' not in items


class TestLowQualityMemoryFilter:
    """测试记忆内容质量校验"""

    def test_first_person_without_subject(self):
        """第一人称开头且无法关联用户 → 低质量"""
        is_lq, reason = L1Buffer._is_low_quality_memory("我想吃煎蛋", None)
        assert is_lq is True
        assert "第一人称" in reason

    def test_first_person_with_subject_passes(self):
        """第一人称开头但能关联用户 → 通过（第一人称检查跳过，但不含即时性内容）"""
        is_lq, _ = L1Buffer._is_low_quality_memory("我是张三，一名程序员", "user_001")
        assert is_lq is False

    def test_garbled_concatenation(self):
        """拼接痕迹：'我想吃煎蛋8岁' → 低质量"""
        is_lq, reason = L1Buffer._is_low_quality_memory("我想吃煎蛋8岁", None)
        assert is_lq is True
        # 第一人称和拼接痕迹都可能命中，任一即可
        assert reason != ""

    def test_immediate_desire(self):
        """即时性欲望 → 低质量"""
        is_lq, reason = L1Buffer._is_low_quality_memory("张三想吃煎蛋", "user_001")
        assert is_lq is True
        assert "即时性" in reason

    def test_too_short(self):
        """内容过短 → 低质量"""
        is_lq, reason = L1Buffer._is_low_quality_memory("好", None)
        assert is_lq is True
        assert "过短" in reason

    def test_too_long(self):
        """内容过长 → 低质量"""
        long_content = "张三是一个非常厉害的程序员" * 30
        is_lq, reason = L1Buffer._is_low_quality_memory(long_content, "user_001")
        assert is_lq is True
        assert "过长" in reason

    def test_valid_memory_passes(self):
        """正常第三人称记忆 → 通过"""
        is_lq, _ = L1Buffer._is_low_quality_memory(
            "张三是Python程序员，正在学习装饰器", "user_001"
        )
        assert is_lq is False

    def test_valid_memory_without_subject_passes(self):
        """无法关联用户但非第一人称 → 通过（由 subjectless 标记处理）"""
        is_lq, _ = L1Buffer._is_low_quality_memory(
            "张三是Python程序员，正在学习装饰器", None
        )
        assert is_lq is False

    def test_age_concatenation_pattern(self):
        """末尾突兀年龄数字 → 低质量"""
        is_lq, reason = L1Buffer._is_low_quality_memory(
            "张三喜欢玩游戏18岁", "user_001"
        )
        assert is_lq is True
        assert "拼接" in reason

    def test_normal_age_not_filtered(self):
        """正常包含年龄的记忆 → 通过"""
        is_lq, _ = L1Buffer._is_low_quality_memory(
            "张三今年18岁，是一名大学生", "user_001"
        )
        assert is_lq is False

    def test_immediate_desire_variants(self):
        """各种即时性欲望变体 → 低质量"""
        for content in ["张三想去玩", "张三想买手机", "张三想睡觉"]:
            is_lq, reason = L1Buffer._is_low_quality_memory(content, "user_001")
            assert is_lq is True, f"应过滤: {content}"
            assert "即时性" in reason


class TestPrivateSessionQueues:
    """私聊会话键隔离与存储归属（私聊 L1 队列修复）

    私聊事件 group_id 为空字符串，调用方以 private:{user_id} 作为
    队列键；总结写入 L2/画像时会话键需还原为空群 ID，保持既有归属行为。
    """

    @pytest.mark.asyncio
    async def test_private_session_keys_are_isolated(self, mock_config):
        """不同私聊用户的会话键进入各自独立的队列"""
        buffer = L1Buffer()
        await buffer.initialize()

        await buffer.add_message("private:111", "user", "用户A的消息", "111")
        await buffer.add_message("private:222", "user", "用户B的消息", "222")

        ctx_a = buffer.get_context("private:111")
        ctx_b = buffer.get_context("private:222")

        assert [m.content for m in ctx_a] == ["用户A的消息"]
        assert [m.content for m in ctx_b] == ["用户B的消息"]
        stats_a = buffer.get_queue_stats("private:111")
        stats_b = buffer.get_queue_stats("private:222")
        assert stats_a is not None and stats_a["message_count"] == 1
        assert stats_b is not None and stats_b["message_count"] == 1

    def test_get_storage_group_id(self):
        """私聊会话键还原为空群 ID；群聊键原样返回"""
        buffer = L1Buffer()

        assert buffer._get_storage_group_id("private:12345") == ""
        assert buffer._get_storage_group_id("87654321") == "87654321"
        assert buffer._get_storage_group_id("") == ""

    @pytest.mark.asyncio
    async def test_private_summary_uses_empty_group_id_for_storage(self, mock_config):
        """私聊队列总结写入 L2/画像时使用空群 ID（保持既有归属行为）"""
        with patch("iris_memory.l1_buffer.buffer.get_config") as mock_get_config:
            mock_get_config.return_value.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.enable": True,
                }.get(key, default)
            )
            buffer = L1Buffer()
            buffer._is_available = True
            buffer._component_manager = None  # 避免 _get_or_create_summarizer 触达

            queue = SegmentedMessageQueue(group_id="private:12345")
            queue.segment_2.append(_make_msg(token_count=10))
            queue.total_tokens = 10
            buffer._queues["private:12345"] = queue

            fake_summarizer = Mock()
            fake_summarizer.should_summarize = Mock(return_value=True)
            fake_summarizer.summarize = AsyncMock(return_value="私聊总结")
            buffer._summarizer = fake_summarizer

            # 副作用桩，捕获归属参数
            buffer._write_summary_to_l2 = AsyncMock(return_value=None)
            buffer._update_profile_after_summary = AsyncMock(return_value=None)
            buffer._clear_images_for_summarized_messages = Mock()

            await buffer._check_and_summarize("private:12345")

            buffer._write_summary_to_l2.assert_awaited_once()
            assert buffer._write_summary_to_l2.await_args.args[0] == ""
            buffer._update_profile_after_summary.assert_awaited_once()
            assert buffer._update_profile_after_summary.await_args.args[0] == ""

    @pytest.mark.asyncio
    async def test_empty_queue_key_rejected(self, mock_config):
        """空队列键拒绝写入，不再创建共享 "" 队列（防污染兜底）"""
        buffer = L1Buffer()
        await buffer.initialize()

        success = await buffer.add_message("", "user", "无会话键的消息", "user_1")

        assert not success
        assert "" not in buffer._queues
        assert buffer.get_context("") == []

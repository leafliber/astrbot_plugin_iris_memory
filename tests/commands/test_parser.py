"""iris_mem 指令解析器测试

覆盖三种 @ 目标形式（outline / message_str / 纯文本）与 scope 校验。
"""

import pytest
from unittest.mock import Mock

from astrbot.api.message_components import At, Plain

from tests.platform.fakes import make_qq_event
from iris_memory.commands.base import ParsedArgs
from iris_memory.commands.parser import CommandParser


class TestParseAtForms:
    """三种 @ 形式的解析"""

    def test_outline_at_form(self):
        """outline 渲染的真实 @（[At:123456]）直接得到目标用户 ID"""
        parsed = CommandParser.parse("iris_mem l2 clear [At:123456]")

        assert parsed.is_valid
        assert parsed.module == "l2"
        assert parsed.sub_command == "clear"
        assert parsed.args.target_user_id == "123456"
        assert parsed.args.target_user_name is None

    def test_message_str_at_form(self):
        """message_str 渲染的 @名字(123456) 同时得到 ID 与名称"""
        parsed = CommandParser.parse("iris_mem l2 clear @张三(123456)")

        assert parsed.is_valid
        assert parsed.args.target_user_id == "123456"
        assert parsed.args.target_user_name == "张三"

    def test_plain_text_at_form(self):
        """纯文本 @张三 仅得到名称（ID 由 extract_target_user_id 反查）"""
        parsed = CommandParser.parse("iris_mem l2 clear @张三")

        assert parsed.is_valid
        assert parsed.args.target_user_name == "张三"
        assert parsed.args.target_user_id is None

    def test_outline_at_without_subcommand(self):
        """[At:...] 不应被误认为子指令"""
        parsed = CommandParser.parse("iris_mem l2 [At:123456]")

        assert parsed.is_valid
        assert parsed.module == "l2"
        assert parsed.sub_command is None
        assert parsed.args.target_user_id == "123456"

    def test_outline_at_conflicts_with_group_scope(self):
        parsed = CommandParser.parse("iris_mem l2 clear [At:123456] --group")

        assert not parsed.is_valid
        assert "不能同时使用" in parsed.error_message

    def test_outline_at_conflicts_with_all_scope(self):
        parsed = CommandParser.parse("iris_mem l2 clear [At:123456] --all")

        assert not parsed.is_valid


class TestParseBasics:
    """基础解析回归"""

    def test_module_and_sub_command(self):
        parsed = CommandParser.parse("iris_mem l1 clear")

        assert parsed.is_valid
        assert parsed.module == "l1"
        assert parsed.sub_command == "clear"

    def test_group_and_all_conflict(self):
        parsed = CommandParser.parse("iris_mem l2 clear --group --all")

        assert not parsed.is_valid

    def test_plain_name_conflicts_with_group_scope(self):
        parsed = CommandParser.parse("iris_mem l2 clear @张三 --group")

        assert not parsed.is_valid

    def test_subcommand_case_insensitive(self):
        parsed = CommandParser.parse("iris_mem L2 CLEAR")

        assert parsed.sub_command == "clear"

    def test_not_a_command(self):
        parsed = CommandParser.parse("今天天气不错")

        assert not parsed.is_valid


class TestExtractTargetUserId:
    """extract_target_user_id 解析链路"""

    @pytest.mark.asyncio
    async def test_direct_id_short_circuit(self):
        """outline/message_str 形式已解析出 ID，直接返回不再反查"""
        event = Mock()
        args = ParsedArgs(target_user_id="123456")

        user_id, error = await CommandParser.extract_target_user_id(event, args)

        assert (user_id, error) == ("123456", None)

    @pytest.mark.asyncio
    async def test_name_matched_via_chain_at(self):
        """纯文本 @名字 通过消息链 At 组件的已解析名称反查 ID"""
        event = make_qq_event(
            group_id="987654321",
            chain=[Plain(text="iris_mem l2 clear @张三"), At(qq="123456", name="张三")],
        )
        args = ParsedArgs(target_user_name="张三")

        user_id, error = await CommandParser.extract_target_user_id(event, args)

        assert (user_id, error) == ("123456", None)

    @pytest.mark.asyncio
    async def test_name_not_found(self):
        """无匹配的 @ 提及时返回错误信息"""
        event = make_qq_event(chain=[Plain(text="iris_mem l2 clear @张三")])
        args = ParsedArgs(target_user_name="张三")

        user_id, error = await CommandParser.extract_target_user_id(event, args)

        assert user_id is None
        assert "未找到用户" in error

    @pytest.mark.asyncio
    async def test_current_user_scope_returns_none(self):
        """current_user 范围不解析目标 ID（由 handler 使用发送者身份）"""
        event = make_qq_event(user_id="10001")
        args = ParsedArgs(raw_args=["clear"])

        user_id, error = await CommandParser.extract_target_user_id(event, args)

        assert (user_id, error) == (None, None)

"""QQ 官方机器人平台适配器测试。

覆盖 qq_official / qq_official_webhook 的四种消息场景（群/单聊/频道/频道私信）、
openid 稳定标签、raw_data 读取与 message_id 键归一化、链优先的引用/图片/@提取、
频道 get_message 兜底与角色映射。

夹具见 fakes.py 的 FakeBotpyRawMessage / make_qq_official_event：
raw 消息对象刻意只有 __slots__（无 __dict__），若适配器错误依赖 __dict__
回退，本文件的测试会立即失败。
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from iris_memory.platform import get_adapter
from iris_memory.platform.factory import GenericAdapter
from iris_memory.platform.qq_official import QQOfficialAdapter

from .fakes import (
    QQ_OFFICIAL_SCENE_C2C,
    QQ_OFFICIAL_SCENE_GROUP,
    QQ_OFFICIAL_SCENE_GUILD_CHANNEL,
    QQ_OFFICIAL_SCENE_GUILD_DM,
    make_qq_official_event,
)


class TestQQOfficialScenes:
    """四种消息场景的基础字段提取"""

    def test_group_scene_basic_fields(self):
        """群场景：group_id 为 group_openid，会话键即群 ID"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GROUP)
        adapter = QQOfficialAdapter()

        assert adapter.get_user_id(event) == "m3a1f2c3d4e5f6a7b8c9d0e1f2a3b4c5"
        assert adapter.get_group_id(event) == "g7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2"
        assert adapter.is_group_message(event) is True
        assert adapter.get_session_id(event) == "g7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2"

    def test_c2c_scene_basic_fields(self):
        """单聊场景：无群 ID，会话键为 private:{user_openid}"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_C2C)
        adapter = QQOfficialAdapter()

        assert adapter.get_user_id(event) == "u5b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
        assert adapter.get_group_id(event) == ""
        assert adapter.is_group_message(event) is False
        assert (
            adapter.get_session_id(event)
            == "private:u5b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
        )

    def test_guild_channel_scene_basic_fields(self):
        """频道场景：group_id 为 channel_id（AstrBot 归一化），用户名真实可用"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL)
        adapter = QQOfficialAdapter()

        assert adapter.get_user_id(event) == "U789GUILD"
        assert adapter.get_group_id(event) == "C1234567890"
        assert adapter.is_group_message(event) is True
        assert adapter.get_user_name(event) == "频道用户"

    def test_guild_dm_scene_basic_fields(self):
        """频道私信场景：按私聊处理"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GUILD_DM)
        adapter = QQOfficialAdapter()

        assert adapter.get_group_id(event) == ""
        assert adapter.is_group_message(event) is False
        assert adapter.get_session_id(event) == "private:U789GUILD"


class TestQQOfficialDisplayNames:
    """匿名场景的 openid 派生稳定标签"""

    def test_group_returns_stable_member_label(self):
        """群场景无昵称：返回 成员_{member_openid 前 6 位}，跨事件稳定"""
        adapter = QQOfficialAdapter()
        event1 = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GROUP)
        event2 = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GROUP, message_id="QQOFFMSG0002"
        )

        name1 = adapter.get_user_name(event1)
        name2 = adapter.get_user_name(event2)

        assert name1 == "成员_m3a1f2"
        assert name1 == name2
        assert adapter.get_user_nickname(event1) == name1

    def test_c2c_returns_user_label(self):
        """单聊场景无昵称：返回 用户_{user_openid 前 6 位}"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_C2C)
        adapter = QQOfficialAdapter()

        assert adapter.get_user_name(event) == "用户_u5b2c3"

    def test_guild_prefers_real_username(self):
        """频道/频道私信场景平台提供真实 username，不生成标签"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL, username="真名"
        )
        adapter = QQOfficialAdapter()

        assert adapter.get_user_name(event) == "真名"

    def test_label_falls_back_to_sender_id_when_raw_missing(self):
        """raw 载荷不可用时回退 sender.user_id 派生标签"""
        from .fakes import FakeGroup, FakeAstrBotMessage, FakeEvent, FakeMessageMember

        message_obj = FakeAstrBotMessage(
            sender=FakeMessageMember(user_id="fallbackid123", nickname=""),
            group=FakeGroup(group_id="g1", group_name=None),
            raw_message={},
        )
        event = FakeEvent(message_obj=message_obj, platform="qq_official")
        adapter = QQOfficialAdapter()

        assert adapter.get_user_name(event) == "成员_fallba"


class TestQQOfficialRawMessage:
    """raw 载荷读取与键名归一化"""

    def test_raw_message_reads_raw_data_and_normalizes_message_id(self):
        """读 raw_data 属性（非 __dict__），并把 id 键归一化为 message_id"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GROUP, message_id="MSGID99"
        )
        adapter = QQOfficialAdapter()

        raw = adapter.get_raw_message(event)

        assert raw["group_openid"] == "g7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2"
        assert raw["message_id"] == "MSGID99"
        assert raw["id"] == "MSGID99"

    def test_raw_message_without_raw_data_attribute(self):
        """raw_message 本身是 dict（非 botpy 对象）时直接采用"""
        from .fakes import FakeAstrBotMessage, FakeEvent

        message_obj = FakeAstrBotMessage(
            raw_message={"id": "PLAIN01", "group_openid": "g1"}
        )
        event = FakeEvent(message_obj=message_obj, platform="qq_official")
        adapter = QQOfficialAdapter()

        raw = adapter.get_raw_message(event)

        assert raw["message_id"] == "PLAIN01"

    def test_raw_message_none_returns_empty(self):
        """raw_message 为 None 时返回空字典"""
        from .fakes import FakeAstrBotMessage, FakeEvent

        event = FakeEvent(
            message_obj=FakeAstrBotMessage(raw_message=None), platform="qq_official"
        )
        adapter = QQOfficialAdapter()

        assert adapter.get_raw_message(event) == {}
        assert adapter.get_group_name(event) == ""


class TestQQOfficialReplyInfo:
    """引用消息提取（链优先）"""

    def test_group_quote_reply_from_chain(self):
        """群引用：AstrBot 已把被引内容解析进链上 Reply 组件"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GROUP,
            quote_id="REFIDX_QUOTE001",
            quote_content="被引用的消息",
        )
        adapter = QQOfficialAdapter()

        info = adapter.get_reply_info(event)

        assert info.has_reply is True
        assert info.message_id == "REFIDX_QUOTE001"
        assert info.content == "被引用的消息"
        # 平台不提供被引者身份，sender_id 默认值 0 应清洗为空
        assert info.user_id == ""
        assert info.user_name == ""

    def test_guild_reference_fallback(self):
        """频道引用：链上无 Reply 时回退 raw message_reference（内容不可得）"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL)
        event.message_obj.raw_message.raw_data["message_reference"] = {
            "message_id": "GUILDREF01"
        }
        adapter = QQOfficialAdapter()

        info = adapter.get_reply_info(event)

        assert info.has_reply is True
        assert info.message_id == "GUILDREF01"
        assert info.content == ""

    def test_no_reply_returns_empty(self):
        """非引用消息返回空 ReplyInfo"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GROUP)
        adapter = QQOfficialAdapter()

        assert adapter.get_reply_info(event).has_reply is False


class TestQQOfficialMentions:
    """被@用户提取与机器人标记 At 排除"""

    def test_group_scene_mentions_empty_by_platform_design(self):
        """群场景 mentions 只下发机器人自身，标记 At 也应排除，返回空"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GROUP)
        adapter = QQOfficialAdapter()

        assert adapter.get_mentioned_users(event) == []

    def test_guild_raw_mentions_extracted(self):
        """频道场景：raw mentions 数组含真实用户，排除 bot 条目后提取"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL,
            extra_mentions=[{"id": "U999", "username": "被@用户", "bot": False}],
        )
        adapter = QQOfficialAdapter()

        result = adapter.get_mentioned_users(event)

        assert result == [("U999", "被@用户")]

    def test_literal_marker_at_excluded(self):
        """链上字面量 At(qq="qq_official") 标记被排除，不误记机器人"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GUILD_DM)
        adapter = QQOfficialAdapter()

        assert adapter.get_mentioned_users(event) == []

    def test_self_id_marker_at_excluded(self):
        """群场景标记 At(qq=self_id) 即便 bot 字段缺失也应被排除"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GROUP)
        # raw mentions 中机器人条目缺失 bot/is_you 字段，仅 id 与 self_id 相同
        event.message_obj.raw_message.raw_data["mentions"] = [{"id": "B999OFFICIAL"}]
        adapter = QQOfficialAdapter()

        assert adapter.get_mentioned_users(event) == []


class TestQQOfficialImages:
    """图片提取（链上 Image 组件，含被引用消息图片）"""

    def test_images_from_chain_and_reply(self):
        """主消息图片 source=user，Reply.chain 内图片 source=forward"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GROUP,
            quote_id="REFIDX_QUOTE001",
            quote_content="看这张图",
            quote_image_url="https://multimedia.nt.qq.com/quote.png",
            attachments=[
                {
                    "content_type": "image/png",
                    "url": "//multimedia.nt.qq.com/main.png",
                    "filename": "main.png",
                }
            ],
        )
        adapter = QQOfficialAdapter()

        images = adapter.get_images(event)

        assert len(images) == 2
        by_source = {img.source: img for img in images}
        assert by_source["user"].url == "https://multimedia.nt.qq.com/main.png"
        assert by_source["user"].format == "png"
        assert by_source["user"].message_id == "QQOFFMSG0001"
        assert by_source["forward"].url == "https://multimedia.nt.qq.com/quote.png"

    def test_no_images_returns_empty(self):
        """无附件消息返回空列表"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_C2C)
        adapter = QQOfficialAdapter()

        assert adapter.get_images(event) == []


class TestQQOfficialRole:
    """群角色提取（频道身份组映射，群场景恒 member）"""

    def test_guild_owner_role(self):
        """频道创建者（roles 含 "4"）映射为 owner"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL, roles=["4", "26"]
        )
        adapter = QQOfficialAdapter()

        assert adapter.get_user_role(event) == "owner"

    def test_guild_admin_role(self):
        """频道管理员（roles 含 "2"）映射为 admin"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL, roles=["2", "5"]
        )
        adapter = QQOfficialAdapter()

        assert adapter.get_user_role(event) == "admin"

    def test_guild_member_role(self):
        """频道普通成员映射为 member"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL, roles=["5"]
        )
        adapter = QQOfficialAdapter()

        assert adapter.get_user_role(event) == "member"

    def test_group_scene_always_member(self):
        """群场景平台不下发角色，恒为 member"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GROUP)
        adapter = QQOfficialAdapter()

        assert adapter.get_user_role(event) == "member"

    def test_private_scene_role(self):
        """单聊场景返回 private"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_C2C)
        adapter = QQOfficialAdapter()

        assert adapter.get_user_role(event) == "private"


class TestQQOfficialMsgById:
    """get_msg_by_id（仅频道场景可用）"""

    def _make_bot(self, payload=None, side_effect=None):
        bot = Mock()
        getter = AsyncMock(return_value=payload, side_effect=side_effect)
        bot.api.get_message = getter
        return bot, getter

    @pytest.mark.asyncio
    async def test_guild_channel_fetch(self):
        """频道场景调用 get_message 并解析 author/content"""
        bot, getter = self._make_bot(
            payload={
                "id": "GUILDMSG01",
                "author": {"id": "U111", "username": "频道张三"},
                "content": "<@!B999OFFICIAL> 历史消息内容",
            }
        )
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL, bot=bot
        )
        adapter = QQOfficialAdapter()

        info = await adapter.get_msg_by_id(event, "GUILDMSG01")

        getter.assert_awaited_once_with(
            channel_id="C1234567890", message_id="GUILDMSG01"
        )
        assert info.has_reply is True
        assert info.user_id == "U111"
        assert info.user_name == "频道张三"
        # content 中的 <@id> 占位符应被去除
        assert info.content == "历史消息内容"

    @pytest.mark.asyncio
    async def test_group_scene_skips_api(self):
        """群/单聊无按 ID 查询 API，不应调用 bot API"""
        bot, getter = self._make_bot(payload={"id": "x"})
        for scene in (QQ_OFFICIAL_SCENE_GROUP, QQ_OFFICIAL_SCENE_C2C):
            event = make_qq_official_event(scene=scene, bot=bot)
            adapter = QQOfficialAdapter()

            info = await adapter.get_msg_by_id(event, "ANYMSG")

            assert info.has_reply is False
        getter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_api_error_degrades_to_empty(self):
        """API 异常时安全降级为空 ReplyInfo"""
        bot, _ = self._make_bot(side_effect=RuntimeError("boom"))
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL, bot=bot
        )
        adapter = QQOfficialAdapter()

        info = await adapter.get_msg_by_id(event, "GUILDMSG01")

        assert info.has_reply is False

    @pytest.mark.asyncio
    async def test_timeout_degrades_to_empty(self):
        """API 超时时安全降级"""
        bot, _ = self._make_bot(side_effect=asyncio.TimeoutError())
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_DM, bot=bot
        )
        adapter = QQOfficialAdapter()

        info = await adapter.get_msg_by_id(event, "DMMSG01")

        assert info.has_reply is False

    @pytest.mark.asyncio
    async def test_bot_missing_returns_empty(self):
        """event.bot 不可用时返回空，不抛异常"""
        event = make_qq_official_event(
            scene=QQ_OFFICIAL_SCENE_GUILD_CHANNEL, bot=None
        )
        adapter = QQOfficialAdapter()

        info = await adapter.get_msg_by_id(event, "GUILDMSG01")

        assert info.has_reply is False

    @pytest.mark.asyncio
    async def test_forward_messages_unsupported(self):
        """平台无合并转发 API，保持基类空实现"""
        event = make_qq_official_event(scene=QQ_OFFICIAL_SCENE_GROUP)
        adapter = QQOfficialAdapter()

        assert await adapter.get_forward_messages(event) == []


class TestQQOfficialFactoryRegistration:
    """工厂注册：两个平台键均返回专用适配器"""

    def test_get_adapter_for_qq_official(self):
        """qq_official 平台键返回 QQOfficialAdapter"""
        event = make_qq_official_event(platform="qq_official")

        adapter = get_adapter(event)

        assert isinstance(adapter, QQOfficialAdapter)

    def test_get_adapter_for_qq_official_webhook(self):
        """qq_official_webhook 复用同一适配器（数据形状一致）"""
        event = make_qq_official_event(platform="qq_official_webhook")

        adapter = get_adapter(event)

        assert isinstance(adapter, QQOfficialAdapter)

    def test_unregistered_platform_still_degrades(self):
        """未注册平台仍降级 GenericAdapter（降级路径不受本次改动影响）"""
        event = make_qq_official_event(platform="telegram")

        adapter = get_adapter(event)

        assert isinstance(adapter, GenericAdapter)

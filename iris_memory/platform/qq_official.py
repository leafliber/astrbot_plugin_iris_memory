"""
Iris Chat Memory - QQ 官方机器人平台适配器

适配 AstrBot 的 qq_official / qq_official_webhook 协议（QQ 开放平台官方机器人，
底层 SDK 为 qq-botpy），从 AstrMessageEvent 提取平台信息。

数据源策略（对齐 AstrBot 4.x qqofficial 实现）：
- 消息链优先：AstrBot 转换消息时已把引用内容（群 msg_elements 引用消息）、
  附件图片、机器人 @ 标记解析进消息链，优先从 event.get_messages() 的组件读取
- raw 载荷补充：场景标识（group_openid/user_openid/channel_id）、频道 mentions、
  频道身份组 roles 等链上没有的信息，从 Patched botpy 消息对象的 raw_data 读取
- botpy 消息对象全部使用 __slots__（无 __dict__），通用适配器的 __dict__ 回退
  必然失败，必须直接读 AstrBot Patched* 挂载的 raw_data / msg_elements 属性

平台能力边界（QQ 开放平台隐私设计，插件侧只能降级）：
- 群聊/单聊场景不提供用户昵称与群名称：用户身份是 openid（按机器人隔离），
  说话人以 openid 前缀派生稳定标签标识（成员_xxxx / 用户_xxxx）
- 群聊 mentions 只下发机器人自身，用户互 @ 不透出
- 群聊/单聊无按消息 ID 查询历史的 API（get_msg_by_id 仅频道场景可用）
- 无合并转发拉取 API（get_forward_messages 保持基类空实现）
- 同一用户在群（member_openid）/单聊（user_openid）/频道（author.id）是
  三个互不关联的 ID 空间，画像与 Person 节点天然按场景隔离
"""

import asyncio
from typing import Any, List, TYPE_CHECKING

from astrbot.api.message_components import At, Image, Plain, Reply

from iris_memory.core import get_logger
from iris_memory.platform.base import PlatformAdapter, ReplyInfo

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from iris_memory.image.models import ImageInfo

logger = get_logger("platform.qq_official")

# 场景标识
SCENE_GROUP = "group"  # QQ 群（group_openid）
SCENE_C2C = "c2c"  # QQ 单聊（user_openid）
SCENE_GUILD_CHANNEL = "guild_channel"  # 频道子频道（channel_id）
SCENE_GUILD_DM = "guild_dm"  # 频道私信
SCENE_UNKNOWN = "unknown"

# 匿名身份标签的 openid 截取长度（hex 载荷 6 位组合空间 1600 万，
# 百人级群聊碰撞概率可忽略）
_OPENID_LABEL_LENGTH = 6


class QQOfficialAdapter(PlatformAdapter):
    """QQ 官方机器人平台适配器

    实现 qq_official / qq_official_webhook 协议的消息信息提取。

    特性：
    - 四种消息场景（群/单聊/频道/频道私信）统一适配
    - 群/单聊无昵称场景以 openid 派生稳定标签标识说话人
    - 引用消息与附件图片从消息链提取（AstrBot 已解析好内容）
    - get_msg_by_id 仅频道场景可用（botpy get_message API）
    - 不依赖 botpy 包：通过 raw 载荷特征字段探测场景

    Examples:
        >>> adapter = QQOfficialAdapter()
        >>> user_id = adapter.get_user_id(event)
        >>> group_id = adapter.get_group_id(event)
    """

    # ------------------------------------------------------------------
    # 内部助手：数据源访问
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_id(value: Any) -> str:
        """清洗 ID 字段（可能为 int、0 或 None）"""
        if value is None or value == "" or value == 0:
            return ""
        return str(value)

    def _get_chain(self, event: Any) -> list[Any]:
        """读取 AstrBot 消息链（event.get_messages()），失败时返回空列表"""
        getter = getattr(event, "get_messages", None)
        if not callable(getter):
            return []
        try:
            chain = getter()
        except Exception as e:
            logger.debug(f"读取消息链失败: {e}")
            return []
        return chain if isinstance(chain, list) else []

    @staticmethod
    def _extract_text_from_chain(chain: list[Any]) -> str:
        """从消息链组件中提取 Plain 文本"""
        parts = []
        for component in chain:
            if isinstance(component, Plain):
                text = getattr(component, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)

    def get_raw_message(self, event: Any) -> dict[str, Any]:
        """获取平台原始消息载荷

        读取 AstrBot Patched botpy 消息对象挂载的 raw_data（原始事件 dict）。
        botpy 消息类全部使用 __slots__，实例无 __dict__，通用的 __dict__
        回退路径对本平台必然失败。

        QQ 载荷的消息 ID 键是 "id"，而插件消费点（message_hook 等）统一读
        "message_id"，这里做键名归一化。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            原始消息字典（含归一化的 message_id 键），解析失败时返回空字典 {}
        """
        try:
            raw_obj = getattr(event.message_obj, "raw_message", None)
        except AttributeError:
            logger.error("无法获取原始消息：event.message_obj.raw_message 不存在")
            raise

        if raw_obj is None:
            logger.debug("原始消息对象为空")
            return {}

        raw_data = getattr(raw_obj, "raw_data", None)
        if isinstance(raw_data, dict):
            result = dict(raw_data)
        elif isinstance(raw_obj, dict):
            result = dict(raw_obj)
        else:
            logger.debug(f"无法解析原始消息对象: {type(raw_obj)}")
            return {}

        if not result.get("message_id") and result.get("id") is not None:
            result["message_id"] = result["id"]
        return result

    def _detect_scene(self, event: Any) -> str:
        """根据 raw 载荷特征字段探测消息场景

        群/频道私信载荷也含 channel_id，判定顺序不能颠倒：
        group_openid > author.user_openid > direct_message > channel_id。
        """
        raw = self.get_raw_message(event)
        if not raw:
            return SCENE_UNKNOWN
        if raw.get("group_openid"):
            return SCENE_GROUP
        author = raw.get("author")
        if isinstance(author, dict) and author.get("user_openid"):
            return SCENE_C2C
        if raw.get("direct_message"):
            return SCENE_GUILD_DM
        if raw.get("channel_id"):
            return SCENE_GUILD_CHANNEL
        return SCENE_UNKNOWN

    def _self_id(self, event: Any) -> str:
        """读取机器人自身 ID（AstrBot 写入 message_obj.self_id）"""
        getter = getattr(event, "get_self_id", None)
        if callable(getter):
            try:
                return str(getter() or "")
            except Exception:
                pass
        return str(getattr(event.message_obj, "self_id", "") or "")

    def _sender_openid(self, event: Any) -> str:
        """从 raw 载荷读取发送者 openid（群 member_openid / 单聊 user_openid）"""
        raw = self.get_raw_message(event)
        author = raw.get("author")
        if isinstance(author, dict):
            openid = author.get("member_openid") or author.get("user_openid")
            if openid:
                return str(openid)
        return ""

    def _display_name(self, event: Any) -> str:
        """解析用户显示名称

        频道/频道私信场景 sender.nickname 是真实 username；
        群/单聊场景平台不提供昵称，以 openid 前缀派生稳定标签：
        群聊 "成员_xxxx"、单聊 "用户_xxxx"，同一 openid 恒定，
        保证 L1 上下文可区分说话人、画像可累积身份线索。
        """
        nickname = ""
        try:
            nickname = event.message_obj.sender.nickname
        except AttributeError:
            logger.error("无法获取用户名称：event.message_obj.sender 结构异常")
            raise
        if nickname:
            return str(nickname)

        openid = self._sender_openid(event)
        if not openid:
            try:
                openid = str(event.message_obj.sender.user_id or "")
            except AttributeError:
                return ""
        if not openid:
            return ""

        try:
            in_group = bool(self.get_group_id(event))
        except AttributeError:
            in_group = False
        prefix = "成员_" if in_group else "用户_"
        return f"{prefix}{openid[:_OPENID_LABEL_LENGTH]}"

    # ------------------------------------------------------------------
    # 基础信息
    # ------------------------------------------------------------------

    def get_user_id(self, event: Any) -> str:
        """获取用户ID（openid 体系）

        群聊为 member_openid、单聊为 user_openid、频道为频道用户 ID，
        均经 AstrBot 归一化到 message_obj.sender.user_id。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            用户ID字符串
        """
        try:
            return str(event.message_obj.sender.user_id)
        except AttributeError:
            logger.error("无法获取用户ID：event.message_obj.sender.user_id 不存在")
            raise

    def get_user_name(self, event: Any) -> str:
        """获取用户显示名称

        频道场景返回真实 username；群/单聊场景返回 openid 派生的
        稳定标签（成员_xxxx / 用户_xxxx）。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            用户显示名称
        """
        return self._display_name(event)

    def get_user_nickname(self, event: Any) -> str:
        """获取用户原始昵称

        平台无群名片概念，与 get_user_name 行为一致；群/单聊场景
        平台不提供昵称，返回 openid 派生稳定标签。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            用户昵称或稳定标签
        """
        return self._display_name(event)

    def get_group_id(self, event: Any) -> str:
        """获取群聊ID

        群聊返回 group_openid、频道返回 channel_id（AstrBot 归一化到
        message_obj.group_id），私聊返回空字符串。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            群聊ID字符串，私聊时返回空字符串
        """
        try:
            group_id = getattr(event.message_obj, "group_id", "")
            return str(group_id) if group_id else ""
        except AttributeError:
            logger.error("无法获取群ID：event.message_obj.group_id 不存在")
            raise

    def get_group_name(self, event: Any) -> str:
        """获取群聊名称

        QQ 官方 API 不提供群/频道名称查询（群场景无接口，频道场景
        需按 channel_id 逐条调用 get_channel_info，代价过高），
        恒返回空字符串。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            空字符串 ""
        """
        return ""

    def get_user_role(self, event: Any) -> str:
        """获取用户在群聊中的角色

        群/单聊场景平台不下发角色信息，恒为 "member"；
        频道场景从 raw 载荷 member.roles 映射（"4" 创建者 → owner，
        "2"/"3" 管理员 → admin，其余 member）。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            角色字符串：owner、admin、member、private
        """
        if not self.is_group_message(event):
            return "private"

        if self._detect_scene(event) != SCENE_GUILD_CHANNEL:
            return "member"

        raw = self.get_raw_message(event)
        member = raw.get("member")
        roles = member.get("roles") if isinstance(member, dict) else None
        if not isinstance(roles, list):
            return "member"

        role_strs = {str(r) for r in roles}
        if "4" in role_strs:
            return "owner"
        if "2" in role_strs or "3" in role_strs:
            return "admin"
        return "member"

    def is_group_message(self, event: "AstrMessageEvent") -> bool:
        """判断是否为群聊消息（群/频道场景均为 True）

        Args:
            event: AstrBot 消息事件对象

        Returns:
            True 表示群聊/频道消息，False 表示单聊/频道私信
        """
        try:
            group_id = self.get_group_id(event)
            return bool(group_id)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 消息链提取
    # ------------------------------------------------------------------

    def get_images(self, event: Any) -> List["ImageInfo"]:
        """获取消息中的图片列表

        从消息链提取 Image 组件（AstrBot 已把附件归一化为 https URL），
        包括当前消息与引用消息（Reply.chain）内的图片。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            图片信息列表
        """
        from iris_memory.image.models import ImageInfo

        images: List["ImageInfo"] = []
        message_id = self._clean_id(getattr(event.message_obj, "message_id", None))

        try:
            for component in self._get_chain(event):
                if isinstance(component, Reply):
                    for sub in component.chain or []:
                        if isinstance(sub, Image):
                            info = self._image_info_from_component(
                                sub, "forward", message_id
                            )
                            if info:
                                images.append(info)
                elif isinstance(component, Image):
                    info = self._image_info_from_component(
                        component, "user", message_id
                    )
                    if info:
                        images.append(info)
        except Exception as e:
            logger.error(f"提取图片信息失败: {e}")
            return images

        if images:
            logger.debug(f"从消息中提取到 {len(images)} 张图片")
        return images

    def _image_info_from_component(
        self, component: Any, source: str, message_id: str
    ) -> "ImageInfo | None":
        """把链上 Image 组件转为 ImageInfo（url 缺失时返回 None）"""
        from iris_memory.image.models import ImageInfo

        url = str(getattr(component, "url", None) or getattr(component, "file", None) or "")
        if not url:
            return None
        return ImageInfo(
            url=url,
            file_path=url,
            format=self._detect_image_format(url),
            size_kb=0,
            source=source,
            message_id=message_id,
        )

    @staticmethod
    def _detect_image_format(url: str) -> str:
        """从 URL 推断图片格式（jpg/jpeg/png/gif/webp）"""
        if not url:
            return ""
        url_lower = url.lower()
        if ".jpg" in url_lower or ".jpeg" in url_lower:
            return "jpg"
        if ".png" in url_lower:
            return "png"
        if ".gif" in url_lower:
            return "gif"
        if ".webp" in url_lower:
            return "webp"
        return ""

    def get_reply_info(self, event: Any) -> ReplyInfo:
        """获取回复/引用消息的关联信息

        主路：消息链 Reply 组件——群引用消息（msg_elements 引用类型）与频道
        message_reference 均已被 AstrBot 解析进链，组件携带被引用消息的 ID、
        纯文本与附件。sender_id/sender_nickname 平台不提供（默认 0/""），
        由上层 L1 Buffer 回填机制兜底。

        回退：raw 载荷的 message_reference.message_id（频道场景，内容不可得）。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            ReplyInfo 实例，非回复消息时返回空 ReplyInfo
        """
        for component in self._get_chain(event):
            if not isinstance(component, Reply):
                continue
            message_id = self._clean_id(component.id)
            if not message_id:
                continue
            content = getattr(component, "message_str", None) or ""
            if not content and component.chain:
                content = self._extract_text_from_chain(component.chain)
            return ReplyInfo(
                message_id=message_id,
                user_id=self._clean_id(component.sender_id),
                user_name=str(getattr(component, "sender_nickname", "") or ""),
                content=str(content),
            )

        try:
            raw = self.get_raw_message(event)
            reference = raw.get("message_reference")
            if isinstance(reference, dict):
                message_id = self._clean_id(reference.get("message_id"))
                if message_id:
                    return ReplyInfo(message_id=message_id)
        except Exception as e:
            logger.error(f"提取回复信息失败: {e}")
        return ReplyInfo()

    def get_mentioned_users(self, event: Any) -> list[tuple[str, str]]:
        """获取消息中 @提及的用户列表

        群聊场景因平台隐私设计 mentions 只下发机器人自身，返回空列表；
        频道场景从 raw 载荷 mentions 数组提取真实用户（排除 bot 条目）。

        消息链上的 At 组件是 AstrBot 塞入的机器人标记（qq 为机器人 ID 或
        字面量 "qq_official"），必须排除，否则机器人自己会被记成被@用户。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            (user_id, user_name) 元组列表
        """
        self_id = self._self_id(event)
        mentioned: list[tuple[str, str]] = []
        seen: set[str] = set()

        for component in self._get_chain(event):
            if not isinstance(component, At):
                continue
            qq = self._clean_id(component.qq)
            if not qq or qq == "all" or qq in seen:
                continue
            # 排除 AstrBot 塞入的机器人标记 At（群场景为 self_id，
            # 频道/单聊场景为字面量 "qq_official"）
            if qq == "qq_official" or (self_id and qq == self_id):
                continue
            seen.add(qq)
            mentioned.append((qq, str(component.name or "")))

        # 频道场景：raw mentions 数组含真实用户（id/username），
        # 排除 bot 条目与机器人自身
        try:
            raw = self.get_raw_message(event)
            for item in raw.get("mentions") or []:
                if not isinstance(item, dict):
                    continue
                user_id = self._clean_id(item.get("id"))
                if not user_id or user_id in seen:
                    continue
                if item.get("bot") is True or item.get("is_you") is True:
                    continue
                if self_id and user_id == self_id:
                    continue
                seen.add(user_id)
                mentioned.append((user_id, str(item.get("username") or "")))
        except Exception as e:
            logger.debug(f"提取 raw mentions 失败: {e}")

        if mentioned:
            logger.debug(f"提取到 {len(mentioned)} 个被@用户")
        return mentioned

    # ------------------------------------------------------------------
    # 平台 API
    # ------------------------------------------------------------------

    async def get_msg_by_id(self, event: Any, message_id: str) -> ReplyInfo:
        """通过消息ID获取消息内容（仅频道场景）

        频道/频道私信走 botpy 的 get_message API
        （GET /channels/{channel_id}/messages/{message_id}）；
        群聊/单聊平台无按 ID 查询历史的 API，直接返回空。
        群引用内容已在消息链 Reply 组件中，本方法仅作频道场景兜底。

        Args:
            event: AstrBot 消息事件对象（event.bot 为 botpy Client）
            message_id: 消息ID

        Returns:
            ReplyInfo 实例，获取失败或场景不支持时返回空 ReplyInfo
        """
        if not message_id:
            return ReplyInfo()

        scene = self._detect_scene(event)
        if scene not in (SCENE_GUILD_CHANNEL, SCENE_GUILD_DM):
            logger.debug(f"场景 {scene} 不支持按消息ID查询，跳过")
            return ReplyInfo()

        raw = self.get_raw_message(event)
        channel_id = self._clean_id(raw.get("channel_id"))
        if not channel_id:
            return ReplyInfo()

        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None)
        getter = getattr(api, "get_message", None)
        if not callable(getter):
            logger.debug("event.bot.api.get_message 不可用")
            return ReplyInfo()

        try:
            result = await asyncio.wait_for(
                getter(channel_id=channel_id, message_id=message_id),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.debug(f"get_message API 超时：message_id={message_id}")
            return ReplyInfo()
        except Exception as e:
            logger.debug(f"get_message API 调用失败：message_id={message_id}, {e}")
            return ReplyInfo()

        if not isinstance(result, dict):
            return ReplyInfo()

        author = result.get("author")
        user_id = ""
        user_name = ""
        if isinstance(author, dict):
            user_id = self._clean_id(author.get("id"))
            user_name = str(author.get("username") or "")

        content = str(result.get("content") or "").strip()
        # 频道消息 content 中的 <@id> 占位符对记忆无意义，直接去除
        if content:
            for token in ("<@!", "<@"):
                while token in content:
                    start = content.find(token)
                    end = content.find(">", start)
                    if end == -1:
                        break
                    content = content[:start] + content[end + 1 :]
            content = content.strip()

        if not content and not user_id:
            return ReplyInfo()

        return ReplyInfo(
            message_id=message_id,
            user_id=user_id,
            user_name=user_name,
            content=content,
        )

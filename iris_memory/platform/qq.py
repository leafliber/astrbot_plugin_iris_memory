"""
Iris Chat Memory - OneBot11 平台适配器

实现 QQ 平台（OneBot11 协议）的适配器，从 AstrMessageEvent 提取平台信息。

OneBot11 协议参考：
- https://github.com/botuniverse/onebot-11

数据源策略（对齐 AstrBot 4.x 实现）：
- 消息链优先：AstrBot 转换消息时已调 API 解析好 Reply（被回复消息的发送者
  与纯文本）和 At（被@用户的名称），优先从 event.get_messages() 的组件读取，
  避免对同一条消息重复调用平台 API
- raw OneBot 载荷补充：群角色（sender.role）、原始昵称等 AstrBot 不透出到
  message_obj.sender 的字段，从 raw_message 的 sender 字典读取
- AstrBot 的 MessageMember 仅含 user_id/nickname（nickname 为 card or
  昵称的合并值），不得依赖其不存在的 card/role 字段
- bot API 调用统一携带 self_id 路由参数（多账号反向 WS 部署下必需）
- 消息段仅支持 array 格式：AstrBot 上游会直接丢弃 string/CQ 码格式的消息
"""

from typing import Any, List, TYPE_CHECKING

from astrbot.api.message_components import At, Plain, Reply

from iris_memory.core import get_logger
from iris_memory.platform.base import (
    ForwardMessage,
    PlatformAdapter,
    ReplyInfo,
)

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from iris_memory.image.models import ImageInfo

logger = get_logger("platform.qq")


class OneBot11Adapter(PlatformAdapter):
    """OneBot11 平台适配器

    实现 QQ 平台（OneBot11 协议）的消息信息提取。

    特性：
    - 支持群聊/私聊识别
    - 群聊时优先返回群名片
    - 支持角色识别（owner/admin/member）
    - 提供原始消息访问

    Examples:
        >>> adapter = OneBot11Adapter()
        >>> user_id = adapter.get_user_id(event)
        >>> group_id = adapter.get_group_id(event)
        >>> is_group = adapter.is_group_message(event)
    """

    # ------------------------------------------------------------------
    # 内部助手：数据源访问
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_id(value: Any) -> str:
        """清洗 OneBot ID 字段（message_id/user_id 等可能为 int、0 或 None）"""
        if value is None or value == "" or value == 0:
            return ""
        return str(value)

    def _raw_sender(self, event: Any) -> dict[str, Any]:
        """读取 raw OneBot 载荷的 sender 字典（card/nickname/role 的真实来源）"""
        try:
            raw_msg = self.get_raw_message(event)
        except Exception:
            return {}
        if not raw_msg:
            return {}
        sender = raw_msg.get("sender")
        return sender if isinstance(sender, dict) else {}

    def _routing_params(self, event: Any) -> dict[str, Any]:
        """构造多账号路由参数（对齐 AstrBot 核心的 self_id 传递方式）

        aiocqhttp 在多个反向 WS 连接并存时，只有携带 self_id 才能路由到
        事件所属的协议端；单连接部署下可省略。
        """
        try:
            self_id = getattr(event.message_obj, "self_id", "")
        except AttributeError:
            return {}
        if isinstance(self_id, (str, int)) and str(self_id).strip():
            return {"self_id": self_id}
        return {}

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

    def _find_chain_component(self, event: Any, component_type: type) -> Any | None:
        """在消息链中查找首个指定类型的组件"""
        for component in self._get_chain(event):
            if isinstance(component, component_type):
                return component
        return None

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

    def get_user_id(self, event: Any) -> str:
        """获取用户ID（QQ号）

        Args:
            event: AstrBot 消息事件对象

        Returns:
            QQ号字符串
        """
        try:
            return str(event.message_obj.sender.user_id)
        except AttributeError:
            logger.error("无法获取用户ID：event.message_obj.sender.user_id 不存在")
            raise

    def get_user_name(self, event: Any) -> str:
        """获取用户显示名称

        群聊时优先返回群名片（如果有），否则返回昵称。群名片与原始昵称
        从 raw OneBot 载荷的 sender 字典读取；载荷不可用时回退到
        message_obj.sender.nickname（AstrBot 已将其合并为 card or 昵称）。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            用户显示名称
        """
        raw_sender = self._raw_sender(event)
        if raw_sender:
            card = raw_sender.get("card") or ""
            if card:
                return str(card)
            nickname = raw_sender.get("nickname") or ""
            if nickname:
                return str(nickname)
        try:
            return str(event.message_obj.sender.nickname)
        except AttributeError:
            logger.error("无法获取用户名称：event.message_obj.sender 结构异常")
            raise

    def get_user_nickname(self, event: Any) -> str:
        """获取用户原始昵称

        从 raw OneBot 载荷的 sender.nickname 读取，不受群名片影响。
        载荷不可用时回退到 message_obj.sender.nickname——注意该值是
        AstrBot 合并的 card or 昵称，群聊下可能等于群名片。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            用户昵称
        """
        raw_sender = self._raw_sender(event)
        if raw_sender:
            nickname = raw_sender.get("nickname") or ""
            if nickname:
                return str(nickname)
        try:
            return str(event.message_obj.sender.nickname)
        except AttributeError:
            logger.error("无法获取用户昵称：event.message_obj.sender.nickname 不存在")
            raise

    def get_group_id(self, event: Any) -> str:
        """获取群聊ID（群号）

        Args:
            event: AstrBot 消息事件对象

        Returns:
            群号字符串，私聊时返回空字符串
        """
        try:
            group_id = getattr(event.message_obj, "group_id", "")
            return str(group_id) if group_id else ""
        except AttributeError:
            logger.error("无法获取群ID：event.message_obj.group_id 不存在")
            raise

    def get_group_name(self, event: Any) -> str:
        """获取群聊名称

        优先读取 AstrBot 结构化字段 message_obj.group.group_name
        （aiocqhttp 适配器在转换消息时填充，缺省哨兵 "N/A" 视为无群名），
        回退到 raw OneBot 载荷的 group_name 字段。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            群名称字符串，无法获取时返回空字符串 ""
        """
        structured = self._structured_group_name(event)
        if structured:
            return structured

        raw_msg = self.get_raw_message(event)
        if raw_msg:
            raw_name = raw_msg.get("group_name")
            if isinstance(raw_name, str) and raw_name and raw_name != "N/A":
                return raw_name

        return ""

    def get_user_role(self, event: Any) -> str:
        """获取用户在群聊中的角色

        从 raw OneBot 载荷的 sender.role 读取（AstrBot 的 MessageMember
        不包含该字段，群主/管理员信息仅在原始载荷中）。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            角色字符串：owner、admin、member、private
        """
        if not self.is_group_message(event):
            return "private"

        raw_sender = self._raw_sender(event)
        role = raw_sender.get("role") if raw_sender else None
        if isinstance(role, str) and role:
            return role
        return "member"

    def get_raw_message(self, event: Any) -> dict[str, Any]:
        """获取平台原始消息对象

        Args:
            event: AstrBot 消息事件对象

        Returns:
            原始消息字典，解析失败时返回空字典
        """
        try:
            raw_msg = getattr(event.message_obj, "raw_message", None)

            if raw_msg is None:
                logger.debug("原始消息对象为空")
                return {}

            if isinstance(raw_msg, dict):
                return raw_msg

            if hasattr(raw_msg, "__dict__"):
                return raw_msg.__dict__

            logger.debug(f"无法解析原始消息对象: {type(raw_msg)}")
            return {}
        except Exception as e:
            logger.error(f"获取原始消息失败: {e}")
            return {}

    def is_group_message(self, event: "AstrMessageEvent") -> bool:
        """判断是否为群聊消息

        Args:
            event: AstrBot 消息事件对象

        Returns:
            True 表示群聊消息，False 表示私聊消息
        """
        try:
            group_id = self.get_group_id(event)
            return bool(group_id)
        except Exception:
            return False

    def get_images(self, event: Any) -> List["ImageInfo"]:
        """获取消息中的图片列表

        从 OneBot11 消息段中提取图片信息。
        支持提取：
        - 当前消息中的图片
        - 引用/回复消息中的图片

        Args:
            event: AstrBot 消息事件对象

        Returns:
            图片信息列表
        """
        from iris_memory.image.models import ImageInfo

        images: List[ImageInfo] = []

        try:
            raw_msg = self.get_raw_message(event)
            if not raw_msg:
                return images

            images.extend(self._extract_images_from_message(raw_msg, "user"))

            images.extend(self._extract_reply_images(raw_msg))

            logger.debug(f"从消息中提取到 {len(images)} 张图片")
            return images

        except Exception as e:
            logger.error(f"提取图片信息失败: {e}")
            return images

    def get_reply_info(self, event: Any) -> ReplyInfo:
        """获取回复/引用消息的关联信息

        优先读取消息链上的 Reply 组件：AstrBot 转换消息时已为每个 reply 段
        调用过 get_msg，组件携带被回复消息的 ID、发送者与纯文本，直接读取
        可避免对同一条消息再次调用平台 API。

        回退：解析 raw OneBot 载荷的 reply 段（仅 id 可靠；user_id/content
        为 go-cqhttp 扩展，多数协议端实现不提供，由上层决定是否再查 API）。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            ReplyInfo 实例，非回复消息时返回空 ReplyInfo
        """
        # 主路：消息链 Reply 组件（sender_id 默认 0、message_str 默认 "" 需清洗）
        chain_reply = self._find_chain_component(event, Reply)
        if chain_reply is not None:
            message_id = self._clean_id(chain_reply.id)
            if message_id:
                content = getattr(chain_reply, "message_str", None) or ""
                if not content and chain_reply.chain:
                    content = self._extract_text_from_chain(chain_reply.chain)
                return ReplyInfo(
                    message_id=message_id,
                    user_id=self._clean_id(chain_reply.sender_id),
                    user_name=str(
                        getattr(chain_reply, "sender_nickname", "") or ""
                    ),
                    content=str(content),
                )

        try:
            raw_msg = self.get_raw_message(event)
            if not raw_msg:
                return ReplyInfo()

            message_segments = raw_msg.get("message", [])

            # AstrBot 上游仅放行 array 格式消息段，string/CQ 码格式不会到达这里
            if not isinstance(message_segments, list):
                return ReplyInfo()

            for segment in message_segments:
                if not isinstance(segment, dict):
                    continue

                if segment.get("type") == "reply":
                    data = segment.get("data", {})

                    reply_info = ReplyInfo(
                        message_id=str(data.get("id", "")),
                        user_id=str(data.get("user_id", ""))
                        if data.get("user_id")
                        else "",
                        user_name="",
                        content="",
                    )

                    if "sender" in data and isinstance(data["sender"], dict):
                        reply_info.user_name = str(data["sender"].get("nickname", ""))

                    if "content" in data:
                        content = data["content"]
                        if isinstance(content, str):
                            reply_info.content = content
                        elif isinstance(content, list):
                            reply_info.content = self._extract_text_from_segments(
                                content
                            )

                    logger.debug(
                        f"提取回复信息：message_id={reply_info.message_id}, "
                        f"user_id={reply_info.user_id}"
                    )
                    return reply_info

            return ReplyInfo()

        except Exception as e:
            logger.error(f"提取回复信息失败: {e}")
            return ReplyInfo()

    def get_mentioned_users(self, event: Any) -> list[tuple[str, str]]:
        """获取消息中 @提及的用户列表

        优先读取消息链上的 At 组件：其 name 字段由 AstrBot 调
        get_group_member_info/get_stranger_info 解析（raw at 段的 data.name
        在多数 OneBot 实现中不存在）。

        回退：解析 raw OneBot 载荷的 at 段（array 格式）。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            (user_id, user_name) 元组列表
        """
        # 主路：消息链 At 组件（名称已由 AstrBot 解析；AtAll 是 At 子类，跳过）
        chain_mentions: list[tuple[str, str]] = []
        for component in self._get_chain(event):
            if not isinstance(component, At):
                continue
            qq = self._clean_id(component.qq)
            if not qq or qq == "all":
                continue
            chain_mentions.append((qq, str(component.name or "")))
        if chain_mentions:
            return chain_mentions

        mentioned: list[tuple[str, str]] = []

        try:
            raw_msg = self.get_raw_message(event)
            if not raw_msg:
                return mentioned

            message_segments = raw_msg.get("message", [])

            # AstrBot 上游仅放行 array 格式消息段
            if not isinstance(message_segments, list):
                return mentioned

            for segment in message_segments:
                if not isinstance(segment, dict):
                    continue

                if segment.get("type") == "at":
                    data = segment.get("data", {})
                    qq = str(data.get("qq", ""))
                    if not qq or qq == "all":
                        # 跳过 @全体成员
                        continue
                    name = str(data.get("name", ""))
                    mentioned.append((qq, name))

            if mentioned:
                logger.debug(f"提取到 {len(mentioned)} 个被@用户")

            return mentioned

        except Exception as e:
            logger.error(f"提取被@用户失败: {e}")
            return mentioned

    def _extract_text_from_segments(self, segments: list[Any]) -> str:
        """从消息段列表中提取纯文本内容

        Args:
            segments: 消息段列表

        Returns:
            拼接后的纯文本内容
        """
        text_parts = []
        for seg in segments:
            if isinstance(seg, dict) and seg.get("type") == "text":
                text_parts.append(seg.get("data", {}).get("text", ""))
        return "".join(text_parts)

    def _extract_images_from_message(
        self, raw_msg: dict[str, Any], source: str = "user"
    ) -> List["ImageInfo"]:
        """从消息段中提取图片

        Args:
            raw_msg: 原始消息字典
            source: 图片来源（user/forward）

        Returns:
            图片信息列表
        """
        from iris_memory.image.models import ImageInfo

        images: List[ImageInfo] = []

        message_segments = raw_msg.get("message", [])

        # AstrBot 上游仅放行 array 格式消息段
        if not isinstance(message_segments, list):
            return images

        for segment in message_segments:
            if not isinstance(segment, dict):
                continue

            if segment.get("type") == "image":
                data = segment.get("data", {})

                image_info = ImageInfo(
                    url=data.get("url"),
                    file_path=data.get("file"),
                    format=self._detect_image_format(data.get("url", "")),
                    size_kb=0,
                    source=source,
                    message_id=str(raw_msg.get("message_id", "") or ""),
                )

                images.append(image_info)

        return images

    def _extract_reply_images(self, raw_msg: dict[str, Any]) -> List["ImageInfo"]:
        """提取引用/回复消息中的图片

        Args:
            raw_msg: 原始消息字典

        Returns:
            图片信息列表
        """
        from iris_memory.image.models import ImageInfo

        images: List[ImageInfo] = []

        message_segments = raw_msg.get("message", [])

        if not isinstance(message_segments, list):
            return images

        for segment in message_segments:
            if not isinstance(segment, dict):
                continue

            if segment.get("type") == "reply":
                data = segment.get("data", {})

                if "content" in data:
                    content = data["content"]
                    if isinstance(content, list):
                        images.extend(
                            self._extract_images_from_message(
                                {"message": content}, "forward"
                            )
                        )

                break

        return images

    def _detect_image_format(self, url: str) -> str:
        """检测图片格式

        从 URL 或文件名推断图片格式。

        Args:
            url: 图片 URL 或文件路径

        Returns:
            图片格式（jpg/jpeg/png/gif/webp）
        """
        if not url:
            return ""

        url_lower = url.lower()

        if ".jpg" in url_lower or ".jpeg" in url_lower:
            return "jpg"
        elif ".png" in url_lower:
            return "png"
        elif ".gif" in url_lower:
            return "gif"
        elif ".webp" in url_lower:
            return "webp"

        return ""

    async def get_msg_by_id(self, event: Any, message_id: str) -> ReplyInfo:
        """通过消息ID获取消息内容（OneBot11 get_msg API）

        调用 OneBot11 的 get_msg API 获取指定消息的详细内容。
        需要事件对象中包含 bot（CQHttp 实例）属性。

        Args:
            event: AstrBot 消息事件对象，需包含 bot 属性
            message_id: 消息ID

        Returns:
            ReplyInfo 实例，包含消息内容和发送者信息；
            获取失败时返回空 ReplyInfo

        Notes:
            - 依赖 aiocqhttp 的 call_action 方法
            - 不同 OneBot11 实现对 get_msg 支持程度不同
            - Lagrange.OneBot 不支持此 API
            - NapCat / go-cqhttp 通常支持此 API
        """
        import asyncio

        if not message_id:
            return ReplyInfo()

        bot = getattr(event, "bot", None)
        if bot is None:
            logger.debug("event.bot 不存在，无法调用 get_msg API")
            return ReplyInfo()

        try:
            result = await asyncio.wait_for(
                bot.call_action(
                    "get_msg",
                    message_id=int(message_id),
                    **self._routing_params(event),
                ),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.debug(f"get_msg API 超时：message_id={message_id}")
            return ReplyInfo()
        except AttributeError:
            logger.debug("bot.call_action 方法不存在，无法调用 get_msg API")
            return ReplyInfo()
        except Exception as e:
            err_str = str(e)
            if "API_NOT_FOUND" in err_str or "api not found" in err_str.lower():
                logger.debug(f"get_msg API 不可用：message_id={message_id}")
            else:
                logger.debug(
                    f"get_msg API 调用失败：message_id={message_id}, error={e}"
                )
            return ReplyInfo()

        if not result or not isinstance(result, dict):
            return ReplyInfo()

        sender = result.get("sender", {})
        user_id = str(sender.get("user_id", "")) if sender else ""
        user_name = ""
        if sender:
            card = sender.get("card", "")
            nickname = sender.get("nickname", "")
            user_name = card or nickname

        message_segments = result.get("message", [])
        content = ""

        if isinstance(message_segments, str):
            content = message_segments
        elif isinstance(message_segments, list):
            content = self._extract_text_from_segments(message_segments)

        if not content:
            raw_message = result.get("raw_message", "")
            if isinstance(raw_message, str) and raw_message:
                content = raw_message

        if not content and not user_id:
            return ReplyInfo()

        return ReplyInfo(
            message_id=message_id,
            user_id=user_id,
            user_name=user_name,
            content=content,
        )

    async def get_forward_messages(self, event: Any) -> List[ForwardMessage]:
        """提取合并转发消息中的所有子消息

        识别 OneBot11 的 forward 消息段，调用 get_forward_msg API 拉取
        子消息列表，提取每条子消息的发送者ID/名称、文本内容、时间戳等。

        Args:
            event: AstrBot 消息事件对象，需包含 bot 属性

        Returns:
            合并转发子消息列表；非合并转发消息或拉取失败时返回空列表

        Notes:
            - 依赖 aiocqhttp 的 call_action 方法
            - 不同 OneBot11 实现对 get_forward_msg 支持程度不同
            - 单个 forward ID 拉取超时 10 秒
        """
        forward_messages: List[ForwardMessage] = []

        try:
            raw_msg = self.get_raw_message(event)
            if not raw_msg:
                return forward_messages

            message_segments = raw_msg.get("message", [])
            if not isinstance(message_segments, list):
                return forward_messages

            # 收集所有 forward 段的 resId
            forward_ids: list[str] = []
            for segment in message_segments:
                if not isinstance(segment, dict):
                    continue
                if segment.get("type") == "forward":
                    forward_id = segment.get("data", {}).get("id")
                    if forward_id:
                        forward_ids.append(str(forward_id))

            if not forward_ids:
                return forward_messages

            bot = getattr(event, "bot", None)
            if bot is None:
                logger.debug("event.bot 不存在，无法调用 get_forward_msg API")
                return forward_messages

            for forward_id in forward_ids:
                sub_messages = await self._fetch_forward_sub_messages(
                    bot, forward_id, self._routing_params(event)
                )
                forward_messages.extend(sub_messages)

            logger.debug(f"提取到 {len(forward_messages)} 条合并转发子消息")

        except Exception as e:
            logger.error(f"提取合并转发消息失败: {e}")
            return forward_messages

        return forward_messages

    async def _fetch_forward_sub_messages(
        self, bot: Any, forward_id: str, routing_params: dict[str, Any] | None = None
    ) -> List[ForwardMessage]:
        """调用 get_forward_msg API 拉取单个合并转发的子消息列表

        Args:
            bot: aiocqhttp Bot 实例
            forward_id: 合并转发消息的 resId
            routing_params: 多账号路由参数（self_id），可选

        Returns:
            子消息列表，失败时返回空列表
        """
        import asyncio

        try:
            result = await asyncio.wait_for(
                bot.call_action(
                    "get_forward_msg",
                    message_id=forward_id,
                    **(routing_params or {}),
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.debug(f"get_forward_msg API 超时：id={forward_id}")
            return []
        except AttributeError:
            logger.debug("bot.call_action 方法不存在，无法调用 get_forward_msg API")
            return []
        except Exception as e:
            err_str = str(e)
            if "API_NOT_FOUND" in err_str or "api not found" in err_str.lower():
                logger.debug(f"get_forward_msg API 不可用：id={forward_id}")
            else:
                logger.debug(
                    f"get_forward_msg API 调用失败：id={forward_id}, error={e}"
                )
            return []

        if not result or not isinstance(result, dict):
            return []

        messages = result.get("messages", [])
        if not isinstance(messages, list):
            return []

        sub_messages: List[ForwardMessage] = []
        for sub_msg in messages:
            if not isinstance(sub_msg, dict):
                continue

            sender = sub_msg.get("sender", {}) or {}
            user_id = str(sender.get("user_id", "")) if sender else ""
            card = sender.get("card", "")
            nickname = sender.get("nickname", "")
            user_name = str(card or nickname)

            content_segments = sub_msg.get("content", [])
            content = ""
            if isinstance(content_segments, str):
                content = content_segments
            elif isinstance(content_segments, list):
                content = self._extract_text_from_segments(content_segments)

            if not content and not user_id:
                continue

            timestamp = sub_msg.get("time", 0)
            if not isinstance(timestamp, int):
                timestamp = 0

            message_id = str(sub_msg.get("message_id", ""))

            sub_messages.append(
                ForwardMessage(
                    user_id=user_id,
                    user_name=user_name,
                    content=content,
                    timestamp=timestamp,
                    message_id=message_id,
                )
            )

        return sub_messages

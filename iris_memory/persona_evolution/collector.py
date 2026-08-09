"""
Iris Chat Memory - 人格自迭代语料采集器

消息采集入口 collect(event)：
- 采集范围排除：私聊、机器人自身消息、cron 合成事件、合并转发；
- 有效消息规则（文档 §7.2）：空/纯图片占位/纯 URL/纯@/纯 Emoji、
  `/` 命令、可见字符 <4 或 >500、重复字符比例过高、
  同用户短时完全重复、is_injection_attempt 命中整条拒绝；
- 入库前 PII 脱敏（邮箱/手机号/长数字账号/URL query）；
- 去重优先 message_id，否则 sha256(platform+group+user+文本+时间桶)。
"""

import hashlib
import re
import time
from typing import Optional, TYPE_CHECKING

from iris_memory.core import get_logger
from iris_memory.platform import get_adapter
from iris_memory.utils.input_sanitizer import is_injection_attempt
from .storage import PersonaEvolutionStorage

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("persona_evolution.collector")

# 可见字符下限/上限（文档 §7.2）
MIN_VISIBLE_CHARS = 4
MAX_VISIBLE_CHARS = 500

# 重复字符比例上限（长消息中单一字符占比超过则视为刷屏/乱码）
_MAX_CHAR_RATIO = 0.7
_MAX_CHAR_RATIO_MIN_LEN = 10

# 同用户短时完全重复消息的判定窗口（秒）
_DUP_WINDOW_SECONDS = 120.0

# 无 message_id 时去重哈希的时间桶（秒）
_TIMESTAMP_BUCKET_SECONDS = 300

# 图片/表情等占位符（剥离后为空则视为纯占位消息）
_PLACEHOLDER = re.compile(
    r"\[(?:"
    r"CQ:[^\]]*"  # OneBot CQ 码（图片/表情/at 等）
    r"|图:[^\]]*"  # 已解析图片描述
    r"|IMG:[^\]]*"  # 待解析图片占位
    r"|图片|表情|动画表情|贴纸|语音|视频"
    r")\]"
)

# 纯 URL 消息
_PURE_URL = re.compile(r"^https?://\S+$", re.IGNORECASE)

# @ 提及（"@昵称 " 或 CQ at 码）
_AT_TOKEN = re.compile(r"@\S+|\[CQ:at[^\]]*\]")

# PII 脱敏
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LONG_DIGITS = re.compile(r"(?<!\d)\d{5,}(?!\d)")
_URL_QUERY = re.compile(r"(https?://[^\s?]+)\?[^\s]*")


def count_visible_chars(text: str) -> int:
    """可见字符数（去除所有空白字符后的长度）"""
    return len(re.sub(r"\s+", "", text))


def is_pure_emoji(text: str) -> bool:
    """判断文本是否只由 Emoji/符号组成

    覆盖 Emoji 区块、杂项符号、变体选择符与 ZWJ；
    含任何字母/数字/汉字/标点则返回 False。
    """
    if not text:
        return False
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        in_emoji_range = (
            0x1F000 <= cp <= 0x1FAFF  # 各类 Emoji 区块
            or 0x2600 <= cp <= 0x27BF  # 杂项符号与装饰符
            or 0x2B00 <= cp <= 0x2BFF  # 箭头与符号补充
            or cp in (0x200D, 0xFE0E, 0xFE0F, 0x20E3)  # ZWJ/变体选择符/组合键帽
            or 0x1F1E6 <= cp <= 0x1F1FF  # 区域指示符（国旗）
        )
        if not in_emoji_range:
            return False
    return True


def sanitize_pii(text: str) -> str:
    """PII 脱敏：邮箱/手机号/长数字账号/URL query 替换为类型占位符"""
    text = _URL_QUERY.sub(r"\1?[查询参数]", text)
    text = _EMAIL.sub("[邮箱]", text)
    text = _PHONE.sub("[手机号]", text)
    text = _LONG_DIGITS.sub("[数字账号]", text)
    return text


def normalize_text(text: str) -> str:
    """规范化：折叠连续空白、去首尾空白"""
    return re.sub(r"\s+", " ", text).strip()


def compute_dedupe_hash(
    *,
    platform: str,
    group_id: str,
    user_id: str,
    normalized_text: str,
    message_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> str:
    """计算去重哈希：优先 message_id，否则按内容+时间桶"""
    if message_id:
        raw = f"mid:{platform}:{message_id}"
    else:
        bucket = int((timestamp or time.time()) // _TIMESTAMP_BUCKET_SECONDS)
        raw = f"{platform}|{group_id}|{user_id}|{normalized_text}|{bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class PersonaCollector:
    """人格自迭代语料采集器

    持有 storage 引用，完成消息过滤、PII 脱敏、去重与语料写入。
    所有公开方法异常由组件层兜底隔离。
    """

    def __init__(self, storage: PersonaEvolutionStorage, store_max_chars: int = 500):
        self._storage = storage
        self._store_max_chars = store_max_chars
        # 同用户短时完全重复判定缓存：{(group_id, user_id): (文本, 时间戳)}
        self._recent_texts: dict[tuple[str, str], tuple[str, float]] = {}

    async def collect(self, event: "AstrMessageEvent") -> Optional[int]:
        """消息采集入口

        Returns:
            新语料行 id；被过滤/去重时返回 None
        """
        # cron 合成事件不进入语料池
        platform = ""
        try:
            platform = str(event.get_platform_name() or "").lower()
        except Exception:
            pass
        if platform == "cron":
            return None

        adapter = get_adapter(event)

        # 只采集群聊消息（排除私聊）
        if not adapter.is_group_message(event):
            return None

        user_id = adapter.get_user_id(event)
        group_id = adapter.get_group_id(event)
        if not user_id or not group_id:
            return None

        # 过滤机器人自己发出的消息
        try:
            self_id = event.get_self_id()
            if self_id and str(user_id) == str(self_id):
                return None
        except Exception:
            pass

        # 合并转发内容第一版直接排除，避免错误归因到转发者
        try:
            forward_messages = await adapter.get_forward_messages(event)
        except Exception as e:
            logger.debug(f"获取合并转发消息失败（按非转发处理）：{e}")
            forward_messages = []
        if forward_messages:
            return None

        raw_text = getattr(event, "message_str", "") or ""

        # 注入检测对原文整条拒绝（不受 sanitizer 开关控制）
        if is_injection_attempt(raw_text):
            logger.info(f"语料采集拒绝注入消息 [群{group_id} 用户{user_id}]")
            return None

        normalized = self._filter_and_clean(raw_text, group_id, user_id)
        if normalized is None:
            return None

        # 平台 message_id（用于优先去重），其他平台防御性置空
        message_id: Optional[str] = None
        try:
            message_obj = getattr(event, "message_obj", None)
            raw_id = getattr(message_obj, "message_id", None) if message_obj else None
            if raw_id is not None and str(raw_id):
                message_id = str(raw_id)
        except Exception:
            message_id = None

        dedupe_hash = compute_dedupe_hash(
            platform=platform,
            group_id=group_id,
            user_id=user_id,
            normalized_text=normalized,
            message_id=message_id,
        )

        user_name = ""
        group_name = ""
        try:
            user_name = adapter.get_user_name(event) or ""
            group_name = adapter.get_group_name(event) or ""
        except Exception:
            pass

        return self._storage.insert_sample(
            platform=platform,
            group_id=group_id,
            group_name=group_name,
            user_id=user_id,
            user_name=user_name,
            normalized_text=normalized,
            dedupe_hash=dedupe_hash,
            message_id=message_id,
        )

    # ------------------------------------------------------------------
    # 内部：过滤与清洗
    # ------------------------------------------------------------------

    def _filter_and_clean(
        self, raw_text: str, group_id: str, user_id: str
    ) -> Optional[str]:
        """有效消息规则过滤 + PII 脱敏 + 规范化

        Returns:
            可入库文本；被过滤时返回 None
        """
        text = (raw_text or "").strip()
        if not text:
            return None

        # `/` 命令
        if text.startswith("/"):
            return None

        # 纯图片/表情占位：剥离占位符后无剩余
        if not _PLACEHOLDER.sub("", text).strip():
            return None

        # 纯 URL
        if _PURE_URL.match(text):
            return None

        # 纯 @ 提及
        if not _AT_TOKEN.sub("", text).strip():
            return None

        # 可见字符长度
        visible = count_visible_chars(text)
        if visible < MIN_VISIBLE_CHARS or visible > MAX_VISIBLE_CHARS:
            return None

        # 纯 Emoji
        if is_pure_emoji(text):
            return None

        # 重复字符比例过高（刷屏/乱码）
        if visible >= _MAX_CHAR_RATIO_MIN_LEN:
            compact = re.sub(r"\s+", "", text)
            most_common = max(compact.count(ch) for ch in set(compact))
            if most_common / len(compact) > _MAX_CHAR_RATIO:
                return None

        # PII 脱敏后规范化
        text = normalize_text(sanitize_pii(text))
        if len(text) > self._store_max_chars:
            text = text[: self._store_max_chars]
        if count_visible_chars(text) < MIN_VISIBLE_CHARS:
            return None

        # 同用户短时完全重复
        now = time.time()
        key = (group_id, user_id)
        last = self._recent_texts.get(key)
        if last and last[0] == text and now - last[1] < _DUP_WINDOW_SECONDS:
            return None
        self._recent_texts[key] = (text, now)
        self._evict_recent_cache(now)

        return text

    def _evict_recent_cache(self, now: float) -> None:
        """惰性清理短时重复缓存（避免长期运行无限增长）"""
        if len(self._recent_texts) <= 1000:
            return
        expired = [
            key
            for key, (_, ts) in self._recent_texts.items()
            if now - ts >= _DUP_WINDOW_SECONDS
        ]
        for key in expired:
            del self._recent_texts[key]

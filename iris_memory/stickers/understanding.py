"""
Iris Chat Memory - 收到的 Telegram 贴纸 → 一句文字描述

流程（由 main.py 的高优先级消息处理器调用，早于 LLM 请求构建）::

    meta = extract_sticker_meta(event.message_obj.raw_message)
    if meta:
        desc = await understanding.describe(meta, frame_getter_from_event(event))
        understanding.rewrite_event(event, desc, meta)

- 按 ``doc_id`` 缓存（stickers_desc.db），命中即不调用模型；
- 未命中：取代表帧（Telethon ``LazyImage.convert_to_file_path``，动画/视频贴纸
  已在适配器里抽中间帧）→ 视觉模型一句话描述 → 校验 → 缓存；
- 同一 doc_id 并发只调用一次；每日未命中次数有上限；整体不超过 ``timeout``；
- 失败返回 None，``rewrite_event`` 保持事件不变（旧行为：原图交给主模型）。

描述只写画面（角色/动作/表情/图上文字），不揣测发送者意图，不含任何打分
或数值（DESIGN §12）。
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Union

try:  # 运行环境（AstrBot 容器）
    from astrbot.api import logger
except Exception:  # pragma: no cover - 独立导入时退化
    import logging

    logger = logging.getLogger("iris.stickers")

try:
    from astrbot.api.message_components import Image, Plain
except Exception:  # pragma: no cover - 独立导入时退化
    Image = None  # type: ignore[assignment]
    Plain = None  # type: ignore[assignment]

from .store import StickerStore

DescribeImage = Callable[[str, str, float], Awaitable[str]]
FrameGetter = Callable[[], Awaitable[Optional[str]]]
ProviderIdGetter = Callable[[], Union[Optional[str], Awaitable[Optional[str]]]]

DEFAULT_TIMEOUT = 15.0
DEFAULT_DAILY_CAP = 100
DESCRIPTION_MAX_LEN = 80
# 同一贴纸描述失败后，这段时间内不再重试（进程内）。
FAILURE_BACKOFF_SECONDS = 600.0

EXTRA_KEY = "iris_sticker"

KIND_STATIC = "static"
KIND_TGS = "tgs"
KIND_VIDEO = "video"
KIND_PHOTO = "photo"
PHOTO_EXTRA_KEY = "iris_photo"
PHOTO_DESCRIPTION_MAX_LEN = 160

_PHOTO_PROMPT = (
    "这是聊天中收到的一张图片。请用不超过100字的中文客观描述："
    "画面主要内容和场景；如果是截图，说明是什么界面、关键信息是什么；"
    "图中出现的文字用「」原样摘录要点。"
    "只描述看到的内容，不要猜测发图人的想法或心情，不要打分或给出任何数字评价，"
    "不要加前缀、标题或解释，直接输出描述。"
)

_PROMPT = (
    "这是聊天中收到的一张表情贴纸（动画贴纸只截取了其中一帧）。"
    "请用一句不超过40字的中文客观描述画面：是什么角色或动物、在做什么动作、"
    "什么表情或情绪；如果图上有文字，用「」原样写出。"
    "只描述画面本身，不要猜测发送者想表达什么，不要打分或给出任何数字评价，"
    "不要加引号、前缀或解释，直接输出这一句话。"
)

_REFUSAL_RE = re.compile(
    r"(无法|不能|没法|抱歉|对不起|看不到|看不清|没有(?:看到|收到)?(?:图|图片|图像)|未(?:提供|收到)"
    r"|sorry|cannot|can't|unable|as an ai)",
    re.IGNORECASE,
)
_SCORE_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:分|星|%|％)|\d+\s*/\s*\d+|评分|打分|分值|置信度)"
)
_PREFIX_RE = re.compile(r"^(?:描述|画面|贴纸描述|图片描述|description)\s*[:：]\s*", re.IGNORECASE)
_WRAP_QUOTES = ("\"\"", "''", "“”", "‘’", "「」", "『』", "``")


@dataclass(frozen=True)
class StickerMeta:
    doc_id: int
    emoji: str
    kind: str
    mime_type: str
    set_id: Optional[int] = None
    set_short_name: Optional[str] = None


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------


def _type_name(obj: Any) -> str:
    return type(obj).__name__


def _kind_for(mime_type: str, attrs: Any) -> str:
    mime = (mime_type or "").lower()
    if mime == "application/x-tgsticker":
        return KIND_TGS
    if mime.startswith("video/") or any(
        _type_name(a) == "DocumentAttributeVideo" for a in attrs
    ):
        return KIND_VIDEO
    for a in attrs:
        if _type_name(a) == "DocumentAttributeFilename":
            name = str(getattr(a, "file_name", "") or "").lower()
            if name.endswith(".tgs"):
                return KIND_TGS
            if name.endswith((".webm", ".mp4")):
                return KIND_VIDEO
    return KIND_STATIC


def extract_sticker_meta(raw_message: Any) -> Optional[StickerMeta]:
    """从 Telethon ``Message`` 读取贴纸元数据；不是贴纸（或无法识别）返回 None。"""
    if raw_message is None:
        return None
    try:
        document = getattr(raw_message, "document", None)
        if document is None:
            media = getattr(raw_message, "media", None)
            document = getattr(media, "document", None)
        if document is None:
            return None
        attrs = list(getattr(document, "attributes", None) or [])
        sticker_attr = next(
            (a for a in attrs if _type_name(a) == "DocumentAttributeSticker"), None
        )
        if sticker_attr is None:
            return None
        doc_id = getattr(document, "id", None)
        if doc_id is None:
            return None
        set_id: Optional[int] = None
        short_name: Optional[str] = None
        stickerset = getattr(sticker_attr, "stickerset", None)
        set_type = _type_name(stickerset) if stickerset is not None else ""
        if set_type == "InputStickerSetID":
            set_id = int(getattr(stickerset, "id"))
        elif set_type == "InputStickerSetShortName":
            short_name = str(getattr(stickerset, "short_name", "") or "") or None
        mime_type = str(getattr(document, "mime_type", "") or "")
        return StickerMeta(
            doc_id=int(doc_id),
            emoji=str(getattr(sticker_attr, "alt", "") or "").strip(),
            kind=_kind_for(mime_type, attrs),
            mime_type=mime_type,
            set_id=set_id,
            set_short_name=short_name,
        )
    except Exception as exc:  # 元数据异常不影响主流程
        logger.debug(f"[stickers] 读取贴纸元数据失败: {type(exc).__name__}")
        return None


# ---------------------------------------------------------------------------
# 校验 / 文本
# ---------------------------------------------------------------------------


def extract_photo_meta(raw_message: Any) -> Optional[StickerMeta]:
    """从 Telethon ``Message`` 读取照片（photo）元数据；不是照片返回 None。

    照片复用 StickerMeta：doc_id = photo.id（Telegram 照片唯一 id），kind=photo。
    """
    if raw_message is None:
        return None
    try:
        photo = getattr(raw_message, "photo", None)
        if photo is None:
            return None
        photo_id = getattr(photo, "id", None)
        if photo_id is None:
            return None
        return StickerMeta(doc_id=int(photo_id), emoji="", kind=KIND_PHOTO, mime_type="image/jpeg")
    except Exception as exc:
        logger.debug(f"[stickers] 读取照片元数据失败: {type(exc).__name__}")
        return None


def build_prompt(meta: Optional[StickerMeta] = None) -> str:
    if meta is not None and meta.kind == KIND_PHOTO:
        return _PHOTO_PROMPT
    if meta is not None and meta.emoji:
        return _PROMPT + f"（贴纸自带的 emoji 是 {meta.emoji}，仅供参考。）"
    return _PROMPT


def sanitize_description(
    text: Any, max_len: int = DESCRIPTION_MAX_LEN, max_lines: int = 2
) -> Optional[str]:
    """清理模型输出；不合格返回 None。"""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    s = re.sub(r"```[a-zA-Z]*", "", s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return None
    if len(lines) > max_lines:  # 多段解释说明模型没有按要求输出
        return None
    s = " ".join(lines)
    s = re.sub(r"^[#>\-\*\s]+", "", s)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = _PREFIX_RE.sub("", s).strip()
    changed = True
    while changed and len(s) >= 2:
        changed = False
        for pair in _WRAP_QUOTES:
            if s[0] == pair[0] and s[-1] == pair[1]:
                inner = s[1:-1]
                # 「」包着的若是整句之外还有别的「」，说明是图上文字，保留
                if pair == "「」" and ("「" in inner or "」" in inner):
                    continue
                s = inner.strip()
                changed = True
    # 方括号会破坏外层 [贴纸 …] 标记
    s = s.translate(str.maketrans({"[": "「", "]": "」", "【": "「", "】": "」"}))
    s = re.sub(r"\s+", " ", s).strip().rstrip("。.")
    if not s or len(s) > max_len:
        return None
    if _REFUSAL_RE.search(s) or _SCORE_RE.search(s):
        return None
    return s


def sticker_text(description: str, emoji: str = "") -> str:
    emoji = (emoji or "").strip()
    if emoji:
        return f"[贴纸 {emoji}：{description}]"
    return f"[贴纸：{description}]"


# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------


class StickerUnderstanding:
    def __init__(
        self,
        store: StickerStore,
        describe_image: DescribeImage,
        clock: Optional[Callable[[], float]] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        daily_cap: int = DEFAULT_DAILY_CAP,
        failure_backoff: float = FAILURE_BACKOFF_SECONDS,
    ) -> None:
        self._store = store
        self._describe_image = describe_image
        self._clock = clock or time.time
        self.timeout = float(timeout)
        self.daily_cap = int(daily_cap)
        self.failure_backoff = float(failure_backoff)
        self._inflight: Dict[int, "asyncio.Task[Optional[str]]"] = {}
        self._failed_at: Dict[int, float] = {}

    # -- 查询 -----------------------------------------------------------------

    async def describe(
        self, meta: Optional[StickerMeta], frame_path_getter: FrameGetter
    ) -> Optional[str]:
        """返回贴纸描述；任何失败返回 None，等待不超过 ``timeout`` 秒。"""
        if meta is None:
            return None
        doc_id = meta.doc_id
        try:
            cached = await asyncio.to_thread(self._store.get, doc_id)
        except Exception as exc:
            logger.warning(f"[stickers] 描述缓存读取失败 doc={doc_id}: {type(exc).__name__}")
            cached = None
        if cached is not None:
            return cached.description

        failed_at = self._failed_at.get(doc_id)
        if failed_at is not None and self._clock() - failed_at < self.failure_backoff:
            return None

        task = self._inflight.get(doc_id)
        if task is None:
            task = asyncio.create_task(self._describe_miss(meta, frame_path_getter))
            self._inflight[doc_id] = task
            task.add_done_callback(lambda t, d=doc_id: self._on_task_done(d, t))
        try:
            return await asyncio.wait_for(asyncio.shield(task), self.timeout)
        except asyncio.TimeoutError:
            logger.info(f"[stickers] 贴纸描述超时 doc={doc_id}，本次按原图处理")
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[stickers] 贴纸描述失败 doc={doc_id}: {type(exc).__name__}")
            return None

    def _on_task_done(self, doc_id: int, task: "asyncio.Task[Optional[str]]") -> None:
        if self._inflight.get(doc_id) is task:
            self._inflight.pop(doc_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None or task.result() is None:
            self._failed_at[doc_id] = self._clock()
        else:
            self._failed_at.pop(doc_id, None)

    async def _describe_miss(
        self, meta: StickerMeta, frame_path_getter: FrameGetter
    ) -> Optional[str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout

        def remaining() -> float:
            return max(0.05, deadline - loop.time())

        try:
            path = await asyncio.wait_for(frame_path_getter(), remaining())
        except asyncio.TimeoutError:
            logger.info(f"[stickers] 贴纸取帧超时 doc={meta.doc_id}")
            return None
        except Exception as exc:
            logger.info(f"[stickers] 贴纸取帧失败 doc={meta.doc_id}: {type(exc).__name__}")
            return None
        if not path or not os.path.isfile(str(path)):
            return None
        if str(path).lower().endswith((".tgs", ".webm", ".mp4", ".mov", ".mkv")):
            # 适配器抽帧失败时回退原文件；视觉模型读不了，按旧行为处理
            return None

        consumed = await asyncio.to_thread(
            self._store.try_consume_miss, self.daily_cap, now=self._clock()
        )
        if not consumed:
            logger.info("[stickers] 今日贴纸描述额度已用完，按原图处理")
            return None

        try:
            raw = await asyncio.wait_for(
                self._describe_image(str(path), build_prompt(meta), remaining()),
                remaining(),
            )
        except asyncio.TimeoutError:
            logger.info(f"[stickers] 视觉模型超时 doc={meta.doc_id}")
            return None
        except Exception as exc:
            logger.info(f"[stickers] 视觉模型调用失败 doc={meta.doc_id}: {type(exc).__name__}")
            return None

        if meta.kind == KIND_PHOTO:
            description = sanitize_description(raw, PHOTO_DESCRIPTION_MAX_LEN, 4)
        else:
            description = sanitize_description(raw)
        if description is None:
            logger.info(f"[stickers] 贴纸描述不合格，已丢弃 doc={meta.doc_id}")
            return None

        model = str(getattr(self._describe_image, "last_model", "") or "")
        try:
            await asyncio.to_thread(
                self._store.put,
                meta.doc_id,
                description,
                set_id=meta.set_id,
                set_short_name=meta.set_short_name,
                emoji=meta.emoji,
                kind=meta.kind,
                model=model,
                created_at=self._clock(),
            )
        except Exception as exc:
            logger.warning(f"[stickers] 描述缓存写入失败 doc={meta.doc_id}: {type(exc).__name__}")
        return description

    async def aclose(self) -> None:
        tasks = list(self._inflight.values())
        self._inflight.clear()
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except BaseException:
                pass

    # -- 事件改写 -------------------------------------------------------------

    @staticmethod
    def rewrite_event(event: Any, description: Optional[str], meta: Optional[StickerMeta]) -> None:
        rewrite_event(event, description, meta)


_ADAPTER_LABEL_RE = re.compile(r"^\s*\[?\s*(?:贴纸|sticker)\s*\]?\s*\S{0,8}\s*$", re.IGNORECASE)


def _is_image(comp: Any) -> bool:
    if Image is not None and isinstance(comp, Image):
        return True
    return _type_name(comp) in {"Image", "LazyImage"}


def _plain_text(comp: Any) -> Optional[str]:
    if Plain is not None and isinstance(comp, Plain):
        return str(getattr(comp, "text", "") or "")
    if _type_name(comp) == "Plain":
        return str(getattr(comp, "text", "") or "")
    return None


def _make_plain(text: str) -> Any:
    if Plain is None:  # pragma: no cover
        raise RuntimeError("astrbot Plain component unavailable")
    return Plain(text=text)


def photo_text(description: str) -> str:
    return f"[图片：{description}]"


def _rewrite_photo_event(event: Any, description: str, meta: StickerMeta) -> None:
    """照片：保留原图（主模型回复时仍可直接看图），在前面加一段文字描述，
    让门控、L1 记忆与后续对话都能用到图片内容。"""
    message_obj = getattr(event, "message_obj", None)
    chain = getattr(message_obj, "message", None)
    if not isinstance(chain, list):
        return
    text = photo_text(description)
    other_text: list[str] = []
    for comp in chain:
        plain = _plain_text(comp)
        if plain is not None and plain.strip():
            other_text.append(plain.strip())
    message_obj.message = [_make_plain(text), *chain]
    message_str = " ".join([text, *other_text])
    try:
        message_obj.message_str = message_str
    except Exception:
        pass
    event.message_str = message_str
    set_extra = getattr(event, "set_extra", None)
    if callable(set_extra):
        try:
            set_extra(PHOTO_EXTRA_KEY, {"photo_id": meta.doc_id, "description": description})
        except Exception:
            pass


def rewrite_event(event: Any, description: Optional[str], meta: Optional[StickerMeta]) -> None:
    """把贴纸图片组件换成文字描述（照片则保留原图并附描述）；description 为 None 时不改动事件。"""
    if description is None or meta is None:
        return
    if meta.kind == KIND_PHOTO:
        _rewrite_photo_event(event, description, meta)
        return
    message_obj = getattr(event, "message_obj", None)
    chain = getattr(message_obj, "message", None)
    if not isinstance(chain, list):
        return
    text = sticker_text(description, meta.emoji)
    new_chain: list = []
    replaced = False
    other_text: list[str] = []
    for comp in chain:
        if _is_image(comp):
            if not replaced:
                new_chain.append(_make_plain(text))
                replaced = True
            continue
        plain = _plain_text(comp)
        if plain is not None:
            if _ADAPTER_LABEL_RE.match(plain):
                continue  # 适配器生成的 "[贴纸] 😀" 占位，已被描述取代
            if plain.strip():
                other_text.append(plain.strip())
        new_chain.append(comp)
    if not replaced:
        new_chain.insert(0, _make_plain(text))
    message_obj.message = new_chain
    message_str = " ".join([text, *other_text])
    try:
        message_obj.message_str = message_str
    except Exception:
        pass
    event.message_str = message_str
    set_extra = getattr(event, "set_extra", None)
    if callable(set_extra):
        try:
            set_extra(
                EXTRA_KEY,
                {"doc_id": meta.doc_id, "emoji": meta.emoji, "description": description},
            )
        except Exception:
            pass


def frame_getter_from_event(event: Any) -> FrameGetter:
    """返回一个取代表帧路径的 async 函数：调用事件链中第一个图片组件的
    ``convert_to_file_path``（Telethon LazyImage 会惰性下载并抽帧）。"""

    async def _get() -> Optional[str]:
        chain = getattr(getattr(event, "message_obj", None), "message", None) or []
        for comp in chain:
            if _is_image(comp):
                convert = getattr(comp, "convert_to_file_path", None)
                if convert is None:
                    return None
                return await convert()
        return None

    return _get


# ---------------------------------------------------------------------------
# 视觉模型调用
# ---------------------------------------------------------------------------


def make_describe_image(llm_manager: Any, provider_id_getter: ProviderIdGetter) -> DescribeImage:
    """经 Iris ``LLMManager.generate_direct`` 让聊天模型看图（走统一调度与 token 统计）。

    - ``image_urls`` 给本地绝对路径，AstrBot provider 会读文件并转成 data URL。
    - ``generate_direct`` 不经过 on_llm_request 钩子，不会被注入记忆/人格。
    - ``provider_id_getter`` 可同步或异步；为空时交给 LLMManager 选默认 provider。
    返回的函数带 ``last_model`` 属性（最近一次使用的 provider id），供缓存记账。
    """

    async def describe_image(image_path: str, prompt: str, timeout: float) -> str:
        pid = provider_id_getter()
        if inspect.isawaitable(pid):
            pid = await pid
        text = await asyncio.wait_for(
            llm_manager.generate_direct(
                prompt=prompt,
                module="sticker_understanding",
                provider_id=str(pid) if pid else None,
                image_urls=[os.path.abspath(image_path)],
                timeout=timeout,
            ),
            timeout + 1.0,
        )
        describe_image.last_model = str(pid or "")  # type: ignore[attr-defined]
        return str(text or "")

    describe_image.last_model = ""  # type: ignore[attr-defined]
    return describe_image

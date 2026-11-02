"""
Iris Chat Memory - 出站账本（DESIGN.md §5.8 / §5.10 / §6）

只有 Telegram 回执（message id）才算发送证据：

- ``deliver``：先把出站行落盘为 ``sending``，再调用平台 ``send_with_receipt``；
  拿到 id → ``sent``；RPCError 类（服务端明确拒绝）→ ``failed``；
  超时 / 连接异常 / 其它异常 / 无 id → ``uncertain``（交给对账）。
  平台没有 ``send_with_receipt`` → 不发送，``failed``（no_receipt_path）。
- 回执重放幂等：同一 tg id 只能归属一行（store.outbound_ids）。
- 被动回复 / 手动发言记账：``record_reply`` / ``record_manual``（同步，service 层 to_thread）。
- 对账匹配：``match_sent_message`` 在 ``get_messages`` 结果中按
  out=True、时间、规范化文本找到对应消息。

日志只含 id / 状态 / 异常类型，不含消息正文。
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from astrbot.api import logger

from . import models as m

__all__ = [
    "SendOutcome",
    "OutboundLedger",
    "classify_send_error",
    "normalize_text",
    "parse_umo",
    "peer_of_umo",
    "resolve_platform",
    "match_sent_message",
    "message_timestamp",
    "ERR_NO_RECEIPT_PATH",
    "ERR_NO_PLATFORM",
]

DEFAULT_SEND_TIMEOUT_S = 60.0

ERR_NO_RECEIPT_PATH = "no_receipt_path"
ERR_NO_PLATFORM = "no_platform"
ERR_NO_RECEIPT_IDS = "no_receipt_ids"
ERR_DUPLICATE_RECEIPT = "duplicate_receipt"
ERR_TIMEOUT = "timeout"


@dataclass
class SendOutcome:
    """一次分段发送的结果"""

    outbound_id: int | None
    status: str  # sent | failed | uncertain | cancelled
    tg_msg_ids: list[int] = field(default_factory=list)
    error: str = ""

    @property
    def sent(self) -> bool:
        return self.status == m.OUT_SENT


# ---------------------------------------------------------------- helpers


def classify_send_error(exc: BaseException) -> str:
    """RPCError 类（Telegram 明确拒绝，消息未发出）→ failed；其余 → uncertain"""
    for cls in type(exc).__mro__:
        if cls.__name__ == "RPCError":
            return m.OUT_FAILED
    return m.OUT_UNCERTAIN


_WS_RE = re.compile(r"\s+", re.UNICODE)


def normalize_text(text: Any) -> str:
    """对账用文本规范化：NFKC + 折叠空白 + 去首尾空白"""
    s = unicodedata.normalize("NFKC", str(text or ""))
    return _WS_RE.sub(" ", s).strip()


def parse_umo(umo: Any) -> tuple[str, str, str] | None:
    """``platform_id:MessageType:session_id`` → 三元组；非法返回 None"""
    parts = str(umo or "").split(":", 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def peer_of_umo(umo: Any) -> int | None:
    """umo 的 session_id → Telegram peer（去掉 forum 话题 ``#thread``）"""
    parsed = parse_umo(umo)
    if parsed is None:
        return None
    peer = parsed[2].partition("#")[0].strip()
    try:
        return int(peer)
    except ValueError:
        return None


def resolve_platform(context: Any, platform_id: str) -> Any:
    """按 platform_id 找平台实例：优先 get_platform_inst，回退遍历 platform_insts"""
    if context is None or not platform_id:
        return None
    getter = getattr(context, "get_platform_inst", None)
    if callable(getter):
        try:
            inst = getter(platform_id)
            if inst is not None:
                return inst
        except Exception as e:
            logger.warning(f"[continuity] get_platform_inst failed: {type(e).__name__}")
    try:
        insts = getattr(getattr(context, "platform_manager", None), "platform_insts", None) or []
        for inst in insts:
            try:
                if inst.meta().id == platform_id:
                    return inst
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[continuity] platform lookup failed: {type(e).__name__}")
    return None


def message_timestamp(msg: Any) -> float | None:
    """Telethon Message.date（UTC aware datetime）或数值 → epoch 秒"""
    date = getattr(msg, "date", None)
    if isinstance(date, datetime):
        return date.timestamp()
    if isinstance(date, (int, float)) and not isinstance(date, bool):
        return float(date)
    return None


def match_sent_message(
    messages: Iterable[Any],
    text: str,
    since_ts: float,
    exclude_ids: Sequence[int] = (),
) -> int | None:
    """在 get_messages 结果里找本账号发出的、时间 ≥ since_ts、规范化文本相等的消息 id"""
    target = normalize_text(text)
    if not target:
        return None
    excluded = {int(i) for i in exclude_ids}
    for msg in messages or ():
        try:
            if getattr(msg, "out", False) is not True:
                continue
            mid = getattr(msg, "id", None)
            if not isinstance(mid, int) or mid <= 0 or mid in excluded:
                continue
            ts = message_timestamp(msg)
            if ts is None or ts < since_ts:
                continue
            body = getattr(msg, "message", None)
            if body is None:
                body = getattr(msg, "text", "")
            if normalize_text(body) == target:
                return mid
        except Exception:
            continue
    return None


def _int_ids(values: Any) -> list[int]:
    out: list[int] = []
    for v in values or ():
        if isinstance(v, bool):
            continue
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i > 0 and i not in out:
            out.append(i)
    return out


# ---------------------------------------------------------------- ledger


class OutboundLedger:
    """出站账本：所有出站状态变化经 ContinuityStore 落盘"""

    def __init__(
        self,
        store: Any,
        *,
        clock: Callable[[], float] = time.time,
        send_timeout_s: float = DEFAULT_SEND_TIMEOUT_S,
    ) -> None:
        self._store = store
        self._clock = clock
        self._send_timeout_s = float(send_timeout_s)

    async def _db(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    # ------------------------------------------------------------ proactive send

    async def create_draft(
        self,
        scope_key: str,
        text: str,
        *,
        turn_id: str | None,
        part_index: int,
        item_ids: Sequence[str] = (),
        origin: str = "proactive",
    ) -> int:
        return await self._db(
            self._store.add_outbound,
            scope_key,
            origin,
            text,
            m.OUT_DRAFT,
            "bot",
            turn_id=turn_id,
            part_index=part_index,
            item_ids=tuple(item_ids),
        )

    async def mark(self, outbound_id: int, status: str, tg_msg_ids: Sequence[int] = (), error: str = "") -> dict | None:
        try:
            return await self._db(self._store.mark_outbound, outbound_id, status, tuple(tg_msg_ids), error)
        except Exception as e:
            logger.warning(
                f"[continuity] mark_outbound id={outbound_id} status={status} failed: {type(e).__name__}"
            )
            return None

    async def deliver(self, outbound_id: int, scope_key: str, platform: Any, umo: str, text: str) -> SendOutcome:
        """发送一条已落盘为 draft 的出站行（见模块说明）"""
        send_fn = getattr(platform, "send_with_receipt", None) if platform is not None else None
        if not callable(send_fn):
            err = ERR_NO_PLATFORM if platform is None else ERR_NO_RECEIPT_PATH
            await self.mark(outbound_id, m.OUT_FAILED, error=err)
            logger.warning(f"[continuity] outbound={outbound_id} scope={scope_key} not sent: {err}")
            return SendOutcome(outbound_id, m.OUT_FAILED, error=err)

        try:
            from astrbot.api.event import MessageChain
            from astrbot.core.platform.message_session import MessageSesion

            session = MessageSesion.from_str(umo)
            chain = MessageChain().message(text)
        except Exception as e:
            err = f"build_failed:{type(e).__name__}"
            await self.mark(outbound_id, m.OUT_FAILED, error=err)
            return SendOutcome(outbound_id, m.OUT_FAILED, error=err)

        # 发送前必须先落盘 sending；落盘失败则不发送（否则崩溃后无从对账）
        try:
            await self._db(self._store.mark_outbound, outbound_id, m.OUT_SENDING)
        except Exception as e:
            err = f"persist_failed:{type(e).__name__}"
            logger.warning(f"[continuity] outbound={outbound_id} not sent: {err}")
            return SendOutcome(outbound_id, m.OUT_FAILED, error=err)

        try:
            raw_ids = await asyncio.wait_for(send_fn(session, chain), timeout=self._send_timeout_s)
        except asyncio.CancelledError:
            await self.mark(outbound_id, m.OUT_UNCERTAIN, error="cancelled_during_send")
            raise
        except asyncio.TimeoutError:
            await self.mark(outbound_id, m.OUT_UNCERTAIN, error=ERR_TIMEOUT)
            logger.warning(f"[continuity] outbound={outbound_id} scope={scope_key} send timeout -> uncertain")
            return SendOutcome(outbound_id, m.OUT_UNCERTAIN, error=ERR_TIMEOUT)
        except Exception as e:
            status = classify_send_error(e)
            err = type(e).__name__
            await self.mark(outbound_id, status, error=err)
            logger.warning(f"[continuity] outbound={outbound_id} scope={scope_key} send error={err} -> {status}")
            return SendOutcome(outbound_id, status, error=err)

        return await self.apply_receipt(outbound_id, scope_key, raw_ids)

    async def apply_receipt(self, outbound_id: int, scope_key: str, raw_ids: Any) -> SendOutcome:
        """登记回执：只有本行独占的 id 才算证据；没有 → uncertain"""
        ids = _int_ids(raw_ids if isinstance(raw_ids, (list, tuple, set)) else ())
        fresh: list[int] = []
        for tid in ids:
            try:
                owner = await self._db(self._store.outbound_by_tg_id, scope_key, tid)
            except Exception:
                owner = None
            if owner is not None and int(owner["id"]) != int(outbound_id):
                logger.warning(
                    f"[continuity] outbound={outbound_id} receipt tg={tid} already bound to outbound={owner['id']}"
                )
                continue
            fresh.append(tid)
        if not fresh:
            err = ERR_DUPLICATE_RECEIPT if ids else ERR_NO_RECEIPT_IDS
            await self.mark(outbound_id, m.OUT_UNCERTAIN, error=err)
            logger.warning(f"[continuity] outbound={outbound_id} scope={scope_key} no usable receipt -> uncertain")
            return SendOutcome(outbound_id, m.OUT_UNCERTAIN, error=err)
        row = await self.mark(outbound_id, m.OUT_SENT, fresh)
        if row is None:
            # 已真实发出但账本写入失败：结果仍按 sent 返回（行保持 sending，对账可修正）
            logger.warning(f"[continuity] outbound={outbound_id} sent but ledger write failed")
        return SendOutcome(outbound_id, m.OUT_SENT, list(fresh))

    # ------------------------------------------------------------ reply / manual

    def record_reply(
        self,
        scope_key: str,
        text: str,
        tg_msg_ids: Iterable[Any],
        *,
        turn_id: str | None = None,
        reply_to_msg_id: int | None = None,
        receipt_supported: bool = True,
    ) -> dict | None:
        """被动回复记账（同步；在 after_message_sent 中经 to_thread 调用）

        - 有 id：sent（回执重放：任一 id 已登记 → 返回已存在行，不新建）；
        - 无 id 且平台支持回执：failed（不写 L1、不建承诺由调用方负责）；
        - 无 id 且平台不支持回执：uncertain + author=unknown（调用方保持旧行为）。
        """
        try:
            ids = _int_ids(tg_msg_ids)
            for tid in ids:
                existing = self._store.outbound_by_tg_id(scope_key, tid)
                if existing is not None:
                    return existing
            if ids:
                oid = self._store.add_outbound(
                    scope_key, "reply", text, m.OUT_SENDING, "bot",
                    turn_id=turn_id, reply_to_msg_id=reply_to_msg_id,
                )
                return self._store.mark_outbound(oid, m.OUT_SENT, ids)
            if receipt_supported:
                oid = self._store.add_outbound(
                    scope_key, "reply", text, m.OUT_FAILED, "bot",
                    turn_id=turn_id, reply_to_msg_id=reply_to_msg_id,
                )
                return self._store.mark_outbound(oid, m.OUT_FAILED, (), ERR_NO_RECEIPT_IDS)
            oid = self._store.add_outbound(
                scope_key, "reply", text, m.OUT_UNCERTAIN, "unknown",
                turn_id=turn_id, reply_to_msg_id=reply_to_msg_id,
            )
            return self._store.get_outbound(oid)
        except Exception as e:
            logger.warning(f"[continuity] record_reply scope={scope_key} failed: {type(e).__name__}")
            return None

    def record_manual(self, scope_key: str, tg_msg_id: Any, text: str) -> dict | None:
        """账号在其他设备手动发出的消息（sender == self）：author=manual, sent；重放幂等"""
        try:
            ids = _int_ids([tg_msg_id])
            if not ids:
                return None
            existing = self._store.outbound_by_tg_id(scope_key, ids[0])
            if existing is not None:
                return existing
            oid = self._store.add_outbound(scope_key, "manual", text, m.OUT_SENDING, "manual")
            return self._store.mark_outbound(oid, m.OUT_SENT, ids)
        except Exception as e:
            logger.warning(f"[continuity] record_manual scope={scope_key} failed: {type(e).__name__}")
            return None

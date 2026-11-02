"""主人私聊短合批 + 显式等待（DESIGN.md §8）。

由 AstrBot 处理器（priority=-10，仅主人私聊、纯文本、非命令）调用：
``verdict = await batcher.gate(scope_key, event_like)``。

event_like 需提供：
- ``message_str``（可写）：放行时被改写为合并文本；
- ``has_media: bool``；
- ``stop()``：判定为 'stop' 时由 gate 调用（适配层映射到 ``stop_event()``）。

规则：
- 媒体事件：直接放行，不参与合批，也不打断显式等待缓冲。
- question / request / wait_done 且不在显式等待：立即放行（0 等待），
  已缓冲的同 burst 片段合并到本事件，被合并的早先事件返回 'stop'。
- 其他文本：静默窗口 ``merge_quiet_ms``，自 burst 首条起硬上限 ``merge_cap_ms``；
  窗口内同 scope 有新消息 → 早先事件立即 'stop'，最后一个事件放行，
  ``message_str`` = 各条文本按到达顺序以 '\\n' 拼接。
- 显式等待（wait_start）：本事件及后续文本事件全部 'stop'（只缓冲），
  直到 wait_done / question / request 到达（该事件携带缓冲全文放行）；
  若 ``explicit_wait_timeout_s`` 内（自最近一条缓冲消息起）无新文本 → 静默退出：
  缓冲丢弃，不放行任何事件，下一条消息按普通规则处理。
  超时为惰性判定（下一条消息到达时检查），不留常驻任务。
- 重启恢复：可选 ``closure_getter(scope_key) -> closure 行``；每个 scope 第一次
  gate 时，若持久化的 closure 为 waiting + explicit_wait=1 且 wait_since 距今
  未超过 ``explicit_wait_timeout_s``，则恢复显式等待（重启前的缓冲文本已丢失）。

并发：asyncio 单线程；scope 状态只在两次 await 之间同步修改，
早先的等待者通过每 scope 的 asyncio.Event 被唤醒后自行退出，不泄漏任务。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger

from . import rules

__all__ = ["MessageBatcher", "PASS", "STOP"]

PASS = "pass"
STOP = "stop"

_DEFAULTS = {"merge_quiet_ms": 1200, "merge_cap_ms": 2000, "explicit_wait_timeout_s": 180}
# 立即放行的类别（完整问题/请求，或“说完了/你怎么看”）
_COMPLETE_KINDS = frozenset({rules.QUESTION, rules.REQUEST, rules.WAIT_DONE})


def _cfg(config: Any, key: str) -> float:
    if isinstance(config, Mapping):
        val = config.get(key, _DEFAULTS[key])
    else:
        val = getattr(config, key, _DEFAULTS[key])
    try:
        return max(0.0, float(val))
    except (TypeError, ValueError):
        return float(_DEFAULTS[key])


@dataclass
class _ScopeState:
    seq: int = 0                       # 最新文本事件序号
    texts: list[str] = field(default_factory=list)
    burst_start: float | None = None
    last_arrival: float = 0.0
    explicit: bool = False
    seed: asyncio.Future | None = None  # 从持久化 closure 恢复显式等待（每 scope 一次）
    waker: asyncio.Event = field(default_factory=asyncio.Event)

    def reset(self) -> None:
        self.texts = []
        self.burst_start = None
        self.explicit = False

    def notify(self) -> None:
        """唤醒当前所有等待者（它们会发现自己已被取代）。"""
        self.waker.set()
        self.waker = asyncio.Event()


class MessageBatcher:
    def __init__(
        self,
        config: Any = None,
        *,
        sleep: Callable[[float], Awaitable[Any]] | None = None,
        clock: Callable[[], float] | None = None,
        closure_getter: Callable[[str], Any] | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        config = config if config is not None else {}
        self.quiet_s = _cfg(config, "merge_quiet_ms") / 1000.0
        self.cap_s = max(self.quiet_s, _cfg(config, "merge_cap_ms") / 1000.0)
        self.wait_timeout_s = _cfg(config, "explicit_wait_timeout_s")
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or time.monotonic
        self._scopes: dict[str, _ScopeState] = {}
        self._closure_getter = closure_getter
        self._wall_clock = wall_clock or time.time

    def set_closure_getter(self, getter: Callable[[str], Any] | None) -> None:
        """事后接入持久化 closure 读取（如 store.get_closure）"""
        self._closure_getter = getter

    # ------------------------------------------------------------------ API

    def in_explicit_wait(self, scope_key: str) -> bool:
        st = self._scopes.get(scope_key)
        if st is None or not st.explicit:
            return False
        return not self._expire_wait(scope_key, st, self._clock())

    async def gate(self, scope_key: str, event: Any) -> str:
        """返回 'pass' 或 'stop'；'stop' 时已调用 event.stop()。异常时放行原事件。"""
        try:
            return await self._gate(scope_key, event)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[continuity] batching gate failed scope={scope_key}: {type(e).__name__}")
            return PASS

    # ------------------------------------------------------------ internals

    async def _gate(self, scope_key: str, event: Any) -> str:
        if bool(getattr(event, "has_media", False)):
            return PASS
        text = str(getattr(event, "message_str", "") or "")
        if not text.strip():
            return PASS

        st = self._scopes.setdefault(scope_key, _ScopeState())
        if self._closure_getter is not None:
            if st.seed is None:
                st.seed = asyncio.ensure_future(self._seed_from_closure(scope_key, st))
            if not st.seed.done():
                await asyncio.shield(st.seed)  # 同 scope 并发事件按到达顺序继续
        now = self._clock()
        self._expire_wait(scope_key, st, now)
        kind = rules.classify(text)

        # 新文本到达：取代所有早先的等待者
        st.seq += 1
        my_seq = st.seq
        st.texts.append(text)
        if st.burst_start is None:
            st.burst_start = now
        st.last_arrival = now
        st.notify()

        if kind == rules.WAIT_START:
            if not st.explicit:
                logger.info(f"[continuity] batching scope={scope_key} explicit_wait=start")
            st.explicit = True
            return self._stop(event)

        if kind in _COMPLETE_KINDS:
            if st.explicit:
                logger.info(f"[continuity] batching scope={scope_key} explicit_wait=done parts={len(st.texts)}")
            return self._flush(st, event)

        if st.explicit:
            return self._stop(event)

        # 普通合批窗口
        while True:
            if st.seq != my_seq or st.explicit:
                return self._stop(event)
            deadline = min(st.last_arrival + self.quiet_s, (st.burst_start or now) + self.cap_s)
            remaining = deadline - self._clock()
            if remaining <= 0:
                return self._flush(st, event)
            await self._wait(st, remaining)

    async def _seed_from_closure(self, scope_key: str, st: _ScopeState) -> None:
        getter = self._closure_getter
        if getter is None:
            return
        try:
            row = await asyncio.to_thread(getter, scope_key)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[continuity] batching closure seed failed scope={scope_key}: {type(e).__name__}")
            return
        if not row or row.get("state") != "waiting" or not row.get("explicit_wait"):
            return
        since = row.get("wait_since")
        try:
            age = float(self._wall_clock()) - float(since)
        except (TypeError, ValueError):
            return
        if age < 0 or age > self.wait_timeout_s:
            return
        if st.explicit:
            return
        st.explicit = True
        # 以单调时钟重建“最近一条缓冲消息”的时刻，超时判定与正常路径一致
        st.last_arrival = self._clock() - age
        logger.info(f"[continuity] batching scope={scope_key} explicit_wait=restored age={int(age)}s")

    def _flush(self, st: _ScopeState, event: Any) -> str:
        merged = "\n".join(st.texts)
        if len(st.texts) > 1:
            event.message_str = merged
        st.reset()
        return PASS

    @staticmethod
    def _stop(event: Any) -> str:
        event.stop()
        return STOP

    def _expire_wait(self, scope_key: str, st: _ScopeState, now: float) -> bool:
        """显式等待超时 → 丢弃缓冲、退出等待。返回是否发生了超时。"""
        if st.explicit and now - st.last_arrival > self.wait_timeout_s:
            logger.info(
                f"[continuity] batching scope={scope_key} explicit_wait=timeout dropped={len(st.texts)}"
            )
            st.reset()
            return True
        return False

    async def _wait(self, st: _ScopeState, delay: float) -> None:
        """睡 delay 秒，或在本 scope 有新消息时提前醒来。"""
        sleeper = asyncio.ensure_future(self._sleep(delay))
        waker = asyncio.ensure_future(st.waker.wait())
        try:
            await asyncio.wait({sleeper, waker}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (sleeper, waker):
                if not t.done():
                    t.cancel()
            await asyncio.gather(sleeper, waker, return_exceptions=True)

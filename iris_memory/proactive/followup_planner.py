"""
FollowUp 跟进规划器

Bot 主动回复后创建跟进期待，独立监控触发者的后续消息，
在短期窗口到期后通过 LLM 判断是否需要继续跟进。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

from iris_memory.config import get_store
from iris_memory.proactive.models import (
    FollowUpDecision,
    FollowUpExpectation,
    FollowUpReplyType,
    ProactiveReplyResult,
)
from iris_memory.proactive.storage.expectation_store import ExpectationStore
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.followup_planner")

# 回调类型
# 返回 True 表示回复已成功发送，False 表示被静音/冷却等阅断
FollowUpReplyCallback = Callable[
    [ProactiveReplyResult, FollowUpExpectation],
    Coroutine[Any, Any, bool],
]

# LLM 判断回调
LLMFollowUpCallback = Callable[
    [FollowUpExpectation],
    Coroutine[Any, Any, Optional[FollowUpDecision]],
]

# FollowUp LLM 判断 Prompt 模板
FOLLOWUP_PROMPT_TEMPLATE = """你是一个智能助手。Bot 刚刚在群聊中主动发起了一条消息，
现在需要判断是否应该继续跟进回复。

【Bot 上次回复】
{bot_reply_summary}

【用户后续发言】（聚合后）
{aggregated_messages}

【群聊上下文】
{group_context}

请判断：
1. 用户是否在回应 Bot 的消息？
2. 是否需要 Bot 继续跟进？
3. 如果需要，应该以什么方式回复？

输出 JSON 格式：
{{"should_reply": true/false, "reason": "简短原因", "reply_type": "acknowledge/continue_topic/emotion_support/question", "suggested_direction": "回复方向提示"}}

注意：
- 适度跟进，不要过度保守
- 如果用户发言与 Bot 话题相关，优先选择跟进
- 如果用户表达困惑、疑问或情感，应该跟进
- 只有在用户明确表示不需要或话题完全无关时才跳过
- 如果用户只是在和其他人聊天，不要强行介入"""


class FollowUpPlanner:
    """FollowUp 跟进规划器

    生命周期：
    1. Bot 主动回复后 → create_expectation()
    2. 用户消息到达 → on_user_message()
    3. 短期窗口到期 → LLM 判断
    4. 需要回复 → 触发回复回调 → 可能再创建新期待
    5. 不需要 / 达到上限 → 清除期待
    """

    def __init__(
        self,
        expectation_store: Optional[ExpectationStore] = None,
        on_followup_reply: Optional[FollowUpReplyCallback] = None,
        on_llm_decide: Optional[LLMFollowUpCallback] = None,
    ) -> None:
        self._cfg = get_store()
        self._store = expectation_store or ExpectationStore()
        self._on_followup_reply = on_followup_reply
        self._on_llm_decide = on_llm_decide

        # group_id -> asyncio.Task (短期窗口定时器)
        self._short_window_timers: Dict[str, asyncio.Task] = {}
        # group_id -> asyncio.Task (FollowUp 窗口超时定时器)
        self._window_timeout_timers: Dict[str, asyncio.Task] = {}
        self._closed = False

    def set_followup_reply_callback(self, callback: FollowUpReplyCallback) -> None:
        """设置跟进回复回调"""
        self._on_followup_reply = callback

    def set_llm_decide_callback(self, callback: LLMFollowUpCallback) -> None:
        """设置 LLM 判断回调"""
        self._on_llm_decide = callback

    def create_expectation(
        self,
        session_key: str,
        group_id: str,
        trigger_user_id: str,
        trigger_message: str,
        bot_reply_summary: str,
        followup_count: int = 0,
        recent_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[FollowUpExpectation]:
        """创建跟进期待

        Args:
            session_key: 会话标识
            group_id: 群组 ID
            trigger_user_id: 触发主动回复的用户
            trigger_message: 触发消息快照
            bot_reply_summary: Bot 回复摘要
            followup_count: 当前跟进次数（累加）
            recent_context: 近期对话上下文

        Returns:
            FollowUpExpectation 对象，如果达到上限则返回 None
        """
        if not self._cfg.get("proactive_reply.followup_enabled", True):
            return None

        # 检查跟进次数上限
        max_followup = self._cfg.get("proactive_reply.max_followup_count", 3)
        if followup_count >= max_followup:
            logger.debug(
                f"FollowUp count {followup_count} >= max "
                f"{max_followup}, not creating expectation"
            )
            return None

        window_seconds = self._cfg.get("proactive_reply.followup_window_seconds", 150)
        now = datetime.now()

        expectation = FollowUpExpectation(
            session_key=session_key,
            group_id=group_id,
            trigger_user_id=trigger_user_id,
            trigger_message=trigger_message,
            bot_reply_summary=bot_reply_summary,
            followup_window_end=now + timedelta(seconds=window_seconds),
            followup_count=followup_count,
            recent_context=recent_context or [],
        )

        # 取消旧的短期窗口定时器和窗口超时定时器
        self._cancel_short_window_timer(group_id)
        self._cancel_window_timeout_timer(group_id)

        # 存储期待
        self._store.put(expectation)

        # 启动 FollowUp 窗口超时定时器
        self._start_window_timeout(group_id, window_seconds)

        logger.info(
            f"FollowUp expectation created: group={group_id}, "
            f"user={trigger_user_id}, count={followup_count}, "
            f"window={window_seconds}s"
        )
        return expectation

    def on_user_message(
        self,
        user_id: str,
        group_id: str,
        message: str,
        sender_name: str = "",
    ) -> bool:
        """处理用户消息

        只监控触发者的消息，聚合到期待中。

        Args:
            user_id: 发言用户 ID
            group_id: 群组 ID
            message: 消息内容
            sender_name: 发送者名称

        Returns:
            True 如果消息被聚合到期待中
        """
        expectation = self._store.get(group_id)
        if expectation is None:
            return False

        # 只监控触发者
        if user_id != expectation.trigger_user_id:
            return False

        # FollowUp 窗口已过期
        if expectation.is_window_expired:
            self._store.remove(group_id)
            self._cancel_short_window_timer(group_id)
            return False

        # 聚合消息
        expectation.aggregated_messages.append({
            "user_id": user_id,
            "sender_name": sender_name,
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })

        # 设置/重置短期窗口
        short_window = self._cfg.get("proactive_reply.followup_short_window_seconds", 10)
        now = datetime.now()
        new_short_end = now + timedelta(seconds=short_window)

        # 短期窗口不超过 FollowUp 窗口
        if new_short_end > expectation.followup_window_end:
            new_short_end = expectation.followup_window_end

        expectation.short_window_end = new_short_end

        # 重置短期窗口定时器
        remaining = (new_short_end - now).total_seconds()
        old_timer = self._short_window_timers.get(group_id)
        logger.debug(
            f"Resetting short window timer for group {group_id}: "
            f"remaining={remaining:.1f}s, "
            f"old_timer={'active' if old_timer and not old_timer.done() else 'none'}"
        )
        self._cancel_short_window_timer(group_id)
        if remaining > 0:
            self._start_short_window_timer(group_id, remaining)

        logger.info(
            f"Message aggregated for group {group_id}: "
            f"total={len(expectation.aggregated_messages)}, "
            f"short_window={remaining:.1f}s, "
            f"trigger_user={expectation.trigger_user_id}, "
            f"current_user={user_id}"
        )
        return True

    def has_active_expectation(self, group_id: str) -> bool:
        """检查某群是否有活跃的跟进期待"""
        return self._store.has_active(group_id)

    def get_expectation(self, group_id: str) -> Optional[FollowUpExpectation]:
        """获取某群的活跃跟进期待（不移除）"""
        return self._store.get(group_id)

    def restart_short_window_timer(self, group_id: str, delay: float) -> None:
        """重新启动短期窗口定时器

        用于在期待重建后恢复聚合消息的定时器。

        Args:
            group_id: 群组 ID
            delay: 延迟秒数
        """
        self._cancel_short_window_timer(group_id)
        if delay > 0:
            self._start_short_window_timer(group_id, delay)
            logger.debug(
                f"Restarted short window timer for group {group_id}: "
                f"delay={delay:.1f}s"
            )

    def clear_expectation(self, group_id: str) -> None:
        """清除某群的跟进期待"""
        self._cancel_short_window_timer(group_id)
        self._cancel_window_timeout_timer(group_id)
        removed = self._store.remove(group_id)
        if removed:
            logger.debug(f"Expectation cleared for group {group_id}")

    # ── 内部定时器管理 ──────────────────────────────────────────

    def _start_short_window_timer(self, group_id: str, delay: float) -> None:
        """启动短期窗口定时器"""
        if self._closed:
            return

        self._short_window_timers[group_id] = asyncio.create_task(
            self._short_window_expired(group_id, delay),
            name=f"followup-short-{group_id}",
        )

    def _start_window_timeout(self, group_id: str, delay: float) -> None:
        """启动 FollowUp 窗口超时定时器"""
        if self._closed:
            return

        self._window_timeout_timers[group_id] = asyncio.create_task(
            self._followup_window_expired(group_id, delay),
            name=f"followup-window-{group_id}",
        )

    async def _short_window_expired(self, group_id: str, delay: float) -> None:
        """短期窗口到期回调"""
        try:
            await asyncio.sleep(delay)

            if self._closed:
                return

            expectation = self._store.get(group_id)
            if expectation is None:
                logger.debug(
                    f"Short window timer fired for group {group_id}, "
                    f"but expectation already cleared, skipping"
                )
                return

            if not expectation.has_aggregated_messages:
                logger.debug(
                    f"Short window timer fired for group {group_id}, "
                    f"but no aggregated messages, skipping"
                )
                return

            logger.info(
                f"Short window timer fired for group {group_id}: "
                f"messages={len(expectation.aggregated_messages)}, "
                f"trigger_user={expectation.trigger_user_id}"
            )

            # 取消 FollowUp 窗口超时定时器，防止重复触发
            self._cancel_window_timeout_timer(group_id)

            # 触发 LLM 决策
            await self._trigger_llm_decision(expectation)

        except asyncio.CancelledError:
            logger.debug(f"Short window timer cancelled for group {group_id}")
        except Exception as e:
            logger.error(f"Short window expired error for group {group_id}: {e}")
        finally:
            self._short_window_timers.pop(group_id, None)

    async def _followup_window_expired(
        self, group_id: str, delay: float
    ) -> None:
        """FollowUp 窗口到期回调"""
        try:
            await asyncio.sleep(delay)

            if self._closed:
                return

            self._window_timeout_timers.pop(group_id, None)

            # 使用 remove() 而非 get()：窗口计时器到期时 get() 会因窗口过期而
            # 自动删除并返回 None，导致跟进逻辑无法执行。remove() 跳过过期检查。
            expectation = self._store.remove(group_id)
            if expectation is None:
                return

            self._cancel_short_window_timer(group_id)

            # followup_after_all_replies 模式下不等待用户发言，直接触发跟进
            # 其他情况下只有用户有发言时才判断
            followup_after_all = self._cfg.get("proactive_reply.followup_after_all_replies", False)
            if expectation.has_aggregated_messages or followup_after_all:
                logger.info(
                    f"FollowUp window expired for group {group_id}: "
                    f"messages={len(expectation.aggregated_messages)}, "
                    f"triggering LLM decision"
                )
                await self._trigger_llm_decision(expectation)
            else:
                logger.debug(
                    f"FollowUp window expired for group {group_id}, "
                    f"no user messages, clearing"
                )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                f"FollowUp window expired error for group {group_id}: {e}"
            )

    async def _trigger_llm_decision(
        self, expectation: FollowUpExpectation
    ) -> None:
        """触发 LLM 判断

        使用 _processing 标记防止短期窗口定时器和 FollowUp 窗口超时定时器
        同时触发导致重复回复。

        Args:
            expectation: 跟进期待
        """
        # 防止竞态：两个定时器可能同时触发此方法
        if getattr(expectation, '_processing', False):
            logger.debug(
                f"FollowUp decision already in progress for group "
                f"{expectation.group_id}, skipping duplicate trigger"
            )
            return
        expectation._processing = True  # type: ignore[attr-defined]

        logger.info(
            f"Triggering LLM decision for group {expectation.group_id}: "
            f"messages={len(expectation.aggregated_messages)}, "
            f"count={expectation.followup_count}"
        )

        try:
            decision = await self._get_llm_decision(expectation)
        except Exception:
            expectation._processing = False  # type: ignore[attr-defined]
            raise

        if decision is None or not decision.should_reply:
            logger.debug(
                f"LLM decided no followup for group {expectation.group_id}: "
                f"{decision.reason if decision else 'LLM returned None'}"
            )
            self._store.remove(expectation.group_id)
            return

        # 需要回复
        logger.info(
            f"LLM decided followup for group {expectation.group_id}: "
            f"type={decision.reply_type.value}, reason={decision.reason}"
        )

        # 构建回复结果
        reply_result = self._build_followup_reply(expectation, decision)

        # 执行回复
        reply_sent = False
        if self._on_followup_reply:
            try:
                reply_sent = await self._on_followup_reply(reply_result, expectation)
            except Exception as e:
                logger.error(f"FollowUp reply callback failed: {e}")

        # 被静音/冷却阻断时，不消耗跟进次数，直接结束
        self._store.remove(expectation.group_id)
        if not reply_sent:
            logger.debug(
                f"FollowUp reply was blocked (quiet hours/rate limit), "
                f"not incrementing count for group {expectation.group_id}"
            )
            return

        # 更新跟进次数并创建新期待
        new_count = expectation.followup_count + 1
        max_followup = self._cfg.get("proactive_reply.max_followup_count", 3)

        if new_count < max_followup:
            # 创建新的期待
            self.create_expectation(
                session_key=expectation.session_key,
                group_id=expectation.group_id,
                trigger_user_id=expectation.trigger_user_id,
                trigger_message=expectation.trigger_message,
                bot_reply_summary=decision.suggested_direction or "跟进回复",
                followup_count=new_count,
                recent_context=expectation.recent_context,
            )
        else:
            logger.debug(
                f"FollowUp count {new_count} reached max, stopping"
            )

    async def _get_llm_decision(
        self, expectation: FollowUpExpectation
    ) -> Optional[FollowUpDecision]:
        """获取 LLM 判断结果

        优先使用外部注入的 LLM 回调，失败时降级到规则判断。

        Returns:
            FollowUpDecision 或 None
        """
        if self._on_llm_decide:
            try:
                result = await self._on_llm_decide(expectation)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"LLM followup decision failed: {e}")

                fallback = self._cfg.get("proactive_reply.followup_fallback_to_rule", True)
                if not fallback:
                    return None

                logger.debug("Falling back to rule-based followup decision")

        # 规则降级：如果有聚合消息且数量 >= 1，则简单回复
        return self._rule_fallback_decision(expectation)

    def _rule_fallback_decision(
        self, expectation: FollowUpExpectation
    ) -> Optional[FollowUpDecision]:
        """规则降级判断

        followup_after_all_replies 模式：即使无用户回应也主动跟进。
        普通模式：有聚合消息才回复。
        """
        followup_after_all = self._cfg.get("proactive_reply.followup_after_all_replies", False)
        if not expectation.has_aggregated_messages:
            if followup_after_all:
                return FollowUpDecision(
                    should_reply=True,
                    reason="主动跟进（Bot 回复后，用户尚未回应）",
                    reply_type=FollowUpReplyType.CONTINUE_TOPIC,
                    suggested_direction="自然延续话题或关心用户状态",
                )
            return FollowUpDecision(
                should_reply=False,
                reason="无用户回应",
            )

        # 合并所有聚合消息文本
        all_text = " ".join(
            m.get("content", "") for m in expectation.aggregated_messages
        )

        # 极短回复（如"嗯""好""哦"）→ 不跟进
        if len(all_text.strip()) < 3:
            return FollowUpDecision(
                should_reply=False,
                reason="用户回复过短，不适合跟进",
            )

        return FollowUpDecision(
            should_reply=True,
            reason="用户有实质回应（规则降级）",
            reply_type=FollowUpReplyType.CONTINUE_TOPIC,
            suggested_direction="继续当前话题",
        )

    @staticmethod
    def _build_followup_reply(
        expectation: FollowUpExpectation,
        decision: FollowUpDecision,
    ) -> ProactiveReplyResult:
        """构建跟进回复结果"""
        reason = f"跟进回复（{decision.reply_type.value}）: {decision.reason}"

        if expectation.has_aggregated_messages:
            # 用户有回应：继续对话
            msg_summary = "\n".join(
                f"  {m.get('sender_name', '用户')}: {m.get('content', '')}"
                for m in expectation.aggregated_messages[-5:]
            )
            trigger_prompt = (
                "【跟进回复场景】\n"
                "你之前说了一些话，用户已经有所回应，现在继续跟进。\n"
                f"你上次说的：{expectation.bot_reply_summary}\n"
                f"用户回应：\n{msg_summary}\n"
                f"回复方向：{decision.suggested_direction}\n"
                "\n行为指导：\n"
                "- 自然延续对话，不要重复之前说过的内容\n"
                "- 简短回应，不要啰嗦\n"
                "- 如果用户明确表示不感兴趣，可以自然结束话题\n"
            )
        else:
            # 用户尚未回应：主动跟进关怀
            trigger_prompt = (
                "【主动跟进场景】\n"
                "你刚刚回复了用户，但用户还没有进一步发言，现在主动跟进延续话题。\n"
                f"你上次说的：{expectation.bot_reply_summary}\n"
                f"回复方向：{decision.suggested_direction}\n"
                "\n行为指导：\n"
                "- 自然地延续或深化刚才的话题，不要生硬地重复同一问题\n"
                "- 简短自然，像朋友闲聊一样\n"
                "- 不要提及\"用户没有回复\"等元信息\n"
            )

        recent_messages = [
            {
                "sender_name": m.get("sender_name", "用户"),
                "content": m.get("content", ""),
            }
            for m in expectation.aggregated_messages[-5:]
        ]

        return ProactiveReplyResult(
            trigger_prompt=trigger_prompt,
            reply_params={"max_tokens": 150, "temperature": 0.7},
            reason=reason,
            group_id=expectation.group_id,
            session_key=expectation.session_key,
            target_user=expectation.trigger_user_id,
            recent_messages=recent_messages,
            source="followup",
        )

    def _cancel_short_window_timer(self, group_id: str) -> None:
        """取消短期窗口定时器"""
        task = self._short_window_timers.pop(group_id, None)
        if task and not task.done():
            task.cancel()

    def _cancel_window_timeout_timer(self, group_id: str) -> None:
        """取消 FollowUp 窗口超时定时器"""
        task = self._window_timeout_timers.pop(group_id, None)
        if task and not task.done():
            task.cancel()

    @property
    def active_expectation_count(self) -> int:
        """活跃期待数量"""
        return self._store.count

    async def close(self) -> None:
        """关闭规划器"""
        self._closed = True

        # 取消所有短期窗口定时器
        for group_id, task in list(self._short_window_timers.items()):
            if not task.done():
                task.cancel()
        self._short_window_timers.clear()

        # 取消所有窗口超时定时器
        for group_id, task in list(self._window_timeout_timers.items()):
            if not task.done():
                task.cancel()
        self._window_timeout_timers.clear()

        # 清除所有期待
        self._store.clear()
        logger.info("FollowUpPlanner closed")


def build_followup_prompt(expectation: FollowUpExpectation) -> str:
    """构建 FollowUp LLM 判断 Prompt

    供外部 LLM provider 使用。

    Args:
        expectation: 跟进期待

    Returns:
        格式化的 prompt 字符串
    """
    msg_lines = []
    for m in expectation.aggregated_messages:
        name = m.get("sender_name", "用户")
        content = m.get("content", "")
        msg_lines.append(f"  {name}: {content}")
    aggregated_text = "\n".join(msg_lines) if msg_lines else "（无用户发言）"

    ctx_lines = []
    for m in expectation.recent_context[-5:]:
        name = m.get("sender_name", "未知")
        content = m.get("content", "")
        ctx_lines.append(f"  {name}: {content}")
    context_text = "\n".join(ctx_lines) if ctx_lines else "（无上下文）"

    return FOLLOWUP_PROMPT_TEMPLATE.format(
        bot_reply_summary=expectation.bot_reply_summary,
        aggregated_messages=aggregated_text,
        group_context=context_text,
    )


def parse_followup_response(response_text: str) -> Optional[FollowUpDecision]:
    """解析 LLM 返回的 FollowUp 决策 JSON

    Args:
        response_text: LLM 返回的文本

    Returns:
        FollowUpDecision 或 None（解析失败）
    """
    try:
        # 尝试从文本中提取 JSON
        if not response_text:
            return None
        text = response_text.strip()
        # 查找 JSON 块
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        data = json.loads(text[start:end])

        should_reply = bool(data.get("should_reply", False))
        reason = str(data.get("reason", ""))
        reply_type_str = str(data.get("reply_type", "acknowledge"))
        suggested_direction = str(data.get("suggested_direction", ""))

        # 映射 reply_type
        type_map = {
            "acknowledge": FollowUpReplyType.ACKNOWLEDGE,
            "continue_topic": FollowUpReplyType.CONTINUE_TOPIC,
            "emotion_support": FollowUpReplyType.EMOTION_SUPPORT,
            "question": FollowUpReplyType.QUESTION,
        }
        reply_type = type_map.get(reply_type_str, FollowUpReplyType.ACKNOWLEDGE)

        return FollowUpDecision(
            should_reply=should_reply,
            reason=reason,
            reply_type=reply_type,
            suggested_direction=suggested_direction,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse followup LLM response: {e}")
        return None

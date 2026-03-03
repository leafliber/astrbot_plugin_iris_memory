"""
主动回复管理器 v3（Facade）

统一入口，编排 SignalQueue + GroupScheduler + FollowUpPlanner。
保持与 v2 相同的外部接口（process_message、白名单管理等），
内部全部替换为新架构。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris_memory.proactive.config import ProactiveConfig
from iris_memory.proactive.followup_planner import (
    FollowUpPlanner,
    build_followup_prompt,
    parse_followup_response,
)
from iris_memory.proactive.group_scheduler import GroupScheduler
from iris_memory.proactive.models import (
    AggregatedDecision,
    FollowUpDecision,
    FollowUpExpectation,
    ProactiveReplyResult,
    Signal,
)
from iris_memory.proactive.reply_sender import ProactiveReplySender
from iris_memory.proactive.signal_generator import SignalGenerator
from iris_memory.proactive.signal_queue import SignalQueue
from iris_memory.proactive.storage.expectation_store import ExpectationStore
from iris_memory.utils.llm_helper import call_llm, parse_llm_json
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.manager")


class ProactiveManager:
    """主动回复管理器 v3（Facade）

    编排流程：
    1. SignalGenerator 从消息中提取信号
    2. SignalQueue 存储信号（私聊排除）
    3. GroupScheduler 定时聚合决策
    4. FollowUpPlanner 独立监控跟进

    保持 v2 兼容的外部接口：
    - process_message()
    - clear_pending_tasks_for_session()
    - 白名单管理（add/remove/is_group_in_whitelist 等）
    - get_stats()
    - close()
    """

    def __init__(
        self,
        plugin_data_path: Path,
        config: Optional[ProactiveConfig] = None,
        llm_provider: Optional[Any] = None,
        enabled: bool = True,
        group_whitelist_mode: bool = False,
        proactive_mode: str = "rule",
        # 以下为 v2 兼容参数（部分映射到 config）
        chroma_manager: Optional[Any] = None,
        embedding_manager: Optional[Any] = None,
        shared_state: Optional[Any] = None,
        personality: str = "balanced",
        quiet_hours: Optional[List[int]] = None,
        max_history: int = 10,
        max_text_tokens: int = 150,
    ) -> None:
        self._plugin_data_path = plugin_data_path
        self._llm_provider = llm_provider
        self._astrbot_context: Optional[Any] = None
        self._llm_provider_id: Optional[str] = None

        # 构建配置
        if config:
            self._config = config
        else:
            self._config = ProactiveConfig(
                enabled=enabled,
                group_whitelist_mode=group_whitelist_mode,
                proactive_mode=proactive_mode,
                quiet_hours=quiet_hours or [23, 7],
                max_reply_tokens=max_text_tokens,
            )

        # 白名单状态（持久化到 KV 存储）
        self._group_whitelist: List[str] = list(self._config.group_whitelist)
        self._group_whitelist_mode: bool = self._config.group_whitelist_mode

        # 核心组件（延迟初始化）
        self._signal_queue: Optional[SignalQueue] = None
        self._signal_generator: Optional[SignalGenerator] = None
        self._group_scheduler: Optional[GroupScheduler] = None
        self._followup_planner: Optional[FollowUpPlanner] = None
        self._expectation_store: Optional[ExpectationStore] = None

        self._initialized = False
        self._init_lock = asyncio.Lock()

        # 主动回复发送器（由服务层注入）
        self._reply_sender: Optional[ProactiveReplySender] = None

        # 群组 UMO 映射（group_id -> unified_msg_origin）
        self._group_umo_map: Dict[str, str] = {}

        # 日统计（简单内存计数）
        self._daily_reply_count: int = 0
        self._daily_reply_date: str = ""
        self._per_user_daily: Dict[str, int] = {}

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._config.enabled = value

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def set_reply_sender(self, sender: ProactiveReplySender) -> None:
        """注入主动回复发送器

        由服务层在所有组件初始化完成后调用。

        Args:
            sender: ProactiveReplySender 实例
        """
        self._reply_sender = sender
        logger.debug("Reply sender injected into ProactiveManager")

    def set_context(
        self,
        astrbot_context: Any,
        llm_provider_id: Optional[str] = None,
    ) -> None:
        """设置 AstrBot 上下文（用于 LLM 确认等内部调用）

        Args:
            astrbot_context: AstrBot Context
            llm_provider_id: LLM Provider ID
        """
        self._astrbot_context = astrbot_context
        self._llm_provider_id = llm_provider_id

    def get_group_umo(self, group_id: str) -> Optional[str]:
        """获取群组的 unified_msg_origin

        Args:
            group_id: 群组 ID

        Returns:
            UMO 字符串，未知返回 None
        """
        return self._group_umo_map.get(group_id)

    # ── 初始化 ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """初始化所有子组件"""
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing ProactiveManager v3...")

            # 1. SignalQueue
            self._signal_queue = SignalQueue(self._config)

            # 2. SignalGenerator
            self._signal_generator = SignalGenerator(self._config)

            # 3. ExpectationStore
            self._expectation_store = ExpectationStore()

            # 4. GroupScheduler
            self._group_scheduler = GroupScheduler(
                config=self._config,
                signal_queue=self._signal_queue,
            )
            self._group_scheduler.set_reply_callback(self._handle_signal_reply)
            if self._llm_provider and self._config.proactive_mode == "hybrid":
                self._group_scheduler.set_llm_confirm_callback(
                    self._handle_llm_confirm
                )

            # 5. FollowUpPlanner
            self._followup_planner = FollowUpPlanner(
                config=self._config,
                expectation_store=self._expectation_store,
            )
            self._followup_planner.set_followup_reply_callback(
                self._handle_followup_reply
            )
            if self._llm_provider:
                self._followup_planner.set_llm_decide_callback(
                    self._handle_followup_llm_decide
                )

            self._initialized = True
            logger.info("ProactiveManager v3 initialized successfully")

    # ── 消息处理（v2 兼容接口）────────────────────────────────

    async def process_message(
        self,
        messages: List[Dict[str, Any]],
        user_id: str,
        session_key: str,
        session_type: str = "group",
        group_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """处理消息，生成信号入队

        v3 不再同步返回回复决定，而是信号入队 + 定时聚合。
        返回值保持 v2 接口兼容性但固定为 None（回复通过异步回调触发）。

        Args:
            messages: 最近消息列表
            user_id: 用户 ID
            session_key: 会话键
            session_type: 会话类型
            group_id: 群组 ID
            extra: 额外上下文

        Returns:
            None（v3 通过异步回调触发回复）
        """
        if not self._config.enabled or not self._initialized:
            return None

        # 白名单过滤
        if not self.is_group_allowed(group_id):
            return None

        # 私聊排除（不入信号队列）
        if session_type == "private" or not group_id:
            return None

        # 静音时段检查
        if self._is_quiet_hours():
            return None

        try:
            # 从消息中提取信号
            extra = extra or {}
            emotion_intensity = float(extra.get("emotion_intensity", 0.0))

            # 取最近一条消息文本
            text = ""
            if messages:
                last_msg = messages[-1]
                text = last_msg.get("text", "") or last_msg.get("content", "")

            if not text:
                return None

            # 生成信号
            signals = self._signal_generator.generate(
                text=text,
                user_id=user_id,
                group_id=group_id,
                session_key=session_key,
                emotion_intensity=emotion_intensity,
            )

            # 入队
            for signal in signals:
                self._signal_queue.enqueue(signal)

            # 更新最后消息时间
            self._signal_queue.update_last_message_time(group_id)

            # 记录群组 UMO（用于后续主动回复发送）
            umo = (extra or {}).get("umo")
            if umo and group_id:
                self._group_umo_map[group_id] = umo

            # 确保群定时器运行
            if self._config.signal_queue_enabled and self._group_scheduler:
                self._group_scheduler.ensure_timer(group_id)

            # 通知 FollowUpPlanner（如有活跃期待）
            if self._followup_planner:
                sender_name = ""
                if messages:
                    sender_name = messages[-1].get("sender_name", "")
                self._followup_planner.on_user_message(
                    user_id=user_id,
                    group_id=group_id,
                    message=text,
                    sender_name=sender_name,
                )

        except Exception as e:
            logger.error(f"ProactiveManager.process_message failed: {e}")

        return None

    def clear_pending_tasks_for_session(
        self,
        user_id: str,
        group_id: Optional[str] = None,
    ) -> None:
        """清除会话的待处理任务

        当 Bot 正常回复后调用，清除该会话的信号和跟进期待。
        注意：如果启用了 followup_after_all_replies，仅清除信号队列，
        保留 FollowUp 期待（因为 notify_bot_reply 会重新创建）。

        Args:
            user_id: 用户 ID
            group_id: 群组 ID
        """
        session_key = self._build_session_key(user_id, group_id)

        if self._signal_queue:
            self._signal_queue.clear_session(session_key)

        # 仅在未启用 followup_after_all_replies 时清除 FollowUp 期待
        if not self._config.followup_after_all_replies:
            if self._followup_planner and group_id:
                self._followup_planner.clear_expectation(group_id)

        logger.debug(f"Cleared pending tasks for session {session_key}")

    def notify_bot_reply(
        self,
        user_id: str,
        group_id: Optional[str],
        user_message: str,
        bot_reply: str,
    ) -> None:
        """Bot 回复后通知，创建 FollowUp 跟进期待

        当 followup_after_all_replies 启用时，由 message_processor
        在每次 Bot 回复后调用此方法，为群聊回复创建 FollowUp 期待。

        Args:
            user_id: 触发用户 ID
            group_id: 群组 ID（私聊为 None，会跳过）
            user_message: 用户原始消息
            bot_reply: Bot 回复内容
        """
        if not self._initialized:
            return

        if not self._config.followup_after_all_replies:
            return

        if not self._config.followup_enabled:
            return

        # 仅群聊创建期待
        if not group_id:
            return

        # 白名单过滤
        if not self.is_group_allowed(group_id):
            return

        if not self._followup_planner:
            return

        session_key = self._build_session_key(user_id, group_id)

        # 先清除旧期待再创建新的
        self._followup_planner.clear_expectation(group_id)

        self._followup_planner.create_expectation(
            session_key=session_key,
            group_id=group_id,
            trigger_user_id=user_id,
            trigger_message=user_message,
            bot_reply_summary=bot_reply[:200] if len(bot_reply) > 200 else bot_reply,
            recent_context=[],
        )

        logger.debug(
            f"FollowUp expectation created after bot reply: "
            f"group={group_id}, user={user_id}"
        )

    # ── 回调处理 ──────────────────────────────────────────────

    async def _handle_signal_reply(self, decision: AggregatedDecision) -> None:
        """处理 SignalQueue 聚合决策触发的回复

        完整流程：
        1. 检查 FollowUp 冲突和每日限额
        2. 构建回复结果
        3. 通过 ReplySender 发送（经过记忆/画像注入 → LLM → 平台发送）
        4. 创建 FollowUp 期待

        Args:
            decision: 聚合决策
        """
        # 检查是否有活跃 FollowUp 期待 → 延迟
        if self._followup_planner and self._followup_planner.has_active_expectation(
            decision.group_id
        ):
            logger.debug(
                f"Active FollowUp expectation for group {decision.group_id}, "
                f"deferring signal reply"
            )
            return

        # 检查每日限额
        if not self._check_daily_limit(decision.target_user_id):
            logger.debug(
                f"Daily limit reached for user {decision.target_user_id}"
            )
            return

        # 构建回复结果
        reply_result = self._build_signal_reply(decision)

        # 通过 ReplySender 发送实际回复
        bot_reply = await self._send_proactive_reply(reply_result)

        if not bot_reply:
            logger.debug(
                f"Signal reply not sent for group {decision.group_id}"
            )
            return

        # 创建 FollowUp 期待
        if self._followup_planner and self._config.followup_enabled:
            trigger_msg = ""
            if decision.recent_messages:
                trigger_msg = decision.recent_messages[-1].get("content", "")

            self._followup_planner.create_expectation(
                session_key=decision.session_key,
                group_id=decision.group_id,
                trigger_user_id=decision.target_user_id,
                trigger_message=trigger_msg,
                bot_reply_summary=bot_reply[:200],
                recent_context=decision.recent_messages,
            )

        self._increment_daily_count(decision.target_user_id)

    async def _handle_followup_reply(
        self,
        reply_result: ProactiveReplyResult,
        expectation: FollowUpExpectation,
    ) -> None:
        """处理 FollowUp 触发的回复

        完整流程：
        1. 清除该群的信号队列
        2. 通过 ReplySender 发送（经过记忆/画像注入 → LLM → 平台发送）
        3. 记录发送结果

        Args:
            reply_result: 回复结果
            expectation: 跟进期待
        """
        # 清除该群的信号
        if self._signal_queue:
            self._signal_queue.clear_group(expectation.group_id)

        # 通过 ReplySender 发送实际回复
        bot_reply = await self._send_proactive_reply(reply_result)

        if bot_reply:
            logger.info(
                f"FollowUp reply sent: group={expectation.group_id}, "
                f"user={expectation.trigger_user_id}, "
                f"count={expectation.followup_count + 1}, "
                f"reply_len={len(bot_reply)}"
            )
        else:
            logger.debug(
                f"FollowUp reply not sent for group={expectation.group_id}"
            )

    async def _send_proactive_reply(
        self,
        reply_result: ProactiveReplyResult,
    ) -> Optional[str]:
        """通过 ReplySender 发送主动回复

        经过记忆上下文注入、LLM 生成、平台发送的完整流程。

        Args:
            reply_result: 主动回复结果

        Returns:
            Bot 回复文本，失败返回 None
        """
        if not self._reply_sender:
            logger.warning(
                "ProactiveManager: reply_sender not set, "
                "cannot send proactive reply. "
                "Ensure set_reply_sender() is called after initialization."
            )
            return None

        try:
            return await self._reply_sender.send_reply(reply_result)
        except Exception as e:
            logger.error(f"Failed to send proactive reply: {e}")
            return None

    async def _handle_llm_confirm(
        self,
        group_id: str,
        signals: List[Signal],
        recent_messages: List[Dict[str, Any]],
    ) -> bool:
        """处理 GroupScheduler 的 LLM 确认请求

        使用 LLM 判断是否应该主动回复。

        Returns:
            是否应回复
        """
        try:
            msg_text = "\n".join(
                f"  {m.get('sender_id', '用户')}: {m.get('content', '')}"
                for m in recent_messages[-5:]
            )
            signal_info = ", ".join(
                f"{s.signal_type.value}(w={s.weight:.2f})"
                for s in signals[:5]
            )

            prompt = (
                "判断以下群聊消息是否需要 Bot 主动回复。\n"
                f"\n检测到的信号：{signal_info}\n"
                f"\n近期对话：\n{msg_text}\n"
                "\n仅回答 yes 或 no。"
            )

            result = await call_llm(
                context=self._astrbot_context,
                provider=self._llm_provider,
                provider_id=self._llm_provider_id,
                prompt=prompt,
            )

            if result.success and result.content:
                return "yes" in result.content.lower().strip()

            return False
        except Exception as e:
            logger.warning(f"LLM confirm failed: {e}")
            return False

    async def _handle_followup_llm_decide(
        self, expectation: FollowUpExpectation
    ) -> Optional[FollowUpDecision]:
        """处理 FollowUpPlanner 的 LLM 判断请求"""
        try:
            prompt = build_followup_prompt(expectation)

            result = await call_llm(
                context=self._astrbot_context,
                provider=self._llm_provider,
                provider_id=self._llm_provider_id,
                prompt=prompt,
            )

            if result.success and result.content:
                return parse_followup_response(result.content)

            return None
        except Exception as e:
            logger.warning(f"FollowUp LLM decision failed: {e}")
            return None

    # ── 回复构建 ──────────────────────────────────────────────

    @staticmethod
    def _build_signal_reply(decision: AggregatedDecision) -> ProactiveReplyResult:
        """从聚合决策构建回复结果"""
        msg_summary = "\n".join(
            f"  {m.get('sender_id', '用户')}: {m.get('content', '')}"
            for m in decision.recent_messages[-5:]
        )

        # 提取信号类型摘要
        signal_types = list({s.signal_type.value for s in decision.signals})

        trigger_prompt = (
            "【主动回复场景】\n"
            "你正在主动向用户发起对话，而不是回复用户的消息。\n"
            f"触发原因：{decision.reason}\n"
            f"检测到的信号：{', '.join(signal_types)}\n"
        )

        if msg_summary:
            trigger_prompt += f"\n近期对话记录：\n{msg_summary}\n"

        trigger_prompt += (
            f"\n对话对象：{decision.target_user_id}\n"
            "\n行为指导：\n"
            "- 你的消息应该自然、简短，像是你忽然想到了什么而发起的对话\n"
            "- 不要提及'系统检测'、'主动回复'等元信息\n"
            "- 结合你对用户的记忆和近期话题来开启对话\n"
            "- 避免重复之前已经讨论过的内容\n"
            "- 语气要符合你的人格设定\n"
            "- 如果是群聊环境，注意适度存在感，不要过度介入\n"
        )

        return ProactiveReplyResult(
            trigger_prompt=trigger_prompt,
            reply_params={
                "max_tokens": 150,
                "temperature": 0.7,
            },
            reason=decision.reason,
            group_id=decision.group_id,
            session_key=decision.session_key,
            target_user=decision.target_user_id,
            recent_messages=decision.recent_messages,
            source="signal_queue",
        )

    # ── 配额控制 ──────────────────────────────────────────────

    def _check_daily_limit(self, user_id: str) -> bool:
        """检查每日限额"""
        self._refresh_daily_counter()

        if self._daily_reply_count >= self._config.max_daily_replies:
            return False

        user_count = self._per_user_daily.get(user_id, 0)
        if user_count >= self._config.max_daily_per_user:
            return False

        return True

    def _increment_daily_count(self, user_id: str) -> None:
        """增加每日计数"""
        self._refresh_daily_counter()
        self._daily_reply_count += 1
        self._per_user_daily[user_id] = self._per_user_daily.get(user_id, 0) + 1

    def _refresh_daily_counter(self) -> None:
        """刷新每日计数器（按日重置）"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._daily_reply_date:
            self._daily_reply_count = 0
            self._per_user_daily.clear()
            self._daily_reply_date = today

    def _is_quiet_hours(self) -> bool:
        """检查是否在静音时段"""
        quiet = self._config.quiet_hours
        if not quiet or len(quiet) < 2:
            return False

        hour = datetime.now().hour
        start, end = quiet[0], quiet[1]

        if start <= end:
            return start <= hour < end
        else:
            # 跨午夜
            return hour >= start or hour < end

    # ── 白名单管理（v2 兼容）──────────────────────────────────

    @property
    def group_whitelist(self) -> List[str]:
        """已加入白名单的群组 ID 列表"""
        return self._group_whitelist

    @group_whitelist.setter
    def group_whitelist(self, value: List[str]) -> None:
        self._group_whitelist = list(value) if value else []

    @property
    def group_whitelist_mode(self) -> bool:
        """True = 白名单模式"""
        return self._group_whitelist_mode

    @group_whitelist_mode.setter
    def group_whitelist_mode(self, value: bool) -> None:
        self._group_whitelist_mode = bool(value)

    def is_group_allowed(self, group_id: Optional[str]) -> bool:
        """判断群组是否允许主动回复"""
        if not group_id:
            return True
        if not self._group_whitelist_mode:
            return True
        return group_id in self._group_whitelist

    def add_group_to_whitelist(self, group_id: str) -> bool:
        """将群组加入白名单"""
        if group_id in self._group_whitelist:
            return False
        self._group_whitelist.append(group_id)
        logger.info(f"Group {group_id} added to proactive whitelist")
        return True

    def remove_group_from_whitelist(self, group_id: str) -> bool:
        """将群组从白名单移除"""
        if group_id not in self._group_whitelist:
            return False
        self._group_whitelist.remove(group_id)
        logger.info(f"Group {group_id} removed from proactive whitelist")
        return True

    def is_group_in_whitelist(self, group_id: str) -> bool:
        """判断群组是否在白名单中"""
        return group_id in self._group_whitelist

    def get_whitelist(self) -> List[str]:
        """获取当前白名单列表"""
        return list(self._group_whitelist)

    def serialize_whitelist(self) -> Dict[str, Any]:
        """序列化白名单状态"""
        return {
            "group_whitelist": self._group_whitelist,
            "group_whitelist_mode": self._group_whitelist_mode,
        }

    def deserialize_whitelist(self, data: Any) -> None:
        """从 KV 存储反序列化白名单状态"""
        if not isinstance(data, dict):
            return
        default_mode = self._group_whitelist_mode
        self._group_whitelist = list(data.get("group_whitelist", []))
        self._group_whitelist_mode = bool(data.get("group_whitelist_mode", False))

        if default_mode != self._group_whitelist_mode and self._group_whitelist_mode:
            logger.info(
                f"Whitelist mode restored from KV storage "
                f"(mode={self._group_whitelist_mode}, "
                f"groups={len(self._group_whitelist)})"
            )
        else:
            logger.debug(
                f"Whitelist loaded: mode={self._group_whitelist_mode}, "
                f"groups={self._group_whitelist}"
            )

    # ── 统计与辅助 ──────────────────────────────────────────────

    def _build_session_key(
        self,
        user_id: str,
        group_id: Optional[str] = None,
    ) -> str:
        """构建会话键"""
        if group_id:
            return f"{user_id}:{group_id}"
        return f"{user_id}:private"

    async def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """获取统计数据"""
        return {
            "enabled": self._config.enabled,
            "mode": self._config.proactive_mode,
            "signal_queue_enabled": self._config.signal_queue_enabled,
            "followup_enabled": self._config.followup_enabled,
            "daily_reply_count": self._daily_reply_count,
            "total_signals": (
                self._signal_queue.total_signals if self._signal_queue else 0
            ),
            "active_groups": (
                self._group_scheduler.active_group_count
                if self._group_scheduler else 0
            ),
            "active_expectations": (
                self._followup_planner.active_expectation_count
                if self._followup_planner else 0
            ),
        }

    async def close(self) -> None:
        """关闭并释放所有资源"""
        if self._group_scheduler:
            await self._group_scheduler.close()
            self._group_scheduler = None

        if self._followup_planner:
            await self._followup_planner.close()
            self._followup_planner = None

        self._signal_queue = None
        self._signal_generator = None
        self._expectation_store = None

        self._initialized = False
        logger.info("ProactiveManager v3 closed")

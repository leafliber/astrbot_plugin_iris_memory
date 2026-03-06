"""
主动回复管理器 v3（Facade）

统一入口，编排 SignalQueue + GroupScheduler + FollowUpPlanner。
保持与 v2 相同的外部接口（process_message、白名单管理等），
内部全部替换为新架构。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from iris_memory.config import get_store
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
from iris_memory.proactive.reply_coordinator import ReplyCoordinator
from iris_memory.proactive.reply_sender import ProactiveReplySender
from iris_memory.proactive.signal_generator import SignalGenerator
from iris_memory.proactive.signal_queue import SignalQueue
from iris_memory.proactive.storage.expectation_store import ExpectationStore
from iris_memory.utils.llm_helper import call_llm, parse_llm_json
from iris_memory.utils.logger import get_logger
from iris_memory.utils.bounded_dict import BoundedDict

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
        llm_provider: Optional[Any] = None,
        enabled: bool = True,
        group_whitelist_mode: bool = False,
        proactive_mode: str = "rule",
        # 以下为 v2 兼容参数（已废弃，仅保留接口兼容性）
        chroma_manager: Optional[Any] = None,
        embedding_manager: Optional[Any] = None,
        shared_state: Optional[Any] = None,
        personality: str = "balanced",
        quiet_hours: Optional[List[int]] = None,
        max_history: int = 10,
        max_text_tokens: int = 150,
        config: Optional[Any] = None,  # 已废弃，保留用于兼容旧测试
    ) -> None:
        self._plugin_data_path = plugin_data_path
        self._llm_provider = llm_provider
        self._astrbot_context: Optional[Any] = None
        self._llm_provider_id: Optional[str] = None

        # 配置直接从全局 get_store() 获取
        self._cfg = get_store()

        # 白名单状态（持久化到 KV 存储）
        self._group_whitelist: List[str] = list(
            self._cfg.get("proactive_reply.group_whitelist", [])
        )
        self._group_whitelist_mode: bool = self._cfg.get(
            "proactive_reply.group_whitelist_mode", False
        )

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
        self._per_user_daily: BoundedDict[str, int] = BoundedDict(max_size=10000)

        # 冷却时间控制（group_id -> 最后回复时间戳）
        self._last_reply_time: Dict[str, float] = {}

        # 群组最近用户活动时间（用于智能静音豁免，group_id -> datetime）
        self._group_last_activity: BoundedDict[str, datetime] = BoundedDict(max_size=2000)

        # 回复协调器：集中管理正常回复与主动回复的竞态关系
        self._reply_coordinator = ReplyCoordinator()

    def _get_cfg(self, key: str, default: Any = None) -> Any:
        """辅助方法：从配置存储获取值"""
        return self._cfg.get(f"proactive_reply.{key}", default)

        # CooldownModule 状态检查回调（group_id -> 是否处于冷却中）
        # 由服务层注入，检查用户通过 /冷却 命令激活的冷却状态
        self._cooldown_checker: Optional[Callable[[str], bool]] = None

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def reply_coordinator(self) -> ReplyCoordinator:
        """回复协调器实例

        供 MessageProcessor 等外部组件访问，用于标记正常回复的起止。
        """
        return self._reply_coordinator

    @property
    def enabled(self) -> bool:
        return self._cfg.get("proactive_reply.enable", False)

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

    def set_cooldown_checker(self, checker: Callable[[str], bool]) -> None:
        """注入 CooldownModule 状态检查回调

        由服务层注入，用于检查用户通过 /冷却 命令激活的冷却状态。
        主动回复在发送前会调用此回调，若群处于冷却中则跳过回复。

        Args:
            checker: 接受 group_id，返回 True 表示该群处于冷却中
        """
        self._cooldown_checker = checker
        logger.debug("Cooldown checker injected into ProactiveManager")

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
            self._signal_queue = SignalQueue()

            # 2. SignalGenerator
            self._signal_generator = SignalGenerator()

            # 3. ExpectationStore
            self._expectation_store = ExpectationStore()

            # 4. GroupScheduler
            self._group_scheduler = GroupScheduler(
                signal_queue=self._signal_queue,
            )
            self._group_scheduler.set_reply_callback(self._handle_signal_reply)
            proactive_mode = self._cfg.get("proactive_reply.proactive_mode", "rule")
            if self._llm_provider and proactive_mode == "hybrid":
                self._group_scheduler.set_llm_confirm_callback(
                    self._handle_llm_confirm
                )

            # 5. FollowUpPlanner
            self._followup_planner = FollowUpPlanner(
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
        if not self._get_cfg("enable", False) or not self._initialized:
            return None

        # 白名单过滤
        if not self.is_group_allowed(group_id):
            return None

        # 私聊排除（不入信号队列）
        if session_type == "private" or not group_id:
            return None

        # 追踪群组活动时间（用于智能静音豁免）
        self._group_last_activity[group_id] = datetime.now()

        # 静音时段检查
        if self._is_quiet_hours(group_id):
            return None

        # 竞态防护：正常回复进行中或冷却期内，抑制信号生成
        if group_id:
            if not self._reply_coordinator.can_send_proactive(
                group_id, self._get_cfg("cooldown_seconds", 60)
            ):
                logger.debug(
                    f"Reply coordinator suppressed proactive signal "
                    f"for group {group_id}"
                )
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
            if True and self._group_scheduler:
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

        # 通过协调器记录正常回复完成，更新时间戳
        if group_id:
            self._reply_coordinator.mark_normal_reply_end(
                group_id, self._get_cfg("cooldown_seconds", 60)
            )

        if self._signal_queue:
            self._signal_queue.clear_session(session_key)

        # 仅在未启用 followup_after_all_replies 时清除 FollowUp 期待
        if not self._get_cfg("followup_after_all_replies", False):
            if self._followup_planner and group_id:
                self._followup_planner.clear_expectation(group_id)

        logger.debug(f"Cleared pending tasks for session {session_key}")

    def notify_message_for_followup(
        self,
        user_id: str,
        group_id: str,
        message: str,
        sender_name: str = "",
    ) -> None:
        """实时消息通知 FollowUpPlanner

        在消息处理流程中直接调用，绕过批量处理延迟，
        确保 FollowUp 短期窗口内能及时收到用户消息。
        仅在有活跃 FollowUp 期待时才执行，开销极小。

        Args:
            user_id: 用户 ID
            group_id: 群组 ID
            message: 消息内容
            sender_name: 发送者名称
        """
        if not self._initialized or not self._followup_planner:
            return
        if not self._followup_planner.has_active_expectation(group_id):
            return
        try:
            self._followup_planner.on_user_message(
                user_id=user_id,
                group_id=group_id,
                message=message,
                sender_name=sender_name,
            )
        except Exception as e:
            logger.warning(f"notify_message_for_followup failed: {e}")

    def notify_bot_reply(
        self,
        user_id: str,
        group_id: Optional[str],
        user_message: str,
        bot_reply: str,
        umo: Optional[str] = None,
    ) -> None:
        """Bot 回复后通知，创建 FollowUp 跟进期待

        当 followup_after_all_replies 启用时，由 message_processor
        在每次 Bot 回复后调用此方法，为群聊回复创建 FollowUp 期待。

        Args:
            user_id: 触发用户 ID
            group_id: 群组 ID（私聊为 None，会跳过）
            user_message: 用户原始消息
            bot_reply: Bot 回复内容
            umo: unified_msg_origin，用于后续主动回复发送
        """
        followup_after_all = self._get_cfg("followup_after_all_replies", False)
        followup_enabled = self._get_cfg("followup_enabled", True)
        logger.debug(
            f"notify_bot_reply called: user={user_id}, group={group_id}, "
            f"initialized={self._initialized}, "
            f"followup_after_all={followup_after_all}, "
            f"followup_enabled={followup_enabled}"
        )
        
        if not self._initialized:
            logger.debug("notify_bot_reply: not initialized, skipping")
            return

        if not self._get_cfg("followup_after_all_replies", False):
            logger.debug("notify_bot_reply: followup_after_all_replies disabled, skipping")
            return

        if not self._get_cfg("followup_enabled", True):
            logger.debug("notify_bot_reply: followup_enabled disabled, skipping")
            return

        # 仅群聊创建期待
        if not group_id:
            logger.debug("notify_bot_reply: not group chat, skipping")
            return

        # 白名单过滤
        if not self.is_group_allowed(group_id):
            logger.debug(f"notify_bot_reply: group {group_id} not in whitelist, skipping")
            return

        if not self._followup_planner:
            logger.warning("notify_bot_reply: _followup_planner is None, skipping")
            return

        # 记录群组 UMO（用于后续 FollowUp 回复发送）
        if umo and group_id:
            self._group_umo_map[group_id] = umo

        session_key = self._build_session_key(user_id, group_id)

        # 检查是否有活跃期待且包含待处理的聚合消息
        # 如果有，保留聚合消息并携带到新期待中，避免丢失用户消息和取消正在运行的定时器
        existing = self._followup_planner.get_expectation(group_id)
        if existing and existing.has_aggregated_messages:
            carried_messages = list(existing.aggregated_messages)
            logger.info(
                f"notify_bot_reply: carrying over {len(carried_messages)} "
                f"aggregated messages from old expectation for group {group_id}"
            )
        else:
            carried_messages = []

        # 先清除旧期待再创建新的
        self._followup_planner.clear_expectation(group_id)

        new_exp = self._followup_planner.create_expectation(
            session_key=session_key,
            group_id=group_id,
            trigger_user_id=user_id,
            trigger_message=user_message,
            bot_reply_summary=bot_reply[:200] if len(bot_reply) > 200 else bot_reply,
            recent_context=[],
        )

        # 恢复聚合消息并重新启动短期窗口定时器
        if new_exp and carried_messages:
            new_exp.aggregated_messages = carried_messages
            short_window = self._get_cfg("followup_short_window_seconds", 10)
            self._followup_planner.restart_short_window_timer(
                group_id, short_window
            )

        logger.debug(
            f"FollowUp expectation created after bot reply: "
            f"group={group_id}, user={user_id}"
        )

    # ── 回调处理 ──────────────────────────────────────────────

    async def _handle_signal_reply(self, decision: AggregatedDecision) -> None:
        """处理 SignalQueue 聚合决策触发的回复

        完整流程：
        1. 检查静音时段、冷却时间、FollowUp 冲突和每日限额
        2. 构建回复结果
        3. 通过 ReplySender 发送（经过记忆/画像注入 → LLM → 平台发送）
        4. 创建 FollowUp 期待

        Args:
            decision: 聚合决策
        """
        # 检查静音时段
        if self._is_quiet_hours(decision.group_id):
            logger.debug(
                f"Quiet hours, skipping signal reply for group {decision.group_id}"
            )
            return

        # 检查冷却时间
        if not self._check_rate_limit(decision.group_id):
            logger.debug(
                f"Rate limit: group {decision.group_id} in cooldown"
            )
            return

        # 检查 CooldownModule（用户通过 /冷却 命令激活的冷却状态）
        if self._cooldown_checker and self._cooldown_checker(decision.group_id):
            logger.debug(
                f"CooldownModule active for group {decision.group_id}, "
                f"skipping signal reply"
            )
            return

        # 检查每日限额（锁外快速拒绝）
        if not self._check_daily_limit(decision.target_user_id):
            logger.debug(
                f"Daily limit reached for user {decision.target_user_id}"
            )
            return

        # 通过协调器守卫发送：获取群锁 + 二次校验正常回复状态 + FollowUp 检查
        async with self._reply_coordinator.proactive_reply_guard(
            decision.group_id, self._get_cfg("cooldown_seconds", 60)
        ) as allowed:
            if not allowed:
                logger.debug(
                    f"Reply coordinator blocked signal reply "
                    f"for group {decision.group_id}"
                )
                return

            # 锁内检查 FollowUp 期待（消除 TOCTOU 窗口）
            if self._followup_planner and self._followup_planner.has_active_expectation(
                decision.group_id
            ):
                logger.debug(
                    f"Active FollowUp expectation for group {decision.group_id}, "
                    f"deferring signal reply"
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

            # 更新冷却时间
            self._update_last_reply_time(decision.group_id)

            # 创建 FollowUp 期待
            if self._followup_planner and self._get_cfg("followup_enabled", True):
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
    ) -> bool:
        """处理 FollowUp 触发的回复

        完整流程：
        1. 检查静音时段、冷却时间和每日限额
        2. 清除该群的信号队列
        3. 通过 ReplySender 发送（经过记忆/画像注入 → LLM → 平台发送）
        4. 记录发送结果

        Args:
            reply_result: 回复结果
            expectation: 跟进期待

        Returns:
            True 表示回复已成功发送，False 表示被阻断（静音/冷却等）
        """
        # 检查静音时段
        if self._is_quiet_hours(expectation.group_id):
            logger.debug(
                f"Quiet hours, skipping followup reply for group {expectation.group_id}"
            )
            return False

        # 检查冷却时间
        if not self._check_rate_limit(expectation.group_id):
            logger.debug(
                f"Rate limit: group {expectation.group_id} in cooldown, "
                f"skipping followup"
            )
            return False

        # 检查 CooldownModule（用户通过 /冷却 命令激活的冷却状态）
        if self._cooldown_checker and self._cooldown_checker(expectation.group_id):
            logger.debug(
                f"CooldownModule active for group {expectation.group_id}, "
                f"skipping followup reply"
            )
            return False

        # 检查每日限额（锁外快速拒绝）
        if not self._check_daily_limit(expectation.trigger_user_id):
            logger.debug(
                f"Daily limit reached for user {expectation.trigger_user_id}, "
                f"skipping followup"
            )
            return False

        # 通过协调器守卫发送：获取群锁 + 二次校验正常回复状态
        reply_sent = False
        async with self._reply_coordinator.proactive_reply_guard(
            expectation.group_id, self._get_cfg("cooldown_seconds", 60)
        ) as allowed:
            if not allowed:
                logger.debug(
                    f"Reply coordinator blocked followup reply "
                    f"for group {expectation.group_id}"
                )
                return False

            # 清除该群的信号
            if self._signal_queue:
                self._signal_queue.clear_group(expectation.group_id)

            # 通过 ReplySender 发送实际回复
            bot_reply = await self._send_proactive_reply(reply_result)

            if bot_reply:
                # 更新冷却时间和计数
                self._update_last_reply_time(expectation.group_id)
                self._increment_daily_count(expectation.trigger_user_id)
                reply_sent = True

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

        return reply_sent

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

    async def send_direct_message(
        self,
        user_id: str,
        group_id: Optional[str],
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """发送预格式化文本消息（不经过 LLM 生成）

        用于记忆回顾通知等需要直接投递文本的场景。

        Args:
            user_id: 目标用户 ID
            group_id: 目标群组 ID
            text: 要发送的文本
            metadata: 可选的附加元数据

        Returns:
            是否发送成功
        """
        if not self._reply_sender:
            logger.warning(
                "ProactiveManager.send_direct_message: "
                "reply_sender not set, cannot send message"
            )
            return False

        umo = self._group_umo_map.get(group_id) if group_id else None
        if not umo:
            logger.warning(
                f"ProactiveManager.send_direct_message: "
                f"no UMO for group={group_id}, cannot send message"
            )
            return False

        try:
            await self._reply_sender.send_text(umo, text)
            logger.debug(
                f"Direct message sent: user={user_id}, group={group_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send direct message: {e}")
            return False

    async def send_review_prompt(
        self,
        user_id: str,
        group_id: Optional[str],
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """发送记忆回顾提示（不经过 LLM 生成）

        用于宽限期回顾等需要直接投递提示文本的场景。

        Args:
            user_id: 目标用户 ID
            group_id: 目标群组 ID
            prompt: 提示文本
            metadata: 可选的附加元数据

        Returns:
            是否发送成功
        """
        return await self.send_direct_message(
            user_id=user_id,
            group_id=group_id,
            text=prompt,
            metadata=metadata,
        )

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

            # 提取信号元数据中的匹配规则
            matched_rules: List[str] = []
            for s in signals:
                rules = s.metadata.get("matched_rules", [])
                matched_rules.extend(rules)
            rules_desc = ", ".join(set(matched_rules)) if matched_rules else "无"

            prompt = (
                "你是一个智能助手，需要判断以下群聊消息是否需要 Bot 主动回复。\n"
                "\n【检测到的信号】\n"
                f"信号详情：{signal_info}\n"
                f"匹配规则：{rules_desc}\n"
                "\n【近期对话】\n"
                f"{msg_text}\n"
                "\n【判断标准】\n"
                "- 用户是否在向 Bot 提问或寻求互动？\n"
                "- 用户是否表达了强烈情感需要回应？\n"
                "- 参与是否自然，还是会显得强行介入？\n"
                "- 宁可保守不回，不要过度介入\n"
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

    def _check_rate_limit(self, group_id: str) -> bool:
        """检查冷却时间限制

        Args:
            group_id: 群组 ID

        Returns:
            True 如果可以通过（冷却时间已过），False 如果还在冷却中
        """
        cooldown = self._get_cfg("cooldown_seconds", 60)
        if cooldown <= 0:
            return True

        last_time = self._last_reply_time.get(group_id)
        if last_time is None:
            return True

        import time
        elapsed = time.time() - last_time
        if elapsed < cooldown:
            logger.debug(
                f"Rate limit: group {group_id} in cooldown "
                f"({elapsed:.0f}s < {cooldown}s)"
            )
            return False

        return True

    def _update_last_reply_time(self, group_id: str) -> None:
        """更新最后回复时间"""
        import time
        self._last_reply_time[group_id] = time.time()

    def _check_daily_limit(self, user_id: str) -> bool:
        """检查每日限额"""
        self._refresh_daily_counter()

        if self._daily_reply_count >= self._get_cfg("max_daily_replies", 20):
            return False

        user_count = self._per_user_daily.get(user_id, 0)
        if user_count >= self._get_cfg("max_daily_per_user", 5):
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

    def _is_quiet_hours(self, group_id: Optional[str] = None) -> bool:
        """检查是否在静音时段

        使用配置的时区偏移（默认 UTC+8 北京时间）来判断当前小时，
        避免服务器时区与用户时区不一致导致静音失效。

        Args:
            group_id: 群组 ID，用于智能静音豁免检测
        """
        quiet = self._cfg.get("proactive_reply.quiet_hours", [23, 7])
        if not quiet or len(quiet) < 2:
            return False

        tz = timezone(timedelta(hours=self._cfg.get("proactive_reply.timezone_offset", 8)))
        hour = datetime.now(tz).hour
        start, end = quiet[0], quiet[1]

        if start <= end:
            in_quiet = start <= hour < end
        else:
            # 跨午夜
            in_quiet = hour >= start or hour < end

        if not in_quiet:
            return False

        # 智能静音：若群组近期有用户活动，豁免静音时段
        exempt_minutes = self._cfg.get("proactive_reply.quiet_hours_activity_exempt_minutes", 20)
        if exempt_minutes > 0 and group_id and group_id in self._group_last_activity:
            last_activity = self._group_last_activity[group_id]
            elapsed = (datetime.now() - last_activity).total_seconds() / 60
            if elapsed <= exempt_minutes:
                logger.debug(
                    f"Smart quiet hours: group {group_id} was active "
                    f"{elapsed:.1f}min ago, exempting from quiet hours"
                )
                return False

        return True

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
            "enabled": self._get_cfg("enable", False),
            "mode": self._cfg.get("proactive_reply.proactive_mode", "rule"),
            "signal_queue_enabled": True,
            "followup_enabled": self._get_cfg("followup_enabled", True),
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

        self._reply_coordinator.clear_all()

        self._initialized = False
        logger.info("ProactiveManager v3 closed")

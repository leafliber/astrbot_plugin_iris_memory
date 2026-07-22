"""
Iris Memory - AstrBot 整合记忆插件（分层记忆 + 统一决策主动回复）

v3.0 架构：
- 记忆侧（源自 Iris Chat Memory 轻量方案）：
  L1 消息上下文缓冲 / L2 记忆库（FAISS + SQLite）/ L3 知识图谱（SQLite）
  + 用户/群聊画像 + 梦境离线加工 + 图片解析
- 主动回复侧（源自 Iris Reply 统一决策模型）：
  chime_in 跟话 / follow_up 跟进 / initiate 发起 / watch 被动评估，
  SignalGate 本地零成本门控 + 单次 LLM 统一决策 + ThreadAnchor 记账

钩子编排（等价于原两插件并存时的兼容性契约）：
  群消息 → on_message（主动回复门控，设 iris_mode extra）
        → on_all_message（入 L1、图片入队）
  门控命中 → 统一决策（llm_generate 直调，不触发钩子）
        → 决策发言：劫持主管线，注入 SPEAK_HINTS（mark_as_temp）
  on_llm_request → 记忆侧：清空 contexts，注入 L1/L2/L3/画像（mark_as_temp）
  on_llm_response → 记忆侧：bot 回复入 L1 → 主动回复侧：按 iris_mode 记账
  after_message_sent → 主动回复侧：入滑动窗口 + 写 ThreadAnchor
  initiate 直发（context.send_message）→ 手动记账 + 回填 L1
"""

# iris_memory 必须在 sys.path 插入后再导入，故 import 不在文件顶部
# ruff: noqa: E402

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional

# 模块导入支持
plugin_root = Path(__file__).parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from iris_memory.config import init_config, Config

from astrbot.api import AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from iris_memory.core import (
    ComponentManager,
    get_logger,
    create_components,
    initialize_components,
    shutdown_components,
    handle_user_message,
    preprocess_llm_request,
    handle_llm_response,
    handle_agent_done,
    handle_pre_request_cleanup,
    handle_initiate_backfill,
    set_component_manager,
)
from iris_memory.tools import (
    SaveKnowledgeTool,
    SaveMemoryTool,
    SearchMemoryTool,
    CorrectMemoryTool,
    SearchKnowledgeGraphTool,
    GetProfileTool,
)
from iris_memory.web import register_all_routes
from iris_memory.commands import (
    get_registry,
    execute_command,
    L1CommandHandler,
    L2CommandHandler,
    L3CommandHandler,
    ProfileCommandHandler,
    AllCommandHandler,
)
from iris_memory.proactive.admin import AdminCommands
from iris_memory.proactive.api import (
    register_web_apis as register_reply_web_apis,
    sync_stats_group_state,
)
from iris_memory.proactive.config import ConfigManager as ReplyConfigManager
from iris_memory.proactive.decision import DecisionCore, DecisionRequest
from iris_memory.proactive.perception import (
    ContextPackager,
    Gatekeeper,
    SlidingWindow,
    WindowMessage,
)
from iris_memory.proactive.prompts import SPEAK_HINTS
from iris_memory.proactive.proactive import ProactiveEngine
from iris_memory.proactive.signals import SignalGate
from iris_memory.proactive.state import StateManager
from iris_memory.proactive.stats import StatsCollector
from iris_memory.proactive.tools import ToolContext
from iris_memory.extras import ErrorFriendlyProcessor, MarkdownStripper

logger = get_logger("main")

PLUGIN_NAME = "astrbot_plugin_iris_memory"

# 旧版（v2.x）数据自动迁移开关；v4 删除本常量与 iris_memory/legacy_migration/ 即彻底移除
LEGACY_MIGRATION_ENABLED = True

_IRIS_ACTIVE_TIMEOUT = 120
_UMO_KV_KEY = "iris_reply:group_umo"


def _detect_passive_trigger(event: AstrMessageEvent, req, context: Context) -> None:
    """检测 LLM 请求是否为被动触发（sampling/主动回复）

    当用户消息不以唤醒前缀开头且未 @机器人 时，LLM 请求可能是由 AstrBot 的
    active_reply/sampling 机制触发的。此时标记事件，供后续钩子
    判断是否跳过图片解析等高 token 消耗操作。

    注：本插件主动回复触发的请求会将 is_at_or_wake_command 置 True，
    天然不会被误判为被动触发，无需额外处理 iris_mode。
    """
    try:
        is_at_or_wake = getattr(event, "is_at_or_wake_command", False)
        if not is_at_or_wake:
            event.set_extra("iris_passive_trigger", True)
            logger.debug(
                "检测到被动触发（sampling/主动回复），is_at_or_wake_command 为 False"
            )
    except Exception as e:
        logger.debug(f"被动触发检测异常（不影响正常流程）：{e}")


class IrisMemoryPlugin(Star):
    """AstrBot 整合记忆插件主类（分层记忆 + 统一决策主动回复）"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context: Context = context

        # ── 记忆侧初始化 ──
        data_dir = StarTools.get_data_dir()
        self.config: Config = init_config(config, data_dir)
        logger.info(f"插件数据目录：{data_dir}")

        components = create_components(context, self)
        self.component_manager: Optional[ComponentManager] = ComponentManager(
            components
        )

        set_component_manager(self.component_manager)

        from iris_memory.image.recorder_bridge import init_recorder_bridge

        init_recorder_bridge(context)

        self._register_llm_tools()
        self._register_command_handlers()
        self._register_web_api()

        # ── extras（自 v2 保留的低成本功能） ──
        self._error_processor = ErrorFriendlyProcessor(self.config)
        self._markdown_stripper = MarkdownStripper(
            context=self.context,
            config=self.config,
        )

        # ── 主动回复侧初始化 ──
        self._reply_config = ReplyConfigManager(config if config else context.get_config())
        self._state = StateManager(self._reply_config)
        self._gatekeeper = Gatekeeper(self._reply_config, self._state)
        self._sliding_window = SlidingWindow(self._reply_config)
        self._context_packager = ContextPackager(self._reply_config)
        self._signals = SignalGate(self._reply_config, self._state)
        self._decision_core = DecisionCore(
            self._reply_config, self._state, self._sliding_window, self._context_packager,
        )
        self._tool_ctx = ToolContext()
        self._admin = AdminCommands(self._state)
        self._stats = StatsCollector()
        self._reply_in_progress: dict[str, float] = {}
        self._passive_active: dict[str, float] = {}
        self._triggering: dict[str, float] = {}
        self._follow_pending: set[str] = set()
        self._group_umo: dict[str, str] = {}
        self._umo_dirty: bool = False
        self._self_id: str = ""
        self._save_task: asyncio.Task | None = None
        self._save_interval = 30
        self._proactive = ProactiveEngine(
            self.context,
            self._reply_config,
            self._state,
            self._sliding_window,
            self._signals,
            self._decision_core,
            self._stats,
            umo_get=lambda gid: self._group_umo.get(gid),
            is_busy=self._is_busy,
            self_id_get=lambda: self._self_id,
            save_fn=lambda: self._state.save_dirty(self._kv_save),
            on_initiate_sent=self._on_initiate_sent,
            text_transform=self._strip_initiate_text,
        )

        logger.info("Iris Memory 整合插件已加载（等待异步初始化）")

    # ========================================================================
    # 记忆侧注册
    # ========================================================================

    def _register_llm_tools(self) -> None:
        """注册记忆侧 LLM Tool"""
        try:
            tools = [
                SaveKnowledgeTool(),
                SaveMemoryTool(),
                SearchMemoryTool(),
                CorrectMemoryTool(),
                SearchKnowledgeGraphTool(),
                GetProfileTool(),
            ]
            self.context.add_llm_tools(*tools)
            logger.info(f"已注册 {len(tools)} 个记忆 LLM Tool")
        except Exception as e:
            logger.error(f"注册记忆 LLM Tool 失败：{e}", exc_info=True)

    def _register_command_handlers(self) -> None:
        """注册记忆侧指令处理器"""
        try:
            registry = get_registry()
            handlers = [
                L1CommandHandler(),
                L2CommandHandler(),
                L3CommandHandler(),
                ProfileCommandHandler(),
                AllCommandHandler(),
            ]
            for handler in handlers:
                registry.register(handler)
            logger.info(f"已注册 {len(handlers)} 个记忆指令处理器")
        except Exception as e:
            logger.error(f"注册记忆指令处理器失败：{e}", exc_info=True)

    def _register_web_api(self) -> None:
        try:
            register_all_routes(self.context)
        except Exception as e:
            logger.error(f"注册记忆 Web API 失败：{e}", exc_info=True)

    # ========================================================================
    # 生命周期
    # ========================================================================

    async def initialize(self) -> None:
        # 1. 记忆组件初始化
        try:
            await initialize_components(self.component_manager)
        except Exception as e:
            logger.error(f"记忆组件初始化失败：{e}", exc_info=True)

        # 2. 旧版（v2.x）数据自动迁移（独立模块，失败不阻断启动）
        if LEGACY_MIGRATION_ENABLED:
            try:
                from iris_memory.legacy_migration import migrate_if_needed

                await migrate_if_needed(
                    self.context, self, StarTools.get_data_dir(), self.component_manager
                )
            except Exception:
                logger.error("旧数据迁移失败（不影响插件启动）", exc_info=True)

        # 3. 主动回复侧初始化
        await self._state.load_all(self._kv_load)
        umo_data = await self._kv_load(_UMO_KV_KEY)
        if isinstance(umo_data, dict):
            self._group_umo = {str(k): str(v) for k, v in umo_data.items()}
        config_overrides = await self._kv_load("iris_reply:config_overrides")
        self._reply_config.load_overrides(config_overrides)
        self._save_task = asyncio.create_task(self._periodic_save())
        self._stats.enabled = self._reply_config.stats_enabled
        register_reply_web_apis(
            context=self.context,
            plugin_name=PLUGIN_NAME,
            config=self._reply_config,
            state=self._state,
            stats=self._stats,
            window=self._sliding_window,
            kv_save=self._kv_save,
        )
        await self._proactive.start()

        # 4. 功能重叠插件检测（重复注入/门控警告）
        for other in ("astrbot_plugin_iris_chat_memory", "astrbot_plugin_iris_reply"):
            try:
                if self.context.get_registered_star(other):
                    logger.warning(
                        f"检测到插件 {other} 已安装，与本插件功能重叠，"
                        "建议停用其一，避免记忆重复注入与主动回复重复门控"
                    )
            except Exception:
                pass

        logger.info("Iris Memory 整合插件异步初始化完成")

    async def terminate(self):
        """插件卸载清理"""
        logger.info("开始关闭插件组件...")
        # 主动回复侧
        await self._proactive.stop()
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        await self._state.save_all(self._kv_save)
        await self._kv_save("iris_reply:config_overrides", self._reply_config.get_overrides())
        await self._kv_save(_UMO_KV_KEY, dict(self._group_umo))
        self._follow_pending.clear()
        self._reply_in_progress.clear()
        self._passive_active.clear()
        self._triggering.clear()
        # 记忆侧
        await shutdown_components(self.component_manager)
        logger.info("Iris Memory 整合插件已卸载")

    # ========================================================================
    # 主动回复侧：状态保存与互斥
    # ========================================================================

    async def _periodic_save(self) -> None:
        while True:
            await asyncio.sleep(self._save_interval)
            try:
                await self._state.save_dirty(self._kv_save)
                await self._kv_save("iris_reply:config_overrides", self._reply_config.get_overrides())
                if self._umo_dirty:
                    self._umo_dirty = False
                    await self._kv_save(_UMO_KV_KEY, dict(self._group_umo))
                self._sliding_window.cleanup(self._state.get_whitelist())
                self._cleanup_stale_active()
                sync_stats_group_state(self._state, self._stats)
            except Exception as e:
                logger.warning(f"Iris Reply: periodic save error: {e}")

    def _cleanup_stale_active(self) -> None:
        now = time.time()
        stale_rip = [gid for gid, ts in self._reply_in_progress.items() if now - ts > _IRIS_ACTIVE_TIMEOUT]
        for gid in stale_rip:
            logger.info(f"Iris Reply: cleaning up stale reply_in_progress for group {gid} (timeout)")
            self._reply_in_progress.pop(gid, None)
        stale_passive = [gid for gid, ts in self._passive_active.items() if now - ts > _IRIS_ACTIVE_TIMEOUT]
        for gid in stale_passive:
            logger.info(f"Iris Reply: cleaning up stale passive for group {gid} (timeout)")
            self._passive_active.pop(gid, None)
        stale_triggering = [gid for gid, ts in self._triggering.items()
                            if gid not in self._reply_in_progress and now - ts > _IRIS_ACTIVE_TIMEOUT]
        for gid in stale_triggering:
            logger.info(f"Iris Reply: cleaning up stale triggering for group {gid}")
            self._triggering.pop(gid, None)

    def _is_busy(self, group_id: str) -> bool:
        return (
            group_id in self._reply_in_progress
            or group_id in self._triggering
            or group_id in self._passive_active
        )

    async def _kv_save(self, key: str, value: Any) -> None:
        await self.put_kv_data(key, value)

    async def _kv_load(self, key: str) -> Any:
        return await self.get_kv_data(key, None)

    def _get_group_id(self, event) -> str | None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
            return None
        return group_id

    async def _get_provider_id(self, event, preferred: str = "") -> str | None:
        if preferred:
            return preferred
        try:
            return await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
        except Exception:
            logger.error("Iris Reply: failed to get provider ID")
            return None

    def _strip_initiate_text(self, text: str) -> str:
        """initiate 直发消息的 Markdown 去除

        直发通路（context.send_message）不触发 on_decorating_result 钩子，
        消息始终以纯文本发送到平台，此处补齐与管线消息一致的 Markdown 去除，
        避免同群内跟话消息与主动发起消息格式处理不一致。
        """
        stripper = self._markdown_stripper
        if not stripper or not text:
            return text
        try:
            if not stripper.should_strip(text, use_t2i=False):
                return text
            return stripper.strip(text)
        except Exception as e:
            logger.warning(f"initiate 消息 Markdown 去除失败：{e}")
            return text

    async def _on_initiate_sent(self, group_id: str, text: str) -> None:
        """initiate 直发成功后，把 bot 发言回填进 L1 缓冲

        直发通路（context.send_message）不触发任何事件钩子，
        若不回填，L1 上下文中将看不到这类发起消息。
        """
        if not self.component_manager or not text:
            return
        try:
            await handle_initiate_backfill(group_id, text, self.component_manager)
        except Exception as e:
            logger.warning(f"initiate 消息回填 L1 失败：{e}")

    # ========================================================================
    # 主动回复侧：LLM 工具
    # ========================================================================

    @filter.llm_tool(name="add_follow_up")
    async def tool_add_follow_up(self, event, user_ids: str = "") -> str:
        """当你希望持续关注某些用户的发言时调用此工具。将在后续消息中匹配指定用户时自动触发回复。

        Args:
            user_ids(string): 逗号分隔的用户ID列表，如 "user1,user2"
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None

        if not uid_list:
            return "error: must provide at least one user_id"

        if len(uid_list) > 10:
            return "error: too many user_ids (max 10 per call)"

        async with self._state.get_lock(group_id):
            self._state.add_anchor_watch(group_id, users=uid_list)
        logger.debug(f"Iris Reply: add_follow_up for group {group_id}, users={uid_list}")
        return f"ok: following users={uid_list}"

    @filter.llm_tool(name="end_follow_up")
    async def tool_end_follow_up(self, event, user_ids: str = "") -> str:
        """当你不再需要关注某些用户时调用此工具，移除对应的跟进记录。不提供参数则移除所有跟进记录。

        Args:
            user_ids(string): 逗号分隔的用户ID列表，如 "user1,user2"
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None

        async with self._state.get_lock(group_id):
            self._state.remove_anchor_watch(group_id, user_ids=uid_list)
        logger.debug(f"Iris Reply: end_follow_up for group {group_id}, users={uid_list}")
        return f"ok: removed follow-up users={uid_list}"

    @filter.llm_tool(name="set_cooldown")
    async def tool_set_cooldown(self, event, minutes: int = 5) -> str:
        """当你认为应该暂时停止主动回复时调用此工具。设置冷却时间，冷却期间不会主动触发任何回复。

        Args:
            minutes(number): 冷却时间（分钟），范围 1-120，默认 5
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        async with self._state.get_lock(group_id):
            actual = self._state.set_cooldown(group_id, minutes)
        logger.debug(f"Iris Reply: set_cooldown for group {group_id}, {actual} min")
        return f"ok: cooldown set for {actual} minutes"

    # ========================================================================
    # 主动回复侧：管理指令
    # ========================================================================

    @filter.command_group("iris_reply")
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    def iris_reply_group(self):
        pass

    @iris_reply_group.command("enable")
    async def cmd_enable(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        self._state.add_to_whitelist(group_id)
        await self._state.save_dirty(self._kv_save)
        event.set_result(f"群 {group_id} 已启用主动回复")

    @iris_reply_group.command("disable")
    async def cmd_disable(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        self._state.remove_from_whitelist(group_id)
        self._sliding_window.remove_group(group_id)
        self._state.remove_group_lock(group_id)
        await self._state.save_dirty(self._kv_save)
        event.set_result(f"群 {group_id} 已禁用主动回复")

    @iris_reply_group.command("status")
    async def cmd_status(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        text = self._admin.get_status(group_id)
        event.set_result(text)

    @iris_reply_group.command("reset")
    async def cmd_reset(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        msg = self._admin.reset_group(group_id)
        await self._state.save_dirty(self._kv_save)
        event.set_result(msg)

    @iris_reply_group.command("cooldown")
    async def cmd_cooldown(self, event, minutes: int = 5) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        msg = self._admin.set_cooldown(group_id, minutes)
        await self._state.save_dirty(self._kv_save)
        event.set_result(msg)

    @iris_reply_group.command("willingness")
    async def cmd_willingness(self, event, level: str = "") -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        if not level.strip():
            current = self._admin.get_willingness(group_id)
            event.set_result(f"群 {group_id} 当前回复意愿: {current}\n可选: 低/中/高 (low/medium/high)")
            return
        msg = self._admin.set_willingness(group_id, level.strip())
        await self._state.save_dirty(self._kv_save)
        event.set_result(msg)

    @iris_reply_group.command("initiate")
    async def cmd_initiate(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        result = await self._proactive.attempt_initiate(group_id, force=True)
        event.set_result(f"主动发起: {result}")

    # ========================================================================
    # AstrBot 钩子
    # ========================================================================

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_message(self, event) -> None:
        """主动回复消息唤醒：门控 → 标记 → 交由 on_llm_request 决策"""
        if not self._reply_config.enabled:
            return

        if not self._gatekeeper.should_process(event):
            return

        group_id = event.get_group_id()
        if not group_id:
            return

        # 缓存会话标识与自身 ID，供主动发起通路使用
        umo = getattr(event, "unified_msg_origin", "")
        if umo and self._group_umo.get(group_id) != umo:
            self._group_umo[group_id] = umo
            self._umo_dirty = True
        if not self._self_id:
            self._self_id = event.get_self_id() or ""

        message_str = event.message_str or ""
        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name() or sender_id

        # 发起后的首次接话：清除 pending，该消息直接获得一次跟进评估资格
        pending_reply = self._state.consume_initiate_pending(group_id)
        is_followed = bool(sender_id and self._state.match_anchor_user(group_id, sender_id))

        score = self._gatekeeper.quality_score(message_str)
        if score < self._reply_config.quality_threshold and not is_followed and not pending_reply:
            return

        self._sliding_window.append(
            group_id,
            WindowMessage(
                sender_id=sender_id,
                sender_name=sender_name,
                content=message_str,
                timestamp=time.time(),
            ),
        )

        if event.is_at_or_wake_command:
            self._triggering.pop(group_id, None)
            self._state.increment_msg_count(group_id)
            self._passive_active[group_id] = time.time()
            event.set_extra("iris_mode", "passive")
            return

        if self._is_busy(group_id) or self._proactive.is_initiating(group_id):
            logger.debug(f"Iris Reply: reply already in progress for group {group_id}")
            return

        async with self._state.get_lock(group_id):
            motive = self._signals.evaluate_message(group_id, sender_id, message_str)

        if not motive and pending_reply:
            motive = "follow_up"
        if not motive:
            return

        is_follow_up = motive == "follow_up"

        if is_follow_up:
            if group_id in self._follow_pending:
                logger.debug(f"Iris Reply: follow-up aggregation pending for group {group_id}")
                return
            self._follow_pending.add(group_id)
            try:
                await asyncio.sleep(self._reply_config.follow_up_aggregate_window)
            finally:
                self._follow_pending.discard(group_id)

            if self._is_busy(group_id):
                return
            if not pending_reply and not self._state.get_anchor(group_id).active:
                return

        if not self._state.can_detect(group_id, follow_up=is_follow_up):
            logger.debug(f"Iris Reply: trigger rate-limited for group {group_id}")
            return

        provider_id = self._reply_config.provider_id
        if not provider_id:
            provider_id = await self._get_provider_id(event)
            if not provider_id:
                logger.error(f"Iris Reply: failed to get provider ID for group {group_id}")
                return

        async with self._state.get_lock(group_id):
            if group_id in self._triggering:
                logger.debug(f"Iris Reply: trigger already in progress for group {group_id}")
                return
            self._state.record_detect_time(group_id)
            self._triggering[group_id] = time.time()

        event.set_extra("iris_decision", {
            "motive": motive,
            "provider_id": provider_id,
        })

        event.is_at_or_wake_command = True
        event.is_wake = True
        if provider_id:
            event.set_extra("selected_provider", provider_id)
        self._tool_ctx.set_context(group_id)
        logger.info(
            f"Iris Reply: {motive} candidate activated for group {group_id}, deferred to on_llm_request"
        )

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent) -> None:
        """记忆侧：全类型消息入 L1 缓冲、图片入队"""
        if self.component_manager:
            await handle_user_message(event, self.component_manager)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("iris_mem")
    async def iris_mem(self, event: AstrMessageEvent) -> None:
        if self.component_manager:
            result = await execute_command(event)
            if result:
                yield event.plain_result(result)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        # 1. 主动回复统一决策（仅处理 _triggering 中的群；可能 stop_event 终止请求）
        if await self._handle_reply_decision(event):
            return

        # 2. 记忆侧：被动触发检测 + 上下文接管 + L1/L2/L3/画像注入
        if self.component_manager:
            _detect_passive_trigger(event, req, self.context)
            await handle_pre_request_cleanup(
                event, req, self.context, self.component_manager
            )
            await preprocess_llm_request(event, req, self.component_manager)

        # 3. 主动回复发言提示：决策通过后由 _handle_reply_decision 暂存，
        # 在记忆注入之后追加，保证 LLM 先看到上下文、再看到发言指令
        hint = event.get_extra("iris_speak_hint")
        if hint:
            req.extra_user_content_parts.append(TextPart(text=hint).mark_as_temp())

    async def _handle_reply_decision(self, event: AstrMessageEvent) -> bool:
        """主动回复统一决策执行点。

        Returns:
            True 表示已调用 event.stop_event()，调用方应立即返回，
            不再进行记忆注入。
        """
        group_id = event.get_group_id()
        if not group_id or group_id not in self._triggering:
            return False

        info = event.get_extra("iris_decision")
        if not info:
            self._triggering.pop(group_id, None)
            return False

        motive = info.get("motive", "")
        provider_id = info.get("provider_id", "")

        req = DecisionRequest(group_id=group_id, wake="message", motive=motive)
        outcome = await self._decision_core.decide(req, self.context.llm_generate, provider_id)

        if outcome.error or outcome.decision is None:
            logger.error(f"Iris Reply: decision LLM call failed for group {group_id}: {outcome.error}")
            self._stats.record_decision_error(group_id, motive)
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return True

        decision = outcome.decision
        logger.info(
            f"Iris Reply: decision raw for group {group_id} (motive={motive}, "
            f"len={len(outcome.raw_text)}): {outcome.raw_text:.500s}"
        )
        self._stats.record_decision(
            group_id, motive,
            system_prompt=outcome.system_prompt,
            user_prompt=outcome.user_prompt,
            response_text=outcome.raw_text,
            decision=decision,
            duration_ms=outcome.duration_ms,
        )
        logger.info(
            f"Iris Reply: decision parsed for group {group_id}: speak={decision.should_speak}, "
            f"drifted={decision.drifted}, watch={decision.watch}, "
            f"watch_keywords={decision.watch_keywords}, cooldown={decision.cooldown_minutes}"
        )

        async with self._state.get_lock(group_id):
            if decision.observation:
                self._state.set_observation(group_id, decision.observation)

        if decision.parse_failed:
            logger.warning(f"Iris Reply: decision parse failed for group {group_id}")
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return True

        if group_id in self._passive_active:
            logger.info(f"Iris Reply: aborting {motive} for group {group_id}, passive reply in progress")
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return True

        if decision.cooldown_minutes:
            async with self._state.get_lock(group_id):
                actual = self._state.set_cooldown(group_id, decision.cooldown_minutes)
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info(f"Iris Reply: decision requested cooldown {actual} min for group {group_id}")
            self._triggering.pop(group_id, None)
            event.stop_event()
            return True

        if decision.drifted:
            async with self._state.get_lock(group_id):
                self._state.close_anchor(group_id)
                self._state.record_drift(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info(f"Iris Reply: topic drifted for group {group_id}, anchor closed")
            self._triggering.pop(group_id, None)
            event.stop_event()
            return True

        if decision.watch or decision.watch_keywords:
            if decision.should_speak:
                event.set_extra("iris_pending_watch", (
                    decision.watch, decision.watch_keywords, decision.why,
                ))
            else:
                async with self._state.get_lock(group_id):
                    self._state.add_anchor_watch(
                        group_id,
                        users=decision.watch or None,
                        keywords=decision.watch_keywords or None,
                        reason=decision.why,
                    )
            logger.info(
                f"Iris Reply: decision watch for group {group_id}, users={decision.watch}, "
                f"keywords={decision.watch_keywords}, reason={decision.why} (speak={decision.should_speak})"
            )

        if not decision.should_speak:
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.debug(f"Iris Reply: decision skip for group {group_id}")
            self._triggering.pop(group_id, None)
            event.stop_event()
            return True

        self._reply_in_progress[group_id] = time.time()
        self._triggering.pop(group_id, None)

        event.set_extra("iris_mode", motive)
        event.set_extra("iris_decision_obs", decision.observation)

        hint = SPEAK_HINTS.get(motive, SPEAK_HINTS["chime_in"])
        # 发言提示延迟到记忆注入之后追加（见 on_llm_request），
        # 保持「上下文在前、指令在后」的提示顺序
        event.set_extra("iris_speak_hint", hint)
        logger.info(f"Iris Reply: decision speak ({motive}) for group {group_id}")
        return False

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        # 1. 记忆侧：bot 回复入 L1
        if self.component_manager:
            await handle_llm_response(event, resp, self.component_manager)
        # 2. 主动回复侧：按 iris_mode 记账
        await self._reply_on_llm_response(event, resp)

    async def _reply_on_llm_response(self, event: AstrMessageEvent, response: LLMResponse) -> None:
        group_id = event.get_group_id()
        if not group_id:
            return

        event.set_extra("iris_llm_replied", True)
        mode = event.get_extra("iris_mode")

        if mode in ("chime_in", "follow_up"):
            self._reply_in_progress.pop(group_id, None)
            self._passive_active.pop(group_id, None)

            async with self._state.get_lock(group_id):
                self._state.record_actual_reply(group_id)

            self._tool_ctx.clear_context()
            await self._state.save_dirty(self._kv_save)
            logger.info(f"Iris Reply: {mode} reply sent for group {group_id}")
        elif mode == "passive":
            self._passive_active.pop(group_id, None)

            async with self._state.get_lock(group_id):
                self._state.record_actual_reply(group_id, count_consecutive=False)

            self._stats.record_passive_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info(f"Iris Reply: passive reply boost applied for group {group_id}")
        else:
            if not self._state.is_whitelisted(group_id):
                return
            async with self._state.get_lock(group_id):
                self._state.record_actual_reply(group_id, count_consecutive=False)
            await self._state.save_dirty(self._kv_save)
            logger.info(f"Iris Reply: normal LLM reply for group {group_id}, boost applied")

    @filter.after_message_sent()
    async def on_message_sent(self, event) -> None:
        """主动回复侧：bot 消息入滑动窗口 + 写 ThreadAnchor"""
        group_id = event.get_group_id()
        if not group_id:
            return
        if not self._state.is_whitelisted(group_id):
            return
        sender_id = event.get_sender_id()
        if not sender_id:
            return
        result = event.get_result()
        bot_text = result.get_plain_text().strip() if result else ""
        if bot_text:
            self._sliding_window.append(group_id, WindowMessage(
                sender_id=event.get_self_id() or "iris",
                sender_name="Iris",
                content=bot_text,
                timestamp=time.time(),
            ))
        mode = event.get_extra("iris_mode")
        if mode in ("chime_in", "follow_up"):
            pending = event.get_extra("iris_pending_watch")
            users = list(pending[0]) if pending else []
            if sender_id not in users:
                users.append(sender_id)
            keywords = list(pending[1]) if pending else []
            reason = pending[2] if pending else ""
            topic = event.get_extra("iris_decision_obs", "")
            async with self._state.get_lock(group_id):
                self._state.write_anchor(
                    group_id,
                    kind=mode,
                    topic=topic,
                    bot_message=bot_text,
                    users=users,
                    keywords=keywords or None,
                    reason=reason,
                )
            await self._state.save_dirty(self._kv_save)
            logger.debug(f"Iris Reply: anchor written ({mode}) for group {group_id}")
        elif mode == "passive":
            provider_id = self._reply_config.provider_id or await self._get_provider_id(event)
            if provider_id:
                await self._passive_watch_eval(group_id, provider_id, sender_id, bot_text)
            else:
                async with self._state.get_lock(group_id):
                    self._state.write_anchor(
                        group_id, kind="passive", bot_message=bot_text, users=[sender_id],
                    )
                await self._state.save_dirty(self._kv_save)
        elif event.get_extra("iris_llm_replied"):
            async with self._state.get_lock(group_id):
                self._state.write_anchor(
                    group_id, kind="reply", bot_message=bot_text, users=[sender_id],
                )
            await self._state.save_dirty(self._kv_save)
            logger.debug(f"Iris Reply: anchor written (reply) for group {group_id}")

    async def _passive_watch_eval(
        self, group_id: str, provider_id: str, fallback_sender: str, bot_text: str,
    ) -> None:
        """被动回复后的跟进评估（motive=watch）：只决定是否建立关注锚点。"""
        if not self._sliding_window.get_messages(group_id):
            return

        req = DecisionRequest(group_id=group_id, wake="message", motive="watch")
        outcome = await self._decision_core.decide(req, self.context.llm_generate, provider_id)

        if outcome.error or outcome.decision is None:
            logger.warning(f"Iris Reply: passive watch eval failed for group {group_id}: {outcome.error}")
            self._stats.record_decision_error(group_id, "watch")
            async with self._state.get_lock(group_id):
                self._state.write_anchor(
                    group_id, kind="passive", bot_message=bot_text, users=[fallback_sender],
                )
            await self._state.save_dirty(self._kv_save)
            return

        decision = outcome.decision
        self._stats.record_decision(
            group_id, "watch",
            system_prompt=outcome.system_prompt,
            user_prompt=outcome.user_prompt,
            response_text=outcome.raw_text,
            decision=decision,
            duration_ms=outcome.duration_ms,
        )
        logger.info(
            f"Iris Reply: passive watch eval for group {group_id}: watch={decision.watch}, "
            f"keywords={decision.watch_keywords}, drifted={decision.drifted}"
        )

        async with self._state.get_lock(group_id):
            if decision.observation:
                self._state.set_observation(group_id, decision.observation)
            if decision.parse_failed:
                self._state.write_anchor(
                    group_id, kind="passive", bot_message=bot_text, users=[fallback_sender],
                )
            elif decision.drifted:
                self._state.close_anchor(group_id)
                self._state.record_drift(group_id)
                logger.info(f"Iris Reply: topic drifted (passive) for group {group_id}, anchor closed")
            elif decision.watch or decision.watch_keywords:
                self._state.write_anchor(
                    group_id,
                    kind="passive",
                    bot_message=bot_text,
                    users=decision.watch or None,
                    keywords=decision.watch_keywords or None,
                    reason=decision.why,
                )
            else:
                self._state.write_anchor(
                    group_id, kind="passive", bot_message=bot_text, users=[fallback_sender],
                )
        await self._state.save_dirty(self._kv_save)

    # AstrBot >= 4.23 才将 on_agent_done 暴露为插件钩子（旧版对话清理路径，默认不走）；
    # 低版本 AstrBot 下不注册该钩子，保证插件可正常加载
    if hasattr(filter, "on_agent_done"):
        @filter.on_agent_done()
        async def on_agent_done(self, event: AstrMessageEvent, run_context, resp) -> None:
            if self.component_manager:
                await handle_agent_done(event, resp, self.context, self.component_manager)

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        """消息发送前处理：错误友好化 + Markdown 去除"""
        result = event.get_result()
        if not result:
            return

        if self._error_processor and self._error_processor.should_process(event):
            self._error_processor.process_result(result)

        if self._markdown_stripper and self._markdown_stripper.should_process(event):
            self._markdown_stripper.process_result(result)

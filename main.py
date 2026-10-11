"""AstrBot hooks; Iris transport and behavior live in iris_memory modules."""

import asyncio
import copy

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import error_response, json_response, request

from .iris_memory import PLUGIN_NAME, VERSION
from .iris_memory.bridge import Bridge
from .iris_memory.commands import run_command
from .iris_memory.config import Config
from .iris_memory.extras import decorate_result, prepare_pure_mention
from .iris_memory.proactive import create_proactive
from .iris_memory.service import TOOL_NAMES, MemoryService
from .iris_memory.status import summarize
from .iris_memory.vendor.iris_client import AsyncIrisClient, RetryPolicy

WARNINGS = {
    "container_host_enabled": (
        "使用 HTTP 时，请求在 Docker 网络中以明文传输令牌，只在可信网络使用；"
        "HTTPS 需要服务端配置有效 TLS。"
    ),
    "astrbot_context_conflict": "请将 provider_ltm_settings.group_message_max_cnt 设置为 0，避免重复上下文。",
    "pure_at_waiting_conflict": "纯 @ 回复需要关闭 platform_settings.empty_mention_waiting。",
    "host_settings_unavailable": "未能检查 AstrBot 上下文配置，请确认 group_message_max_cnt=0。",
}


class IrisMemoryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.bridge = None
        self.service = None
        self.settings = None
        self.proactive = None
        self.warnings = set()
        self._page_route = f"/{PLUGIN_NAME}/status"
        self._proactive_route = f"/{PLUGIN_NAME}/proactive"
        try:
            self.settings = Config.from_mapping(config)
            if self.settings.enabled and self.settings.token:
                client = AsyncIrisClient(
                    self.settings.base_url,
                    token=self.settings.token,
                    timeout=max(
                        self.settings.request_timeout,
                        self.settings.prepare_timeout,
                        self.settings.tool_timeout,
                    ),
                    retry=RetryPolicy(max_retries=1),
                )
                self.bridge = Bridge(
                    self.settings,
                    client,
                    StarTools.get_data_dir(PLUGIN_NAME),
                    logger.warning,
                    self._send_active,
                )
                self.service = MemoryService(self.bridge)
            else:
                logger.info("Iris: disabled_or_missing_token")
        except Exception:
            logger.warning("Iris: invalid_configuration")
        context.register_web_api(self._page_route, self.page_status, ["GET"], "Iris 连接与状态")
        context.register_web_api(
            self._proactive_route, self.page_proactive, ["POST"], "Iris 主动回复设置"
        )
        self.check_host_config()
        # Proactive owns only proactive policy / scheduling; it reuses this bridge.
        try:
            self.proactive = create_proactive(self)
        except Exception:
            logger.warning("Iris: proactive_initialization_failed")

    def check_host_config(self, event=None) -> None:
        try:
            cfg = (
                self.context.get_config(umo=event.unified_msg_origin)
                if event
                else self.context.get_config()
            )
            codes = set()
            if cfg.get("provider_ltm_settings", {}).get("group_message_max_cnt") != 0:
                codes.add("astrbot_context_conflict")
            if (
                self.settings
                and self.settings.pure_at_reply
                and cfg.get("platform_settings", {}).get("empty_mention_waiting", True)
            ):
                codes.add("pure_at_waiting_conflict")
        except Exception:
            codes = {"host_settings_unavailable"}
        if self.settings and self.settings.allow_container_host:
            codes.add("container_host_enabled")
        for code in codes - self.warnings:
            logger.warning("Iris: " + code + " — " + WARNINGS[code])
        # Include warnings found in routed per-session AstrBot profiles.
        self.warnings.update(codes)

    def warning_text(self) -> str:
        return "".join("\n提示：" + WARNINGS[code] for code in sorted(self.warnings))

    async def _send_active(self, origin: str, text: str) -> bool:
        # The bridge owns reminder consumption/backflow; policy owns permission to send.
        return bool(self.proactive and await self.proactive.send_reminder(origin, text))

    async def _send_to_session(self, origin: str, text: str) -> bool:
        return await self.context.send_message(origin, MessageChain().message(text))

    async def initialize(self) -> None:
        if self.bridge:
            self.bridge.start()
        if self.proactive:
            await self.proactive.start()

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10000)
    async def receive(self, event: AstrMessageEvent) -> None:
        if self.bridge:
            self.check_host_config(event)
            await self.bridge.receive(event)
            try:
                await prepare_pure_mention(self.context, event, self.settings)
            except Exception:
                self.bridge.record("pure_at_degraded")
            if self.proactive:
                await self.proactive.on_message(event)

    @filter.on_llm_request(priority=10000)
    async def before_llm(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        tools = getattr(req, "func_tool", None)
        if tools is not None:
            # Avoid mutating a ToolSet shared by other requests or plugins.
            req.func_tool = copy.copy(tools)
            for name in (*TOOL_NAMES, "set_cooldown"):
                if (
                    not self.service
                    or not self.settings.tools_enabled
                    or not getattr(self.settings, name + "_enabled", False)
                ):
                    req.func_tool.remove_tool(name)
        if self.bridge:
            await self.bridge.prepare(event, req)
            if self.proactive:
                await self.proactive.before_llm(event, req)

    @filter.on_decorating_result()
    async def decorate(self, event: AstrMessageEvent) -> None:
        if not self.settings or not self.settings.enabled:
            return
        try:
            decorate_result(
                event, self.settings, self.context.get_config(umo=event.unified_msg_origin)
            )
        except Exception:
            logger.warning("Iris: decorate_failed")

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        if self.bridge:
            await self.bridge.after_sent(event)
            if self.proactive:
                # Still NOT a delivery receipt. PL-2 must consult confirmed sends.
                await self.proactive.after_sent(event)

    @filter.command("iris_mem")
    async def iris_mem(self, event: AstrMessageEvent):
        yield event.plain_result(await run_command(self, event))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("iris_reply")
    async def iris_reply(self, event: AstrMessageEvent):
        if not event.is_admin():
            yield event.plain_result("仅 AstrBot 管理员可以管理主动回复。")
            return
        if not self.proactive:
            yield event.plain_result("Iris 未连接，请检查插件配置及本机服务。")
            return
        yield event.plain_result(await self.proactive.command(event))

    @filter.llm_tool(name="set_cooldown")
    async def set_cooldown(self, event: AstrMessageEvent, minutes: int = 5) -> str:
        """管理员要求暂时停止当前会话的主动回复时设置冷却，不会开启主动回复。

        Args:
            minutes(number): 冷却整数分钟，1—120，默认 5。
        """
        return await self._tool("set_cooldown", event, minutes=minutes)

    async def _tool(self, name, event, **kwargs) -> str:
        if name == "set_cooldown":
            if (
                not self.settings
                or not self.settings.tools_enabled
                or not self.settings.set_cooldown_enabled
            ):
                return "这个 Iris 工具已关闭。"
            if not self.proactive:
                return "Iris 未连接，请检查插件配置及本机服务。"
            return await self.proactive.set_cooldown(event, **kwargs)
        if not self.service:
            return "Iris 未连接，请检查插件配置及本机服务。"
        result = await self.service.tool(name, event, **kwargs)
        if (
            self.proactive
            and name in {"add_follow_up", "end_follow_up"}
            and self.settings.tools_enabled
            and getattr(self.settings, name + "_enabled", False)
        ):
            self.proactive.invalidate_goals(event)
        return result

    @filter.llm_tool(name="search_memory")
    async def search_memory(
        self, event: AstrMessageEvent, query: str, person: str = "", include_forgotten: bool = False
    ) -> str:
        """查询当前入口可见的记忆，保留立场与相信程度；检索不等于事实确认。

        Args:
            query(string): 查询内容。
            person(string): 可选人物名字或 Iris 主体 ID，空字符串表示不限定人物。
            include_forgotten(boolean): 是否深度查询遗忘中的记忆，不会恢复记忆。
        """
        return await self._tool(
            "search_memory", event, query=query, person=person, include_forgotten=include_forgotten
        )

    @filter.llm_tool(name="save_memory")
    async def save_memory(self, event: AstrMessageEvent, content: str = "") -> str:
        """请求立即学习本入口已收到的真实对话。不是直接写入，不能宣称已经记住。

        Args:
            content(string): 用户希望学习的事项说明；不会作为伪造原话写入。
        """
        return await self._tool("save_memory", event, content=content)

    @filter.llm_tool(name="correct_memory")
    async def correct_memory(self, event: AstrMessageEvent, correction: str = "") -> str:
        """请求 Iris 学习对话中用户的纠正。不是按 ID 编辑记忆，不能宣称已修改。

        Args:
            correction(string): 纠正事项说明；实际依据是本入口已收到的对话。
        """
        return await self._tool("correct_memory", event, correction=correction)

    @filter.llm_tool(name="get_profile")
    async def get_profile(self, event: AstrMessageEvent, person: str) -> str:
        """按人物定向查询当前入口可见的记忆，作为画像材料。

        Args:
            person(string): 人物名字或 Iris 主体 ID；有同名时请先澄清身份。
        """
        return await self._tool("get_profile", event, person=person)

    @filter.llm_tool(name="search_knowledge_graph")
    async def search_knowledge_graph(
        self, event: AstrMessageEvent, person: str, query: str = ""
    ) -> str:
        """兼容旧工具名称，执行人物定向记忆查询；不提供独立知识图谱。

        Args:
            person(string): 人物名字或 Iris 主体 ID。
            query(string): 可选的相关问题。
        """
        return await self._tool("search_knowledge_graph", event, person=person, query=query)

    @filter.llm_tool(name="add_follow_up")
    async def add_follow_up(
        self, event: AstrMessageEvent, content: str, deadline: str = "", reminder_minutes: int = 0
    ) -> str:
        """创建本入口 Iris 跟进目标。仅在对话明确要求或承诺跟进时使用。

        Args:
            content(string): 明确的跟进事项。
            deadline(string): 截止日期或带时区的 ISO 时间，空字符串表示无截止。
            reminder_minutes(number): 提前提醒的整数分钟，0 表示到期提醒。
        """
        return await self._tool(
            "add_follow_up",
            event,
            content=content,
            deadline=deadline,
            reminder_minutes=reminder_minutes,
        )

    @filter.llm_tool(name="end_follow_up")
    async def end_follow_up(
        self, event: AstrMessageEvent, goal_id: int, state: str = "completed"
    ) -> str:
        """明确完成或放弃本入口目标。不能因为已发送提醒就调用此工具。

        Args:
            goal_id(number): 本入口的目标整数 ID。
            state(string): completed（完成）或 abandoned（放弃）。
        """
        return await self._tool("end_follow_up", event, goal_id=goal_id, state=state)

    async def page_status(self):
        if not request.username:
            return error_response("请先登录 AstrBot。", status_code=401)
        settings = self.settings
        data = {
            "version": VERSION,
            "connection": "disabled",
            "health": None,
            "token_configured": bool(settings and settings.token),
            "allow_container_host": bool(settings and settings.allow_container_host),
            "management_url": settings.base_url.rstrip("/")
            if settings
            else "http://127.0.0.1:8080",
            "warnings": [WARNINGS[code] for code in sorted(self.warnings)],
            "entry_rules": {
                "prefix": settings.entry_prefix,
                "template": settings.entry_template,
                "overrides": settings.entry_overrides,
            }
            if settings
            else {},
            "entries": [],
            "proactive": self.proactive.snapshot() if self.proactive else None,
        }
        if self.bridge:
            data["entries"] = [
                {"origin": e.origin, "entry_id": e.id, "kind": e.kind}
                for e in self.bridge.entries.values()
            ]
            data["counts"] = dict(self.bridge.counts)
            try:
                async with asyncio.timeout(settings.tool_timeout):
                    data["health"] = summarize(await self.bridge.client.status())
                data["connection"] = "connected"
            except Exception:
                data["connection"] = "unavailable"
        return json_response(data, headers={"Cache-Control": "no-store"})

    async def page_proactive(self):
        if not request.username:
            return error_response("请先登录 AstrBot。", status_code=401)
        if not self.proactive:
            return error_response("Iris 未连接。", status_code=503)
        body = await request.json()
        if (
            not isinstance(body, dict)
            or set(body) != {"origin", "changes"}
            or not isinstance(body["origin"], str)
        ):
            return error_response("请提供会话和设置。", status_code=400)
        try:
            data = self.proactive.update(body["origin"], body["changes"])
        except (ValueError, TypeError):
            return error_response("会话不存在或设置无效。", status_code=400)
        return json_response(data, headers={"Cache-Control": "no-store"})

    async def terminate(self) -> None:
        try:
            if self.proactive:
                await self.proactive.close()
        finally:
            if self.bridge:
                await self.bridge.close()
            # Remove only our own handler; don't discard a replacement on reload.
            registered = self.context.registered_web_apis
            registered[:] = [
                api
                for api in registered
                if not (
                    (api[0] == self._page_route and api[1] == self.page_status)
                    or (api[0] == self._proactive_route and api[1] == self.page_proactive)
                )
            ]

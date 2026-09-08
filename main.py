"""AstrBot entry point. Core and SDK remain lazy optional dependencies."""

import copy
import uuid

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from .iris_memory.api import PagesAPI
from .iris_memory.control import Control
from .iris_memory.errors import IrisError
from .iris_memory.host import AstrBotHost
from .iris_memory.identity import Identity

TOOLS = {
    "iris_recall": "memory",
    "iris_remember": "memory",
    "iris_correct": "memory",
    "iris_forget": "memory",
    "iris_task": "proactive",
}


@register(
    "astrbot_plugin_iris_memory", "AirObject", "轻量记忆、人格和主动陪伴", "4.0.0-dev.1"
)
class IrisMemoryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.control = Control(
            StarTools.get_data_dir("astrbot_plugin_iris_memory"),
            AstrBotHost(context, logger),
            config.get("core_mode", "local"),
        )
        self.api = PagesAPI(self.control)

    async def initialize(self):
        await self.control.start()
        # AstrBot owns routing/authentication and replaces this exact route on reload.
        self.context.register_web_api(
            "/astrbot_plugin_iris_memory/v4/query",
            self.web_query,
            ["POST"],
            "Iris v4 Pages",
        )
        self.context.register_web_api(
            "/astrbot_plugin_iris_memory/v4/status",
            self.web_status,
            ["GET"],
            "Iris v4 status",
        )

    async def web_status(self):
        return await self._web("status", {})

    async def web_query(self):
        from astrbot.api.web import error_response, request

        try:
            data = await request.json(default={})
            if not isinstance(data, dict):
                raise ValueError
            return await self._web(data.get("action", ""), data.get("data", {}))
        except (ValueError, TypeError):
            return error_response("请求格式无效", status_code=400)

    async def _web(self, action, data):
        from astrbot.api.web import error_response, json_response, request

        try:
            result = await self.api.dispatch(action, data, username=request.username)
            return json_response({"status": "ok", "data": result})
        except IrisError as exc:
            status = (
                401
                if exc.code == "unauthenticated"
                else 409
                if "conflict" in exc.code
                else 400
            )
            return error_response(str(exc), status_code=status, data=exc.as_dict())
        except (KeyError, TypeError, ValueError):
            return error_response("字段缺失或格式无效", status_code=400)
        except Exception as exc:
            self.control.logs.emit(
                "pages.failed", level="ERROR", error=exc, action=action
            )
            return error_response("操作失败，请查看插件日志", status_code=500)

    async def terminate(self):
        await self.control.close()

    def identity(self, event):
        cached = event.get_extra("iris_v4_identity")
        if cached is None:
            cached = Identity.from_event(event)
            if not cached.message_id:
                from dataclasses import replace

                cached = replace(cached, message_id=uuid.uuid4().hex)
            event.set_extra("iris_v4_identity", cached)
        return cached

    @filter.command("iris")
    async def iris_command(self, event: AstrMessageEvent, action: str = "status"):
        identity = self.identity(event)
        if action != "status":
            yield event.plain_result(
                "配置、人格和任务请使用插件 Pages；/iris status 查看当前会话键。"
            )
            return
        await self.control.remember_conversation(identity)
        running = "、".join(self.control.modules) or "仅控制页面"
        yield event.plain_result(
            f"Iris v4 · {self.control.mode}\n会话键：{identity.key}\n会话绑定键：{identity.session_key}\n模块：{running}\n采集：{'开启' if self.control.allowed(identity) else '关闭'}"
        )

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def observe_message(self, event: AstrMessageEvent):
        c, identity = self.control, self.identity(event)
        if not c.allowed(identity):
            return
        try:
            await c.remember_conversation(identity)
            if "memory" in c.modules:
                await c.run("memory", "capture", identity, event.message_str)
            if "media" in c.modules:
                from astrbot.api.message_components import Image

                images = [
                    await item.convert_to_file_path()
                    for item in event.get_messages()
                    if isinstance(item, Image)
                ]
                if images:
                    await c.run("media", "describe", identity, images)
        except Exception as exc:
            c.logs.emit(
                "hook.observe_failed",
                level="ERROR",
                error=exc,
                conversation=identity.key,
            )

    @filter.on_waiting_llm_request()
    async def prepare_persona(self, event: AstrMessageEvent):
        c = self.control
        if "persona" not in c.modules:
            return
        try:
            await c.run("persona", "prepare", self.identity(event))
        except Exception as exc:
            event.set_extra("iris_v4_persona_failed", True)
            c.logs.emit("hook.persona_failed", level="ERROR", error=exc)

    @filter.on_llm_request()
    async def inject_context(self, event: AstrMessageEvent, req):
        c, identity = self.control, self.identity(event)
        if req.func_tool:
            # Never mutate the host/shared tool registry.
            req.func_tool = copy.copy(req.func_tool)
            req.func_tool.tools = list(req.func_tool.tools)
            for name, module in TOOLS.items():
                if module not in c.modules or not c.allowed(identity):
                    req.func_tool.remove_tool(name)
        old_part = event.get_extra("iris_v4_part")
        req.extra_user_content_parts = [
            p for p in req.extra_user_content_parts if p is not old_part
        ]
        event.set_extra("iris_v4_part", None)
        event.set_extra("iris_v4_plan", None)
        if (
            "context" not in c.modules
            or not c.allowed(identity)
            or event.get_extra("iris_v4_persona_failed")
        ):
            return
        try:
            if "persona" in c.modules:
                await c.host.verify_persona(c, identity, req)
            tool_schema = req.func_tool.openai_schema() if req.func_tool else []
            plan = await c.run(
                "context",
                "build",
                identity,
                req.prompt or event.message_str,
                system=req.system_prompt,
                history=req.contexts,
                tools=tool_schema,
                extra=[str(item) for item in req.extra_user_content_parts],
            )
            if "persona" in c.modules:
                await c.host.verify_persona(c, identity, req)
            if plan["text"]:
                from astrbot.core.agent.message import TextPart

                part = TextPart(text=plan["text"]).mark_as_temp()
                # Remove only this hook's previous object on a host retry.
                old = event.get_extra("iris_v4_part")
                req.extra_user_content_parts = [
                    p for p in req.extra_user_content_parts if p is not old
                ] + [part]
                event.set_extra("iris_v4_part", part)
            event.set_extra("iris_v4_plan", plan)
        except Exception as exc:
            c.logs.emit("hook.context_failed", level="ERROR", error=exc)

    @filter.on_llm_response()
    async def report_context(self, event: AstrMessageEvent, resp):
        plan = event.get_extra("iris_v4_plan")
        if plan and "context" in self.control.modules:
            event.set_extra("iris_v4_plan", None)
            try:
                await self.control.run("context", "report", plan, visible=True)
            except Exception as exc:
                self.control.logs.emit("hook.usage_failed", level="ERROR", error=exc)

    async def tool(self, event, method, *args):
        from .iris_memory.storage import encode

        try:
            return encode(
                await self.control.run("memory", method, self.identity(event), *args)
            )
        except IrisError as exc:
            return encode({"error": exc.as_dict()})

    @filter.llm_tool(name="iris_recall")
    async def recall_tool(self, event: AstrMessageEvent, query: str):
        """查询当前会话与人格范围内的记忆。

        Args:
            query(string): 要查找的事实或主题。
        """
        return await self.tool(event, "recall", query)

    @filter.llm_tool(name="iris_remember")
    async def remember_tool(self, event: AstrMessageEvent, text: str):
        """在用户明确要求记住时保存一条事实，不保存推测。

        Args:
            text(string): 用户明确提供的事实。
        """
        return await self.tool(event, "remember", text)

    @filter.llm_tool(name="iris_correct")
    async def correct_tool(
        self, event: AstrMessageEvent, claim_id: str, text: str, revision: int
    ):
        """按照用户明确纠正修订当前会话记忆。

        Args:
            claim_id(string): 召回返回的 claim 资源 ID。
            text(string): 用户纠正后的事实。
            revision(number): 召回返回的当前修订号。
        """
        return await self.tool(event, "correct", claim_id, text, revision)

    @filter.llm_tool(name="iris_forget")
    async def forget_tool(self, event: AstrMessageEvent, claim_id: str):
        """仅在用户明确要求遗忘时删除指定记忆。

        Args:
            claim_id(string): 当前会话 claim 资源 ID。
        """
        return await self.tool(event, "forget", claim_id)

    @filter.llm_tool(name="iris_task")
    async def task_tool(self, event: AstrMessageEvent, title: str, due_at_us: int):
        """按用户明确要求创建定时提醒，主动模块开启时到点发送。

        Args:
            title(string): 提醒内容。
            due_at_us(number): 提醒时间的 UTC Unix 微秒时间戳。
        """
        self.control.require("proactive")
        return await self.tool(event, "create_task", title, due_at_us)

"""Entry-scoped operations shared by commands, tools and the PL-2 extension."""

import asyncio
import json
from typing import Any

from .bridge import Bridge
from .mapping import input_message, map_entry, stable_key
from .prompt import memory_material
from .vendor.iris_client import IrisError

TOOL_NAMES = (
    "search_memory",
    "save_memory",
    "correct_memory",
    "get_profile",
    "search_knowledge_graph",
    "add_follow_up",
    "end_follow_up",
)


def error_text(error: Exception) -> str:
    if isinstance(error, IrisError):
        return {
            401: "Iris 令牌无效或已撤销，请在插件配置中更新。",
            403: "Iris 未授权这个入口或对象，请检查令牌范围。",
            404: "Iris 对象不存在或已删除，请重新查询。",
            409: "目标已变化或合并，请重新查询目标后明确选择，未自动重试修改。",
            413: "内容超过 Iris 接口限制，请缩短后重试。",
            429: "Iris 请求较多，请稍后重试。",
            503: "Iris 暂时不可用，请稍后重试。",
        }.get(error.status_code, "Iris 请求未完成，请检查配置或稍后重试。")
    return "Iris 请求未完成，请检查服务是否启动，或稍后重试。"


def memories_text(data: dict) -> str:
    # Preserve uncertainty and provenance; never present a search as a new fact.
    rows = [memory_material(memory) for memory in data.get("memories", [])]
    if not rows:
        return "当前入口可见范围内未找到相关记忆。"
    return "以下是检索材料，需结合立场与相信程度判断：\n" + json.dumps(rows, ensure_ascii=False)


class MemoryService:
    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self.config = bridge.config
        self.client = bridge.client

    def entry(self, event: Any):
        # Never accept an LLM-supplied entry_id or trust an old mapping after reload.
        return map_entry(event, self.config)

    async def ensure_received(self, event: Any) -> None:
        await self.bridge.receive(event)
        receipt = event.get_extra("iris_received")
        if receipt is None or not await asyncio.shield(receipt):
            raise RuntimeError("input_not_accepted")

    async def search(self, event: Any, text: str = "", person: str = "", forgotten: bool = False):
        options = {"text": text, "limit": 8, "include_forgotten": forgotten}
        if person.strip():
            options["people"] = [person.strip()]
        return await self.client.search(options, entry_id=self.entry(event).id)

    async def learn(self, event: Any) -> str:
        await self.ensure_received(event)
        result = await self.client.learn(self.entry(event).id)
        if not result.get("accepted"):
            return "Iris 未接受本次学习请求，请检查状态。"
        suffix = (
            "模型用途暂停，恢复后处理。" if result.get("paused") else "请通过 status 查看后续结果。"
        )
        return "已请求立即学习本入口对话中的陈述与纠正；接受请求不代表已形成或修改记忆。" + suffix

    async def goals(self, event: Any, *, offset: int = 0) -> dict:
        entry_id = self.entry(event).id
        page = await self.client.list_goals(
            {"entry_id": entry_id, "state": "open", "limit": 100, "offset": offset}
        )
        # Some cores also return applicable global goals. They are not this tool's writable objects.
        return {
            **page,
            "items": [g for g in page.get("items", []) if g.get("entry_id") == entry_id],
        }

    async def add_goal(
        self, event: Any, content: str, deadline: str = "", reminder_minutes: int = 0
    ):
        if not isinstance(content, str) or not content.strip():
            return "请提供跟进内容。"
        if type(reminder_minutes) is not int or not 0 <= reminder_minutes <= 525600:
            return "提醒提前量应为 0—525600 的整数分钟。"
        await self.ensure_received(event)
        entry = self.entry(event)
        message = input_message(event, entry)
        goal = {
            "content": content.strip(),
            "entry_id": entry.id,
            "host_key": "astrbot:"
            + stable_key(
                entry.id, message["dedupe_key"], content.strip(), deadline, str(reminder_minutes)
            ),
            "reminder_minutes": reminder_minutes,
        }
        if deadline:
            goal["deadline"] = deadline
        result = await self.client.create_goal(goal)
        kept = result["goal"]
        return (
            f"已提交本入口跟进目标，ID {kept['id']}；是否合并由 Iris 去重处理。提醒不等于目标完成。"
        )

    async def end_goal(self, event: Any, goal_id: int, state: str = "completed") -> str:
        if type(goal_id) is not int or goal_id <= 0 or state not in {"completed", "abandoned"}:
            return "请提供有效目标 ID，结束状态只能是 completed 或 abandoned。"
        offset = 0
        while True:
            page = await self.goals(event, offset=offset)
            target = next((g for g in page["items"] if g["id"] == goal_id), None)
            if target:
                # PATCH has no entry_id field in host v1. Verify ownership before
                # sending its revision; conflicts / canonical_id are never followed.
                await self.client.update_goal(
                    goal_id, {"state": state, "expected_revision": target["revision"]}
                )
                return "目标已标记为完成。" if state == "completed" else "目标已标记为放弃。"
            offset += 100
            if offset >= page.get("total", 0):
                return "未找到本入口的开放目标；全局或其他入口的目标请在 Iris 管理页处理。"

    async def tool(self, name: str, event: Any, **args) -> str:
        if (
            name not in TOOL_NAMES
            or not self.config.tools_enabled
            or not getattr(self.config, name + "_enabled", False)
        ):
            return "这个 Iris 工具已关闭。"
        try:
            async with asyncio.timeout(self.config.tool_timeout):
                if name == "search_memory":
                    return memories_text(
                        await self.search(
                            event,
                            args.get("query", ""),
                            args.get("person", ""),
                            args.get("include_forgotten", False),
                        )
                    )
                if name in {"get_profile", "search_knowledge_graph"}:
                    person = args.get("person", "").strip()
                    if not person:
                        return "请提供人物名字或 Iris 主体 ID；不提供独立知识图谱。"
                    return memories_text(await self.search(event, args.get("query", ""), person))
                if name in {"save_memory", "correct_memory"}:
                    # Arguments describe intent only. Learn actual received dialogue,
                    # never fabricate a user message from generated tool arguments.
                    return await self.learn(event)
                if name == "add_follow_up":
                    return await self.add_goal(event, **args)
                return await self.end_goal(event, **args)
        except Exception as error:
            self.bridge.record("tool_request_failed")
            return error_text(error)

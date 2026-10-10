import asyncio
import re

from .service import error_text, memories_text
from .status import status_text, summarize

HELP = """/iris_mem status — Iris 连接、健康、积压和用量
/iris_mem search 关键词 — 查询本入口可见记忆
/iris_mem persona — 当前 Iris 人格版本与说明
/iris_mem goals — 本入口开放目标
/iris_mem learn — 请求立即学习本入口对话
/iris_mem help — 帮助
删除、编辑、导出导入请使用 Iris 管理页。"""
MANAGEMENT = {
    "delete",
    "remove",
    "clear",
    "edit",
    "export",
    "import",
    "backup",
    "restore",
    "删除",
    "编辑",
    "导出",
    "导入",
    "l1",
    "l2",
    "l3",
    "all",
}


async def run_command(plugin, event) -> str:
    text = event.get_message_str().strip()
    tail = re.sub(r"^/?iris_mem\b", "", text, count=1).strip()
    action, _, argument = tail.partition(" ")
    action = action.lower() or "help"
    link = plugin.settings.base_url.rstrip("/") if plugin.settings else "http://127.0.0.1:8080"
    if action in MANAGEMENT:
        return f"此操作请在 Iris 管理页完成：{link}"
    if action == "help":
        return HELP + "\nIris 管理页：" + link
    if not plugin.service:
        return "Iris 未连接：请先启动服务，在插件配置中填写地址和令牌后重载。"
    try:
        async with asyncio.timeout(plugin.settings.tool_timeout):
            service = plugin.service
            if action == "status":
                message = status_text(summarize(await service.client.status()))
                return message + "\n入口：" + service.entry(event).id + plugin.warning_text()
            if action == "search":
                if not argument.strip():
                    return "用法：/iris_mem search 关键词"
                return memories_text(await service.search(event, argument.strip()))
            if action == "persona":
                await service.ensure_received(event)
                data = await service.client.prepare(
                    service.entry(event).id,
                    {
                        "judge": False,
                        "recent_limit": 0,
                        "memory_limit": 0,
                        "goal_limit": 0,
                    },
                )
                persona = data.get("persona") or {}
                content = str(persona.get("content") or "未设置，继续沿用 AstrBot 人设。")
                return (
                    f"Iris 人格版本：{persona.get('version') or '未发布'}\n{content}\n"
                    f"更新说明：由 Iris 学习与整理生成，在 Iris 管理页确认更新：{link}"
                )
            if action == "goals":
                page = await service.goals(event)
                rows = [
                    f"#{g['id']} {g['content']}（截止：{g.get('deadline') or '无'}）"
                    for g in page["items"]
                ]
                return (
                    ("本入口开放目标（最多 100 条）：\n" + "\n".join(rows))
                    if rows
                    else "本入口暂无开放目标。"
                )
            if action == "learn":
                return await service.learn(event)
            return HELP + "\nIris 管理页：" + link
    except Exception as error:
        plugin.bridge.record("command_request_failed")
        return error_text(error)

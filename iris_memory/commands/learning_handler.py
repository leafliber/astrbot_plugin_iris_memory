"""
Iris Chat Memory - 学习模块指令处理器

处理学习模块的管理指令：
- stats（默认）：三表计数与 pending/approved 分布；
- show <jargon|pattern|shot>：列出当前群条目（每类最多 20 条）；
- clear：按范围清空学习数据；
- jargon disable <词> / jargon enable <词>：手动禁用/恢复词条注入。
"""

from typing import Optional, TYPE_CHECKING

from iris_memory.core import get_logger, get_component_manager
from iris_memory.learning import LearningComponent
from iris_memory.learning.storage import (
    STATUS_ACTIVE,
    STATUS_DORMANT,
    STATUS_APPROVED,
    STATUS_DISABLED,
    STATUS_PENDING,
)
from iris_memory.platform import get_adapter
from .base import CommandHandler, CommandResult, ParsedArgs, DeleteScope

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("commands.learning")

# show 子指令每类最多列出的条数
_SHOW_LIMIT = 20

# 状态中文显示
_STATUS_LABELS = {
    STATUS_PENDING: "待审查",
    STATUS_APPROVED: "已通过",
    STATUS_DISABLED: "已禁用",
    STATUS_ACTIVE: "生效中",
    STATUS_DORMANT: "休眠",
}

# show 类别词到表名
_SHOW_TABLES = {
    "jargon": "jargon",
    "pattern": "expression_pattern",
    "shot": "few_shot",
}

_UNAVAILABLE_MSG = "学习模块不可用或未启用（learning.enable）"


def _truncate(text: str, limit: int = 50) -> str:
    """列表展示用文本截断"""
    text = (text or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


class LearningCommandHandler(CommandHandler):
    """学习模块指令处理器

    支持的子指令：
    - stats: 查看统计信息（默认）
    - show <jargon|pattern|shot>: 列出当前群条目
    - clear: 清空学习数据
    - jargon disable <词> / jargon enable <词>: 禁用/恢复词条
    """

    @property
    def name(self) -> str:
        return "learning"

    @property
    def description(self) -> str:
        return "学习模块管理（表达风格/暗语/对话样例）"

    @property
    def sub_commands(self) -> dict[str, str]:
        return {
            "stats": "查看统计信息（默认）",
            "show <jargon|pattern|shot>": "列出当前群条目",
            "remember <上下文> => <表达示例>": "手动记入风格学习（直接生效）",
            "clear": "清空学习数据",
            "jargon disable <词>": "禁用词条注入",
            "jargon enable <词>": "恢复词条注入",
        }

    async def handle(
        self,
        event: "AstrMessageEvent",
        args: ParsedArgs,
        sub_command: Optional[str] = None,
    ) -> CommandResult:
        """处理学习模块指令"""
        if sub_command == "stats" or sub_command is None:
            return await self._handle_stats(event, args)
        elif sub_command == "show":
            return await self._handle_show(event, args)
        elif sub_command == "remember":
            return await self._handle_remember(event, args)
        elif sub_command == "clear":
            return await self._handle_clear(event, args)
        elif sub_command == "jargon":
            return await self._handle_jargon(event, args)
        elif sub_command == "help":
            return CommandResult(success=True, message=self.get_help_text())
        else:
            return CommandResult(
                success=False,
                message=f"未知的子指令: {sub_command}\n{self.get_help_text()}",
            )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _get_component(self) -> Optional[LearningComponent]:
        """取学习组件（不可用返回 None）"""
        try:
            manager = get_component_manager()
        except RuntimeError:
            return None
        if not manager:
            return None
        component = manager.get_component("learning", LearningComponent)
        if not component or not component.is_available or not component.storage:
            return None
        return component

    async def _handle_stats(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理统计查询"""
        component = self._get_component()
        if not component:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)

        stats = component.storage.get_stats()

        def _fmt(table_key: str, label: str) -> str:
            table_stats = stats.get(table_key, {})
            dist = table_stats.get("by_status", {})
            dist_text = "，".join(
                f"{_STATUS_LABELS.get(s, s)} {c}" for s, c in dist.items()
            )
            return f"{label}: {table_stats.get('total', 0)} 条（{dist_text or '无'}）"

        message = (
            "📊 学习模块统计\n"
            + _fmt("expression_pattern", "表达模式")
            + "\n"
            + _fmt("few_shot", "对话样例")
            + "\n"
            + _fmt("jargon", "暗语词条")
        )
        return CommandResult(success=True, message=message, details=stats)

    async def _handle_show(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理条目列表查询"""
        component = self._get_component()
        if not component:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)

        # raw_args 含子指令本身（["show", category]），类别词取下标 1
        category = args.raw_args[1] if len(args.raw_args) > 1 else ""
        table = _SHOW_TABLES.get(category)
        if not table:
            return CommandResult(
                success=False,
                message="用法: iris_mem learning show <jargon|pattern|shot>",
            )

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)
        rows = component.storage.list_by_group(table, group_id, _SHOW_LIMIT)
        if not rows:
            return CommandResult(success=True, message=f"当前群暂无 {category} 条目")

        if category == "jargon":
            lines = [
                f"- {_truncate(r['term'], 20)}｜证据 {r['evidence_count']}"
                f"｜{_truncate(r.get('meaning') or '（无含义）')}"
                f"｜{_STATUS_LABELS.get(r['status'], r['status'])}"
                for r in rows
            ]
        elif category == "pattern":
            lines = [
                f"- [{r['scene']}] {_truncate(r['expression'])}"
                f"｜命中 {r['hit_count']} 次"
                f"｜{_STATUS_LABELS.get(r['status'], r['status'])}"
                for r in rows
            ]
        else:
            lines = [
                f"- 用户：{_truncate(r['user_text'])}\n"
                f"  回复：{_truncate(r['bot_text'])}"
                f"｜{_STATUS_LABELS.get(r['status'], r['status'])}"
                for r in rows
            ]

        message = f"📖 当前群 {category} 条目（前 {len(rows)} 条）\n" + "\n".join(lines)
        return CommandResult(success=True, message=message)

    async def _handle_remember(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """手动记住对话上下文并链入表达方式与对话样例。

        格式：``iris_mem learning remember <上下文> => <表达示例>``。
        人工手令代表明确意图，few_shot 对话对与表达模式均以 approved
        直接落库（跳过 LLM 审查），立即参与回复注入。
        """
        component = self._get_component()
        if not component:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)

        # raw_args[0] 是 "remember" 本身；其余拼回后按 => 拆分
        text = " ".join(args.raw_args[1:]).strip()
        if "=>" not in text:
            return CommandResult(
                success=False,
                message=(
                    "格式错误，正确用法：\n"
                    "iris_mem learning remember <上下文> => <表达示例>"
                ),
            )
        context_part, expression_part = text.split("=>", 1)
        context_part = context_part.strip()
        expression_part = expression_part.strip()
        if not context_part or not expression_part:
            return CommandResult(
                success=False, message="上下文与表达示例均不能为空"
            )

        storage = component.storage
        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event) or ""
        user_id = adapter.get_user_id(event) or ""

        from iris_memory.core.persona import resolve_persona

        manager = get_component_manager()
        persona_id = await resolve_persona(manager, event)

        # 表达模式的 scene：无 LLM 参与，用上下文前 20 字作场景标签
        scene = context_part.replace("\n", " ")[:20]

        try:
            pair_id = storage.insert_pair(
                group_id,
                user_id,
                context_part,
                expression_part,
                persona_id=persona_id,
                approved=True,
            )
            pattern_id = storage.insert_pattern(
                group_id,
                scene,
                expression_part,
                source_pair_id=pair_id,
                persona_id=persona_id,
                approved=True,
            )
        except Exception as e:
            logger.error(f"learning remember 写入失败：{e}", exc_info=True)
            return CommandResult(success=False, message=f"写入失败：{e}")

        logger.info(
            f"learning remember 操作: pair_id={pair_id}, pattern_id={pattern_id}, "
            f"persona={persona_id}"
        )
        return CommandResult(
            success=True,
            message=(
                "✅ 已记入风格学习（直接生效）\n"
                f"上下文: {_truncate(context_part)}\n"
                f"表达示例: {_truncate(expression_part)}\n"
                f"对话样例 #{pair_id} · 表达模式 #{pattern_id}"
            ),
            details={"pair_id": pair_id, "pattern_id": pattern_id},
        )

    async def _handle_clear(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理清空操作"""
        component = self._get_component()
        if not component:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)

        storage = component.storage
        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)
        current_user_id = adapter.get_user_id(event)

        scope = args.scope

        if scope == DeleteScope.ALL:
            storage.clear_all()
            message = "✅ 已清空所有学习数据"

        elif scope == DeleteScope.GROUP:
            storage.clear_by_group(group_id)
            message = "✅ 已清空当前群聊的学习数据"

        elif scope == DeleteScope.SPECIFIED_USER:
            target_user_id = args.target_user_id
            if not target_user_id:
                return CommandResult(success=False, message="无法获取目标用户 ID")
            storage.clear_by_user(target_user_id, group_id)
            message = (
                f"✅ 已清空用户 {args.target_user_name or target_user_id}"
                " 在当前群聊的对话样例（表达模式/暗语无用户维度，不随用户清理）"
            )

        else:
            storage.clear_by_user(current_user_id, group_id)
            message = "✅ 已清空你的对话样例（表达模式/暗语无用户维度，不随用户清理）"

        logger.info(f"learning clear 操作: scope={scope.value}")
        return CommandResult(
            success=True, message=message, details={"scope": scope.value}
        )

    async def _handle_jargon(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理词条禁用/恢复：jargon disable <词> / jargon enable <词>"""
        component = self._get_component()
        if not component:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)

        # raw_args 含子指令本身（["jargon", action, term]）
        if len(args.raw_args) < 3:
            return CommandResult(
                success=False,
                message="用法: iris_mem learning jargon <disable|enable> <词>",
            )
        action, term = args.raw_args[1], args.raw_args[2]
        if action not in ("disable", "enable"):
            return CommandResult(
                success=False,
                message="用法: iris_mem learning jargon <disable|enable> <词>",
            )

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)
        status = STATUS_DISABLED if action == "disable" else STATUS_ACTIVE
        component.storage.set_jargon_status(group_id, term, status)

        action_label = "禁用" if action == "disable" else "恢复"
        logger.info(f"learning jargon {action}: group={group_id}, term={term}")
        return CommandResult(
            success=True,
            message=f"✅ 已{action_label}词条「{term}」的注入",
            details={"term": term, "status": status},
        )

"""LLM Tool 统一注册入口

全部 9 个工具（记忆侧 6 个 + 主动回复侧 3 个）由 build_llm_tools()
构建、_validate_tools() 校验、register_llm_tools() 注册并统一修正
插件归属（handler_module_path）。

注意：AstrBot 4.27.2 的 Context.add_llm_tools() 会按批内第一个工具的
__module__ 重新计算并覆盖同批工具的 handler_module_path，因此归属必须
在调用之后统一写入。
"""

from collections import Counter
from collections.abc import Sequence

from astrbot.core.agent.tool import FunctionTool

from iris_memory.core import get_logger
from iris_memory.proactive.state import StateManager
from iris_memory.proactive.tools import ToolContext

from .correct_memory import CorrectMemoryTool
from .get_profile import GetProfileTool
from .proactive import AddFollowUpTool, EndFollowUpTool, SetCooldownTool
from .save_knowledge import SaveKnowledgeTool
from .save_memory import SaveMemoryTool
from .search_knowledge_graph import SearchKnowledgeGraphTool
from .search_memory import SearchMemoryTool

logger = get_logger("tools")

EXPECTED_TOOL_NAMES = frozenset(
    {
        "save_knowledge",
        "save_memory",
        "search_memory",
        "correct_memory",
        "search_knowledge_graph",
        "get_profile",
        "add_follow_up",
        "end_follow_up",
        "set_cooldown",
    }
)


def build_llm_tools(
    *,
    state: StateManager,
    tool_context: ToolContext,
) -> list[FunctionTool]:
    """构建全部 LLM Tool，每次调用都返回全新实例（不跨热重载复用）。"""
    tools = [
        SaveKnowledgeTool(),
        SaveMemoryTool(),
        SearchMemoryTool(),
        CorrectMemoryTool(),
        SearchKnowledgeGraphTool(),
        GetProfileTool(),
        AddFollowUpTool(state, tool_context),
        EndFollowUpTool(state, tool_context),
        SetCooldownTool(state, tool_context),
    ]
    _validate_tools(tools)
    return tools


def _validate_tools(tools: Sequence[FunctionTool]) -> None:
    names = [tool.name for tool in tools]
    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"重复的 LLM Tool 名称: {', '.join(duplicates)}")

    actual = set(names)
    if actual != EXPECTED_TOOL_NAMES:
        missing = sorted(EXPECTED_TOOL_NAMES - actual)
        unexpected = sorted(actual - EXPECTED_TOOL_NAMES)
        raise ValueError(
            f"LLM Tool 集合不完整: missing={missing}, unexpected={unexpected}"
        )


def register_llm_tools(
    *,
    context,
    owner_module: str,
    state: StateManager,
    tool_context: ToolContext,
) -> tuple[FunctionTool, ...]:
    """通过 AstrBot 公共 API 注册全部工具，并在之后统一写入插件归属。

    Args:
        context: AstrBot Star Context（使用其 add_llm_tools() 公共接口）。
        owner_module: 插件主模块路径，必须由插件类提供
            （self.__class__.__module__），不允许硬编码。
        state: 主动回复侧状态管理器。
        tool_context: 主动回复工具上下文。

    Returns:
        已注册的工具对象元组，供插件实例保存以便诊断与测试。
    """
    if not owner_module:
        raise ValueError("LLM Tool owner_module 不能为空")

    tools = build_llm_tools(state=state, tool_context=tool_context)
    context.add_llm_tools(*tools)

    # add_llm_tools() 会覆盖 handler_module_path，归属必须在其后写入
    for tool in tools:
        tool.handler_module_path = owner_module

    invalid = [
        tool.name for tool in tools if tool.handler_module_path != owner_module
    ]
    if invalid:
        raise RuntimeError(f"LLM Tool 归属写入失败: {invalid}")

    return tuple(tools)

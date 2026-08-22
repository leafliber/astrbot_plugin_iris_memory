"""Registry 单元测试：工具集合完整性、Schema 回归、归属写入与重复注册。"""

from types import SimpleNamespace

import pytest
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.star.context import Context

from iris_memory.proactive.config import ConfigManager
from iris_memory.proactive.state import StateManager
from iris_memory.proactive.tools import ToolContext
from iris_memory.tools import EXPECTED_TOOL_NAMES, build_llm_tools, register_llm_tools
from iris_memory.tools.registry import _validate_tools

# 9 个工具的精确 Schema 快照。注册重构必须保持 Schema 深度相等，
# 任何调整（新增 required/default/minimum/maximum、修改类型）都应单独提交。
EXPECTED_SCHEMAS = {
    "save_knowledge": {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "description": "节点列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "节点类型（如 Person, Event, Concept）",
                        },
                        "name": {"type": "string", "description": "实体名称"},
                        "content": {"type": "string", "description": "实体描述"},
                        "confidence": {
                            "type": "number",
                            "description": "置信度（0.0-1.0）",
                            "default": 1.0,
                        },
                    },
                    "required": ["label", "name", "content"],
                },
            },
            "edges": {
                "type": "array",
                "description": "边列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_name": {
                            "type": "string",
                            "description": "源实体名称（必须在nodes中定义）",
                        },
                        "target_name": {
                            "type": "string",
                            "description": "目标实体名称（必须在nodes中定义）",
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "关系类型（如 KNOWS, RELATED_TO）",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "置信度（0.0-1.0）",
                            "default": 1.0,
                        },
                    },
                    "required": ["source_name", "target_name", "relation_type"],
                },
            },
        },
        "required": ["nodes"],
    },
    "save_memory": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "记忆内容（简洁明确，不超过500字）",
            },
            "confidence": {
                "type": "number",
                "description": "置信度（0.0-1.0，表示记忆的可靠性）",
                "default": 1.0,
            },
            "importance": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "重要度：high=长期稳定的核心信息（身份/职业/亲密关系/重大事件/持久偏好），"
                    "medium=一般性事实与阶段性信息，low=边缘信息"
                ),
                "default": "medium",
            },
            "ttl_hours": {
                "type": "number",
                "description": (
                    "存活时长（小时，可选）。仅对确实会过期的临时事实设置，"
                    "如「明天考试」「这周末搬家」；长期信息不要设置"
                ),
            },
            "scope": {
                "type": "string",
                "enum": ["group", "global"],
                "description": (
                    "作用域：group=仅当前群可见可检索（默认）；"
                    "global=全局共享，所有群与私聊均可检索。"
                    "仅当信息属于 bot 自身/主人等跨群通用事实时才用 global"
                ),
                "default": "group",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "标签列表（可选，用于分类记忆）",
            },
        },
        "required": ["content"],
    },
    "search_memory": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "查询文本（描述你想查找的记忆）",
            },
            "top_k": {
                "type": "integer",
                "description": "返回的记忆数量（默认5条，最多20条）",
                "default": 5,
            },
            "with_graph_context": {
                "type": "boolean",
                "description": "是否同时从知识图谱获取关联实体的上下文（默认false）",
                "default": False,
            },
        },
        "required": ["query"],
    },
    "correct_memory": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "要修正的记忆ID（格式：mem_xxxxxxxxxx）",
            },
            "correction": {
                "type": "string",
                "description": "修正后的正确内容",
            },
            "reason": {
                "type": "string",
                "description": "修正原因（为什么原记忆是错误的）",
            },
        },
        "required": ["memory_id", "correction", "reason"],
    },
    "search_knowledge_graph": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词（实体名称或描述）",
            },
            "label": {
                "type": "string",
                "description": "节点类型过滤（可选，如 Person, Event, Concept, Location, Item, Topic）",
            },
            "expand_depth": {
                "type": "integer",
                "description": "关系扩展深度（默认1层，最多2层。1层=直接关联，2层=间接关联）",
                "default": 1,
            },
        },
        "required": ["query"],
    },
    "get_profile": {
        "type": "object",
        "properties": {
            "target_type": {
                "type": "string",
                "description": "查询类型：user（用户画像）或 group（群聊画像），默认user",
                "default": "user",
            },
            "target_id": {
                "type": "string",
                "description": "用户ID或群聊ID（可选，不传则自动获取当前用户/群聊）",
            },
        },
        "required": [],
    },
    # 3 个主动回复工具：与原 @filter.llm_tool 装饰器实际生成的 Schema 一致
    "add_follow_up": {
        "type": "object",
        "properties": {
            "user_ids": {
                "type": "string",
                "description": '逗号分隔的用户ID列表，如 "user1,user2"',
            },
        },
    },
    "end_follow_up": {
        "type": "object",
        "properties": {
            "user_ids": {
                "type": "string",
                "description": '逗号分隔的用户ID列表，如 "user1,user2"',
            },
        },
    },
    "set_cooldown": {
        "type": "object",
        "properties": {
            "minutes": {
                "type": "number",
                "description": "冷却时间（分钟），范围 1-120，默认 5",
            },
        },
    },
}

assert set(EXPECTED_SCHEMAS) == EXPECTED_TOOL_NAMES


@pytest.fixture
def state():
    return StateManager(ConfigManager({}, hidden_get={}.get))


@pytest.fixture
def tool_context():
    ctx = ToolContext()
    yield ctx
    ctx.clear_context()


@pytest.fixture
def tools(state, tool_context):
    return build_llm_tools(state=state, tool_context=tool_context)


def _make_context() -> Context:
    context = object.__new__(Context)
    context.provider_manager = SimpleNamespace(llm_tools=ToolSet())
    return context


def test_build_returns_nine_tools(tools):
    assert len(tools) == 9


def test_tool_names_match_expected(tools):
    assert {tool.name for tool in tools} == EXPECTED_TOOL_NAMES


def test_no_duplicate_tool_names(tools):
    names = [tool.name for tool in tools]
    assert len(names) == len(set(names))


def test_all_tools_are_function_tools(tools):
    assert all(isinstance(tool, FunctionTool) for tool in tools)


def test_all_tools_override_call_without_handler(tools):
    for tool in tools:
        is_override_call = any(
            "call" in ty.__dict__ and ty.__dict__["call"] is not FunctionTool.call
            for ty in type(tool).mro()
        )
        assert is_override_call, f"{tool.name} 未覆盖 call()"
        assert tool.handler is None, f"{tool.name} 不应依赖 decorator handler"


def test_factory_returns_fresh_instances(state, tool_context):
    first = build_llm_tools(state=state, tool_context=tool_context)
    second = build_llm_tools(state=state, tool_context=tool_context)
    first_by_name = {tool.name: tool for tool in first}
    second_by_name = {tool.name: tool for tool in second}
    for name in EXPECTED_TOOL_NAMES:
        assert first_by_name[name] is not second_by_name[name]


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOL_NAMES))
def test_tool_schema_regression(tools, tool_name):
    tool = next(tool for tool in tools if tool.name == tool_name)
    assert tool.parameters == EXPECTED_SCHEMAS[tool_name]


def test_validate_tools_rejects_duplicates():
    dup = FunctionTool(name="save_memory", description="", parameters={})
    other = FunctionTool(name="save_memory", description="", parameters={})
    with pytest.raises(ValueError, match="重复"):
        _validate_tools([dup, other])


def test_validate_tools_rejects_incomplete_set():
    tool = FunctionTool(name="save_memory", description="", parameters={})
    with pytest.raises(ValueError, match="不完整"):
        _validate_tools([tool])


def test_register_requires_owner_module(state, tool_context):
    with pytest.raises(ValueError, match="owner_module"):
        register_llm_tools(
            context=_make_context(),
            owner_module="",
            state=state,
            tool_context=tool_context,
        )


def test_register_writes_plugin_ownership(state, tool_context):
    context = _make_context()
    registered = register_llm_tools(
        context=context,
        owner_module="data.plugins.astrbot_plugin_iris_memory.main",
        state=state,
        tool_context=tool_context,
    )

    tools = context.provider_manager.llm_tools.func_list
    assert len(tools) == 9
    assert len(registered) == 9
    assert {tool.handler_module_path for tool in tools} == {
        "data.plugins.astrbot_plugin_iris_memory.main"
    }
    assert not any(
        tool.handler_module_path.startswith("iris_memory.tools.") for tool in tools
    )


def test_double_registration_replaces_without_duplicates(state, tool_context):
    """模拟热重载：同一 ToolSet 连续注册两次，工具不重复、实例被替换。"""
    context = _make_context()
    owner = "data.plugins.astrbot_plugin_iris_memory.main"

    first = register_llm_tools(
        context=context,
        owner_module=owner,
        state=state,
        tool_context=tool_context,
    )
    second = register_llm_tools(
        context=context,
        owner_module=owner,
        state=state,
        tool_context=tool_context,
    )

    tools = context.provider_manager.llm_tools.func_list
    assert len(tools) == 9
    assert {tool.name for tool in tools} == EXPECTED_TOOL_NAMES

    first_by_name = {tool.name: tool for tool in first}
    second_by_name = {tool.name: tool for tool in second}
    for tool in tools:
        assert tool is second_by_name[tool.name]
        assert tool is not first_by_name[tool.name]
        assert tool.handler_module_path == owner

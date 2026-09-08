"""Single description of every Pages setting; mode stays in AstrBot's schema."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import IrisError


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    group: str
    kind: str
    default: object
    hint: str = ""
    minimum: float | None = None
    maximum: float | None = None
    secret: bool = False


FIELDS = (
    Field("remote_url", "Core 服务地址", "Core", "string", ""),
    Field(
        "remote_app_instance",
        "远程应用实例 ID",
        "Core",
        "string",
        "",
        "服务端启用 Active Surface 时填写，用于租约；不抢占其他宿主",
    ),
    Field("remote_token", "应用 Bearer Token", "Core", "string", "", secret=True),
    Field(
        "remote_bindings",
        "远程会话绑定",
        "Core",
        "object",
        {},
        "会话键/人格 ID → agent_id、space_id；必须与服务端授权一致",
    ),
    Field(
        "remote_actors",
        "远程用户实体映射",
        "Core",
        "object",
        {},
        "realm/用户 ID → entity_id；群聊写入必须配置每位用户，避免共享实体",
    ),
    Field(
        "request_timeout",
        "Core 请求期限（秒）",
        "Core",
        "number",
        10,
        minimum=1,
        maximum=60,
    ),
    Field(
        "allow_development_sqlite",
        "开发 SQLite 例外",
        "Core",
        "boolean",
        False,
        "仅开发验证使用，不能代替正式运行库兼容性认证",
    ),
    Field("embedding_provider", "Embedding Provider ID", "模型与成本", "string", ""),
    Field(
        "embedding_dimension",
        "向量维度",
        "模型与成本",
        "integer",
        1024,
        minimum=1,
        maximum=8192,
    ),
    Field(
        "embedding_revision",
        "向量模型修订指纹",
        "模型与成本",
        "string",
        "v1",
        "模型/维度改变前需处理旧索引；不自动复用不兼容空间",
    ),
    Field("chat_provider", "主动/学习模型 Provider ID", "模型与成本", "string", ""),
    Field("cognitive_provider", "记忆提炼模型 Provider ID", "模型与成本", "string", ""),
    Field(
        "daily_model_calls",
        "每日插件模型调用上限",
        "模型与成本",
        "integer",
        20,
        minimum=0,
        maximum=10000,
    ),
    Field(
        "daily_model_tokens",
        "每日插件模型 Token 预算",
        "模型与成本",
        "integer",
        30000,
        minimum=0,
        maximum=10000000,
    ),
    Field(
        "model_timeout",
        "模型超时（秒）",
        "模型与成本",
        "number",
        30,
        minimum=1,
        maximum=120,
    ),
    Field(
        "allowed_conversations",
        "启用会话键",
        "记忆",
        "list",
        [],
        "空列表不采集任何会话；通过 /iris status 获取会话键",
    ),
    Field(
        "context_tokens",
        "动态记忆 Token 上限",
        "记忆",
        "integer",
        1200,
        minimum=0,
        maximum=16000,
    ),
    Field(
        "model_window",
        "模型上下文窗口",
        "记忆",
        "integer",
        32768,
        minimum=1024,
        maximum=2000000,
    ),
    Field(
        "output_reserve",
        "输出预留 Token",
        "记忆",
        "integer",
        2048,
        minimum=128,
        maximum=65536,
    ),
    Field(
        "recall_candidates", "候选上限", "记忆", "integer", 12, minimum=1, maximum=100
    ),
    Field(
        "queue_items",
        "交付队列条数上限",
        "记忆",
        "integer",
        256,
        minimum=1,
        maximum=10000,
    ),
    Field(
        "queue_bytes",
        "交付队列字节上限",
        "记忆",
        "integer",
        2097152,
        minimum=16384,
        maximum=16777216,
    ),
    Field(
        "queue_ttl_hours",
        "交付队列有效期（小时）",
        "记忆",
        "integer",
        24,
        minimum=1,
        maximum=168,
    ),
    Field("default_persona", "默认人格 ID", "人格", "string", ""),
    Field("timezone", "时区", "主动", "string", "Asia/Shanghai"),
    Field("quiet_start", "静音起始小时", "主动", "integer", 1, minimum=0, maximum=23),
    Field("quiet_end", "静音结束小时", "主动", "integer", 7, minimum=0, maximum=23),
    Field(
        "proactive_cooldown",
        "会话发言冷却（秒）",
        "主动",
        "integer",
        300,
        minimum=10,
        maximum=86400,
    ),
    Field(
        "catchup_seconds",
        "提醒补发窗口（秒）",
        "主动",
        "integer",
        3600,
        minimum=0,
        maximum=86400,
    ),
    Field(
        "initiate_after_seconds",
        "静默主动发起间隔（秒）",
        "主动",
        "integer",
        0,
        minimum=0,
        maximum=604800,
    ),
    Field(
        "log_level",
        "详细日志级别",
        "日志",
        "string",
        "INFO",
        "DEBUG / INFO / WARNING / ERROR",
    ),
    Field("log_bodies", "记录脱敏模型正文", "日志", "boolean", False),
    Field(
        "log_megabytes",
        "日志容量（MiB）",
        "日志",
        "integer",
        100,
        minimum=1,
        maximum=1024,
    ),
    Field(
        "log_days", "日志最长保留（天）", "日志", "integer", 7, minimum=1, maximum=90
    ),
)
MODULES = {
    "memory": ("记忆", ()),
    "context": ("上下文", ("memory",)),
    "persona": ("人格采用", ()),
    "proactive": ("主动陪伴", ("context",)),
    "learning": ("人格学习", ("persona", "memory")),
    "media": ("图片描述", ("memory",)),
    "maintenance": ("后台维护", ("memory",)),
    "diagnostics": ("完整运行日志", ()),
}


def defaults():
    return {
        **{f.key: copy.deepcopy(f.default) for f in FIELDS},
        "modules": {key: False for key in MODULES},
    }


def describe():
    return [asdict(f) for f in FIELDS]


def validate(current, changes):
    if not isinstance(changes, dict):
        raise IrisError("invalid_config", "配置必须是 JSON 对象")
    result = copy.deepcopy(current)
    definitions = {f.key: f for f in FIELDS}
    for key, value in changes.items():
        if key == "modules":
            if not isinstance(value, dict) or set(value) - set(MODULES):
                raise IrisError("invalid_config", "模块配置无效")
            if any(type(enabled) is not bool for enabled in value.values()):
                raise IrisError("invalid_config", "模块开关必须是布尔值")
            result[key].update(value)
            continue
        field = definitions.get(key)
        if field is None:
            raise IrisError("invalid_config", f"未知配置项：{key}")
        types = {
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
            "list": (list,),
            "object": (dict,),
        }
        if type(value) not in types[field.kind]:
            raise IrisError("invalid_config", f"{field.label}类型错误")
        if isinstance(value, str) and len(value) > 8192:
            raise IrisError("invalid_config", f"{field.label}过长")
        if field.minimum is not None and not field.minimum <= value <= field.maximum:
            raise IrisError("invalid_config", f"{field.label}超出范围")
        result[key] = value
    for name, (_, deps) in MODULES.items():
        if result["modules"][name] and any(not result["modules"][d] for d in deps):
            raise IrisError("dependency_disabled", f"{name} 依赖 {'、'.join(deps)}")
    if result["remote_url"]:
        url = urlsplit(result["remote_url"])
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise IrisError(
                "invalid_config", "Core 地址必须是不含凭据、查询或片段的 HTTP(S) 地址"
            )
    if result["log_level"] not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise IrisError("invalid_config", "日志级别无效")
    if len(result["allowed_conversations"]) > 512 or any(
        not isinstance(s, str) or len(s) > 256 for s in result["allowed_conversations"]
    ):
        raise IrisError("invalid_config", "启用会话列表无效")
    if len(result["remote_bindings"]) > 512:
        raise IrisError("invalid_config", "远程绑定数量超出上限")
    seen_agents = set()
    for key, binding in result["remote_bindings"].items():
        if (
            not isinstance(key, str)
            or not isinstance(binding, dict)
            or set(binding) - {"agent_id", "space_id", "entity_id", "realm"}
            or not all(
                isinstance(binding.get(k), str) and 0 < len(binding[k]) <= 256
                for k in ("agent_id", "space_id")
            )
        ):
            raise IrisError("invalid_config", "远程绑定须包含 agent_id 与 space_id")
        if binding["agent_id"] in seen_agents:
            raise IrisError("scope_shared", "不同会话/人格不能共用一个远程 Agent")
        seen_agents.add(binding["agent_id"])
        if any(
            not isinstance(v, str) or not 1 <= len(v) <= 256 for v in binding.values()
        ):
            raise IrisError("invalid_config", "远程绑定字段须为非空字符串")
    if len(result["remote_actors"]) > 2000 or any(
        not isinstance(k, str)
        or not 1 <= len(k) <= 400
        or not isinstance(v, str)
        or not 1 <= len(v) <= 256
        for k, v in result["remote_actors"].items()
    ):
        raise IrisError("invalid_config", "远程用户实体映射无效")
    if result["output_reserve"] >= result["model_window"]:
        raise IrisError("invalid_config", "输出预留必须小于模型窗口")
    try:
        ZoneInfo(result["timezone"])
    except (ZoneInfoNotFoundError, ValueError):
        raise IrisError("invalid_config", "时区无效") from None
    return result


def public_settings(values):
    result = copy.deepcopy(values)
    for field in FIELDS:
        if field.secret:
            result[field.key] = ""
            result[field.key + "_configured"] = bool(values[field.key])
    return result

"""
Iris Chat Memory - Persona 解析器

在请求时从 AstrBot 会话上下文解析当前 persona_id，供 profile / L2 / L1 等
模块做人格隔离。解析结果在同一事件对象生命周期内缓存（event extra），
避免 on_all_message → on_llm_request → tools 链路重复查库。

解析策略：
1. persona 隔离未启用 → 直接返回 "default"
2. req 可用（on_llm_request 路径）→ req.conversation.persona_id（权威来源）
3. 否则经 conversation_manager 查当前会话 → conv.persona_id
4. None / "[%None]" → "default"

不依赖 persona_manager.resolve_selected_persona（避免额外副作用与 AstrBot 版本耦合）。
"""

from typing import TYPE_CHECKING, Optional

from iris_memory.core import Component, get_logger

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.star import Context
    from iris_memory.core.components import ComponentManager

logger = get_logger("persona")

# event extra 上的缓存键
_EXTRA_KEY = "iris_persona_id"

# 视为 default 的 persona 值
_DEFAULT_VALUES = (None, "", "default", "[%None]")


class PersonaResolver(Component):
    """Persona 解析组件

    在请求时解析当前事件的 persona_id。注册名 "persona_resolver"，
    EAGER 初始化（无重资源）。
    """

    def __init__(self, context: "Context"):
        super().__init__()
        self._context = context

    @property
    def name(self) -> str:
        return "persona_resolver"

    async def initialize(self) -> None:
        self._is_available = True
        logger.info(
            f"PersonaResolver 初始化成功，人格隔离：{'启用' if self.is_enabled() else '未启用'}"
        )

    async def shutdown(self) -> None:
        self._reset_state()

    def is_enabled(self) -> bool:
        """persona 隔离是否启用"""
        try:
            from iris_memory.config import get_config

            return bool(get_config().get("isolation_config.enable_persona_isolation"))
        except RuntimeError:
            return False

    async def resolve(
        self,
        event: "AstrMessageEvent",
        req: Optional[object] = None,
    ) -> str:
        """解析 persona_id，结果缓存到 event extra

        Args:
            event: AstrBot 消息事件
            req: 可选的 ProviderRequest（on_llm_request 路径传入）

        Returns:
            persona_id 字符串，未启用隔离时恒为 "default"
        """
        if not self.is_enabled():
            return "default"

        # 同一事件生命周期内复用
        try:
            cached = event.get_extra(_EXTRA_KEY)
            if cached:
                return str(cached)
        except Exception:
            pass

        persona_id = await self._do_resolve(event, req)

        try:
            event.set_extra(_EXTRA_KEY, persona_id)
        except Exception:
            pass

        return persona_id

    async def _do_resolve(
        self,
        event: "AstrMessageEvent",
        req: Optional[object],
    ) -> str:
        # 1) req.conversation.persona_id（on_llm_request 路径，权威来源）
        persona_id = self._extract_from_req(req)
        if persona_id:
            return persona_id

        # 2) 经 conversation_manager 查当前会话
        persona_id = await self._extract_from_conversation(event)
        if persona_id:
            return persona_id

        return "default"

    @staticmethod
    def _extract_from_req(req: Optional[object]) -> str:
        if req is None:
            return ""
        conversation = getattr(req, "conversation", None)
        if conversation is None:
            return ""
        pid = getattr(conversation, "persona_id", None)
        return _normalize(pid)

    async def _extract_from_conversation(self, event: "AstrMessageEvent") -> str:
        conv_mgr = getattr(self._context, "conversation_manager", None)
        if conv_mgr is None:
            return ""
        try:
            umo = getattr(event, "unified_msg_origin", None)
            if not umo:
                return ""
            cid = await conv_mgr.get_curr_conversation_id(umo)
            if not cid:
                return ""
            conv = await conv_mgr.get_conversation(umo, cid)
            if conv is None:
                return ""
            pid = getattr(conv, "persona_id", None)
            return _normalize(pid)
        except Exception as e:
            logger.debug(f"经会话管理器解析 persona_id 失败，回退 default: {e}")
            return ""


def _normalize(pid: object) -> str:
    """规范化 persona_id：default 类值返回空串（由调用方兜底 "default"）"""
    if pid is None:
        return ""
    s = str(pid).strip()
    if s in ("", "default", "[%None]"):
        return ""
    return s


async def resolve_persona(
    component_manager: "ComponentManager",
    event: "AstrMessageEvent",
    req: Optional[object] = None,
) -> str:
    """便捷函数：从组件管理器取 PersonaResolver 并解析

    Args:
        component_manager: 组件管理器
        event: 消息事件
        req: 可选 ProviderRequest

    Returns:
        persona_id；隔离未启用或解析器不可用时返回 "default"
    """
    resolver = component_manager.get_available_component("persona_resolver")
    if resolver is None or not isinstance(resolver, PersonaResolver):
        return "default"
    return await resolver.resolve(event, req)

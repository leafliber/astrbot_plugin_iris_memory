"""
LLM 请求钩子处理模块

负责处理 LLM 请求前的钩子逻辑，包括：
- L1 上下文注入
- 用户画像注入
- 图片解析（related 模式）
- L2 记忆注入
- L3 知识图谱注入

注入策略：
所有动态内容（L1/画像/L2/L3）统一通过 req.extra_user_content_parts 注入，
作为额外的用户消息内容块放在本轮用户输入之后。不修改 req.system_prompt、
req.contexts 和 req.prompt。

这样做的理由：
- L1/L2/L3/画像均为每轮变化的动态上下文，不属于稳定角色设定
- 注入 system_prompt 会使系统提示词每轮变化，破坏模型服务端的提示词缓存，
  显著增加请求成本和首 token 延迟
- extra_user_content_parts 适合承载"本轮相关记忆片段"等动态上下文
- 不修改 req.contexts 和 req.prompt，避免与 AstrBot 对话管理冲突

所有注入的 TextPart 均调用 mark_as_temp() 标记为临时内容（需 AstrBot >= 4.24.0），
使其不持久化到会话历史中。若当前 AstrBot 版本不支持 mark_as_temp()，则自动跳过。
"""

from typing import TYPE_CHECKING, List, Optional, cast
import re

from iris_memory.core import get_logger

_KG_STOPWORDS = frozenset(
    {
        "什么",
        "怎么",
        "如何",
        "为什么",
        "这个",
        "那个",
        "今天",
        "昨天",
        "明天",
        "喜欢",
        "觉得",
        "想要",
        "可以",
        "知道",
        "一下",
        "一些",
        "告诉",
        "请问",
        "还是",
        "的话",
        "不是",
        "没有",
        "现在",
        "已经",
        "应该",
        "可能",
        "因为",
        "所以",
        "但是",
    }
)

_QUOTED_PATTERN = re.compile(r'[""「」『』]([^""「」『』]+)[""「」『』]')
_CHINESE_WORD_PATTERN = re.compile(r"[一-龥]{2,6}")

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import ProviderRequest
    from iris_memory.core.components import ComponentManager
    from iris_memory.l1_buffer import L1Buffer
    from iris_memory.l1_buffer.models import ContextMessage
    from iris_memory.l2_memory.models import MemorySearchResult
    from iris_memory.profile.models import GroupProfile, UserProfile

logger = get_logger("llm_request_hook")


async def preprocess_llm_request(
    event: "AstrMessageEvent",
    req: "ProviderRequest",
    component_manager: "ComponentManager",
) -> None:
    """LLM 请求钩子处理

    执行所有 LLM 对话前的预处理逻辑。

    注入策略：
    - req.extra_user_content_parts: L1/画像/L2/L3（全部作为动态上下文内容块）
    - req.system_prompt: 不修改
    - req.contexts: 不修改
    - req.prompt: 不修改

    Args:
        event: AstrBot 消息事件对象
        req: LLM 提供者请求对象
        component_manager: 组件管理器实例
    """
    # 故障隔离：每个步骤独立 try/except，单步异常不影响其余已收集的上下文。
    # 历史 bug：无顶层 try/except，[t.result() for t in done] 会 re-raise，
    # 异常穿透使已收集的 L1/画像/L2/L3 注入全丢，违背故障隔离约定。

    l1_text = ""
    profile_text = ""
    l2_text = ""
    l2_results = []
    l3_text = ""
    inject_meta: dict = {"l1": {}, "profile": {}, "l2": {}, "l3": {}}

    try:
        await _parse_images_if_related_mode(event, req, component_manager)
    except Exception as e:
        logger.error(f"图片解析（related 模式）失败，已隔离：{e}", exc_info=True)

    try:
        l1_text = await _collect_l1_context(
            event, req, component_manager, meta=inject_meta["l1"]
        )
    except Exception as e:
        logger.error(f"L1 上下文收集失败，已隔离：{e}", exc_info=True)

    try:
        profile_text = await _collect_user_profile(
            event, component_manager, meta=inject_meta["profile"]
        )
    except Exception as e:
        logger.error(f"用户画像收集失败，已隔离：{e}", exc_info=True)

    try:
        l2_text, l2_results = await _collect_l2_memory(
            event, component_manager, meta=inject_meta["l2"]
        )
    except Exception as e:
        logger.error(f"L2 记忆检索失败，已隔离：{e}", exc_info=True)

    user_message = ""
    if hasattr(event, "message_str") and event.message_str:
        user_message = event.message_str
    elif hasattr(event, "get_message_str"):
        user_message = event.get_message_str()

    try:
        l3_text = await _collect_l3_knowledge_graph(
            event, component_manager, l2_results, user_message, meta=inject_meta["l3"]
        )
    except Exception as e:
        logger.error(f"L3 知识图谱检索失败，已隔离：{e}", exc_info=True)

    combined = _inject_to_extra_user_content_parts(
        req, l1_text, profile_text, l2_text, l3_text
    )

    _record_injection_log(
        event,
        req,
        l1_text=l1_text,
        profile_text=profile_text,
        l2_text=l2_text,
        l3_text=l3_text,
        meta=inject_meta,
        combined=combined,
        user_message=user_message,
    )

    _log_final_context(req)


def _record_injection_log(
    event: "AstrMessageEvent",
    req: "ProviderRequest",
    *,
    l1_text: str,
    profile_text: str,
    l2_text: str,
    l3_text: str,
    meta: dict,
    combined: str,
    user_message: str,
) -> None:
    """写入统一运行日志（injection 类型），失败不影响主流程"""
    try:
        from iris_memory.core.run_log import get_run_log_manager

        group_id = ""
        session_id = ""
        try:
            from iris_memory.platform import get_adapter

            adapter = get_adapter(event)
            group_id = adapter.get_group_id(event) or ""
            session_id = adapter.get_session_id(event) or ""
        except Exception:
            pass

        sections = {
            "l1_context": {"chars": len(l1_text), "injected": bool(l1_text), **meta.get("l1", {})},
            "profile": {"chars": len(profile_text), "injected": bool(profile_text), **meta.get("profile", {})},
            "l2_memory": {"chars": len(l2_text), "injected": bool(l2_text), **meta.get("l2", {})},
            "l3_kg": {"chars": len(l3_text), "injected": bool(l3_text), **meta.get("l3", {})},
        }
        injected_count = sum(1 for s in sections.values() if s["injected"])

        image_meta = getattr(req, "_iris_image_meta", None)
        if not isinstance(image_meta, dict):
            image_meta = None

        if injected_count:
            title = f"注入 {injected_count} 个 section（共 {len(combined)} 字符）"
        else:
            title = "所有 section 均为空，未注入"

        get_run_log_manager().record(
            "injection",
            title,
            success=injected_count > 0,
            group_id=group_id,
            session_id=session_id,
            user_message=user_message,
            injected_sections=injected_count,
            total_chars=len(combined),
            sections=sections,
            image=image_meta,
            content=combined,
        )
    except Exception as e:
        logger.debug(f"注入运行日志记录失败（已忽略）：{e}")


def _inject_to_extra_user_content_parts(
    req: "ProviderRequest",
    l1_text: str,
    profile_text: str,
    l2_text: str,
    l3_text: str,
) -> str:
    """将所有动态内容注入到 req.extra_user_content_parts

    所有内容（L1/画像/L2/L3）均为每轮变化的动态上下文，通过
    extra_user_content_parts 注入，避免修改 system_prompt 导致
    提示词缓存失效。

    注入顺序：L1 上下文 → 画像 → L2 记忆 → L3 知识图谱

    Args:
        req: LLM 提供者请求对象
        l1_text: L1 对话上下文文本
        profile_text: 用户画像文本
        l2_text: 相关记忆文本
        l3_text: 知识图谱文本

    Returns:
        实际注入的合并文本，无内容注入时返回空字符串
    """
    sections = [
        ("l1_context", l1_text),
        ("profile", profile_text),
        ("l2_memory", l2_text),
        ("l3_kg", l3_text),
    ]

    parts = []
    inject_summary = []
    for section_name, content in sections:
        if content:
            parts.append(f"<iris:{section_name}>\n{content}\n</iris:{section_name}>")
            inject_summary.append(f"{section_name}({len(content)}字)")
        else:
            inject_summary.append(f"{section_name}(空)")

    if not parts:
        logger.debug("注入摘要：所有 section 均为空，跳过注入")
        return ""

    logger.debug(f"注入摘要：{', '.join(inject_summary)}")

    combined = "\n\n".join(parts)

    from astrbot.core.agent.message import TextPart as _TextPart

    text_part = _TextPart(text=combined)

    if hasattr(text_part, "mark_as_temp"):
        text_part.mark_as_temp()

    req.extra_user_content_parts.append(text_part)

    return combined


async def _build_image_map(
    l1_buffer: "L1Buffer",
    group_id: str,
    component_manager: "ComponentManager",
) -> dict:
    """构建图片解析结果映射表

    从 L1 Buffer 图片队列和缓存中获取已解析的图片内容，
    按 message_id 和时间窗口关联到对应的消息。

    Args:
        l1_buffer: L1 Buffer 实例
        group_id: 群聊 ID
        component_manager: 组件管理器实例

    Returns:
        映射表，key 为 message_id 或 (user_id, timestamp_window)，value为图片描述列表
    """
    from iris_memory.image import ImageParseStatus

    cache_manager = component_manager.get_component("image_cache")

    all_images = l1_buffer.get_images(group_id, only_pending=False)
    if not all_images:
        return {}

    image_map: dict[str, list[str]] = {}

    for img_item in all_images:
        if img_item.status != ImageParseStatus.SUCCESS:
            continue

        desc: Optional[str] = None

        if cache_manager and cache_manager.is_available:
            cached = await cache_manager.get_cache(img_item.image_hash)
            if cached and cached.content:
                desc = cached.content

        if not desc:
            continue

        if img_item.message_id:
            key = img_item.message_id
        else:
            ts = img_item.timestamp
            key = f"{img_item.user_id}:{ts.hour}:{ts.minute}"

        if key not in image_map:
            image_map[key] = []
        image_map[key].append(desc)

    return image_map


def _get_inline_image_desc(
    msg: "ContextMessage",
    msg_id: Optional[str],
    image_map: dict,
) -> str:
    """获取消息的行内图片描述

    根据消息 ID 或时间窗口匹配图片解析结果，
    返回行内格式的图片描述文本。

    Args:
        msg: 上下文消息
        msg_id: 消息 ID
        image_map: 图片映射表

    Returns:
        行内图片描述，如 " [图:一只猫的照片]"，无匹配时返回空字符串
    """
    if not image_map:
        return ""

    descs: Optional[list[str]] = None

    if msg_id and msg_id in image_map:
        descs = image_map[msg_id]
    else:
        ts = msg.timestamp
        source = msg.source
        key = f"{source}:{ts.hour}:{ts.minute}"
        descs = image_map.get(key)

    if not descs:
        return ""

    if len(descs) == 1:
        return f" [图:{descs[0]}]"

    parts = "；".join(descs)
    return f" [图:{parts}]"


async def _collect_l1_context(
    event: "AstrMessageEvent",
    req: "ProviderRequest",
    component_manager: "ComponentManager",
    meta: Optional[dict] = None,
) -> str:
    """收集 L1 上下文文本

    将 L1 上下文格式化为纯文本返回，作为动态上下文注入到 extra_user_content_parts。
    不模拟为 user/assistant 对话格式，因为 L1 是辅助上下文而非真实对话历史。

    Args:
        event: AstrBot 消息事件对象
        req: LLM 提供者请求对象
        component_manager: 组件管理器实例
        meta: 可选的运行日志元信息字典，函数会填充消息数与截断统计

    Returns:
        格式化的 L1 上下文文本，不可用时返回空字符串
    """
    from iris_memory.platform import get_adapter

    buffer = component_manager.get_available_component("l1_buffer")
    if not buffer:
        logger.debug("L1 Buffer 组件不可用，跳过上下文注入")
        if meta is not None:
            meta["skipped"] = "component_unavailable"
        return ""

    from iris_memory.config import get_config

    config = get_config()

    l1_buffer = cast("L1Buffer", buffer)

    adapter = get_adapter(event)
    group_id = adapter.get_group_id(event)
    # L1 队列键使用会话 ID（私聊为 private:{user_id}），
    # 与写入侧 message_hook 保持一致，避免不同私聊用户共享空字符串队列
    session_id = adapter.get_session_id(event)

    max_length = cast(int, config.get("l1_buffer.inject_queue_length", 50))

    messages = l1_buffer.get_context(session_id, max_length)
    if not messages:
        logger.debug(f"群聊 {group_id} 的 L1 上下文为空，跳过注入")
        if meta is not None:
            meta["skipped"] = "empty"
        return ""

    current_user_id = adapter.get_user_id(event)
    excluded_current = False
    if (
        current_user_id
        and messages
        and messages[-1].role == "user"
        and messages[-1].source == current_user_id
    ):
        messages = messages[:-1]
        excluded_current = True

    if not messages:
        logger.debug(f"群聊 {group_id} 排除当前消息后 L1 上下文为空，跳过注入")
        if meta is not None:
            meta["skipped"] = "empty_after_excluding_current"
        return ""

    max_content_chars = cast(int, config.get("l1_inject_max_content_chars"))

    lines = []
    if group_id:
        lines.append("【近期群聊记录】")
        lines.append(
            "以下是群里最近的对话，帮助你了解当前话题。"
            "消息按从旧到新排列，越靠下越接近当前对话。"
            "其中 [图] 标记为对话中发送的图片的辅助描述，"
            "仅用于辅助理解对话内容："
        )
    else:
        lines.append("【近期对话记录】")
        lines.append(
            "以下是你们最近的对话。"
            "消息按从旧到新排列，越靠下越接近当前对话。"
            "其中 [图] 标记为对话中发送的图片的辅助描述，"
            "仅用于辅助理解对话内容："
        )

    lines.append("← 较早")

    msg_id_map: dict[str, tuple[str, str, str]] = {}
    for msg in messages:
        if msg.metadata:
            mid = msg.metadata.get("message_id")
            if mid:
                uname = (
                    msg.metadata.get("user_name", "") if msg.role == "user" else "Bot"
                )
                uid = msg.source if msg.role == "user" and msg.source else ""
                msg_id_map[str(mid)] = (msg.content, uname, uid)

    truncated_messages = 0
    truncated_replies = 0
    for idx, msg in enumerate(messages):
        content = msg.content
        role = msg.role

        if max_content_chars > 0 and len(content) > max_content_chars:
            truncated_messages += 1
            logger.debug(
                f"L1 上下文消息内容截断：群聊 {group_id}，"
                f"消息 #{idx} 原始 {len(msg.content)} 字符 → {max_content_chars} 字符"
            )
            content = content[:max_content_chars] + "..."

        if role == "user":
            user_name = msg.metadata.get("user_name") if msg.metadata else None
            user_id = msg.source if msg.source else None
            reply_content = msg.metadata.get("reply_content") if msg.metadata else None
            reply_user_name = (
                msg.metadata.get("reply_user_name") if msg.metadata else None
            )
            reply_user_id = msg.metadata.get("reply_user_id") if msg.metadata else None

            if not reply_content and msg.metadata:
                reply_mid = msg.metadata.get("reply_message_id")
                if reply_mid and str(reply_mid) in msg_id_map:
                    ref_content, ref_name, ref_uid = msg_id_map[str(reply_mid)]
                    reply_content = ref_content
                    if not reply_user_name and ref_name:
                        reply_user_name = ref_name
                    if not reply_user_id and ref_uid:
                        reply_user_id = ref_uid

            reply_tag = ""
            if reply_content:
                ref_sender = reply_user_name or "某人"
                if reply_user_id and ref_sender != reply_user_id:
                    ref_sender = f"{ref_sender}({reply_user_id})"
                if len(reply_content) > 80:
                    truncated_replies += 1
                    logger.debug(
                        f"L1 回复内容截断：群聊 {group_id}，"
                        f"消息 #{idx} 回复原始 {len(reply_content)} 字符 → 80 字符"
                    )
                    reply_content = reply_content[:80] + "..."
                reply_tag = f" ↩️回复{ref_sender}「{reply_content}」"
            elif msg.metadata and msg.metadata.get("reply_message_id"):
                reply_tag = " ↩️回复了某条消息"

            sender = user_name or "对方"
            if user_id and user_id != sender:
                sender = f"{sender}({user_id})"

            lines.append(f"{sender}:{reply_tag} {content}")
        elif role == "assistant":
            lines.append(f"Bot: {content}")

    lines.append("→ 较近（紧邻当前消息）")

    try:
        req._l1_context_count = len(messages)
    except AttributeError:
        pass

    if meta is not None:
        meta["message_count"] = len(messages)
        meta["excluded_current_message"] = excluded_current
        meta["truncated_messages"] = truncated_messages
        meta["truncated_replies"] = truncated_replies
        meta["max_content_chars"] = max_content_chars

    logger.debug(f"已收集 {len(messages)} 条 L1 上下文消息到群聊 {group_id}")

    return "\n".join(lines)


async def _collect_user_profile(
    event: "AstrMessageEvent",
    component_manager: "ComponentManager",
    meta: Optional[dict] = None,
) -> str:
    """收集用户画像文本（不直接修改 req）

    Args:
        event: AstrBot 消息事件对象
        component_manager: 组件管理器实例
        meta: 可选的运行日志元信息字典

    Returns:
        格式化的画像文本，不可用时返回空字符串
    """
    from iris_memory.config import get_config
    from iris_memory.platform import get_adapter

    config = get_config()
    if not config.get("profile.enable"):
        if meta is not None:
            meta["skipped"] = "disabled"
        return ""

    enable_auto_injection = config.get("profile.enable_auto_injection")
    if enable_auto_injection is not None and not enable_auto_injection:
        if meta is not None:
            meta["skipped"] = "auto_injection_disabled"
        return ""

    profile_storage = component_manager.get_available_component("profile")
    if not profile_storage:
        logger.debug("画像系统组件不可用，跳过画像注入")
        if meta is not None:
            meta["skipped"] = "component_unavailable"
        return ""

    adapter = get_adapter(event)
    group_id = adapter.get_group_id(event)
    user_id = adapter.get_user_id(event)

    if not user_id:
        logger.debug("无法获取用户ID，跳过画像注入")
        if meta is not None:
            meta["skipped"] = "no_user_id"
        return ""

    effective_group_id = (
        group_id if config.get("isolation_config.enable_group_isolation") else "default"
    )

    from iris_memory.core.persona import resolve_persona
    from iris_memory.profile import GroupProfileManager, UserProfileManager

    persona_id = await resolve_persona(component_manager, event)

    group_manager = GroupProfileManager(profile_storage)
    user_manager = UserProfileManager(profile_storage)

    group_profile = await group_manager.get_or_create(group_id, persona_id)
    user_profile = await user_manager.get_or_create(
        user_id, effective_group_id, persona_id
    )

    profile_text = _format_profiles_for_injection(group_profile, user_profile)

    if profile_text:
        logger.debug(f"已收集画像信息：群聊 {group_id} 用户 {user_id}")

    if meta is not None:
        meta["user_id"] = user_id
        meta["group_id"] = group_id or ""
        meta["persona_id"] = persona_id or ""

    return profile_text


async def _rewrite_query_for_retrieval(
    user_message: str, component_manager: "ComponentManager"
) -> Optional[str]:
    """查询改写：从用户消息中提取检索意图

    使用 LLM 将用户原始消息改写为更适合向量检索的查询文本。
    例如："你还记得我喜欢什么吗？" → "用户偏好 喜好"

    Args:
        user_message: 用户原始消息
        component_manager: 组件管理器实例

    Returns:
        改写后的查询文本，失败时返回 None（使用原始消息）
    """
    from iris_memory.config import get_config
    import asyncio

    config = get_config()

    if not config.get("l2_query_rewrite_enable", True):
        return None

    llm_manager = component_manager.get_component("llm_manager")
    if not llm_manager or not llm_manager.is_available:
        return None

    prompt = (
        "从用户消息中提取用于记忆检索的关键词，空格分隔，不要解释。\n"
        "提取核心实体、事件和偏好，去除口语语气词。\n"
        f"用户消息：{user_message}\n\n搜索关键词："
    )

    timeout_ms = config.get("l2_query_rewrite_timeout_ms", 3000)

    try:
        rewritten = await asyncio.wait_for(
            llm_manager.generate_direct(prompt=prompt, module="l2_query_rewrite"),
            timeout=timeout_ms / 1000.0,
        )

        rewritten = rewritten.strip()

        if not rewritten or rewritten == "无":
            logger.debug("查询改写结果为空，使用原始消息")
            return None

        logger.debug(f"查询改写：'{user_message[:30]}...' -> '{rewritten}'")
        return rewritten

    except asyncio.TimeoutError:
        logger.debug(f"查询改写超时（{timeout_ms}ms），使用原始消息")
        return None
    except Exception as e:
        logger.debug(f"查询改写失败：{e}，使用原始消息")
        return None


async def _collect_l2_memory(
    event: "AstrMessageEvent",
    component_manager: "ComponentManager",
    meta: Optional[dict] = None,
) -> tuple[str, List["MemorySearchResult"]]:
    """收集 L2 记忆文本（不直接修改 req）

    执行 L2 向量检索并返回格式化文本和检索结果。

    Args:
        event: AstrBot 消息事件对象
        component_manager: 组件管理器实例
        meta: 可选的运行日志元信息字典，函数会填充检索与预算裁剪统计

    Returns:
        (格式化的记忆文本, L2 检索结果列表)
    """
    from iris_memory.config import get_config
    from iris_memory.platform import get_adapter

    config = get_config()

    if not config.get("l2_memory.enable"):
        logger.debug("L2 记忆库未启用，跳过记忆注入")
        if meta is not None:
            meta["skipped"] = "disabled"
        return "", []

    l2_status = component_manager.check_component("l2_memory")
    if l2_status == "disabled":
        logger.debug("L2 记忆库未启用，跳过记忆注入")
        if meta is not None:
            meta["skipped"] = "disabled"
        return "", []
    if l2_status == "initializing":
        logger.debug("L2 记忆库正在初始化中，跳过记忆注入")
        if meta is not None:
            meta["skipped"] = "initializing"
        return "", []
    if l2_status != "available":
        logger.debug("L2 记忆库组件不可用，跳过记忆注入")
        if meta is not None:
            meta["skipped"] = "component_unavailable"
        return "", []

    adapter = get_adapter(event)
    group_id = adapter.get_group_id(event)

    from iris_memory.core.persona import resolve_persona

    persona_id = await resolve_persona(component_manager, event)

    query_text = ""
    if hasattr(event, "message_str") and event.message_str:
        query_text = event.message_str
    elif hasattr(event, "get_message_str"):
        query_text = event.get_message_str()

    if not query_text:
        logger.debug("无法获取用户消息，跳过记忆检索")
        if meta is not None:
            meta["skipped"] = "no_query"
        return "", []

    try:
        rewritten_query = await _rewrite_query_for_retrieval(
            query_text, component_manager
        )
        search_query = rewritten_query if rewritten_query else query_text

        from iris_memory.l2_memory import MemoryRetriever

        llm_manager = component_manager.get_component("llm_manager")
        retriever = MemoryRetriever(component_manager, llm_manager)

        l2_results = await retriever.retrieve(
            query=search_query,
            group_id=group_id,
            persona_id=persona_id,
        )

        context_text = ""
        injected_count = 0
        budget_tokens = 0
        if l2_results:
            max_tokens = config.get("token_budget_max_tokens", 2000)
            budget_tokens = max_tokens
            trimmed = MemoryRetriever.trim_by_token_budget(l2_results, max_tokens)
            injected_count = len(trimmed)
            context_lines = ["## 相关记忆"]
            for i, result in enumerate(trimmed, 1):
                context_lines.append(f"{i}. {result.entry.content}")
            context_text = "\n".join(context_lines)

        if context_text:
            logger.debug(f"已收集检索记忆到群聊 {group_id}")

        if meta is not None:
            meta["query"] = query_text
            meta["rewritten_query"] = rewritten_query or ""
            meta["result_count"] = len(l2_results)
            meta["injected_count"] = injected_count
            meta["dropped_by_budget"] = len(l2_results) - injected_count
            meta["budget_tokens"] = budget_tokens

        return context_text, l2_results

    except Exception as e:
        logger.error(f"L2 记忆注入失败: {e}", exc_info=True)
        if meta is not None:
            meta["error"] = str(e)
        return "", []


async def _collect_l3_knowledge_graph(
    event: "AstrMessageEvent",
    component_manager: "ComponentManager",
    l2_results: List["MemorySearchResult"],
    user_message: str = "",
    meta: Optional[dict] = None,
) -> str:
    """收集 L3 知识图谱文本（不直接修改 req）

    检索策略：
    1. 优先基于 L2 记忆关联的节点 ID 进行路径扩展
    2. 若 L2 结果无节点 ID，则基于用户消息关键词搜索图谱
    3. 两种策略的结果合并去重

    Args:
        event: AstrBot 消息事件对象
        component_manager: 组件管理器实例
        l2_results: L2 检索结果
        user_message: 用户当前消息文本
        meta: 可选的运行日志元信息字典，函数会填充检索策略与裁剪统计

    Returns:
        格式化的图谱文本，不可用时返回空字符串
    """
    from iris_memory.config import get_config
    from iris_memory.platform import get_adapter

    config = get_config()

    if not config.get("l3_kg.enable"):
        logger.debug("L3 知识图谱未启用，跳过图谱注入")
        if meta is not None:
            meta["skipped"] = "disabled"
        return ""

    l3_status = component_manager.check_component("l3_kg")
    if l3_status == "disabled":
        logger.debug("L3 知识图谱未启用，跳过图谱注入")
        if meta is not None:
            meta["skipped"] = "disabled"
        return ""
    if l3_status == "initializing":
        logger.debug("L3 知识图谱正在初始化中，跳过图谱注入")
        if meta is not None:
            meta["skipped"] = "initializing"
        return ""
    if l3_status != "available":
        logger.debug("L3 知识图谱组件不可用，跳过图谱注入")
        if meta is not None:
            meta["skipped"] = "component_unavailable"
        return ""

    kg_adapter = component_manager.get_available_component("l3_kg")

    adapter = get_adapter(event)
    group_id = adapter.get_group_id(event)

    # 与 L2 retriever 对齐：群记忆隔离关闭时不按 group_id 过滤，跨群共享图谱
    if not config.get("isolation_config.enable_group_memory_isolation"):
        group_id = None

    try:
        from iris_memory.l3_kg import GraphRetriever

        retriever = GraphRetriever(kg_adapter)

        all_nodes: dict[str, dict] = {}
        all_edges: dict[str, dict] = {}
        strategy = "none"
        keywords_used: List[str] = []

        memory_node_ids: List[str] = []
        if l2_results:
            for result in l2_results:
                metadata = result.entry.metadata
                node_id = (
                    metadata.get("memory_node_id")
                    or metadata.get("kg_node_id")
                    or metadata.get("node_id")
                    or metadata.get("entity_id")
                )
                if node_id:
                    memory_node_ids.append(node_id)

            if not memory_node_ids:
                l2_memory_ids = [r.entry.id for r in l2_results]
                if l2_memory_ids:
                    try:
                        memory_node_ids = (
                            await kg_adapter.get_node_ids_by_source_memory_ids(
                                l2_memory_ids
                            )
                        )
                        if memory_node_ids:
                            logger.debug(
                                f"通过来源记忆反向查找找到 {len(memory_node_ids)} 个图谱节点"
                            )
                    except Exception as e:
                        logger.debug(f"来源记忆反向查找失败: {e}")

        if memory_node_ids:
            nodes, edges = await retriever.retrieve_with_expansion(
                memory_node_ids=memory_node_ids, group_id=group_id
            )
            for n in nodes:
                nid = n.get("id")
                if nid:
                    all_nodes[nid] = n
            for e in edges:
                eid = f"{e.get('source', '')}-{e.get('relation_type', '')}-{e.get('target', '')}"
                all_edges[eid] = e
            strategy = "l2_expansion"
            logger.debug(f"基于 L2 节点扩展：{len(nodes)} 节点，{len(edges)} 边")

        if not memory_node_ids and user_message:
            keywords = _extract_kg_keywords(user_message)
            if keywords:
                keywords_used = keywords
                nodes, edges = await retriever.retrieve_by_keywords(
                    keywords=keywords, group_id=group_id
                )
                for n in nodes:
                    nid = n.get("id")
                    if nid:
                        all_nodes[nid] = n
                for e in edges:
                    eid = f"{e.get('source', '')}-{e.get('relation_type', '')}-{e.get('target', '')}"
                    all_edges[eid] = e
                strategy = "keywords"
                logger.debug(f"基于关键词检索：{len(nodes)} 节点，{len(edges)} 边")

        if not all_nodes:
            if meta is not None:
                meta["strategy"] = strategy
                meta["keywords"] = keywords_used
                meta["skipped"] = "no_nodes"
            return ""

        l3_max_tokens = cast(int, config.get("l3_max_inject_tokens", 600))

        graph_text, included_node_ids = retriever.format_for_context(
            list(all_nodes.values()),
            list(all_edges.values()),
            max_tokens=l3_max_tokens,
        )

        if graph_text:
            # 仅对实际注入文本的节点更新访问计数，避免被 token 预算
            # 裁剪掉的节点 access_count 被错误抬升，污染淘汰评分
            await retriever.update_access_count(list(included_node_ids))

            logger.debug(f"图谱检索完成（{len(all_nodes)} 节点，{len(all_edges)} 边）")

        if meta is not None:
            meta["strategy"] = strategy
            meta["keywords"] = keywords_used
            meta["node_count"] = len(all_nodes)
            meta["edge_count"] = len(all_edges)
            meta["included_nodes"] = len(included_node_ids)
            meta["budget_tokens"] = l3_max_tokens

        return graph_text

    except Exception as e:
        logger.error(f"L3 知识图谱注入失败: {e}", exc_info=True)
        if meta is not None:
            meta["error"] = str(e)
        return ""


def _extract_kg_keywords(text: str) -> List[str]:
    """从用户消息中提取知识图谱检索关键词

    Args:
        text: 用户消息文本

    Returns:
        关键词列表
    """
    if not text:
        return []

    keywords: List[str] = []

    quoted = _QUOTED_PATTERN.findall(text)
    keywords.extend(quoted)

    chinese_words = _CHINESE_WORD_PATTERN.findall(text)
    filtered = [w for w in chinese_words if w not in _KG_STOPWORDS]
    keywords.extend(filtered)

    seen = set()
    unique: List[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)

    return unique[:8]


def _format_profiles_for_injection(
    group_profile: "GroupProfile", user_profile: "UserProfile"
) -> str:
    """格式化画像为注入文本

    将用户画像和群聊画像分开显示，节约 token。

    Args:
        group_profile: 群聊画像对象
        user_profile: 用户画像对象

    Returns:
        格式化的画像文本，任一部分为空则不注入该部分
    """
    from iris_memory.config import get_config
    from iris_memory.profile.models import favorability_level

    config = get_config()
    favorability_enabled = config.get("profile.favorability_enable", True)

    sections = []

    user_parts = [f"【发送者】ID: {user_profile.user_id}"]
    if user_profile.user_name:
        user_parts.append(f"昵称: {user_profile.user_name}")
    if user_profile.historical_names:
        user_parts.append(f"曾用昵称: {', '.join(user_profile.historical_names)}")
    if user_profile.personality_tags:
        user_parts.append(f"性格: {', '.join(user_profile.personality_tags)}")
    if user_profile.interests:
        user_parts.append(f"兴趣: {', '.join(user_profile.interests)}")
    if user_profile.occupation:
        user_parts.append(f"职业: {user_profile.occupation}")
    if user_profile.language_style:
        user_parts.append(f"语言风格: {user_profile.language_style}")
    if user_profile.communication_style:
        user_parts.append(f"沟通偏好: {user_profile.communication_style}")
    if user_profile.emotional_baseline:
        user_parts.append(f"情感: {user_profile.emotional_baseline}")
    if favorability_enabled and user_profile.favorability > 0:
        level = favorability_level(user_profile.favorability)
        user_parts.append(f"好感度: {int(user_profile.favorability)}({level})")
    if user_profile.bot_relationship:
        user_parts.append(f"称呼: {user_profile.bot_relationship}")
    if user_profile.important_dates:
        dates_str = ", ".join(
            f"{d['date']}({d['description']})" for d in user_profile.important_dates
        )
        user_parts.append(f"重要日期: {dates_str}")
    if user_profile.important_events:
        user_parts.append(f"重要事件: {', '.join(user_profile.important_events)}")
    if user_profile.taboo_topics:
        user_parts.append(f"禁忌: {', '.join(user_profile.taboo_topics)}")
    if user_profile.custom_fields:
        custom_str = ", ".join(
            f"{k}: {v}" for k, v in user_profile.custom_fields.items()
        )
        user_parts.append(custom_str)

    if len(user_parts) > 1:
        sections.append("\n".join(user_parts))

    group_parts = ["【群聊】"]
    if group_profile.interests:
        group_parts.append(f"兴趣: {', '.join(group_profile.interests)}")
    if group_profile.atmosphere_tags:
        group_parts.append(f"氛围: {', '.join(group_profile.atmosphere_tags)}")
    if group_profile.long_term_tags:
        group_parts.append(f"核心特征: {', '.join(group_profile.long_term_tags)}")
    if group_profile.blacklist_topics:
        group_parts.append(f"禁忌: {', '.join(group_profile.blacklist_topics)}")
    if group_profile.custom_fields:
        custom_str = ", ".join(
            f"{k}: {v}" for k, v in group_profile.custom_fields.items()
        )
        group_parts.append(custom_str)

    if len(group_parts) > 1:
        sections.append("\n".join(group_parts))

    return "\n\n".join(sections) if sections else ""


def _is_passive_trigger(event: "AstrMessageEvent") -> bool:
    """检查当前 LLM 请求是否为被动触发（sampling/主动回复）

    通过检查 event extras 中的 iris_passive_trigger 标志判断。
    该标志由 main.py 中的 _detect_passive_trigger() 设置，
    其依据是 event.is_at_or_wake_command 是否为 False。

    被动触发时，用户未通过 @机器人 或唤醒前缀主动请求，LLM 请求由 AstrBot 的
    active_reply/sampling 机制触发。此时图片解析是不必要的，
    跳过可节省 token 和配额。

    Args:
        event: AstrBot 消息事件对象

    Returns:
        True 表示被动触发，应跳过图片解析
    """
    try:
        if hasattr(event, "get_extra"):
            return bool(event.get_extra("iris_passive_trigger"))
    except Exception:
        pass
    return False


async def _parse_images_if_related_mode(
    event: "AstrMessageEvent",
    req: "ProviderRequest",
    component_manager: "ComponentManager",
) -> None:
    """解析图片并替换 L1 消息中的占位符（related 模式）

    仅在 related 模式下解析 L1 Buffer 范围内的图片。
    all 模式已在消息钩子中处理。

    流程：
    1. 获取 L1Buffer 图片队列中的待解析图片
    2. 检查缓存，命中则直接替换占位符
    3. 批量解析（并发控制、数量限制）
    4. 结果存入缓存
    5. 替换 L1 消息中的 [IMG:xxx] 占位符为 [图:描述]

    Args:
        event: AstrBot 消息事件对象
        req: LLM 提供者请求对象
        component_manager: 组件管理器实例
    """
    from iris_memory.config import get_config
    from iris_memory.platform import get_adapter
    from iris_memory.image import ImageParser, ImageParseStatus, ImageParseCache
    import asyncio

    config = get_config()
    if not config.get("l1_buffer.image_parsing.enable"):
        return

    if config.get("image_skip_on_passive_trigger", True):
        if _is_passive_trigger(event):
            logger.info("被动触发（sampling/主动回复），跳过图片解析以节省 token")
            req._iris_image_meta = {"skipped": "passive_trigger"}
            return

    mode = config.get("l1_buffer.image_parsing.mode", "related")

    if mode == "all":
        return

    if mode != "related":
        logger.warning(f"未知的图片解析模式：{mode}")
        return

    adapter = get_adapter(event)
    session_id = adapter.get_session_id(event)

    buffer = component_manager.get_available_component("l1_buffer")
    if not buffer:
        return

    l1_buffer = cast("L1Buffer", buffer)

    cache_manager = component_manager.get_available_component("image_cache")
    quota_manager = component_manager.get_available_component("image_quota")
    llm_manager = component_manager.get_available_component("llm_manager")

    if not llm_manager:
        logger.warning("LLM Manager 不可用，跳过图片解析")
        return

    max_parse = config.get("image_max_parse_per_request")
    max_concurrent = config.get("image_max_concurrent_parse")

    pending_images = l1_buffer.get_images(session_id, limit=max_parse, only_pending=True)

    if not pending_images:
        return

    images_to_parse = []
    cached_results = []

    for img_item in pending_images:
        if cache_manager and cache_manager.is_available:
            cached = await cache_manager.get_cache(img_item.image_hash)
            if cached:
                l1_buffer.mark_image_parsed(
                    session_id, img_item.image_hash, ImageParseStatus.SUCCESS
                )
                placeholder = f"[IMG:{img_item.image_hash.removeprefix('ph:')[:12]}]"
                l1_buffer.replace_image_placeholder(
                    session_id, placeholder, f"[图:{cached.content}]"
                )
                cached_results.append((img_item, cached))
                continue

        images_to_parse.append(img_item)

    if cached_results:
        logger.debug(f"从缓存读取 {len(cached_results)} 条图片解析结果并替换占位符")

    if not images_to_parse:
        total_cached = len(cached_results)
        if total_cached > 0:
            logger.info(f"已从缓存获取 {total_cached} 条图片解析结果并替换占位符")
        return

    if quota_manager and quota_manager.is_available:
        has_quota = await quota_manager.check_quota()
        if not has_quota:
            logger.info("图片解析配额已耗尽，跳过解析")
            req._iris_image_meta = {"skipped": "quota_exhausted"}
            return

        quota_used = await quota_manager.use_quota(len(images_to_parse))
        if not quota_used:
            logger.warning("图片解析配额使用失败")
            return

    provider = config.get("l1_buffer.image_parsing.provider", "")

    from iris_memory.image.recorder_bridge import get_recorder_bridge

    parser = ImageParser(llm_manager, provider, recorder_bridge=get_recorder_bridge())

    logger.info(f"开始解析 {len(images_to_parse)} 张图片（related 模式）")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def parse_with_semaphore(img_item):
        async with semaphore:
            if not img_item.image_info or not img_item.image_info.has_url:
                return (img_item, None)
            result = await parser.parse(img_item.image_info)
            return (img_item, result)

    parse_timeout_ms = cast(int, config.get("image_parse_timeout_ms", 30000))

    task_to_img: dict = {}
    for img in images_to_parse:
        task_to_img[asyncio.ensure_future(parse_with_semaphore(img))] = img

    if parse_timeout_ms and parse_timeout_ms > 0:
        done, pending = await asyncio.wait(
            task_to_img, timeout=parse_timeout_ms / 1000.0
        )
    else:
        done, pending = await asyncio.wait(task_to_img)

    if pending:
        for t in pending:
            t.cancel()
            img_item = task_to_img[t]
            l1_buffer.mark_image_parsed(
                session_id, img_item.image_hash, ImageParseStatus.FAILED
            )
            placeholder = f"[IMG:{img_item.image_hash.removeprefix('ph:')[:12]}]"
            l1_buffer.replace_image_placeholder(session_id, placeholder, "")
        logger.warning(
            f"图片解析整体超时（{parse_timeout_ms}ms），"
            f"{len(pending)}/{len(images_to_parse)} 张未完成，已标记失败"
        )
        # 等待被取消的任务真正结束，避免任务泄漏与 "Task was destroyed but
        # it is pending" 警告，并确保 parse_with_semaphore 内的 httpx 连接等资源被回收
        await asyncio.gather(*pending, return_exceptions=True)

    parse_results = [t.result() for t in done]

    success_count = 0
    for img_item, result in parse_results:
        if result is None:
            l1_buffer.mark_image_parsed(
                session_id, img_item.image_hash, ImageParseStatus.FAILED
            )
            placeholder = f"[IMG:{img_item.image_hash.removeprefix('ph:')[:12]}]"
            l1_buffer.replace_image_placeholder(session_id, placeholder, "")
            continue

        if not result.success:
            logger.warning(f"图片解析失败：{result.error_message}")
            l1_buffer.mark_image_parsed(
                session_id, img_item.image_hash, ImageParseStatus.FAILED
            )
            placeholder = f"[IMG:{img_item.image_hash.removeprefix('ph:')[:12]}]"
            l1_buffer.replace_image_placeholder(session_id, placeholder, "")
            continue

        if not result.content:
            logger.debug("图片解析结果为空")
            l1_buffer.mark_image_parsed(
                session_id, img_item.image_hash, ImageParseStatus.FAILED
            )
            placeholder = f"[IMG:{img_item.image_hash.removeprefix('ph:')[:12]}]"
            l1_buffer.replace_image_placeholder(session_id, placeholder, "")
            continue

        if cache_manager and cache_manager.is_available:
            cache = ImageParseCache(
                image_hash=img_item.image_hash,
                content=result.content,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            await cache_manager.set_cache(cache)

        l1_buffer.mark_image_parsed(
            session_id, img_item.image_hash, ImageParseStatus.SUCCESS
        )

        placeholder = f"[IMG:{img_item.image_hash.removeprefix('ph:')[:12]}]"
        l1_buffer.replace_image_placeholder(
            session_id, placeholder, f"[图:{result.content}]"
        )

        success_count += 1

    # 退还解析失败/跳过/超时的预扣配额，避免静默耗尽
    if quota_manager and quota_manager.is_available:
        failed_count = len(images_to_parse) - success_count
        if failed_count > 0:
            await quota_manager.release_quota(failed_count)

    total_replaced = success_count + len(cached_results)
    req._iris_image_meta = {
        "mode": "related",
        "parsed": success_count,
        "cached": len(cached_results),
        "failed": len(images_to_parse) - success_count,
    }
    if total_replaced > 0:
        logger.info(
            f"已解析 {success_count} 张新图片，缓存 {len(cached_results)} 张，"
            f"共 {total_replaced} 条图片解析结果已替换占位符"
        )


def _log_final_context(req: "ProviderRequest") -> None:
    """输出最终上下文内容的 debug 日志

    在所有注入完成后，输出完整的上下文信息用于问题排查。

    Args:
        req: LLM 提供者请求对象
    """
    from iris_memory.config import get_config

    config = get_config()
    if not config.get("enable_context_logging", False):
        return

    log_parts = ["\n" + "=" * 60 + "\n[LLM 请求上下文详情]\n" + "=" * 60]

    if req.system_prompt:
        log_parts.append(
            f"\n[System Prompt]\n{'-' * 40}\n{req.system_prompt}\n{'-' * 40}"
        )
    else:
        log_parts.append("\n[System Prompt]\n(无)")

    if req.extra_user_content_parts:
        log_parts.append(
            f"\n[Extra User Content Parts] (共 {len(req.extra_user_content_parts)} 个)"
        )
        import re as _re

        _iris_tag_pattern = _re.compile(r"<iris:(\w+)>\n(.*?)\n</iris:\1>", _re.DOTALL)
        _section_truncation = 300

        for i, part in enumerate(req.extra_user_content_parts, 1):
            text = getattr(part, "text", None) or str(part)
            tag_sections = _iris_tag_pattern.findall(text)
            if tag_sections:
                log_parts.append(f"  [{i}] ({len(text)} 字符)")
                for sec_name, sec_content in tag_sections:
                    display = sec_content
                    if len(display) > _section_truncation:
                        display = (
                            display[: _section_truncation // 2]
                            + f"\n  ... 省略 {len(display) - _section_truncation} 字 ...\n"
                            + display[-_section_truncation // 2 :]
                        )
                    log_parts.append(f"    <iris:{sec_name}> ({len(sec_content)} 字)")
                    for line in display.split("\n"):
                        log_parts.append(f"      {line}")
            else:
                if len(text) > 500:
                    text = text[:500] + "..."
                log_parts.append(f"  [{i}] {text}")
    else:
        log_parts.append("\n[Extra User Content Parts]\n(无)")

    if req.contexts:
        log_parts.append(f"\n[Contexts] (共 {len(req.contexts)} 条)")
        for i, ctx in enumerate(req.contexts, 1):
            role = ctx.get("role", "unknown")
            content = ctx.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            log_parts.append(f"  [{i}] {role}: {content}")
    else:
        log_parts.append("\n[Contexts]\n(无)")

    if hasattr(req, "functions") and req.functions:
        log_parts.append(f"\n[Functions] (共 {len(req.functions)} 个)")
        for i, func in enumerate(req.functions, 1):
            name = (
                func.get("name", "unknown")
                if isinstance(func, dict)
                else getattr(func, "name", "unknown")
            )
            log_parts.append(f"  [{i}] {name}")

    log_parts.append("\n" + "=" * 60)

    logger.debug("\n".join(log_parts))

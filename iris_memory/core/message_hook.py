"""
消息钩子处理模块

负责处理用户发送的消息钩子，包括：
- 添加用户消息到 L1 Buffer
- 用户画像更新
- 图片解析（all 模式）
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

from iris_memory.core import get_logger

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from iris_memory.core.components import ComponentManager
    from iris_memory.l1_buffer import L1Buffer

logger = get_logger("message_hook")

_name_cache: OrderedDict = OrderedDict()
_NAME_CACHE_MAX_SIZE = 1000


def _get_cached_name(key: str) -> str | None:
    if key in _name_cache:
        _name_cache.move_to_end(key)
        return _name_cache[key]
    return None


def _set_cached_name(key: str, name: str) -> None:
    if key in _name_cache:
        _name_cache.move_to_end(key)
        _name_cache[key] = name
    else:
        if len(_name_cache) >= _NAME_CACHE_MAX_SIZE:
            _name_cache.popitem(last=False)
        _name_cache[key] = name


async def handle_user_message(
    event: "AstrMessageEvent", component_manager: "ComponentManager"
) -> None:
    """处理用户消息钩子

    执行所有用户消息的处理逻辑（按顺序执行）：
    1. 添加用户消息到 L1 Buffer
    2. 用户画像更新
    3. 图片入队到 L1 Buffer 图片队列
    4. 图片解析（all 模式）

    Args:
        event: AstrBot 消息事件对象
        component_manager: 组件管理器实例
    """
    await _add_to_l1_buffer(event, component_manager)
    await _queue_images_to_l1_buffer(event, component_manager)
    await _parse_images_if_enabled(event, component_manager)


def _backfill_reply_from_buffer(
    l1_buffer: "L1Buffer",
    session_id: str,
    reply_message_id: str,
    metadata: dict,
) -> None:
    """从 L1 Buffer 中回填被回复消息的内容

    当平台未提供 reply_content 时，尝试从 L1 Buffer 中
    通过 message_id 查找被回复的消息，回填其内容和发送者名称。

    Args:
        l1_buffer: L1 Buffer 实例
        session_id: 会话 ID（L1 队列键，私聊为 private:{user_id}）
        reply_message_id: 被回复消息的ID
        metadata: 待回填的 metadata 字典
    """
    try:
        messages = l1_buffer.get_context(session_id)
        for msg in reversed(messages):
            msg_mid = msg.metadata.get("message_id") if msg.metadata else None
            if msg_mid and str(msg_mid) == str(reply_message_id):
                if msg.content and "reply_content" not in metadata:
                    metadata["reply_content"] = msg.content
                if "reply_user_name" not in metadata and msg.metadata:
                    msg_name = msg.metadata.get("user_name")
                    if msg_name:
                        metadata["reply_user_name"] = msg_name
                logger.debug(
                    f"从 L1 Buffer 回填回复信息：message_id={reply_message_id}"
                )
                return
    except Exception as e:
        logger.debug(f"回填回复信息失败: {e}")


async def _update_profile_names(
    component_manager: "ComponentManager",
    group_id: str,
    group_name: str,
    user_id: str,
    user_name: str,
    persona_id: str = "default",
) -> None:
    """更新用户昵称和群聊名称（内部函数）

    当用户昵称或群聊名称发生变化时，更新画像中的名称字段。
    用户昵称变化会记录到 historical_names。
    使用内存缓存避免重复数据库操作。

    Args:
        component_manager: 组件管理器实例
        group_id: 群聊ID
        group_name: 群聊名称
        user_id: 用户ID
        user_name: 用户昵称
        persona_id: 人格ID
    """
    from iris_memory.config import get_config

    config = get_config()
    if not config.get("profile.enable"):
        return

    profile_storage = component_manager.get_available_component("profile")
    if not profile_storage:
        return

    group_key = f"group:{persona_id}:{group_id}"
    effective_group_id = (
        group_id if config.get("isolation_config.enable_group_isolation") else "default"
    )
    user_key = f"user:{persona_id}:{effective_group_id}:{user_id}"

    group_name_changed = group_name and _get_cached_name(group_key) != group_name
    user_name_changed = user_name and _get_cached_name(user_key) != user_name

    if not group_name_changed and not user_name_changed:
        return

    try:
        from iris_memory.profile import GroupProfileManager, UserProfileManager

        group_manager = GroupProfileManager(profile_storage)
        user_manager = UserProfileManager(profile_storage)

        if group_name_changed:
            await group_manager.update_group_name(group_id, group_name, persona_id)
            _set_cached_name(group_key, group_name)

        if user_name_changed:
            await user_manager.update_user_name(
                user_id, effective_group_id, user_name, persona_id
            )
            _set_cached_name(user_key, user_name)

    except Exception as e:
        logger.error(f"更新画像名称失败: {e}", exc_info=True)


async def _add_to_l1_buffer(
    event: "AstrMessageEvent", component_manager: "ComponentManager"
) -> None:
    """添加用户消息到 L1 Buffer（内部函数）

    Args:
        event: AstrBot 消息事件对象
        component_manager: 组件管理器实例
    """
    from iris_memory.platform import get_adapter

    content = event.message_str
    if not content:
        logger.debug("消息内容为空，跳过添加")
        return

    from iris_memory.utils import sanitize_input

    content = sanitize_input(content, source="user_message")

    buffer = component_manager.get_available_component("l1_buffer")
    if not buffer:
        logger.debug("L1 Buffer 组件不可用，跳过消息添加")
        return

    l1_buffer = cast("L1Buffer", buffer)

    adapter = get_adapter(event)
    group_id = adapter.get_group_id(event)
    # L1 队列键使用会话 ID：私聊为 private:{user_id}，避免不同私聊用户
    # 共用空字符串队列导致上下文互相污染；画像等仍使用原始群 ID
    session_id = adapter.get_session_id(event)
    user_id = adapter.get_user_id(event)
    user_name = adapter.get_user_name(event)
    group_name = adapter.get_group_name(event)

    # 解析 persona_id（用于画像与 L1 消息的隔离命名空间）
    from iris_memory.core.persona import resolve_persona

    persona_id = await resolve_persona(component_manager, event)

    metadata = {}
    if user_name:
        metadata["user_name"] = user_name

    raw_msg = adapter.get_raw_message(event)
    message_id = str(raw_msg.get("message_id", "")) if raw_msg else ""
    if message_id:
        metadata["message_id"] = message_id

    reply_info = adapter.get_reply_info(event)
    if reply_info.has_reply:
        metadata["reply_message_id"] = reply_info.message_id
        if reply_info.user_id:
            metadata["reply_user_id"] = reply_info.user_id
        if reply_info.user_name:
            metadata["reply_user_name"] = reply_info.user_name
        if reply_info.content:
            metadata["reply_content"] = reply_info.content
        elif reply_info.message_id:
            _backfill_reply_from_buffer(
                l1_buffer, session_id, reply_info.message_id, metadata
            )
            if "reply_content" not in metadata:
                api_reply = await adapter.get_msg_by_id(event, reply_info.message_id)
                if api_reply.has_reply:
                    if api_reply.content:
                        metadata["reply_content"] = api_reply.content
                    if not metadata.get("reply_user_name") and api_reply.user_name:
                        metadata["reply_user_name"] = api_reply.user_name
                    if not metadata.get("reply_user_id") and api_reply.user_id:
                        metadata["reply_user_id"] = api_reply.user_id
                    logger.debug(
                        f"从平台 API 回填回复信息：message_id={reply_info.message_id}"
                    )

    await l1_buffer.add_message(
        group_id=session_id,
        role="user",
        content=content,
        source=user_id,
        metadata=metadata,
        persona_id=persona_id,
    )

    logger.debug(f"已添加用户消息到会话 {session_id} 的 L1 Buffer")

    # 展开合并转发消息：拼接为一条消息入队，超长按 token 限制截断
    try:
        forward_messages = await adapter.get_forward_messages(event)
    except Exception as e:
        logger.debug(f"获取合并转发消息失败: {e}")
        forward_messages = []

    if forward_messages:
        from iris_memory.config import get_config as _get_cfg
        from iris_memory.utils import count_tokens

        max_single_tokens = cast(
            int, _get_cfg().get("l1_max_single_message_tokens", 500)
        )

        # 预留 token 预算给前后缀与单条子消息结构开销
        reserve_tokens = 30
        budget = max(max_single_tokens - reserve_tokens, 64)

        # 按子消息累加，超过预算则停止；保证每条子消息完整
        included_lines: list[str] = []
        included_count = 0
        used_tokens = 0
        truncated = False

        for fwd in forward_messages:
            if not fwd.content:
                continue
            fwd_content = sanitize_input(fwd.content, source="user_message")
            if not fwd_content:
                continue
            user_label = fwd.user_name or "用户"
            line = f"[{user_label}]: {fwd_content}"
            line_tokens = count_tokens(line)

            if used_tokens + line_tokens > budget and included_lines:
                truncated = True
                break
            included_lines.append(line)
            included_count += 1
            used_tokens += line_tokens

        if included_lines:
            header = "【合并转发内容】"
            parts = [header] + included_lines
            if truncated:
                total = sum(1 for f in forward_messages if f.content)
                parts.append(f"（已截断，共 {total} 条，已展示 {included_count} 条）")
            combined_content = "\n".join(parts)

            fwd_metadata: dict[str, Any] = {
                "forward": True,
                "forward_total": len(forward_messages),
                "forward_included": included_count,
                "forward_truncated": truncated,
            }
            await l1_buffer.add_message(
                group_id=session_id,
                role="user",
                content=combined_content,
                source=user_id,
                metadata=fwd_metadata,
                persona_id=persona_id,
            )
            logger.debug(
                f"合并转发消息已入队：会话 {session_id}，"
                f"包含 {included_count}/{len(forward_messages)} 条子消息，"
                f"truncated={truncated}"
            )

    await _update_profile_names(
        component_manager, group_id, group_name, user_id, user_name, persona_id
    )


async def update_l1_buffer(
    event: "AstrMessageEvent",
    component_manager: "ComponentManager",
    role: str,
    content: str,
) -> None:
    """更新 L1 Buffer（添加用户消息或助手响应）

    此函数用于特殊场景（如添加助手响应），
    普通用户消息应使用 handle_user_message() 处理。

    Args:
        event: AstrBot 消息事件对象
        component_manager: 组件管理器实例
        role: 消息角色（"user" 或 "assistant"）
        content: 消息内容
    """
    from iris_memory.platform import get_adapter

    buffer = component_manager.get_available_component("l1_buffer")
    if not buffer:
        logger.debug("L1 Buffer 组件不可用，跳过消息更新")
        return

    l1_buffer = cast("L1Buffer", buffer)

    adapter = get_adapter(event)
    session_id = adapter.get_session_id(event)
    user_id = adapter.get_user_id(event)

    from iris_memory.core.persona import resolve_persona

    persona_id = await resolve_persona(component_manager, event)

    await l1_buffer.add_message(
        group_id=session_id,
        role=role,
        content=content,
        source=user_id if role == "user" else "assistant",
        persona_id=persona_id,
    )

    logger.debug(f"已添加 {role} 消息到会话 {session_id} 的 L1 Buffer")


async def _queue_images_to_l1_buffer(
    event: "AstrMessageEvent", component_manager: "ComponentManager"
) -> None:
    """提取图片并入队到 L1 Buffer 图片队列（内部函数）

    支持 pHash 感知哈希去重和无效图过滤。
    入队时检查缓存：
    - 已缓存：直接将图片描述前置到 L1 消息内容，图片标记为 SUCCESS
    - 未缓存：前置占位符 [IMG:{hash_prefix}] 到 L1 消息内容，图片标记为 PENDING

    Args:
        event: AstrBot 消息事件对象
        component_manager: 组件管理器实例
    """
    from iris_memory.config import get_config
    from iris_memory.platform import get_adapter
    from iris_memory.image import ImageQueueItem, ImageParseStatus
    from iris_memory.image.image_utils import (
        compute_image_hash,
        is_similar_image,
        check_invalid_image,
        detect_image_extension,
    )

    config = get_config()
    if not config.get("l1_buffer.image_parsing.enable"):
        return

    adapter = get_adapter(event)
    images = adapter.get_images(event)

    if not images:
        return

    buffer = component_manager.get_available_component("l1_buffer")
    if not buffer:
        return

    l1_buffer = cast("L1Buffer", buffer)
    session_id = adapter.get_session_id(event)
    user_id = adapter.get_user_id(event)

    # 解析 persona_id：占位消息（prepend 失败时新建）必须携带正确人格归属，
    # 否则该占位常作为最后入队消息，buffer.py 用 messages[-1].persona_id 决定
    # 画像与 L2 摘要归属，default 占位会污染人格命名空间。
    # resolve_persona 在同一 event 上缓存，此处复用 _add_to_l1_buffer 的解析结果。
    from iris_memory.core.persona import resolve_persona

    persona_id = await resolve_persona(component_manager, event)

    raw_msg = adapter.get_raw_message(event)
    message_id = raw_msg.get("message_id", "")

    use_phash = config.get("image_phash_enable")
    phash_threshold = config.get("image_phash_threshold")
    use_filter = config.get("image_filter_enable")
    filter_min_size = config.get("image_filter_min_size", 16)
    filter_std_threshold = config.get("image_filter_std_threshold", 5.0)

    cache_manager = component_manager.get_available_component("image_cache")

    existing_hashes: list[str] = []
    if use_phash:
        existing_hashes = l1_buffer.get_all_phash_hashes()

    image_suffixes: list[str] = []
    queued_count = 0
    for image_info in images:
        # ---- 提前下载图片数据（用于 pHash、过滤、本地缓存） ----
        image_data: bytes | None = None
        if image_info.url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(image_info.url, follow_redirects=True)
                    if resp.status_code < 400 and resp.content:
                        image_data = resp.content
            except Exception as e:
                logger.debug(f"图片下载失败：{e}")

        # ---- 哈希计算（有数据时走 pHash，否则走 URL MD5） ----
        image_hash = await compute_image_hash(
            image_data=image_data,
            url=image_info.url,
            use_phash=use_phash,
        )

        if not image_hash:
            continue

        if use_phash and image_hash.startswith("ph:"):
            is_dup = False
            for existing in existing_hashes:
                if is_similar_image(image_hash, existing, threshold=phash_threshold):
                    is_dup = True
                    logger.debug(f"pHash 去重：跳过相似图片 {image_hash[:16]}...")
                    break
            if is_dup:
                continue
            existing_hashes.append(image_hash)

        # ---- 无效图过滤（复用已下载数据） ----
        if use_filter and image_data:
            is_invalid, reason = await check_invalid_image(
                image_data,
                min_size=filter_min_size,
                std_threshold=filter_std_threshold,
            )
            if is_invalid:
                logger.debug(f"无效图过滤：跳过 {image_hash[:16]}... ({reason})")
                continue

        # ---- 本地缓存（避免 URL 过期后无法访问） ----
        if image_data:
            try:
                raw_hash = image_hash.removeprefix("ph:")
                ext = detect_image_extension(image_data, image_info.url or "")
                cache_dir = config.data_dir / "image_cache" / raw_hash[:2]
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"{raw_hash}{ext}"
                cache_path.write_bytes(image_data)
                image_info.file_path = str(cache_path)
                logger.debug(f"已缓存图片到本地：{cache_path.name}")
            except Exception as e:
                logger.debug(f"图片缓存写入失败：{e}")

        hash_prefix = image_hash.removeprefix("ph:")[:12]

        cached_desc = None
        if cache_manager and cache_manager.is_available:
            cached = await cache_manager.get_cache(image_hash)
            if cached and cached.content:
                cached_desc = cached.content

        if cached_desc:
            image_suffixes.append(f"[图:{cached_desc}]")
            queue_item = ImageQueueItem(
                image_hash=image_hash,
                image_url=image_info.url or "",
                image_info=image_info,
                message_id=message_id,
                group_id=session_id,
                user_id=user_id,
                status=ImageParseStatus.SUCCESS,
            )
        else:
            placeholder = f"[IMG:{hash_prefix}]"
            image_suffixes.append(placeholder)
            queue_item = ImageQueueItem(
                image_hash=image_hash,
                image_url=image_info.url or "",
                image_info=image_info,
                message_id=message_id,
                group_id=session_id,
                user_id=user_id,
                status=ImageParseStatus.PENDING,
            )

        l1_buffer.add_image(session_id, queue_item)
        queued_count += 1

    if image_suffixes:
        prefix = "".join(image_suffixes)
        prepended = l1_buffer.prepend_to_last_message(
            session_id, prefix, same_source=user_id
        )
        if not prepended:
            user_name = adapter.get_user_name(event)
            metadata: dict[str, Any] = {}
            if user_name:
                metadata["user_name"] = user_name
            if message_id:
                metadata["message_id"] = message_id

            await l1_buffer.add_message(
                group_id=session_id,
                role="user",
                content=prefix,
                source=user_id,
                metadata=metadata,
                persona_id=persona_id,
            )

    if queued_count > 0:
        logger.debug(f"已入队 {queued_count} 张图片到 L1 Buffer 图片队列")


async def _parse_images_if_enabled(
    event: "AstrMessageEvent", component_manager: "ComponentManager"
) -> None:
    """解析图片（all 模式）

    仅在 all 模式下解析图片。
    related 模式在 LLM 请求钩子中处理。

    Args:
        event: AstrBot 消息事件对象
        component_manager: 组件管理器实例
    """
    from iris_memory.config import get_config
    from iris_memory.platform import get_adapter
    from iris_memory.image import ImageParser, ImageParseStatus, ImageParseCache

    config = get_config()
    if not config.get("l1_buffer.image_parsing.enable"):
        return

    mode = config.get("l1_buffer.image_parsing.mode", "related")

    if mode == "related":
        return

    if mode != "all":
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
    pending_images = l1_buffer.get_images(session_id, limit=max_parse, only_pending=True)

    if not pending_images:
        return

    images_to_parse = []
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
                continue

        images_to_parse.append(img_item)

    if not images_to_parse:
        return

    if quota_manager and quota_manager.is_available:
        has_quota = await quota_manager.check_quota()
        if not has_quota:
            logger.info("图片解析配额已耗尽，跳过解析")
            return

        quota_used = await quota_manager.use_quota(len(images_to_parse))
        if not quota_used:
            logger.warning("图片解析配额使用失败")
            return

    provider = config.get("l1_buffer.image_parsing.provider", "")

    from iris_memory.image.recorder_bridge import get_recorder_bridge

    parser = ImageParser(llm_manager, provider, recorder_bridge=get_recorder_bridge())

    logger.info(f"开始解析 {len(images_to_parse)} 张图片（all 模式）")

    # 用 (img_item, image_info) 配对，避免过滤后列表与 images_to_parse 下标错位。
    # 历史 bug：image_infos 按 has_url 过滤（更短），parse_batch 按它返回结果，
    # 但回写用 parse_results[i] 索引未过滤的 images_to_parse[i]，存在仅 file_path
    # 无 url 的图片时两列表错位，解析结果归属错误 img_item，缓存写入（以 image_hash
    # 为键）与 mark_image_parsed 全部错位，被过滤项还永久残留 [IMG:...] 占位符。
    parse_pairs = [
        (img, img.image_info)
        for img in images_to_parse
        if img.image_info and img.image_info.has_url
    ]

    # 无 url 的图片无法解析，单独标记 FAILED 并清占位符，不留残留
    no_url_items = [
        img
        for img in images_to_parse
        if not (img.image_info and img.image_info.has_url)
    ]
    for img_item in no_url_items:
        logger.debug(f"图片无 URL，跳过解析：{img_item.image_hash[:8]}")
        l1_buffer.mark_image_parsed(
            session_id, img_item.image_hash, ImageParseStatus.FAILED
        )
        placeholder = f"[IMG:{img_item.image_hash.removeprefix('ph:')[:12]}]"
        l1_buffer.replace_image_placeholder(session_id, placeholder, "")

    if not parse_pairs:
        # 全部无 url，退还预扣配额
        if quota_manager and quota_manager.is_available and no_url_items:
            await quota_manager.release_quota(len(no_url_items))
        return

    image_infos = [info for _img, info in parse_pairs]
    try:
        parse_results = await parser.parse_batch(image_infos)
    except Exception as e:
        logger.error(f"parse_batch 异常，退还全部预扣配额：{e}", exc_info=True)
        if quota_manager and quota_manager.is_available:
            await quota_manager.release_quota(len(images_to_parse))
        # 标记所有可解析图片为 FAILED 并清占位符
        for img_item, _info in parse_pairs:
            l1_buffer.mark_image_parsed(
                session_id, img_item.image_hash, ImageParseStatus.FAILED
            )
            placeholder = f"[IMG:{img_item.image_hash.removeprefix('ph:')[:12]}]"
            l1_buffer.replace_image_placeholder(session_id, placeholder, "")
        return

    success_count = 0
    for (img_item, _info), result in zip(parse_pairs, parse_results):
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

    # 退还解析失败的预扣配额，避免静默耗尽
    if quota_manager and quota_manager.is_available:
        failed_count = (len(parse_pairs) - success_count) + len(no_url_items)
        if failed_count > 0:
            await quota_manager.release_quota(failed_count)

    logger.info(
        f"已解析 {success_count}/{len(images_to_parse)} 张图片"
        f"（无 URL 跳过 {len(no_url_items)} 张）"
    )

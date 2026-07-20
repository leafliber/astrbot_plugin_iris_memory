"""
Iris Chat Memory - L1 消息缓冲组件

提供三段式 FIFO 消息队列管理、自动总结触发等功能。
支持群聊隔离和人格切换时清空所有队列。

三段式设计：
- L1-1（segment_1）：最新段，接收新消息，注入上下文
- L1-2（segment_2）：主体段，注入上下文，总结时的目标段
- L1-3（segment_3）：缓冲段，不注入上下文，总结时辅助理解
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, TYPE_CHECKING, cast
from pathlib import Path
from datetime import datetime
import asyncio
import re

from iris_memory.core import Component, get_logger
from iris_memory.config import get_config
from iris_memory.platform.base import PRIVATE_SESSION_PREFIX
from iris_memory.utils import count_tokens
from .models import ContextMessage, SegmentedMessageQueue
from .summarizer import Summarizer

if TYPE_CHECKING:
    from iris_memory.core.components import ComponentManager
    from iris_memory.profile import GroupProfileManager, UserProfileManager
    from iris_memory.profile.storage import ProfileStorage
    from iris_memory.profile.models import UserProfile

logger = get_logger("buffer")


def _as_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _as_str_list(value: object) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def _as_str_dict(value: object) -> dict[str, str] | None:
    if isinstance(value, dict):
        return value  # type: ignore[return-value]
    return None


def _as_float(value: object) -> float | None:
    """将 LLM 返回的数值字段转为 float（拒绝 bool）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


class L1Buffer(Component):
    """L1 消息缓冲组件

    管理三段式 FIFO 消息队列，支持：
    - 按群聊隔离存储消息
    - L1-1/L1-2/L1-3 三段式 FIFO 队列
    - 自动总结触发（L1-3 满时或 token 超限时）
    - 总结时输入全量上下文，只总结 L1-2
    - 总结后段位转移

    Attributes:
        _queues: 消息队列字典 {group_id: SegmentedMessageQueue}
        _summarizer: 总结器实例（延迟初始化）
        _component_manager: 组件管理器引用（用于获取 LLMManager）
        _provider: 总结使用的 Provider ID
    """

    def __init__(self):
        super().__init__()
        self._queues: Dict[str, SegmentedMessageQueue] = {}
        self._image_queues: Dict[str, List[Any]] = {}
        self._summarizer: Optional[Summarizer] = None
        self._component_manager: Optional["ComponentManager"] = None
        self._provider: str = ""
        self._summarizing_locks: Dict[str, asyncio.Lock] = {}
        self._summary_fail_counts: Dict[str, int] = {}
        self._background_tasks: set[asyncio.Task] = set()
        logger.debug("L1Buffer 实例已创建")

    @property
    def name(self) -> str:
        return "l1_buffer"

    async def initialize(self) -> None:
        try:
            config = get_config()

            if not config.get("l1_buffer.enable"):
                logger.info("L1 缓冲已禁用")
                self._is_available = False
                self._init_error = "L1 缓冲已禁用"
                return

            self._provider = str(config.get("l1_buffer.summary_provider", ""))

            self._is_available = True
            logger.info("L1 缓冲组件初始化成功")

        except Exception as e:
            self._is_available = False
            self._init_error = str(e)
            logger.error(f"L1 缓冲组件初始化失败：{e}", exc_info=True)
            raise

    def set_component_manager(self, manager: "ComponentManager") -> None:
        self._component_manager = manager
        logger.debug("L1Buffer 已获取 ComponentManager 引用")

    async def shutdown(self) -> None:
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        self.clear_all()
        self._summarizing_locks.clear()
        self._summary_fail_counts.clear()
        self._reset_state()
        logger.info("L1 缓冲组件已关闭")

    def _get_or_create_summarizer(self) -> Optional[Summarizer]:
        if self._summarizer is not None:
            return self._summarizer

        if not self._component_manager:
            logger.warning("ComponentManager 未设置，无法创建 Summarizer")
            return None

        from iris_memory.llm import LLMManager

        llm_manager = self._component_manager.get_component("llm_manager")

        if not llm_manager or not llm_manager.is_available:
            logger.warning("LLMManager 不可用，无法创建 Summarizer")
            return None

        assert isinstance(llm_manager, LLMManager)

        self._summarizer = Summarizer(llm_manager=llm_manager, provider=self._provider)
        logger.info("Summarizer 已延迟创建")

        return self._summarizer

    def _get_queue_key(self, group_id: str) -> str:
        """返回队列键

        调用方应传入会话 ID（见 PlatformAdapter.get_session_id）：
        群聊为群号，私聊为 f"private:{user_id}"，保证不同私聊用户
        不会共用同一个队列。此处不再做额外映射。
        """
        return group_id

    def _get_storage_group_id(self, queue_key: str) -> str:
        """从队列键推导 L2 记忆/画像归属使用的群 ID

        私聊会话键（private:{user_id}）还原为空字符串——L2 检索与画像
        注入对私聊均使用空群 ID（不按群过滤、归入空群桶），保持既有归属
        行为不变；群聊队列键即群号，原样返回。
        """
        if queue_key.startswith(PRIVATE_SESSION_PREFIX):
            return ""
        return queue_key

    def _get_or_create_queue(self, group_id: str) -> SegmentedMessageQueue:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._queues:
            config = get_config()
            self._queues[queue_key] = SegmentedMessageQueue(
                group_id=queue_key,
                segment_1_length=cast(int, config.get("l1_segment_1_length", 10)),
                segment_3_length=cast(int, config.get("l1_segment_3_length", 10)),
                total_length=cast(int, config.get("l1_buffer.inject_queue_length", 50)),
            )
            logger.debug(f"创建新队列：{queue_key}")

        return self._queues[queue_key]

    async def add_message(
        self,
        group_id: str,
        role: str,
        content: str,
        source: str,
        metadata: Optional[Dict] = None,
        persona_id: str = "default",
    ) -> bool:
        if not self._is_available:
            logger.warning("L1 缓冲不可用，跳过消息添加")
            return False

        if not group_id:
            # 空队列键会导致所有无群会话（私聊）混入同一个队列，
            # 造成跨用户上下文污染；调用方应传入会话 ID
            # （见 PlatformAdapter.get_session_id，私聊为 private:{user_id}）
            logger.warning(
                "L1 队列键为空，拒绝写入（调用方应使用 get_session_id 生成会话键）"
            )
            return False

        valid_roles = ("user", "assistant", "system")
        if role not in valid_roles:
            raise ValueError(f"无效的消息角色：{role!r}，合法值为 {valid_roles}")

        config = get_config()

        token_count = count_tokens(content)

        max_single_tokens = cast(int, config.get("l1_max_single_message_tokens", 500))
        if token_count > max_single_tokens:
            logger.warning(
                f"消息 Token 数 {token_count} 超过限制 {max_single_tokens}，已丢弃"
            )
            return False

        message = ContextMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            token_count=token_count,
            source=source,
            metadata=metadata or {},
            persona_id=persona_id,
        )

        queue = self._get_or_create_queue(group_id)
        queue.add_message(message)

        logger.debug(
            "消息已入队：%r, role=%r, tokens=%d, queue_size=%d, "
            "seg1=%d seg2=%d seg3=%d",
            group_id,
            role,
            token_count,
            len(queue),
            len(queue.segment_1),
            len(queue.segment_2),
            len(queue.segment_3),
        )

        self._schedule_summarize(group_id)

        return True

    def get_context(
        self, group_id: str, max_length: Optional[int] = None
    ) -> list[ContextMessage]:
        if not self._is_available:
            logger.warning("L1 缓冲不可用，返回空上下文")
            return []

        config = get_config()
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._queues:
            return []

        queue = self._queues[queue_key]

        if max_length is None:
            max_length = cast(int, config.get("l1_buffer.inject_queue_length", 50))

        messages = queue.inject_messages

        if len(messages) > max_length:
            messages = messages[-max_length:]

        logger.debug(
            f"获取上下文：{group_id}, 返回 {len(messages)}/{len(queue)} 条消息 "
            f"(不含 L1-3)"
        )

        return messages

    def clear_context(self, group_id: str) -> None:
        self.clear_by_group(group_id)

    def clear_all(self) -> int:
        total_messages = sum(len(q) for q in self._queues.values())
        # 收集所有图片项用于清理缓存文件
        all_image_items: list[Any] = []
        for img_list in self._image_queues.values():
            all_image_items.extend(img_list)
        self._queues.clear()
        self._image_queues.clear()
        self._summarizing_locks.clear()
        self._cleanup_image_cache_files(all_image_items)
        logger.info(f"已清空所有队列，共 {total_messages} 条消息")
        return total_messages

    def clear_by_user(self, user_id: str, group_id: Optional[str] = None) -> int:
        total_removed = 0

        if group_id:
            queue_key = self._get_queue_key(group_id)
            if queue_key in self._queues:
                queue = self._queues[queue_key]
                removed = queue.remove_user_messages(user_id)
                total_removed += removed
                logger.info(
                    f"已从队列 {queue_key} 删除用户 {user_id} 的 {removed} 条消息"
                )
        else:
            for queue_key, queue in self._queues.items():
                removed = queue.remove_user_messages(user_id)
                total_removed += removed
                if removed > 0:
                    logger.info(
                        f"已从队列 {queue_key} 删除用户 {user_id} 的 {removed} 条消息"
                    )

        return total_removed

    def clear_by_group(self, group_id: str) -> int:
        queue_key = self._get_queue_key(group_id)

        if queue_key in self._queues:
            queue = self._queues[queue_key]
            old_size = len(queue)
            queue.clear()
            logger.info(f"已清空队列：{queue_key}，原 {old_size} 条消息")
            self.clear_images_for_queue(group_id)
            self._summarizing_locks.pop(queue_key, None)
            return old_size

        return 0

    def _schedule_summarize(self, group_id: str) -> None:
        """将总结检查调度为后台任务，避免阻塞调用方（如 session lock 内的钩子）。

        _check_and_summarize 内部已有 asyncio.Lock 防止并发执行，
        多次调度是安全的：后续任务会看到锁被持有后立即返回。
        """
        task = asyncio.create_task(self._check_and_summarize(group_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _check_and_summarize(self, group_id: str) -> None:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._summarizing_locks:
            self._summarizing_locks[queue_key] = asyncio.Lock()

        if self._summarizing_locks[queue_key].locked():
            logger.debug(f"群聊 {queue_key} 正在总结中，跳过重复触发")
            return

        async with self._summarizing_locks[queue_key]:
            summarizer = self._get_or_create_summarizer()

            if not summarizer:
                logger.debug("Summarizer 不可用，跳过总结检查")
                return

            if queue_key not in self._queues:
                return

            queue = self._queues[queue_key]

            if not summarizer.should_summarize(queue):
                return

            try:
                target_messages = list(queue.segment_2)

                if not target_messages:
                    logger.debug(f"队列 {queue_key} L1-2 为空，无需总结")
                    return

                all_messages = queue.all_messages

                logger.info(
                    f"开始总结队列：{queue_key}，"
                    f"全量上下文 {len(all_messages)} 条，"
                    f"总结目标 L1-2 {len(target_messages)} 条"
                )

                summary = await summarizer.summarize(
                    context_messages=all_messages,
                    target_messages=target_messages,
                )

                if summary:
                    logger.info(f"总结完成：{queue_key}, 长度：{len(summary)}")

                    # L2/画像归属使用原始群 ID：私聊会话键还原为空字符串，
                    # 避免私聊总结写入读不到的 private: 命名空间
                    storage_group_id = self._get_storage_group_id(queue_key)

                    await self._write_summary_to_l2(
                        storage_group_id, target_messages, summary
                    )

                    await self._update_profile_after_summary(
                        storage_group_id, target_messages, summary
                    )

                    self._summary_fail_counts[queue_key] = 0
                else:
                    logger.warning(f"总结返回空，队列 {queue_key}")

                    fail_count = self._summary_fail_counts.get(queue_key, 0) + 1
                    self._summary_fail_counts[queue_key] = fail_count

                    if fail_count >= 2:
                        logger.warning(
                            f"队列 {queue_key} 总结连续失败 {fail_count} 次，"
                            f"清除 L1-2 段（可能包含审核不通过的消息）"
                        )
                        removed = queue.clear_segment_2()
                        self._clear_images_for_summarized_messages(queue_key, removed)
                        self._summary_fail_counts[queue_key] = 0
                    else:
                        logger.info(
                            f"队列 {queue_key} 空总结第 {fail_count} 次失败"
                            f"（阈值 2），保留 L1-2 段下轮重试"
                        )
                    # 无论是否达到阈值，空总结都不应推进段位——
                    # 否则未写入 L2 的内容会被 rotate_after_summary 丢弃，
                    # 使重试阈值形同虚设。与 except 分支保持一致。
                    return

                queue.rotate_after_summary(summarized_messages=target_messages)
                self._clear_images_for_summarized_messages(queue_key, target_messages)

                logger.info(
                    f"总结完成，段位转移后：L1-1={len(queue.segment_1)}, "
                    f"L1-2={len(queue.segment_2)}, L1-3={len(queue.segment_3)}, "
                    f"total_tokens={queue.total_tokens}"
                )

            except Exception as e:
                logger.error(f"总结队列 {queue_key} 失败：{e}", exc_info=True)

                fail_count = self._summary_fail_counts.get(queue_key, 0) + 1
                self._summary_fail_counts[queue_key] = fail_count

                if fail_count >= 2:
                    logger.warning(
                        f"队列 {queue_key} 总结连续异常 {fail_count} 次，"
                        f"清除 L1-2 段（可能包含审核不通过的消息）"
                    )
                    if queue_key in self._queues:
                        removed = queue.clear_segment_2()
                        self._clear_images_for_summarized_messages(queue_key, removed)
                    self._summary_fail_counts[queue_key] = 0

    async def _update_profile_after_summary(
        self, group_id: str, messages: list[ContextMessage], summary: str
    ) -> None:
        config = get_config()
        if not config.get("profile.enable"):
            return

        if not self._component_manager:
            return

        profile_storage = self._component_manager.get_component("profile")
        if not profile_storage or not profile_storage.is_available:
            return

        try:
            from iris_memory.profile import GroupProfileManager, UserProfileManager
            from iris_memory.profile.storage import ProfileStorage

            assert isinstance(profile_storage, ProfileStorage)

            group_manager = GroupProfileManager(profile_storage)
            user_manager = UserProfileManager(profile_storage)

            # 从消息中取 persona（取最新消息的 persona——会话切换时以最近活跃为准）
            persona_id = messages[-1].persona_id if messages else "default"

            user_messages_by_id: dict[str, list[str]] = {}
            for msg in messages:
                if msg.role == "user" and msg.source:
                    user_messages_by_id.setdefault(msg.source, []).append(msg.content)

            effective_group_id = (
                group_id
                if config.get("isolation_config.enable_group_isolation")
                else "default"
            )

            # 总结成功后递增计数器，使 should_update_mid 的"按总结次数触发"生效。
            # 生产代码此前从未调用 increment_summary_count，导致该路径完全失效。
            await group_manager.increment_summary_count(group_id, persona_id)
            for uid in user_messages_by_id:
                await user_manager.increment_summary_count(
                    uid, effective_group_id, persona_id
                )

            group_profile_obj = await group_manager.get_or_create(group_id, persona_id)
            group_should_mid = group_manager.should_update_mid(group_profile_obj)
            group_should_long = group_manager.should_update_long(group_profile_obj)

            if group_should_mid and group_should_long:
                await self._update_group_combined(
                    group_id, messages, group_manager, profile_storage, persona_id
                )
            elif group_should_mid:
                await self._update_group_mid_term(
                    group_id, messages, group_manager, profile_storage, persona_id
                )
            elif group_should_long:
                await self._update_group_long_term(
                    group_id, messages, group_manager, profile_storage, persona_id
                )

            for user_id, user_msgs in user_messages_by_id.items():
                user_profile_obj = await user_manager.get_or_create(
                    user_id, effective_group_id, persona_id
                )

                user_should_mid = user_manager.should_update_mid(user_profile_obj)
                user_should_long = user_manager.should_update_long(user_profile_obj)

                if user_should_mid and user_should_long:
                    await self._update_user_combined(
                        user_id,
                        effective_group_id,
                        user_msgs,
                        user_manager,
                        user_profile_obj,
                        profile_storage,
                        persona_id,
                    )
                elif user_should_mid:
                    await self._update_user_mid_term(
                        user_id,
                        effective_group_id,
                        user_msgs,
                        user_manager,
                        user_profile_obj,
                        profile_storage,
                        persona_id,
                    )
                elif user_should_long:
                    await self._update_user_long_term(
                        user_id,
                        effective_group_id,
                        user_msgs,
                        user_manager,
                        user_profile_obj,
                        profile_storage,
                        persona_id,
                    )

            logger.debug(f"总结后更新画像完成: {group_id}")

        except Exception as e:
            logger.error(f"更新画像失败: {e}", exc_info=True)

    async def _update_group_mid_term(
        self,
        group_id: str,
        messages: list[ContextMessage],
        group_manager: GroupProfileManager,
        profile_storage: ProfileStorage,
        persona_id: str = "default",
    ) -> None:
        assert self._component_manager is not None
        llm_manager = self._component_manager.get_component("llm_manager")
        if not llm_manager or not llm_manager.is_available:
            return

        try:
            from iris_memory.llm import LLMManager
            from iris_memory.profile import ProfileAnalyzer
            from iris_memory.profile.models import UpdateTier

            assert isinstance(llm_manager, LLMManager)
            analyzer = ProfileAnalyzer(llm_manager)
            group_profile_obj = await group_manager.get_or_create(group_id, persona_id)
            from iris_memory.profile.models import profile_to_dict

            current_profile_dict = profile_to_dict(group_profile_obj)

            msg_texts = [msg.content for msg in messages if msg.content]
            result = await analyzer.analyze_group_profile(
                msg_texts, current_profile_dict, tier=UpdateTier.MID
            )

            if result:
                await group_manager.update_from_analysis(
                    group_id=group_id,
                    interests=_as_str_list(result.get("interests")),
                    atmosphere_tags=_as_str_list(result.get("atmosphere_tags")),
                    custom_fields=_as_str_dict(result.get("custom_fields")),
                    tier=UpdateTier.MID,
                    confidence=0.7,
                    persona_id=persona_id,
                )
                logger.info(f"群聊画像中期更新完成: {group_id}")

        except Exception as e:
            logger.error(f"群聊画像中期更新失败: {e}", exc_info=True)

    async def _update_group_long_term(
        self,
        group_id: str,
        messages: list[ContextMessage],
        group_manager: GroupProfileManager,
        profile_storage: ProfileStorage,
        persona_id: str = "default",
    ) -> None:
        assert self._component_manager is not None
        llm_manager = self._component_manager.get_component("llm_manager")
        if not llm_manager or not llm_manager.is_available:
            return

        try:
            from iris_memory.llm import LLMManager
            from iris_memory.profile import ProfileAnalyzer
            from iris_memory.profile.models import UpdateTier

            assert isinstance(llm_manager, LLMManager)
            analyzer = ProfileAnalyzer(llm_manager)
            group_profile_obj = await group_manager.get_or_create(group_id, persona_id)
            from iris_memory.profile.models import profile_to_dict

            current_profile_dict = profile_to_dict(group_profile_obj)

            msg_texts = [msg.content for msg in messages if msg.content]
            result = await analyzer.analyze_group_profile(
                msg_texts, current_profile_dict, tier=UpdateTier.LONG
            )

            if result:
                await group_manager.update_long_term_from_analysis(
                    group_id=group_id,
                    long_term_tags=_as_str_list(result.get("long_term_tags")),
                    blacklist_topics=_as_str_list(result.get("blacklist_topics")),
                    interests=_as_str_list(result.get("interests")),
                    atmosphere_tags=_as_str_list(result.get("atmosphere_tags")),
                    custom_fields=_as_str_dict(result.get("custom_fields")),
                    confidence=0.8,
                    persona_id=persona_id,
                )
                logger.info(f"群聊画像长期更新完成: {group_id}")

        except Exception as e:
            logger.error(f"群聊画像长期更新失败: {e}", exc_info=True)

    async def _update_user_mid_term(
        self,
        user_id: str,
        group_id: str,
        user_messages: list[str],
        user_manager: UserProfileManager,
        user_profile_obj: UserProfile,
        profile_storage: ProfileStorage,
        persona_id: str = "default",
    ) -> None:
        assert self._component_manager is not None
        llm_manager = self._component_manager.get_component("llm_manager")
        if not llm_manager or not llm_manager.is_available:
            return

        try:
            from iris_memory.llm import LLMManager
            from iris_memory.profile import ProfileAnalyzer
            from iris_memory.profile.models import UpdateTier, profile_to_dict

            assert isinstance(llm_manager, LLMManager)
            analyzer = ProfileAnalyzer(llm_manager)
            current_profile_dict = profile_to_dict(user_profile_obj)

            result = await analyzer.analyze_user_profile(
                user_messages, current_profile_dict, tier=UpdateTier.MID
            )

            if result:
                await user_manager.update_from_analysis(
                    user_id=user_id,
                    group_id=group_id,
                    personality_tags=_as_str_list(result.get("personality_tags")),
                    interests=_as_str_list(result.get("interests")),
                    occupation=_as_str(result.get("occupation")),
                    language_style=_as_str(result.get("language_style")),
                    communication_style=_as_str(result.get("communication_style")),
                    emotional_baseline=_as_str(result.get("emotional_baseline")),
                    favorability_delta=_as_float(result.get("favorability_delta")),
                    custom_fields=_as_str_dict(result.get("custom_fields")),
                    tier=UpdateTier.MID,
                    confidence=0.7,
                    persona_id=persona_id,
                )
                logger.info(f"用户画像中期更新完成: {user_id}")

        except Exception as e:
            logger.error(f"用户画像中期更新失败: {e}", exc_info=True)

    async def _update_user_long_term(
        self,
        user_id: str,
        group_id: str,
        user_messages: list[str],
        user_manager: UserProfileManager,
        user_profile_obj: UserProfile,
        profile_storage: ProfileStorage,
        persona_id: str = "default",
    ) -> None:
        assert self._component_manager is not None
        llm_manager = self._component_manager.get_component("llm_manager")
        if not llm_manager or not llm_manager.is_available:
            return

        try:
            from iris_memory.llm import LLMManager
            from iris_memory.profile import ProfileAnalyzer
            from iris_memory.profile.models import UpdateTier, profile_to_dict

            assert isinstance(llm_manager, LLMManager)
            analyzer = ProfileAnalyzer(llm_manager)
            current_profile_dict = profile_to_dict(user_profile_obj)

            result = await analyzer.analyze_user_profile(
                user_messages, current_profile_dict, tier=UpdateTier.LONG
            )

            if result:
                await user_manager.update_long_term_from_analysis(
                    user_id=user_id,
                    group_id=group_id,
                    occupation=_as_str(result.get("occupation")),
                    bot_relationship=_as_str(result.get("bot_relationship")),
                    important_events=_as_str_list(result.get("important_events")),
                    taboo_topics=_as_str_list(result.get("taboo_topics")),
                    important_dates=result.get("important_dates"),  # type: ignore[arg-type]
                    personality_tags=_as_str_list(result.get("personality_tags")),
                    interests=_as_str_list(result.get("interests")),
                    language_style=_as_str(result.get("language_style")),
                    communication_style=_as_str(result.get("communication_style")),
                    emotional_baseline=_as_str(result.get("emotional_baseline")),
                    custom_fields=_as_str_dict(result.get("custom_fields")),
                    confidence=0.8,
                    persona_id=persona_id,
                )
                logger.info(f"用户画像长期更新完成: {user_id}")

        except Exception as e:
            logger.error(f"用户画像长期更新失败: {e}", exc_info=True)

    async def _update_user_combined(
        self,
        user_id: str,
        group_id: str,
        user_messages: list[str],
        user_manager: UserProfileManager,
        user_profile_obj: UserProfile,
        profile_storage: ProfileStorage,
        persona_id: str = "default",
    ) -> None:
        """合并 MID+LONG 为单次 LLM 调用，分别派发到 MID/LONG 更新路径。"""
        assert self._component_manager is not None
        llm_manager = self._component_manager.get_component("llm_manager")
        if not llm_manager or not llm_manager.is_available:
            return

        try:
            from iris_memory.llm import LLMManager
            from iris_memory.profile import ProfileAnalyzer
            from iris_memory.profile.models import UpdateTier, profile_to_dict

            assert isinstance(llm_manager, LLMManager)
            analyzer = ProfileAnalyzer(llm_manager)
            current_profile_dict = profile_to_dict(user_profile_obj)

            result = await analyzer.analyze_user_profile(
                user_messages,
                current_profile_dict,
                tier=UpdateTier.MID,
                combined=True,
            )

            if not result:
                return

            await user_manager.update_from_analysis(
                user_id=user_id,
                group_id=group_id,
                personality_tags=_as_str_list(result.get("personality_tags")),
                interests=_as_str_list(result.get("interests")),
                occupation=_as_str(result.get("occupation")),
                language_style=_as_str(result.get("language_style")),
                communication_style=_as_str(result.get("communication_style")),
                emotional_baseline=_as_str(result.get("emotional_baseline")),
                favorability_delta=_as_float(result.get("favorability_delta")),
                custom_fields=_as_str_dict(result.get("custom_fields")),
                tier=UpdateTier.MID,
                confidence=0.7,
                persona_id=persona_id,
            )
            await user_manager.update_long_term_from_analysis(
                user_id=user_id,
                group_id=group_id,
                occupation=_as_str(result.get("occupation")),
                bot_relationship=_as_str(result.get("bot_relationship")),
                important_events=_as_str_list(result.get("important_events")),
                taboo_topics=_as_str_list(result.get("taboo_topics")),
                important_dates=result.get("important_dates"),  # type: ignore[arg-type]
                confidence=0.8,
                persona_id=persona_id,
            )
            logger.info(f"用户画像合并更新完成: {user_id}")

        except Exception as e:
            logger.error(f"用户画像合并更新失败: {e}", exc_info=True)

    async def _update_group_combined(
        self,
        group_id: str,
        messages: list[ContextMessage],
        group_manager: GroupProfileManager,
        profile_storage: ProfileStorage,
        persona_id: str = "default",
    ) -> None:
        """合并群聊 MID+LONG 为单次 LLM 调用，分别派发到 MID/LONG 更新路径。"""
        assert self._component_manager is not None
        llm_manager = self._component_manager.get_component("llm_manager")
        if not llm_manager or not llm_manager.is_available:
            return

        try:
            from iris_memory.llm import LLMManager
            from iris_memory.profile import ProfileAnalyzer
            from iris_memory.profile.models import UpdateTier, profile_to_dict

            assert isinstance(llm_manager, LLMManager)
            analyzer = ProfileAnalyzer(llm_manager)
            group_profile_obj = await group_manager.get_or_create(group_id, persona_id)
            current_profile_dict = profile_to_dict(group_profile_obj)

            msg_texts = [msg.content for msg in messages if msg.content]
            result = await analyzer.analyze_group_profile(
                msg_texts,
                current_profile_dict,
                tier=UpdateTier.MID,
                combined=True,
            )

            if not result:
                return

            await group_manager.update_from_analysis(
                group_id=group_id,
                interests=_as_str_list(result.get("interests")),
                atmosphere_tags=_as_str_list(result.get("atmosphere_tags")),
                custom_fields=_as_str_dict(result.get("custom_fields")),
                tier=UpdateTier.MID,
                confidence=0.7,
                persona_id=persona_id,
            )
            await group_manager.update_long_term_from_analysis(
                group_id=group_id,
                long_term_tags=_as_str_list(result.get("long_term_tags")),
                blacklist_topics=_as_str_list(result.get("blacklist_topics")),
                confidence=0.8,
                persona_id=persona_id,
            )
            logger.info(f"群聊画像合并更新完成: {group_id}")

        except Exception as e:
            logger.error(f"群聊画像合并更新失败: {e}", exc_info=True)

    async def _write_summary_to_l2(
        self, group_id: str, messages: list[ContextMessage], summary: str
    ) -> Optional[str]:
        config = get_config()
        if not config.get("l2_memory.enable"):
            logger.debug("L2 记忆库未启用，跳过写入")
            return None

        if not self._component_manager:
            return None

        l2_adapter = self._component_manager.get_component("l2_memory")
        if not l2_adapter or not l2_adapter.is_available:
            logger.debug("L2 记忆库组件不可用，跳过写入")
            return None

        try:
            from iris_memory.l2_memory import MemoryRetriever
            from .summarizer import parse_summary_response, confidence_to_float

            retriever = MemoryRetriever(self._component_manager)

            # 从消息取 persona（取最新消息的 persona）
            persona_id = messages[-1].persona_id if messages else "default"

            name_to_id = self._build_name_to_id_map(messages)
            active_users = list(
                set(msg.source for msg in messages if msg.role == "user" and msg.source)
            )

            parsed = parse_summary_response(summary)
            summary_items = parsed.get("memories", [])
            json_parsed = parsed.get("json_parsed", False)

            if not json_parsed and summary_items:
                logger.warning(
                    f"总结响应未通过 JSON 解析，已使用文本回退提取 "
                    f"{len(summary_items)} 条记忆。模型输出可能不符合 JSON 格式要求，"
                    f"建议更换支持 JSON 输出的模型以提升总结质量。"
                )

            # 仅当 JSON 解析失败时才尝试行式回退。
            # JSON 解析成功但 memories 为空，属于模型判定“无值得记录的信息”的正常结果，
            # 结构性文本误当作记忆导入 L2。
            if not summary_items and not json_parsed:
                fallback_items = self._parse_summary_items(summary)
                if fallback_items:
                    summary_items = [
                        {"content": item, "confidence": "medium"}
                        for item in fallback_items
                    ]
                    logger.warning(
                        f"总结响应 JSON 解析失败且文本回退仅提取到 "
                        f"{len(fallback_items)} 条记忆（来自行式解析）。"
                        f"如大量出现此情况，建议更换支持 JSON 输出的模型。"
                        f"\n--- LLM 原始返回 ---\n{summary}\n--- 结束 ---"
                    )
            elif json_parsed and not summary_items:
                logger.debug(
                    "总结 JSON 解析成功，但 memories 为空（无值得记录的信息），跳过 L2 写入"
                )

            if not summary_items:
                logger.debug(f"总结解析后无有效条目，原内容：{summary[:100]}...")
                return None

            filtered_items = [
                item for item in summary_items if item.get("confidence") != "low"
            ]
            if len(filtered_items) < len(summary_items):
                logger.info(
                    f"过滤低置信度记忆：{len(summary_items)} -> {len(filtered_items)} 条"
                )

            max_per_summary = cast(int, config.get("l1_max_memories_per_summary", 10))
            if len(filtered_items) > max_per_summary:
                priority_order = {"high": 0, "medium": 1, "low": 2}
                filtered_items.sort(
                    key=lambda x: priority_order.get(x.get("confidence", "medium"), 1)
                )
                filtered_items = filtered_items[:max_per_summary]
                logger.info(
                    f"限制记忆数量：截取前 {max_per_summary} 条（按置信度优先）"
                )

            memory_ids = []
            quality_filtered = 0
            for item in filtered_items:
                content = item.get("content", "")
                if not content:
                    continue

                confidence_str = item.get("confidence", "medium")
                confidence_value = confidence_to_float(confidence_str)

                user_id = self._extract_user_from_item(content, name_to_id)

                # 内容质量校验：拦截第一人称片段、即时性内容、拼接痕迹等
                # 低质量提取，避免无效记忆污染 L2。
                is_low_quality, reason = self._is_low_quality_memory(content, user_id)
                if is_low_quality:
                    quality_filtered += 1
                    logger.info(f"过滤低质量记忆：{content[:50]}（原因：{reason}）")
                    continue

                metadata = {
                    "group_id": group_id,
                    "source": "l1_summary",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": confidence_value,
                    "confidence_level": confidence_str,
                    "kg_processed": False,
                }

                if user_id:
                    metadata["user_id"] = user_id
                else:
                    # 标记为无主体记忆：总结未能关联到具体用户。
                    # 遗忘清洗阶段会对无主体记忆加速淘汰，避免无主信息
                    # 长期占据 L2 并流入下游 L3 图谱产生孤儿节点。
                    metadata["subjectless"] = True

                if active_users:
                    metadata["active_users"] = ",".join(active_users)

                memory_id = await retriever.add_from_summary(
                    content, metadata, persona_id
                )
                if memory_id:
                    memory_ids.append(memory_id)

            if memory_ids:
                high_count = sum(
                    1
                    for item in filtered_items[: len(memory_ids)]
                    if item.get("confidence") == "high"
                )
                medium_count = sum(
                    1
                    for item in filtered_items[: len(memory_ids)]
                    if item.get("confidence") == "medium"
                )
                quality_msg = (
                    f"，质量过滤 {quality_filtered} 条" if quality_filtered else ""
                )
                logger.info(
                    f"已将 {len(memory_ids)} 条记忆写入 L2 记忆库 "
                    f"(high={high_count}, medium={medium_count}{quality_msg})"
                )
                return memory_ids[0]
            else:
                if quality_filtered:
                    logger.info(
                        f"写入 L2 记忆库：全部 {quality_filtered} 条记忆被质量校验过滤"
                    )
                else:
                    logger.debug("写入 L2 记忆库：无新记忆（可能全部去重）")
                return None

        except Exception as e:
            logger.error(f"写入 L2 记忆库失败: {e}", exc_info=True)
            return None

    def _build_name_to_id_map(self, messages: list[ContextMessage]) -> dict[str, str]:
        name_to_id: dict[str, str] = {}
        for msg in messages:
            if msg.role == "user" and msg.source and msg.metadata:
                user_name = msg.metadata.get("user_name")
                if user_name and user_name not in name_to_id:
                    name_to_id[user_name] = msg.source
        return name_to_id

    def _extract_user_from_item(
        self, item: str, name_to_id: dict[str, str]
    ) -> Optional[str]:
        if not name_to_id:
            return None

        for user_name, user_id in sorted(
            name_to_id.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if user_name in item:
                return user_id

        return None

    # 第一人称代词开头模式——这些是原始对话片段而非提取后的记忆。
    # 记忆应以第三人称表述（如"张三喜欢Python"），不应以"我想"/"我是"开头。
    _FIRST_PERSON_PATTERNS: list[re.Pattern[str]] = [
        re.compile(
            r"^(我想|我要|我喜欢|我爱|我讨厌|我是|我在|我有|我去|我正|我的|我觉|我认为|我觉得|我打算|我准备|我希望|我需要|我正在|我最近|我今天|我昨天|我明天|我这|我那|我会|我能|我可以|我想要)"
        ),
    ]

    # 即时性内容模式——这些是临时性内容，不具备长期价值。
    _IMMEDIATE_DESIRE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"(想吃|想喝|想去|想看|想玩|想买|想睡|想休息|想吃煎蛋|想做饭)"),
    ]

    @classmethod
    def _is_low_quality_memory(
        cls, content: str, user_id: Optional[str]
    ) -> tuple[bool, str]:
        """检查记忆内容是否为低质量提取

        检测 LLM 常见的不当提取模式：
        1. 第一人称开头且无法关联用户——记忆应以第三人称表述
        2. 即时性内容——"想吃""想去"等临时欲望不具备长期价值
        3. 内容过短——可能是碎片化提取
        4. 拼接痕迹——多个不相关片段被合并为一条

        Args:
            content: 记忆内容文本
            user_id: 从对话中解析出的用户 ID（None 表示无法关联主体）

        Returns:
            (是否低质量, 原因说明)
        """
        stripped = content.strip()

        # 过短：可能是碎片化提取
        if len(stripped) < 4:
            return True, "内容过短（<4字符），可能是碎片化提取"

        # 过长：可能不是精炼的记忆而是整段对话复制
        if len(stripped) > 300:
            return True, "内容过长（>300字符），可能不是精炼提取"

        # 第一人称开头且无法关联用户——记忆应以第三人称表述
        # 例如"我想吃煎蛋"而非"张三想吃煎蛋"
        if user_id is None:
            for pattern in cls._FIRST_PERSON_PATTERNS:
                if pattern.match(stripped):
                    return True, "第一人称开头且无法关联用户，应为第三人称表述"

        # 即时性内容——"想吃""想去"等临时欲望
        for pattern in cls._IMMEDIATE_DESIRE_PATTERNS:
            if pattern.search(stripped):
                return True, "即时性内容（临时欲望），不具备长期价值"

        # 拼接痕迹检测：内容中包含明显不连贯的语义断裂
        # 如"我想吃煎蛋8岁"——"想吃煎蛋"和"8岁"是两个不相关的片段
        # 启发式：末尾突然出现年龄/数字且与前文无语义连接
        if re.search(r"[^\d]{2,}\d+岁$", stripped):
            return True, "疑似拼接痕迹（末尾突兀年龄数字）"

        return False, ""

    def _parse_summary_items(self, summary: str, min_length: int = 5) -> list[str]:
        items = []
        lines = summary.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 跳过 Markdown 代码块标记（```json、``` 等）
            if line.startswith("```"):
                continue
            # 跳过 JSON 结构性行：以花括号开头（含 {"content":...} 这类对象片段）、
            # 纯方括号、或形如 "key": 的键名行，避免把 JSON 骨架当作记忆
            if line[0] in "{}":
                continue
            if line in ("[", "]"):
                continue
            if re.match(r'^"[a-zA-Z_]+"[\s]*:', line):
                continue

            if line in ("无", "无有效信息", "无有效记忆", "无有价值的信息"):
                continue

            if line.startswith("- "):
                line = line[2:]
            elif line.startswith("• "):
                line = line[2:]
            elif len(line) > 2 and line[0].isdigit() and line[1] in ".、)":
                line = line[2:].strip()

            line = line.strip()
            if len(line) >= min_length:
                items.append(line)

        return items

    def get_queue_stats(self, group_id: str) -> Optional[Dict]:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._queues:
            return None

        queue = self._queues[queue_key]

        return {
            "group_id": queue_key,
            "message_count": len(queue),
            "total_tokens": queue.total_tokens,
            "segment_1_count": len(queue.segment_1),
            "segment_2_count": len(queue.segment_2),
            "segment_3_count": len(queue.segment_3),
        }

    def get_stats(self) -> Dict[str, Any]:
        config = get_config()

        total_messages = 0
        total_tokens = 0
        queue_count = len(self._queues)

        max_queue_length = 0
        for queue in self._queues.values():
            total_messages += len(queue)
            total_tokens += queue.total_tokens
            if len(queue) > max_queue_length:
                max_queue_length = len(queue)

        return {
            "queue_count": queue_count,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "max_capacity": cast(int, config.get("l1_buffer.inject_queue_length", 50)),
            "max_queue_length": max_queue_length,
        }

    def get_all_queues_stats(self) -> List[Dict[str, Any]]:
        queues_stats = []
        for group_id, queue in self._queues.items():
            queues_stats.append(
                {
                    "group_id": group_id,
                    "message_count": len(queue),
                    "total_tokens": queue.total_tokens,
                    "segment_1_count": len(queue.segment_1),
                    "segment_2_count": len(queue.segment_2),
                    "segment_3_count": len(queue.segment_3),
                }
            )
        return queues_stats

    # ========================================================================
    # 图片队列管理
    # ========================================================================

    def append_to_last_message(self, group_id: str, suffix: str) -> bool:
        if not self._is_available:
            return False

        queue_key = self._get_queue_key(group_id)
        if queue_key not in self._queues:
            return False

        queue = self._queues[queue_key]
        if queue.segment_1:
            last_msg = queue.segment_1[-1]
        elif queue.segment_2:
            last_msg = queue.segment_2[-1]
        elif queue.segment_3:
            last_msg = queue.segment_3[-1]
        else:
            return False

        old_tokens = last_msg.token_count
        last_msg.content += suffix
        last_msg.token_count = count_tokens(last_msg.content)

        token_diff = last_msg.token_count - old_tokens
        queue.total_tokens += token_diff

        logger.debug(
            f"已追加文本到最后消息：{queue_key}, 追加长度={len(suffix)}, "
            f"token差={token_diff}"
        )
        return True

    def prepend_to_last_message(
        self, group_id: str, prefix: str, same_source: str = ""
    ) -> bool:
        if not self._is_available:
            return False

        queue_key = self._get_queue_key(group_id)
        if queue_key not in self._queues:
            return False

        queue = self._queues[queue_key]
        if queue.segment_1:
            last_msg = queue.segment_1[-1]
        elif queue.segment_2:
            last_msg = queue.segment_2[-1]
        elif queue.segment_3:
            last_msg = queue.segment_3[-1]
        else:
            return False

        if same_source and last_msg.source != same_source:
            return False

        old_tokens = last_msg.token_count
        last_msg.content = prefix + last_msg.content
        last_msg.token_count = count_tokens(last_msg.content)

        token_diff = last_msg.token_count - old_tokens
        queue.total_tokens += token_diff

        logger.debug(
            f"已前置文本到最后消息：{queue_key}, 前置长度={len(prefix)}, "
            f"token差={token_diff}"
        )
        return True

    def replace_image_placeholder(
        self, group_id: str, placeholder: str, description: str
    ) -> bool:
        if not self._is_available:
            return False

        queue_key = self._get_queue_key(group_id)
        if queue_key not in self._queues:
            return False

        queue = self._queues[queue_key]
        replaced = False

        for msg in queue.all_messages:
            if placeholder in msg.content:
                old_tokens = msg.token_count
                msg.content = msg.content.replace(placeholder, description)
                msg.token_count = count_tokens(msg.content)

                token_diff = msg.token_count - old_tokens
                queue.total_tokens += token_diff

                replaced = True
                logger.debug(
                    f"已替换图片占位符：{queue_key}, "
                    f"{placeholder} -> {description[:30]}..."
                )

        return replaced

    def add_image(self, group_id: str, image_item: Any) -> None:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._image_queues:
            self._image_queues[queue_key] = []

        self._image_queues[queue_key].append(image_item)
        logger.debug(
            f"图片已入队：{queue_key}, hash={image_item.image_hash[:8]}..., "
            f"队列大小={len(self._image_queues[queue_key])}"
        )

    def get_images(
        self, group_id: str, limit: Optional[int] = None, only_pending: bool = True
    ) -> List[Any]:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._image_queues:
            return []

        images = self._image_queues[queue_key]

        if only_pending:
            from iris_memory.image import ImageParseStatus

            images = [img for img in images if img.status == ImageParseStatus.PENDING]

        if limit is not None:
            images = images[:limit]

        return images

    def mark_image_parsed(self, group_id: str, image_hash: str, status: Any) -> bool:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._image_queues:
            return False

        for img in self._image_queues[queue_key]:
            if img.image_hash == image_hash:
                img.status = status
                logger.debug(
                    f"图片状态已更新：{queue_key}, hash={image_hash[:8]}..., "
                    f"status={status.value}"
                )
                return True

        return False

    def clear_images_for_message(self, group_id: str, message_id: str) -> int:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._image_queues:
            return 0

        original_count = len(self._image_queues[queue_key])
        removed_items = [
            img for img in self._image_queues[queue_key] if img.message_id == message_id
        ]
        self._image_queues[queue_key] = [
            img for img in self._image_queues[queue_key] if img.message_id != message_id
        ]

        removed_count = original_count - len(self._image_queues[queue_key])
        if removed_count > 0:
            self._cleanup_image_cache_files(removed_items)
            logger.debug(f"已清理消息 {message_id} 的 {removed_count} 张图片")

        return removed_count

    def clear_images_for_queue(self, group_id: str) -> int:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._image_queues:
            return 0

        removed_items = list(self._image_queues[queue_key])
        removed_count = len(removed_items)
        del self._image_queues[queue_key]

        if removed_count > 0:
            self._cleanup_image_cache_files(removed_items)
            logger.debug(f"已清理队列 {queue_key} 的 {removed_count} 张图片")

        return removed_count

    def get_all_phash_hashes(self) -> list[str]:
        """获取所有队列中 pHash 格式的图片哈希列表

        用于跨队列 pHash 去重检查。

        Returns:
            所有以 'ph:' 开头的图片哈希列表
        """
        hashes: list[str] = []
        for img_list in self._image_queues.values():
            for img in img_list:
                if img.image_hash.startswith("ph:"):
                    hashes.append(img.image_hash)
        return hashes

    def get_image_stats(self, group_id: str) -> Optional[Dict[str, Any]]:
        queue_key = self._get_queue_key(group_id)

        if queue_key not in self._image_queues:
            return None

        from iris_memory.image import ImageParseStatus

        images = self._image_queues[queue_key]
        pending_count = sum(
            1 for img in images if img.status == ImageParseStatus.PENDING
        )
        success_count = sum(
            1 for img in images if img.status == ImageParseStatus.SUCCESS
        )
        failed_count = sum(1 for img in images if img.status == ImageParseStatus.FAILED)

        return {
            "group_id": queue_key,
            "total_count": len(images),
            "pending_count": pending_count,
            "success_count": success_count,
            "failed_count": failed_count,
        }

    def _clear_images_for_summarized_messages(
        self, queue_key: str, messages: list[ContextMessage]
    ) -> int:
        if queue_key not in self._image_queues:
            return 0

        message_ids = set()
        for msg in messages:
            msg_id = msg.metadata.get("message_id")
            if msg_id:
                message_ids.add(msg_id)

        if not message_ids:
            return 0

        original_count = len(self._image_queues[queue_key])
        removed_items = [
            img
            for img in self._image_queues[queue_key]
            if img.message_id in message_ids
        ]
        self._image_queues[queue_key] = [
            img
            for img in self._image_queues[queue_key]
            if img.message_id not in message_ids
        ]

        removed_count = original_count - len(self._image_queues[queue_key])
        if removed_count > 0:
            self._cleanup_image_cache_files(removed_items)
            logger.debug(f"已清理被总结消息的 {removed_count} 张图片")

        return removed_count

    @staticmethod
    def _cleanup_image_cache_files(items: list[Any]) -> None:
        """删除图片队列项对应的本地缓存文件

        仅删除位于插件 image_cache 目录下的文件，避免误删平台原始文件。

        Args:
            items: 被移除的 ImageQueueItem 列表
        """
        for item in items:
            try:
                info = getattr(item, "image_info", None)
                if not info:
                    continue
                fp = getattr(info, "file_path", None)
                if not fp:
                    continue
                path = Path(fp)
                if not path.is_absolute():
                    continue
                # 仅清理 image_cache 目录下的文件，不删平台原始文件
                if "image_cache" not in path.parts:
                    continue
                if path.exists():
                    path.unlink()
                    logger.debug(f"已删除图片缓存：{path.name}")
            except Exception as e:
                logger.debug(f"删除图片缓存文件失败：{e}")

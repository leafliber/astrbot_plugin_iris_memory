"""
Iris Chat Memory - 群聊画像管理器

封装群聊画像的业务逻辑。
仅保留中长期字段更新。
"""

from typing import List, Optional

from iris_memory.core import get_logger
from iris_memory.config import get_config
from .storage import ProfileStorage, profile_lock
from .models import (
    GroupProfile,
    UpdateTier,
    merge_list_field,
    merge_custom_fields,
    ProfileConfig,
)

logger = get_logger("profile")


class GroupProfileManager:
    """群聊画像管理器

    封装群聊画像的业务逻辑，提供：
    - 获取/创建画像
    - 更新简单字段（实时，无LLM）
    - 更新中期字段（LLM分析，按频率触发）
    - 更新长期字段（LLM分析，高置信度触发）
    - 智能合并：列表字段合并而非替换

    Attributes:
        _storage: 画像存储组件实例
    """

    def __init__(self, storage: ProfileStorage):
        self._storage = storage

    async def get_or_create(
        self, group_id: str, persona_id: str = "default"
    ) -> GroupProfile:
        """获取或创建群聊画像

        Args:
            group_id: 群聊ID
            persona_id: 人格ID

        Returns:
            群聊画像对象
        """
        profile = await self._storage.get_group_profile(group_id, persona_id)

        if not profile:
            profile = GroupProfile(group_id=group_id)
            logger.info(f"创建新群聊画像: {group_id}")

        return profile

    @profile_lock("group")
    async def update_group_name(
        self, group_id: str, group_name: str, persona_id: str = "default"
    ) -> None:
        """更新群聊名称

        Args:
            group_id: 群聊ID
            group_name: 群聊名称
            persona_id: 人格ID
        """
        if not group_name:
            return

        profile = await self.get_or_create(group_id, persona_id)

        if group_name != profile.group_name:
            profile.group_name = group_name
            await self._storage.save_group_profile(
                profile, increment_version=True, persona_id=persona_id
            )
            logger.debug(f"更新群聊名称: {group_id} -> {group_name}")

    @profile_lock("group")
    async def update_from_analysis(
        self,
        group_id: str,
        interests: Optional[List[str]] = None,
        atmosphere_tags: Optional[List[str]] = None,
        custom_fields: Optional[dict] = None,
        tier: UpdateTier = UpdateTier.MID,
        confidence: float = 0.7,
        persona_id: str = "default",
    ) -> None:
        """从分析结果更新字段（LLM分析后调用）

        根据更新层级和置信度智能合并字段：
        - 列表字段：合并新旧值，新值优先

        Args:
            group_id: 群聊ID
            interests: 群聊兴趣点列表
            atmosphere_tags: 氛围标签列表
            custom_fields: 自定义字段字典
            tier: 更新层级
            confidence: 本次分析的整体置信度
            persona_id: 人格ID
        """
        profile = await self.get_or_create(group_id, persona_id)
        updated = False

        if interests is not None:
            meta = profile.get_field_meta("interests")
            merged = merge_list_field(profile.interests, interests)
            if merged != profile.interests:
                profile.interests = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("interests", meta)
                updated = True

        if atmosphere_tags is not None:
            meta = profile.get_field_meta("atmosphere_tags")
            merged = merge_list_field(profile.atmosphere_tags, atmosphere_tags)
            if merged != profile.atmosphere_tags:
                profile.atmosphere_tags = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("atmosphere_tags", meta)
                updated = True

        if custom_fields:
            merged_fields, fields_changed = merge_custom_fields(
                profile.custom_fields, custom_fields, confidence=confidence
            )
            if fields_changed:
                profile.custom_fields = merged_fields
                updated = True

        tracker = profile.get_update_tracker()
        if tier == UpdateTier.MID:
            tracker.record_mid_update()
        elif tier == UpdateTier.LONG:
            tracker.record_long_update()
        profile.set_update_tracker(tracker)

        await self._storage.save_group_profile(
            profile, increment_version=updated, persona_id=persona_id
        )
        if updated:
            logger.info(f"从分析结果更新群聊画像: {group_id} (tier={tier.value})")

    @profile_lock("group")
    async def update_long_term_from_analysis(
        self,
        group_id: str,
        long_term_tags: Optional[List[str]] = None,
        blacklist_topics: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        atmosphere_tags: Optional[List[str]] = None,
        custom_fields: Optional[dict] = None,
        confidence: float = 0.8,
        persona_id: str = "default",
    ) -> None:
        """从长期分析结果更新字段

        长期字段要求更高置信度，且更难被覆盖。

        Args:
            group_id: 群聊ID
            long_term_tags: 核心特征标签
            blacklist_topics: 禁忌话题
            interests: 兴趣点（如有变化）
            atmosphere_tags: 氛围标签（如有变化）
            custom_fields: 自定义字段
            confidence: 置信度
            persona_id: 人格ID
        """
        profile = await self.get_or_create(group_id, persona_id)
        updated = False

        if long_term_tags is not None:
            meta = profile.get_field_meta("long_term_tags")
            merged = merge_list_field(
                profile.long_term_tags, long_term_tags, max_items=10
            )
            if merged != profile.long_term_tags:
                profile.long_term_tags = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("long_term_tags", meta)
                updated = True

        if blacklist_topics is not None:
            blacklist_changed = False
            for topic in blacklist_topics:
                if topic and topic not in profile.blacklist_topics:
                    profile.blacklist_topics.append(topic)
                    blacklist_changed = True
                    updated = True
            if blacklist_changed:
                profile.blacklist_topics = profile.blacklist_topics[-10:]
                meta = profile.get_field_meta("blacklist_topics")
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("blacklist_topics", meta)

        if interests is not None:
            meta = profile.get_field_meta("interests")
            merged = merge_list_field(profile.interests, interests)
            if merged != profile.interests:
                profile.interests = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("interests", meta)
                updated = True

        if atmosphere_tags is not None:
            meta = profile.get_field_meta("atmosphere_tags")
            merged = merge_list_field(profile.atmosphere_tags, atmosphere_tags)
            if merged != profile.atmosphere_tags:
                profile.atmosphere_tags = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("atmosphere_tags", meta)
                updated = True

        if custom_fields:
            merged_fields, fields_changed = merge_custom_fields(
                profile.custom_fields, custom_fields, confidence=confidence
            )
            if fields_changed:
                profile.custom_fields = merged_fields
                updated = True

        tracker = profile.get_update_tracker()
        tracker.record_long_update()
        profile.set_update_tracker(tracker)

        await self._storage.save_group_profile(
            profile, increment_version=updated, persona_id=persona_id
        )
        if updated:
            logger.info(f"从长期分析更新群聊画像: {group_id}")

    def should_update_mid(self, profile: GroupProfile) -> bool:
        """判断群聊画像是否需要中期更新

        Args:
            profile: 群聊画像

        Returns:
            是否需要更新
        """
        config = get_config()
        interval_summaries = ProfileConfig.get_mid_update_interval_summaries(config)
        interval_hours = ProfileConfig.get_mid_update_interval_hours(config)

        tracker = profile.get_update_tracker()
        return tracker.should_update_mid(interval_summaries, interval_hours)

    def should_update_long(self, profile: GroupProfile) -> bool:
        """判断群聊画像是否需要长期更新

        Args:
            profile: 群聊画像

        Returns:
            是否需要更新
        """
        config = get_config()
        interval_hours = ProfileConfig.get_long_update_interval_hours(config)

        tracker = profile.get_update_tracker()
        return tracker.should_update_long(interval_hours)

    async def increment_summary_count(
        self, group_id: str, persona_id: str = "default"
    ) -> None:
        """总结次数 +1（中期更新触发条件之一）

        应在每次 L2 总结成功后调用。生产代码此前从未调用此方法，
        导致 summary_count_since_mid_update 恒为 0，
        "按总结次数触发中期更新"完全失效。
        """
        profile = await self.get_or_create(group_id, persona_id)
        tracker = profile.get_update_tracker()
        tracker.increment_summary_count()
        profile.set_update_tracker(tracker)
        await self._storage.save_group_profile(profile, persona_id=persona_id)

    @profile_lock("group")
    async def add_long_term_tag(
        self, group_id: str, tag: str, persona_id: str = "default"
    ) -> None:
        """添加长期标签（人工或高质量LLM更新）"""
        profile = await self.get_or_create(group_id, persona_id)

        if tag not in profile.long_term_tags:
            profile.long_term_tags.append(tag)
            meta = profile.get_field_meta("long_term_tags")
            meta.record_update(1.0, source="manual")
            profile.set_field_meta("long_term_tags", meta)
            await self._storage.save_group_profile(profile, persona_id=persona_id)
            logger.info(f"添加群聊长期标签: {group_id} -> {tag}")

    @profile_lock("group")
    async def add_blacklist_topic(
        self, group_id: str, topic: str, persona_id: str = "default"
    ) -> None:
        """添加禁忌话题"""
        profile = await self.get_or_create(group_id, persona_id)

        if topic not in profile.blacklist_topics:
            profile.blacklist_topics.append(topic)
            meta = profile.get_field_meta("blacklist_topics")
            meta.record_update(1.0, source="manual")
            profile.set_field_meta("blacklist_topics", meta)
            await self._storage.save_group_profile(profile, persona_id=persona_id)
            logger.info(f"添加群聊禁忌话题: {group_id} -> {topic}")

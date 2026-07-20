"""
Iris Chat Memory - 用户画像管理器

封装用户画像的业务逻辑。
仅保留中长期字段更新。
"""

from typing import List, Optional, Dict

from iris_memory.core import get_logger
from iris_memory.config import get_config
from .storage import ProfileStorage, profile_lock
from .models import (
    UserProfile,
    UpdateTier,
    merge_list_field,
    should_overwrite_field,
    merge_custom_fields,
    ProfileConfig,
)

logger = get_logger("profile")


class UserProfileManager:
    """用户画像管理器

    封装用户画像的业务逻辑，提供：
    - 获取/创建画像
    - 更新中期字段（LLM分析，按频率触发）
    - 更新长期字段（LLM分析，高置信度触发）
    - 智能合并：列表字段合并而非替换，置信度控制覆盖

    Attributes:
        _storage: 画像存储组件实例
    """

    def __init__(self, storage: ProfileStorage):
        self._storage = storage

    async def get_or_create(
        self, user_id: str, group_id: str = "default", persona_id: str = "default"
    ) -> UserProfile:
        """获取或创建用户画像

        Args:
            user_id: 用户ID
            group_id: 群聊ID（全局模式传 "default"）
            persona_id: 人格ID

        Returns:
            用户画像对象
        """
        profile = await self._storage.get_user_profile(user_id, group_id, persona_id)

        if not profile:
            profile = UserProfile(user_id=user_id)
            mode = "全局" if group_id == "default" else f"群聊{group_id}"
            logger.info(f"创建新用户画像: {user_id} ({mode})")

        return profile

    @profile_lock("user")
    async def update_user_name(
        self,
        user_id: str,
        group_id: str = "default",
        user_name: Optional[str] = None,
        persona_id: str = "default",
    ) -> None:
        """更新用户昵称

        Args:
            user_id: 用户ID
            group_id: 群聊ID
            user_name: 用户昵称
            persona_id: 人格ID
        """
        if not user_name:
            return

        profile = await self.get_or_create(user_id, group_id, persona_id)

        if user_name != profile.user_name:
            if profile.user_name and profile.user_name not in profile.historical_names:
                profile.historical_names.append(profile.user_name)
            profile.user_name = user_name
            await self._storage.save_user_profile(
                profile, group_id, increment_version=True, persona_id=persona_id
            )
            logger.debug(f"更新用户昵称: {user_id} -> {user_name}")

    @profile_lock("user")
    async def update_from_analysis(
        self,
        user_id: str,
        group_id: str,
        personality_tags: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        occupation: Optional[str] = None,
        language_style: Optional[str] = None,
        communication_style: Optional[str] = None,
        emotional_baseline: Optional[str] = None,
        favorability_delta: Optional[float] = None,
        custom_fields: Optional[dict] = None,
        tier: UpdateTier = UpdateTier.MID,
        confidence: float = 0.7,
        persona_id: str = "default",
    ) -> None:
        """从分析结果更新字段（LLM分析后调用）

        根据更新层级和置信度智能合并字段：
        - 列表字段：合并新旧值，新值优先
        - 字符串字段：置信度更高时覆盖
        - 好感度：按 delta 增量更新，夹紧到 [0, 100]

        Args:
            user_id: 用户ID
            group_id: 群聊ID
            personality_tags: 性格标签列表
            interests: 兴趣爱好列表
            occupation: 职业/身份
            language_style: 语言风格
            communication_style: 期望的沟通偏好（简洁/详细/随意/正式）
            emotional_baseline: 情感基线（稳定/敏感/乐观/低落）
            favorability_delta: 好感度变化量（-20~+20，由配置夹紧）
            custom_fields: 自定义字段字典
            tier: 更新层级
            confidence: 本次分析的整体置信度
            persona_id: 人格ID
        """
        profile = await self.get_or_create(user_id, group_id, persona_id)
        updated = False

        if personality_tags is not None:
            meta = profile.get_field_meta("personality_tags")
            merged = merge_list_field(profile.personality_tags, personality_tags)
            if merged != profile.personality_tags:
                profile.personality_tags = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("personality_tags", meta)
                updated = True

        if interests is not None:
            meta = profile.get_field_meta("interests")
            merged = merge_list_field(profile.interests, interests)
            if merged != profile.interests:
                profile.interests = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("interests", meta)
                updated = True

        if language_style is not None:
            meta = profile.get_field_meta("language_style")
            long_term_confidence = (
                confidence if tier == UpdateTier.LONG else confidence * 0.8
            )
            if should_overwrite_field(
                profile.language_style,
                language_style,
                meta.confidence,
                long_term_confidence,
            ):
                profile.language_style = language_style
                meta.record_update(long_term_confidence, source="llm")
                profile.set_field_meta("language_style", meta)
                updated = True

        if communication_style is not None:
            meta = profile.get_field_meta("communication_style")
            if should_overwrite_field(
                profile.communication_style,
                communication_style,
                meta.confidence,
                confidence,
            ):
                profile.communication_style = communication_style
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("communication_style", meta)
                updated = True

        if emotional_baseline is not None:
            meta = profile.get_field_meta("emotional_baseline")
            if should_overwrite_field(
                profile.emotional_baseline,
                emotional_baseline,
                meta.confidence,
                confidence,
            ):
                profile.emotional_baseline = emotional_baseline
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("emotional_baseline", meta)
                updated = True

        if favorability_delta is not None:
            config = get_config()
            if config.get("profile.favorability_enable"):
                max_delta = float(config.get("profile_favorability_max_delta", 20.0))  # type: ignore[arg-type]
                clamped_delta = max(
                    -max_delta, min(max_delta, float(favorability_delta))
                )
                old_val = profile.favorability
                new_val = max(0.0, min(100.0, old_val + clamped_delta))
                if new_val != old_val:
                    profile.favorability = new_val
                    meta = profile.get_field_meta("favorability")
                    meta.record_update(confidence, source="llm")
                    profile.set_field_meta("favorability", meta)
                    updated = True

        if occupation is not None:
            meta = profile.get_field_meta("occupation")
            if should_overwrite_field(
                profile.occupation, occupation, meta.confidence, confidence
            ):
                profile.occupation = occupation
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("occupation", meta)
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

        await self._storage.save_user_profile(
            profile, group_id, increment_version=updated, persona_id=persona_id
        )
        if updated:
            logger.info(
                f"从分析结果更新用户画像: {user_id} (群聊: {group_id}, tier={tier.value})"
            )

    @profile_lock("user")
    async def update_long_term_from_analysis(
        self,
        user_id: str,
        group_id: str,
        occupation: Optional[str] = None,
        bot_relationship: Optional[str] = None,
        important_events: Optional[List[str]] = None,
        taboo_topics: Optional[List[str]] = None,
        important_dates: Optional[List[Dict[str, str]]] = None,
        personality_tags: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        language_style: Optional[str] = None,
        communication_style: Optional[str] = None,
        emotional_baseline: Optional[str] = None,
        custom_fields: Optional[dict] = None,
        confidence: float = 0.8,
        persona_id: str = "default",
    ) -> None:
        """从长期分析结果更新字段

        长期字段要求更高置信度，且更难被覆盖。

        Args:
            user_id: 用户ID
            group_id: 群聊ID
            occupation: 职业/身份
            bot_relationship: 对bot的称呼/关系设定
            important_events: 重要事件列表
            taboo_topics: 禁忌话题列表
            important_dates: 重要纪念日列表
            personality_tags: 性格标签（如有变化）
            interests: 兴趣爱好（如有变化）
            communication_style: 期望的沟通偏好
            emotional_baseline: 情感基线
            custom_fields: 自定义字段
            confidence: 置信度
            persona_id: 人格ID
        """
        profile = await self.get_or_create(user_id, group_id, persona_id)
        updated = False

        if occupation is not None:
            meta = profile.get_field_meta("occupation")
            if should_overwrite_field(
                profile.occupation, occupation, meta.confidence, confidence
            ):
                profile.occupation = occupation
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("occupation", meta)
                updated = True

        if bot_relationship is not None:
            meta = profile.get_field_meta("bot_relationship")
            if should_overwrite_field(
                profile.bot_relationship, bot_relationship, meta.confidence, confidence
            ):
                profile.bot_relationship = bot_relationship
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("bot_relationship", meta)
                updated = True

        if important_events is not None:
            events_changed = False
            for event in important_events:
                if event and event not in profile.important_events:
                    profile.important_events.append(event)
                    events_changed = True
                    updated = True
            if events_changed:
                profile.important_events = profile.important_events[-10:]
                meta = profile.get_field_meta("important_events")
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("important_events", meta)

        if taboo_topics is not None:
            taboo_changed = False
            for topic in taboo_topics:
                if topic and topic not in profile.taboo_topics:
                    profile.taboo_topics.append(topic)
                    taboo_changed = True
                    updated = True
            if taboo_changed:
                profile.taboo_topics = profile.taboo_topics[-10:]
                meta = profile.get_field_meta("taboo_topics")
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("taboo_topics", meta)

        if important_dates is not None:
            dates_changed = False
            for date_entry in important_dates:
                if date_entry and date_entry not in profile.important_dates:
                    profile.important_dates.append(date_entry)
                    dates_changed = True
                    updated = True
            if dates_changed:
                profile.important_dates = profile.important_dates[-10:]
                meta = profile.get_field_meta("important_dates")
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("important_dates", meta)

        if personality_tags is not None:
            meta = profile.get_field_meta("personality_tags")
            merged = merge_list_field(profile.personality_tags, personality_tags)
            if merged != profile.personality_tags:
                profile.personality_tags = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("personality_tags", meta)
                updated = True

        if interests is not None:
            meta = profile.get_field_meta("interests")
            merged = merge_list_field(profile.interests, interests)
            if merged != profile.interests:
                profile.interests = merged
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("interests", meta)
                updated = True

        if language_style is not None:
            meta = profile.get_field_meta("language_style")
            if should_overwrite_field(
                profile.language_style,
                language_style,
                meta.confidence,
                confidence,
            ):
                profile.language_style = language_style
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("language_style", meta)
                updated = True

        if communication_style is not None:
            meta = profile.get_field_meta("communication_style")
            if should_overwrite_field(
                profile.communication_style,
                communication_style,
                meta.confidence,
                confidence,
            ):
                profile.communication_style = communication_style
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("communication_style", meta)
                updated = True

        if emotional_baseline is not None:
            meta = profile.get_field_meta("emotional_baseline")
            if should_overwrite_field(
                profile.emotional_baseline,
                emotional_baseline,
                meta.confidence,
                confidence,
            ):
                profile.emotional_baseline = emotional_baseline
                meta.record_update(confidence, source="llm")
                profile.set_field_meta("emotional_baseline", meta)
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

        await self._storage.save_user_profile(
            profile, group_id, increment_version=updated, persona_id=persona_id
        )
        if updated:
            logger.info(f"从长期分析更新用户画像: {user_id} (群聊: {group_id})")

    def should_update_mid(self, profile: UserProfile) -> bool:
        """判断用户画像是否需要中期更新

        Args:
            profile: 用户画像

        Returns:
            是否需要更新
        """
        config = get_config()
        interval_summaries = ProfileConfig.get_mid_update_interval_summaries(config)
        interval_hours = ProfileConfig.get_mid_update_interval_hours(config)

        tracker = profile.get_update_tracker()
        return tracker.should_update_mid(interval_summaries, interval_hours)

    def should_update_long(self, profile: UserProfile) -> bool:
        """判断用户画像是否需要长期更新

        Args:
            profile: 用户画像

        Returns:
            是否需要更新
        """
        config = get_config()
        interval_hours = ProfileConfig.get_long_update_interval_hours(config)

        tracker = profile.get_update_tracker()
        return tracker.should_update_long(interval_hours)

    async def increment_summary_count(
        self, user_id: str, group_id: str, persona_id: str = "default"
    ) -> None:
        """总结次数 +1（中期更新触发条件之一）

        应在每次 L2 总结成功后调用。生产代码此前从未调用此方法，
        导致 summary_count_since_mid_update 恒为 0，
        "按总结次数触发中期更新"完全失效。
        """
        profile = await self.get_or_create(user_id, group_id, persona_id)
        tracker = profile.get_update_tracker()
        tracker.increment_summary_count()
        profile.set_update_tracker(tracker)
        await self._storage.save_user_profile(
            profile, group_id=group_id, persona_id=persona_id
        )

    @profile_lock("user")
    async def set_bot_relationship(
        self,
        user_id: str,
        group_id: str,
        relationship: str,
        persona_id: str = "default",
    ) -> None:
        """设置用户对 bot 的称呼/关系设定"""
        profile = await self.get_or_create(user_id, group_id, persona_id)
        profile.bot_relationship = relationship
        meta = profile.get_field_meta("bot_relationship")
        meta.record_update(1.0, source="manual")
        profile.set_field_meta("bot_relationship", meta)
        await self._storage.save_user_profile(profile, group_id, persona_id=persona_id)
        logger.info(f"设置用户对bot的关系: {user_id} -> {relationship}")

    @profile_lock("user")
    async def add_important_date(
        self,
        user_id: str,
        group_id: str,
        date: str,
        description: str,
        persona_id: str = "default",
    ) -> None:
        """添加重要纪念日"""
        profile = await self.get_or_create(user_id, group_id, persona_id)

        date_entry = {"date": date, "description": description}
        if date_entry not in profile.important_dates:
            profile.important_dates.append(date_entry)
            meta = profile.get_field_meta("important_dates")
            meta.record_update(1.0, source="manual")
            profile.set_field_meta("important_dates", meta)
            await self._storage.save_user_profile(
                profile, group_id, persona_id=persona_id
            )
            logger.info(f"添加用户重要日期: {user_id} -> {date} ({description})")

    @profile_lock("user")
    async def add_taboo_topic(
        self,
        user_id: str,
        group_id: str,
        topic: str,
        persona_id: str = "default",
    ) -> None:
        """添加禁忌话题"""
        profile = await self.get_or_create(user_id, group_id, persona_id)

        if topic not in profile.taboo_topics:
            profile.taboo_topics.append(topic)
            meta = profile.get_field_meta("taboo_topics")
            meta.record_update(1.0, source="manual")
            profile.set_field_meta("taboo_topics", meta)
            await self._storage.save_user_profile(
                profile, group_id, persona_id=persona_id
            )
            logger.info(f"添加用户禁忌话题: {user_id} -> {topic}")

    @profile_lock("user")
    async def add_important_event(
        self,
        user_id: str,
        group_id: str,
        event: str,
        persona_id: str = "default",
    ) -> None:
        """添加历史重要事件"""
        profile = await self.get_or_create(user_id, group_id, persona_id)

        if event not in profile.important_events:
            profile.important_events.append(event)
            meta = profile.get_field_meta("important_events")
            meta.record_update(1.0, source="manual")
            profile.set_field_meta("important_events", meta)
            await self._storage.save_user_profile(
                profile, group_id, persona_id=persona_id
            )
            logger.info(f"添加用户重要事件: {user_id} -> {event}")

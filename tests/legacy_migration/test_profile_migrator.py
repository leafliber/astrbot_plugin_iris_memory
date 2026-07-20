"""profile_migrator 测试"""

import pytest

from iris_memory.legacy_migration.detector import LegacyDetection
from iris_memory.legacy_migration.profile_migrator import (
    DEFAULT_FAVORABILITY,
    map_legacy_persona,
    migrate_profiles,
)
from iris_memory.profile.models import UserProfile


def _detection(personas):
    return LegacyDetection(kv_keys={"user_personas": personas})


def _full_persona(**overrides):
    data = {
        "user_id": "u1",
        "display_name": "小明",
        "interests": {"编程": 0.9, "游戏": 0.7, "音乐": 0.8},
        "emotional_baseline": "平静",
        "preferred_reply_style": "brief",
        "topic_blacklist": ["政治", "宗教"],
        # 以下字段新版无对应，应被丢弃
        "trust_level": 0.8,
        "intimacy_level": 0.6,
        "personality_openness": 0.7,
        "hourly_distribution": [0.0] * 24,
        "habits": ["熬夜"],
    }
    data.update(overrides)
    return data


class TestMapLegacyPersona:
    """字段映射正确性"""

    def test_full_mapping(self):
        profile = map_legacy_persona("u1", _full_persona())
        assert profile.user_id == "u1"
        assert profile.user_name == "小明"
        # interests dict 按分数降序
        assert profile.interests == ["编程", "音乐", "游戏"]
        assert profile.emotional_baseline == "平静"
        assert profile.communication_style == "简洁"
        assert profile.taboo_topics == ["政治", "宗教"]
        assert profile.favorability == DEFAULT_FAVORABILITY == 50.0
        # 无对应字段不进入新画像
        assert not hasattr(profile, "trust_level")

    def test_reply_style_mapping(self):
        assert map_legacy_persona("u", {"preferred_reply_style": "brief"}).communication_style == "简洁"
        assert map_legacy_persona("u", {"preferred_reply_style": "detailed"}).communication_style == "详细"
        assert map_legacy_persona("u", {"preferred_reply_style": "default"}).communication_style == ""
        assert map_legacy_persona("u", {}).communication_style == ""

    def test_interests_list_form(self):
        profile = map_legacy_persona("u", {"interests": ["篮球", "足球"]})
        assert profile.interests == ["篮球", "足球"]

    def test_interests_truncated_to_10(self):
        interests = {f"兴趣{i}": 1.0 - i * 0.01 for i in range(20)}
        profile = map_legacy_persona("u", {"interests": interests})
        assert len(profile.interests) == 10

    def test_garbage_fields_safe(self):
        profile = map_legacy_persona(
            "u",
            {
                "interests": "not-a-dict",
                "topic_blacklist": "not-a-list",
                "display_name": 123,
                "emotional_baseline": None,
            },
        )
        assert profile.interests == []
        assert profile.taboo_topics == []
        assert profile.user_name == ""
        assert profile.emotional_baseline == ""


class TestMigrateProfiles:
    """迁移流程"""

    @pytest.mark.asyncio
    async def test_skipped_no_data(self, component_manager):
        stats = await migrate_profiles(LegacyDetection(), component_manager)
        assert stats["status"] == "skipped_no_data"

    @pytest.mark.asyncio
    async def test_bad_kv_value(self, component_manager):
        stats = await migrate_profiles(
            LegacyDetection(kv_keys={"user_personas": ["not", "a", "dict"]}),
            component_manager,
        )
        assert stats["status"] == "error"

    @pytest.mark.asyncio
    async def test_skipped_adapter_unavailable(
        self, component_manager, profile_storage
    ):
        profile_storage.is_available = False
        stats = await migrate_profiles(_detection({"u1": _full_persona()}), component_manager)
        assert stats["status"] == "skipped_adapter_unavailable"

    @pytest.mark.asyncio
    async def test_migrate_into_default_namespace(
        self, component_manager, profile_storage
    ):
        personas = {
            "u1": _full_persona(),
            "u2": _full_persona(user_id="u2", display_name="小红"),
            "u3": "not-a-dict",  # 垃圾条目 → 跳过
        }
        stats = await migrate_profiles(_detection(personas), component_manager)

        assert stats["status"] == "ok"
        assert stats["total"] == 3
        assert stats["imported"] == 2
        assert stats["skipped"] == 1
        assert stats["errors"] == 0

        # 迁移到 default 群 / default 人格命名空间
        profile = profile_storage.profiles.get(("u1", "default", "default"))
        assert profile is not None
        assert isinstance(profile, UserProfile)
        assert profile.user_name == "小明"
        assert profile.favorability == 50.0

    @pytest.mark.asyncio
    async def test_existing_profile_not_overwritten(
        self, component_manager, profile_storage
    ):
        existing = UserProfile(user_id="u1", user_name="新画像昵称", favorability=88.0)
        profile_storage.profiles[("u1", "default", "default")] = existing

        stats = await migrate_profiles(_detection({"u1": _full_persona()}), component_manager)
        assert stats["imported"] == 0
        assert stats["skipped"] == 1

        kept = profile_storage.profiles[("u1", "default", "default")]
        assert kept.user_name == "新画像昵称"
        assert kept.favorability == 88.0

    @pytest.mark.asyncio
    async def test_user_id_fallback_to_kv_key(
        self, component_manager, profile_storage
    ):
        personas = {"qq_12345": {"display_name": "无ID用户"}}
        stats = await migrate_profiles(_detection(personas), component_manager)
        assert stats["imported"] == 1
        profile = profile_storage.profiles.get(("qq_12345", "default", "default"))
        assert profile is not None
        assert profile.user_name == "无ID用户"

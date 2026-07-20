"""kv_migrator 测试"""

import pytest

from iris_memory.legacy_migration.detector import LegacyDetection
from iris_memory.legacy_migration.kv_migrator import (
    NEW_WHITELIST_KEY,
    migrate_kv,
)


class TestWhitelistMigration:
    """主动回复白名单迁移"""

    @pytest.mark.asyncio
    async def test_merge_with_existing(self, star):
        star.kv[NEW_WHITELIST_KEY] = ["g1"]
        detection = LegacyDetection(
            kv_keys={
                "proactive_reply_whitelist": {
                    "group_whitelist": ["g1", "g2", "g3"],
                    "group_whitelist_mode": True,
                }
            }
        )

        stats = await migrate_kv(detection, star)
        assert stats["status"] == "ok"
        assert stats["whitelist_total"] == 3
        assert stats["whitelist_migrated"] == 2  # g2、g3 为新增
        assert star.kv[NEW_WHITELIST_KEY] == ["g1", "g2", "g3"]

    @pytest.mark.asyncio
    async def test_write_when_no_existing(self, star):
        detection = LegacyDetection(
            kv_keys={
                "proactive_reply_whitelist": {
                    "group_whitelist": ["g2", "g1"],
                    "group_whitelist_mode": True,
                }
            }
        )

        stats = await migrate_kv(detection, star)
        assert stats["whitelist_migrated"] == 2
        assert star.kv[NEW_WHITELIST_KEY] == ["g1", "g2"]

    @pytest.mark.asyncio
    async def test_empty_groups_no_write(self, star):
        detection = LegacyDetection(
            kv_keys={
                "proactive_reply_whitelist": {
                    "group_whitelist": [],
                    "group_whitelist_mode": False,
                }
            }
        )

        stats = await migrate_kv(detection, star)
        assert stats["whitelist_total"] == 0
        assert stats["whitelist_migrated"] == 0
        assert NEW_WHITELIST_KEY not in star.kv

    @pytest.mark.asyncio
    async def test_no_duplicate_write_when_identical(self, star):
        star.kv[NEW_WHITELIST_KEY] = ["g1", "g2"]
        detection = LegacyDetection(
            kv_keys={
                "proactive_reply_whitelist": {
                    "group_whitelist": ["g2", "g1"],
                    "group_whitelist_mode": True,
                }
            }
        )

        stats = await migrate_kv(detection, star)
        assert stats["whitelist_migrated"] == 0
        assert star.kv[NEW_WHITELIST_KEY] == ["g1", "g2"]

    @pytest.mark.asyncio
    async def test_malformed_whitelist_ignored(self, star):
        detection = LegacyDetection(
            kv_keys={"proactive_reply_whitelist": ["not", "a", "dict"]}
        )
        stats = await migrate_kv(detection, star)
        assert stats["status"] == "ok"
        assert NEW_WHITELIST_KEY not in star.kv

    @pytest.mark.asyncio
    async def test_put_failure_reported(self, star):
        async def failing_put(key, value):
            raise RuntimeError("KV 后端故障")

        star.put_kv_data = failing_put  # type: ignore[assignment]
        detection = LegacyDetection(
            kv_keys={
                "proactive_reply_whitelist": {
                    "group_whitelist": ["g1"],
                    "group_whitelist_mode": True,
                }
            }
        )

        stats = await migrate_kv(detection, star)
        assert stats["status"] == "error"


class TestNotMigratedKeys:
    """只检测不迁移的旧 KV 键"""

    @pytest.mark.asyncio
    async def test_not_migrated_listed(self, star):
        detection = LegacyDetection(
            kv_keys={
                "sessions": {"s1": {}},
                "chat_history": {"g1": []},
                "group_activity": {"g1": {}},
                "user_personas": {"u1": {}},  # 由画像迁移器处理，不在此列出
                "proactive_reply_whitelist": {"group_whitelist": ["g1"]},
            }
        )

        stats = await migrate_kv(detection, star)
        assert set(stats["not_migrated"]) == {"sessions", "chat_history", "group_activity"}

    @pytest.mark.asyncio
    async def test_nothing_detected(self, star):
        stats = await migrate_kv(LegacyDetection(), star)
        assert stats["status"] == "ok"
        assert stats["not_migrated"] == []

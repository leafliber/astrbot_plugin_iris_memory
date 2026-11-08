"""用户画像 group_id 归一化与历史数据迁移测试"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot_plugin_iris_memory.iris_memory.profile.models import (
    UserProfile,
    profile_to_dict,
)
from astrbot_plugin_iris_memory.iris_memory.profile.storage import (
    ProfileStorage,
    effective_group_id,
)


class MemoryKV:
    """行为与 AstrBot KV 存储一致的内存实现。"""

    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})
        self.get_kv_data = AsyncMock(side_effect=self._get)
        self.put_kv_data = AsyncMock(side_effect=self._put)
        self.delete_kv_data = AsyncMock(side_effect=self._delete)

    async def _get(self, key, default=None):
        return self.data.get(key, default)

    async def _put(self, key, value):
        self.data[key] = value

    async def _delete(self, key):
        self.data.pop(key, None)


def _storage(data: dict | None = None) -> tuple[ProfileStorage, MemoryKV]:
    kv = MemoryKV(data)
    storage = ProfileStorage(MagicMock())
    storage._storage = kv
    storage._is_available = True
    return storage, kv


def _profile_dict(user_id: str, **fields) -> dict:
    data = profile_to_dict(UserProfile(user_id=user_id))
    data.update(fields)
    return data


@pytest.mark.parametrize(
    "raw,expected",
    [("", "default"), (None, "default"), ("  ", "default"), ("g1", "g1")],
)
def test_effective_group_id(raw, expected):
    assert effective_group_id(raw) == expected


@pytest.mark.asyncio
async def test_empty_group_and_default_share_one_profile():
    storage, kv = _storage({"persona_index": ["default"]})

    await storage.save_user_profile(UserProfile(user_id="u1"), group_id="")

    assert "user_profile:default:default:u1" in kv.data
    assert "user_profile:default::u1" not in kv.data
    assert await storage.get_user_profile("u1", "default") is not None
    assert await storage.get_user_profile("u1", "") is not None


@pytest.mark.asyncio
async def test_migration_merges_and_backs_up_source():
    old = _profile_dict("u1", occupation="工程师", interests=["a"], version=3)
    new = _profile_dict("u1", occupation="", interests=["b"], version=2)
    storage, kv = _storage(
        {
            "user_group_index:default": ["", "default"],
            "user_index:default:": ["u1"],
            "user_index:default:default": ["u1"],
            "user_profile:default::u1": old,
            "user_profile:default:default:u1": new,
        }
    )

    assert await storage.migrate_empty_group_keys() == 1

    merged = kv.data["user_profile:default:default:u1"]
    assert merged["occupation"] == "工程师"
    assert merged["interests"] == ["b", "a"]
    assert merged["version"] == 3
    assert kv.data["user_profile_backup:default:__empty__:u1"] == old
    assert "user_profile:default::u1" not in kv.data
    assert "user_index:default:" not in kv.data
    assert kv.data["user_group_index:default"] == ["default"]
    assert kv.data["user_index:default:default"] == ["u1"]


@pytest.mark.asyncio
async def test_migration_moves_profile_missing_from_default():
    old = _profile_dict("u2", occupation="学生")
    storage, kv = _storage(
        {
            "user_group_index:default": [""],
            "user_index:default:": ["u2"],
            "user_profile:default::u2": old,
        }
    )

    await storage.migrate_empty_group_keys()

    assert kv.data["user_profile:default:default:u2"]["occupation"] == "学生"
    assert kv.data["user_index:default:default"] == ["u2"]
    assert kv.data["user_group_index:default"] == ["default"]


@pytest.mark.asyncio
async def test_migration_is_idempotent():
    storage, kv = _storage(
        {
            "user_group_index:default": [""],
            "user_index:default:": ["u1"],
            "user_profile:default::u1": _profile_dict("u1"),
        }
    )

    assert await storage.migrate_empty_group_keys() == 1
    snapshot = dict(kv.data)
    assert await storage.migrate_empty_group_keys() == 0
    assert kv.data == snapshot


@pytest.mark.asyncio
async def test_listing_shows_each_user_once_with_stale_indexes():
    storage, _ = _storage(
        {
            "user_group_index:default": ["", "default"],
            "user_index:default:default": ["u1", "u1"],
            "user_profile:default:default:u1": _profile_dict("u1"),
        }
    )

    users = await storage.list_all_users()

    assert [user["user_id"] for user in users] == ["u1"]

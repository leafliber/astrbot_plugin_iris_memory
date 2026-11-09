"""用户身份映射测试"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.identity import (
    parse_user_id_aliases,
    resolve_user_id,
)
from astrbot_plugin_iris_memory.iris_memory.platform.generic import GenericAdapter
from astrbot_plugin_iris_memory.iris_memory.profile.models import (
    UserProfile,
    profile_to_dict,
)
from astrbot_plugin_iris_memory.iris_memory.profile.storage import ProfileStorage

CONFIG = "astrbot_plugin_iris_memory.iris_memory.config.get_config"
STORAGE_CONFIG = "astrbot_plugin_iris_memory.iris_memory.profile.storage.get_config"


def _config(aliases) -> Mock:
    config = Mock()
    config.get = Mock(
        side_effect=lambda key, default=None: {"identity.user_id_aliases": aliases}.get(
            key, default
        )
    )
    return config


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", {}),
        ("   ", {}),
        ("not json", {}),
        ("[1, 2]", {}),
        ('{"admin": "123"}', {"admin": "123"}),
        ('{" admin ": " 123 ", "": "x", "y": ""}', {"admin": "123"}),
        ({"admin": 123}, {"admin": "123"}),
        (None, {}),
    ],
)
def test_parse_user_id_aliases(raw, expected):
    assert parse_user_id_aliases(raw) == expected


def test_resolve_user_id_maps_only_listed_ids():
    with patch(CONFIG, return_value=_config('{"admin": "123"}')):
        assert resolve_user_id("admin") == "123"
        assert resolve_user_id("someone") == "someone"


def test_resolve_user_id_before_config_initialized():
    with patch(CONFIG, side_effect=RuntimeError("not initialized")):
        assert resolve_user_id("admin") == "admin"


def test_generic_adapter_applies_alias():
    event = Mock()
    event.message_obj.sender.user_id = "admin"

    with patch(CONFIG, return_value=_config('{"admin": "123"}')):
        assert GenericAdapter().get_user_id(event) == "123"


class MemoryKV:
    def __init__(self, data: dict):
        self.data = dict(data)
        self.get_kv_data = AsyncMock(side_effect=self._get)
        self.put_kv_data = AsyncMock(side_effect=self._put)
        self.delete_kv_data = AsyncMock(side_effect=self._delete)

    async def _get(self, key, default=None):
        return self.data.get(key, default)

    async def _put(self, key, value):
        self.data[key] = value

    async def _delete(self, key):
        self.data.pop(key, None)


def _storage(data: dict) -> tuple[ProfileStorage, MemoryKV]:
    kv = MemoryKV(data)
    storage = ProfileStorage(MagicMock())
    storage._storage = kv
    storage._is_available = True
    return storage, kv


def _profile(user_id: str, **fields) -> dict:
    data = profile_to_dict(UserProfile(user_id=user_id))
    data.update(fields)
    return data


@pytest.mark.asyncio
async def test_alias_migration_merges_into_target():
    source = _profile("admin", occupation="工程师")
    storage, kv = _storage(
        {
            "user_index:default:default": ["admin", "123"],
            "user_profile:default:default:admin": source,
            "user_profile:default:default:123": _profile("123", interests=["a"]),
        }
    )

    with patch(STORAGE_CONFIG, return_value=_config('{"admin": "123"}')):
        assert await storage.migrate_alias_profiles() == 1
        assert await storage.migrate_alias_profiles() == 0

    merged = kv.data["user_profile:default:default:123"]
    assert merged["user_id"] == "123"
    assert merged["occupation"] == "工程师"
    assert merged["interests"] == ["a"]
    assert kv.data["user_index:default:default"] == ["123"]
    assert "user_profile:default:default:admin" not in kv.data
    assert kv.data["user_profile_backup:default:default:admin"] == source


@pytest.mark.asyncio
async def test_alias_migration_creates_missing_target():
    storage, kv = _storage(
        {
            "user_index:default:default": ["admin"],
            "user_profile:default:default:admin": _profile("admin", occupation="x"),
        }
    )

    with patch(STORAGE_CONFIG, return_value=_config('{"admin": "123"}')):
        await storage.migrate_alias_profiles()

    assert kv.data["user_profile:default:default:123"]["user_id"] == "123"
    assert kv.data["user_index:default:default"] == ["123"]


@pytest.mark.asyncio
async def test_no_aliases_leaves_data_untouched():
    data = {"user_index:default:default": ["admin"]}
    storage, kv = _storage(data)

    with patch(STORAGE_CONFIG, return_value=_config("")):
        assert await storage.migrate_alias_profiles() == 0

    assert kv.data == data

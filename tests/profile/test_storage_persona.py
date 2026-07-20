"""Profile storage persona 隔离与读写对称性测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from iris_memory.profile.storage import ProfileStorage
from iris_memory.profile.models import GroupProfile, UserProfile


def _make_storage(isolation_on: bool):
    context = MagicMock()
    context.get_kv_data = AsyncMock(return_value=None)
    context.put_kv_data = AsyncMock()
    storage = ProfileStorage(context)
    storage._is_available = True
    storage._is_available = True
    # 让 get_config 在 storage 模块内返回可控配置
    cfg = Mock()
    cfg.get = Mock(
        side_effect=lambda key, default=None: {
            "isolation_config.enable_persona_isolation": isolation_on,
        }.get(key, default)
    )
    return storage, cfg


class TestPersonaNormalization:
    @pytest.mark.asyncio
    async def test_isolation_off_forces_default_even_if_persona_passed(self):
        storage, cfg = _make_storage(isolation_on=False)
        with pytest.MonkeyPatch().context() as m:
            m.setattr("iris_memory.profile.storage.get_config", lambda: cfg)
            await storage.save_group_profile(
                GroupProfile(group_id="g1"), persona_id="yuki"
            )
        # 隔离关闭时即便传 yuki，键也应固化到 default
        args = context_put_args(storage)
        assert args[0] == "group_profile:default:g1"

    @pytest.mark.asyncio
    async def test_isolation_on_uses_passed_persona(self):
        storage, cfg = _make_storage(isolation_on=True)
        with pytest.MonkeyPatch().context() as m:
            m.setattr("iris_memory.profile.storage.get_config", lambda: cfg)
            await storage.save_group_profile(
                GroupProfile(group_id="g1"), persona_id="yuki"
            )
        assert context_put_args(storage)[0] == "group_profile:yuki:g1"


def context_put_args(storage):
    return storage._storage.put_kv_data.call_args.args


class TestReadWriteSymmetry:
    """读写键必须对称：同 persona 下 save 与 get 用同一键"""

    @pytest.mark.asyncio
    async def test_group_profile_save_get_same_key(self):
        storage, cfg = _make_storage(isolation_on=True)
        with pytest.MonkeyPatch().context() as m:
            m.setattr("iris_memory.profile.storage.get_config", lambda: cfg)
            await storage.save_group_profile(
                GroupProfile(group_id="g1", group_name="群1"), persona_id="yuki"
            )
            await storage.get_group_profile("g1", "yuki")
        # get 用了和 save 相同的键
        get_key = storage._storage.get_kv_data.call_args.args[0]
        put_key = storage._storage.put_kv_data.call_args.args[0]
        assert get_key == put_key == "group_profile:yuki:g1"

    @pytest.mark.asyncio
    async def test_user_profile_save_get_same_key(self):
        storage, cfg = _make_storage(isolation_on=True)
        with pytest.MonkeyPatch().context() as m:
            m.setattr("iris_memory.profile.storage.get_config", lambda: cfg)
            await storage.save_user_profile(
                UserProfile(user_id="u1"), group_id="g1", persona_id="yuki"
            )
            await storage.get_user_profile("u1", "g1", "yuki")
        get_key = storage._storage.get_kv_data.call_args.args[0]
        put_key = storage._storage.put_kv_data.call_args.args[0]
        assert get_key == put_key == "user_profile:yuki:g1:u1"

    @pytest.mark.asyncio
    async def test_different_personas_use_different_keys(self):
        storage, cfg = _make_storage(isolation_on=True)
        with pytest.MonkeyPatch().context() as m:
            m.setattr("iris_memory.profile.storage.get_config", lambda: cfg)
            await storage.save_user_profile(
                UserProfile(user_id="u1"), group_id="g1", persona_id="yuki"
            )
            await storage.save_user_profile(
                UserProfile(user_id="u1"), group_id="g1", persona_id="aria"
            )
        keys = [c.args[0] for c in storage._storage.put_kv_data.call_args_list]
        assert "user_profile:yuki:g1:u1" in keys
        assert "user_profile:aria:g1:u1" in keys

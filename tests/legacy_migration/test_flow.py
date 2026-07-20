"""migrate_if_needed 端到端流程测试：备份、幂等、故障隔离、中止"""

import json
import sys
from unittest.mock import Mock

import pytest

import iris_memory.legacy_migration as legacy_migration
from iris_memory.config import init_config
from iris_memory.config.config import reset_config
from iris_memory.legacy_migration import (
    BACKUP_DIRNAME,
    KV_BACKUP_FILENAME,
    MIGRATION_DONE_KEY,
    l3_migrator,
    migrate_if_needed,
)

from .conftest import FakeUserConfig, make_legacy_kg_db


@pytest.fixture(autouse=True)
def _reset_iris_config():
    yield
    reset_config()


def _prepare_legacy_env(tmp_path, star):
    """构造完整的旧数据环境：chroma 目录 + KG 库 + 旧 KV + 旧配置"""
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"legacy chroma data")

    kg_db = make_legacy_kg_db(
        tmp_path / "knowledge_graph.db",
        nodes=[
            {"id": "n1", "name": "小明", "display_name": "小明", "node_type": "person"},
            {"id": "n2", "name": "北京", "display_name": "北京", "node_type": "location"},
        ],
        edges=[
            {"id": "e1", "source_id": "n1", "target_id": "n2", "relation_type": "lives_in"},
        ],
    )

    star.kv["user_personas"] = {
        "u1": {"user_id": "u1", "display_name": "小明", "interests": {"编程": 0.9}}
    }
    star.kv["proactive_reply_whitelist"] = {
        "group_whitelist": ["g100", "g200"],
        "group_whitelist_mode": True,
    }
    star.kv["sessions"] = {"s1": {"data": "x"}}

    raw_config = FakeUserConfig({"persona": {"enabled": False}})
    init_config(raw_config, tmp_path)
    return chroma, kg_db, raw_config


class TestFullFlow:
    @pytest.mark.asyncio
    async def test_full_migration(
        self, tmp_path, star, component_manager, monkeypatch
    ):
        chroma, kg_db, raw_config = _prepare_legacy_env(tmp_path, star)
        # 屏蔽真实 chromadb：走缺包跳过分支
        monkeypatch.setitem(sys.modules, "chromadb", None)

        await migrate_if_needed(Mock(), star, tmp_path, component_manager)

        # ── 完成标志 ──
        flag = star.kv.get(MIGRATION_DONE_KEY)
        assert flag is not None
        assert flag["status"] == "done"
        assert set(flag.keys()) >= {"l2", "l3", "profile", "kv", "config", "backup_dir"}

        # chromadb 缺失 → L2 跳过，但不影响其他迁移器
        assert flag["l2"]["status"] == "skipped_missing_chromadb"
        assert flag["l3"]["status"] == "ok"
        assert flag["profile"]["status"] == "ok"
        assert flag["config"]["status"] == "ok"

        # ── 备份创建 ──
        backup_dir = tmp_path / BACKUP_DIRNAME
        assert (backup_dir / "chroma" / "chroma.sqlite3").read_bytes() == b"legacy chroma data"
        assert (backup_dir / "knowledge_graph.db").exists()
        kv_backup = json.loads((backup_dir / KV_BACKUP_FILENAME).read_text(encoding="utf-8"))
        assert "user_personas" in kv_backup["kv"]
        assert "persona.enabled" in kv_backup["config_keys"]
        # 原始数据未删除
        assert chroma.exists() and kg_db.exists()

        # ── L3 已迁移 ──
        l3 = component_manager.get_component("l3_kg")
        assert len(l3.nodes) == 2
        assert len(l3.edges) == 1

        # ── 画像已迁移 ──
        profile_storage = component_manager.get_component("profile")
        assert ("u1", "default", "default") in profile_storage.profiles

        # ── 白名单已迁移 ──
        assert star.kv["iris_reply:whitelist"] == ["g100", "g200"]

        # ── 配置已迁移并持久化 ──
        assert raw_config["profile"]["enable"] is False
        assert raw_config.save_calls == 1

        # ── 等待后台组件被调用 ──
        assert component_manager.wait_calls

    @pytest.mark.asyncio
    async def test_no_legacy_data_writes_flag(self, tmp_path, star, component_manager):
        init_config(FakeUserConfig(), tmp_path)

        await migrate_if_needed(Mock(), star, tmp_path, component_manager)

        flag = star.kv.get(MIGRATION_DONE_KEY)
        assert flag is not None
        assert flag["status"] == "no_legacy_data"
        # 无数据时不创建备份目录
        assert not (tmp_path / BACKUP_DIRNAME).exists()


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_second_run_skips(
        self, tmp_path, star, component_manager, monkeypatch
    ):
        _prepare_legacy_env(tmp_path, star)
        monkeypatch.setitem(sys.modules, "chromadb", None)

        await migrate_if_needed(Mock(), star, tmp_path, component_manager)
        l3 = component_manager.get_component("l3_kg")
        nodes_after_first = len(l3.nodes)
        whitelist_after_first = list(star.kv["iris_reply:whitelist"])

        # 第二次运行：整体跳过，各存储无变化
        await migrate_if_needed(Mock(), star, tmp_path, component_manager)
        assert len(l3.nodes) == nodes_after_first
        assert star.kv["iris_reply:whitelist"] == whitelist_after_first

    @pytest.mark.asyncio
    async def test_existing_flag_short_circuits(
        self, tmp_path, star, component_manager
    ):
        star.kv[MIGRATION_DONE_KEY] = {"status": "done", "finished_at": "2025-01-01"}
        # 准备旧数据，验证有数据也不迁移
        _prepare_legacy_env(tmp_path, star)

        await migrate_if_needed(Mock(), star, tmp_path, component_manager)

        l3 = component_manager.get_component("l3_kg")
        assert l3.nodes == {}
        assert "iris_reply:whitelist" not in star.kv


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_single_migrator_failure_does_not_block_others(
        self, tmp_path, star, component_manager, monkeypatch
    ):
        _prepare_legacy_env(tmp_path, star)
        monkeypatch.setitem(sys.modules, "chromadb", None)

        async def exploding_migrate_l3(*args):
            raise RuntimeError("L3 迁移器爆炸")

        monkeypatch.setattr(l3_migrator, "migrate_l3", exploding_migrate_l3)

        await migrate_if_needed(Mock(), star, tmp_path, component_manager)

        flag = star.kv.get(MIGRATION_DONE_KEY)
        assert flag is not None
        assert flag["l3"]["status"] == "error"
        assert "爆炸" in flag["l3"]["error"]
        # 其他迁移器不受影响
        assert flag["profile"]["status"] == "ok"
        assert flag["kv"]["status"] == "ok"
        assert star.kv["iris_reply:whitelist"] == ["g100", "g200"]


class TestBackupFailure:
    @pytest.mark.asyncio
    async def test_backup_failure_aborts(
        self, tmp_path, star, component_manager, monkeypatch
    ):
        _prepare_legacy_env(tmp_path, star)
        monkeypatch.setitem(sys.modules, "chromadb", None)

        def exploding_copytree(*args, **kwargs):
            raise OSError("磁盘已满")

        monkeypatch.setattr(legacy_migration.shutil, "copytree", exploding_copytree)

        await migrate_if_needed(Mock(), star, tmp_path, component_manager)

        # 中止：不写标志（下回启动重试），各迁移器未执行
        assert MIGRATION_DONE_KEY not in star.kv
        l3 = component_manager.get_component("l3_kg")
        assert l3.nodes == {}
        assert "iris_reply:whitelist" not in star.kv
        # 原始数据完好
        assert (tmp_path / "chroma" / "chroma.sqlite3").exists()
        assert (tmp_path / "knowledge_graph.db").exists()


class TestNeverRaises:
    @pytest.mark.asyncio
    async def test_unexpected_error_swallowed(
        self, tmp_path, star, component_manager, monkeypatch
    ):
        init_config(FakeUserConfig(), tmp_path)

        async def exploding_get(key, default=None):
            raise RuntimeError("KV 后端完全故障")

        star.get_kv_data = exploding_get  # type: ignore[assignment]

        # 不应抛出
        await migrate_if_needed(Mock(), star, tmp_path, component_manager)

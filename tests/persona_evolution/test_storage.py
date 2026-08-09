"""人格自迭代存储层测试（真实 SQLite，不 mock）"""

import sqlite3
import time

import pytest

from iris_memory.persona_evolution import EvolutionJob
from iris_memory.persona_evolution.storage import (
    SCHEMA_VERSION,
    PersonaEvolutionStorage,
)

from .conftest import insert_sample


class TestSchema:
    """建表、WAL/busy_timeout/user_version 迁移框架"""

    def test_all_tables_created(self, storage):
        rows = storage._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        tables = {r["name"] for r in rows}
        assert {
            "evolution_jobs",
            "style_samples",
            "evolution_runs",
            "persona_revisions",
            "revision_samples",
        } <= tables

    def test_sample_indexes_created(self, storage):
        rows = storage._db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        indexes = {r["name"] for r in rows}
        assert {
            "idx_pe_samples_created",
            "idx_pe_samples_group_created",
            "idx_pe_samples_user_created",
            "idx_pe_samples_group_user_created",
        } <= indexes

    def test_wal_and_busy_timeout(self, storage):
        journal = storage._db.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal.lower() == "wal"
        busy = storage._db.execute("PRAGMA busy_timeout").fetchone()[0]
        assert int(busy) == 5000

    def test_user_version_after_init(self, storage):
        assert storage.get_schema_version() == SCHEMA_VERSION
        # 幂等：再次执行不改变版本
        storage.init_schema()
        assert storage.get_schema_version() == SCHEMA_VERSION

    def test_migration_failure_rolls_back(self, tmp_path, monkeypatch):
        """迁移 SQL 出错时回滚并抛出（由组件降级），user_version 不变"""
        from iris_memory.persona_evolution import storage as storage_mod

        monkeypatch.setitem(
            storage_mod._MIGRATIONS, 0, "THIS IS NOT VALID SQL AT ALL;"
        )
        s = PersonaEvolutionStorage(tmp_path / "fail.db")
        with pytest.raises(sqlite3.Error):
            s.init_schema()
        assert s.get_schema_version() == 0
        s.close()

    def test_reopen_existing_db(self, tmp_path):
        """重启后数据完整、schema 版本保留"""
        db = tmp_path / "reopen.db"
        s1 = PersonaEvolutionStorage(db)
        s1.init_schema()
        job_id = s1.create_job(EvolutionJob(persona_id="p1"))
        s1.close()

        s2 = PersonaEvolutionStorage(db)
        s2.init_schema()
        assert s2.get_schema_version() == SCHEMA_VERSION
        assert s2.get_job(job_id) is not None
        s2.close()


class TestJobCRUD:
    """Job 增删查改与 persona_id 唯一约束"""

    def test_create_and_get(self, storage):
        job = EvolutionJob(
            persona_id="assistant_v2",
            name="测试任务",
            source_group_ids=["g1", "g2"],
            source_user_ids=["u1"],
            protected_fragments=["你是 Iris"],
        )
        job_id = storage.create_job(job)
        loaded = storage.get_job(job_id)
        assert loaded is not None
        assert loaded.persona_id == "assistant_v2"
        assert loaded.name == "测试任务"
        assert loaded.source_group_ids == ["g1", "g2"]
        assert loaded.source_user_ids == ["u1"]
        assert loaded.protected_fragments == ["你是 Iris"]
        assert loaded.edit_mode == "managed_block"
        assert loaded.approval_mode == "auto"
        assert loaded.status == "active"
        assert loaded.trigger_sample_count == 100
        assert loaded.min_interval_hours == 24

    def test_get_by_persona(self, storage):
        storage.create_job(EvolutionJob(persona_id="p1"))
        assert storage.get_job_by_persona("p1") is not None
        assert storage.get_job_by_persona("missing") is None

    def test_persona_id_unique(self, storage):
        storage.create_job(EvolutionJob(persona_id="p1"))
        with pytest.raises(ValueError, match="persona_id"):
            storage.create_job(EvolutionJob(persona_id="p1"))

    def test_list_jobs(self, storage):
        storage.create_job(EvolutionJob(persona_id="p1"))
        storage.create_job(EvolutionJob(persona_id="p2"))
        assert len(storage.list_jobs()) == 2

    def test_update_job(self, storage):
        job_id = storage.create_job(EvolutionJob(persona_id="p1"))
        assert storage.update_job(
            job_id,
            {"status": "paused", "source_group_ids": ["g9"], "consecutive_failures": 2},
        )
        loaded = storage.get_job(job_id)
        assert loaded.status == "paused"
        assert loaded.source_group_ids == ["g9"]
        assert loaded.consecutive_failures == 2

    def test_update_job_field_whitelist(self, storage):
        job_id = storage.create_job(EvolutionJob(persona_id="p1"))
        with pytest.raises(ValueError, match="不允许更新"):
            storage.update_job(job_id, {"persona_id": "p2"})
        with pytest.raises(ValueError, match="不允许更新"):
            storage.update_job(job_id, {"id; DROP TABLE evolution_jobs": 1})


class TestScopeCount:
    """四种群/用户范围组合计数（空数组=不限）"""

    @pytest.fixture(autouse=True)
    def _seed(self, storage):
        # g1: u1×2, u2×1；g2: u1×1, u2×1
        insert_sample(storage, group_id="g1", user_id="u1", text="g1u1 第一条消息")
        insert_sample(storage, group_id="g1", user_id="u1", text="g1u1 第二条消息")
        insert_sample(storage, group_id="g1", user_id="u2", text="g1u2 的消息")
        insert_sample(storage, group_id="g2", user_id="u1", text="g2u1 的消息")
        insert_sample(storage, group_id="g2", user_id="u2", text="g2u2 的消息")

    def test_all_groups_all_users(self, storage):
        assert storage.count_samples() == 5
        assert storage.count_samples([], []) == 5

    def test_specific_groups_all_users(self, storage):
        assert storage.count_samples(group_ids=["g1"]) == 3
        assert storage.count_samples(group_ids=["g1", "g2"]) == 5

    def test_all_groups_specific_users(self, storage):
        assert storage.count_samples(user_ids=["u1"]) == 3
        assert storage.count_samples(user_ids=["u2"]) == 2

    def test_specific_groups_specific_users(self, storage):
        assert storage.count_samples(group_ids=["g1"], user_ids=["u1"]) == 2
        assert storage.count_samples(group_ids=["g2"], user_ids=["u2"]) == 1

    def test_since_id_incremental(self, storage):
        first_two = storage.count_samples()
        cursor = storage.get_latest_sample_id()
        insert_sample(storage, group_id="g3", user_id="u3", text="新增消息")
        assert storage.count_samples(since_id=cursor) == 1
        assert storage.count_samples() == first_two + 1


class TestSampleDedupe:
    """dedupe_hash 唯一约束"""

    def test_duplicate_dedupe_hash_ignored(self, storage):
        first = insert_sample(storage, text="重复内容", dedupe_hash="h1")
        second = insert_sample(storage, text="重复内容", dedupe_hash="h1")
        assert first is not None
        assert second is None
        assert storage.count_samples() == 1

    def test_distinct_hash_both_inserted(self, storage):
        assert insert_sample(storage, text="甲", dedupe_hash="h1") is not None
        assert insert_sample(storage, text="乙", dedupe_hash="h2") is not None
        assert storage.count_samples() == 2


class TestPrune:
    """语料保留淘汰（30 天 / 全局上限，最旧先删）"""

    def test_retention_days(self, storage):
        now = time.time()
        insert_sample(storage, text="老消息", created_at=now - 31 * 86400)
        insert_sample(storage, text="新消息", created_at=now)
        removed = storage.prune_samples(retention_days=30, max_count=20000)
        assert removed == 1
        rows = storage.fetch_samples()
        assert len(rows) == 1
        assert rows[0]["normalized_text"] == "新消息"

    def test_max_count_oldest_first(self, storage):
        now = time.time()
        for i in range(10):
            insert_sample(storage, text=f"消息{i}", created_at=now - 100 + i)
        removed = storage.prune_samples(retention_days=30, max_count=6)
        assert removed == 4
        rows = storage.fetch_samples()
        assert len(rows) == 6
        # 最旧的 0-3 被删除，保留 4-9
        assert rows[0]["normalized_text"] == "消息4"

    def test_prune_nulls_revision_sample_id(self, storage):
        """删除语料后 revision_samples.sample_id 置空但保留 hash"""
        now = time.time()
        sid = insert_sample(storage, text="老消息", created_at=now - 31 * 86400)
        assert sid is not None
        storage._db.execute(
            "INSERT INTO revision_samples (revision_id, sample_id, sample_hash)"
            " VALUES (1, ?, 'abc123')",
            (sid,),
        )
        storage._db.commit()
        storage.prune_samples(retention_days=30, max_count=20000)
        row = storage._db.execute(
            "SELECT * FROM revision_samples WHERE revision_id=1"
        ).fetchone()
        assert row["sample_id"] is None
        assert row["sample_hash"] == "abc123"

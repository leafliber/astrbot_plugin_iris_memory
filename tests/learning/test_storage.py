"""LearningStorage 新存储模型测试。"""

import sqlite3
import time

import pytest

from iris_memory.learning.storage import LearningStorage


class TestSchemaAndCommonTables:
    def test_schema_idempotent_and_empty_stats(self, storage):
        storage.init_schema()
        stats = storage.get_stats()
        assert stats["jargon"]["total"] == 0
        assert stats["jargon_candidate"]["total"] == 0

    def test_old_jargon_schema_fails_clearly_without_migration(self, tmp_path):
        path = tmp_path / "old.db"
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE jargon(id INTEGER PRIMARY KEY, term TEXT, count INTEGER)")
        db.commit()
        db.close()
        storage = LearningStorage(path)
        with pytest.raises(RuntimeError, match="不提供兼容迁移"):
            storage.init_schema()
        storage.close()

    def test_pair_and_pattern_flows(self, storage):
        pair = storage.insert_pair("g", "u", "你好", "你好呀", "m1")
        pattern = storage.insert_pattern("g", "chat", "你好呀", pair)
        assert storage.get_pending_pairs(10)[0]["id"] == pair
        assert storage.get_pending_patterns(10)[0]["id"] == pattern
        storage.update_status("few_shot", [pair], "approved")
        storage.update_status("expression_pattern", [pattern], "approved")
        assert storage.get_approved_few_shots("g", 5)
        assert storage.get_approved_patterns("g", 5)

    def test_status_validation(self, storage):
        pair = storage.insert_pair("g", "u", "问", "答")
        with pytest.raises(ValueError):
            storage.update_status("few_shot", [pair], "active")
        with pytest.raises(ValueError):
            storage.update_status("sqlite_master", [pair], "approved")

    def test_pattern_decay(self, storage):
        pattern = storage.insert_pattern("g", "chat", "旧表达")
        storage.update_status("expression_pattern", [pattern], "approved")
        storage._db.execute(
            "UPDATE expression_pattern SET created_at=? WHERE id=?",
            (time.time() - 30 * 86400, pattern),
        )
        storage._db.commit()
        assert storage.decay_patterns(15, 300) == 1


class TestCandidateStorage:
    def _record(self, storage, user="u1", now=None, term="绝绝子", message_hash="h1"):
        return storage.record_jargon_observations(
            "g", user, message_hash, f"这波{term}",
            [{"term": term, "left": ["波"], "right": ["<E>"]}],
            now or time.time(), 1800,
        )

    def test_record_and_snapshot(self, storage):
        now = time.time()
        assert self._record(storage, now=now) == 1
        snapshot = storage.get_jargon_candidate_snapshots(14, now + 1)[0]
        assert snapshot["term"] == "绝绝子"
        assert snapshot["message_count"] == 1
        assert snapshot["user_counts"] == {"u1": 1}
        assert snapshot["contexts"][0]["text"] == "这波绝绝子"

    def test_cooldown(self, storage):
        now = time.time()
        self._record(storage, now=now)
        assert self._record(storage, now=now + 60, message_hash="h2") == 0
        assert self._record(storage, now=now + 1801, message_hash="h3") == 1
        assert storage.get_jargon_candidate_snapshots(14)[0]["message_count"] == 2

    def test_management_list_folds_same_source_fragments(self, storage):
        now = time.time()
        observations = [
            {"term": "可以把钱给我", "left": ["<B>"], "right": ["<E>"]},
            {"term": "以把钱给我", "left": ["可"], "right": ["<E>"]},
            {"term": "把钱给我", "left": ["以"], "right": ["<E>"]},
        ]
        for index in range(6):
            storage.record_jargon_observations(
                "g", f"u{index}", "same-message", "可以把钱给我",
                observations, now + index, 1800,
            )
        storage.record_jargon_observations(
            "g", "rail-user", "rail-message", "高铁",
            [{"term": "高铁", "left": ["<B>"], "right": ["<E>"]}],
            now, 1800,
        )

        items, total = storage.query_jargon_candidate_clusters()

        assert total == 2
        phrase = next(item for item in items if item["term"] == "可以把钱给我")
        assert phrase["cluster_size"] == 3
        assert set(phrase["cluster_terms"]) == {"可以把钱给我", "以把钱给我", "把钱给我"}
        assert phrase["message_count"] == 6
        assert phrase["user_count"] == 6
        assert storage.get_jargon_candidate_cluster_stats() == {
            "total": 2, "by_status": {"collecting": 2},
        }

    def test_claim_and_promote(self, storage):
        self._record(storage)
        cid = storage.get_jargon_candidate_snapshots(14)[0]["id"]
        claimed = storage.claim_jargon_candidates([cid], "token", time.time())
        assert [r["id"] for r in claimed] == [cid]
        assert storage.apply_jargon_verdict(
            [cid], "token", "approve", "slang", "绝绝子", "非常棒",
            0.9, "多用户稳定使用", 8, aliases=["绝子"],
        )
        active = storage.get_active_jargon("g")
        assert active[0]["term"] == "绝绝子"
        assert active[0]["aliases"] == ["绝子"]
        assert active[0]["evidence_count"] == 8

    def test_review_token_cas(self, storage):
        self._record(storage)
        cid = storage.get_jargon_candidate_snapshots(14)[0]["id"]
        storage.claim_jargon_candidates([cid], "new", time.time())
        assert not storage.apply_jargon_verdict(
            [cid], "old", "approve", "slang", "绝绝子", "非常棒", 0.9, "", 6
        )

    def test_llm_daily_budget_and_interval(self, storage):
        now = time.time()
        assert storage.reserve_jargon_llm_call("2026-08-09", 3, 2, 3600, now)
        assert not storage.reserve_jargon_llm_call("2026-08-09", 3, 2, 3600, now + 60)
        assert storage.reserve_jargon_llm_call("2026-08-09", 2, 2, 3600, now + 3601)
        assert not storage.reserve_jargon_llm_call("2026-08-09", 1, 2, 3600, now + 7202)
        usage = storage.get_jargon_usage("2026-08-09")
        assert usage["call_count"] == 2 and usage["candidate_count"] == 5

    def test_maintenance(self, storage):
        old = time.time() - 70 * 86400
        self._record(storage, now=old)
        result = storage.maintain_jargon(time.time(), 14, 30, 60, 60, 3000)
        assert result["expired"] == 1


class TestFormalJargon:
    def test_manual_insert_update_and_status(self, storage):
        jid = storage.insert_jargon("g", "yyds", "永远的神", 0.9)
        row = storage.list_rows("jargon")[0]
        assert row["id"] == jid and row["category"] == "manual"
        assert storage.update_row("jargon", jid, {"status": "dormant"})
        assert storage.get_active_jargon("g") == []
        storage.set_jargon_status("g", "yyds", "active")
        assert storage.get_active_jargon("g")

    def test_list_order_by_evidence(self, storage):
        a = storage.insert_jargon("g", "词甲", "甲", 0.8)
        b = storage.insert_jargon("g", "词乙", "乙", 0.8)
        storage._db.execute("UPDATE jargon SET evidence_count=3 WHERE id=?", (a,))
        storage._db.execute("UPDATE jargon SET evidence_count=9 WHERE id=?", (b,))
        storage._db.commit()
        assert [r["term"] for r in storage.list_by_group("jargon", "g")] == ["词乙", "词甲"]

    def test_clear_group_cascades_candidates(self, storage):
        storage.insert_jargon("g", "yyds", "永远的神", 0.9)
        storage.record_jargon_observations(
            "g", "u", "h", "绝绝子", [{"term": "绝绝子", "left": [], "right": []}],
            time.time(), 0,
        )
        storage.clear_by_group("g")
        assert storage.get_stats()["jargon"]["total"] == 0
        assert storage.get_stats()["jargon_candidate"]["total"] == 0


class TestWebCrud:
    def test_list_count_update_delete_and_groups(self, storage):
        jid = storage.insert_jargon("g1", "yyds", "永远的神", 0.9)
        storage.insert_pair("g2", "u", "问", "答")
        assert storage.count_rows("jargon", "g1") == 1
        assert storage.update_row("jargon", jid, {"meaning": "永远滴神"})
        assert storage.list_rows("jargon", group_id="g1")[0]["meaning"] == "永远滴神"
        assert storage.list_groups() == ["g1", "g2"]
        assert storage.delete_rows("jargon", [jid]) == 1

    def test_invalid_table_and_field(self, storage):
        with pytest.raises(ValueError):
            storage.list_rows("sqlite_master")
        jid = storage.insert_jargon("g", "yyds", "永远的神", 0.9)
        with pytest.raises(ValueError):
            storage.update_row("jargon", jid, {"group_id": "other"})


class TestBackupAndDelete:
    def test_export_import_roundtrip_and_duplicate_skip(self, storage, tmp_path):
        pair_id = storage.insert_pair("g", "u", "问", "答", "m1")
        storage.insert_pattern("g", "问候", "你好呀", pair_id)
        storage.insert_jargon("g", "yyds", "永远的神", 0.9)
        storage.record_jargon_observations(
            "g", "u", "hash-1", "这波绝绝子",
            [{"term": "绝绝子", "left": ["波"], "right": []}],
            time.time(), 0,
        )

        exported = storage.export_all()
        restored = LearningStorage(tmp_path / "restored.db")
        restored.init_schema()
        try:
            stats = restored.import_from_data(exported)
            assert stats["error_count"] == 0
            assert stats["imported_count"] >= 5
            assert restored.count_rows("few_shot") == 1
            assert restored.count_rows("expression_pattern") == 1
            assert restored.count_rows("jargon") == 1
            assert restored.count_jargon_candidates() == 1
            pattern = restored.list_rows("expression_pattern")[0]
            assert pattern["source_pair_id"] == restored.list_rows("few_shot")[0]["id"]

            duplicate = restored.import_from_data(exported)
            assert duplicate["error_count"] == 0
            assert duplicate["skipped_count"] >= 5

            deleted = restored.delete_all()
            assert deleted["total"] >= 5
            assert restored.get_stats()["few_shot"]["total"] == 0
            assert restored.count_jargon_candidates() == 0
        finally:
            restored.close()

    def test_import_rejects_unknown_tables(self, storage):
        with pytest.raises(ValueError, match="未知表"):
            storage.import_from_data({"tables": {"sqlite_master": []}})

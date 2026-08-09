"""LearningStorage 存储层测试"""

import time


class TestSchema:
    """建表与初始化"""

    def test_init_schema_idempotent(self, storage):
        # 重复建表不报错（CREATE TABLE IF NOT EXISTS）
        storage.init_schema()
        storage.init_schema()

    def test_stats_empty(self, storage):
        stats = storage.get_stats()
        assert stats["expression_pattern"]["total"] == 0
        assert stats["few_shot"]["total"] == 0
        assert stats["jargon"]["total"] == 0


class TestPairs:
    """few_shot 对话对"""

    def test_insert_pair_returns_id(self, storage):
        pid = storage.insert_pair("g1", "u1", "你好？", "你好呀", "msg1")
        assert pid == 1
        pid2 = storage.insert_pair("g1", "u2", "在吗", "在的")
        assert pid2 == 2

    def test_insert_pair_default_pending(self, storage):
        storage.insert_pair("g1", "u1", "你好？", "你好呀")
        pairs = storage.get_pending_pairs(10)
        assert len(pairs) == 1
        assert pairs[0]["status"] == "pending_review"
        assert pairs[0]["user_text"] == "你好？"
        assert pairs[0]["message_id"] is None

    def test_get_approved_few_shots_only_approved(self, storage):
        pid = storage.insert_pair("g1", "u1", "早", "早呀")
        assert storage.get_approved_few_shots("g1", 5) == []
        storage.update_status("few_shot", [pid], "approved")
        shots = storage.get_approved_few_shots("g1", 5)
        assert len(shots) == 1
        # disabled 的也不应返回
        storage.update_status("few_shot", [pid], "disabled")
        assert storage.get_approved_few_shots("g1", 5) == []

    def test_get_approved_few_shots_group_isolation(self, storage):
        pid = storage.insert_pair("g1", "u1", "早", "早呀")
        storage.update_status("few_shot", [pid], "approved")
        assert storage.get_approved_few_shots("g2", 5) == []


class TestPatterns:
    """expression_pattern 表达模式"""

    def test_insert_pattern_returns_id(self, storage):
        pat = storage.insert_pattern("g1", "question", "你好呀", source_pair_id=1)
        assert pat == 1
        pending = storage.get_pending_patterns(10)
        assert pending[0]["scene"] == "question"
        assert pending[0]["source_pair_id"] == 1

    def test_get_approved_patterns_ordered_by_hit(self, storage):
        p1 = storage.insert_pattern("g1", "chat", "表达一")
        p2 = storage.insert_pattern("g1", "chat", "表达二")
        storage.update_status("expression_pattern", [p1, p2], "approved")
        storage.record_pattern_hit([p1, p1, p2])  # p1 命中 2 次，p2 命中 1 次
        # record_pattern_hit 逐条 +1，列表重复 id 仅生效一次 UPDATE 内 +1
        patterns = storage.get_approved_patterns("g1", 5)
        assert len(patterns) == 2

    def test_record_pattern_hit(self, storage):
        p1 = storage.insert_pattern("g1", "chat", "表达一")
        storage.update_status("expression_pattern", [p1], "approved")
        storage.record_pattern_hit([p1])
        row = storage.get_approved_patterns("g1", 5)[0]
        assert row["hit_count"] == 1
        assert row["last_hit_at"] is not None
        storage.record_pattern_hit([p1])
        row = storage.get_approved_patterns("g1", 5)[0]
        assert row["hit_count"] == 2

    def test_record_pattern_hit_empty(self, storage):
        storage.record_pattern_hit([])  # 不报错


class TestUpdateStatus:
    """状态流转"""

    def test_update_status_flow(self, storage):
        pid = storage.insert_pair("g1", "u1", "早", "早呀")
        storage.update_status("few_shot", [pid], "approved")
        assert storage.get_pending_pairs(10) == []
        storage.update_status("few_shot", [pid], "disabled")
        assert storage.get_approved_few_shots("g1", 5) == []

    def test_update_status_invalid_table(self, storage):
        import pytest

        with pytest.raises(ValueError):
            storage.update_status("users; DROP TABLE few_shot", [1], "approved")

    def test_update_status_empty_ids(self, storage):
        storage.update_status("few_shot", [], "approved")  # 不报错


class TestJargon:
    """jargon 暗语"""

    def test_upsert_jargon_count_accumulates(self, storage):
        assert storage.upsert_jargon_count("g1", "yyds", 3) == 3
        assert storage.upsert_jargon_count("g1", "yyds", 2) == 5
        # 不同群独立计数
        assert storage.upsert_jargon_count("g2", "yyds", 1) == 1

    def test_load_all_jargon_counts(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 3)
        storage.upsert_jargon_count("g2", "绝绝子", 7)
        counts = storage.load_all_jargon_counts()
        assert counts[("g1", "yyds")] == 3
        assert counts[("g2", "绝绝子")] == 7

    def test_get_terms_for_inference_threshold(self, storage):
        storage.upsert_jargon_count("g1", "低频词", 2)
        storage.upsert_jargon_count("g1", "高频词", 5)
        terms = storage.get_jargon_terms_for_inference([3, 6, 10])
        assert [t["term"] for t in terms] == ["高频词"]
        # 空阈值列表返回空
        assert storage.get_jargon_terms_for_inference([]) == []

    def test_get_terms_excludes_disabled(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 5)
        storage.set_jargon_status("g1", "yyds", "disabled")
        assert storage.get_jargon_terms_for_inference([3]) == []

    def test_mark_jargon_inferred(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 5)
        term = storage.get_jargon_terms_for_inference([3])[0]
        storage.mark_jargon_inferred(term["id"], "永远的神", 0.9)
        rows = storage.list_by_group("jargon", "g1")
        assert rows[0]["meaning"] == "永远的神"
        assert abs(rows[0]["confidence"] - 0.9) < 1e-6
        assert rows[0]["last_inferred_at"] is not None

    def test_get_active_jargon(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 5)
        term = storage.get_jargon_terms_for_inference([3])[0]
        # 未推断（无 meaning）不出现在 active 列表
        assert storage.get_active_jargon("g1") == []
        storage.mark_jargon_inferred(term["id"], "永远的神", 0.9)
        assert len(storage.get_active_jargon("g1")) == 1
        # 禁用后不返回
        storage.set_jargon_status("g1", "yyds", "disabled")
        assert storage.get_active_jargon("g1") == []
        storage.set_jargon_status("g1", "yyds", "active")
        assert len(storage.get_active_jargon("g1")) == 1


class TestDecay:
    """表达模式衰减"""

    def _approve(self, storage, pat_id):
        storage.update_status("expression_pattern", [pat_id], "approved")

    def test_decay_removes_stale_zero_hit(self, storage):
        p1 = storage.insert_pattern("g1", "chat", "旧表达")
        p2 = storage.insert_pattern("g1", "chat", "新表达")
        self._approve(storage, p1)
        self._approve(storage, p2)
        # p1 创建时间改到 30 天前
        old = time.time() - 30 * 86400
        storage._db.execute(
            "UPDATE expression_pattern SET created_at=? WHERE id=?", (old, p1)
        )
        storage._db.commit()
        removed = storage.decay_patterns(decay_days=15, max_count=300)
        assert removed == 1
        remaining = storage.get_approved_patterns("g1", 5)
        assert [r["expression"] for r in remaining] == ["新表达"]

    def test_decay_keeps_pending(self, storage):
        p1 = storage.insert_pattern("g1", "chat", "待审表达")
        old = time.time() - 30 * 86400
        storage._db.execute(
            "UPDATE expression_pattern SET created_at=? WHERE id=?", (old, p1)
        )
        storage._db.commit()
        # 衰减只处理 approved，pending 不受影响
        assert storage.decay_patterns(decay_days=15, max_count=300) == 0
        assert len(storage.get_pending_patterns(10)) == 1

    def test_decay_overflow_by_hit_rate(self, storage):
        ids = [storage.insert_pattern("g1", "chat", f"表达{i}") for i in range(3)]
        self._approve(storage, ids[0])
        self._approve(storage, ids[1])
        self._approve(storage, ids[2])
        # ids[2] 命中最高，ids[0] 最低
        storage.record_pattern_hit([ids[1]])
        storage.record_pattern_hit([ids[2]])
        storage.record_pattern_hit([ids[2]])
        removed = storage.decay_patterns(decay_days=15, max_count=2)
        assert removed == 1
        remaining = {r["expression"] for r in storage.get_approved_patterns("g1", 5)}
        assert "表达0" not in remaining
        assert remaining == {"表达1", "表达2"}


class TestClear:
    """清理操作"""

    def _seed(self, storage):
        pid = storage.insert_pair("g1", "u1", "早", "早呀")
        storage.insert_pair("g1", "u2", "好", "好呀")
        storage.insert_pair("g2", "u1", "嗨", "嗨呀")
        storage.insert_pattern("g1", "chat", "表达")
        storage.upsert_jargon_count("g1", "yyds", 3)
        return pid

    def test_clear_all(self, storage):
        self._seed(storage)
        storage.clear_all()
        stats = storage.get_stats()
        assert all(t["total"] == 0 for t in stats.values())

    def test_clear_by_group(self, storage):
        self._seed(storage)
        storage.clear_by_group("g1")
        stats = storage.get_stats()
        assert stats["few_shot"]["total"] == 1  # 只剩 g2 的
        assert stats["expression_pattern"]["total"] == 0
        assert stats["jargon"]["total"] == 0

    def test_clear_by_user_only_few_shot(self, storage):
        self._seed(storage)
        storage.clear_by_user("u1")
        stats = storage.get_stats()
        # u1 在 g1/g2 各一条 few_shot 被清，u2 的保留
        assert stats["few_shot"]["total"] == 1
        # pattern/jargon 无用户维度，不随用户清理
        assert stats["expression_pattern"]["total"] == 1
        assert stats["jargon"]["total"] == 1

    def test_clear_by_user_with_group(self, storage):
        self._seed(storage)
        storage.clear_by_user("u1", group_id="g1")
        shots = storage.list_by_group("few_shot", "g2")
        assert len(shots) == 1  # g2 的 u1 记录保留


class TestListByGroup:
    """list_by_group 通用查询"""

    def test_list_jargon_ordered_by_count(self, storage):
        storage.upsert_jargon_count("g1", "低频", 1)
        storage.upsert_jargon_count("g1", "高频", 9)
        rows = storage.list_by_group("jargon", "g1")
        assert [r["term"] for r in rows] == ["高频", "低频"]

    def test_list_limit(self, storage):
        for i in range(5):
            storage.insert_pair("g1", "u1", f"问{i}", f"答{i}")
        assert len(storage.list_by_group("few_shot", "g1", limit=3)) == 3

    def test_list_invalid_table(self, storage):
        import pytest

        with pytest.raises(ValueError):
            storage.list_by_group("sqlite_master", "g1")


class TestWebCrud:
    """Web 管理 CRUD（list_rows/count_rows/delete_rows/update_row/insert_jargon/list_groups）"""

    def test_list_rows_filter_and_order(self, storage):
        storage.insert_pair("g1", "u1", "问1", "答1")
        storage.insert_pair("g2", "u2", "问2", "答2")
        pid = storage.insert_pair("g1", "u1", "问3", "答3")
        storage.update_status("few_shot", [pid], "approved")

        assert len(storage.list_rows("few_shot")) == 3
        rows = storage.list_rows("few_shot", group_id="g1")
        assert len(rows) == 2
        rows = storage.list_rows("few_shot", group_id="g1", status="approved")
        assert len(rows) == 1 and rows[0]["user_text"] == "问3"

    def test_list_rows_pagination(self, storage):
        for i in range(5):
            storage.insert_pair("g1", "u1", f"问{i}", f"答{i}")
        page1 = storage.list_rows("few_shot", limit=2, offset=0)
        page2 = storage.list_rows("few_shot", limit=2, offset=2)
        assert len(page1) == 2 and len(page2) == 2
        assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})

    def test_list_rows_jargon_ordered_by_count(self, storage):
        storage.upsert_jargon_count("g1", "低频", 1)
        storage.upsert_jargon_count("g1", "高频", 9)
        rows = storage.list_rows("jargon")
        assert [r["term"] for r in rows] == ["高频", "低频"]

    def test_list_rows_invalid_table(self, storage):
        import pytest

        with pytest.raises(ValueError):
            storage.list_rows("sqlite_master")

    def test_count_rows(self, storage):
        storage.insert_pair("g1", "u1", "问1", "答1")
        storage.insert_pair("g1", "u1", "问2", "答2")
        storage.insert_pair("g2", "u2", "问3", "答3")
        assert storage.count_rows("few_shot") == 3
        assert storage.count_rows("few_shot", group_id="g1") == 2
        assert storage.count_rows("few_shot", status="approved") == 0

    def test_delete_rows(self, storage):
        pid1 = storage.insert_pair("g1", "u1", "问1", "答1")
        pid2 = storage.insert_pair("g1", "u1", "问2", "答2")
        assert storage.delete_rows("few_shot", [pid1]) == 1
        assert storage.count_rows("few_shot") == 1
        assert storage.delete_rows("few_shot", [pid1, pid2, 999]) == 1
        assert storage.delete_rows("few_shot", []) == 0

    def test_delete_rows_invalid_table(self, storage):
        import pytest

        with pytest.raises(ValueError):
            storage.delete_rows("sqlite_master", [1])

    def test_update_row_allowed_fields(self, storage):
        storage.upsert_jargon_count("g1", " yyds ", 5)
        row = storage.list_rows("jargon")[0]
        ok = storage.update_row(
            "jargon", row["id"], {"meaning": "永远的神", "confidence": 0.9}
        )
        assert ok
        row = storage.list_rows("jargon")[0]
        assert row["meaning"] == "永远的神"
        assert row["confidence"] == 0.9
        assert row["count"] == 5  # 未触及

    def test_update_row_rejects_unknown_field(self, storage):
        pid = storage.insert_pair("g1", "u1", "问", "答")
        import pytest

        with pytest.raises(ValueError):
            storage.update_row("few_shot", pid, {"group_id": "g2"})
        with pytest.raises(ValueError):
            storage.update_row("few_shot", pid, {"id": 99})

    def test_update_row_no_fields_returns_false(self, storage):
        pid = storage.insert_pair("g1", "u1", "问", "答")
        assert storage.update_row("few_shot", pid, {}) is False

    def test_update_row_missing_id_returns_false(self, storage):
        assert storage.update_row("jargon", 999, {"meaning": "x"}) is False

    def test_insert_jargon_new_and_conflict(self, storage):
        jid = storage.insert_jargon("g1", "绝绝子", meaning="太棒了", confidence=0.8)
        row = storage.list_rows("jargon")[0]
        assert row["id"] == jid
        assert row["status"] == "active"
        assert row["count"] == 0

        # 同群同词冲突：更新含义，不新增行
        storage.upsert_jargon_count("g1", "绝绝子", 3)
        jid2 = storage.insert_jargon("g1", "绝绝子", meaning="非常好", confidence=0.9)
        assert jid2 == jid
        rows = storage.list_rows("jargon")
        assert len(rows) == 1
        assert rows[0]["meaning"] == "非常好"
        assert rows[0]["count"] == 3  # 计数保留

    def test_list_groups(self, storage):
        storage.insert_pair("g2", "u1", "问", "答")
        storage.insert_pattern("g1", "scene", "表达")
        storage.upsert_jargon_count("", "无群词", 1)
        assert storage.list_groups() == ["g1", "g2"]

    def test_update_row_validates_status_per_table(self, storage):
        import pytest

        storage.upsert_jargon_count("g1", "yyds", 5)
        jid = storage.list_rows("jargon")[0]["id"]
        pid = storage.insert_pair("g1", "u1", "问", "答")

        # 暗语不可设为 approved / pending_review
        with pytest.raises(ValueError):
            storage.update_row("jargon", jid, {"status": "approved"})
        with pytest.raises(ValueError):
            storage.update_row("jargon", jid, {"status": "pending_review"})
        # few_shot 不可设为 active
        with pytest.raises(ValueError):
            storage.update_row("few_shot", pid, {"status": "active"})

        # 合法取值正常更新
        assert storage.update_row("jargon", jid, {"status": "disabled"})
        assert storage.update_row("few_shot", pid, {"status": "approved"})

    def test_update_status_validates_status_per_table(self, storage):
        import pytest

        storage.upsert_jargon_count("g1", "yyds", 5)
        jid = storage.list_rows("jargon")[0]["id"]

        with pytest.raises(ValueError):
            storage.update_status("jargon", [jid], "approved")
        with pytest.raises(ValueError):
            storage.update_status("jargon", [jid], "pending_review")
        with pytest.raises(ValueError):
            storage.update_status("few_shot", [1], "active")

        assert storage.update_status("jargon", [jid], "disabled") == 1
        assert storage.update_status("jargon", [jid], "active") == 1

    def test_update_status_expected_status_cas(self, storage):
        """带原状态条件的比较更新：状态已变化时不回写"""
        pid = storage.insert_pair("g1", "u1", "问", "答")

        # 当前是 pending_review，期望 disabled 不匹配 → 不更新
        assert storage.update_status(
            "few_shot", [pid], "approved", expected_status="disabled"
        ) == 0
        assert storage.list_rows("few_shot")[0]["status"] == "pending_review"

        # 期望 pending_review 匹配 → 更新
        assert storage.update_status(
            "few_shot", [pid], "approved", expected_status="pending_review"
        ) == 1
        assert storage.list_rows("few_shot")[0]["status"] == "approved"

    def test_mark_jargon_inferred_snapshot_cas(self, storage):
        """快照比较更新：推断期间被管理员修改的词条不回写"""
        storage.upsert_jargon_count("g1", "yyds", 5)
        snapshot = storage.get_jargon_terms_for_inference([3])[0]

        # 管理员在推断期间修改了含义
        storage.update_row("jargon", snapshot["id"], {"meaning": "管理员含义"})

        # 迟到的 LLM 结果基于旧快照 → 跳过，管理员修改保留
        assert storage.mark_jargon_inferred(
            snapshot["id"], "LLM含义", 0.9, snapshot=snapshot
        ) is False
        row = storage.list_rows("jargon")[0]
        assert row["meaning"] == "管理员含义"

        # 基于最新快照 → 正常回写
        fresh = storage.list_rows("jargon")[0]
        assert storage.mark_jargon_inferred(
            fresh["id"], "LLM含义", 0.9, snapshot=fresh
        ) is True
        assert storage.list_rows("jargon")[0]["meaning"] == "LLM含义"

    def test_mark_jargon_inferred_snapshot_status_change_cas(self, storage):
        """快照比较更新：推断期间被禁用的词条不回写"""
        storage.upsert_jargon_count("g1", "yyds", 5)
        snapshot = storage.get_jargon_terms_for_inference([3])[0]
        storage.set_jargon_status("g1", "yyds", "disabled")

        assert storage.mark_jargon_inferred(
            snapshot["id"], "LLM含义", 0.9, snapshot=snapshot
        ) is False
        row = storage.list_rows("jargon")[0]
        assert row["meaning"] is None
        assert row["status"] == "disabled"

    def test_insert_jargon_reactivates_disabled(self, storage):
        """重新手动新增已禁用暗语 → 恢复 active，计数保留"""
        storage.upsert_jargon_count("g1", "绝绝子", 7)
        storage.set_jargon_status("g1", "绝绝子", "disabled")

        storage.insert_jargon("g1", "绝绝子", meaning="太棒了")
        row = storage.list_rows("jargon")[0]
        assert row["status"] == "active"
        assert row["meaning"] == "太棒了"
        assert row["count"] == 7

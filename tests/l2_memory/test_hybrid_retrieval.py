"""L2 混合检索测试：FTS5 双写维护、关键词检索、RRF 融合与命中强化"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import numpy as np
import pytest

from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l2_memory.models import MemoryEntry, MemorySearchResult


@pytest.fixture
def mock_config():
    config = Mock()
    config.get = Mock(
        side_effect=lambda key, default=None: {
            "l2_memory.enable": True,
            "l2_timeout_ms": 2000,
            "l2_similarity_threshold": 0.90,
            "l2_memory.embedding_source": "provider",
            "l2_hybrid_rrf_k": 60,
            "l2_hybrid_keyword_pool_factor": 3,
            "l2_hybrid_enable_fts_debug": False,
        }.get(key, default)
    )
    config.data_dir = Path(tempfile.mkdtemp())
    return config


@pytest.fixture
def adapter():
    """真实 SQLite + FTS5，FAISS 索引为可控 mock"""
    a = L2MemoryAdapter()
    a._is_available = True
    a._persist_dir = Path(tempfile.mkdtemp())
    a._embedding_dimensions = 8

    mock_index = Mock()
    mock_index.ntotal = 0

    def fake_add(vectors, ids):
        mock_index.ntotal += len(ids)

    mock_index.add_with_ids = fake_add
    mock_index.remove_ids = Mock()
    mock_index.search = Mock(return_value=(np.array([[-1.0]]), np.array([[-1]])))
    mock_index.reconstruct = Mock(
        return_value=np.array([0.1] * 8, dtype=np.float32)
    )
    a._index = mock_index

    a._db = a._open_db(a._persist_dir / "metadata.db")
    a._free_list = []
    a._actual_embedding_model = "test-model"
    a._find_similar_unlocked = Mock(return_value=None)
    a._embed = AsyncMock(return_value=[[0.1] * 8])
    return a


def _fts_count(a) -> int:
    """真实 FTS 索引文档数（docsize 影子表，可检测孤儿/漏索引）"""
    with a._lock:
        return a._db.execute("SELECT COUNT(*) FROM memories_fts_docsize").fetchone()[0]


class TestFTSIndexMaintenance:
    def test_fts_table_created(self, adapter):
        assert adapter._fts_ok is True
        with adapter._lock:
            row = adapter._db.execute(
                "SELECT name FROM sqlite_master WHERE name='memories_fts'"
            ).fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_add_syncs_fts(self, adapter):
        mid = await adapter.add_memory("用户喜欢吃苹果和香蕉", metadata={})
        assert mid is not None
        assert _fts_count(adapter) == 1
        hits = adapter._search_with_keyword("吃苹果", None, 10, "default")
        assert len(hits) == 1 and hits[0].entry.id == mid

    @pytest.mark.asyncio
    async def test_delete_syncs_fts(self, adapter):
        mid = await adapter.add_memory("用户喜欢吃苹果和香蕉", metadata={})
        await adapter.delete_entries([mid])
        assert _fts_count(adapter) == 0
        assert adapter._search_with_keyword("吃苹果", None, 10, "default") == []

    @pytest.mark.asyncio
    async def test_update_content_syncs_fts(self, adapter):
        mid = await adapter.add_memory("旧内容关于咖啡的偏好", metadata={})
        updated = await adapter.batch_update_contents([(mid, "新内容关于茶叶的偏好")])
        assert updated == 1
        assert adapter._search_with_keyword("咖啡的", None, 10, "default") == []
        new_hits = adapter._search_with_keyword("茶叶的", None, 10, "default")
        assert len(new_hits) == 1 and new_hits[0].entry.id == mid

    @pytest.mark.asyncio
    async def test_delete_by_group_syncs_fts(self, adapter):
        await adapter.add_memory(
            "群组甲的专属记忆内容", metadata={"group_id": "g1"}
        )
        await adapter.add_memory(
            "群组乙的专属记忆内容", metadata={"group_id": "g2"}
        )
        assert _fts_count(adapter) == 2
        await adapter.delete_by_group("g1", "default")
        assert _fts_count(adapter) == 1

    @pytest.mark.asyncio
    async def test_delete_all_syncs_fts(self, adapter):
        await adapter.add_memory("第一条可以被检索的记忆", metadata={})
        await adapter.add_memory("第二条可以被检索的记忆", metadata={})
        assert _fts_count(adapter) == 2
        await adapter.delete_all()
        assert _fts_count(adapter) == 0

    @pytest.mark.asyncio
    async def test_startup_rebuild_clears_orphan_index(self, adapter):
        keep = await adapter.add_memory("启动重建后仍应命中的记忆甲", metadata={})
        gone = await adapter.add_memory("绕过同步被直接删除的记忆乙", metadata={})
        # 绕过双写直接删 memories -> FTS 残留孤儿索引项
        with adapter._lock:
            adapter._db.execute(
                "DELETE FROM memories WHERE memory_id = ?", (gone,)
            )
            adapter._db.commit()
        assert _fts_count(adapter) == 2

        adapter._ensure_fts_consistent()

        assert _fts_count(adapter) == 1
        assert any(
            r.entry.id == keep
            for r in adapter._search_with_keyword("仍应命中", None, 10, "default")
        )
        assert adapter._search_with_keyword("直接删除", None, 10, "default") == []

    @pytest.mark.asyncio
    async def test_startup_rebuild_indexes_missing_rows(self, adapter):
        # 模拟旧版本/手工写入：memories 有行但 FTS 未索引
        with adapter._lock:
            adapter._db.execute(
                "INSERT INTO memories "
                "(faiss_idx, memory_id, content, metadata, persona_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (999, "mem_legacy", "旧版本写入未建索引的记忆丙", "{}", "default"),
            )
            adapter._db.commit()
        assert adapter._search_with_keyword("未建索引", None, 10, "default") == []

        adapter._ensure_fts_consistent()

        hits = adapter._search_with_keyword("未建索引", None, 10, "default")
        assert len(hits) == 1 and hits[0].entry.id == "mem_legacy"


class TestKeywordSearch:
    def test_short_query_returns_none(self, adapter):
        assert adapter._build_fts_match_expr("你好") is None
        assert adapter._build_fts_match_expr("  ") is None

    def test_query_quotes_escaped(self, adapter):
        expr = adapter._build_fts_match_expr('包含"引号"的查询词')
        assert expr is not None
        assert '""' in expr

    @pytest.mark.asyncio
    async def test_keyword_search_group_isolation(self, adapter):
        await adapter.add_memory("隔离测试的群组记忆甲", metadata={"group_id": "g1"})
        await adapter.add_memory("隔离测试的群组记忆乙", metadata={"group_id": "g2"})
        hits = adapter._search_with_keyword("群组记忆甲", "g1", 10, "default")
        assert len(hits) == 1
        assert hits[0].entry.group_id == "g1"

    @pytest.mark.asyncio
    async def test_keyword_search_persona_isolation(self, adapter):
        await adapter.add_memory(
            "人格隔离的记忆内容样本", metadata={}, persona_id="yuki"
        )
        assert (
            adapter._search_with_keyword("人格隔离", None, 10, "default") == []
        )
        assert (
            len(adapter._search_with_keyword("人格隔离", None, 10, "yuki")) == 1
        )


class TestRRFFusion:
    def _res(self, mid: str) -> MemorySearchResult:
        return MemorySearchResult(
            entry=MemoryEntry(id=mid, content=mid), score=0.9, distance=0.1
        )

    def test_doc_in_both_paths_ranks_first(self):
        vec = [self._res("a"), self._res("b")]
        kw = [self._res("b"), self._res("c")]
        fused = L2MemoryAdapter._rrf_fuse(vec, kw, rrf_k=60, top_k=5)
        assert fused[0].entry.id == "b"
        assert len(fused) == 3

    def test_respects_top_k(self):
        vec = [self._res(f"m{i}") for i in range(10)]
        fused = L2MemoryAdapter._rrf_fuse(vec, [], rrf_k=60, top_k=3)
        assert len(fused) == 3

    def test_rrf_score_is_rank_based(self):
        fused = L2MemoryAdapter._rrf_fuse(
            [self._res("only_vec")], [], rrf_k=60, top_k=5
        )
        assert abs(fused[0].score - 1.0 / 61) < 1e-9


class TestHybridRetrieve:
    @pytest.mark.asyncio
    async def test_hybrid_combines_both_paths(self, adapter, mock_config):
        mem_a = await adapter.add_memory("小明最喜欢吃苹果和香蕉", metadata={})
        mem_b = await adapter.add_memory("今天天气很好大家一起出去玩", metadata={})

        # 向量路只命中 A（mock），关键词路用查询词的连续子串命中 B（真实 FTS）
        vec_result = MemorySearchResult(
            entry=(await adapter.get_entry_by_id(mem_a)), score=0.9, distance=0.1
        )
        adapter._search_with_vector = Mock(return_value=[vec_result])

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            results = await adapter.retrieve_hybrid(
                "天气很好", top_k=10, relevance_threshold=0.0
            )

        ids = [r.entry.id for r in results]
        assert mem_a in ids and mem_b in ids

    @pytest.mark.asyncio
    async def test_hybrid_vector_threshold_filters_vector_path(
        self, adapter, mock_config
    ):
        mem_a = await adapter.add_memory("阈值过滤测试的记忆甲", metadata={})
        low = MemorySearchResult(
            entry=(await adapter.get_entry_by_id(mem_a)), score=0.1, distance=0.9
        )
        adapter._search_with_vector = Mock(return_value=[low])

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            results = await adapter.retrieve_hybrid(
                "阈值过滤测试的记忆甲", top_k=10, relevance_threshold=0.5
            )

        # 向量路被阈值过滤，但关键词路仍命中 -> 结果仍包含该条
        assert any(r.entry.id == mem_a for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_keeps_keyword_results_when_vector_path_fails(
        self, adapter, mock_config
    ):
        mem_id = await adapter.add_memory("向量故障时仍可关键词降级命中", metadata={})
        adapter._embed = AsyncMock(side_effect=RuntimeError("embedding unavailable"))

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            results = await adapter.retrieve_hybrid("关键词降级", top_k=10)

        assert [r.entry.id for r in results] == [mem_id]

    @pytest.mark.asyncio
    async def test_hybrid_keeps_vector_results_when_keyword_path_fails(
        self, adapter, mock_config
    ):
        mem_id = await adapter.add_memory("关键词故障时仍可向量降级命中", metadata={})
        vector_hit = MemorySearchResult(
            entry=(await adapter.get_entry_by_id(mem_id)), score=0.9, distance=0.1
        )
        adapter._search_with_vector = Mock(return_value=[vector_hit])
        adapter._search_with_keyword = Mock(side_effect=RuntimeError("fts unavailable"))

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            results = await adapter.retrieve_hybrid("任意查询", top_k=10)

        assert [r.entry.id for r in results] == [mem_id]

    @pytest.mark.asyncio
    async def test_hybrid_unavailable_returns_empty(self):
        a = L2MemoryAdapter()
        a._is_available = False
        a._try_recover = AsyncMock(return_value=None)
        assert await a.retrieve_hybrid("任意查询") == []

    @pytest.mark.asyncio
    async def test_rebuild_fts_index_and_status(self, adapter):
        await adapter.add_memory("用于重建状态检查的记忆丁", metadata={})
        status = adapter.get_fts_status()
        assert status["available"] is True
        assert status["memory_rows"] == 1
        assert await adapter.rebuild_fts_index() is True
        assert adapter._search_with_keyword("状态检查", None, 10, "default") != []

    @pytest.mark.asyncio
    async def test_retrieve_debug_returns_both_lanes(self, adapter, mock_config):
        await adapter.add_memory("调试载荷中的苹果记忆", metadata={})
        adapter._search_with_vector = Mock(return_value=[])

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            payload = await adapter.retrieve_debug("苹果记忆", top_k=5)

        assert payload["fts"]["available"] is True
        assert payload["rrf_k"] == 60
        assert payload["relevance_threshold"] > 0
        # 向量路被 mock 为空，关键词路真实命中
        assert payload["vector"] == []
        assert len(payload["keyword"]) == 1
        assert len(payload["fused"]) == 1
        assert payload["fused"][0]["content"] == "调试载荷中的苹果记忆"


class TestHitReinforcement:
    def _config(self, reinforce: bool, step: float = 0.1):
        config = Mock()
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_enable_hit_reinforcement": reinforce,
                "l2_hit_reinforcement_step": step,
            }.get(key, default)
        )
        return config

    @pytest.mark.asyncio
    async def test_hit_boosts_importance_and_syncs_level(self, adapter):
        mid = await adapter.add_memory("被命中强化测试的记忆甲", metadata={})
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories SET metadata = json_set(metadata, "
                "'$.importance', 0.2, '$.importance_level', 'low') WHERE memory_id = ?",
                (mid,),
            )
            adapter._db.commit()

        with patch(
            "iris_memory.l2_memory.adapter.get_config",
            return_value=self._config(True, step=0.5),
        ):
            await adapter.batch_update_access([mid])

        with adapter._lock:
            meta = json.loads(
                adapter._db.execute(
                    "SELECT metadata FROM memories WHERE memory_id = ?", (mid,)
                ).fetchone()[0]
            )
        # 0.2 + 0.5·(1-0.2) = 0.6，跨过 0.35 → medium
        assert abs(meta["importance"] - 0.6) < 1e-6
        assert meta["importance_level"] == "medium"
        assert meta["access_count"] == 1

    @pytest.mark.asyncio
    async def test_reinforcement_is_asymptotic(self, adapter):
        mid = await adapter.add_memory("渐近自限测试的记忆乙", metadata={})
        with patch(
            "iris_memory.l2_memory.adapter.get_config",
            return_value=self._config(True, step=0.5),
        ):
            for _ in range(5):
                await adapter.batch_update_access([mid])

        with adapter._lock:
            meta = json.loads(
                adapter._db.execute(
                    "SELECT metadata FROM memories WHERE memory_id = ?", (mid,)
                ).fetchone()[0]
            )
        assert meta["importance"] < 1.0
        assert meta["importance_level"] == "high"

    @pytest.mark.asyncio
    async def test_reinforcement_disabled_keeps_importance(self, adapter):
        mid = await adapter.add_memory("关闭强化测试的记忆丙", metadata={})
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories SET metadata = json_set(metadata, "
                "'$.importance', 0.2) WHERE memory_id = ?",
                (mid,),
            )
            adapter._db.commit()

        with patch(
            "iris_memory.l2_memory.adapter.get_config",
            return_value=self._config(False),
        ):
            await adapter.batch_update_access([mid])

        with adapter._lock:
            meta = json.loads(
                adapter._db.execute(
                    "SELECT metadata FROM memories WHERE memory_id = ?", (mid,)
                ).fetchone()[0]
            )
        assert meta["importance"] == 0.2
        assert meta["access_count"] == 1


class TestArchiveAndRestore:
    def _config(self):
        config = Mock()
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_archive_retention_days": 30,
            }.get(key, default)
        )
        return config

    @pytest.mark.asyncio
    async def test_eviction_archives_instead_of_hard_delete(self, adapter):
        mid = await adapter.add_memory("将被梦境淘汰归档的记忆", metadata={})

        count = await adapter.evict_memories([mid])

        assert count == 1
        assert adapter._count_db() == 0
        assert _fts_count(adapter) == 0
        assert await adapter.get_archived_count() == 1
        archived = await adapter.list_archived_memories()
        assert archived[0]["id"] == mid
        assert archived[0]["has_vector"] is True

    @pytest.mark.asyncio
    async def test_eviction_rolls_back_when_faiss_delete_fails(self, adapter):
        mid = await adapter.add_memory("归档删除失败时必须完整回滚", metadata={})
        adapter._index.remove_ids = Mock(side_effect=RuntimeError("faiss failure"))

        count = await adapter.evict_memories([mid])

        assert count == 0
        assert adapter._count_db() == 1
        assert await adapter.get_archived_count() == 0
        assert _fts_count(adapter) == 1

    @pytest.mark.asyncio
    async def test_restore_returns_memory_to_live(self, adapter):
        mid = await adapter.add_memory("归档后需要恢复的记忆甲", metadata={"group_id": "g1"})
        await adapter.evict_memories([mid])
        assert adapter._search_with_keyword("需要恢复", None, 10, "default") == []

        ok = await adapter.restore_archived_memory(mid)

        assert ok is True
        assert adapter._count_db() == 1
        assert await adapter.get_archived_count() == 0
        hits = adapter._search_with_keyword("需要恢复", None, 10, "default")
        assert len(hits) == 1 and hits[0].entry.id == mid
        assert hits[0].entry.metadata.get("group_id") == "g1"

    @pytest.mark.asyncio
    async def test_restore_without_vector_reembeds(self, adapter):
        mid = await adapter.add_memory("丢失向量后恢复的记忆乙", metadata={})
        await adapter.evict_memories([mid])
        # 模拟向量损坏：清空归档向量
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories_archive SET vector = NULL WHERE memory_id = ?",
                (mid,),
            )
            adapter._db.commit()

        ok = await adapter.restore_archived_memory(mid)

        assert ok is True
        assert adapter._count_db() == 1

    @pytest.mark.asyncio
    async def test_restore_rejects_duplicate_live_id(self, adapter):
        mid = await adapter.add_memory("冲突恢复测试的记忆丙", metadata={})
        await adapter.evict_memories([mid])
        # 正式库出现同 ID 记忆（极端场景）
        await adapter.add_memory("占位的新记忆内容丁", metadata={}, skip_dedup=True)
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories SET memory_id = ? WHERE content = ?",
                (mid, "占位的新记忆内容丁"),
            )
            adapter._db.commit()

        assert await adapter.restore_archived_memory(mid) is False
        assert await adapter.get_archived_count() == 1

    @pytest.mark.asyncio
    async def test_purge_expired_archives(self, adapter):
        mid = await adapter.add_memory("超期清除测试的记忆戊", metadata={})
        await adapter.evict_memories([mid])
        # 把归档时间拨到 31 天前
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories_archive SET archived_at = '2020-01-01T00:00:00'"
            )
            adapter._db.commit()

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=self._config()
        ):
            removed = await adapter.purge_expired_archives()

        assert removed == 1
        assert await adapter.get_archived_count() == 0

    @pytest.mark.asyncio
    async def test_delete_archived_memory(self, adapter):
        mid = await adapter.add_memory("归档后彻底删除的记忆己", metadata={})
        await adapter.evict_memories([mid])

        assert await adapter.delete_archived_memory(mid) is True
        assert await adapter.get_archived_count() == 0
        assert await adapter.restore_archived_memory(mid) is False


class TestTTLExpiration:
    @pytest.mark.asyncio
    async def test_expired_memory_filtered_from_keyword_path(self, adapter):
        mid = await adapter.add_memory("带过期时间的临时记忆A", metadata={})
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories SET metadata = json_set(metadata, "
                "'$.expires_at', '2020-01-01T00:00:00') WHERE memory_id = ?",
                (mid,),
            )
            adapter._db.commit()

        assert adapter._search_with_keyword("临时记忆", None, 10, "default") == []

    @pytest.mark.asyncio
    async def test_non_expired_memory_still_hits(self, adapter):
        mid = await adapter.add_memory("未来过期的有效记忆B", metadata={})
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories SET metadata = json_set(metadata, "
                "'$.expires_at', '2999-01-01T00:00:00') WHERE memory_id = ?",
                (mid,),
            )
            adapter._db.commit()

        hits = adapter._search_with_keyword("有效记忆", None, 10, "default")
        assert len(hits) == 1 and hits[0].entry.id == mid

    @pytest.mark.asyncio
    async def test_expired_memory_filtered_from_vector_path(self, adapter):
        mid = await adapter.add_memory("向量路过期过滤的记忆C", metadata={})
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories SET metadata = json_set(metadata, "
                "'$.expires_at', '2020-01-01T00:00:00') WHERE memory_id = ?",
                (mid,),
            )
            adapter._db.commit()

        adapter._index.search = Mock(return_value=(np.array([[0.9]]), np.array([[0]])))
        results = adapter._search_with_vector(
            np.array([[0.1] * 8], dtype=np.float32), None, 10, "default"
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_purge_expired_memories(self, adapter):
        expired = await adapter.add_memory("已过期待清除的记忆D", metadata={})
        alive = await adapter.add_memory("未过期保留的记忆E", metadata={})
        with adapter._lock:
            adapter._db.execute(
                "UPDATE memories SET metadata = json_set(metadata, "
                "'$.expires_at', '2020-01-01T00:00:00') WHERE memory_id = ?",
                (expired,),
            )
            adapter._db.commit()

        removed = await adapter.purge_expired_memories()

        assert removed == 1
        assert adapter._count_db() == 1
        remaining = await adapter.get_entry_by_id(alive)
        assert remaining is not None


class TestGlobalScope:
    @pytest.mark.asyncio
    async def test_set_memory_scope(self, adapter):
        mid = await adapter.add_memory("作用域切换测试的记忆", metadata={"group_id": "g1"})

        assert await adapter.set_memory_scope(mid, "global") is True
        entry = await adapter.get_entry_by_id(mid)
        assert entry.metadata.get("scope") == "global"

        assert await adapter.set_memory_scope(mid, "group") is True
        entry = await adapter.get_entry_by_id(mid)
        assert "scope" not in entry.metadata

        assert await adapter.set_memory_scope(mid, "bogus") is False
        assert await adapter.set_memory_scope("mem_missing", "global") is False

    @pytest.mark.asyncio
    async def test_global_memory_hits_other_group_keyword_path(self, adapter):
        mid = await adapter.add_memory(
            "全局共享的机器人设定记忆", metadata={"group_id": "g1", "scope": "global"}
        )
        await adapter.add_memory(
            "其他群的普通记忆内容甲", metadata={"group_id": "g2"}
        )

        # 以 g2 身份检索：全局记忆应豁免群隔离
        hits = adapter._search_with_keyword("机器人设定", "g2", 10, "default")
        assert len(hits) == 1 and hits[0].entry.id == mid

    @pytest.mark.asyncio
    async def test_global_memory_hits_other_group_vector_path(self, adapter):
        mid = await adapter.add_memory(
            "全局向量豁免测试的记忆乙", metadata={"group_id": "g1", "scope": "global"}
        )
        adapter._index.search = Mock(return_value=(np.array([[0.9]]), np.array([[0]])))

        results = adapter._search_with_vector(
            np.array([[0.1] * 8], dtype=np.float32), "g2", 10, "default"
        )
        assert len(results) == 1 and results[0].entry.id == mid

    @pytest.mark.asyncio
    async def test_group_clear_skips_global(self, adapter):
        await adapter.add_memory(
            "群清理解除外的全局记忆丙", metadata={"group_id": "g1", "scope": "global"}
        )
        await adapter.add_memory(
            "群清理命中的普通记忆丁", metadata={"group_id": "g1"}
        )

        removed = await adapter.delete_by_group("g1", "default")

        assert removed == 1
        assert adapter._count_db() == 1

    @pytest.mark.asyncio
    async def test_delete_all_removes_global(self, adapter):
        await adapter.add_memory(
            "全清也要删除的全局记忆戊", metadata={"group_id": "g1", "scope": "global"}
        )

        removed = await adapter.delete_all()

        assert removed == 1
        assert adapter._count_db() == 0

    @pytest.mark.asyncio
    async def test_fusion_demotes_global(self, adapter):
        def make(mid: str, scope: str) -> MemorySearchResult:
            return MemorySearchResult(
                entry=MemoryEntry(
                    id=mid, content=mid, metadata={"scope": scope}
                ),
                score=0.03,
                distance=0.97,
            )

        config = Mock()
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_hybrid_global_scope_factor": 0.5,
            }.get(key, default)
        )

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=config
        ):
            demoted = adapter._apply_global_demotion(
                [make("mem_g", "global"), make("mem_l", "group")]
            )

        # 0.03 × 0.5 = 0.015 < 0.03，全局记忆被降到群记忆之后
        assert demoted[0].entry.id == "mem_l"
        assert demoted[1].entry.id == "mem_g"
        assert abs(demoted[1].score - 0.015) < 1e-9

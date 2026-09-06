"""L2 持久化韧性回归测试。

覆盖以下修复的回归保护：
- FAISS 索引原子写（tmp + os.replace，不留半截文件）
- 索引损坏后的启动恢复（改名保留 + 从 SQLite 对账重建）
- 索引/DB 脱同步时的 ID 集对账重建（增量修复 + 嵌入重试退避）
- checkpoint 任务强引用与 _checkpointing 复位
- 嵌入模型迁移失败时备份保留（不删唯一全量副本）
- 周期索引审计任务：运行期自愈脱同步、shutdown 干净取消
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.config import init_config
from astrbot_plugin_iris_memory.iris_memory.config.config import reset_config
from astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter import L2MemoryAdapter

DIM = 8


@pytest.fixture(autouse=True)
def _reset_iris_config():
    yield
    reset_config()


@pytest.fixture
def adapter(tmp_path: Path) -> L2MemoryAdapter:
    """带真实 SQLite 与真实 FAISS 的最小可用适配器。"""
    from astrbot_plugin_iris_memory.iris_memory.config import get_config

    init_config(
        {
            "l1_buffer": {"enable": True},
            "l2_memory": {
                "enable": True,
                "embedding_source": "provider",
            },
        },
        tmp_path,
    )
    # checkpoint 阈值是 hidden 键，走隐藏配置通道
    get_config().set_hidden("l2_checkpoint_writes", 3)
    adapter = L2MemoryAdapter()
    adapter._is_available = True
    adapter._persist_dir = tmp_path / "faiss" / "memory_default"
    adapter._persist_dir.mkdir(parents=True)
    adapter._embedding_dimensions = DIM
    adapter._actual_embedding_model = "test-model"
    adapter._db = adapter._open_db(adapter._persist_dir / "metadata.db")
    # _embed 默认返回固定向量（各测试可再覆盖）
    adapter._embed = AsyncMock(
        side_effect=lambda texts: [[0.1] * DIM for _ in texts]
    )
    # 嵌入重试退避置 0，避免重试路径的真实 sleep 拖慢测试
    adapter._embed_retry_backoff = 0.0
    return adapter


def _seed_db(adapter: L2MemoryAdapter, ids: list[int]) -> None:
    """直接向 SQLite 插入 memories 行（绕过 FAISS 写入）。"""
    with adapter._lock:
        for faiss_idx in ids:
            adapter._db.execute(
                "INSERT OR REPLACE INTO memories"
                " (faiss_idx, memory_id, content, metadata, persona_id)"
                " VALUES (?, ?, ?, ?, 'default')",
                (faiss_idx, f"mem_{faiss_idx:04d}", f"内容 {faiss_idx}", "{}"),
            )
        adapter._db.commit()


class TestAtomicIndexWrite:
    def test_write_and_read_back_roundtrip(self, adapter: L2MemoryAdapter):
        import faiss
        import numpy as np

        index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
        index.add_with_ids(
            np.array([[0.1] * DIM, [0.2] * DIM], dtype=np.float32),
            np.array([1, 2], dtype=np.int64),
        )
        path = adapter._persist_dir / "index.faiss"
        adapter._write_index_atomic(index, path)

        loaded = faiss.read_index(str(path))
        assert loaded.ntotal == 2
        # 不残留临时文件
        leftovers = list(adapter._persist_dir.glob("index.faiss.*.tmp"))
        assert leftovers == []

    def test_failed_write_keeps_old_file_intact(self, adapter: L2MemoryAdapter):
        import faiss
        import numpy as np

        index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
        index.add_with_ids(
            np.array([[0.1] * DIM], dtype=np.float32),
            np.array([1], dtype=np.int64),
        )
        path = adapter._persist_dir / "index.faiss"
        adapter._write_index_atomic(index, path)

        with patch(
            "faiss.write_index", side_effect=OSError("磁盘已满")
        ):
            with pytest.raises(OSError):
                adapter._write_index_atomic(index, path)

        # 旧文件仍是完整可读的旧内容
        loaded = faiss.read_index(str(path))
        assert loaded.ntotal == 1
        assert list(adapter._persist_dir.glob("index.faiss.*.tmp")) == []


class TestCorruptIndexRecovery:
    @pytest.mark.asyncio
    async def test_corrupt_index_renamed_and_rebuilt_from_db(
        self, adapter: L2MemoryAdapter
    ):
        _seed_db(adapter, [1, 2, 3])
        # 写入截断的伪索引文件
        (adapter._persist_dir / "index.faiss").write_bytes(b"\x00garbage")

        await adapter._load_existing(0)

        # 损坏文件保留现场
        assert (adapter._persist_dir / "index.faiss.corrupt").exists()
        # 索引已按 DB 行重建
        assert adapter._index is not None
        assert adapter._index.ntotal == 3

    @pytest.mark.asyncio
    async def test_valid_index_with_drift_reconciled(self, adapter: L2MemoryAdapter):
        import faiss
        import numpy as np

        _seed_db(adapter, [1, 2, 3])
        # 索引只含 1、2，缺 3（模拟 checkpoint 丢失窗口）
        index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
        index.add_with_ids(
            np.array([[0.1] * DIM, [0.2] * DIM], dtype=np.float32),
            np.array([1, 2], dtype=np.int64),
        )
        faiss.write_index(index, str(adapter._persist_dir / "index.faiss"))

        await adapter._load_existing(0)

        assert adapter._index.ntotal == 3
        # 增量修复：只重嵌入缺失的第 3 行，不触碰存量向量
        embedded = [t for call in adapter._embed.await_args_list for t in call.args[0]]
        assert embedded == ["内容 3"]
        # 修复结果已原子落盘
        on_disk = faiss.read_index(str(adapter._persist_dir / "index.faiss"))
        assert on_disk.ntotal == 3

    @pytest.mark.asyncio
    async def test_index_with_stale_extra_vectors_reconciled(
        self, adapter: L2MemoryAdapter
    ):
        import faiss
        import numpy as np

        _seed_db(adapter, [1])
        # 索引含已删除的 99（模拟删除已 commit、索引未 checkpoint）
        index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
        index.add_with_ids(
            np.array([[0.1] * DIM, [0.2] * DIM], dtype=np.float32),
            np.array([1, 99], dtype=np.int64),
        )
        faiss.write_index(index, str(adapter._persist_dir / "index.faiss"))

        await adapter._load_existing(0)

        assert adapter._index.ntotal == 1
        # 摘除脏向量无需嵌入
        adapter._embed.assert_not_awaited()
        on_disk = faiss.read_index(str(adapter._persist_dir / "index.faiss"))
        assert on_disk.ntotal == 1

    @pytest.mark.asyncio
    async def test_consistent_index_untouched(self, adapter: L2MemoryAdapter):
        import faiss
        import numpy as np

        _seed_db(adapter, [1, 2])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
        index.add_with_ids(
            np.array([[0.1] * DIM, [0.2] * DIM], dtype=np.float32),
            np.array([1, 2], dtype=np.int64),
        )
        faiss.write_index(index, str(adapter._persist_dir / "index.faiss"))
        embed = adapter._embed

        await adapter._load_existing(0)

        assert adapter._index.ntotal == 2
        embed.assert_not_awaited()


class TestReconcileRetryAndRuntimeHeal:
    """启动对账失败后，适配器保持可用并在稍后重试时自愈。"""

    @pytest.mark.asyncio
    async def test_transient_embed_failure_retried_then_repaired(
        self, adapter: L2MemoryAdapter
    ):
        _seed_db(adapter, [1, 2, 3])
        # 无索引文件：_open_storage 建空索引，全部行进入缺失集
        calls = {"n": 0}

        async def flaky(texts):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("provider 503")
            return [[0.1] * DIM for _ in texts]

        adapter._embed = AsyncMock(side_effect=flaky)

        await adapter._load_existing(0)

        # 前两次失败经退避重试后成功，索引完整重建
        assert adapter._index.ntotal == 3
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_reconcile_failure_keeps_partial_progress(
        self, adapter: L2MemoryAdapter
    ):
        import faiss

        _seed_db(adapter, [1, 2, 3])
        calls = {"n": 0}

        async def provider_down(texts):
            # 前 3 次调用全部失败：单批重试耗尽，对账以失败告终
            calls["n"] += 1
            if calls["n"] <= 3:
                raise RuntimeError("provider down")
            return [[0.1] * DIM for _ in texts]

        adapter._embed = AsyncMock(side_effect=provider_down)

        # 对账失败但不得抛出（适配器保持可用，检索退化为漏召回）
        await adapter._load_existing(0)
        assert adapter._index.ntotal == 0

        # Provider 恢复后再次对账（等价于周期审计触发）：全量补回
        adapter._embed = AsyncMock(
            side_effect=lambda texts: [[0.1] * DIM for _ in texts]
        )
        await adapter._reconcile_index_with_db()

        assert adapter._index.ntotal == 3
        on_disk = faiss.read_index(str(adapter._persist_dir / "index.faiss"))
        assert on_disk.ntotal == 3

    @pytest.mark.asyncio
    async def test_late_write_race_does_not_duplicate_ids(
        self, adapter: L2MemoryAdapter
    ):
        import numpy as np

        _seed_db(adapter, [1, 2])
        adapter._index = adapter._create_index(DIM)
        rows = [(1, "内容 1"), (2, "内容 2")]

        # 模拟：重嵌入期间槽位 1 已被其他写入路径补上
        adapter._index.add_with_ids(
            np.array([[0.5] * DIM], dtype=np.float32), np.array([1], dtype=np.int64)
        )

        added = await asyncio.to_thread(
            adapter._add_missing_vectors_locked,
            rows,
            [[0.1] * DIM, [0.2] * DIM],
        )

        assert added == 1
        assert adapter._index.ntotal == 2


class TestIndexAuditLifecycle:
    """周期审计任务的生命周期与自愈行为。"""

    @pytest.mark.asyncio
    async def test_audit_disabled_by_config(self, adapter: L2MemoryAdapter):
        from astrbot_plugin_iris_memory.iris_memory.config import get_config

        get_config().set_hidden("l2_index_audit_interval_sec", 0)
        adapter._start_index_audit()

        assert adapter._audit_task is None

    @pytest.mark.asyncio
    async def test_audit_task_started_then_cancelled_on_shutdown(
        self, adapter: L2MemoryAdapter
    ):
        from astrbot_plugin_iris_memory.iris_memory.config import get_config

        get_config().set_hidden("l2_index_audit_interval_sec", 60)
        adapter._index = adapter._create_index(DIM)

        adapter._start_index_audit()
        task = adapter._audit_task
        assert task is not None and not task.done()

        # 幂等：重复启动不产生第二个任务
        adapter._start_index_audit()
        assert adapter._audit_task is task

        await adapter.shutdown()

        assert adapter._audit_task is None
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_audit_loop_repairs_runtime_drift(self, adapter: L2MemoryAdapter):
        """运行期脱同步（如 checkpoint 写失败）由审计轮次自动修复。"""
        import faiss
        import numpy as np

        from astrbot_plugin_iris_memory.iris_memory.config import get_config

        _seed_db(adapter, [1, 2])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
        index.add_with_ids(
            np.array([[0.1] * DIM], dtype=np.float32), np.array([1], dtype=np.int64)
        )
        faiss.write_index(index, str(adapter._persist_dir / "index.faiss"))
        adapter._index = index

        # 快速走完一轮审计（跳过循环 sleep，直接执行单轮对账逻辑）
        get_config().set_hidden("l2_index_audit_interval_sec", 60)
        await adapter._reconcile_index_with_db()

        assert adapter._index.ntotal == 2
        embedded = [t for call in adapter._embed.await_args_list for t in call.args[0]]
        assert embedded == ["内容 2"]


class TestCheckpointTaskLifecycle:
    @pytest.mark.asyncio
    async def test_checkpoint_task_referenced_and_flag_resets(
        self, adapter: L2MemoryAdapter
    ):
        adapter._index = adapter._create_index(DIM)
        adapter._dirty = True
        # 配置阈值为 3
        adapter._pending_writes = 0
        for _ in range(3):
            adapter._mark_dirty()

        task = adapter._checkpoint_task
        assert task is not None, "checkpoint 任务必须保存强引用"
        await asyncio.wait_for(task, timeout=5)

        assert adapter._checkpointing is False
        assert adapter._checkpoint_task is None or adapter._checkpoint_task.done()
        assert (adapter._persist_dir / "index.faiss").exists()

    @pytest.mark.asyncio
    async def test_stale_done_callback_does_not_reset_new_checkpoint(
        self, adapter: L2MemoryAdapter
    ):
        loop = asyncio.get_running_loop()
        old_task = loop.create_future()
        old_task.set_result(None)
        new_task = loop.create_future()
        adapter._checkpoint_task = new_task
        adapter._checkpointing = True

        adapter._on_checkpoint_done(old_task)

        assert adapter._checkpointing is True
        assert adapter._checkpoint_task is new_task
        new_task.cancel()

    @pytest.mark.asyncio
    async def test_checkpoint_exception_surfaces_in_log_and_flag_resets(
        self, adapter: L2MemoryAdapter
    ):
        adapter._index = adapter._create_index(DIM)
        adapter._dirty = True
        with patch.object(
            adapter,
            "_write_index_atomic",
            side_effect=OSError("写盘失败"),
        ):
            adapter._pending_writes = 0
            for _ in range(3):
                adapter._mark_dirty()
            task = adapter._checkpoint_task
            assert task is not None
            await asyncio.wait_for(task, timeout=5)

        assert adapter._checkpointing is False


class TestMigrationBackupPreservation:
    @pytest.mark.asyncio
    async def test_import_failure_keeps_backups(self, adapter: L2MemoryAdapter):
        _seed_db(adapter, [1, 2, 3])
        adapter._save_meta()

        with patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.io.MemoryImporter.import_from_file",
            new=AsyncMock(side_effect=RuntimeError("Embedding Provider 超时")),
        ):
            ok = await adapter._migrate_on_model_change("new-model", DIM)

        assert ok is False
        backup_dir = adapter._persist_dir.parent / "migration_backup"
        backups = list(backup_dir.glob("*_migration_backup.json"))
        archives = list(backup_dir.glob("*_migration_archives.db"))
        assert backups, "迁移失败时全量备份必须保留"
        assert archives

    @pytest.mark.asyncio
    async def test_successful_migration_cleans_backups(self, adapter: L2MemoryAdapter):
        _seed_db(adapter, [1, 2])
        adapter._save_meta()

        ok = await adapter._migrate_on_model_change("new-model", DIM)

        assert ok is True
        backup_dir = adapter._persist_dir.parent / "migration_backup"
        assert list(backup_dir.glob("*_migration_backup.json")) == []
        assert list(backup_dir.glob("*_migration_archives.db")) == []


class TestDbFetchallDiscipline:
    @pytest.mark.asyncio
    async def test_get_all_entries_returns_rows(self, adapter: L2MemoryAdapter):
        _seed_db(adapter, [1, 2, 5])
        entries = await adapter.get_all_entries()
        assert sorted(e.id for e in entries) == ["mem_0001", "mem_0002", "mem_0005"]


class TestHiddenConfigResilience:
    def test_dirty_flag_restored_when_persist_fails(self, tmp_path: Path):
        from astrbot_plugin_iris_memory.iris_memory.config.hidden_config import HiddenConfigManager
        from astrbot_plugin_iris_memory.iris_memory.config.defaults import HiddenConfig

        manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        manager.set("llm_provider_rpm", 99)
        # 第一次写盘成功后再次制造失败场景
        with patch(
            "astrbot_plugin_iris_memory.iris_memory.utils.persistence.atomic_write_text",
            side_effect=OSError("磁盘满"),
        ):
            manager.set("llm_provider_rpm", 99)
            assert manager._dirty is True, "写盘失败必须保留脏标志"

        # 磁盘恢复后，下一次变更能把挂起的值写出
        manager.set("llm_provider_rpm", 60)
        data = json.loads((tmp_path / "hidden_config.json").read_text())
        assert data["llm_provider_rpm"] == 60

    def test_load_sanitizes_wrong_types(self, tmp_path: Path):
        from astrbot_plugin_iris_memory.iris_memory.config.hidden_config import HiddenConfigManager
        from astrbot_plugin_iris_memory.iris_memory.config.defaults import HiddenConfig

        defaults = HiddenConfig()
        int_key = next(
            f.name for f in defaults.__dataclass_fields__.values()
            if f.type in ("int", "float") or "int" in str(f.type)
        )
        path = tmp_path / "hidden_config.json"
        path.write_text(
            json.dumps({int_key: "not-a-number", "__foreign__": {"ok": 1}}),
            encoding="utf-8",
        )

        manager = HiddenConfigManager(path, defaults)
        # 错误类型回退默认值而不是污染运行时
        assert manager.get(int_key) == getattr(defaults, int_key)
        # 外部槽位按设计原样保留
        assert manager.get("__foreign__") == {"ok": 1}

    def test_reset_to_defaults_notifies_observers(self, tmp_path: Path):
        from astrbot_plugin_iris_memory.iris_memory.config.hidden_config import HiddenConfigManager
        from astrbot_plugin_iris_memory.iris_memory.config.defaults import HiddenConfig

        manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        manager.set("llm_provider_rpm", 99)

        notified: list[tuple[str, object, object]] = []
        manager.add_observer(lambda k, old, new: notified.append((k, old, new)))

        manager.reset_to_defaults()

        # 观察者必须真正收到被覆盖键的重置通知（此前遍历空 dict 是死代码）
        reset_keys = {key for key, _old, _new in notified}
        assert "llm_provider_rpm" in reset_keys
        old_values = {key: old for key, old, _new in notified}
        assert old_values["llm_provider_rpm"] == 99


class TestQueryRewriteInflightCleanup:
    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_leak_inflight_entry(self):
        from astrbot_plugin_iris_memory.iris_memory.core import llm_request_hook as hook

        hook._QUERY_REWRITE_INFLIGHT.clear()
        release = asyncio.Event()
        started = asyncio.Event()

        async def slow_rewrite():
            started.set()
            await release.wait()
            return "关键词"

        async def waiter():
            task = asyncio.create_task(slow_rewrite())
            hook._QUERY_REWRITE_INFLIGHT["k"] = task

            def _discard(done, *, entry_key="k"):
                if hook._QUERY_REWRITE_INFLIGHT.get(entry_key) is done:
                    hook._QUERY_REWRITE_INFLIGHT.pop(entry_key, None)

            task.add_done_callback(_discard)
            try:
                await asyncio.shield(task)
            finally:
                if task.done() and hook._QUERY_REWRITE_INFLIGHT.get("k") is task:
                    hook._QUERY_REWRITE_INFLIGHT.pop("k", None)

        waiter_task = asyncio.create_task(waiter())
        await started.wait()
        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        # 等待者已取消，此时入表项仍在（task 未完成）
        assert "k" in hook._QUERY_REWRITE_INFLIGHT

        # 任务完成后由 done callback 清理，不产生僵尸条目
        release.set()
        await asyncio.sleep(0.05)
        assert "k" not in hook._QUERY_REWRITE_INFLIGHT
        hook._QUERY_REWRITE_INFLIGHT.clear()


class TestL3LikeEscaping:
    @pytest.mark.asyncio
    async def test_wildcard_keyword_does_not_match_everything(self, tmp_path: Path):
        import sqlite3 as sq

        from astrbot_plugin_iris_memory.iris_memory.l3_kg.adapter import L3KGAdapter

        init_config({"l3_kg": {"enable": True}}, tmp_path)
        adapter = L3KGAdapter()
        adapter._is_available = True
        adapter._db = sq.connect(":memory:", check_same_thread=False)
        adapter._db.row_factory = sq.Row
        with adapter._db_lock:
            adapter._create_schema_unlocked()
            adapter._db.execute(
                "INSERT INTO nodes (id, label, name, content) VALUES (?, ?, ?, ?)",
                ("n1", "person", "小明", "普通内容"),
            )
            adapter._db.commit()

        # %% 若未转义会命中全表
        results = await adapter.search_nodes("%%%")
        assert results == []
        results = await adapter.search_nodes("小明")
        assert len(results) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,args,expected",
    [
        ("update_access", ("m1",), False),
        ("batch_update_access", (["m1"],), 0),
        ("delete_by_group", ("g1",), 0),
        ("delete_by_user", ("u1",), 0),
    ],
)
async def test_shutdown_connection_race_returns_failure(
    adapter, operation, args, expected
):
    """可用标志尚未重置但连接已关闭时，操作应正常退回失败结果。"""
    adapter._index = adapter._create_index(DIM)
    adapter._db.close()
    adapter._db = None
    assert await getattr(adapter, operation)(*args) == expected



class TestSyncRecoveryRaces:
    @pytest.mark.asyncio
    async def test_delete_during_embedding_does_not_resurrect_vector(self, adapter):
        _seed_db(adapter, [1, 2])
        adapter._index = adapter._create_index(DIM)

        async def embed_then_delete(texts):
            await adapter.delete_entries(["mem_0001"])
            return [[0.1] * DIM for _ in texts]

        adapter._embed = AsyncMock(side_effect=embed_then_delete)
        await adapter._reconcile_index_with_db()
        assert adapter._index_ids_unlocked() == {2}
        assert adapter._count_db() == 1

    @pytest.mark.asyncio
    async def test_edit_during_embedding_does_not_restore_old_content(self, adapter):
        _seed_db(adapter, [1])
        adapter._index = adapter._create_index(DIM)

        async def embed_then_edit(texts):
            adapter._db_write(
                "UPDATE memories SET content = '新内容' WHERE faiss_idx = 1"
            )
            return [[0.1] * DIM for _ in texts]

        adapter._embed = AsyncMock(side_effect=embed_then_edit)
        await adapter._reconcile_index_with_db()
        assert adapter._index.ntotal == 0
        adapter._embed = AsyncMock(return_value=[[0.2] * DIM])
        await adapter._reconcile_index_with_db()
        adapter._embed.assert_awaited_once_with(["新内容"])
        assert adapter._index_ids_unlocked() == {1}

    @pytest.mark.asyncio
    async def test_consistent_ids_still_remove_stale_free_slots(self, adapter):
        _seed_db(adapter, [1])
        adapter._index = adapter._create_index(DIM)
        adapter._add_missing_vectors_locked([(1, "内容 1")], [[0.1] * DIM])
        adapter._free_list = [1, 3, 3]
        await adapter._reconcile_index_with_db()
        assert adapter._free_list == [3]
        adapter._embed.assert_not_awaited()

    def test_extra_vector_recheck_preserves_reused_slot(self, adapter):
        _seed_db(adapter, [1])
        adapter._index = adapter._create_index(DIM)
        adapter._add_missing_vectors_locked([(1, "内容 1")], [[0.1] * DIM])
        adapter._remove_index_ids_locked([1])
        assert adapter._index_ids_unlocked() == {1}

    @pytest.mark.asyncio
    async def test_recovery_validates_dimensions_before_native_faiss(self, adapter):
        _seed_db(adapter, [1])
        adapter._index = adapter._create_index(DIM)
        adapter._embed = AsyncMock(return_value=[[0.1] * (DIM + 1)])
        await adapter._reconcile_index_with_db()
        assert adapter._index.ntotal == 0
        assert adapter._count_db() == 1
        adapter._embed = AsyncMock(return_value=[[0.1] * DIM])
        await adapter._reconcile_index_with_db()
        assert adapter._index.ntotal == 1

    @pytest.mark.asyncio
    async def test_loaded_dimension_mismatch_defers_repair_to_migration(self, adapter):
        import faiss

        _seed_db(adapter, [1])
        wrong_index = adapter._create_index(DIM + 1)
        faiss.write_index(wrong_index, str(adapter._persist_dir / "index.faiss"))
        await adapter._load_existing(0)
        assert adapter._embedding_dimensions == DIM
        assert adapter._index.d == DIM + 1
        adapter._embed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancelled_later_batch_keeps_first_batch_on_disk(self, adapter):
        import faiss

        _seed_db(adapter, list(range(65)))
        adapter._index = adapter._create_index(DIM)

        async def embed(texts):
            if len(texts) == 1:
                raise asyncio.CancelledError()
            return [[0.1] * DIM for _ in texts]

        adapter._embed = AsyncMock(side_effect=embed)
        with pytest.raises(asyncio.CancelledError):
            await adapter._reconcile_index_with_db()
        loaded = faiss.read_index(str(adapter._persist_dir / "index.faiss"))
        assert loaded.ntotal == 64
        assert adapter._count_db() == 65

    @pytest.mark.asyncio
    async def test_partial_migration_preserves_full_backup(self, adapter):
        from types import SimpleNamespace
        import json

        _seed_db(adapter, [1, 2, 3])
        stats = SimpleNamespace(
            total_count=3, imported_count=1, skipped_count=0, error_count=2
        )
        with patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.io.MemoryImporter.import_from_file",
            new=AsyncMock(return_value=stats),
        ):
            assert not await adapter._migrate_on_model_change("new-model", DIM)
        backups = list(
            (adapter._persist_dir.parent / "migration_backup").glob(
                "*_migration_backup.json"
            )
        )
        assert len(backups) == 1
        data = json.loads(backups[0].read_text())
        assert len(data["entries"]) == 3

    @pytest.mark.asyncio
    async def test_failed_final_save_keeps_migration_backup(self, adapter):
        _seed_db(adapter, [1, 2])
        with patch.object(
            adapter, "_write_index_atomic", side_effect=OSError("disk full")
        ):
            assert not await adapter._migrate_on_model_change("new-model", DIM)
        assert list(
            (adapter._persist_dir.parent / "migration_backup").glob(
                "*_migration_backup.json"
            )
        )


@pytest.mark.asyncio
async def test_partial_export_preserves_live_collection_and_archive(adapter):
    """导出只返回部分正式记忆时，不得删除旧库或归档。"""
    from types import SimpleNamespace

    _seed_db(adapter, [1, 2, 3])
    adapter._index = adapter._create_index(DIM)
    await adapter._reconcile_index_with_db()
    assert await adapter.evict_memories(["mem_0003"]) == 1
    delete_collection = AsyncMock(wraps=adapter.delete_collection)
    with (
        patch.object(adapter, "delete_collection", delete_collection),
        patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.io.MemoryExporter.export_all",
            new=AsyncMock(return_value=SimpleNamespace(total_count=2, exported_count=1)),
        ),
    ):
        assert not await adapter._migrate_on_model_change("new-model", DIM)
    delete_collection.assert_not_awaited()
    assert adapter._count_db() == 2
    assert await adapter.get_archived_count() == 1
    assert adapter._index_ids_unlocked() == {1, 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["partial_import", "final_save"])
async def test_failed_migration_keeps_archive_snapshot(adapter, failure):
    """失败迁移保留完整正式记忆备份和独立归档快照。"""
    import sqlite3
    from types import SimpleNamespace

    _seed_db(adapter, [1, 2, 3])
    adapter._index = adapter._create_index(DIM)
    await adapter._reconcile_index_with_db()
    assert await adapter.evict_memories(["mem_0003"]) == 1
    if failure == "partial_import":
        failure_patch = patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.io.MemoryImporter.import_from_file",
            new=AsyncMock(return_value=SimpleNamespace(
                total_count=2, imported_count=1, skipped_count=0, error_count=1
            )),
        )
    else:
        failure_patch = patch.object(
            adapter, "_write_index_atomic", side_effect=OSError("disk full")
        )
    with failure_patch:
        assert not await adapter._migrate_on_model_change("new-model", DIM)
    backup_dir = adapter._persist_dir.parent / "migration_backup"
    backups = list(backup_dir.glob("*_migration_backup.json"))
    archives = list(backup_dir.glob("*_migration_archives.db"))
    assert len(backups) == len(archives) == 1
    assert len(json.loads(backups[0].read_text())["entries"]) == 2
    with sqlite3.connect(archives[0]) as db:
        assert db.execute("SELECT memory_id FROM memories_archive").fetchall() == [
            ("mem_0003",)
        ]
    assert await adapter.get_archived_count() == 1


@pytest.mark.asyncio
async def test_repair_filters_free_list_against_current_db(adapter):
    """对账快照后被占用的槽位不得作为 free-list 保存。"""
    _seed_db(adapter, [1])
    adapter._index = adapter._create_index(DIM)
    adapter._free_list = [1, 2, 3, 3]
    snapshot = {1}
    _seed_db(adapter, [2])
    adapter._persist_repaired_index_locked(snapshot)
    assert adapter._free_list == [3]
    assert adapter._load_meta()["free_list"] == [3]


@pytest.mark.asyncio
async def test_shutdown_waits_for_checkpoint_and_reconcile(adapter):
    """关闭连接前必须等待落盘任务和正在进行的对账。"""
    _seed_db(adapter, [1])
    adapter._index = adapter._create_index(DIM)
    checkpoint_started = asyncio.Event()
    release_checkpoint = asyncio.Event()

    async def checkpoint():
        checkpoint_started.set()
        await release_checkpoint.wait()
        assert adapter._db is not None

    checkpoint_task = asyncio.create_task(checkpoint())
    adapter._checkpoint_task = checkpoint_task
    await checkpoint_started.wait()
    await adapter._reconcile_lock.acquire()
    shutdown = asyncio.create_task(adapter.shutdown())
    try:
        await asyncio.sleep(0)
        assert not shutdown.done()
        release_checkpoint.set()
        await checkpoint_task
        await asyncio.sleep(0)
        assert not shutdown.done()
        assert adapter._db is not None
    finally:
        release_checkpoint.set()
        adapter._reconcile_lock.release()
        await asyncio.wait_for(shutdown, timeout=5)
    assert adapter._db is None
    assert adapter._index is None

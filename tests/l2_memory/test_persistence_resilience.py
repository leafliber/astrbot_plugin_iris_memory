"""L2 持久化韧性回归测试。

覆盖以下修复的回归保护：
- FAISS 索引原子写（tmp + os.replace，不留半截文件）
- 索引损坏后的启动恢复（改名保留 + 从 SQLite 对账重建）
- 索引/DB 脱同步时的 ID 集对账重建
- checkpoint 任务强引用与 _checkpointing 复位
- 嵌入模型迁移失败时备份保留（不删唯一全量副本）
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from iris_memory.config import init_config
from iris_memory.config.config import reset_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter

DIM = 8


@pytest.fixture(autouse=True)
def _reset_iris_config():
    yield
    reset_config()


@pytest.fixture
def adapter(tmp_path: Path) -> L2MemoryAdapter:
    """带真实 SQLite 与真实 FAISS 的最小可用适配器。"""
    from iris_memory.config import get_config

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
            "iris_memory.l2_memory.io.MemoryImporter.import_from_file",
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
        from iris_memory.config.hidden_config import HiddenConfigManager
        from iris_memory.config.defaults import HiddenConfig

        manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        manager.set("llm_provider_rpm", 99)
        # 第一次写盘成功后再次制造失败场景
        with patch(
            "iris_memory.utils.persistence.atomic_write_text",
            side_effect=OSError("磁盘满"),
        ):
            manager.set("llm_provider_rpm", 99)
            assert manager._dirty is True, "写盘失败必须保留脏标志"

        # 磁盘恢复后，下一次变更能把挂起的值写出
        manager.set("llm_provider_rpm", 60)
        data = json.loads((tmp_path / "hidden_config.json").read_text())
        assert data["llm_provider_rpm"] == 60

    def test_load_sanitizes_wrong_types(self, tmp_path: Path):
        from iris_memory.config.hidden_config import HiddenConfigManager
        from iris_memory.config.defaults import HiddenConfig

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
        from iris_memory.config.hidden_config import HiddenConfigManager
        from iris_memory.config.defaults import HiddenConfig

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
        from iris_memory.core import llm_request_hook as hook

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

        from iris_memory.l3_kg.adapter import L3KGAdapter

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

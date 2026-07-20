"""L2 记忆库 persona 列+过滤 隔离测试"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l2_memory.io import MemoryExporter, MemoryImporter


@pytest.fixture
def persona_adapter():
    """真实 SQLite + mock FAISS 的适配器，用于验证 persona 列与过滤"""
    adapter = L2MemoryAdapter()
    adapter._is_available = True
    adapter._persist_dir = Path(tempfile.mkdtemp())
    adapter._embedding_dimensions = 8
    adapter._persona_id = "default"

    mock_index = Mock()
    mock_index.ntotal = 0
    mock_index.d = 8

    def fake_add_with_ids(vectors, ids):
        mock_index.ntotal += len(ids)

    mock_index.add_with_ids = fake_add_with_ids
    # search 返回空（隔离测试主要看 SQLite 过滤，不依赖向量命中）
    mock_index.search = Mock(return_value=(np.array([[]]), np.array([[]])))
    mock_index.remove_ids = Mock()
    adapter._index = mock_index

    adapter._db = adapter._open_db(adapter._persist_dir / "metadata.db")
    adapter._free_list = []
    adapter._dirty = False
    adapter._actual_embedding_model = "test-model"
    adapter._embedding_source = "provider"
    adapter._embedding_provider = None
    return adapter


class TestPersonaColumn:
    def test_persona_id_column_exists(self, persona_adapter):
        cols = {
            row[1]
            for row in persona_adapter._db.execute(
                "PRAGMA table_info(memories)"
            ).fetchall()
        }
        assert "persona_id" in cols

    @pytest.mark.asyncio
    async def test_add_memory_writes_persona_id(self, persona_adapter):
        # mock 嵌入，跳过 dedup（skip_dedup=True）
        with patch.object(
            persona_adapter, "_embed", new=AsyncMock(return_value=[[0.0] * 8])
        ):
            mid = await persona_adapter.add_memory(
                "content-yuki",
                metadata={"group_id": "g1"},
                persona_id="yuki",
                skip_dedup=True,
            )
        assert mid is not None
        row = persona_adapter._db.execute(
            "SELECT persona_id FROM memories WHERE memory_id = ?", (mid,)
        ).fetchone()
        assert row[0] == "yuki"


class TestPersonaFiltering:
    @pytest.mark.asyncio
    async def test_get_all_entries_filters_by_persona(self, persona_adapter):
        with patch.object(
            persona_adapter, "_embed", new=AsyncMock(return_value=[[0.0] * 8])
        ):
            await persona_adapter.add_memory("a", persona_id="yuki", skip_dedup=True)
            await persona_adapter.add_memory("b", persona_id="aria", skip_dedup=True)
            await persona_adapter.add_memory("c", persona_id="default", skip_dedup=True)

        yuki = await persona_adapter.get_all_entries(persona_id="yuki")
        aria = await persona_adapter.get_all_entries(persona_id="aria")
        all_entries = await persona_adapter.get_all_entries()  # None = 全部

        assert len(yuki) == 1 and yuki[0].content == "a"
        assert len(aria) == 1 and aria[0].content == "b"
        assert len(all_entries) == 3

    @pytest.mark.asyncio
    async def test_get_all_persona_ids(self, persona_adapter):
        with patch.object(
            persona_adapter, "_embed", new=AsyncMock(return_value=[[0.0] * 8])
        ):
            await persona_adapter.add_memory("a", persona_id="yuki", skip_dedup=True)
            await persona_adapter.add_memory("b", persona_id="aria", skip_dedup=True)

        ids = await persona_adapter.get_all_persona_ids()
        assert set(ids) == {"yuki", "aria"}

    @pytest.mark.asyncio
    async def test_delete_all_scoped_to_persona(self, persona_adapter):
        with patch.object(
            persona_adapter, "_embed", new=AsyncMock(return_value=[[0.0] * 8])
        ):
            await persona_adapter.add_memory("a", persona_id="yuki", skip_dedup=True)
            await persona_adapter.add_memory("b", persona_id="aria", skip_dedup=True)

        count = await persona_adapter.delete_all(persona_id="yuki")
        assert count == 1

        remaining = await persona_adapter.get_all_entries()
        assert len(remaining) == 1
        assert remaining[0].content == "b"

    @pytest.mark.asyncio
    async def test_get_stats_global(self, persona_adapter):
        with patch.object(
            persona_adapter, "_embed", new=AsyncMock(return_value=[[0.0] * 8])
        ):
            await persona_adapter.add_memory("a", persona_id="yuki", skip_dedup=True)
            await persona_adapter.add_memory("b", persona_id="yuki", skip_dedup=True)
        stats = await persona_adapter.get_stats()
        assert stats["total_count"] == 2


@pytest.fixture
def migration_adapter():
    """真实 SQLite + mock FAISS（search 返回无命中）的适配器，用于迁移回合测试。

    与 persona_adapter 不同，search 返回 index=-1（无命中），使重导入路径中
    add_memory 的去重检查 _find_similar_unlocked 正常返回 None 而不抛异常。
    """
    adapter = L2MemoryAdapter()
    adapter._is_available = True
    adapter._persist_dir = Path(tempfile.mkdtemp())
    adapter._embedding_dimensions = 8
    adapter._persona_id = "default"

    mock_index = Mock()
    mock_index.ntotal = 0
    mock_index.d = 8

    def fake_add_with_ids(vectors, ids):
        mock_index.ntotal += len(ids)

    mock_index.add_with_ids = fake_add_with_ids
    # search 返回无命中（index=-1），dedup 不会误杀重导入的条目
    mock_index.search = Mock(return_value=(np.array([[-1.0]]), np.array([[-1]])))
    mock_index.remove_ids = Mock()
    adapter._index = mock_index

    adapter._db = adapter._open_db(adapter._persist_dir / "metadata.db")
    adapter._free_list = []
    adapter._dirty = False
    adapter._actual_embedding_model = "test-model"
    adapter._embedding_source = "provider"
    adapter._embedding_provider = None
    return adapter


class TestMigrationPersonaPreservation:
    """回归：模型迁移「导出→删库→重导入」必须保留 persona_id 隔离"""

    @pytest.mark.asyncio
    async def test_export_import_roundtrip_preserves_persona(self, migration_adapter):
        adapter = migration_adapter
        embed = AsyncMock(return_value=[[0.0] * 8])
        with patch.object(adapter, "_embed", new=embed):
            await adapter.add_memory("a", persona_id="yuki", skip_dedup=True)
            await adapter.add_memory("b", persona_id="aria", skip_dedup=True)
            await adapter.add_memory("c", persona_id="default", skip_dedup=True)

        # 1. 导出全部
        backup = adapter._persist_dir / "_migration_backup.json"
        exporter = MemoryExporter(adapter)
        await exporter.export_all(backup)

        data = json.loads(backup.read_text(encoding="utf-8"))
        # 导出 JSON 必须逐条携带 persona_id
        pids = {e["persona_id"] for e in data["entries"]}
        assert pids == {"yuki", "aria", "default"}

        # 2. 模拟迁移步骤 2：清空旧数据（保留适配器可用）
        with adapter._lock:
            adapter._db.execute("DELETE FROM memories")
            adapter._db.commit()
            adapter._index.ntotal = 0
            adapter._free_list = []
        assert len(await adapter.get_all_entries()) == 0

        # 3. 重新导入
        importer = MemoryImporter(adapter)
        with patch.object(adapter, "_embed", new=embed):
            stats = await importer.import_from_file(backup, skip_duplicates=False)
        assert stats.imported_count == 3

        # 4. 人格隔离保持：按 persona 过滤后各归各位
        yuki = await adapter.get_all_entries(persona_id="yuki")
        aria = await adapter.get_all_entries(persona_id="aria")
        default = await adapter.get_all_entries(persona_id="default")

        assert len(yuki) == 1 and yuki[0].content == "a"
        assert yuki[0].persona_id == "yuki"
        assert len(aria) == 1 and aria[0].content == "b"
        assert aria[0].persona_id == "aria"
        assert len(default) == 1 and default[0].content == "c"
        assert default[0].persona_id == "default"

    @pytest.mark.asyncio
    async def test_update_content_preserves_persona_id(self, migration_adapter):
        """回归：update_content 不得把 persona_id 重置为 default"""
        adapter = migration_adapter
        embed = AsyncMock(return_value=[[0.0] * 8])
        with patch.object(adapter, "_embed", new=embed):
            mid = await adapter.add_memory(
                "原始内容", persona_id="yuki", skip_dedup=True
            )

        with patch.object(adapter, "_embed", new=embed):
            ok = await adapter.update_content(mid, "修改后的内容")
        assert ok

        row = adapter._db.execute(
            "SELECT persona_id FROM memories WHERE memory_id = ?", (mid,)
        ).fetchone()
        assert row[0] == "yuki"

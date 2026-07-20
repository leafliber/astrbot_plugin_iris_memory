"""l2_migrator 测试（stub chromadb 注入 + 缺包分支 + 真实 chromadb 回合）"""

import sys
import types

import pytest

from iris_memory.legacy_migration.detector import LegacyDetection
from iris_memory.legacy_migration.l2_migrator import (
    BATCH_SIZE,
    map_legacy_metadata,
    migrate_l2,
)


def _chroma_detection(tmp_path):
    chroma = tmp_path / "chroma"
    chroma.mkdir(exist_ok=True)
    (chroma / "chroma.sqlite3").write_bytes(b"dummy")
    return LegacyDetection(chroma_dir=chroma)


def _make_metadata(i: int, **overrides):
    meta = {
        "user_id": f"u{i % 3}",
        "group_id": f"g{i % 2}",
        "persona_id": "default" if i % 2 == 0 else "yuki",
        "sender_name": f"用户{i}",
        "storage_layer": "episodic",
        "scope": "shared" if i % 2 == 0 else "private",
        "created_time": f"2025-01-{(i % 28) + 1:02d}T00:00:00",
        "last_access_time": "2025-06-01T00:00:00",
        "access_count": i,
        "confidence": 0.7,
        "summarized": i % 2 == 0,
        "importance_score": 0.6,
        "rif_score": 0.4,
        "type": "fact",
    }
    meta.update(overrides)
    return meta


class StubCollection:
    """模拟 chromadb Collection（支持分页）"""

    def __init__(self, ids, documents, metadatas):
        self._ids = ids
        self._documents = documents
        self._metadatas = metadatas
        self.get_calls = []

    def count(self):
        return len(self._ids)

    def get(self, limit=None, offset=0, include=None):
        self.get_calls.append({"limit": limit, "offset": offset, "include": include})
        end = offset + limit if limit else None
        return {
            "ids": self._ids[offset:end],
            "documents": self._documents[offset:end],
            "metadatas": self._metadatas[offset:end],
        }


class StubClient:
    """模拟 chromadb PersistentClient"""

    def __init__(self, collection, default_name_exists=True):
        self._collection = collection
        self._default_name_exists = default_name_exists

    def get_collection(self, name):
        if name == "iris_memory" and self._default_name_exists:
            return self._collection
        if name == "other_collection" and not self._default_name_exists:
            return self._collection
        raise ValueError(f"Collection {name} does not exist")

    def list_collections(self):
        return ["other_collection"]


def _install_stub_chromadb(monkeypatch, collection, default_name_exists=True):
    client = StubClient(collection, default_name_exists)
    stub_module = types.SimpleNamespace(
        PersistentClient=lambda path, **kwargs: client,
    )
    monkeypatch.setitem(sys.modules, "chromadb", stub_module)
    return client


class TestMapLegacyMetadata:
    """metadata 映射正确性"""

    def test_full_mapping(self):
        mapped = map_legacy_metadata(_make_metadata(0), "old_id_1")
        assert mapped["timestamp"] == "2025-01-01T00:00:00"
        assert mapped["group_id"] == "g0"
        assert mapped["user_id"] == "u0"
        assert mapped["access_count"] == 0
        assert mapped["confidence"] == 0.7
        assert mapped["source"] == "summary"
        assert mapped["kg_processed"] is True
        assert mapped["original_storage_layer"] == "episodic"
        assert mapped["original_scope"] == "shared"
        assert mapped["legacy_id"] == "old_id_1"
        assert mapped["migrated_from"] == "iris_memory_v2"
        assert mapped["last_access_time"] == "2025-06-01T00:00:00"
        assert mapped["importance_score"] == 0.6
        assert mapped["rif_score"] == 0.4

    def test_source_tool_when_not_summarized(self):
        mapped = map_legacy_metadata(_make_metadata(1), "x")
        assert mapped["source"] == "tool"

    def test_defaults_and_none_dropped(self):
        mapped = map_legacy_metadata({}, "x")
        assert mapped["timestamp"]  # 回退为当前时间
        assert mapped["access_count"] == 0
        assert mapped["confidence"] == 0.5
        assert "last_access_time" not in mapped
        assert "type" not in mapped  # None 被剔除

    def test_bad_numeric_values(self):
        mapped = map_legacy_metadata(
            {"access_count": "abc", "confidence": object()}, "x"
        )
        assert mapped["access_count"] == 0
        assert mapped["confidence"] == 0.5


class TestMigrateL2Branches:
    """各跳过/失败分支"""

    @pytest.mark.asyncio
    async def test_skipped_no_data(self, component_manager):
        stats = await migrate_l2(LegacyDetection(), component_manager)
        assert stats["status"] == "skipped_no_data"

    @pytest.mark.asyncio
    async def test_skipped_adapter_unavailable(
        self, tmp_path, component_manager, l2_adapter
    ):
        l2_adapter.is_available = False
        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["status"] == "skipped_adapter_unavailable"

    @pytest.mark.asyncio
    async def test_missing_chromadb(self, tmp_path, component_manager, monkeypatch):
        # sys.modules 中置 None 会让 import 抛出 ImportError
        monkeypatch.setitem(sys.modules, "chromadb", None)
        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["status"] == "skipped_missing_chromadb"
        assert component_manager.get_component("l2_memory").added == []

    @pytest.mark.asyncio
    async def test_no_collection(self, tmp_path, component_manager, monkeypatch):
        class EmptyClient:
            def get_collection(self, name):
                raise ValueError("not found")

            def list_collections(self):
                return []

        stub = types.SimpleNamespace(PersistentClient=lambda path: EmptyClient())
        monkeypatch.setitem(sys.modules, "chromadb", stub)
        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["status"] == "skipped_no_collection"

    @pytest.mark.asyncio
    async def test_open_failure(self, tmp_path, component_manager, monkeypatch):
        def raiser(path):
            raise RuntimeError("旧版 chroma 数据格式不兼容")

        stub = types.SimpleNamespace(PersistentClient=raiser)
        monkeypatch.setitem(sys.modules, "chromadb", stub)
        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["status"] == "error"
        assert "不兼容" in stats["error"]


class TestMigrateL2WithStub:
    """stub chromadb 下的映射/分批验证"""

    def _build_collection(self, n):
        ids = [f"chroma_id_{i}" for i in range(n)]
        docs = [f"旧记忆内容 {i}" for i in range(n)]
        metas = [_make_metadata(i) for i in range(n)]
        return StubCollection(ids, docs, metas)

    @pytest.mark.asyncio
    async def test_mapping_and_import(self, tmp_path, component_manager, monkeypatch):
        collection = self._build_collection(5)
        _install_stub_chromadb(monkeypatch, collection)

        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["status"] == "ok"
        assert stats["total"] == 5
        assert stats["imported"] == 5
        assert stats["errors"] == 0

        added = component_manager.get_component("l2_memory").added
        assert len(added) == 5

        first = added[0]
        assert first["content"] == "旧记忆内容 0"
        assert first["persona_id"] == "default"
        assert first["skip_dedup"] is True  # 保真导入，不去重
        meta = first["metadata"]
        assert meta["legacy_id"] == "chroma_id_0"
        assert meta["kg_processed"] is True
        assert meta["timestamp"] == "2025-01-01T00:00:00"

        # 奇数条目 persona_id 为 yuki
        assert added[1]["persona_id"] == "yuki"

    @pytest.mark.asyncio
    async def test_batching_and_progress(
        self, tmp_path, component_manager, monkeypatch
    ):
        n = BATCH_SIZE * 2 + 50
        collection = self._build_collection(n)
        _install_stub_chromadb(monkeypatch, collection)

        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["imported"] == n

        # 分页调用：3 批，offset 递增
        offsets = [c["offset"] for c in collection.get_calls]
        assert offsets == [0, BATCH_SIZE, BATCH_SIZE * 2]
        for call in collection.get_calls:
            assert call["limit"] == BATCH_SIZE
            assert "embeddings" not in (call["include"] or [])

    @pytest.mark.asyncio
    async def test_empty_content_skipped(
        self, tmp_path, component_manager, monkeypatch
    ):
        collection = StubCollection(
            ["a", "b", "c"],
            ["有内容", "", None],
            [_make_metadata(0), _make_metadata(1), _make_metadata(2)],
        )
        _install_stub_chromadb(monkeypatch, collection)

        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["imported"] == 1
        assert stats["skipped"] == 2

    @pytest.mark.asyncio
    async def test_collection_name_fallback(
        self, tmp_path, component_manager, monkeypatch
    ):
        collection = self._build_collection(2)
        _install_stub_chromadb(monkeypatch, collection, default_name_exists=False)

        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["status"] == "ok"
        assert stats["imported"] == 2

    @pytest.mark.asyncio
    async def test_empty_collection(self, tmp_path, component_manager, monkeypatch):
        collection = self._build_collection(0)
        _install_stub_chromadb(monkeypatch, collection)

        stats = await migrate_l2(_chroma_detection(tmp_path), component_manager)
        assert stats["status"] == "ok"
        assert stats["total"] == 0
        assert stats["imported"] == 0


try:
    import chromadb as _chromadb_check  # noqa: F401

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb 未安装")
class TestMigrateL2RealChroma:
    """真实 chromadb 回合测试（验证 PersistentClient/get API 用法与旧数据形态一致）"""

    @pytest.mark.asyncio
    async def test_real_chroma_roundtrip(self, tmp_path, component_manager):
        import chromadb

        chroma_dir = tmp_path / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.create_collection("iris_memory")
        n = 3
        collection.add(
            ids=[f"real_id_{i}" for i in range(n)],
            documents=[f"真实旧记忆 {i}" for i in range(n)],
            metadatas=[_make_metadata(i) for i in range(n)],
            embeddings=[[0.1 + i * 0.01] * 16 for i in range(n)],
        )
        assert collection.count() == n
        # 释放客户端，避免与迁移器重复占用
        del client, collection

        detection = LegacyDetection(chroma_dir=chroma_dir)
        stats = await migrate_l2(detection, component_manager)

        assert stats["status"] == "ok"
        assert stats["total"] == n
        assert stats["imported"] == n

        added = component_manager.get_component("l2_memory").added
        assert {a["metadata"]["legacy_id"] for a in added} == {
            f"real_id_{i}" for i in range(n)
        }
        by_id = {a["metadata"]["legacy_id"]: a for a in added}
        assert by_id["real_id_0"]["metadata"]["source"] == "summary"
        assert by_id["real_id_1"]["persona_id"] == "yuki"
        assert by_id["real_id_1"]["metadata"]["original_scope"] == "private"

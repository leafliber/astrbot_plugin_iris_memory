"""detector 分支覆盖测试"""

import pytest

from iris_memory.legacy_migration.detector import (
    detect_legacy_config_keys,
    detect_legacy_data,
)


class TestDetectFiles:
    """文件型旧数据检测"""

    @pytest.mark.asyncio
    async def test_no_legacy_data(self, star, tmp_path):
        detection = await detect_legacy_data(star, tmp_path, None)
        assert not detection.has_anything
        assert not detection.has_file_data
        assert detection.summary_text() == "无"

    @pytest.mark.asyncio
    async def test_detect_chroma_dir(self, star, tmp_path):
        chroma = tmp_path / "chroma"
        chroma.mkdir()
        (chroma / "chroma.sqlite3").write_bytes(b"dummy")

        detection = await detect_legacy_data(star, tmp_path, None)
        assert detection.chroma_dir == chroma
        assert detection.has_file_data

    @pytest.mark.asyncio
    async def test_empty_chroma_dir_ignored(self, star, tmp_path):
        (tmp_path / "chroma").mkdir()

        detection = await detect_legacy_data(star, tmp_path, None)
        assert detection.chroma_dir is None
        assert not detection.has_anything

    @pytest.mark.asyncio
    async def test_detect_kg_db(self, star, tmp_path):
        kg_db = tmp_path / "knowledge_graph.db"
        kg_db.write_bytes(b"SQLite format 3")

        detection = await detect_legacy_data(star, tmp_path, None)
        assert detection.kg_db_path == kg_db

    @pytest.mark.asyncio
    async def test_empty_kg_db_ignored(self, star, tmp_path):
        (tmp_path / "knowledge_graph.db").write_bytes(b"")

        detection = await detect_legacy_data(star, tmp_path, None)
        assert detection.kg_db_path is None


class TestDetectKV:
    """KV 旧键检测"""

    @pytest.mark.asyncio
    async def test_detect_kv_keys(self, star, tmp_path):
        star.kv["user_personas"] = {"u1": {"user_id": "u1"}}
        star.kv["proactive_reply_whitelist"] = {"group_whitelist": ["g1"]}

        detection = await detect_legacy_data(star, tmp_path, None)
        assert set(detection.kv_keys) == {"user_personas", "proactive_reply_whitelist"}
        assert detection.kv_keys["user_personas"]["u1"]["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_empty_kv_containers_ignored(self, star, tmp_path):
        star.kv["user_personas"] = {}
        star.kv["sessions"] = []

        detection = await detect_legacy_data(star, tmp_path, None)
        assert detection.kv_keys == {}

    @pytest.mark.asyncio
    async def test_kv_error_isolated(self, star, tmp_path):
        async def failing_get(key, default=None):
            if key == "user_personas":
                raise RuntimeError("KV 后端故障")
            return star.kv.get(key, default)

        star.get_kv_data = failing_get  # type: ignore[assignment]
        star.kv["sessions"] = {"s1": {}}

        detection = await detect_legacy_data(star, tmp_path, None)
        assert "user_personas" not in detection.kv_keys
        assert detection.kv_keys.get("sessions") == {"s1": {}}


class TestDetectConfig:
    """旧配置键检测"""

    def test_detect_config_keys(self):
        raw = {
            "persona": {"enabled": False},
            "embedding": {"source": "local", "local_model": "m3e"},
            "l2_memory": {"enable": True},  # 新版键不应命中
            "knowledge_graph": {"other": 1},  # 段存在但键不命中
        }
        found = detect_legacy_config_keys(raw)
        assert set(found) == {
            "persona.enabled",
            "embedding.source",
            "embedding.local_model",
        }

    def test_detect_config_keys_empty(self):
        assert detect_legacy_config_keys(None) == []
        assert detect_legacy_config_keys({}) == []
        assert detect_legacy_config_keys({"l2_memory": {"enable": True}}) == []

    @pytest.mark.asyncio
    async def test_summary_text(self, star, tmp_path):
        (tmp_path / "chroma").mkdir()
        (tmp_path / "chroma" / "x").write_bytes(b"x")
        star.kv["sessions"] = {"a": 1}

        detection = await detect_legacy_data(star, tmp_path, None)
        text = detection.summary_text()
        assert "ChromaDB" in text
        assert "sessions" in text

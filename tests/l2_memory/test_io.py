"""L2 记忆导入导出测试"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock
import json
import tempfile

from iris_memory.l2_memory.io import (
    MemoryExporter,
    MemoryImporter,
    ExportStats,
    ImportStats,
    MemoryExport,
    export_memories,
    import_memories,
)
from iris_memory.l2_memory.models import MemoryEntry
from iris_memory.l2_memory.adapter import L2MemoryAdapter


class TestExportStats:
    """导出统计测试"""

    def test_create_export_stats(self):
        """测试创建导出统计"""
        stats = ExportStats(
            total_count=100,
            exported_count=80,
            skipped_count=20,
            file_size_bytes=1024,
            export_time="2024-01-01T00:00:00",
        )

        assert stats.total_count == 100
        assert stats.exported_count == 80
        assert stats.skipped_count == 20
        assert stats.file_size_bytes == 1024


class TestImportStats:
    """导入统计测试"""

    def test_create_import_stats(self):
        """测试创建导入统计"""
        stats = ImportStats(
            total_count=50,
            imported_count=45,
            skipped_count=3,
            error_count=2,
            import_time="2024-01-01T00:00:00",
        )

        assert stats.total_count == 50
        assert stats.imported_count == 45
        assert stats.skipped_count == 3
        assert stats.error_count == 2


class TestMemoryExport:
    """导出数据结构测试"""

    def test_create_memory_export(self):
        """测试创建导出数据"""
        export = MemoryExport(
            version="1.0",
            persona_id="default",
            export_time="2024-01-01T00:00:00",
            entries=[{"id": "mem_001", "content": "测试记忆", "metadata": {}}],
        )

        assert export.version == "1.0"
        assert export.persona_id == "default"
        assert len(export.entries) == 1


class TestMemoryExporter:
    """导出器测试"""

    @pytest.fixture
    def mock_adapter(self):
        """创建模拟适配器"""
        adapter = Mock(spec=L2MemoryAdapter)
        adapter.is_available = True
        adapter._persona_id = "default"
        return adapter

    @pytest.fixture
    def sample_entries(self):
        """创建示例条目"""
        return [
            MemoryEntry(
                id="mem_001",
                content="用户喜欢吃苹果",
                metadata={
                    "group_id": "group_123",
                    "timestamp": "2024-01-01T10:00:00",
                    "access_count": 5,
                    "confidence": 0.9,
                },
            ),
            MemoryEntry(
                id="mem_002",
                content="用户今天吃了香蕉",
                metadata={
                    "group_id": "group_456",
                    "timestamp": "2024-01-02T10:00:00",
                    "access_count": 2,
                    "confidence": 0.8,
                },
            ),
            MemoryEntry(
                id="mem_003",
                content="用户不喜欢吃榴莲",
                metadata={
                    "group_id": "group_123",
                    "timestamp": "2024-01-03T10:00:00",
                    "access_count": 1,
                    "confidence": 0.7,
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_export_all(self, mock_adapter, sample_entries):
        """测试导出所有记忆"""
        mock_adapter.get_all_entries = AsyncMock(return_value=sample_entries)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            exporter = MemoryExporter(mock_adapter)
            stats = await exporter.export_to_file(temp_path)

            assert stats.total_count == 3
            assert stats.exported_count == 3
            assert stats.skipped_count == 0
            assert temp_path.exists()

            # 验证文件内容
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["version"] == "1.0"
            assert len(data["entries"]) == 3
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_export_with_group_filter(self, mock_adapter, sample_entries):
        """测试按群聊ID筛选导出"""
        mock_adapter.get_all_entries = AsyncMock(return_value=sample_entries)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            exporter = MemoryExporter(mock_adapter)
            stats = await exporter.export_to_file(temp_path, group_id="group_123")

            assert stats.total_count == 3
            assert stats.exported_count == 2  # 只有 group_123 的条目
            assert stats.skipped_count == 1

            # 验证文件内容
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert len(data["entries"]) == 2
            for entry in data["entries"]:
                assert entry["metadata"]["group_id"] == "group_123"
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_export_with_time_filter(self, mock_adapter, sample_entries):
        """测试按时间筛选导出"""
        mock_adapter.get_all_entries = AsyncMock(return_value=sample_entries)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            exporter = MemoryExporter(mock_adapter)

            start_time = datetime(2024, 1, 2, 0, 0, 0)
            stats = await exporter.export_to_file(temp_path, start_time=start_time)

            assert stats.exported_count == 2  # 1月2日和1月3日的条目
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_export_unavailable_adapter(self):
        """测试适配器不可用时的导出"""
        adapter = Mock(spec=L2MemoryAdapter)
        adapter.is_available = False

        exporter = MemoryExporter(adapter)
        stats = await exporter.export_to_file(Path("test.json"))

        assert stats.total_count == 0
        assert stats.exported_count == 0

    def test_export_to_json(self, mock_adapter, sample_entries):
        """测试导出为 JSON 字符串"""
        exporter = MemoryExporter(mock_adapter)
        json_str = exporter.export_to_json(sample_entries)

        data = json.loads(json_str)
        assert data["version"] == "1.0"
        assert len(data["entries"]) == 3


class TestMemoryImporter:
    """导入器测试"""

    @pytest.fixture
    def mock_adapter(self):
        """创建模拟适配器"""
        adapter = Mock(spec=L2MemoryAdapter)
        adapter.is_available = True
        adapter._persona_id = "default"
        return adapter

    @pytest.fixture
    def sample_export_file(self):
        """创建示例导出文件"""
        export_data = {
            "version": "1.0",
            "persona_id": "default",
            "export_time": "2024-01-01T00:00:00",
            "entries": [
                {
                    "id": "mem_001",
                    "content": "测试记忆1",
                    "metadata": {"group_id": "group_123"},
                },
                {
                    "id": "mem_002",
                    "content": "测试记忆2",
                    "metadata": {"group_id": "group_456"},
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(export_data, f)
            temp_path = Path(f.name)

        return temp_path

    @pytest.mark.asyncio
    async def test_import_from_file(self, mock_adapter, sample_export_file):
        """测试从文件导入"""
        mock_adapter.add_memory = AsyncMock(return_value="mem_new_001")

        try:
            importer = MemoryImporter(mock_adapter)
            stats = await importer.import_from_file(sample_export_file)

            assert stats.total_count == 2
            assert stats.imported_count == 2
            assert stats.error_count == 0
        finally:
            sample_export_file.unlink()

    @pytest.mark.asyncio
    async def test_import_with_skip_duplicates(self, mock_adapter, sample_export_file):
        """测试跳过重复记忆"""
        # 第一条成功，第二条返回 None（重复）
        mock_adapter.add_memory = AsyncMock(side_effect=["mem_001", None])

        try:
            importer = MemoryImporter(mock_adapter)
            stats = await importer.import_from_file(
                sample_export_file, skip_duplicates=True
            )

            assert stats.total_count == 2
            assert stats.imported_count == 1
            assert stats.skipped_count == 1
        finally:
            sample_export_file.unlink()

    @pytest.mark.asyncio
    async def test_import_with_metadata_update(self, mock_adapter, sample_export_file):
        """测试更新元数据导入"""
        mock_adapter.add_memory = AsyncMock(return_value="mem_new")

        try:
            importer = MemoryImporter(mock_adapter)
            stats = await importer.import_from_file(
                sample_export_file,
                update_metadata=True,
                metadata_updates={"imported": True},
            )

            assert stats.imported_count == 2

            # 验证元数据已更新
            calls = mock_adapter.add_memory.call_args_list
            for call in calls:
                metadata = call[0][1]
                assert metadata.get("imported")
        finally:
            sample_export_file.unlink()

    @pytest.mark.asyncio
    async def test_import_unavailable_adapter(self, sample_export_file):
        """测试适配器不可用时的导入"""
        adapter = Mock(spec=L2MemoryAdapter)
        adapter.is_available = False

        try:
            importer = MemoryImporter(adapter)
            stats = await importer.import_from_file(sample_export_file)

            assert stats.total_count == 0
            assert stats.imported_count == 0
        finally:
            sample_export_file.unlink()

    @pytest.mark.asyncio
    async def test_import_nonexistent_file(self, mock_adapter):
        """测试导入不存在的文件"""
        importer = MemoryImporter(mock_adapter)
        stats = await importer.import_from_file(Path("nonexistent.json"))

        assert stats.error_count == 1

    @pytest.mark.asyncio
    async def test_import_invalid_json(self, mock_adapter):
        """测试导入无效 JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {{{")
            temp_path = Path(f.name)

        try:
            importer = MemoryImporter(mock_adapter)
            stats = await importer.import_from_file(temp_path)

            assert stats.error_count == 1
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_import_old_format(self, mock_adapter):
        """测试导入旧格式（列表）"""
        old_format_data = [{"id": "mem_001", "content": "旧格式记忆", "metadata": {}}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(old_format_data, f)
            temp_path = Path(f.name)

        mock_adapter.add_memory = AsyncMock(return_value="mem_new")

        try:
            importer = MemoryImporter(mock_adapter)
            stats = await importer.import_from_file(temp_path)

            assert stats.total_count == 1
            assert stats.imported_count == 1
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_import_from_json_string(self, mock_adapter):
        """测试从 JSON 字符串导入"""
        json_str = json.dumps(
            [{"id": "mem_001", "content": "测试记忆", "metadata": {}}]
        )

        mock_adapter.add_memory = AsyncMock(return_value="mem_new")

        importer = MemoryImporter(mock_adapter)
        stats = await importer.import_from_json(json_str)

        assert stats.total_count == 1
        assert stats.imported_count == 1

    @pytest.mark.asyncio
    async def test_import_entries_directly(self, mock_adapter):
        """测试直接导入条目"""
        entries = [
            MemoryEntry(id="mem_001", content="记忆1", metadata={}),
            MemoryEntry(id="mem_002", content="记忆2", metadata={}),
        ]

        mock_adapter.add_memory = AsyncMock(return_value="mem_new")

        importer = MemoryImporter(mock_adapter)
        stats = await importer.import_entries(entries)

        assert stats.total_count == 2
        assert stats.imported_count == 2

    @pytest.mark.asyncio
    async def test_import_preserves_per_entry_persona_id(self, mock_adapter):
        """导入时必须按条目透传 persona_id，否则迁移会塌缩人格隔离"""
        export_data = {
            "version": "1.0",
            "persona_id": "default",
            "export_time": "2024-01-01T00:00:00",
            "entries": [
                {
                    "id": "mem_001",
                    "content": "yuki 的记忆",
                    "metadata": {"group_id": "g1"},
                    "persona_id": "yuki",
                },
                {
                    "id": "mem_002",
                    "content": "aria 的记忆",
                    "metadata": {"group_id": "g1"},
                    "persona_id": "aria",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(export_data, f)
            temp_path = Path(f.name)

        mock_adapter.add_memory = AsyncMock(return_value="mem_new")
        try:
            importer = MemoryImporter(mock_adapter)
            stats = await importer.import_from_file(temp_path)

            assert stats.imported_count == 2
            calls = mock_adapter.add_memory.call_args_list
            assert calls[0].kwargs["persona_id"] == "yuki"
            assert calls[1].kwargs["persona_id"] == "aria"
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_import_falls_back_to_file_persona_id(self, mock_adapter):
        """旧版导出无逐条 persona_id 时回退到文件级 persona_id"""
        export_data = {
            "version": "1.0",
            "persona_id": "legacy",
            "export_time": "2024-01-01T00:00:00",
            "entries": [
                # 旧格式：无 persona_id 字段
                {"id": "mem_001", "content": "旧记忆", "metadata": {}},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(export_data, f)
            temp_path = Path(f.name)

        mock_adapter.add_memory = AsyncMock(return_value="mem_new")
        try:
            importer = MemoryImporter(mock_adapter)
            await importer.import_from_file(temp_path)

            assert mock_adapter.add_memory.call_args.kwargs["persona_id"] == "legacy"
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_import_entries_preserves_persona_id(self, mock_adapter):
        """import_entries 直接导入时透传 entry.persona_id"""
        entries = [
            MemoryEntry(id="mem_001", content="记忆1", metadata={}, persona_id="yuki"),
            MemoryEntry(id="mem_002", content="记忆2", metadata={}, persona_id="aria"),
        ]

        mock_adapter.add_memory = AsyncMock(return_value="mem_new")

        importer = MemoryImporter(mock_adapter)
        await importer.import_entries(entries)

        calls = mock_adapter.add_memory.call_args_list
        assert calls[0].kwargs["persona_id"] == "yuki"
        assert calls[1].kwargs["persona_id"] == "aria"

    @pytest.mark.asyncio
    async def test_skip_duplicates_false_passes_skip_dedup_true(
        self, mock_adapter, sample_export_file
    ):
        """回归：skip_duplicates=False 必须传 skip_dedup=True 关闭去重

        历史 bug：skip_duplicates 仅决定计入 skipped 还是 error，不传
        skip_dedup=True，add_memory 默认始终去重（阈值 0.87），相似度
        ≥0.87 的条目被静默丢弃。迁移以 skip_duplicates=False 期望原样导入。
        """
        mock_adapter.add_memory = AsyncMock(return_value="mem_new")

        try:
            importer = MemoryImporter(mock_adapter)
            await importer.import_from_file(sample_export_file, skip_duplicates=False)

            for call in mock_adapter.add_memory.call_args_list:
                assert call.kwargs.get("skip_dedup") is True, (
                    "skip_duplicates=False 时必须传 skip_dedup=True 关闭去重"
                )
        finally:
            sample_export_file.unlink()

    @pytest.mark.asyncio
    async def test_skip_duplicates_true_passes_skip_dedup_false(
        self, mock_adapter, sample_export_file
    ):
        """对照：skip_duplicates=True 时 skip_dedup=False（启用去重）"""
        mock_adapter.add_memory = AsyncMock(return_value="mem_new")

        try:
            importer = MemoryImporter(mock_adapter)
            await importer.import_from_file(sample_export_file, skip_duplicates=True)

            for call in mock_adapter.add_memory.call_args_list:
                assert call.kwargs.get("skip_dedup") is False
        finally:
            sample_export_file.unlink()

    @pytest.mark.asyncio
    async def test_import_entries_skip_duplicates_false(self, mock_adapter):
        """import_entries 同样映射 skip_duplicates→skip_dedup"""
        entries = [
            MemoryEntry(id="mem_001", content="记忆1", metadata={}),
            MemoryEntry(id="mem_002", content="记忆2", metadata={}),
        ]

        mock_adapter.add_memory = AsyncMock(return_value="mem_new")

        importer = MemoryImporter(mock_adapter)
        await importer.import_entries(entries, skip_duplicates=False)

        for call in mock_adapter.add_memory.call_args_list:
            assert call.kwargs.get("skip_dedup") is True


class TestConvenienceFunctions:
    """便捷函数测试"""

    @pytest.fixture
    def mock_adapter(self):
        """创建模拟适配器"""
        adapter = Mock(spec=L2MemoryAdapter)
        adapter.is_available = True
        adapter._persona_id = "default"
        adapter.get_all_entries = AsyncMock(return_value=[])
        adapter.add_memory = AsyncMock(return_value="mem_new")
        return adapter

    @pytest.mark.asyncio
    async def test_export_memories(self, mock_adapter):
        """测试便捷导出函数"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            stats = await export_memories(mock_adapter, temp_path)
            assert isinstance(stats, ExportStats)
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_import_memories(self, mock_adapter):
        """测试便捷导入函数"""
        # 创建测试文件
        export_data = {
            "version": "1.0",
            "persona_id": "default",
            "export_time": "2024-01-01T00:00:00",
            "entries": [{"id": "mem_001", "content": "测试", "metadata": {}}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(export_data, f)
            temp_path = Path(f.name)

        try:
            stats = await import_memories(mock_adapter, temp_path)
            assert isinstance(stats, ImportStats)
            assert stats.imported_count == 1
        finally:
            temp_path.unlink()

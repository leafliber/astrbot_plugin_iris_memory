"""
Web 模块单元测试

测试 WebService 和 StandaloneWebServer 的核心功能。
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict, List, Optional

from iris_memory.web.web_service import WebService


# ── Fixtures ──

class MockCollection:
    """Mock ChromaDB Collection"""

    def __init__(self, data: Optional[Dict] = None):
        self._data = data or {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

    def count(self) -> int:
        return len(self._data["ids"])

    def get(self, where=None, include=None):
        return self._data

    def query(self, query_embeddings=None, n_results=10, where=None, include=None):
        return {
            "ids": [self._data.get("ids", [])[:n_results]],
            "documents": [self._data.get("documents", [])[:n_results]],
            "metadatas": [self._data.get("metadatas", [])[:n_results]],
            "embeddings": [[]],
            "distances": [[0.1] * min(n_results, len(self._data.get("ids", [])))],
        }


class MockChromaManager:
    """Mock ChromaManager"""

    def __init__(self, data: Optional[Dict] = None):
        self.collection = MockCollection(data)
        self._is_ready = True
        self._data = data or {"ids": [], "documents": [], "metadatas": []}

    @property
    def is_ready(self):
        return self._is_ready

    def _build_where_clause(self, filters):
        if len(filters) > 1:
            return {"$and": [{k: v} for k, v in filters.items()]}
        return filters

    async def query_memories(self, query_text="", user_id="", group_id=None,
                             top_k=10, storage_layer=None):
        return []

    async def count_memories(self, user_id="", group_id=None, storage_layer=None):
        return self.collection.count()

    async def delete_memory(self, memory_id: str) -> bool:
        """只有当记忆存在于 ChromaDB 时才返回 True"""
        return memory_id in self._data.get("ids", [])

    async def add_memory(self, memory) -> str:
        return getattr(memory, "id", "test-id")


class MockKGStorage:
    """Mock KGStorage"""

    def __init__(self):
        self._lock = AsyncMock()
        self._conn = MagicMock()
        self._node_cache = {}

    async def search_nodes(self, query="", user_id=None, group_id=None,
                           node_type=None, limit=20):
        return []

    async def get_node(self, node_id: str):
        return None

    async def get_neighbors(self, node_id: str, limit=50):
        return []

    async def get_stats(self, user_id=None, group_id=None):
        return {"nodes": 5, "edges": 10}

    async def delete_by_memory_id(self, memory_id: str) -> int:
        return 1

    def _invalidate_cache(self, **kwargs):
        pass


class MockKGModule:
    """Mock KnowledgeGraphModule"""

    def __init__(self):
        self.enabled = True
        self.storage = MockKGStorage()

    async def get_stats(self, user_id=None, group_id=None):
        return await self.storage.get_stats(user_id, group_id)

    async def delete_user_data(self, user_id: str, group_id=None):
        return 1

    async def delete_all(self):
        return 5


class MockMemory:
    """Mock Memory 对象 - 模拟 iris_memory.models.memory.Memory"""

    def __init__(self, memory_id: str = "wm-1", content: str = "工作记忆内容"):
        self.id = memory_id
        self.content = content
        self.user_id = "user1"
        self.group_id = None
        self.sender_name = "User1"
        # 使用简单属性而不是 Mock 对象，以便 _memory_to_dict 正确处理
        self.type = Mock()
        self.type.value = "fact"
        self.storage_layer = Mock()
        self.storage_layer.value = "working"
        self.scope = Mock()
        self.scope.value = "user_private"
        self.confidence = 0.8
        self.importance_score = 0.6
        self.created_time = datetime.now()
        self.summary = "工作记忆摘要"


class MockSessionManager:
    """Mock SessionManager"""

    def __init__(self):
        # 使用 MockMemory 而不是普通的 Mock
        self.working_memory_cache = {"user1:private": [MockMemory("wm-1", "工作记忆1")]}

    def get_all_sessions(self):
        return {"user1:private": {"created": "2024-01-01"}, "user2:group1": {"created": "2024-01-02"}}

    async def remove_working_memory(self, user_id: str, group_id: str, memory_id: str) -> bool:
        """模拟删除工作记忆"""
        session_key = f"{user_id}:{group_id or 'private'}"
        if session_key in self.working_memory_cache:
            memories = self.working_memory_cache[session_key]
            original_count = len(memories)
            self.working_memory_cache[session_key] = [
                m for m in memories if m.id != memory_id
            ]
            return len(memories) > len(self.working_memory_cache[session_key])
        return False


@pytest.fixture
def sample_memory_data():
    """样例记忆数据"""
    return {
        "ids": ["mem-1", "mem-2", "mem-3"],
        "documents": ["我喜欢编程", "今天心情很好", "Python是最好的语言"],
        "metadatas": [
            {"user_id": "user1", "group_id": "", "sender_name": "Alice",
             "type": "fact", "storage_layer": "episodic", "scope": "user_private",
             "confidence": 0.8, "importance_score": 0.6, "created_time": "2024-01-15T10:00:00",
             "summary": "用户喜欢编程"},
            {"user_id": "user1", "group_id": "", "sender_name": "Alice",
             "type": "emotion", "storage_layer": "working", "scope": "user_private",
             "confidence": 0.9, "importance_score": 0.5, "created_time": "2024-01-16T10:00:00",
             "summary": "心情好"},
            {"user_id": "user2", "group_id": "group1", "sender_name": "Bob",
             "type": "fact", "storage_layer": "episodic", "scope": "group_shared",
             "confidence": 0.7, "importance_score": 0.4, "created_time": "2024-01-17T10:00:00",
             "summary": "Python观点"},
        ],
        "embeddings": [],
    }


@pytest.fixture
def mock_service(sample_memory_data):
    """Mock MemoryService"""
    service = MagicMock()
    service.chroma_manager = MockChromaManager(sample_memory_data)
    service.session_manager = MockSessionManager()
    service.kg = MockKGModule()
    service._user_personas = {"user1": Mock()}
    service._user_emotional_states = {}
    service.health_check.return_value = {"status": "healthy", "modules": {}}
    return service


@pytest.fixture
def web_service(mock_service):
    """WebService实例"""
    return WebService(mock_service)


# ================================================================
# Dashboard Tests
# ================================================================

class TestDashboardStats:
    """仪表盘统计测试"""

    @pytest.mark.asyncio
    async def test_get_dashboard_stats(self, web_service):
        """测试获取仪表盘统计"""
        stats = await web_service.get_dashboard_stats()
        assert "system" in stats
        assert "memories" in stats
        assert "knowledge_graph" in stats
        assert "health" in stats
        assert stats["health"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_system_stats(self, web_service):
        """测试系统统计"""
        stats = await web_service._get_system_stats()
        assert stats["total_users"] == 2  # user1, user2
        assert stats["total_sessions"] == 2
        assert stats["total_personas"] == 1

    @pytest.mark.asyncio
    async def test_memory_overview(self, web_service):
        """测试记忆总览 - 包含工作记忆和ChromaDB记忆"""
        overview = await web_service._get_memory_overview()
        # 3 from ChromaDB + 1 from working_memory_cache
        assert overview["total_count"] == 4
        # 工作记忆来自 SessionManager.working_memory_cache
        assert overview["by_layer"]["working"] >= 1

    @pytest.mark.asyncio
    async def test_kg_overview(self, web_service):
        """测试知识图谱总览"""
        overview = await web_service._get_kg_overview()
        assert overview["enabled"] is True
        assert overview["nodes"] == 5
        assert overview["edges"] == 10

    @pytest.mark.asyncio
    async def test_kg_overview_disabled(self, mock_service):
        """测试知识图谱禁用时"""
        mock_service.kg = None
        ws = WebService(mock_service)
        overview = await ws._get_kg_overview()
        assert overview["enabled"] is False
        assert overview["nodes"] == 0

    @pytest.mark.asyncio
    async def test_memory_trend(self, web_service):
        """测试记忆趋势"""
        trend = await web_service.get_memory_trend(days=7)
        assert len(trend) == 7
        for item in trend:
            assert "date" in item
            assert "count" in item

    @pytest.mark.asyncio
    async def test_memory_trend_empty(self, mock_service):
        """测试空数据的趋势"""
        mock_service.chroma_manager = MockChromaManager()
        ws = WebService(mock_service)
        trend = await ws.get_memory_trend(days=7)
        # 空数据也应该返回7天的结构
        assert isinstance(trend, list)


# ================================================================
# Memory Management Tests
# ================================================================

class TestMemoryManagement:
    """记忆管理测试"""

    @pytest.mark.asyncio
    async def test_search_memories_web_no_query(self, web_service):
        """测试无搜索词的记忆列表 - 包含 ChromaDB 记忆和工作记忆"""
        result = await web_service.search_memories_web()
        assert "items" in result
        assert "total" in result
        # 3 from ChromaDB + 1 from working memory
        assert result["total"] == 4

    @pytest.mark.asyncio
    async def test_search_memories_web_with_user(self, web_service):
        """测试按用户过滤"""
        result = await web_service.search_memories_web(user_id="user1")
        assert "items" in result
        assert result["total"] >= 0

    @pytest.mark.asyncio
    async def test_search_memories_pagination(self, web_service):
        """测试分页"""
        result = await web_service.search_memories_web(page=1, page_size=2)
        assert result["page"] == 1
        assert result["page_size"] == 2
        assert len(result["items"]) <= 2

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, web_service):
        """测试删除单条记忆"""
        success, msg = await web_service.delete_memory_by_id("mem-1")
        assert success is True
        assert "成功" in msg

    @pytest.mark.asyncio
    async def test_delete_memory_no_storage(self, mock_service):
        """测试存储未就绪时删除 - 会尝试删除工作记忆"""
        mock_service.chroma_manager = None
        ws = WebService(mock_service)
        # wm-1 存在于工作记忆中，应该删除成功
        success, msg = await ws.delete_memory_by_id("wm-1")
        assert success is True
        assert "成功" in msg

    @pytest.mark.asyncio
    async def test_delete_memory_not_found_anywhere(self, mock_service):
        """测试记忆在任何地方都不存在时删除"""
        mock_service.chroma_manager = None
        # 清空工作记忆
        mock_service.session_manager.working_memory_cache = {}
        ws = WebService(mock_service)
        success, msg = await ws.delete_memory_by_id("non-existent")
        assert success is False
        assert "不存在" in msg or "删除失败" in msg

    @pytest.mark.asyncio
    async def test_batch_delete(self, web_service):
        """测试批量删除"""
        result = await web_service.batch_delete_memories(["mem-1", "mem-2"])
        assert result["success_count"] == 2
        assert result["fail_count"] == 0

    @pytest.mark.asyncio
    async def test_delete_working_memory_success(self, web_service):
        """测试删除工作记忆成功"""
        # 先验证工作记忆存在
        result = await web_service.search_memories_web(storage_layer="working")
        initial_count = result["total"]
        assert initial_count >= 1

        # 删除工作记忆
        success, msg = await web_service.delete_memory_by_id("wm-1")
        assert success is True
        assert "成功" in msg

    @pytest.mark.asyncio
    async def test_delete_working_memory_not_found(self, web_service):
        """测试删除不存在的工作记忆"""
        success, msg = await web_service.delete_memory_by_id("non-existent-id")
        assert success is False
        assert "不存在" in msg or "删除失败" in msg

    @pytest.mark.asyncio
    async def test_batch_delete_includes_working_memory(self, web_service):
        """测试批量删除包含工作记忆"""
        # 包含 ChromaDB 记忆和工作记忆
        result = await web_service.batch_delete_memories(["mem-1", "wm-1"])
        assert result["success_count"] == 2
        assert result["fail_count"] == 0

    @pytest.mark.asyncio
    async def test_delete_chromadb_memory_first(self, mock_service):
        """测试优先删除 ChromaDB 记忆"""
        ws = WebService(mock_service)
        # mem-1 存在于 ChromaDB，应该优先从 ChromaDB 删除
        success, msg = await ws.delete_memory_by_id("mem-1")
        assert success is True
        assert "成功" in msg


# ================================================================
# Working Memory Repository Tests
# ================================================================

class TestWorkingMemoryRepository:
    """工作记忆仓库测试"""

    @pytest.mark.asyncio
    async def test_list_all_includes_working_memory(self, mock_service):
        """测试 list_all 包含工作记忆"""
        from iris_memory.web.data.memory_repo import MemoryRepositoryImpl

        repo = MemoryRepositoryImpl(mock_service)
        result = await repo.list_all()

        # 应该包含 ChromaDB 的 3 条 + 工作记忆的 1 条
        assert result["total"] == 4

    @pytest.mark.asyncio
    async def test_list_all_filter_by_working_layer(self, mock_service):
        """测试按 working 层过滤"""
        from iris_memory.web.data.memory_repo import MemoryRepositoryImpl

        repo = MemoryRepositoryImpl(mock_service)
        result = await repo.list_all(storage_layer="working")

        # 应该只返回工作记忆
        assert result["total"] == 1
        assert result["items"][0]["storage_layer"] == "working"

    @pytest.mark.asyncio
    async def test_delete_working_memory_via_repo(self, mock_service):
        """测试通过仓库删除工作记忆"""
        from iris_memory.web.data.memory_repo import MemoryRepositoryImpl

        repo = MemoryRepositoryImpl(mock_service)

        # 删除工作记忆
        success, msg = await repo.delete("wm-1")
        assert success is True
        assert "成功" in msg

    @pytest.mark.asyncio
    async def test_delete_chromadb_memory_via_repo(self, mock_service):
        """测试通过仓库删除 ChromaDB 记忆"""
        from iris_memory.web.data.memory_repo import MemoryRepositoryImpl

        repo = MemoryRepositoryImpl(mock_service)

        # 删除 ChromaDB 记忆
        success, msg = await repo.delete("mem-1")
        assert success is True
        assert "成功" in msg

    @pytest.mark.asyncio
    async def test_batch_delete_mixed_memories(self, mock_service):
        """测试批量删除混合记忆（ChromaDB + 工作记忆）"""
        from iris_memory.web.data.memory_repo import MemoryRepositoryImpl

        repo = MemoryRepositoryImpl(mock_service)

        # 批量删除包含两种类型的记忆
        result = await repo.batch_delete(["mem-1", "mem-2", "wm-1"])
        assert result["success_count"] == 3
        assert result["fail_count"] == 0


# ================================================================
# Knowledge Graph Tests
# ================================================================

class TestKnowledgeGraph:
    """知识图谱测试"""

    @pytest.mark.asyncio
    async def test_search_kg_nodes_empty(self, web_service):
        """测试搜索KG节点（空结果）"""
        nodes = await web_service.search_kg_nodes(query="test")
        assert isinstance(nodes, list)

    @pytest.mark.asyncio
    async def test_search_kg_disabled(self, mock_service):
        """测试KG禁用时搜索"""
        mock_service.kg = None
        ws = WebService(mock_service)
        nodes = await ws.search_kg_nodes(query="test")
        assert nodes == []

    @pytest.mark.asyncio
    async def test_get_kg_graph_data_empty(self, web_service):
        """测试获取空图谱数据"""
        graph = await web_service.get_kg_graph_data(user_id="user1")
        assert "nodes" in graph
        assert "edges" in graph

    @pytest.mark.asyncio
    async def test_delete_kg_node_disabled(self, mock_service):
        """测试KG禁用时删除节点"""
        mock_service.kg = None
        ws = WebService(mock_service)
        success, msg = await ws.delete_kg_node("node-1")
        assert success is False
        assert "未启用" in msg

    @pytest.mark.asyncio
    async def test_delete_kg_edge_disabled(self, mock_service):
        """测试KG禁用时删除边"""
        mock_service.kg = None
        ws = WebService(mock_service)
        success, msg = await ws.delete_kg_edge("edge-1")
        assert success is False
        assert "未启用" in msg


# ================================================================
# Export Tests
# ================================================================

class TestExport:
    """数据导出测试"""

    @pytest.mark.asyncio
    async def test_export_memories_json(self, web_service):
        """测试JSON格式导出记忆"""
        data, content_type, filename = await web_service.export_memories(format="json")
        assert content_type == "application/json"
        assert filename.endswith(".json")
        parsed = json.loads(data)
        assert "memories" in parsed
        assert parsed["total_count"] == 3

    @pytest.mark.asyncio
    async def test_export_memories_csv(self, web_service):
        """测试CSV格式导出记忆"""
        data, content_type, filename = await web_service.export_memories(format="csv")
        assert content_type == "text/csv"
        assert filename.endswith(".csv")
        assert "id" in data  # CSV header
        assert "content" in data

    @pytest.mark.asyncio
    async def test_export_memories_empty(self, mock_service):
        """测试空数据导出"""
        mock_service.chroma_manager = MockChromaManager()
        ws = WebService(mock_service)
        data, content_type, filename = await ws.export_memories(format="json")
        parsed = json.loads(data)
        assert parsed["memories"] == []

    @pytest.mark.asyncio
    async def test_export_memories_no_storage(self, mock_service):
        """测试存储未就绪时导出"""
        mock_service.chroma_manager = None
        ws = WebService(mock_service)
        data, content_type, filename = await ws.export_memories()
        assert "text/plain" in content_type

    @pytest.mark.asyncio
    async def test_export_kg_json(self, web_service):
        """测试JSON格式导出KG"""
        # Mock the KG storage for export
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        web_service._service.kg.storage._conn = mock_conn

        data, content_type, filename = await web_service.export_kg(format="json")
        assert content_type == "application/json"
        assert filename.endswith(".json")

    @pytest.mark.asyncio
    async def test_export_kg_disabled(self, mock_service):
        """测试KG禁用时导出"""
        mock_service.kg = None
        ws = WebService(mock_service)
        data, content_type, filename = await ws.export_kg()
        parsed = json.loads(data)
        assert "error" in parsed


# ================================================================
# Import Tests
# ================================================================

class TestImport:
    """数据导入测试"""

    @pytest.mark.asyncio
    async def test_import_memories_json(self, web_service):
        """测试JSON格式导入记忆"""
        import_data = json.dumps({
            "memories": [
                {"content": "测试记忆1", "user_id": "user1"},
                {"content": "测试记忆2", "user_id": "user1", "type": "fact"},
            ]
        })
        result = await web_service.import_memories(import_data, format="json")
        assert result["success_count"] == 2
        assert result["fail_count"] == 0

    @pytest.mark.asyncio
    async def test_import_memories_csv(self, web_service):
        """测试CSV格式导入记忆"""
        csv_data = "content,user_id,type\n测试记忆1,user1,fact\n测试记忆2,user2,emotion\n"
        result = await web_service.import_memories(csv_data, format="csv")
        assert result["success_count"] == 2

    @pytest.mark.asyncio
    async def test_import_memories_invalid_content(self, web_service):
        """测试缺少content的导入"""
        import_data = json.dumps({
            "memories": [{"user_id": "user1"}]  # 无 content
        })
        result = await web_service.import_memories(import_data, format="json")
        assert result["fail_count"] == 1
        assert any("content" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_import_memories_invalid_type(self, web_service):
        """测试无效type的导入"""
        import_data = json.dumps({
            "memories": [{"content": "test", "type": "invalid_type"}]
        })
        result = await web_service.import_memories(import_data, format="json")
        assert result["fail_count"] == 1

    @pytest.mark.asyncio
    async def test_import_memories_no_storage(self, mock_service):
        """测试存储未就绪时导入"""
        mock_service.chroma_manager = None
        ws = WebService(mock_service)
        result = await ws.import_memories('{"memories":[]}', format="json")
        assert any("未就绪" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_import_memories_too_large(self, web_service):
        """测试文件过大"""
        # 51MB data
        result = await web_service.import_memories("x" * (51 * 1024 * 1024), format="json")
        assert any("过大" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_import_kg_json(self, web_service):
        """测试JSON格式导入KG"""
        from iris_memory.knowledge_graph.kg_models import KGNode, KGEdge

        # 设置 storage mock 的 upsert 方法
        web_service._service.kg.storage.upsert_node = AsyncMock(
            return_value=KGNode(name="test")
        )
        web_service._service.kg.storage.upsert_edge = AsyncMock(
            return_value=KGEdge(source_id="n1", target_id="n2")
        )

        import_data = json.dumps({
            "nodes": [
                {"name": "张三", "node_type": "person", "user_id": "user1"},
                {"name": "北京", "node_type": "location"},
            ],
            "edges": [
                {"source_id": "n1", "target_id": "n2", "relation_type": "lives_in"},
            ],
        })
        result = await web_service.import_kg(import_data, format="json")
        assert result["nodes_imported"] == 2
        assert result["edges_imported"] == 1

    @pytest.mark.asyncio
    async def test_import_kg_invalid_node(self, web_service):
        """测试无效节点导入"""
        web_service._service.kg.storage.upsert_node = AsyncMock()

        import_data = json.dumps({
            "nodes": [{"node_type": "person"}],  # 缺少 name
            "edges": [],
        })
        result = await web_service.import_kg(import_data, format="json")
        assert any("name" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_import_kg_invalid_edge(self, web_service):
        """测试无效边导入"""
        import_data = json.dumps({
            "nodes": [],
            "edges": [{"source_id": "n1"}],  # 缺少 target_id
        })
        result = await web_service.import_kg(import_data, format="json")
        assert any("target_id" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_import_kg_disabled(self, mock_service):
        """测试KG禁用时导入"""
        mock_service.kg = None
        ws = WebService(mock_service)
        result = await ws.import_kg('{"nodes":[],"edges":[]}', format="json")
        assert any("未启用" in e for e in result["errors"])


# ================================================================
# Validation Tests
# ================================================================

class TestValidation:
    """数据验证测试"""

    def test_validate_memory_import_valid(self, web_service):
        """测试有效记忆数据验证"""
        valid, err = web_service._validate_memory_import({"content": "test"})
        assert valid is True

    def test_validate_memory_import_no_content(self, web_service):
        """测试无content验证"""
        valid, err = web_service._validate_memory_import({})
        assert valid is False
        assert "content" in err

    def test_validate_memory_import_long_content(self, web_service):
        """测试过长content验证"""
        valid, err = web_service._validate_memory_import({"content": "x" * 10001})
        assert valid is False
        assert "过长" in err

    def test_validate_memory_import_invalid_layer(self, web_service):
        """测试无效storage_layer验证"""
        valid, err = web_service._validate_memory_import({"content": "test", "storage_layer": "invalid"})
        assert valid is False
        assert "storage_layer" in err

    def test_validate_kg_node_import_valid(self, web_service):
        """测试有效KG节点验证"""
        valid, err = web_service._validate_kg_node_import({"name": "test"})
        assert valid is True

    def test_validate_kg_node_import_no_name(self, web_service):
        """测试无name验证"""
        valid, err = web_service._validate_kg_node_import({})
        assert valid is False

    def test_validate_kg_edge_import_valid(self, web_service):
        """测试有效KG边验证"""
        valid, err = web_service._validate_kg_edge_import({"source_id": "a", "target_id": "b"})
        assert valid is True

    def test_validate_kg_edge_import_no_source(self, web_service):
        """测试无source_id验证"""
        valid, err = web_service._validate_kg_edge_import({"target_id": "b"})
        assert valid is False

    def test_validate_kg_edge_import_invalid_relation(self, web_service):
        """测试无效relation_type验证"""
        valid, err = web_service._validate_kg_edge_import(
            {"source_id": "a", "target_id": "b", "relation_type": "invalid"}
        )
        assert valid is False


# ================================================================
# CSV Conversion Tests
# ================================================================

class TestCSVConversion:
    """CSV 转换测试"""

    def test_memories_to_csv_empty(self, web_service):
        """测试空数据CSV"""
        csv = web_service._memories_to_csv([])
        assert "id" in csv

    def test_memories_to_csv_with_data(self, web_service):
        """测试有数据CSV"""
        items = [
            {"id": "1", "content": "测试记忆", "user_id": "u1", "type": "fact"},
            {"id": "2", "content": "另一条记忆", "user_id": "u2", "type": "emotion"},
        ]
        csv = web_service._memories_to_csv(items)
        assert "测试记忆" in csv
        assert "另一条记忆" in csv

    def test_memories_to_csv_special_chars(self, web_service):
        """测试特殊字符CSV"""
        items = [{"id": "1", "content": 'he said "hello, world"', "user_id": "u1"}]
        csv = web_service._memories_to_csv(items)
        assert "hello" in csv

    def test_kg_nodes_to_csv_empty(self, web_service):
        """测试空KG节点CSV"""
        csv = web_service._kg_nodes_to_csv([])
        assert "id" in csv

    def test_kg_edges_to_csv_empty(self, web_service):
        """测试空KG边CSV"""
        csv = web_service._kg_edges_to_csv([])
        assert "id" in csv

    def test_parse_csv_memories(self, web_service):
        """测试CSV记忆解析"""
        csv = "content,user_id\ntest,u1\nhello,u2\n"
        items = web_service._parse_csv_memories(csv)
        assert len(items) == 2

    def test_parse_json_memories_list(self, web_service):
        """测试JSON列表格式解析"""
        data = json.dumps([{"content": "a"}, {"content": "b"}])
        items = web_service._parse_json_memories(data)
        assert len(items) == 2

    def test_parse_json_memories_dict(self, web_service):
        """测试JSON字典格式解析"""
        data = json.dumps({"memories": [{"content": "a"}]})
        items = web_service._parse_json_memories(data)
        assert len(items) == 1

    def test_parse_json_memories_invalid(self, web_service):
        """测试无效JSON解析"""
        items = web_service._parse_json_memories("invalid json")
        assert items == []

    def test_parse_csv_kg(self, web_service):
        """测试CSV KG解析"""
        csv = "# NODES\nid,name,node_type\nn1,张三,person\n# EDGES\nid,source_id,target_id\ne1,n1,n2\n"
        nodes, edges = web_service._parse_csv_kg(csv)
        assert len(nodes) == 1
        assert len(edges) == 1

    def test_parse_json_kg(self, web_service):
        """测试JSON KG解析"""
        data = json.dumps({"nodes": [{"name": "test"}], "edges": [{"source_id": "a", "target_id": "b"}]})
        nodes, edges = web_service._parse_json_kg(data)
        assert len(nodes) == 1
        assert len(edges) == 1

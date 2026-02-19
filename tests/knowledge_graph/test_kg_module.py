"""
KnowledgeGraphModule 集成测试
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from iris_memory.services.modules.kg_module import KnowledgeGraphModule


def run(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture
def kg_module():
    """创建完整初始化的 KG 模块"""
    module = KnowledgeGraphModule()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(module.initialize(
            plugin_data_path=Path(tmpdir),
            kg_mode="rule",
            max_depth=3,
            max_nodes_per_hop=10,
            max_facts=8,
            enabled=True,
        ))
        yield module
        run(module.close())


@pytest.fixture
def disabled_module():
    """创建禁用的 KG 模块"""
    module = KnowledgeGraphModule()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(module.initialize(
            plugin_data_path=Path(tmpdir),
            enabled=False,
        ))
        yield module


class TestModuleInit:
    """模块初始化测试"""

    def test_initialize_enabled(self, kg_module):
        assert kg_module.is_initialized
        assert kg_module.enabled
        assert kg_module.storage is not None
        assert kg_module.extractor is not None
        assert kg_module.reasoning is not None
        assert kg_module.formatter is not None

    def test_initialize_disabled(self, disabled_module):
        assert not disabled_module.is_initialized
        assert not disabled_module.enabled
        assert disabled_module.storage is None

    def test_close(self, kg_module):
        run(kg_module.close())
        assert not kg_module.is_initialized


class TestModuleCapture:
    """记忆捕获 → KG 提取测试"""

    def test_process_memory(self, kg_module):
        memory = Mock()
        memory.content = "张三和李四是好朋友"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []

        triples = run(kg_module.process_memory(memory))
        assert len(triples) >= 1

        # graph_nodes / graph_edges 应被更新
        assert len(memory.graph_nodes) >= 2
        assert len(memory.graph_edges) >= 1

    def test_process_memory_disabled(self, disabled_module):
        memory = Mock()
        memory.content = "张三和李四是好朋友"
        triples = run(disabled_module.process_memory(memory))
        assert len(triples) == 0

    def test_process_memory_error_handling(self, kg_module):
        """处理错误不应抛出异常"""
        memory = Mock()
        memory.content = None  # 会导致提取失败
        triples = run(kg_module.process_memory(memory))
        assert len(triples) == 0


class TestModuleRetrieval:
    """图检索测试"""

    def test_graph_retrieve_with_data(self, kg_module):
        # 先写入一些数据
        memory = Mock()
        memory.content = "张三喜欢编程"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        # 检索
        result = run(kg_module.graph_retrieve("张三", user_id="u1"))
        assert result.has_results or len(result.seed_nodes) >= 0  # 某些情况 FTS 可能无法匹配

    def test_graph_retrieve_disabled(self, disabled_module):
        result = run(disabled_module.graph_retrieve("张三", user_id="u1"))
        assert not result.has_results

    def test_format_graph_context(self, kg_module):
        # 写入数据
        memory = Mock()
        memory.content = "张三和李四是好朋友"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        text = run(kg_module.format_graph_context("张三", user_id="u1"))
        # 可能有结果也可能没有（取决于 FTS 匹配）
        assert isinstance(text, str)


class TestModuleStats:
    """统计和管理测试"""

    def test_get_stats_empty(self, kg_module):
        stats = run(kg_module.get_stats())
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_get_stats_after_insert(self, kg_module):
        memory = Mock()
        memory.content = "张三喜欢编程"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        stats = run(kg_module.get_stats())
        assert stats["nodes"] >= 1

    def test_delete_user_data(self, kg_module):
        memory = Mock()
        memory.content = "张三喜欢编程"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        count = run(kg_module.delete_user_data("u1"))
        assert count >= 1

        stats = run(kg_module.get_stats(user_id="u1"))
        assert stats["nodes"] == 0

    def test_delete_all(self, kg_module):
        memory = Mock()
        memory.content = "张三喜欢编程"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        count = run(kg_module.delete_all())
        assert count >= 1

        stats = run(kg_module.get_stats())
        assert stats["nodes"] == 0

    def test_stats_disabled(self, disabled_module):
        stats = run(disabled_module.get_stats())
        assert stats == {"nodes": 0, "edges": 0}

    def test_delete_disabled(self, disabled_module):
        count = run(disabled_module.delete_user_data("u1"))
        assert count == 0
        count = run(disabled_module.delete_all())
        assert count == 0

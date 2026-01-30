"""
ChromaManager测试
测试Chroma向量数据库管理器的核心功能
"""

import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
import numpy as np

from iris_memory.storage.chroma_manager import ChromaManager
from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer, MemoryType, ModalityType


@pytest.fixture
def mock_config():
    """模拟配置对象"""
    class MockConfig:
        def __init__(self):
            self.chroma_config = {
                'embedding_model': 'BAAI/bge-m3',
                'embedding_dimension': 1024,
                'collection_name': 'test_collection',
                'auto_detect_dimension': True
            }

    return MockConfig()


@pytest.fixture
def mock_data_path(tmp_path):
    """临时数据目录"""
    return tmp_path / "test_data"


@pytest.fixture
def mock_plugin_context():
    """模拟插件上下文"""
    context = Mock()
    return context


@pytest_asyncio.fixture
async def chroma_manager(mock_config, mock_data_path, mock_plugin_context):
    """ChromaManager实例"""
    manager = ChromaManager(mock_config, mock_data_path, mock_plugin_context)
    return manager


@pytest.fixture
def chroma_manager_sync(mock_config, mock_data_path, mock_plugin_context):
    """ChromaManager同步实例（用于同步测试）"""
    manager = ChromaManager(mock_config, mock_data_path, mock_plugin_context)
    return manager


class TestChromaManagerInit:
    """测试初始化功能"""
    
    @pytest.mark.asyncio
    async def test_init_basic(self, chroma_manager):
        """测试基本初始化"""
        assert chroma_manager.config is not None
        assert chroma_manager.data_path is not None
        assert chroma_manager.collection is None
        assert chroma_manager.client is None
    
    @pytest.mark.asyncio
    async def test_get_config_nested_key(self, chroma_manager):
        """测试嵌套配置读取"""
        model = chroma_manager._get_config("chroma_config.embedding_model", "default")
        assert model == "BAAI/bge-m3"
        
        dim = chroma_manager._get_config("chroma_config.embedding_dimension", 512)
        assert dim == 1024
    
    @pytest.mark.asyncio
    async def test_get_config_missing_key(self, chroma_manager):
        """测试缺失配置键"""
        value = chroma_manager._get_config("missing.key", "default_value")
        assert value == "default_value"

    @pytest.mark.asyncio
    async def test_get_config_with_dict(self, chroma_manager):
        """测试字典配置读取"""
        chroma_manager.config.test_dict = {"key1": "value1", "key2": "value2"}
        value = chroma_manager._get_config("test_dict.key1", "default")
        assert value == "value1"


class TestChromaManagerInitialization:
    """测试Chroma客户端初始化"""
    
    @pytest.mark.asyncio
    async def test_initialize_with_chromadb_available(
        self, chroma_manager, monkeypatch
    ):
        """测试ChromaDB可用时的初始化"""
        # Mock chromadb
        mock_client = MagicMock()
        mock_collection = MagicMock()

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Not exists")
        mock_client.create_collection.return_value = mock_collection

        # Mock Settings
        mock_settings = MagicMock()

        monkeypatch.setattr("iris_memory.storage.chroma_manager.chromadb", mock_chromadb)
        monkeypatch.setattr("iris_memory.storage.chroma_manager.Settings", mock_settings)

        # Mock embedding manager
        chroma_manager.embedding_manager = MagicMock()
        chroma_manager.embedding_manager.initialize = AsyncMock()
        chroma_manager.embedding_manager.get_dimension = MagicMock(return_value=1024)
        chroma_manager.embedding_manager.get_model = MagicMock(return_value="BAAI/bge-m3")

        await chroma_manager.initialize()

        assert chroma_manager.client is not None
        assert chroma_manager.collection is not None
        mock_chromadb.PersistentClient.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_without_chromadb(self, chroma_manager, monkeypatch):
        """测试ChromaDB不可用时抛出错误"""
        monkeypatch.setattr("iris_memory.storage.chroma_manager.chromadb", None)
        
        with pytest.raises(ImportError, match="chromadb is not installed"):
            await chroma_manager.initialize()
    
    @pytest.mark.asyncio
    async def test_initialize_existing_collection(
        self, chroma_manager, monkeypatch
    ):
        """测试使用现有集合"""
        mock_client = MagicMock()
        mock_existing_collection = MagicMock()
        mock_client.get_collection.return_value = mock_existing_collection

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_settings = MagicMock()

        monkeypatch.setattr("iris_memory.storage.chroma_manager.chromadb", mock_chromadb)
        monkeypatch.setattr("iris_memory.storage.chroma_manager.Settings", mock_settings)

        # Mock embedding manager
        chroma_manager.embedding_manager = MagicMock()
        chroma_manager.embedding_manager.initialize = AsyncMock()
        chroma_manager.embedding_manager.detect_existing_dimension = AsyncMock(return_value=768)
        chroma_manager.embedding_manager.get_dimension = MagicMock(return_value=768)
        chroma_manager.embedding_manager.get_model = MagicMock(return_value="existing_model")

        await chroma_manager.initialize()

        assert chroma_manager.collection == mock_existing_collection
        assert chroma_manager.embedding_dimension == 768

    @pytest.mark.asyncio
    async def test_initialize_auto_detect_dimension(
        self, chroma_manager, monkeypatch
    ):
        """测试自动检测维度"""
        mock_client = MagicMock()
        mock_existing_collection = MagicMock()
        mock_client.get_collection.return_value = mock_existing_collection

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_settings = MagicMock()

        monkeypatch.setattr("iris_memory.storage.chroma_manager.chromadb", mock_chromadb)
        monkeypatch.setattr("iris_memory.storage.chroma_manager.Settings", mock_settings)

        # Mock embedding manager with dimension detection
        chroma_manager.embedding_manager = MagicMock()
        chroma_manager.embedding_manager.initialize = AsyncMock()
        chroma_manager.embedding_manager.detect_existing_dimension = AsyncMock(return_value=1536)
        chroma_manager.embedding_manager.get_dimension = MagicMock(return_value=1536)
        chroma_manager.embedding_manager.get_model = MagicMock(return_value="BAAI/bge-m3")

        chroma_manager.auto_detect_dimension = True
        chroma_manager.embedding_dimension = 1024  # 初始值

        await chroma_manager.initialize()

        assert chroma_manager.embedding_dimension == 1536


class TestChromaManagerAddMemory:
    """测试添加记忆功能"""
    
    @pytest.mark.asyncio
    async def test_add_memory_success(
        self, chroma_manager, monkeypatch
    ):
        """测试成功添加记忆"""
        # Mock collection
        mock_collection = MagicMock()
        chroma_manager.collection = mock_collection
        
        # Mock embedding generation
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        # Create test memory
        memory = Memory(
            id="test_001",
            content="This is a test memory",
            user_id="user_123",
            group_id="group_456",
            type=MemoryType.FACT,
            modality=ModalityType.TEXT,
            storage_layer=StorageLayer.EPISODIC
        )
        
        memory_id = await chroma_manager.add_memory(memory)
        
        assert memory_id == "test_001"
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args[1]['ids'] == [memory.id]
        assert len(call_args[1]['embeddings']) == 1
    
    @pytest.mark.asyncio
    async def test_add_memory_with_existing_embedding(
        self, chroma_manager
    ):
        """测试添加已有嵌入的记忆"""
        mock_collection = MagicMock()
        chroma_manager.collection = mock_collection
        
        # Create memory with embedding
        memory = Memory(
            id="test_002",
            content="Test content with embedding",
            user_id="user_123",
            group_id="group_456",
            type=MemoryType.FACT,
            modality=ModalityType.TEXT
        )
        memory.embedding = np.random.rand(1024)
        
        memory_id = await chroma_manager.add_memory(memory)
        
        assert memory_id == "test_002"
        mock_collection.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_memory_metadata_construction(
        self, chroma_manager, monkeypatch
    ):
        """测试元数据构建"""
        mock_collection = MagicMock()
        chroma_manager.collection = mock_collection
        
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        memory = Memory(
            id="test_003",
            content="Test metadata",
            user_id="user_123",
            group_id="group_456",
            type=MemoryType.FACT,
            modality=ModalityType.TEXT,
            storage_layer=StorageLayer.SEMANTIC,
            quality_level=5,
            sensitivity_level=2,
            rif_score=0.8,
            importance_score=0.9,
            is_user_requested=True
        )
        memory.metadata = {"custom_key": "custom_value"}
        
        await chroma_manager.add_memory(memory)
        
        call_args = mock_collection.add.call_args
        metadata = call_args[1]['metadatas'][0]
        
        assert metadata['user_id'] == 'user_123'
        assert metadata['group_id'] == 'group_456'
        assert metadata['type'] == 'fact'
        assert metadata['custom_key'] == 'custom_value'
        assert metadata['rif_score'] == 0.8
        assert metadata['is_user_requested'] == True
    
    @pytest.mark.asyncio
    async def test_add_memory_private_chat(
        self, chroma_manager, monkeypatch
    ):
        """测试私聊场景（group_id=None）"""
        mock_collection = MagicMock()
        chroma_manager.collection = mock_collection
        
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        memory = Memory(
            id="test_004",
            content="Private message",
            user_id="user_123",
            group_id=None,  # 私聊
            type=MemoryType.FACT,
            modality=ModalityType.TEXT
        )
        
        await chroma_manager.add_memory(memory)
        
        call_args = mock_collection.add.call_args
        metadata = call_args[1]['metadatas'][0]
        
        assert metadata['group_id'] == ""


class TestChromaManagerQueryMemories:
    """测试查询记忆功能"""
    
    @pytest.mark.asyncio
    async def test_query_memories_success(
        self, chroma_manager, monkeypatch
    ):
        """测试成功查询记忆"""
        # Mock embedding generation
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        # Mock collection query result
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['mem_001', 'mem_002']],
            'documents': [['Document 1', 'Document 2']],
            'embeddings': [[np.random.rand(1024).tolist(), np.random.rand(1024).tolist()]],
            'metadatas': [[
                {
                    'user_id': 'user_123',
                    'group_id': 'group_456',
                    'type': 'fact',
                    'modality': 'text',
                    'quality_level': 3,
                    'sensitivity_level': 0,
                    'storage_layer': 'episodic',
                    'created_time': datetime.now().isoformat(),
                    'last_access_time': datetime.now().isoformat(),
                    'access_count': 5,
                    'rif_score': 0.7,
                    'importance_score': 0.6,
                    'is_user_requested': False
                },
                {
                    'user_id': 'user_123',
                    'group_id': 'group_456',
                    'type': 'fact',
                    'modality': 'text',
                    'quality_level': 4,
                    'sensitivity_level': 1,
                    'storage_layer': 'semantic',
                    'created_time': datetime.now().isoformat(),
                    'last_access_time': datetime.now().isoformat(),
                    'access_count': 2,
                    'rif_score': 0.5,
                    'importance_score': 0.8,
                    'is_user_requested': True
                }
            ]]
        }
        chroma_manager.collection = mock_collection
        
        results = await chroma_manager.query_memories(
            "test query",
            user_id="user_123",
            group_id="group_456",
            top_k=10
        )
        
        assert len(results) == 2
        assert results[0].id == 'mem_001'
        assert results[1].id == 'mem_002'
        assert results[0].user_id == 'user_123'
        assert results[0].storage_layer == StorageLayer.EPISODIC
    
    @pytest.mark.asyncio
    async def test_query_memories_with_storage_layer_filter(
        self, chroma_manager, monkeypatch
    ):
        """测试带存储层过滤的查询"""
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['mem_001']],
            'documents': [['Document 1']],
            'embeddings': [[np.random.rand(1024).tolist()]],
            'metadatas': [[
                {
                    'user_id': 'user_123',
                    'group_id': 'group_456',
                    'type': 'fact',
                    'storage_layer': 'semantic'
                }
            ]]
        }
        chroma_manager.collection = mock_collection
        
        results = await chroma_manager.query_memories(
            "test query",
            user_id="user_123",
            group_id="group_456",
            storage_layer=StorageLayer.SEMANTIC
        )
        
        assert len(results) == 1
        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs['where']['storage_layer'] == 'semantic'
    
    @pytest.mark.asyncio
    async def test_query_memories_private_chat(
        self, chroma_manager, monkeypatch
    ):
        """测试私聊查询（group_id=None）"""
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['mem_001']],
            'documents': [['Document 1']],
            'embeddings': [[np.random.rand(1024).tolist()]],
            'metadatas': [[
                {
                    'user_id': 'user_123',
                    'group_id': '',
                    'type': 'fact',
                    'storage_layer': 'episodic'
                }
            ]]
        }
        chroma_manager.collection = mock_collection
        
        results = await chroma_manager.query_memories(
            "test query",
            user_id="user_123",
            group_id=None
        )
        
        assert len(results) == 1
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs['where']['group_id'] == ""
    
    @pytest.mark.asyncio
    async def test_query_memories_empty_result(
        self, chroma_manager, monkeypatch
    ):
        """测试空查询结果"""
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]]
        }
        chroma_manager.collection = mock_collection
        
        results = await chroma_manager.query_memories(
            "test query",
            user_id="user_123",
            group_id="group_456"
        )
        
        assert len(results) == 0


class TestChromaManagerUpdateMemory:
    """测试更新记忆功能"""
    
    @pytest.mark.asyncio
    async def test_update_memory_success(
        self, chroma_manager, monkeypatch
    ):
        """测试成功更新记忆"""
        mock_collection = MagicMock()
        chroma_manager.collection = mock_collection
        
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        memory = Memory(
            id="test_001",
            content="Updated content",
            user_id="user_123",
            group_id="group_456",
            type=MemoryType.FACT,
            modality=ModalityType.TEXT,
            storage_layer=StorageLayer.EPISODIC,
            access_count=10,
            rif_score=0.9,
            importance_score=0.8
        )
        
        success = await chroma_manager.update_memory(memory)
        
        assert success is True
        mock_collection.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_memory_failure(
        self, chroma_manager, monkeypatch
    ):
        """测试更新失败"""
        mock_collection = MagicMock()
        mock_collection.update.side_effect = Exception("Update failed")
        chroma_manager.collection = mock_collection
        
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=np.random.rand(1024).tolist())
        
        memory = Memory(
            id="test_001",
            content="Updated content",
            user_id="user_123",
            group_id="group_456"
        )
        
        success = await chroma_manager.update_memory(memory)
        
        assert success is False


class TestChromaManagerDeleteMemory:
    """测试删除记忆功能"""
    
    @pytest.mark.asyncio
    async def test_delete_memory_success(self, chroma_manager):
        """测试成功删除记忆"""
        mock_collection = MagicMock()
        chroma_manager.collection = mock_collection
        
        success = await chroma_manager.delete_memory("test_001")
        
        assert success is True
        mock_collection.delete.assert_called_once_with(ids=['test_001'])
    
    @pytest.mark.asyncio
    async def test_delete_memory_failure(self, chroma_manager):
        """测试删除失败"""
        mock_collection = MagicMock()
        mock_collection.delete.side_effect = Exception("Delete failed")
        chroma_manager.collection = mock_collection
        
        success = await chroma_manager.delete_memory("test_001")
        
        assert success is False


class TestChromaManagerDeleteSession:
    """测试删除会话功能"""
    
    @pytest.mark.asyncio
    async def test_delete_session_success(self, chroma_manager):
        """测试成功删除会话"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['mem_001', 'mem_002', 'mem_003']
        }
        chroma_manager.collection = mock_collection
        
        success = await chroma_manager.delete_session(
            user_id="user_123",
            group_id="group_456"
        )
        
        assert success is True
        mock_collection.delete.assert_called_once_with(
            ids=['mem_001', 'mem_002', 'mem_003']
        )
    
    @pytest.mark.asyncio
    async def test_delete_session_private_chat(self, chroma_manager):
        """测试删除私聊会话（group_id=None）"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['mem_001', 'mem_002']
        }
        chroma_manager.collection = mock_collection
        
        success = await chroma_manager.delete_session(
            user_id="user_123",
            group_id=None
        )
        
        assert success is True
        mock_collection.get.assert_called_once_with(
            where={'user_id': 'user_123', 'group_id': ''}
        )
    
    @pytest.mark.asyncio
    async def test_delete_session_no_memories(self, chroma_manager):
        """测试删除没有记忆的会话"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'ids': []}
        chroma_manager.collection = mock_collection
        
        success = await chroma_manager.delete_session(
            user_id="user_123",
            group_id="group_456"
        )
        
        assert success is False


class TestChromaManagerGetAllMemories:
    """测试获取所有记忆功能"""
    
    @pytest.mark.asyncio
    async def test_get_all_memories_success(self, chroma_manager):
        """测试成功获取所有记忆"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['mem_001', 'mem_002'],
            'documents': [['Document 1'], ['Document 2']],
            'embeddings': [
                [np.random.rand(1024).tolist()],
                [np.random.rand(1024).tolist()]
            ],
            'metadatas': [[
                {
                    'user_id': 'user_123',
                    'group_id': 'group_456',
                    'type': 'fact',
                    'storage_layer': 'episodic'
                },
                {
                    'user_id': 'user_123',
                    'group_id': 'group_456',
                    'type': 'fact',
                    'storage_layer': 'semantic'
                }
            ]]
        }
        chroma_manager.collection = mock_collection
        
        memories = await chroma_manager.get_all_memories(
            user_id="user_123",
            group_id="group_456"
        )
        
        assert len(memories) == 2
        assert memories[0].id == 'mem_001'
        assert memories[1].id == 'mem_002'
    
    @pytest.mark.asyncio
    async def test_get_all_memories_with_storage_filter(self, chroma_manager):
        """测试带存储层过滤获取记忆"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['mem_001'],
            'documents': [['Document 1']],
            'metadatas': [[
                {
                    'user_id': 'user_123',
                    'group_id': 'group_456',
                    'storage_layer': 'semantic'
                }
            ]]
        }
        chroma_manager.collection = mock_collection
        
        memories = await chroma_manager.get_all_memories(
            user_id="user_123",
            group_id="group_456",
            storage_layer=StorageLayer.SEMANTIC
        )
        
        assert len(memories) == 1
        mock_collection.get.assert_called_once_with(
            where={'user_id': 'user_123', 'group_id': 'group_456', 'storage_layer': 'semantic'}
        )


class TestChromaManagerCountMemories:
    """测试统计记忆数量功能"""
    
    @pytest.mark.asyncio
    async def test_count_memories_success(self, chroma_manager):
        """测试成功统计记忆"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['mem_001', 'mem_002', 'mem_003', 'mem_004', 'mem_005']
        }
        chroma_manager.collection = mock_collection
        
        count = await chroma_manager.count_memories(
            user_id="user_123",
            group_id="group_456"
        )
        
        assert count == 5
    
    @pytest.mark.asyncio
    async def test_count_memories_empty(self, chroma_manager):
        """测试统计空记忆"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'ids': []}
        chroma_manager.collection = mock_collection
        
        count = await chroma_manager.count_memories(
            user_id="user_123",
            group_id="group_456"
        )
        
        assert count == 0


class TestChromaManagerEmbeddingGeneration:
    """测试嵌入生成功能"""
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(
        self, chroma_manager, monkeypatch
    ):
        """测试成功生成嵌入"""
        chroma_manager.embedding_manager = AsyncMock()
        expected_embedding = np.random.rand(1024).tolist()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=expected_embedding)
        
        embedding = await chroma_manager._generate_embedding("Test text")
        
        assert embedding == expected_embedding
        chroma_manager.embedding_manager.embed.assert_called_once_with("Test text", 1024)
    
    @pytest.mark.asyncio
    async def test_generate_embedding_fallback_to_hash(
        self, chroma_manager, monkeypatch
    ):
        """测试嵌入生成失败时的降级到哈希"""
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_manager.embed = AsyncMock(side_effect=Exception("API failed"))
        
        embedding = await chroma_manager._generate_embedding("Test text for fallback")
        
        # 降级应该生成指定维度的向量
        assert len(embedding) == 1024
        assert all(0.0 <= x <= 1.0 for x in embedding)
    
    @pytest.mark.asyncio
    async def test_generate_embedding_different_dimensions(
        self, chroma_manager, monkeypatch
    ):
        """测试不同维度的嵌入生成"""
        chroma_manager.embedding_manager = AsyncMock()
        chroma_manager.embedding_dimension = 768
        expected_embedding = np.random.rand(768).tolist()
        chroma_manager.embedding_manager.embed = AsyncMock(return_value=expected_embedding)
        
        embedding = await chroma_manager._generate_embedding("Test text")
        
        assert len(embedding) == 768
        chroma_manager.embedding_manager.embed.assert_called_once_with("Test text", 768)


class TestChromaManagerResultToMemory:
    """测试结果转换为Memory对象"""

    def test_result_to_memory_basic(self, chroma_manager_sync):
        """测试基本转换"""
        memory_data = {
            'id': 'mem_001',
            'content': 'Test content',
            'embedding': np.random.rand(1024).tolist(),
            'metadata': {
                'user_id': 'user_123',
                'group_id': 'group_456',
                'type': 'fact',
                'modality': 'text',
                'quality_level': 3,
                'sensitivity_level': 1,
                'storage_layer': 'episodic',
                'created_time': datetime.now().isoformat(),
                'last_access_time': datetime.now().isoformat(),
                'access_count': 5,
                'rif_score': 0.7,
                'importance_score': 0.6,
                'is_user_requested': True,
                'custom_key': 'custom_value'
            }
        }

        memory = chroma_manager_sync._result_to_memory(memory_data)

        assert memory.id == 'mem_001'
        assert memory.content == 'Test content'
        assert memory.user_id == 'user_123'
        assert memory.group_id == 'group_456'
        assert memory.type == MemoryType.FACT
        assert memory.modality == ModalityType.TEXT
        assert memory.storage_layer == StorageLayer.EPISODIC
        assert memory.quality_level == 3
        assert memory.sensitivity_level == 1
        assert memory.access_count == 5
        assert memory.rif_score == 0.7
        assert memory.importance_score == 0.6
        assert memory.is_user_requested is True
        assert 'custom_key' in memory.metadata
        assert memory.metadata['custom_key'] == 'custom_value'

    def test_result_to_memory_private_chat(self, chroma_manager_sync):
        """测试私聊记忆转换（group_id为空字符串）"""
        memory_data = {
            'id': 'mem_002',
            'content': 'Private message',
            'embedding': None,
            'metadata': {
                'user_id': 'user_123',
                'group_id': '',  # 空字符串表示私聊
                'type': 'fact',
                'storage_layer': 'episodic'
            }
        }

        memory = chroma_manager_sync._result_to_memory(memory_data)

        assert memory.id == 'mem_002'
        assert memory.group_id is None  # 空字符串转换为None

    def test_result_to_memory_missing_metadata(self, chroma_manager_sync):
        """测试缺失元数据时的默认值"""
        memory_data = {
            'id': 'mem_003',
            'content': 'Minimal memory',
            'embedding': None,
            'metadata': {
                'user_id': 'user_123',
                'group_id': 'group_456'
            }
        }

        memory = chroma_manager_sync._result_to_memory(memory_data)
        
        assert memory.quality_level == 3  # 默认值
        assert memory.sensitivity_level == 0  # 默认值
        assert memory.access_count == 0  # 默认值
        assert memory.rif_score == 0.5  # 默认值
        assert memory.importance_score == 0.5  # 默认值
        assert memory.is_user_requested is False  # 默认值


class TestChromaManagerClose:
    """测试关闭功能"""
    
    @pytest.mark.asyncio
    async def test_close_success(self, chroma_manager):
        """测试成功关闭"""
        mock_client = MagicMock()
        chroma_manager.client = mock_client
        
        await chroma_manager.close()
        
        mock_client.reset.assert_called_once()
        assert chroma_manager.client is not None  # 客户端仍然存在但已重置

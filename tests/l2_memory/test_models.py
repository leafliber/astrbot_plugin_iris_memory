"""L2 数据模型测试"""

from datetime import datetime
from iris_memory.l2_memory.models import MemoryEntry, MemorySearchResult


class TestMemoryEntry:
    """MemoryEntry 测试"""

    def test_create_entry(self):
        """测试创建记忆条目"""
        entry = MemoryEntry(
            id="mem_001",
            content="用户喜欢吃苹果",
            metadata={
                "group_id": "group_123",
                "timestamp": datetime.now().isoformat(),
                "access_count": 1,
                "confidence": 0.85,
            },
        )

        assert entry.id == "mem_001"
        assert entry.content == "用户喜欢吃苹果"
        assert entry.metadata["group_id"] == "group_123"
        assert entry.embedding is None

    def test_entry_with_embedding(self):
        """测试带向量的记忆条目"""
        embedding = [0.1, 0.2, 0.3]
        entry = MemoryEntry(id="mem_002", content="测试向量", embedding=embedding)

        assert entry.embedding == embedding

    def test_to_dict(self):
        """测试转换为字典"""
        entry = MemoryEntry(
            id="mem_003", content="测试内容", metadata={"group_id": "group_123"}
        )

        data = entry.to_dict()

        assert data["id"] == "mem_003"
        assert data["content"] == "测试内容"
        assert data["metadata"]["group_id"] == "group_123"
        assert "embedding" not in data  # to_dict 不包含 embedding

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "id": "mem_004",
            "content": "测试内容",
            "metadata": {"user_id": "user_456"},
            "embedding": [0.1, 0.2],
        }

        entry = MemoryEntry.from_dict(data)

        assert entry.id == "mem_004"
        assert entry.content == "测试内容"
        assert entry.metadata == {"user_id": "user_456"}
        assert entry.embedding == [0.1, 0.2]

    def test_property_accessors(self):
        """测试属性访问器"""
        timestamp = datetime.now().isoformat()
        entry = MemoryEntry(
            id="mem_005",
            content="测试",
            metadata={
                "group_id": "group_123",
                "timestamp": timestamp,
                "access_count": 5,
                "last_access_time": timestamp,
                "confidence": 0.9,
            },
        )

        assert entry.group_id == "group_123"
        assert entry.timestamp == timestamp
        assert entry.access_count == 5
        assert entry.last_access_time == timestamp
        assert entry.confidence == 0.9

    def test_property_defaults(self):
        """测试属性默认值"""
        entry = MemoryEntry(id="mem_006", content="测试")

        assert entry.group_id is None
        assert entry.timestamp is None
        assert entry.access_count == 0
        assert entry.last_access_time is None
        assert entry.confidence == 0.5

    def test_persona_id_default(self):
        """persona_id 默认为 default"""
        entry = MemoryEntry(id="mem_007", content="测试")
        assert entry.persona_id == "default"

    def test_persona_id_roundtrip(self):
        """to_dict/from_dict 必须透传 persona_id（迁移回合关键）"""
        entry = MemoryEntry(
            id="mem_008",
            content="人格隔离测试",
            metadata={"group_id": "g1"},
            persona_id="yuki",
        )
        data = entry.to_dict()
        assert data["persona_id"] == "yuki"

        restored = MemoryEntry.from_dict(data)
        assert restored.persona_id == "yuki"

    def test_from_dict_legacy_without_persona_id(self):
        """旧版导出无 persona_id 字段时回退为 default"""
        data = {
            "id": "mem_009",
            "content": "旧格式",
            "metadata": {},
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.persona_id == "default"


class TestMemorySearchResult:
    """MemorySearchResult 测试"""

    def test_create_result(self):
        """测试创建检索结果"""
        entry = MemoryEntry(
            id="mem_001", content="测试内容", metadata={"group_id": "group_123"}
        )

        result = MemorySearchResult(entry=entry, score=0.85, distance=0.15)

        assert result.entry == entry
        assert result.score == 0.85
        assert result.distance == 0.15

    def test_to_dict(self):
        """测试转换为字典"""
        entry = MemoryEntry(
            id="mem_001", content="测试内容", metadata={"group_id": "group_123"}
        )

        result = MemorySearchResult(entry=entry, score=0.9, distance=0.1)

        data = result.to_dict()

        assert data["id"] == "mem_001"
        assert data["content"] == "测试内容"
        assert data["score"] == 0.9
        assert data["distance"] == 0.1
        assert data["metadata"]["group_id"] == "group_123"

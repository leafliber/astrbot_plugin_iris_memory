"""
Memory模型单元测试
测试Memory数据模型的功能
"""

import pytest
from datetime import datetime
from iris_memory.models.memory import Memory
from iris_memory.core.types import (
    MemoryType, ModalityType, QualityLevel,
    SensitivityLevel, StorageLayer
)


class TestMemoryInit:
    """测试Memory初始化"""
    
    def test_init_minimal(self):
        """测试最小初始化"""
        memory = Memory(id="mem_001")
        
        assert memory.id == "mem_001"
        assert memory.content == ""
        assert memory.user_id == ""
    
    def test_init_full(self):
        """测试完整初始化"""
        memory = Memory(
            id="mem_001",
            content="测试内容",
            user_id="user_123",
            group_id="group_456",
            type=MemoryType.FACT,
            modality=ModalityType.TEXT,
            storage_layer=StorageLayer.EPISODIC,
            quality_level=QualityLevel.CONFIRMED,
            sensitivity_level=SensitivityLevel.PRIVATE
        )
        
        assert memory.id == "mem_001"
        assert memory.content == "测试内容"
        assert memory.user_id == "user_123"
        assert memory.group_id == "group_456"
        assert memory.type == MemoryType.FACT


class TestMemoryMethods:
    """测试Memory方法"""
    
    def test_should_upgrade_to_episodic(self):
        """测试升级到情景记忆的条件"""
        memory = Memory(
            id="mem_001",
            storage_layer=StorageLayer.WORKING,
            quality_level=QualityLevel.CONFIRMED,
            access_count=10,
            importance_score=0.9
        )

        result = memory.should_upgrade_to_episodic()
        assert result is True
    
    def test_should_not_upgrade_low_quality(self):
        """测试低质量且无足够访问/重要性的记忆不应升级"""
        memory = Memory(
            id="mem_001",
            quality_level=QualityLevel.UNCERTAIN,
            access_count=1,
            importance_score=0.3,
            emotional_weight=0.2,
            confidence=0.3,
            rif_score=0.2,
            is_user_requested=False,
            storage_layer=StorageLayer.WORKING
        )
        
        result = memory.should_upgrade_to_episodic()
        assert result is False
    
    def test_update_access(self):
        """测试更新访问"""
        memory = Memory(
            id="mem_001",
            access_count=5,
            last_access_time=datetime(2024, 1, 1, 10, 0, 0)
        )
        
        old_count = memory.access_count
        memory.update_access()
        
        assert memory.access_count == old_count + 1
        assert memory.last_access_time > datetime(2024, 1, 1, 10, 0, 0)
    
    def test_to_dict(self):
        """测试序列化"""
        memory = Memory(
            id="mem_001",
            content="测试",
            user_id="user_123"
        )
        
        data = memory.to_dict()
        
        assert data['id'] == "mem_001"
        assert data['content'] == "测试"
        assert 'created_time' in data
    
    def test_from_dict(self):
        """测试反序列化"""
        data = {
            'id': 'mem_002',
            'content': '测试内容',
            'user_id': 'user_456',
            'type': 'fact',
            'storage_layer': 'episodic'
        }
        
        memory = Memory.from_dict(data)
        
        assert memory.id == 'mem_002'
        assert memory.content == '测试内容'
        assert memory.user_id == 'user_456'
        assert memory.type == MemoryType.FACT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

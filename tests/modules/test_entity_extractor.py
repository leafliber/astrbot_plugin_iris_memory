"""
实体提取器单元测试
测试EntityExtractor的功能
"""

import pytest
from unittest.mock import Mock
from iris_memory.analysis.entity.entity_extractor import EntityExtractor, EntityType


class TestEntityExtractorInit:
    """测试初始化"""
    
    def test_init(self):
        """测试基本初始化"""
        extractor = EntityExtractor()
        
        assert extractor is not None
    
    def test_init_with_config(self):
        """测试带配置初始化"""
        from datetime import datetime
        reference_date = datetime(2026, 1, 31)
        extractor = EntityExtractor(reference_date=reference_date)

        assert extractor.reference_date == reference_date


class TestEntityExtractorExtraction:
    """测试实体提取"""

    def test_extract_person(self):
        """测试提取人名实体"""
        extractor = EntityExtractor()
        text = "张三和李四昨天见面了"

        entities = extractor.extract_entities(text)

        # 应该提取出人名
        assert any(e.entity_type == EntityType.PERSON for e in entities)

    def test_extract_location(self):
        """测试提取地点实体"""
        extractor = EntityExtractor()
        text = "我去了北京和上海"

        entities = extractor.extract_entities(text)

        # 应该提取出地点
        assert any(e.entity_type == EntityType.LOCATION for e in entities)

    def test_extract_empty_text(self):
        """测试空文本"""
        extractor = EntityExtractor()

        entities = extractor.extract_entities("")

        assert len(entities) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

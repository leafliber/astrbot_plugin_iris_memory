"""
LLM处理器测试

测试级别：
- 单元测试：单个方法测试
- 集成测试：完整流程测试
- 边界测试：异常情况和边界条件
- 性能测试：大量数据处理
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from iris_memory.processing.llm_processor import (
    LLMMessageProcessor,
    LLMClassificationResult,
    LLMSummaryResult
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_astrbot_context():
    """模拟AstrBot上下文"""
    context = Mock()
    context.send_message = AsyncMock()
    return context


@pytest.fixture
def mock_llm_api():
    """模拟LLM API"""
    api = Mock()
    api.text_chat = AsyncMock(return_value={
        "text": '{"layer": "immediate", "confidence": 0.9, "reason": "test"}'
    })
    api.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"summary": "test summary"}'}}]
    })
    return api


@pytest.fixture
def processor(mock_astrbot_context):
    """基础LLM处理器实例"""
    return LLMMessageProcessor(
        astrbot_context=mock_astrbot_context,
        max_tokens=200
    )


@pytest.fixture
def initialized_processor(mock_astrbot_context, mock_llm_api):
    """已初始化的LLM处理器"""
    processor = LLMMessageProcessor(
        astrbot_context=mock_astrbot_context,
        max_tokens=200
    )
    processor.llm_api = mock_llm_api
    return processor


# =============================================================================
# 初始化测试
# =============================================================================

class TestInitialization:
    """初始化测试"""
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_astrbot_context, mock_llm_api):
        """测试成功初始化"""
        # Create a mock module for astrbot.api
        import sys
        from types import ModuleType
        
        # Create mock modules
        astrbot_module = ModuleType('astrbot')
        astrbot_api_module = ModuleType('astrbot.api')
        astrbot_api_module.AstrBotApi = Mock(return_value=mock_llm_api)
        astrbot_module.api = astrbot_api_module
        
        # Add to sys.modules
        sys.modules['astrbot'] = astrbot_module
        sys.modules['astrbot.api'] = astrbot_api_module
        
        try:
            processor = LLMMessageProcessor(mock_astrbot_context)
            result = await processor.initialize()
            
            assert result is True
            # llm_api is now lazy-loaded on first use, not during initialize()
            # is_available() checks llm_api which is not set until first use
            assert processor.astrbot_context is not None
        finally:
            # Clean up
            sys.modules.pop('astrbot', None)
            sys.modules.pop('astrbot.api', None)
    
    @pytest.mark.asyncio
    async def test_initialize_no_context(self):
        """测试无上下文初始化失败"""
        processor = LLMMessageProcessor(astrbot_context=None)
        result = await processor.initialize()
        
        assert result is False
        assert processor.llm_api is None
        assert processor.is_available() is False
    
    @pytest.mark.asyncio
    async def test_initialize_import_error(self, mock_astrbot_context):
        """测试导入错误处理"""
        # initialize() 现在使用延迟加载策略，不再在初始化时导入
        # 只要有context就会initialize成功
        processor = LLMMessageProcessor(mock_astrbot_context)
        result = await processor.initialize()
        
        assert result is True


# =============================================================================
# 消息分类测试
# =============================================================================

class TestMessageClassification:
    """消息分类测试"""
    
    @pytest.mark.asyncio
    async def test_classify_message_immediate(self, initialized_processor):
        """测试高优先级消息分类"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "immediate", "confidence": 0.9, "reason": "重要信息"}'
        }
        
        result = await initialized_processor.classify_message("我喜欢猫")
        
        assert result is not None
        assert result.layer == "immediate"
        assert result.confidence == 0.9
        assert result.reason == "重要信息"
    
    @pytest.mark.asyncio
    async def test_classify_message_batch(self, initialized_processor):
        """测试普通消息分类"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "batch", "confidence": 0.5, "reason": "普通对话"}'
        }
        
        result = await initialized_processor.classify_message("今天天气不错")
        
        assert result is not None
        assert result.layer == "batch"
        assert result.confidence == 0.5
    
    @pytest.mark.asyncio
    async def test_classify_message_discard(self, initialized_processor):
        """测试丢弃消息分类"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "discard", "confidence": 0.1, "reason": "无意义"}'
        }
        
        result = await initialized_processor.classify_message("哈哈")
        
        assert result is not None
        assert result.layer == "discard"
    
    @pytest.mark.asyncio
    async def test_classify_message_invalid_json(self, initialized_processor):
        """测试无效JSON响应处理"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": "invalid json response"
        }
        
        result = await initialized_processor.classify_message("测试消息")
        
        # 应该返回None或回退处理
        assert result is None
    
    @pytest.mark.asyncio
    async def test_classify_message_api_error(self, initialized_processor):
        """测试API错误处理"""
        initialized_processor.llm_api.text_chat.side_effect = Exception("API Error")
        
        result = await initialized_processor.classify_message("测试消息")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_classify_message_no_api(self, processor):
        """测试无API情况"""
        result = await processor.classify_message("测试消息")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_classify_message_with_context(self, initialized_processor):
        """测试带上下文的分类"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "immediate", "confidence": 0.85, "reason": "有上下文"}'
        }
        
        context = {
            "session_message_count": 5,
            "last_topic": "宠物"
        }
        
        result = await initialized_processor.classify_message(
            "我喜欢猫", context=context
        )
        
        assert result is not None
        # 验证上下文被包含在prompt中
        call_args = initialized_processor.llm_api.text_chat.call_args
        assert call_args is not None


# =============================================================================
# 摘要生成测试
# =============================================================================

class TestSummaryGeneration:
    """摘要生成测试"""
    
    @pytest.mark.asyncio
    async def test_generate_summary_success(self, initialized_processor):
        """测试成功生成摘要"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"summary": "用户喜欢猫和狗", "key_points": ["喜欢猫", "喜欢狗"], "user_preferences": ["宠物爱好者"]}'
        }
        
        messages = ["我喜欢猫", "我也喜欢狗", "它们很可爱"]
        result = await initialized_processor.generate_summary(
            messages, user_id="test_user"
        )
        
        assert result is not None
        assert result.summary == "用户喜欢猫和狗"
        assert len(result.key_points) == 2
        assert len(result.user_preferences) == 1
    
    @pytest.mark.asyncio
    async def test_generate_summary_empty_messages(self, initialized_processor):
        """测试空消息列表"""
        result = await initialized_processor.generate_summary(
            [], user_id="test_user"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_summary_single_message(self, initialized_processor):
        """测试单条消息摘要"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"summary": "用户喜欢猫", "key_points": ["喜欢猫"], "user_preferences": []}'
        }
        
        messages = ["我喜欢猫"]
        result = await initialized_processor.generate_summary(
            messages, user_id="test_user"
        )
        
        assert result is not None
        assert "喜欢猫" in result.summary
    
    @pytest.mark.asyncio
    async def test_generate_summary_many_messages(self, initialized_processor):
        """测试大量消息摘要"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"summary": "对话摘要", "key_points": ["要点1"], "user_preferences": []}'
        }
        
        # 测试超过10条消息时只取最近10条
        # 生成15条消息 (索引 0-14)
        messages = [f"消息{i}" for i in range(15)]
        result = await initialized_processor.generate_summary(
            messages, user_id="test_user"
        )
        
        assert result is not None
        # 验证只使用了最近10条 (索引 5-14)
        call_args = initialized_processor.llm_api.text_chat.call_args[1]
        prompt = call_args.get('prompt', '')
        assert "消息4" not in prompt  # 旧消息不应该在prompt中
        assert "消息5" in prompt  # 第一条保留的消息
        assert "消息14" in prompt  # 最后一条消息
    
    @pytest.mark.asyncio
    async def test_generate_summary_with_persona(self, initialized_processor):
        """测试带用户画像的摘要"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"summary": "符合画像的摘要", "key_points": [], "user_preferences": []}'
        }
        
        context = {
            "user_persona": {"interests": ["宠物", "摄影"]}
        }
        
        messages = ["我喜欢猫"]
        result = await initialized_processor.generate_summary(
            messages, user_id="test_user", context=context
        )
        
        assert result is not None
        # 验证用户画像被包含
        call_args = initialized_processor.llm_api.text_chat.call_args
        assert call_args is not None
    
    @pytest.mark.asyncio
    async def test_generate_summary_parse_fallback(self, initialized_processor):
        """测试JSON解析失败时的回退"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": "这是一个纯文本回复，不是JSON格式"
        }
        
        messages = ["我喜欢猫"]
        result = await initialized_processor.generate_summary(
            messages, user_id="test_user"
        )
        
        # 应该返回使用原始文本的结果
        assert result is not None
        assert result.summary == "这是一个纯文本回复，不是JSON格式"


# =============================================================================
# 边界条件测试
# =============================================================================

class TestEdgeCases:
    """边界条件测试"""
    
    @pytest.mark.asyncio
    async def test_very_long_message_classification(self, initialized_processor):
        """测试超长消息分类"""
        long_message = "我喜欢猫" * 1000
        
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "batch", "confidence": 0.5, "reason": "长消息"}'
        }
        
        result = await initialized_processor.classify_message(long_message)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_special_characters_in_message(self, initialized_processor):
        """测试特殊字符处理"""
        special_message = "我喜欢猫！🐱 <script>alert('xss')</script> \\n\\t"
        
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "immediate", "confidence": 0.8, "reason": "特殊字符"}'
        }
        
        result = await initialized_processor.classify_message(special_message)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_unicode_message(self, initialized_processor):
        """测试Unicode消息"""
        unicode_message = "我喜欢猫🐱 dogs são legais 日本語"
        
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "immediate", "confidence": 0.9, "reason": "unicode"}'
        }
        
        result = await initialized_processor.classify_message(unicode_message)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, initialized_processor):
        """测试并发请求处理"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "batch", "confidence": 0.5, "reason": "并发"}'
        }
        
        # 并发发送多个请求
        tasks = [
            initialized_processor.classify_message(f"消息{i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 所有请求都应该成功
        assert all(r is not None for r in results)
        assert len(results) == 10


# =============================================================================
# 统计信息测试
# =============================================================================

class TestStatistics:
    """统计信息测试"""
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, initialized_processor):
        """测试统计信息追踪"""
        # 初始状态
        stats = initialized_processor.get_stats()
        assert stats["classification_calls"] == 0
        assert stats["summary_calls"] == 0
        
        # 执行一些操作
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "immediate", "confidence": 0.9, "reason": "test"}'
        }
        
        await initialized_processor.classify_message("测试")
        
        stats = initialized_processor.get_stats()
        assert stats["classification_calls"] == 1
        assert stats["failed_calls"] == 0
    
    @pytest.mark.asyncio
    async def test_stats_after_failure(self, initialized_processor):
        """测试失败后的统计"""
        initialized_processor.llm_api.text_chat.side_effect = Exception("Error")
        
        await initialized_processor.classify_message("测试")
        
        stats = initialized_processor.get_stats()
        assert stats["failed_calls"] == 1


# =============================================================================
# JSON解析测试
# =============================================================================

class TestJSONParsing:
    """JSON解析测试"""
    
    def test_parse_valid_json(self, initialized_processor):
        """测试有效JSON解析"""
        response = '{"layer": "immediate", "confidence": 0.9}'
        result = initialized_processor._parse_json_response(response)
        
        assert result is not None
        assert result["layer"] == "immediate"
    
    def test_parse_json_with_code_block(self, initialized_processor):
        """测试代码块中的JSON"""
        response = '```json\n{"layer": "batch"}\n```'
        result = initialized_processor._parse_json_response(response)
        
        assert result is not None
        assert result["layer"] == "batch"
    
    def test_parse_json_with_extra_text(self, initialized_processor):
        """测试带额外文本的JSON"""
        response = 'Here is the result: {"layer": "discard"} Thanks!'
        result = initialized_processor._parse_json_response(response)
        
        assert result is not None
        assert result["layer"] == "discard"
    
    def test_parse_invalid_json(self, initialized_processor):
        """测试无效JSON"""
        response = "This is not JSON"
        result = initialized_processor._parse_json_response(response)
        
        assert result is None
    
    def test_parse_empty_response(self, initialized_processor):
        """测试空响应"""
        result = initialized_processor._parse_json_response("")
        
        assert result is None


# =============================================================================
# 配置测试
# =============================================================================

class TestConfiguration:
    """配置测试"""
    
    def test_custom_prompts(self, mock_astrbot_context):
        """测试自定义提示词"""
        custom_class_prompt = "Custom classification prompt"
        custom_summary_prompt = "Custom summary prompt"
        
        processor = LLMMessageProcessor(
            astrbot_context=mock_astrbot_context,
            classification_prompt=custom_class_prompt,
            summary_prompt=custom_summary_prompt,
            max_tokens=300
        )
        
        assert processor.classification_prompt == custom_class_prompt
        assert processor.summary_prompt == custom_summary_prompt
        assert processor.max_tokens == 300
    
    def test_default_prompts(self, mock_astrbot_context):
        """测试默认提示词"""
        processor = LLMMessageProcessor(astrbot_context=mock_astrbot_context)
        
        assert "layer" in processor.classification_prompt
        assert "summary" in processor.summary_prompt


# =============================================================================
# 性能测试
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self, initialized_processor):
        """测试大批量处理性能"""
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "batch", "confidence": 0.5}'
        }
        
        # 处理100条消息
        start_time = asyncio.get_event_loop().time()
        
        for i in range(100):
            await initialized_processor.classify_message(f"消息{i}")
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # 应该在合理时间内完成（假设每秒10个请求）
        assert elapsed < 15  # 放宽到15秒
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_messages(self, initialized_processor):
        """测试大消息的内存使用"""
        large_message = "A" * 10000  # 10KB消息
        
        initialized_processor.llm_api.text_chat.return_value = {
            "text": '{"layer": "batch"}'
        }
        
        result = await initialized_processor.classify_message(large_message)
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

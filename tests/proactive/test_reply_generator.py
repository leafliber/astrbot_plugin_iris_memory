"""
回复生成器测试

测试回复生成的核心功能：
- 语调选择
- 提示词构建
- LLM调用
- 记忆引用
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from iris_memory.proactive.reply_generator import (
    ProactiveReplyGenerator,
    GeneratedReply
)
from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine
from iris_memory.models.emotion_state import EmotionalState


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_provider():
    """模拟LLM provider"""
    provider = Mock()
    provider.id = "test-provider-id"
    provider.text_chat = AsyncMock(return_value={
        "text": "我理解你的感受，有什么我可以帮你的吗？"
    })
    return provider


@pytest.fixture
def mock_astrbot_context(mock_llm_provider):
    """模拟AstrBot上下文（带LLM provider）"""
    context = Mock()
    context.get_using_provider = Mock(return_value=mock_llm_provider)
    # 模拟 llm_generate 方法，返回带 completion_text 的响应对象
    llm_response = Mock()
    llm_response.completion_text = "我理解你的感受，有什么我可以帮你的吗？"
    context.llm_generate = AsyncMock(return_value=llm_response)
    return context


@pytest.fixture
def mock_retrieval_engine():
    """模拟检索引擎"""
    engine = Mock(spec=MemoryRetrievalEngine)
    engine.retrieve = AsyncMock(return_value=[
        Mock(content="用户喜欢猫"),
        Mock(content="用户讨厌狗")
    ])
    return engine


@pytest.fixture
def generator(mock_retrieval_engine):
    """基础生成器（无LLM provider，get_using_provider返回None）"""
    ctx = Mock()
    ctx.get_using_provider = Mock(return_value=None)
    return ProactiveReplyGenerator(
        astrbot_context=ctx,
        retrieval_engine=mock_retrieval_engine,
        config={
            "max_reply_tokens": 150,
            "reply_temperature": 0.7
        }
    )


@pytest.fixture
def initialized_generator(mock_astrbot_context, mock_retrieval_engine):
    """已初始化的生成器（有LLM provider）"""
    return ProactiveReplyGenerator(
        astrbot_context=mock_astrbot_context,
        retrieval_engine=mock_retrieval_engine
    )


# =============================================================================
# 初始化测试
# =============================================================================

class TestInitialization:
    """初始化测试"""
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_astrbot_context, mock_retrieval_engine):
        """测试成功初始化"""
        # Create mock modules for astrbot
        import sys
        from types import ModuleType
        
        astrbot_module = ModuleType('astrbot')
        astrbot_api_module = ModuleType('astrbot.api')
        mock_llm_api = Mock()
        astrbot_api_module.AstrBotApi = Mock(return_value=mock_llm_api)
        astrbot_module.api = astrbot_api_module
        
        sys.modules['astrbot'] = astrbot_module
        sys.modules['astrbot.api'] = astrbot_api_module
        
        try:
            generator = ProactiveReplyGenerator(
                astrbot_context=mock_astrbot_context,
                retrieval_engine=mock_retrieval_engine
            )
            result = await generator.initialize()
            
            assert result is True
        finally:
            sys.modules.pop('astrbot', None)
            sys.modules.pop('astrbot.api', None)
    
    @pytest.mark.asyncio
    async def test_initialize_no_context(self, mock_retrieval_engine):
        """测试无上下文初始化"""
        generator = ProactiveReplyGenerator(
            astrbot_context=None,
            retrieval_engine=mock_retrieval_engine
        )
        result = await generator.initialize()
        
        assert result is False
    
    def test_configuration(self, mock_astrbot_context, mock_retrieval_engine):
        """测试配置"""
        generator = ProactiveReplyGenerator(
            astrbot_context=mock_astrbot_context,
            retrieval_engine=mock_retrieval_engine,
            config={
                "max_reply_tokens": 200,
                "reply_temperature": 0.5
            }
        )
        
        assert generator.max_tokens == 200
        assert generator.temperature == 0.5


# =============================================================================
# 回复生成测试
# =============================================================================

class TestReplyGeneration:
    """回复生成测试"""
    
    @pytest.mark.asyncio
    async def test_generate_reply_success(self, initialized_generator):
        """测试成功生成回复"""
        messages = ["我很难过"]
        
        result = await initialized_generator.generate_reply(
            messages=messages,
            user_id="test_user"
        )
        
        assert result is not None
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_generate_reply_empty_messages(self, initialized_generator):
        """测试空消息列表"""
        result = await initialized_generator.generate_reply(
            messages=[],
            user_id="test_user"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_reply_no_api(self, generator):
        """测试无API情况"""
        result = await generator.generate_reply(
            messages=["测试"],
            user_id="test_user"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_reply_with_memories(self, initialized_generator, mock_retrieval_engine):
        """测试带记忆的回复"""
        messages = ["我喜欢什么动物？"]
        
        result = await initialized_generator.generate_reply(
            messages=messages,
            user_id="test_user"
        )
        
        # 验证检索引擎被调用
        mock_retrieval_engine.retrieve.assert_called_once()
        
        assert result is not None
        assert len(result.referenced_memories) > 0


# =============================================================================
# 语调选择测试
# =============================================================================

class TestToneSelection:
    """语调选择测试"""
    
    def test_happy_emotion_tone(self, initialized_generator):
        """测试开心情绪语调"""
        emotion = {"primary": "happy", "intensity": 0.6}
        
        tone = initialized_generator._determine_tone(emotion, None)
        
        assert tone == "cheerful"
    
    def test_sad_emotion_tone(self, initialized_generator):
        """测试悲伤情绪语调"""
        emotion = {"primary": "sad", "intensity": 0.6}
        
        tone = initialized_generator._determine_tone(emotion, None)
        
        assert tone == "supportive"
    
    def test_angry_emotion_tone(self, initialized_generator):
        """测试愤怒情绪语调"""
        emotion = {"primary": "angry", "intensity": 0.6}
        
        tone = initialized_generator._determine_tone(emotion, None)
        
        assert tone == "empathetic"
    
    def test_high_intensity_happy(self, initialized_generator):
        """测试高强度开心"""
        emotion = {"primary": "happy", "intensity": 0.9}
        
        tone = initialized_generator._determine_tone(emotion, None)
        
        assert tone == "encouraging"
    
    def test_high_intensity_sad(self, initialized_generator):
        """测试高强度悲伤"""
        emotion = {"primary": "sad", "intensity": 0.9}
        
        tone = initialized_generator._determine_tone(emotion, None)
        
        assert tone == "empathetic"
    
    def test_neutral_emotion(self, initialized_generator):
        """测试中性情绪"""
        emotion = {"primary": "neutral", "intensity": 0.5}
        
        tone = initialized_generator._determine_tone(emotion, None)
        
        assert tone == "neutral"


# =============================================================================
# 提示词构建测试
# =============================================================================

class TestPromptBuilding:
    """提示词构建测试"""
    
    def test_prompt_includes_messages(self, initialized_generator):
        """测试提示词包含消息"""
        messages = ["消息1", "消息2"]
        
        prompt = initialized_generator._build_reply_prompt(
            messages=messages,
            memories=[],
            tone="neutral"
        )
        
        assert "消息1" in prompt
        assert "消息2" in prompt
    
    def test_prompt_includes_memories(self, initialized_generator):
        """测试提示词包含记忆"""
        messages = ["测试"]
        memories = ["用户喜欢猫", "用户讨厌狗"]
        
        prompt = initialized_generator._build_reply_prompt(
            messages=messages,
            memories=memories,
            tone="neutral"
        )
        
        assert "用户喜欢猫" in prompt
        assert "用户讨厌狗" in prompt
    
    def test_prompt_includes_tone(self, initialized_generator):
        """测试提示词包含语调"""
        messages = ["测试"]
        
        prompt = initialized_generator._build_reply_prompt(
            messages=messages,
            memories=[],
            tone="supportive"
        )
        
        assert "supportive" in prompt or "温暖支持" in prompt
    
    def test_prompt_limits_messages(self, initialized_generator):
        """测试提示词限制消息数量"""
        # 生成10条消息 (索引 0-9)
        messages = [f"消息{i}" for i in range(10)]
        
        prompt = initialized_generator._build_reply_prompt(
            messages=messages,
            memories=[],
            tone="neutral"
        )
        
        # 只应该包含最近5条 (索引 5-9)
        assert "消息4" not in prompt  # Not included
        assert "消息5" in prompt      # First included
        assert "消息9" in prompt      # Last included


# =============================================================================
# LLM调用测试
# =============================================================================

class TestLLMCalling:
    """LLM调用测试"""
    
    @pytest.mark.asyncio
    async def test_text_chat_method(self, initialized_generator, mock_llm_provider):
        """测试text_chat方法"""
        response = await initialized_generator._call_llm(mock_llm_provider, "测试提示词")
        
        assert response is not None
        mock_llm_provider.text_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_text_chat_dict_response(self, initialized_generator, mock_llm_provider):
        """测试text_chat返回dict响应"""
        mock_llm_provider.text_chat.return_value = {"text": "回复内容"}
        
        response = await initialized_generator._call_llm(mock_llm_provider, "测试提示词")
        
        assert response == "回复内容"
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, initialized_generator, mock_llm_provider):
        """测试API错误处理"""
        mock_llm_provider.text_chat.side_effect = Exception("API Error")
        
        response = await initialized_generator._call_llm(mock_llm_provider, "测试")
        
        assert response is None
    
    @pytest.mark.asyncio
    async def test_no_provider(self, generator):
        """测试无provider"""
        response = await generator._call_llm(None, "测试")
        
        assert response is None


# =============================================================================
# 回复提取测试
# =============================================================================

class TestReplyExtraction:
    """回复提取测试"""
    
    def test_extract_normal_reply(self, initialized_generator):
        """测试正常回复提取"""
        response = "  这是一个回复  "
        
        reply = initialized_generator._extract_reply(response)
        
        assert reply == "这是一个回复"
    
    def test_extract_with_quotes(self, initialized_generator):
        """测试带引号的回复"""
        response = '"带引号的回复"'
        
        reply = initialized_generator._extract_reply(response)
        
        assert reply == "带引号的回复"
    
    def test_extract_long_reply(self, initialized_generator):
        """测试长回复截断"""
        response = "A" * 300  # 超长回复
        
        reply = initialized_generator._extract_reply(response)
        
        assert len(reply) <= 200
        assert reply.endswith("...")
    
    def test_extract_empty_reply(self, initialized_generator):
        """测试空回复"""
        reply = initialized_generator._extract_reply("")
        
        assert reply == "我在听，请继续说。"


# =============================================================================
# 记忆引用测试
# =============================================================================

class TestMemoryReferencing:
    """记忆引用测试"""
    
    @pytest.mark.asyncio
    async def test_memories_included_in_reply(self, initialized_generator, mock_retrieval_engine):
        """测试记忆包含在回复中"""
        mock_retrieval_engine.retrieve.return_value = [
            Mock(content="用户喜欢蓝色"),
            Mock(content="用户喜欢夏天")
        ]
        
        result = await initialized_generator.generate_reply(
            messages=["我喜欢什么颜色？"],
            user_id="test_user"
        )
        
        assert "用户喜欢蓝色" in result.referenced_memories
    
    @pytest.mark.asyncio
    async def test_no_memories_found(self, initialized_generator, mock_retrieval_engine):
        """测试无记忆情况"""
        mock_retrieval_engine.retrieve.return_value = []
        
        result = await initialized_generator.generate_reply(
            messages=["测试"],
            user_id="test_user"
        )
        
        assert len(result.referenced_memories) == 0
    
    @pytest.mark.asyncio
    async def test_memory_limit(self, initialized_generator, mock_retrieval_engine):
        """测试记忆数量限制"""
        # Return only 3 memories to match top_k=3 request in generate_reply
        mock_retrieval_engine.retrieve.return_value = [
            Mock(content=f"记忆{i}") for i in range(3)
        ]
        
        result = await initialized_generator.generate_reply(
            messages=["测试"],
            user_id="test_user"
        )
        
        # Should reference exactly 3 memories (top_k=3 in retrieve call)
        assert result is not None
        if hasattr(result, 'referenced_memories'):
            assert len(result.referenced_memories) == 3


# =============================================================================
# 上下文传递测试
# =============================================================================

class TestContextPassing:
    """上下文传递测试"""
    
    @pytest.mark.asyncio
    async def test_emotional_state_passed(self, initialized_generator, mock_retrieval_engine):
        """测试情感状态传递"""
        emotional_state = Mock(spec=EmotionalState)
        
        await initialized_generator.generate_reply(
            messages=["测试"],
            user_id="test_user",
            emotional_state=emotional_state
        )
        
        # 验证情感状态传递给检索引擎
        call_kwargs = mock_retrieval_engine.retrieve.call_args[1]
        assert call_kwargs.get("emotional_state") == emotional_state
    
    @pytest.mark.asyncio
    async def test_reply_context_used(self, initialized_generator):
        """测试回复上下文使用"""
        reply_context = {"reason": "用户提问", "signals": {"question": 0.9}}
        
        prompt = initialized_generator._build_reply_prompt(
            messages=["测试"],
            memories=[],
            tone="neutral",
            reply_context=reply_context
        )
        
        assert "用户提问" in prompt


# =============================================================================
# 边界测试
# =============================================================================

class TestEdgeCases:
    """边界测试"""
    
    @pytest.mark.asyncio
    async def test_very_long_message(self, initialized_generator):
        """测试超长消息"""
        long_message = "A" * 10000
        
        result = await initialized_generator.generate_reply(
            messages=[long_message],
            user_id="test_user"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_special_characters(self, initialized_generator):
        """测试特殊字符"""
        message = "你好🐱 <script> \\n\\t"
        
        result = await initialized_generator.generate_reply(
            messages=[message],
            user_id="test_user"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_unicode_message(self, initialized_generator):
        """测试Unicode消息"""
        message = "你好🐱 日本語 العربية"
        
        result = await initialized_generator.generate_reply(
            messages=[message],
            user_id="test_user"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_many_messages(self, initialized_generator):
        """测试大量消息"""
        messages = [f"消息{i}" for i in range(20)]
        
        result = await initialized_generator.generate_reply(
            messages=messages,
            user_id="test_user"
        )
        
        assert result is not None


# =============================================================================
# 可用性测试
# =============================================================================

class TestAvailability:
    """可用性测试"""
    
    def test_available_with_context(self, initialized_generator):
        """测试有上下文时可用"""
        assert initialized_generator.is_available() is True
    
    def test_not_available_without_context(self):
        """测试无上下文时不可用"""
        gen = ProactiveReplyGenerator(astrbot_context=None)
        assert gen.is_available() is False


# =============================================================================
# 性能测试
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_multiple_replies_performance(self, initialized_generator):
        """测试多次回复性能"""
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        for i in range(10):
            await initialized_generator.generate_reply(
                messages=[f"消息{i}"],
                user_id="test_user"
            )
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # 10次回复应该在5秒内完成
        assert elapsed < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

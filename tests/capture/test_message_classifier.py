"""
消息分类器测试

测试分类器的三种模式：
- local: 仅本地规则
- llm: 仅LLM分类
- hybrid: 混合模式（默认）
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from iris_memory.capture.message_classifier import (
    MessageClassifier,
    ProcessingLayer,
    ClassificationResult
)
from iris_memory.capture.trigger_detector import TriggerDetector
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.processing.llm_processor import LLMMessageProcessor, LLMClassificationResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def trigger_detector():
    """触发器检测器"""
    return TriggerDetector()


@pytest.fixture
def emotion_analyzer():
    """情感分析器"""
    analyzer = Mock(spec=EmotionAnalyzer)
    analyzer.analyze_emotion = AsyncMock(return_value={
        "primary": "neutral",
        "intensity": 0.5,
        "confidence": 0.8
    })
    return analyzer


@pytest.fixture
def mock_llm_processor():
    """模拟LLM处理器"""
    processor = Mock(spec=LLMMessageProcessor)
    processor.is_available = Mock(return_value=True)
    processor.classify_message = AsyncMock(return_value=LLMClassificationResult(
        layer="batch",
        confidence=0.6,
        reason="test",
        metadata={}
    ))
    return processor


@pytest.fixture
def local_classifier(trigger_detector, emotion_analyzer):
    """本地模式分类器"""
    return MessageClassifier(
        trigger_detector=trigger_detector,
        emotion_analyzer=emotion_analyzer,
        llm_processor=None,
        config={"llm_processing_mode": "local"}
    )


@pytest.fixture
def llm_classifier(trigger_detector, emotion_analyzer, mock_llm_processor):
    """LLM模式分类器"""
    return MessageClassifier(
        trigger_detector=trigger_detector,
        emotion_analyzer=emotion_analyzer,
        llm_processor=mock_llm_processor,
        config={"llm_processing_mode": "llm"}
    )


@pytest.fixture
def hybrid_classifier(trigger_detector, emotion_analyzer, mock_llm_processor):
    """混合模式分类器"""
    return MessageClassifier(
        trigger_detector=trigger_detector,
        emotion_analyzer=emotion_analyzer,
        llm_processor=mock_llm_processor,
        config={
            "llm_processing_mode": "hybrid",
            "immediate_trigger_confidence": 0.8,
            "immediate_emotion_intensity": 0.7
        }
    )


# =============================================================================
# 本地模式测试
# =============================================================================

class TestLocalMode:
    """本地模式测试"""
    
    @pytest.mark.asyncio
    async def test_negative_sample_discard(self, local_classifier):
        """测试负样本丢弃"""
        result = await local_classifier.classify("哈哈")
        
        assert result.layer == ProcessingLayer.DISCARD
        assert result.confidence == 1.0
        assert result.source == "local"
    
    @pytest.mark.asyncio
    async def test_high_confidence_trigger_immediate(self, local_classifier, emotion_analyzer):
        """测试高置信度触发器立即处理"""
        # 显式触发器应该有高置信度
        result = await local_classifier.classify("请记住我喜欢猫")
        
        assert result.layer == ProcessingLayer.IMMEDIATE
        assert result.confidence >= 0.8
        assert "trigger" in result.reason
    
    @pytest.mark.asyncio
    async def test_high_emotion_immediate(self, local_classifier, emotion_analyzer):
        """测试高情感强度立即处理"""
        emotion_analyzer.analyze_emotion.return_value = {
            "primary": "happy",
            "intensity": 0.9,  # 高强度
            "confidence": 0.8
        }
        
        result = await local_classifier.classify("我太开心了！")
        
        assert result.layer == ProcessingLayer.IMMEDIATE
        assert "emotion" in result.reason
    
    @pytest.mark.asyncio
    async def test_normal_message_batch(self, local_classifier):
        """测试普通消息批量处理"""
        result = await local_classifier.classify("今天天气不错")
        
        assert result.layer == ProcessingLayer.BATCH
        assert result.confidence == 0.5
        assert result.source == "local"
    
    @pytest.mark.asyncio
    async def test_preference_trigger(self, local_classifier):
        """测试偏好触发器"""
        result = await local_classifier.classify("我喜欢喝咖啡")
        
        # 偏好触发器应该有高置信度
        assert result.layer == ProcessingLayer.IMMEDIATE
    
    @pytest.mark.asyncio
    async def test_emotion_trigger(self, local_classifier, emotion_analyzer):
        """测试情感触发器"""
        emotion_analyzer.analyze_emotion.return_value = {
            "primary": "sad",
            "intensity": 0.75,  # Above threshold (0.7) to trigger immediate processing
            "confidence": 0.8
        }
        
        result = await local_classifier.classify("我觉得很难过")
        
        assert result.layer == ProcessingLayer.IMMEDIATE
    
    @pytest.mark.asyncio
    async def test_short_message_discard(self, local_classifier):
        """测试短消息丢弃"""
        result = await local_classifier.classify("好")
        
        assert result.layer == ProcessingLayer.DISCARD
    
    @pytest.mark.asyncio
    async def test_greeting_discard(self, local_classifier):
        """测试问候语丢弃"""
        result = await local_classifier.classify("你好")
        
        assert result.layer == ProcessingLayer.DISCARD


# =============================================================================
# LLM模式测试
# =============================================================================

class TestLLMMode:
    """LLM模式测试"""
    
    @pytest.mark.asyncio
    async def test_llm_classify_immediate(self, llm_classifier, mock_llm_processor):
        """测试LLM分类立即处理"""
        mock_llm_processor.classify_message.return_value = LLMClassificationResult(
            layer="immediate",
            confidence=0.9,
            reason="LLM决定",
            metadata={}
        )
        
        result = await llm_classifier.classify("重要消息")
        
        assert result.layer == ProcessingLayer.IMMEDIATE
        assert result.source == "llm"
        assert result.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_llm_classify_batch(self, llm_classifier, mock_llm_processor):
        """测试LLM分类批量处理"""
        mock_llm_processor.classify_message.return_value = LLMClassificationResult(
            layer="batch",
            confidence=0.6,
            reason="LLM批量",
            metadata={}
        )
        
        result = await llm_classifier.classify("普通消息")
        
        assert result.layer == ProcessingLayer.BATCH
        assert result.source == "llm"
    
    @pytest.mark.asyncio
    async def test_llm_classify_discard(self, llm_classifier, mock_llm_processor):
        """测试LLM分类丢弃"""
        mock_llm_processor.classify_message.return_value = LLMClassificationResult(
            layer="discard",
            confidence=0.2,
            reason="LLM丢弃",
            metadata={}
        )
        
        result = await llm_classifier.classify("无意义消息")
        
        assert result.layer == ProcessingLayer.DISCARD
    
    @pytest.mark.asyncio
    async def test_llm_fallback_to_local(self, llm_classifier, mock_llm_processor):
        """测试LLM失败回退到本地"""
        mock_llm_processor.classify_message.return_value = None
        
        result = await llm_classifier.classify("我喜欢猫")
        
        # 应该回退到本地处理
        assert result.source == "local"
    
    @pytest.mark.asyncio
    async def test_llm_not_available_fallback(self, trigger_detector, emotion_analyzer):
        """测试LLM不可用回退"""
        mock_processor = Mock(spec=LLMMessageProcessor)
        mock_processor.is_available.return_value = False
        
        classifier = MessageClassifier(
            trigger_detector=trigger_detector,
            emotion_analyzer=emotion_analyzer,
            llm_processor=mock_processor,
            config={"llm_processing_mode": "llm"}
        )
        
        result = await classifier.classify("我喜欢猫")
        
        assert result.source == "local"


# =============================================================================
# 混合模式测试
# =============================================================================

class TestHybridMode:
    """混合模式测试"""
    
    @pytest.mark.asyncio
    async def test_high_confidence_skip_llm(self, hybrid_classifier, mock_llm_processor):
        """测试高置信度跳过LLM"""
        # 使用显式触发器，本地置信度应该很高
        result = await hybrid_classifier.classify("请记住这个")
        
        # 高置信度应该直接使用本地结果，不调用LLM
        assert result.source == "local"
        mock_llm_processor.classify_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_low_confidence_use_llm(self, hybrid_classifier, mock_llm_processor):
        """测试低置信度使用LLM确认"""
        # 创建一个边缘情况的消息（没有明显触发器，情感中性）
        result = await hybrid_classifier.classify("这是一条普通的消息")
        
        # 本地分类置信度中等时，应该调用LLM进行确认
        mock_llm_processor.classify_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_llm_confirms_local(self, hybrid_classifier, mock_llm_processor):
        """测试LLM确认本地结果"""
        mock_llm_processor.classify_message.return_value = LLMClassificationResult(
            layer="immediate",
            confidence=0.85,
            reason="LLM确认",
            metadata={}
        )
        
        result = await hybrid_classifier.classify("边缘消息")
        
        assert result.source == "llm"
        assert result.confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_llm_low_confidence_use_local(self, hybrid_classifier, mock_llm_processor):
        """测试LLM低置信度使用本地结果"""
        mock_llm_processor.classify_message.return_value = LLMClassificationResult(
            layer="batch",
            confidence=0.4,  # 低置信度
            reason="不确定",
            metadata={}
        )
        
        result = await hybrid_classifier.classify("边缘消息")
        
        # 应该使用本地结果
        assert result.source == "local"
    
    @pytest.mark.asyncio
    async def test_llm_failure_use_local(self, hybrid_classifier, mock_llm_processor):
        """测试LLM失败使用本地"""
        mock_llm_processor.classify_message.side_effect = Exception("LLM Error")
        
        result = await hybrid_classifier.classify("边缘消息")
        
        assert result.source == "local"
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, hybrid_classifier):
        """测试统计追踪"""
        # 执行几次分类
        await hybrid_classifier.classify("哈哈")  # discard
        await hybrid_classifier.classify("请记住")  # immediate, local
        
        stats = hybrid_classifier.get_stats()
        assert stats["local_classifications"] >= 2


# =============================================================================
# 配置测试
# =============================================================================

class TestConfiguration:
    """配置测试"""
    
    def test_custom_thresholds(self, trigger_detector, emotion_analyzer):
        """测试自定义阈值"""
        classifier = MessageClassifier(
            trigger_detector=trigger_detector,
            emotion_analyzer=emotion_analyzer,
            config={
                "immediate_trigger_confidence": 0.9,  # 更高的阈值
                "immediate_emotion_intensity": 0.8
            }
        )
        
        assert classifier.immediate_trigger_confidence == 0.9
        assert classifier.immediate_emotion_intensity == 0.8
    
    def test_default_thresholds(self, trigger_detector, emotion_analyzer):
        """测试默认阈值"""
        classifier = MessageClassifier(
            trigger_detector=trigger_detector,
            emotion_analyzer=emotion_analyzer
        )
        
        assert classifier.immediate_trigger_confidence == 0.8
        assert classifier.immediate_emotion_intensity == 0.7


# =============================================================================
# 上下文传递测试
# =============================================================================

class TestContextPassing:
    """上下文传递测试"""
    
    @pytest.mark.asyncio
    async def test_context_passed_to_emotion_analyzer(self, local_classifier, emotion_analyzer):
        """测试上下文传递给情感分析器"""
        context = {"session_id": "test123", "history": []}
        
        await local_classifier.classify("测试消息", context)
        
        # 验证上下文被传递
        call_args = emotion_analyzer.analyze_emotion.call_args
        assert call_args[0][1] == context
    
    @pytest.mark.asyncio
    async def test_context_passed_to_llm(self, llm_classifier, mock_llm_processor):
        """测试上下文传递给LLM"""
        context = {"user_type": "premium"}
        
        await llm_classifier.classify("测试消息", context)
        
        # 验证上下文被传递 (context是第二个位置参数)
        call_args = mock_llm_processor.classify_message.call_args
        assert call_args[0][1] == context


# =============================================================================
# 边界测试
# =============================================================================

class TestEdgeCases:
    """边界测试"""
    
    @pytest.mark.asyncio
    async def test_empty_message(self, local_classifier):
        """测试空消息"""
        result = await local_classifier.classify("")
        
        assert result.layer == ProcessingLayer.DISCARD
    
    @pytest.mark.asyncio
    async def test_whitespace_message(self, local_classifier):
        """测试空白消息"""
        result = await local_classifier.classify("   \n\t  ")
        
        assert result.layer == ProcessingLayer.DISCARD
    
    @pytest.mark.asyncio
    async def test_very_long_message(self, local_classifier):
        """测试超长消息"""
        long_message = "我喜欢猫" * 1000
        
        result = await local_classifier.classify(long_message)
        
        # 应该能够处理，不会崩溃
        assert isinstance(result, ClassificationResult)
    
    @pytest.mark.asyncio
    async def test_special_characters(self, local_classifier):
        """测试特殊字符"""
        message = "我喜欢猫！🐱 <tag> \\n\\t @mention #hashtag"
        
        result = await local_classifier.classify(message)
        
        assert isinstance(result, ClassificationResult)
    
    @pytest.mark.asyncio
    async def test_unicode_message(self, local_classifier):
        """测试Unicode消息"""
        message = "我喜欢猫🐱 dogs 日本語 العربية"
        
        result = await local_classifier.classify(message)
        
        assert isinstance(result, ClassificationResult)


# =============================================================================
# 性能测试
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_local_mode_performance(self, local_classifier):
        """测试本地模式性能"""
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        # 处理100条消息
        for i in range(100):
            await local_classifier.classify(f"测试消息{i}")
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # 本地模式应该很快（100条<1秒）
        assert elapsed < 1.0
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, local_classifier):
        """测试内存效率"""
        import gc
        
        gc.collect()
        
        # 处理大量消息
        for i in range(1000):
            await local_classifier.classify(f"消息{i}")
        
        gc.collect()
        
        # 如果内存泄漏，这里可能会失败
        # 实际断言取决于具体的内存监控方法


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

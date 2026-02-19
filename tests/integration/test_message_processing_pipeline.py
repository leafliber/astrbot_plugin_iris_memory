"""
消息处理管道集成测试

测试完整的端到端流程：
1. 消息接收 -> 分类 -> 批量处理 -> 主动回复
2. 确保各模块协同工作
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from iris_memory.core.types import TriggerMatch


# =============================================================================
# 端到端流程测试
# =============================================================================

@pytest.mark.asyncio
class TestEndToEndPipeline:
    """端到端管道测试"""
    
    async def test_simple_message_flow(self):
        """测试简单消息流程"""
        # 模拟完整的消息处理流程
        from iris_memory.capture.message_classifier import MessageClassifier, ProcessingLayer
        from iris_memory.capture.batch_processor import MessageBatchProcessor
        from iris_memory.capture.capture_engine import MemoryCaptureEngine
        
        # 模拟组件
        capture_engine = Mock(spec=MemoryCaptureEngine)
        capture_engine.capture_memory = AsyncMock(return_value=Mock(
            id="test_memory",
            storage_layer=Mock(value="working")
        ))
        capture_engine.trigger_detector = Mock()
        capture_engine.trigger_detector._is_negative_sample = Mock(return_value=False)
        capture_engine.trigger_detector.detect_triggers = Mock(return_value=[])
        
        # 创建分类器
        classifier = MessageClassifier(
            trigger_detector=capture_engine.trigger_detector,
            config={"llm_processing_mode": "local"}
        )
        
        # 创建批量处理器
        batch_processor = MessageBatchProcessor(
            capture_engine=capture_engine,
            threshold_count=3
        )
        await batch_processor.start()
        
        try:
            # 模拟消息处理
            messages = ["我喜欢猫", "今天天气不错", "明天见"]
            
            for msg in messages:
                # 分类
                classification = await classifier.classify(msg)
                
                if classification.layer == ProcessingLayer.BATCH:
                    # 加入批量队列
                    await batch_processor.add_message(
                        content=msg,
                        user_id="test_user"
                    )
            
            # 验证批量处理被触发
            # （由于阈值是3，最后一条应该触发处理）
            
        finally:
            await batch_processor.stop()
    
    async def test_high_value_message_flow(self):
        """测试高价值消息流程"""
        from iris_memory.capture.message_classifier import MessageClassifier, ProcessingLayer
        
        # 模拟触发器检测器
        trigger_detector = Mock()
        trigger_detector._is_negative_sample = Mock(return_value=False)
        trigger_detector.detect_triggers = Mock(return_value=[
            TriggerMatch(type=Mock(), pattern="记住", confidence=0.9, position=0)
        ])
        
        classifier = MessageClassifier(
            trigger_detector=trigger_detector,
            config={"immediate_trigger_confidence": 0.8}
        )
        
        # 高价值消息应该被立即处理
        classification = await classifier.classify("请记住我喜欢猫")
        
        assert classification.layer == ProcessingLayer.IMMEDIATE
    
    async def test_discard_message_flow(self):
        """测试丢弃消息流程"""
        from iris_memory.capture.message_classifier import MessageClassifier, ProcessingLayer
        
        trigger_detector = Mock()
        trigger_detector._is_negative_sample = Mock(return_value=True)
        trigger_detector.detect_triggers = Mock(return_value=[])
        
        classifier = MessageClassifier(
            trigger_detector=trigger_detector
        )
        
        # 闲聊消息应该被丢弃
        classification = await classifier.classify("哈哈")
        
        assert classification.layer == ProcessingLayer.DISCARD


# =============================================================================
# 主动回复集成测试
# =============================================================================

@pytest.mark.asyncio
class TestProactiveReplyIntegration:
    """主动回复集成测试"""
    
    async def test_proactive_reply_full_flow(self):
        """测试主动回复完整流程"""
        from iris_memory.proactive.proactive_reply_detector import ProactiveReplyDetector
        from iris_memory.proactive.proactive_manager import ProactiveReplyManager
        
        # 模拟情感分析器
        emotion_analyzer = Mock()
        emotion_analyzer.analyze_emotion = AsyncMock(return_value={
            "primary": "sad",
            "intensity": 0.8,
            "confidence": 0.9
        })
        
        # 创建检测器
        detector = ProactiveReplyDetector(
            emotion_analyzer=emotion_analyzer
        )
        
        # 测试需要回复的消息
        result = await detector.analyze(
            messages=["我好难过，你能陪我聊聊吗？"],
            user_id="test_user"
        )
        
        assert result.should_reply is True
    
    async def test_proactive_reply_with_cooldown(self):
        """测试主动回复冷却机制"""
        from iris_memory.proactive.proactive_manager import ProactiveReplyManager
        
        # 模拟组件
        detector = Mock()
        detector.analyze = AsyncMock(return_value=Mock(
            should_reply=True,
            urgency=Mock(value="high"),
            reason="test",
            suggested_delay=0,
            reply_context={}
        ))
        
        manager = ProactiveReplyManager(
            reply_detector=detector,
            event_queue=asyncio.Queue(),
            astrbot_context=Mock(),
            config={
                "enable_proactive_reply": True,
                "reply_cooldown": 60
            }
        )
        await manager.initialize()
        
        try:
            # 第一次处理
            await manager.handle_batch(["消息1"], user_id="user1")
            
            # 立即第二次处理（在冷却期内）
            await manager.handle_batch(["消息2"], user_id="user1")
            
            # 应该只有一个任务（冷却防重）
            assert manager.pending_tasks.qsize() == 1
            
        finally:
            await manager.stop()


# =============================================================================
# LLM集成测试
# =============================================================================

@pytest.mark.asyncio
class TestLLMIntegration:
    """LLM集成测试"""
    
    async def test_llm_classification_integration(self):
        """测试LLM分类集成"""
        from iris_memory.processing.llm_processor import LLMMessageProcessor
        from iris_memory.capture.message_classifier import MessageClassifier
        
        # 模拟LLM处理器
        llm_processor = Mock(spec=LLMMessageProcessor)
        llm_processor.is_available.return_value = True
        llm_processor.classify_message = AsyncMock(return_value=Mock(
            layer="immediate",
            confidence=0.9,
            reason="LLM决定",
            metadata={}
        ))
        
        # 创建LLM模式分类器
        classifier = MessageClassifier(
            llm_processor=llm_processor,
            config={"llm_processing_mode": "llm"}
        )
        
        # 分类消息
        result = await classifier.classify("测试消息")
        
        # 验证LLM被调用
        llm_processor.classify_message.assert_called_once()
        assert result.source == "llm"
    
    async def test_llm_fallback_to_local(self):
        """测试LLM失败回退到本地"""
        from iris_memory.processing.llm_processor import LLMMessageProcessor
        from iris_memory.capture.message_classifier import MessageClassifier
        
        # 模拟失败的LLM处理器
        llm_processor = Mock(spec=LLMMessageProcessor)
        llm_processor.is_available.return_value = True
        llm_processor.classify_message = AsyncMock(return_value=None)
        
        # 模拟触发器检测器
        trigger_detector = Mock()
        trigger_detector._is_negative_sample = Mock(return_value=False)
        trigger_detector.detect_triggers = Mock(return_value=[])
        
        classifier = MessageClassifier(
            trigger_detector=trigger_detector,
            llm_processor=llm_processor,
            config={"llm_processing_mode": "llm"}
        )
        
        result = await classifier.classify("普通消息")
        
        # 应该回退到本地
        assert result.source == "local"


# =============================================================================
# 错误恢复测试
# =============================================================================

@pytest.mark.asyncio
class TestErrorRecovery:
    """错误恢复测试"""
    
    async def test_batch_processor_error_recovery(self):
        """测试批量处理器错误恢复"""
        from iris_memory.capture.batch_processor import MessageBatchProcessor
        
        # 模拟会失败的捕获引擎
        capture_engine = Mock()
        capture_engine.capture_memory = AsyncMock(side_effect=Exception("捕获失败"))
        
        batch_processor = MessageBatchProcessor(
            capture_engine=capture_engine,
            threshold_count=2
        )
        await batch_processor.start()
        
        try:
            # 添加消息
            await batch_processor.add_message("消息1", "user1")
            await batch_processor.add_message("消息2", "user1")
            
            # 等待处理
            await asyncio.sleep(0.5)
            
            # 不应该崩溃，应该继续运行
            assert batch_processor.is_running is True
            
        finally:
            await batch_processor.stop()
    
    async def test_proactive_manager_error_recovery(self):
        """测试主动回复管理器错误恢复"""
        from iris_memory.proactive.proactive_manager import ProactiveReplyManager
        
        # 模拟检测器
        detector = Mock()
        detector.analyze = AsyncMock(return_value=Mock(
            should_reply=True,
            urgency=Mock(value="high"),
            reason="test",
            suggested_delay=0,
            reply_context={}
        ))
        
        # 使用一个会在put_nowait时失败的事件队列来触发错误
        mock_queue = asyncio.Queue()
        mock_context = Mock()
        
        manager = ProactiveReplyManager(
            reply_detector=detector,
            event_queue=mock_queue,
            astrbot_context=mock_context,
            config={"enable_proactive_reply": True}
        )
        await manager.initialize()
        
        try:
            # 发送消息，让任务入队
            await manager.handle_batch(["消息"], "user1")
            # 等待处理循环处理任务（会因 ProactiveMessageEvent 构造失败而记录错误）
            await asyncio.sleep(0.5)
            
            # 验证管理器仍在运行（错误恢复）
            assert manager.is_running is True
            
        finally:
            await manager.stop()


# =============================================================================
# 性能测试
# =============================================================================

@pytest.mark.slow
@pytest.mark.asyncio
class TestPerformance:
    """性能测试"""
    
    async def test_high_volume_message_processing(self):
        """测试高容量消息处理"""
        from iris_memory.capture.message_classifier import MessageClassifier
        
        trigger_detector = Mock()
        trigger_detector._is_negative_sample = Mock(return_value=False)
        trigger_detector.detect_triggers = Mock(return_value=[])
        
        classifier = MessageClassifier(
            trigger_detector=trigger_detector,
            config={"llm_processing_mode": "local"}
        )
        
        start_time = asyncio.get_event_loop().time()
        
        # 处理1000条消息
        for i in range(1000):
            await classifier.classify(f"消息{i}")
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # 应该在1秒内完成
        assert elapsed < 1.0
    
    async def test_concurrent_batch_processing(self):
        """测试并发批量处理"""
        from iris_memory.capture.batch_processor import MessageBatchProcessor
        
        capture_engine = Mock()
        capture_engine.capture_memory = AsyncMock(return_value=Mock(
            id="test",
            storage_layer=Mock(value="working")
        ))
        
        batch_processor = MessageBatchProcessor(
            capture_engine=capture_engine,
            threshold_count=100
        )
        await batch_processor.start()
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 并发添加消息到不同会话
            tasks = [
                batch_processor.add_message(f"消息{i}", f"user{i % 10}")
                for i in range(100)
            ]
            
            await asyncio.gather(*tasks)
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # 应该在2秒内完成
            assert elapsed < 2.0
            
        finally:
            await batch_processor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

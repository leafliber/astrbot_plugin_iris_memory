"""
批量消息处理器测试

测试批量处理的核心功能：
- 消息队列管理
- 阈值触发机制
- 多种处理模式（summary/filter/hybrid）
- 主动回复集成
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from iris_memory.capture.batch_processor import (
    MessageBatchProcessor,
    QueuedMessage
)
from iris_memory.capture.capture_engine import MemoryCaptureEngine
from iris_memory.processing.llm_processor import LLMMessageProcessor, LLMSummaryResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_capture_engine():
    """模拟捕获引擎"""
    engine = Mock(spec=MemoryCaptureEngine)
    engine.capture_memory = AsyncMock(return_value=Mock(
        id="test_memory_id",
        storage_layer=Mock(value="working")
    ))
    return engine


@pytest.fixture
def mock_llm_processor():
    """模拟LLM处理器"""
    processor = Mock(spec=LLMMessageProcessor)
    processor.is_available = Mock(return_value=True)
    processor.generate_summary = AsyncMock(return_value=LLMSummaryResult(
        summary="LLM生成的摘要",
        key_points=["要点1", "要点2"],
        user_preferences=["偏好1"],
        token_used=50
    ))
    return processor


@pytest.fixture
def mock_proactive_manager():
    """模拟主动回复管理器"""
    manager = MagicMock()
    manager.process_message = AsyncMock()
    return manager


@pytest.fixture
def basic_processor(mock_capture_engine):
    """基础处理器（无LLM，无主动回复）"""
    return MessageBatchProcessor(
        capture_engine=mock_capture_engine,
        threshold_count=5,
        threshold_interval=300,
        processing_mode="hybrid"
    )


@pytest.fixture
def llm_processor(mock_capture_engine, mock_llm_processor):
    """带LLM的处理器"""
    return MessageBatchProcessor(
        capture_engine=mock_capture_engine,
        llm_processor=mock_llm_processor,
        threshold_count=5,
        threshold_interval=300,
        processing_mode="hybrid",
        use_llm_summary=True
    )


@pytest.fixture
def full_processor(mock_capture_engine, mock_llm_processor, mock_proactive_manager):
    """完整处理器（LLM + 主动回复）"""
    return MessageBatchProcessor(
        capture_engine=mock_capture_engine,
        llm_processor=mock_llm_processor,
        proactive_manager=mock_proactive_manager,
        threshold_count=5,
        threshold_interval=300,
        processing_mode="hybrid",
        use_llm_summary=True
    )


@pytest_asyncio.fixture
async def started_processor(basic_processor):
    """已启动的处理器"""
    await basic_processor.start()
    yield basic_processor
    await basic_processor.stop()


# =============================================================================
# 初始化和生命周期测试
# =============================================================================

class TestLifecycle:
    """生命周期测试"""
    
    @pytest.mark.asyncio
    async def test_start_stop(self, basic_processor):
        """测试启动和停止"""
        await basic_processor.start()
        assert basic_processor.is_running is True
        assert basic_processor.cleanup_task is not None
        
        await basic_processor.stop()
        assert basic_processor.is_running is False
    
    @pytest.mark.asyncio
    async def test_stop_processes_remaining(self, mock_capture_engine):
        """测试停止时处理剩余消息"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=100  # 高阈值，不会自动触发
        )
        
        await processor.start()
        
        # 添加一些消息
        await processor.add_message("消息1", "user1")
        await processor.add_message("消息2", "user1")
        
        # 停止时应该处理剩余消息
        await processor.stop()
        
        # 验证捕获引擎被调用
        mock_capture_engine.capture_memory.assert_called()
    
    @pytest.mark.asyncio
    async def test_multiple_start_stop(self, basic_processor):
        """测试多次启动停止"""
        await basic_processor.start()
        await basic_processor.stop()
        await basic_processor.start()
        await basic_processor.stop()
        
        # 不应该抛出异常
        assert True


# =============================================================================
# 消息队列测试
# =============================================================================

class TestMessageQueue:
    """消息队列测试"""
    
    @pytest.mark.asyncio
    async def test_add_message_creates_queue(self, started_processor):
        """测试添加消息创建队列"""
        result = await started_processor.add_message("测试消息", "user1")
        
        assert "user1:private" in started_processor.message_queues
        assert len(started_processor.message_queues["user1:private"]) == 1
        assert result is False  # 未达到阈值，不触发处理
    
    @pytest.mark.asyncio
    async def test_add_message_group_chat(self, started_processor):
        """测试群聊消息"""
        await started_processor.add_message("群消息", "user1", group_id="group123")
        
        assert "user1:group123" in started_processor.message_queues
    
    @pytest.mark.asyncio
    async def test_add_message_triggers_processing(self, mock_capture_engine):
        """测试消息触发处理"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=3
        )
        await processor.start()
        
        # 添加3条消息，触发处理
        await processor.add_message("消息1", "user1")
        await processor.add_message("消息2", "user1")
        result = await processor.add_message("消息3", "user1")
        
        assert result is True  # 触发处理
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_queue_per_session(self, started_processor):
        """测试每个会话独立队列"""
        await started_processor.add_message("消息1", "user1")
        await started_processor.add_message("消息2", "user2")
        await started_processor.add_message("消息3", "user1", group_id="group1")
        
        assert len(started_processor.message_queues) == 3
        assert "user1:private" in started_processor.message_queues
        assert "user2:private" in started_processor.message_queues
        assert "user1:group1" in started_processor.message_queues


# =============================================================================
# 阈值测试
# =============================================================================

class TestThresholds:
    """阈值测试"""
    
    @pytest.mark.asyncio
    async def test_count_threshold(self, mock_capture_engine):
        """测试数量阈值"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=5
        )
        await processor.start()
        
        # 添加4条，不触发
        for i in range(4):
            result = await processor.add_message(f"消息{i}", "user1")
            assert result is False
        
        # 第5条触发
        result = await processor.add_message("消息4", "user1")
        assert result is True
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_time_threshold(self, mock_capture_engine):
        """测试时间阈值"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=100,  # 高数量阈值
            threshold_interval=1  # 1秒时间阈值
        )
        await processor.start()
        
        await processor.add_message("消息", "user1")
        
        # 等待超过时间阈值
        await asyncio.sleep(1.5)
        
        # 清理循环应该触发处理
        # 由于清理循环是异步的，我们需要等待
        await asyncio.sleep(0.5)
        
        # 队列应该被清空
        assert len(processor.message_queues.get("user1:private", [])) == 0
        
        await processor.stop()


# =============================================================================
# 处理模式测试
# =============================================================================

class TestProcessingModes:
    """处理模式测试"""
    
    @pytest.mark.asyncio
    async def test_summary_mode(self, mock_capture_engine):
        """测试摘要模式"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=2,
            processing_mode="summary"
        )
        await processor.start()
        
        # 使用较长的消息（>=15字符）避免被短消息合并
        await processor.add_message("我喜欢猫，猫咪真的很可爱，每天撸猫超幸福", "user1")
        await processor.add_message("我也喜欢狗，金毛犬特别温顺，遛狗很开心", "user1")
        
        await processor.stop()
        
        # 验证生成了摘要记忆
        mock_capture_engine.capture_memory.assert_called()
    
    @pytest.mark.asyncio
    async def test_filter_mode(self, mock_capture_engine):
        """测试筛选模式"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=3,
            processing_mode="filter"
        )
        await processor.start()
        
        await processor.add_message("短", "user1")
        await processor.add_message("我喜欢猫，这是一个很长的消息，包含重要信息", "user1")
        await processor.add_message("哈哈", "user1")
        
        await processor.stop()
        
        # 验证只有高价值消息被捕获
        # 长消息应该被捕获
        assert mock_capture_engine.capture_memory.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_hybrid_mode(self, mock_capture_engine):
        """测试混合模式"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=3,
            processing_mode="hybrid"
        )
        await processor.start()
        
        await processor.add_message("我喜欢猫", "user1")  # 高价值
        await processor.add_message("今天天气不错", "user1")  # 普通
        await processor.add_message("明天见", "user1")  # 普通
        
        await processor.stop()
        
        # 高价值消息应该单独捕获
        assert mock_capture_engine.capture_memory.call_count >= 1


# =============================================================================
# LLM集成测试
# =============================================================================

class TestLLMIntegration:
    """LLM集成测试"""
    
    @pytest.mark.asyncio
    async def test_llm_summary_generation(self, mock_capture_engine, mock_llm_processor):
        """测试LLM摘要生成"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            llm_processor=mock_llm_processor,
            threshold_count=2,
            processing_mode="summary",
            use_llm_summary=True
        )
        await processor.start()
        
        # 使用较长的消息（>=15字符）避免被短消息合并
        await processor.add_message("这是第一条消息，它包含了比较长的重要内容", "user1")
        await processor.add_message("这是第二条消息，它也包含了比较长的重要内容", "user1")
        
        await processor.stop()
        
        # 验证LLM被调用生成摘要
        mock_llm_processor.generate_summary.assert_called()
    
    @pytest.mark.asyncio
    async def test_llm_not_available_fallback(self, mock_capture_engine):
        """测试LLM不可用回退"""
        mock_llm = Mock(spec=LLMMessageProcessor)
        mock_llm.is_available.return_value = False
        
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            llm_processor=mock_llm,
            use_llm_summary=True,
            threshold_count=2
        )
        await processor.start()
        
        await processor.add_message("消息1", "user1")
        await processor.add_message("消息2", "user1")
        
        await processor.stop()
        
        # 应该使用本地摘要
        mock_capture_engine.capture_memory.assert_called()
    
    @pytest.mark.asyncio
    async def test_llm_summary_failure_fallback(self, mock_capture_engine, mock_llm_processor):
        """测试LLM摘要失败回退"""
        mock_llm_processor.generate_summary.return_value = None
        
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            llm_processor=mock_llm_processor,
            use_llm_summary=True,
            threshold_count=2
        )
        await processor.start()
        
        await processor.add_message("消息1", "user1")
        await processor.add_message("消息2", "user1")
        
        await processor.stop()
        
        # 应该使用本地摘要
        mock_capture_engine.capture_memory.assert_called()


# =============================================================================
# 主动回复集成测试
# =============================================================================

class TestProactiveReplyIntegration:
    """主动回复集成测试"""
    
    @pytest.mark.asyncio
    async def test_proactive_reply_triggered(self, full_processor):
        """测试主动回复触发"""
        await full_processor.start()
        
        await full_processor.add_message("在吗？", "user1")
        await full_processor.add_message("我想问你个问题", "user1")
        
        await full_processor.stop()
        
        # 验证主动回复管理器被调用
        full_processor.proactive_manager.process_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_proactive_reply_context(self, full_processor):
        """测试主动回复上下文传递"""
        await full_processor.start()
        
        await full_processor.add_message("消息", "user1")
        
        await full_processor.stop()
        
        # 验证主动回复处理器被调用
        full_processor.proactive_manager.process_message.assert_called()


# =============================================================================
# 本地摘要测试
# =============================================================================

class TestLocalSummary:
    """本地摘要测试"""
    
    def test_extract_key_sentences(self, basic_processor):
        """测试关键句提取"""
        messages = [
            "我喜欢猫",
            "它们很可爱",
            "我也喜欢狗"
        ]
        
        summary = basic_processor._generate_local_summary(messages)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_keyword_scoring(self, basic_processor):
        """测试关键词评分"""
        messages = [
            "我喜欢猫",  # 有"喜欢"
            "短",  # 太短
            "明天见"  # 普通
        ]
        
        summary = basic_processor._generate_local_summary(messages)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_empty_messages(self, basic_processor):
        """测试空消息列表"""
        summary = basic_processor._generate_local_summary([])
        
        assert summary == ""


# =============================================================================
# 高价值消息检测测试
# =============================================================================

class TestHighValueDetection:
    """高价值消息检测测试"""
    
    def test_preference_keywords(self, basic_processor):
        """测试偏好关键词"""
        msg = Mock()
        msg.content = "我喜欢喝咖啡"
        
        is_high = basic_processor._is_high_value_message(msg)
        
        assert is_high is True
    
    def test_plan_keywords(self, basic_processor):
        """测试计划关键词"""
        msg = Mock()
        msg.content = "我计划去旅行"
        
        is_high = basic_processor._is_high_value_message(msg)
        
        assert is_high is True
    
    def test_short_message(self, basic_processor):
        """测试短消息"""
        msg = Mock()
        msg.content = "好"
        
        is_high = basic_processor._is_high_value_message(msg)
        
        assert is_high is False
    
    def test_long_message(self, basic_processor):
        """测试长消息"""
        msg = Mock()
        msg.content = "A" * 101  # 超过100字符阈值
        
        is_high = basic_processor._is_high_value_message(msg)
        
        assert is_high is True


# =============================================================================
# 统计测试
# =============================================================================

class TestStatistics:
    """统计测试"""
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, started_processor):
        """测试统计追踪"""
        await started_processor.add_message("消息1", "user1")
        await started_processor.add_message("消息2", "user1")
        
        stats = started_processor.get_stats()
        
        assert "messages_processed" in stats or "batches_processed" in stats
    
    @pytest.mark.asyncio
    async def test_batch_stats(self, mock_capture_engine):
        """测试批次统计"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=2
        )
        await processor.start()
        
        await processor.add_message("消息1", "user1")
        await processor.add_message("消息2", "user1")
        
        await processor.stop()
        
        stats = processor.get_stats()
        assert stats["batches_processed"] >= 1
        assert stats["messages_processed"] >= 2


# =============================================================================
# 边界测试
# =============================================================================

class TestEdgeCases:
    """边界测试"""
    
    @pytest.mark.asyncio
    async def test_empty_message(self, started_processor):
        """测试空消息"""
        result = await started_processor.add_message("", "user1")
        
        # 应该能处理，不崩溃
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_very_long_message(self, started_processor):
        """测试超长消息"""
        long_message = "A" * 10000
        
        result = await started_processor.add_message(long_message, "user1")
        
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_special_characters(self, started_processor):
        """测试特殊字符"""
        message = "🐱 <script> \\n\\t @user #tag"
        
        result = await started_processor.add_message(message, "user1")
        
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, started_processor):
        """测试并发会话"""
        # 并发添加消息到不同会话
        tasks = [
            started_processor.add_message(f"消息{i}", f"user{i % 5}")
            for i in range(20)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert all(isinstance(r, bool) for r in results)


# =============================================================================
# 性能测试
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_high_throughput(self, mock_capture_engine):
        """测试高吞吐量"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=100
        )
        await processor.start()
        
        start_time = asyncio.get_event_loop().time()
        
        # 快速添加100条消息
        for i in range(100):
            await processor.add_message(f"消息{i}", "user1")
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        await processor.stop()
        
        # 应该在合理时间内完成
        assert elapsed < 2.0
    
    @pytest.mark.asyncio
    async def test_many_sessions(self, mock_capture_engine):
        """测试大量会话"""
        processor = MessageBatchProcessor(
            capture_engine=mock_capture_engine,
            threshold_count=1000
        )
        await processor.start()
        
        # 创建100个会话
        for i in range(100):
            await processor.add_message("消息", f"user{i}")
        
        assert len(processor.message_queues) == 100
        
        await processor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

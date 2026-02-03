"""
主动回复管理器测试

测试管理器的核心功能：
- 任务队列管理
- 冷却时间管理
- 每日限制
- 回复流程协调
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
from datetime import datetime, timedelta

from iris_memory.proactive.proactive_manager import (
    ProactiveReplyManager,
    ProactiveReplyTask
)
from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDetector,
    ProactiveReplyDecision,
    ReplyUrgency
)
from iris_memory.proactive.reply_generator import ProactiveReplyGenerator, GeneratedReply


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_reply_detector():
    """模拟回复检测器"""
    detector = Mock(spec=ProactiveReplyDetector)
    detector.analyze = AsyncMock(return_value=ProactiveReplyDecision(
        should_reply=True,
        urgency=ReplyUrgency.HIGH,
        reason="test",
        suggested_delay=0,
        reply_context={"emotion": {}, "signals": {}}
    ))
    return detector


@pytest.fixture
def mock_reply_generator():
    """模拟回复生成器"""
    generator = Mock(spec=ProactiveReplyGenerator)
    generator.astrbot_context = Mock()
    generator.generate_reply = AsyncMock(return_value=GeneratedReply(
        content="这是一个测试回复",
        emotion_tone="neutral",
        referenced_memories=[],
        confidence=0.8,
        metadata={}
    ))
    return generator


@pytest.fixture
def mock_message_sender():
    """模拟消息发送器"""
    sender = Mock()
    sender.send = AsyncMock(return_value=Mock(success=True, message_id="msg_123"))
    sender.is_available = Mock(return_value=True)
    return sender


@pytest_asyncio.fixture
async def enabled_manager(mock_reply_detector, mock_reply_generator, mock_message_sender):
    """启用的管理器"""
    with patch('iris_memory.proactive.proactive_manager.MessageSender', return_value=mock_message_sender):
        manager = ProactiveReplyManager(
            reply_detector=mock_reply_detector,
            reply_generator=mock_reply_generator,
            config={
                "enable_proactive_reply": True,
                "reply_cooldown": 60,
                "max_daily_replies": 20
            }
        )
        await manager.initialize()
        yield manager
        await manager.stop()


@pytest_asyncio.fixture
async def disabled_manager(mock_reply_detector, mock_reply_generator):
    """禁用的管理器"""
    manager = ProactiveReplyManager(
        reply_detector=mock_reply_detector,
        reply_generator=mock_reply_generator,
        config={
            "enable_proactive_reply": False
        }
    )
    yield manager


# =============================================================================
# 初始化测试
# =============================================================================

class TestInitialization:
    """初始化测试"""
    
    @pytest.mark.asyncio
    async def test_initialize_enabled(self, mock_reply_detector, mock_reply_generator):
        """测试启用状态初始化"""
        with patch('iris_memory.proactive.proactive_manager.MessageSender') as MockSender:
            mock_sender = Mock()
            mock_sender.is_available.return_value = True
            MockSender.return_value = mock_sender
            
            manager = ProactiveReplyManager(
                reply_detector=mock_reply_detector,
                reply_generator=mock_reply_generator,
                config={"enable_proactive_reply": True}
            )
            await manager.initialize()
            
            assert manager.is_running is True
            assert manager.processing_task is not None
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_initialize_disabled(self, mock_reply_detector, mock_reply_generator):
        """测试禁用状态初始化"""
        manager = ProactiveReplyManager(
            reply_detector=mock_reply_detector,
            reply_generator=mock_reply_generator,
            config={"enable_proactive_reply": False}
        )
        await manager.initialize()
        
        assert manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_initialize_sender_unavailable(self, mock_reply_detector, mock_reply_generator):
        """测试发送器不可用"""
        with patch('iris_memory.proactive.proactive_manager.MessageSender') as MockSender:
            mock_sender = Mock()
            mock_sender.is_available.return_value = False
            MockSender.return_value = mock_sender
            
            manager = ProactiveReplyManager(
                reply_detector=mock_reply_detector,
                reply_generator=mock_reply_generator,
                config={"enable_proactive_reply": True}
            )
            await manager.initialize()
            
            assert manager.enabled is False


# =============================================================================
# 批量处理测试
# =============================================================================

class TestHandleBatch:
    """批量处理测试"""
    
    @pytest.mark.asyncio
    async def test_handle_batch_creates_task(self, enabled_manager):
        """测试批量处理创建任务"""
        messages = ["消息1", "消息2"]
        
        await enabled_manager.handle_batch(
            messages=messages,
            user_id="user123"
        )
        
        # 任务应该被加入队列
        assert enabled_manager.pending_tasks.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_handle_batch_disabled(self, disabled_manager):
        """测试禁用时批量处理"""
        messages = ["消息1"]
        
        await disabled_manager.handle_batch(
            messages=messages,
            user_id="user123"
        )
        
        # 不应该创建任务
        assert disabled_manager.pending_tasks.qsize() == 0
    
    @pytest.mark.asyncio
    async def test_handle_batch_no_reply_needed(self, mock_reply_detector, mock_reply_generator, mock_message_sender):
        """测试不需要回复的情况"""
        mock_reply_detector.analyze.return_value = ProactiveReplyDecision(
            should_reply=False,
            urgency=ReplyUrgency.IGNORE,
            reason="不需要回复",
            suggested_delay=0,
            reply_context={}
        )
        
        with patch('iris_memory.proactive.proactive_manager.MessageSender', return_value=mock_message_sender):
            manager = ProactiveReplyManager(
                reply_detector=mock_reply_detector,
                reply_generator=mock_reply_generator,
                config={"enable_proactive_reply": True}
            )
            await manager.initialize()
            
            await manager.handle_batch(
                messages=["哈哈"],
                user_id="user123"
            )
            
            # 不应该创建任务
            assert manager.pending_tasks.qsize() == 0
            assert manager.stats["replies_skipped"] == 1
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_handle_batch_empty_messages(self, enabled_manager):
        """测试空消息列表"""
        await enabled_manager.handle_batch(
            messages=[],
            user_id="user123"
        )
        
        assert enabled_manager.pending_tasks.qsize() == 0


# =============================================================================
# 冷却时间测试
# =============================================================================

class TestCooldown:
    """冷却时间测试"""
    
    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicate(self, enabled_manager):
        """测试冷却时间防止重复"""
        messages = ["消息"]
        
        # 第一次处理
        await enabled_manager.handle_batch(messages=messages, user_id="user123")
        assert enabled_manager.pending_tasks.qsize() == 1
        
        # 立即第二次处理（在冷却期内）
        await enabled_manager.handle_batch(messages=messages, user_id="user123")
        assert enabled_manager.pending_tasks.qsize() == 1  # 不应该增加
    
    @pytest.mark.asyncio
    async def test_cooldown_per_session(self, enabled_manager):
        """测试每个会话独立冷却"""
        messages = ["消息"]
        
        # user1 的消息
        await enabled_manager.handle_batch(messages=messages, user_id="user1")
        
        # user2 的消息（不同会话，不受冷却影响）
        await enabled_manager.handle_batch(messages=messages, user_id="user2")
        
        # 应该有两个任务
        assert enabled_manager.pending_tasks.qsize() == 2
    
    @pytest.mark.asyncio
    async def test_cooldown_group_vs_private(self, enabled_manager):
        """测试群聊和私聊独立冷却"""
        messages = ["消息"]
        
        # 私聊
        await enabled_manager.handle_batch(messages=messages, user_id="user1", group_id=None)
        
        # 群聊（相同用户）
        await enabled_manager.handle_batch(messages=messages, user_id="user1", group_id="group1")
        
        assert enabled_manager.pending_tasks.qsize() == 2


# =============================================================================
# 每日限制测试
# =============================================================================

class TestDailyLimit:
    """每日限制测试"""
    
    @pytest.mark.asyncio
    async def test_daily_limit_enforced(self, mock_reply_detector, mock_reply_generator, mock_message_sender):
        """测试每日限制执行"""
        with patch('iris_memory.proactive.proactive_manager.MessageSender', return_value=mock_message_sender):
            manager = ProactiveReplyManager(
                reply_detector=mock_reply_detector,
                reply_generator=mock_reply_generator,
                config={
                    "enable_proactive_reply": True,
                    "max_daily_replies": 2
                }
            )
            await manager.initialize()
            
            # 设置已达到限制
            manager.daily_reply_count["user123"] = 2
            
            await manager.handle_batch(
                messages=["消息"],
                user_id="user123"
            )
            
            # 不应该创建任务
            assert manager.pending_tasks.qsize() == 0
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_daily_limit_per_user(self, enabled_manager):
        """测试每个用户独立限制"""
        messages = ["消息"]
        
        # 用户1达到限制
        enabled_manager.daily_reply_count["user1"] = 20
        
        # 用户2未达限制
        await enabled_manager.handle_batch(messages=messages, user_id="user2")
        
        assert enabled_manager.pending_tasks.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_reset_daily_counts(self, enabled_manager):
        """测试重置每日计数"""
        enabled_manager.daily_reply_count["user1"] = 10
        enabled_manager.daily_reply_count["user2"] = 5
        
        enabled_manager.reset_daily_counts()
        
        assert len(enabled_manager.daily_reply_count) == 0


# =============================================================================
# 任务处理测试
# =============================================================================

class TestTaskProcessing:
    """任务处理测试"""
    
    @pytest.mark.asyncio
    async def test_task_processing_success(self, enabled_manager):
        """测试任务处理成功"""
        messages = ["消息"]
        
        await enabled_manager.handle_batch(messages=messages, user_id="user123")
        
        # 等待处理
        await asyncio.sleep(0.5)
        
        assert enabled_manager.stats["replies_sent"] >= 1
    
    @pytest.mark.asyncio
    async def test_task_processing_with_delay(self, mock_reply_detector, mock_reply_generator, mock_message_sender):
        """测试带延迟的任务处理"""
        mock_reply_detector.analyze.return_value = ProactiveReplyDecision(
            should_reply=True,
            urgency=ReplyUrgency.MEDIUM,
            reason="test",
            suggested_delay=1,  # 1秒延迟
            reply_context={}
        )
        
        with patch('iris_memory.proactive.proactive_manager.MessageSender', return_value=mock_message_sender):
            manager = ProactiveReplyManager(
                reply_detector=mock_reply_detector,
                reply_generator=mock_reply_generator,
                config={"enable_proactive_reply": True}
            )
            await manager.initialize()
            
            await manager.handle_batch(messages=["消息"], user_id="user123")
            
            # 立即检查，还没处理
            assert manager.stats["replies_sent"] == 0
            
            # 等待延迟+处理时间
            await asyncio.sleep(2)
            
            assert manager.stats["replies_sent"] >= 1
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_task_processing_failure(self, mock_reply_detector, mock_reply_generator, mock_message_sender):
        """测试任务处理失败"""
        mock_reply_generator.generate_reply.return_value = None  # 生成失败
        
        with patch('iris_memory.proactive.proactive_manager.MessageSender', return_value=mock_message_sender):
            manager = ProactiveReplyManager(
                reply_detector=mock_reply_detector,
                reply_generator=mock_reply_generator,
                config={"enable_proactive_reply": True}
            )
            await manager.initialize()
            
            await manager.handle_batch(messages=["消息"], user_id="user123")
            await asyncio.sleep(0.5)
            
            assert manager.stats["replies_failed"] >= 1
            
            await manager.stop()


# =============================================================================
# 停止处理测试
# =============================================================================

class TestStopProcessing:
    """停止处理测试"""
    
    @pytest.mark.asyncio
    async def test_stop_processes_pending(self, mock_reply_detector, mock_reply_generator, mock_message_sender):
        """测试停止时处理待处理任务"""
        with patch('iris_memory.proactive.proactive_manager.MessageSender', return_value=mock_message_sender):
            manager = ProactiveReplyManager(
                reply_detector=mock_reply_detector,
                reply_generator=mock_reply_generator,
                config={"enable_proactive_reply": True}
            )
            await manager.initialize()
            
            # 添加一些任务
            for i in range(3):
                await manager.handle_batch(messages=[f"消息{i}"], user_id=f"user{i}")
            
            # 停止
            await manager.stop()
            
            # 验证任务被处理
            assert manager.stats["replies_sent"] >= 1


# =============================================================================
# 统计信息测试
# =============================================================================

class TestStatistics:
    """统计信息测试"""
    
    @pytest.mark.asyncio
    async def test_get_stats(self, enabled_manager):
        """测试获取统计信息"""
        await enabled_manager.handle_batch(messages=["消息"], user_id="user123")
        await asyncio.sleep(0.5)
        
        stats = enabled_manager.get_stats()
        
        assert "replies_sent" in stats
        assert "replies_skipped" in stats
        assert "replies_failed" in stats
        assert "pending_tasks" in stats
    
    def test_initial_stats(self, enabled_manager):
        """测试初始统计"""
        stats = enabled_manager.get_stats()
        
        assert stats["replies_sent"] == 0
        assert stats["replies_skipped"] == 0
        assert stats["replies_failed"] == 0


# =============================================================================
# 上下文传递测试
# =============================================================================

class TestContextPassing:
    """上下文传递测试"""
    
    @pytest.mark.asyncio
    async def test_context_passed_to_detector(self, enabled_manager):
        """测试上下文传递给检测器"""
        context = {
            "time_span": 3600,
            "user_persona": {"name": "Test"}
        }
        
        await enabled_manager.handle_batch(
            messages=["消息"],
            user_id="user123",
            context=context
        )
        
        # 验证检测器收到上下文
        call_args = enabled_manager.reply_detector.analyze.call_args
        assert call_args[1]["context"] == context
    
    @pytest.mark.asyncio
    async def test_context_passed_to_generator(self, enabled_manager):
        """测试上下文传递给生成器"""
        emotional_state = Mock()
        
        await enabled_manager.handle_batch(
            messages=["消息"],
            user_id="user123",
            context={"emotional_state": emotional_state}
        )
        
        await asyncio.sleep(0.5)
        
        # 验证生成器收到情感状态
        call_args = enabled_manager.reply_generator.generate_reply.call_args
        assert call_args[1].get("emotional_state") == emotional_state


# =============================================================================
# 配置测试
# =============================================================================

class TestConfiguration:
    """配置测试"""
    
    def test_custom_cooldown(self, mock_reply_detector, mock_reply_generator):
        """测试自定义冷却时间"""
        manager = ProactiveReplyManager(
            reply_detector=mock_reply_detector,
            reply_generator=mock_reply_generator,
            config={"reply_cooldown": 120}
        )
        
        assert manager.cooldown_seconds == 120
    
    def test_custom_daily_limit(self, mock_reply_detector, mock_reply_generator):
        """测试自定义每日限制"""
        manager = ProactiveReplyManager(
            reply_detector=mock_reply_detector,
            reply_generator=mock_reply_generator,
            config={"max_daily_replies": 50}
        )
        
        assert manager.max_daily_replies == 50


# =============================================================================
# 边界测试
# =============================================================================

class TestEdgeCases:
    """边界测试"""
    
    @pytest.mark.asyncio
    async def test_empty_message_list(self, enabled_manager):
        """测试空消息列表"""
        await enabled_manager.handle_batch(
            messages=[],
            user_id="user123"
        )
        
        assert enabled_manager.pending_tasks.qsize() == 0
    
    @pytest.mark.asyncio
    async def test_very_long_message(self, enabled_manager):
        """测试超长消息"""
        long_message = "A" * 10000
        
        await enabled_manager.handle_batch(
            messages=[long_message],
            user_id="user123"
        )
        
        # 应该能处理，不崩溃
        assert enabled_manager.pending_tasks.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_special_characters(self, enabled_manager):
        """测试特殊字符"""
        message = "🐱 <script> \\n\\t"
        
        await enabled_manager.handle_batch(
            messages=[message],
            user_id="user123"
        )
        
        assert enabled_manager.pending_tasks.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_batches(self, enabled_manager):
        """测试并发批量处理"""
        tasks = [
            enabled_manager.handle_batch([f"消息{i}"], f"user{i % 5}")
            for i in range(20)
        ]
        
        await asyncio.gather(*tasks)
        
        # 应该有任务被创建
        assert enabled_manager.pending_tasks.qsize() <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

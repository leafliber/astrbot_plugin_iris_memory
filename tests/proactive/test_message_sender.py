"""
消息发送器测试

测试消息发送的核心功能：
- 发送方法检测
- 私聊发送
- 群聊发送
- 错误处理
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional

from iris_memory.proactive.message_sender import MessageSender, SendResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_context_send():
    """模拟上下文发送方法"""
    # 使用spec限制属性，避免自动创建provider/platform等属性导致检测到provider_send
    context = Mock(spec=['send_message'])
    context.send_message = AsyncMock(return_value="msg_id_123")
    return context


@pytest.fixture
def mock_platform_send():
    """模拟平台发送方法"""
    # 使用spec_set严格限制属性，确保hasattr对未定义属性返回False
    context = Mock(spec_set=['platform'])
    platform_mock = Mock(spec_set=['send_private_msg', 'send_group_msg'])
    platform_mock.send_private_msg = AsyncMock(return_value="private_id_123")
    platform_mock.send_group_msg = AsyncMock(return_value="group_id_123")
    context.platform = platform_mock
    return context


@pytest.fixture
def mock_service_send():
    """模拟服务发送方法"""
    context = Mock(spec=['message_service'])
    context.message_service = Mock(spec=['send'])
    context.message_service.send = AsyncMock(return_value="service_id_123")
    return context


@pytest.fixture
def mock_event_send():
    """模拟事件发送方法"""
    context = Mock(spec=['_event', '_send_callback'])
    context._event = Mock()
    context._send_callback = AsyncMock(return_value=True)
    return context


@pytest.fixture
def context_sender(mock_context_send):
    """上下文发送器"""
    return MessageSender(mock_context_send)


@pytest.fixture
def platform_sender(mock_platform_send):
    """平台发送器"""
    return MessageSender(mock_platform_send)


@pytest.fixture
def service_sender(mock_service_send):
    """服务发送器"""
    return MessageSender(mock_service_send)


# =============================================================================
# 发送方法检测测试
# =============================================================================

class TestSendMethodDetection:
    """发送方法检测测试"""
    
    def test_detect_context_send(self, mock_context_send):
        """检测上下文发送"""
        sender = MessageSender(mock_context_send)
        
        assert sender.send_method == "context_send"
        assert sender.is_available() is True
    
    def test_detect_platform_send(self, mock_platform_send):
        """检测平台发送"""
        sender = MessageSender(mock_platform_send)
        
        assert sender.send_method == "platform_send"
        assert sender.is_available() is True
    
    def test_detect_service_send(self, mock_service_send):
        """检测服务发送"""
        sender = MessageSender(mock_service_send)
        
        assert sender.send_method == "service_send"
        assert sender.is_available() is True
    
    def test_no_send_method(self):
        """测试无发送方法"""
        # 使用spec=object确保Mock不会自动创建属性
        context = Mock(spec=object)
        # 没有任何发送方法
        
        sender = MessageSender(context)
        
        assert sender.send_method is None
        assert sender.is_available() is False
    
    def test_no_context(self):
        """测试无上下文"""
        sender = MessageSender(None)
        
        assert sender.send_method is None
        assert sender.is_available() is False


# =============================================================================
# 上下文发送测试
# =============================================================================

class TestContextSend:
    """上下文发送测试"""
    
    @pytest.mark.asyncio
    async def test_send_private_message(self, context_sender):
        """测试发送私聊消息"""
        result = await context_sender.send(
            content="测试消息",
            user_id="user123",
            group_id=None,
            umo="platform:FriendMessage:user123"
        )
        
        assert result.success is True
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_send_group_message(self, context_sender):
        """测试发送群聊消息"""
        result = await context_sender.send(
            content="测试消息",
            user_id="user123",
            group_id="group456",
            umo="platform:GroupMessage:group456"
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_context_send_without_umo(self, context_sender):
        """测试无 umo 时发送失败"""
        result = await context_sender.send(
            content="测试消息",
            user_id="user123"
        )
        
        assert result.success is False
        assert "unified_msg_origin" in result.error
    
    @pytest.mark.asyncio
    async def test_context_send_error(self, context_sender):
        """测试上下文发送错误"""
        context_sender.astrbot_context.send_message.side_effect = Exception("发送失败")
        
        result = await context_sender.send(
            content="测试消息",
            user_id="user123",
            umo="platform:FriendMessage:user123"
        )
        
        assert result.success is False
        assert "发送失败" in result.error


# =============================================================================
# 平台发送测试
# =============================================================================

class TestPlatformSend:
    """平台发送测试"""
    
    @pytest.mark.asyncio
    async def test_send_private(self, platform_sender):
        """测试私聊发送"""
        result = await platform_sender.send(
            content="私聊消息",
            user_id="user123",
            group_id=None
        )
        
        assert result.success is True
        assert result.message_id == "private_id_123"
        
        # 验证调用私聊方法
        platform_sender.astrbot_context.platform.send_private_msg.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_group(self, platform_sender):
        """测试群聊发送"""
        result = await platform_sender.send(
            content="群聊消息",
            user_id="user123",
            group_id="group456"
        )
        
        assert result.success is True
        assert result.message_id == "group_id_123"
        
        # 验证调用群聊方法
        platform_sender.astrbot_context.platform.send_group_msg.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_platform_send_error(self, platform_sender):
        """测试平台发送错误"""
        platform_sender.astrbot_context.platform.send_private_msg.side_effect = Exception("平台错误")
        
        result = await platform_sender.send(
            content="测试消息",
            user_id="user123"
        )
        
        assert result.success is False
        assert "平台错误" in result.error


# =============================================================================
# 服务发送测试
# =============================================================================

class TestServiceSend:
    """服务发送测试"""
    
    @pytest.mark.asyncio
    async def test_service_send_success(self, service_sender):
        """测试服务发送成功"""
        result = await service_sender.send(
            content="服务消息",
            user_id="user123",
            group_id="group456"
        )
        
        assert result.success is True
        assert result.message_id == "service_id_123"
    
    @pytest.mark.asyncio
    async def test_service_send_error(self, service_sender):
        """测试服务发送错误"""
        service_sender.astrbot_context.message_service.send.side_effect = Exception("服务错误")
        
        result = await service_sender.send(
            content="测试消息",
            user_id="user123"
        )
        
        assert result.success is False
        assert "服务错误" in result.error


# =============================================================================
# 回退测试
# =============================================================================

class TestFallbackMethods:
    """回退方法测试"""
    
    @pytest.mark.asyncio
    async def test_unknown_send_method(self):
        """测试未知发送方法"""
        context = Mock()
        context.unknown_method = AsyncMock()
        
        sender = MessageSender(context)
        sender.send_method = "unknown_method"
        
        result = await sender.send(
            content="测试消息",
            user_id="user123"
        )
        
        assert result.success is False
        assert "Unknown send method" in result.error
    
    @pytest.mark.asyncio
    async def test_unavailable_sender(self):
        """测试不可用的发送器"""
        sender = MessageSender(None)
        
        result = await sender.send(
            content="测试消息",
            user_id="user123"
        )
        
        assert result.success is False
        assert "No send method available" in result.error


# =============================================================================
# 发送内容测试
# =============================================================================

class TestSendContent:
    """发送内容测试"""
    
    @pytest.mark.asyncio
    async def test_send_empty_content(self, context_sender):
        """测试发送空内容"""
        result = await context_sender.send(
            content="",
            user_id="user123",
            umo="fakeid:GroupMessage:test_group"
        )
        
        # 空内容应该也能发送
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_send_long_content(self, context_sender):
        """测试发送长内容"""
        long_content = "A" * 10000
        
        result = await context_sender.send(
            content=long_content,
            user_id="user123",
            umo="fakeid:GroupMessage:test_group"
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_send_special_characters(self, context_sender):
        """测试发送特殊字符"""
        content = "你好🐱 <script> \\n\\t @user #tag"
        
        result = await context_sender.send(
            content=content,
            user_id="user123",
            umo="fakeid:GroupMessage:test_group"
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_send_unicode(self, context_sender):
        """测试发送Unicode"""
        content = "你好🐱 日本語 العربية"
        
        result = await context_sender.send(
            content=content,
            user_id="user123",
            umo="fakeid:GroupMessage:test_group"
        )
        
        assert result.success is True


# =============================================================================
# Session信息测试
# =============================================================================

class TestSessionInfo:
    """Session信息测试"""
    
    @pytest.mark.asyncio
    async def test_send_with_session_info(self, context_sender):
        """测试带Session信息发送"""
        session_info = {"platform": "wechat", "chat_type": "private"}
        
        result = await context_sender.send(
            content="测试消息",
            user_id="user123",
            group_id=None,
            session_info=session_info,
            umo="fakeid:GroupMessage:test_group"
        )
        
        assert result.success is True


# =============================================================================
# 边界测试
# =============================================================================

class TestEdgeCases:
    """边界测试"""
    
    @pytest.mark.asyncio
    async def test_send_with_none_user_id(self, context_sender):
        """测试None用户ID"""
        result = await context_sender.send(
            content="测试消息",
            user_id=None
        )
        
        # 应该尝试发送
        assert isinstance(result, SendResult)
    
    @pytest.mark.asyncio
    async def test_send_with_empty_group_id(self, context_sender):
        """测试空群组ID"""
        result = await context_sender.send(
            content="测试消息",
            user_id="user123",
            group_id=""
        )
        
        assert isinstance(result, SendResult)
    
    @pytest.mark.asyncio
    async def test_context_method_exception(self):
        """测试上下文方法异常"""
        context = Mock()
        context.send_message = Mock(side_effect=AttributeError("No such method"))
        
        sender = MessageSender(context)
        sender.send_method = "context_send"
        
        result = await sender.send(
            content="测试消息",
            user_id="user123"
        )
        
        assert result.success is False


# =============================================================================
# 并发测试
# =============================================================================

class TestConcurrency:
    """并发测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_sends(self, context_sender):
        """测试并发发送"""
        tasks = [
            context_sender.send(f"消息{i}", f"user{i}", umo="fakeid:GroupMessage:test_group")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert all(r.success for r in results)
        assert len(results) == 10


if __name__ == "__main__":
    import asyncio
    pytest.main([__file__, "-v"])

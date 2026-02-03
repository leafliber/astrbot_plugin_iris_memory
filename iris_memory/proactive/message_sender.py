"""
消息发送器
通过AstrBot主动发送消息到会话
"""
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

from iris_memory.utils.logger import get_logger

logger = get_logger("message_sender")


@dataclass
class SendResult:
    """发送结果"""
    success: bool
    message_id: Optional[str]
    error: Optional[str]


class MessageSender:
    """消息发送器"""
    
    def __init__(self, astrbot_context=None):
        self.astrbot_context = astrbot_context
        self.send_method = None
        self._detect_send_method()
    
    def _detect_send_method(self):
        """检测可用的发送方法"""
        if not self.astrbot_context:
            return
        
        # 方法1: 通过 context 的 send_message
        if hasattr(self.astrbot_context, 'send_message'):
            self.send_method = "context_send"
        # 方法2: 通过 platform 对象
        elif hasattr(self.astrbot_context, 'platform'):
            platform = self.astrbot_context.platform
            if hasattr(platform, 'send_private_msg') or hasattr(platform, 'send_group_msg'):
                self.send_method = "platform_send"
        # 方法3: 通过 message_service
        elif hasattr(self.astrbot_context, 'message_service'):
            self.send_method = "service_send"
        # 方法4: 通过 event 对象
        elif hasattr(self.astrbot_context, '_event'):
            self.send_method = "event_send"
        
        logger.info(f"Message sender method: {self.send_method}")
    
    async def send(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str] = None,
        session_info: Optional[Dict] = None
    ) -> SendResult:
        """发送消息"""
        if not self.send_method:
            return SendResult(
                success=False,
                message_id=None,
                error="No send method available"
            )
        
        try:
            if self.send_method == "context_send":
                return await self._send_via_context(content, user_id, group_id)
            elif self.send_method == "platform_send":
                return await self._send_via_platform(content, user_id, group_id)
            elif self.send_method == "service_send":
                return await self._send_via_service(content, user_id, group_id)
            elif self.send_method == "event_send":
                return await self._send_via_event(content, user_id, group_id)
            else:
                return SendResult(
                    success=False,
                    message_id=None,
                    error=f"Unknown send method: {self.send_method}"
                )
                
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return SendResult(
                success=False,
                message_id=None,
                error=str(e)
            )
    
    async def _send_via_context(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str]
    ) -> SendResult:
        """通过 context 发送"""
        try:
            target = {
                "user_id": user_id,
                "group_id": group_id
            }
            
            result = await self.astrbot_context.send_message(
                target=target,
                message=content
            )
            
            return SendResult(
                success=True,
                message_id=str(result) if result else None,
                error=None
            )
            
        except Exception as e:
            return SendResult(
                success=False,
                message_id=None,
                error=str(e)
            )
    
    async def _send_via_platform(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str]
    ) -> SendResult:
        """通过 platform 发送"""
        try:
            platform = self.astrbot_context.platform
            
            message = {
                "type": "text",
                "content": content
            }
            
            if group_id:
                result = await platform.send_group_msg(
                    group_id=group_id,
                    message=message
                )
            else:
                result = await platform.send_private_msg(
                    user_id=user_id,
                    message=message
                )
            
            return SendResult(
                success=True,
                message_id=str(result) if result else None,
                error=None
            )
            
        except Exception as e:
            return SendResult(
                success=False,
                message_id=None,
                error=str(e)
            )
    
    async def _send_via_service(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str]
    ) -> SendResult:
        """通过 message_service 发送"""
        try:
            service = self.astrbot_context.message_service
            
            result = await service.send(
                user_id=user_id,
                group_id=group_id,
                content=content
            )
            
            return SendResult(
                success=True,
                message_id=str(result) if result else None,
                error=None
            )
            
        except Exception as e:
            return SendResult(
                success=False,
                message_id=None,
                error=str(e)
            )
    
    async def _send_via_event(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str]
    ) -> SendResult:
        """通过 event 发送（备用方案）"""
        try:
            # 尝试使用注册的回调
            if hasattr(self.astrbot_context, '_send_callback'):
                result = await self.astrbot_context._send_callback(
                    content=content,
                    user_id=user_id,
                    group_id=group_id
                )
                return SendResult(
                    success=result,
                    message_id=None,
                    error=None if result else "Callback returned False"
                )
            
            return SendResult(
                success=False,
                message_id=None,
                error="No event send method available"
            )
            
        except Exception as e:
            return SendResult(
                success=False,
                message_id=None,
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.send_method is not None

"""
消息发送器
通过AstrBot主动发送消息到会话
"""
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
        """检测可用的发送方法
        
        优先检测顺序（基于 AstrBot 实际 API）：
        1. provider 对象（AstrBot 标准接口，v4.14.4+ 推荐）
        2. platform 对象（向下兼容）
        3. context.send_message（备用方案）
        4. message_service（特殊情况）
        5. event 对象（最后备用）
        """
        if not self.astrbot_context:
            logger.warning("No AstrBot context provided")
            return
        
        ctx = self.astrbot_context
        
        # 方法1: 通过 provider 对象（AstrBot v4.14.4+ 标准接口）
        if hasattr(ctx, 'provider') and ctx.provider:
            provider = ctx.provider
            if hasattr(provider, 'send_private_msg') and hasattr(provider, 'send_group_msg'):
                self.send_method = "provider_send"
                logger.info(f"Message sender method: provider_send (AstrBot v4.14+ standard)")
                return
        
        # 方法2: 通过 platform 对象（向下兼容）
        if hasattr(ctx, 'platform') and ctx.platform:
            platform = ctx.platform
            if hasattr(platform, 'send_private_msg') and hasattr(platform, 'send_group_msg'):
                self.send_method = "platform_send"
                logger.info(f"Message sender method: platform_send (legacy compatibility)")
                return
        
        # 方法3: 通过 send_message 方法（备用）
        if hasattr(ctx, 'send_message') and callable(getattr(ctx, 'send_message')):
            self.send_method = "context_send"
            logger.info(f"Message sender method: context_send (backup)")
            return
        
        # 方法4: 通过 message_service
        if hasattr(ctx, 'message_service') and ctx.message_service:
            self.send_method = "service_send"
            logger.info(f"Message sender method: service_send")
            return
        
        # 方法5: 通过 event 对象（最后备用）
        if hasattr(ctx, '_event'):
            self.send_method = "event_send"
            logger.info(f"Message sender method: event_send (last resort)")
            return
        
        logger.warning("No valid send method detected. Proactive reply will be disabled.")
        logger.debug(f"Available context attributes: {dir(ctx) if ctx else 'None'}")
    
    async def send(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str] = None,
        session_info: Optional[Dict] = None,
        umo: str = ""
    ) -> SendResult:
        """发送消息"""
        if not self.send_method:
            return SendResult(
                success=False,
                message_id=None,
                error="No send method available"
            )
        
        try:
            if self.send_method == "provider_send":
                return await self._send_via_provider(content, user_id, group_id)
            elif self.send_method == "platform_send":
                return await self._send_via_platform(content, user_id, group_id)
            elif self.send_method == "service_send":
                return await self._send_via_service(content, user_id, group_id)
            elif self.send_method == "context_send":
                return await self._send_via_context(content, user_id, group_id, umo=umo)
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
    
    async def _send_via_provider(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str]
    ) -> SendResult:
        """通过 AstrBot provider 对象发送（标准接口）"""
        try:
            provider = self.astrbot_context.provider
            
            # 构建标准消息格式
            from astrbot.api.message_components import Plain
            message_chain = [Plain(content)]
            
            if group_id:
                # 群聊消息
                result = await provider.send_group_msg(
                    group_id=group_id,
                    message=message_chain
                )
            else:
                # 私聊消息
                result = await provider.send_private_msg(
                    user_id=user_id,
                    message=message_chain
                )
            
            logger.debug(f"Message sent via provider to {'group' if group_id else 'user'} {group_id or user_id}")
            
            return SendResult(
                success=True,
                message_id=str(result) if result else None,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Provider send failed: {e}")
            return SendResult(
                success=False,
                message_id=None,
                error=str(e)
            )
    
    async def _send_via_context(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str],
        umo: str = ""
    ) -> SendResult:
        """通过 Context.send_message(session, message_chain) 发送
        
        AstrBot API 签名:
            Context.send_message(session: str | MessageSession, message_chain: MessageChain) -> bool
        
        session 可以是 unified_msg_origin 字符串，格式为 "platform_id:message_type:session_id"
        """
        if not umo:
            return SendResult(
                success=False,
                message_id=None,
                error="Cannot send via context without unified_msg_origin (session)"
            )
        
        try:
            from astrbot.api.message_components import Plain
            message_chain = [Plain(content)]
            
            result = await self.astrbot_context.send_message(
                umo,           # session (unified_msg_origin)
                message_chain  # message_chain
            )
            
            return SendResult(
                success=bool(result),
                message_id=None,
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

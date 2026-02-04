"""
主动回复管理器
协调检测、生成、发送整个流程
"""
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from iris_memory.utils.logger import get_logger
from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDetector, ProactiveReplyDecision
)
from iris_memory.proactive.reply_generator import (
    ProactiveReplyGenerator, GeneratedReply
)
from iris_memory.proactive.message_sender import MessageSender

logger = get_logger("proactive_manager")


@dataclass
class ProactiveReplyTask:
    """主动回复任务"""
    messages: List[str]
    user_id: str
    group_id: Optional[str]
    decision: ProactiveReplyDecision
    context: Dict[str, Any]
    umo: str = ""  # 新增：unified_msg_origin


class ProactiveReplyManager:
    """主动回复管理器"""
    
    def __init__(
        self,
        astrbot_context=None,
        reply_detector: Optional[ProactiveReplyDetector] = None,
        reply_generator: Optional[ProactiveReplyGenerator] = None,
        config: Optional[Dict] = None
    ):
        self.config = config or {}
        self.reply_detector = reply_detector
        self.reply_generator = reply_generator
        self.message_sender = None
        
        # 配置
        self.enabled = self.config.get("enable_proactive_reply", True)
        self.cooldown_seconds = self.config.get("reply_cooldown", 60)
        self.max_daily_replies = self.config.get("max_daily_replies", 20)
        
        # 群聊白名单（空列表表示允许所有群聊）
        self.group_whitelist = self.config.get("group_whitelist", [])
        if isinstance(self.group_whitelist, str):
            self.group_whitelist = [self.group_whitelist] if self.group_whitelist else []
        
        # 状态跟踪
        self.last_reply_time: Dict[str, float] = {}
        self.daily_reply_count: Dict[str, int] = {}
        self.pending_tasks: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # 统计
        self.stats = {
            "replies_sent": 0,
            "replies_skipped": 0,
            "replies_failed": 0
        }
    
    async def initialize(self):
        """初始化"""
        if not self.enabled:
            logger.info("Proactive reply is disabled")
            return
        
        # 初始化发送器
        context = None
        if self.reply_generator:
            context = self.reply_generator.astrbot_context
        
        self.message_sender = MessageSender(context)
        
        if not self.message_sender.is_available():
            logger.warning("Message sender not available, proactive reply disabled")
            self.enabled = False
            return
        
        # 启动处理循环
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_loop())
        
        logger.info("Proactive reply manager initialized")
    
    async def stop(self):
        """停止"""
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # 处理剩余的任务
        while not self.pending_tasks.empty():
            try:
                task = self.pending_tasks.get_nowait()
                await self._process_task(task)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error processing pending task during shutdown: {e}")
        
        logger.info("Proactive reply manager stopped")
    
    async def handle_batch(
        self,
        messages: List[str],
        user_id: str,
        group_id: Optional[str] = None,
        context: Optional[Dict] = None,
        umo: str = ""
    ):
        """处理批量消息，判断是否需要主动回复"""
        if not self.enabled or not messages:
            return
        
        # 检查冷却时间
        session_key = f"{user_id}:{group_id or 'private'}"
        if self._is_in_cooldown(session_key):
            logger.debug(f"Proactive reply in cooldown for {session_key}")
            return
        
        # 检查每日限制
        if self._is_daily_limit_reached(user_id):
            logger.debug(f"Daily proactive reply limit reached for {user_id}")
            return
        
        # 检查群聊白名单
        if group_id and self.group_whitelist:
            if str(group_id) not in self.group_whitelist:
                logger.debug(f"Group {group_id} not in proactive reply whitelist, skipping")
                return
        
        # 使用检测器分析
        if not self.reply_detector:
            return
        
        try:
            decision = await self.reply_detector.analyze(
                messages=messages,
                user_id=user_id,
                group_id=group_id,
                context=context
            )
            
            if decision.should_reply:
                # 创建任务
                task = ProactiveReplyTask(
                    messages=messages,
                    user_id=user_id,
                    group_id=group_id,
                    decision=decision,
                    context=context or {},
                    umo=umo
                )
                
                # 加入队列
                await self.pending_tasks.put(task)
                
                # 更新冷却时间
                self.last_reply_time[session_key] = asyncio.get_event_loop().time()
                
                logger.info(f"Proactive reply queued for {session_key}, "
                           f"urgency: {decision.urgency.value}")
            else:
                self.stats["replies_skipped"] += 1
                
        except Exception as e:
            logger.error(f"Error in proactive reply detection: {e}")
    
    async def _process_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 获取任务
                task = await asyncio.wait_for(
                    self.pending_tasks.get(),
                    timeout=1.0
                )
                
                # 处理任务
                await self._process_task(task)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in proactive reply process loop: {e}")
    
    async def _process_task(self, task: ProactiveReplyTask):
        """处理回复任务"""
        try:
            # 延迟发送（根据紧急度）
            delay = task.decision.suggested_delay
            if delay > 0:
                await asyncio.sleep(delay)
            
            # 生成回复
            if not self.reply_generator:
                return
            
            reply = await self.reply_generator.generate_reply(
                messages=task.messages,
                user_id=task.user_id,
                group_id=task.group_id,
                reply_context=task.decision.reply_context,
                emotional_state=task.context.get("emotional_state"),
                umo=task.umo
            )
            
            if not reply or not reply.content:
                logger.warning(f"Failed to generate reply for {task.user_id}")
                self.stats["replies_failed"] += 1
                return
            
            # 发送回复
            result = await self.message_sender.send(
                content=reply.content,
                user_id=task.user_id,
                group_id=task.group_id,
                session_info=task.context.get("session_info")
            )
            
            if result.success:
                self.stats["replies_sent"] += 1
                
                # 更新每日计数
                self.daily_reply_count[task.user_id] = \
                    self.daily_reply_count.get(task.user_id, 0) + 1
                
                logger.info(f"Proactive reply sent to {task.user_id}: "
                           f"{reply.content[:50]}...")
            else:
                self.stats["replies_failed"] += 1
                logger.error(f"Failed to send proactive reply: {result.error}")
                
        except Exception as e:
            logger.error(f"Error processing proactive reply task: {e}")
            self.stats["replies_failed"] += 1
    
    def _is_in_cooldown(self, session_key: str) -> bool:
        """检查是否在冷却中"""
        if session_key not in self.last_reply_time:
            return False
        
        elapsed = asyncio.get_event_loop().time() - self.last_reply_time[session_key]
        return elapsed < self.cooldown_seconds
    
    def _is_daily_limit_reached(self, user_id: str) -> bool:
        """检查是否达到每日限制"""
        count = self.daily_reply_count.get(user_id, 0)
        return count >= self.max_daily_replies
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "pending_tasks": self.pending_tasks.qsize(),
            "last_reply_times": len(self.last_reply_time),
            "daily_counts": self.daily_reply_count.copy()
        }
    
    def reset_daily_counts(self):
        """重置每日计数"""
        self.daily_reply_count.clear()
        logger.info("Daily proactive reply counts reset")

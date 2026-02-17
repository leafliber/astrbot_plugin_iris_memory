"""
主动回复管理器
协调检测、事件注入整个流程

重构后架构：
不再自行调用 LLM 生成回复和直接发送，
而是构造合成事件注入 AstrBot 事件队列，
让主动回复经过完整的 Pipeline 处理流程：
  人格注入 → 插件 Hook（记忆检索等）→ LLM 生成 → 结果装饰 → 发送
"""
import asyncio
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass

from iris_memory.utils.logger import get_logger
from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDetector, ProactiveReplyDecision
)
from iris_memory.proactive.proactive_event import ProactiveMessageEvent

if TYPE_CHECKING:
    from iris_memory.core.config_manager import ConfigManager

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
    """主动回复管理器
    
    重构后不再持有 ReplyGenerator / MessageSender，
    改为通过事件队列注入合成事件来触发完整 Pipeline。
    """
    
    def __init__(
        self,
        astrbot_context=None,
        reply_detector: Optional[ProactiveReplyDetector] = None,
        reply_generator=None,
        event_queue: Optional[asyncio.Queue] = None,
        config: Optional[Dict] = None,
        config_manager: Optional['ConfigManager'] = None
    ):
        self.config = config or {}
        self.reply_detector = reply_detector
        self.reply_generator = reply_generator
        self.astrbot_context = astrbot_context
        self.event_queue = event_queue
        self._config_manager = config_manager
        
        # 配置（默认值，会被动态配置覆盖）
        self.enabled = self.config.get("enable_proactive_reply", True)
        self._default_cooldown = self.config.get("reply_cooldown", 60)
        self._default_max_daily = self.config.get("max_daily_replies", 20)
        
        # 群聊白名单（静态配置，空列表表示允许所有群聊）
        self.group_whitelist = self.config.get("group_whitelist", [])
        if isinstance(self.group_whitelist, str):
            self.group_whitelist = [self.group_whitelist] if self.group_whitelist else []
        elif not isinstance(self.group_whitelist, list):
            self.group_whitelist = []
        self.group_whitelist = [str(group_id) for group_id in self.group_whitelist if group_id]
        
        # 群聊白名单模式（开启后需管理员用指令控制各群聊的主动回复开关）
        self.group_whitelist_mode = self.config.get("group_whitelist_mode", False)
        # 动态群聊白名单（通过指令管理，仅在 group_whitelist_mode 开启时生效）
        dynamic_whitelist = self.config.get("dynamic_whitelist", [])
        if isinstance(dynamic_whitelist, str):
            dynamic_whitelist = [dynamic_whitelist] if dynamic_whitelist else []
        elif not isinstance(dynamic_whitelist, list):
            dynamic_whitelist = []
        self._dynamic_whitelist: set = set(str(group_id) for group_id in dynamic_whitelist if group_id)
        
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
        
        # 检查事件队列是否可用
        if not self.event_queue:
            # 尝试从 context 获取
            if self.astrbot_context and hasattr(self.astrbot_context, '_event_queue'):
                self.event_queue = self.astrbot_context._event_queue
            if not self.event_queue:
                logger.warning("Event queue not available, proactive reply disabled")
                self.enabled = False
                return
        
        # 启动处理循环
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_loop())
        
        logger.info("Proactive reply manager initialized (event queue mode)")
    
    async def stop(self):
        """停止（热更新友好）"""
        logger.info("[Hot-Reload] Stopping ProactiveReplyManager...")
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error cancelling processing task: {e}")
            self.processing_task = None
        
        # 处理剩余的任务
        while not self.pending_tasks.empty():
            try:
                task = self.pending_tasks.get_nowait()
                await self._process_task(task, skip_delay=True)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error processing pending task during shutdown: {e}")
        
        logger.info("[Hot-Reload] ProactiveReplyManager stopped")
    
    def _get_cooldown_seconds(self, group_id: Optional[str] = None) -> int:
        """获取冷却时间"""
        if self._config_manager:
            return self._config_manager.get_cooldown_seconds(group_id)
        return self._default_cooldown
    
    def _get_max_daily_replies(self, group_id: Optional[str] = None) -> int:
        """获取每日最大回复数"""
        if self._config_manager:
            return self._config_manager.get_max_daily_replies(group_id)
        return self._default_max_daily
    
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
        if self._is_in_cooldown(session_key, group_id):
            logger.debug(f"Proactive reply in cooldown for {session_key}")
            return
        
        # 检查每日限制
        if self._is_daily_limit_reached(user_id, group_id):
            logger.debug(f"Daily proactive reply limit reached for {user_id}")
            return
        
        # 检查群聊白名单
        if group_id:
            if not self._is_group_allowed(group_id):
                logger.debug(f"Group {group_id} not allowed for proactive reply, skipping")
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
    
    async def _process_task(self, task: ProactiveReplyTask, skip_delay: bool = False):
        """处理回复任务：构造合成事件并注入事件队列"""
        try:
            # 延迟发送（根据紧急度）
            delay = task.decision.suggested_delay
            if delay > 0 and not skip_delay:
                await asyncio.sleep(delay)
            
            # 检查事件队列和 context
            if not self.event_queue or not self.astrbot_context:
                logger.warning("Event queue or context not available, skip proactive reply")
                self.stats["replies_failed"] += 1
                return
            
            # 构建触发提示（LLM 的 prompt，不是最终回复）
            trigger_prompt = self._build_trigger_prompt(task)
            
            if not trigger_prompt:
                logger.warning(f"Failed to build trigger prompt for {task.user_id}")
                self.stats["replies_failed"] += 1
                return
            
            # 构建主动回复上下文（附加到合成事件的 extras）
            # 获取发送者名称（从上下文中获取）
            sender_name = task.context.get("sender_name", "")
            
            # 准备近期消息数据（供 _build_proactive_directive 使用）
            recent_messages = []
            for msg in task.messages[-5:]:
                recent_messages.append({
                    "sender_name": sender_name or task.user_id,
                    "content": msg[:200]  # 截断过长消息
                })
            
            # 提取情感摘要
            reply_context = task.decision.reply_context or {}
            emotion_data = reply_context.get("emotion", {})
            emotion_summary = ""
            if emotion_data:
                primary = emotion_data.get("primary", "")
                intensity = emotion_data.get("intensity", 0)
                if primary:
                    emotion_summary = f"{primary}（强度 {intensity:.1f}）"
            
            proactive_context = {
                "reason": task.decision.reason,
                "urgency": task.decision.urgency.value,
                "reply_context": reply_context,
                "message_count": len(task.messages),
                "user_id": task.user_id,
                "group_id": task.group_id,
                "recent_messages": recent_messages,
                "emotion_summary": emotion_summary,
                "target_user": sender_name or task.user_id,
            }
            
            # 构造合成事件
            proactive_event = ProactiveMessageEvent(
                context=self.astrbot_context,
                umo=task.umo,
                trigger_prompt=trigger_prompt,
                user_id=task.user_id,
                sender_name=sender_name,
                group_id=task.group_id,
                proactive_context=proactive_context,
            )
            
            # 注入事件队列
            self.event_queue.put_nowait(proactive_event)
            
            self.stats["replies_sent"] += 1
            
            # 更新每日计数
            count_key = f"{task.user_id}:{task.group_id or 'private'}"
            self.daily_reply_count[count_key] = \
                self.daily_reply_count.get(count_key, 0) + 1
            
            logger.info(
                f"Proactive reply event dispatched for {task.user_id}, "
                f"urgency: {task.decision.urgency.value}, "
                f"reason: {task.decision.reason}"
            )
                
        except Exception as e:
            logger.error(f"Error processing proactive reply task: {e}")
            self.stats["replies_failed"] += 1
    
    def _build_trigger_prompt(self, task: ProactiveReplyTask) -> str:
        """构建触发提示词
        
        这段文字会作为合成事件的 message_str，经过 on_all_messages 后
        变成 event.request_llm() 的 prompt，再经过 build_main_agent
        的 _decorate_llm_request（注入人格、技能等），最终由 LLM 生成回复。
        
        关键：不在此处生成最终回复，只提供触发意图和背景上下文。
        """
        reply_context = task.decision.reply_context or {}
        reason = reply_context.get("reason", task.decision.reason)
        
        # 最近用户消息摘要
        recent_messages = task.messages[-5:] if task.messages else []
        messages_summary = "\n".join(
            f"- {msg[:100]}" for msg in recent_messages
        )
        
        # 情感信息
        emotion_info = ""
        emotion_data = reply_context.get("emotion", {})
        if emotion_data:
            primary = emotion_data.get("primary", "")
            intensity = emotion_data.get("intensity", 0)
            if primary:
                emotion_info = f"\n用户当前情绪：{primary}（强度 {intensity:.1f}）"
        
        prompt = (
            f"你现在要主动回复这位用户。\n"
            f"回复原因：{reason}\n"
            f"用户最近的消息：\n{messages_summary}"
            f"{emotion_info}\n\n"
            f"请根据你的人格和与这位用户的记忆，生成一条自然、简短的主动消息。"
            f"像朋友一样自然地接话或关心对方，不要说明你是在主动回复。"
        )
        
        return prompt
    
    def _is_in_cooldown(self, session_key: str, group_id: Optional[str] = None) -> bool:
        """检查是否在冷却中"""
        if session_key not in self.last_reply_time:
            return False
        
        elapsed = asyncio.get_running_loop().time() - self.last_reply_time[session_key]
        return elapsed < self._get_cooldown_seconds(group_id)
    
    def _is_daily_limit_reached(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """检查是否达到每日限制"""
        count_key = f"{user_id}:{group_id or 'private'}"
        count = self.daily_reply_count.get(count_key, 0)
        return count >= self._get_max_daily_replies(group_id)
    
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
    
    # ========== 群聊白名单管理 ==========
    
    def _is_group_allowed(self, group_id: str) -> bool:
        """检查群聊是否允许主动回复
        
        判断逻辑：
        1. 如果开启了群聊白名单模式，只允许在动态白名单中的群聊
        2. 如果没有开启白名单模式，检查静态白名单（空列表表示允许所有）
        """
        group_id_str = str(group_id)
        
        if self.group_whitelist_mode:
            # 白名单模式：仅允许动态白名单中的群聊
            return group_id_str in self._dynamic_whitelist
        
        # 非白名单模式：检查静态白名单（空列表表示允许所有）
        if self.group_whitelist:
            return group_id_str in self.group_whitelist
        return True
    
    def add_group_to_whitelist(self, group_id: str) -> bool:
        """将群聊加入动态白名单
        
        Returns:
            bool: 是否成功添加（如果已存在则返回false）
        """
        group_id_str = str(group_id)
        if group_id_str in self._dynamic_whitelist:
            return False
        self._dynamic_whitelist.add(group_id_str)
        logger.info(f"Group {group_id} added to proactive reply whitelist")
        return True
    
    def remove_group_from_whitelist(self, group_id: str) -> bool:
        """将群聊从动态白名单移除
        
        Returns:
            bool: 是否成功移除（如果不存在则返回false）
        """
        group_id_str = str(group_id)
        if group_id_str not in self._dynamic_whitelist:
            return False
        self._dynamic_whitelist.discard(group_id_str)
        logger.info(f"Group {group_id} removed from proactive reply whitelist")
        return True
    
    def is_group_in_whitelist(self, group_id: str) -> bool:
        """检查群聊是否在动态白名单中"""
        return str(group_id) in self._dynamic_whitelist
    
    def get_whitelist(self) -> list:
        """获取动态白名单列表"""
        return sorted(self._dynamic_whitelist)
    
    def serialize_whitelist(self) -> list:
        """序列化动态白名单（用于KV存储）"""
        return sorted(self._dynamic_whitelist)
    
    def deserialize_whitelist(self, data: list) -> None:
        """反序列化动态白名单（从 KV 存储加载）"""
        if isinstance(data, list):
            self._dynamic_whitelist = set(str(g) for g in data)
            logger.info(f"Loaded {len(self._dynamic_whitelist)} groups to proactive reply whitelist")

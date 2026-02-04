"""
批量消息处理器 - 支持主动回复
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

from iris_memory.utils.logger import get_logger
from iris_memory.capture.capture_engine import MemoryCaptureEngine
from iris_memory.models.memory import Memory
from iris_memory.processing.llm_processor import LLMMessageProcessor, LLMSummaryResult

if TYPE_CHECKING:
    from iris_memory.proactive.proactive_manager import ProactiveReplyManager

logger = get_logger("batch_processor")


@dataclass
class QueuedMessage:
    """队列中的消息"""
    content: str
    user_id: str
    group_id: Optional[str]
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    umo: str = ""  # 新增：unified_msg_origin


class MessageBatchProcessor:
    """消息批量处理器"""
    
    # 自动保存间隔（秒）
    AUTO_SAVE_INTERVAL = 60
    
    def __init__(
        self,
        capture_engine: MemoryCaptureEngine,
        llm_processor: Optional[LLMMessageProcessor] = None,
        proactive_manager: Optional['ProactiveReplyManager'] = None,
        threshold_count: int = 5,
        threshold_interval: int = 300,
        processing_mode: str = "hybrid",
        use_llm_summary: bool = False,
        summary_prompt: str = None,
        on_save_callback: Optional[callable] = None
    ):
        self.capture_engine = capture_engine
        self.llm_processor = llm_processor
        self.proactive_manager = proactive_manager
        self.threshold_count = threshold_count
        self.threshold_interval = threshold_interval
        self.processing_mode = processing_mode
        self.use_llm_summary = use_llm_summary
        self.summary_prompt = summary_prompt
        self.on_save_callback = on_save_callback  # 保存回调函数
        
        # 消息队列: {session_key: [QueuedMessage]}
        self.message_queues: Dict[str, List[QueuedMessage]] = {}
        self.last_process_time: Dict[str, float] = {}
        
        # 后台任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self.auto_save_task: Optional[asyncio.Task] = None
        self.is_running = False
        self._last_save_time: float = time.time()
        self._dirty: bool = False  # 标记是否有未保存的更改
        
        # 统计
        self.stats = {
            "batches_processed": 0,
            "messages_processed": 0,
            "llm_summaries": 0,
            "local_summaries": 0,
            "auto_saves": 0
        }
    
    async def start(self):
        """启动处理器"""
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info(f"MessageBatchProcessor started (LLM: {self.use_llm_summary}, "
                   f"Proactive: {self.proactive_manager is not None}, "
                   f"AutoSave: {self.AUTO_SAVE_INTERVAL}s)")
    
    async def stop(self):
        """停止处理器"""
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
        
        # 处理剩余消息
        await self._process_all_queues()
        
        # 最终保存
        await self._trigger_save()
        
        logger.info("MessageBatchProcessor stopped")
    
    async def add_message(
        self,
        content: str,
        user_id: str,
        group_id: Optional[str] = None,
        context: Dict = None,
        umo: str = ""
    ) -> bool:
        """添加消息到队列"""
        session_key = f"{user_id}:{group_id or 'private'}"
        
        if session_key not in self.message_queues:
            self.message_queues[session_key] = []
            self.last_process_time[session_key] = time.time()
        
        self.message_queues[session_key].append(QueuedMessage(
            content=content,
            user_id=user_id,
            group_id=group_id,
            context=context or {},
            umo=umo
        ))
        
        # 标记有未保存的更改
        self._dirty = True
        
        logger.debug(f"Message queued for {session_key}, "
                    f"queue size: {len(self.message_queues[session_key])}")
        
        # 检查是否触发处理
        should_process = await self._check_threshold(session_key)
        if should_process:
            await self._process_queue(session_key)
            return True
        
        return False
    
    async def _check_threshold(self, session_key: str) -> bool:
        """检查是否达到处理阈值"""
        queue = self.message_queues.get(session_key, [])
        last_time = self.last_process_time.get(session_key, 0)
        
        # 数量阈值
        if len(queue) >= self.threshold_count:
            return True
        
        # 时间阈值
        if time.time() - last_time >= self.threshold_interval:
            return True
        
        return False
    
    async def _cleanup_loop(self):
        """清理循环 - 定期检查时间阈值"""
        # 根据时间阈值动态调整检查频率
        # 最短1秒，最长60秒
        check_interval = max(1, min(60, self.threshold_interval / 2))
        
        while self.is_running:
            try:
                await asyncio.sleep(check_interval)
                
                current_time = time.time()
                for session_key in list(self.message_queues.keys()):
                    last_time = self.last_process_time.get(session_key, 0)
                    if current_time - last_time >= self.threshold_interval:
                        if self.message_queues[session_key]:
                            await self._process_queue(session_key)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _process_queue(self, session_key: str):
        """处理指定队列"""
        queue = self.message_queues.get(session_key, [])
        if not queue:
            return
        
        self.stats["batches_processed"] += 1
        self.stats["messages_processed"] += len(queue)
        
        logger.info(f"Processing batch for {session_key}, count: {len(queue)}")
        
        try:
            # 1. 首先处理消息捕获
            if self.processing_mode == "summary":
                await self._process_summary_mode(session_key, queue)
            elif self.processing_mode == "filter":
                await self._process_filter_mode(session_key, queue)
            else:  # hybrid
                await self._process_hybrid_mode(session_key, queue)
            
            # 2. 然后触发主动回复判断
            await self._trigger_proactive_reply(queue)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
        finally:
            # 清空队列
            self.message_queues[session_key] = []
            self.last_process_time[session_key] = time.time()
    
    async def _trigger_proactive_reply(self, queue: List[QueuedMessage]):
        """触发主动回复判断"""
        if not self.proactive_manager or not queue:
            return
        
        try:
            first_msg = queue[0]
            messages = [msg.content for msg in queue]
            
            # 计算时间跨度
            time_span = queue[-1].timestamp - queue[0].timestamp if len(queue) > 1 else 0
            
            await self.proactive_manager.handle_batch(
                messages=messages,
                user_id=first_msg.user_id,
                group_id=first_msg.group_id,
                context={
                    "time_span": time_span,
                    "user_persona": first_msg.context.get("user_persona", {}),
                    "emotional_state": first_msg.context.get("emotional_state"),
                    "session_info": {"message_count": len(queue)}
                },
                umo=first_msg.umo
            )
        except Exception as e:
            logger.error(f"Proactive reply trigger failed: {e}")
    
    async def _process_summary_mode(
        self,
        session_key: str,
        queue: List[QueuedMessage]
    ):
        """摘要模式 - 支持LLM摘要"""
        messages = [msg.content for msg in queue]
        
        # 尝试使用LLM生成摘要
        if self.use_llm_summary and self.llm_processor:
            summary_result = await self._generate_llm_summary(queue)
            if summary_result:
                await self._capture_summary_memory(
                    session_key, queue, summary_result.summary,
                    source="llm", metadata={
                        "key_points": summary_result.key_points,
                        "preferences": summary_result.user_preferences
                    }
                )
                return
        
        # 使用本地摘要
        summary = self._generate_local_summary(messages)
        await self._capture_summary_memory(
            session_key, queue, summary,
            source="local", metadata={}
        )
    
    async def _generate_llm_summary(
        self,
        queue: List[QueuedMessage]
    ) -> Optional[LLMSummaryResult]:
        """使用LLM生成摘要"""
        if not self.llm_processor or not self.llm_processor.is_available():
            return None
        
        messages = [msg.content for msg in queue]
        first_msg = queue[0]
        
        context = {
            "user_persona": first_msg.context.get("user_persona", {}),
            "session_info": {
                "message_count": len(queue),
                "time_span": queue[-1].timestamp - queue[0].timestamp if len(queue) > 1 else 0
            }
        }
        
        result = await self.llm_processor.generate_summary(
            messages=messages,
            user_id=first_msg.user_id,
            context=context
        )
        
        if result:
            self.stats["llm_summaries"] += 1
            logger.info(f"LLM summary generated: {result.summary[:50]}...")
        
        return result
    
    def _generate_local_summary(self, messages: List[str]) -> str:
        """生成本地摘要（规则基础）"""
        self.stats["local_summaries"] += 1
        
        # 1. 提取关键句
        key_sentences = []
        value_keywords = ["喜欢", "讨厌", "想要", "计划", "目标", "工作", "家庭"]
        
        for msg in messages:
            sentences = msg.split("。")
            for sent in sentences:
                if len(sent) > 10:
                    score = sum(2 for kw in value_keywords if kw in sent)
                    score += len(sent) / 100
                    key_sentences.append((sent, score))
        
        key_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in key_sentences[:3]]
        
        if top_sentences:
            return "对话要点：" + "；".join(top_sentences)
        
        return "对话记录：" + " | ".join(messages[:2])
    
    async def _capture_summary_memory(
        self,
        session_key: str,
        queue: List[QueuedMessage],
        summary: str,
        source: str,
        metadata: Dict
    ):
        """捕获摘要记忆"""
        if not queue:
            return
        
        first_msg = queue[0]
        
        memory = await self.capture_engine.capture_memory(
            message=summary,
            user_id=first_msg.user_id,
            group_id=first_msg.group_id,
            context={
                "batch_summary": True,
                "source_count": len(queue),
                "summary_source": source,
                "key_points": metadata.get("key_points", []),
                "preferences": metadata.get("preferences", [])
            }
        )
        
        if memory:
            logger.info(f"Created {source} summary memory: {memory.id}")
    
    async def _process_filter_mode(self, session_key: str, queue: List[QueuedMessage]):
        """筛选模式 - 支持LLM辅助筛选"""
        for msg in queue:
            # 使用LLM判断价值
            if (self.use_llm_summary and self.llm_processor and 
                self.llm_processor.is_available()):
                
                llm_result = await self.llm_processor.classify_message(
                    msg.content, msg.context
                )
                
                if llm_result and llm_result.layer == "immediate":
                    await self.capture_engine.capture_memory(
                        message=msg.content,
                        user_id=msg.user_id,
                        group_id=msg.group_id,
                        context={**msg.context, "llm_selected": True}
                    )
                    continue
            
            # 本地规则筛选
            if self._is_high_value_message(msg):
                await self.capture_engine.capture_memory(
                    message=msg.content,
                    user_id=msg.user_id,
                    group_id=msg.group_id,
                    context=msg.context
                )
    
    async def _process_hybrid_mode(self, session_key: str, queue: List[QueuedMessage]):
        """混合模式 - 逐条捕获所有消息，高价值消息额外生成摘要"""
        high_value = []
        normal = []

        # 第一步：分类消息
        for msg in queue:
            is_high = await self._is_high_value_async(msg)
            if is_high:
                high_value.append(msg)
            else:
                normal.append(msg)

        # 第二步：逐条捕获所有消息（关键修复：确保每条消息都被捕获）
        captured_count = 0
        for msg in queue:
            try:
                memory = await self.capture_engine.capture_memory(
                    message=msg.content,
                    user_id=msg.user_id,
                    group_id=msg.group_id,
                    context={**msg.context, "batch_processed": True, "high_value": msg in high_value}
                )
                if memory:
                    captured_count += 1
                    logger.debug(f"Captured memory from batch: {memory.id} (type={memory.type.value})")
            except Exception as e:
                logger.warning(f"Failed to capture memory for message: {e}")

        if captured_count > 0:
            logger.info(f"Batch capture completed: {captured_count}/{len(queue)} messages captured")

        # 第三步：对高价值消息和普通消息分别生成摘要（额外增强）
        all_messages = high_value + normal
        if len(all_messages) >= 2:
            # 尝试生成LLM摘要
            if self.use_llm_summary and self.llm_processor:
                summary_result = await self._generate_llm_summary(all_messages)
                if summary_result:
                    await self._capture_summary_memory(
                        session_key, all_messages, summary_result.summary,
                        "llm", {
                            "key_points": summary_result.key_points,
                            "preferences": summary_result.user_preferences,
                            "message_count": len(all_messages),
                            "high_value_count": len(high_value)
                        }
                    )
                    return

            # 使用本地摘要作为后备
            summary = self._generate_local_summary([m.content for m in all_messages])
            await self._capture_summary_memory(
                session_key, all_messages, summary, "local",
                {"message_count": len(all_messages), "high_value_count": len(high_value)}
            )
    
    async def _is_high_value_async(self, msg: QueuedMessage) -> bool:
        """异步判断高价值消息"""
        if self.use_llm_summary and self.llm_processor:
            result = await self.llm_processor.classify_message(
                msg.content, msg.context
            )
            if result:
                return result.layer == "immediate"
        
        return self._is_high_value_message(msg)
    
    def _is_high_value_message(self, msg: QueuedMessage) -> bool:
        """本地规则判断高价值消息"""
        content = msg.content
        
        # 偏好关键词（喜欢、讨厌等）
        preference_keywords = ["喜欢", "讨厌", "爱", "偏好", " desire", "want", "hate"]
        # 计划关键词（计划、目标、打算等）
        plan_keywords = ["计划", "目标", "打算", "准备", "要", "will", "plan"]
        # 高价值主题词
        high_value_keywords = ["工作", "家庭", "学习", "生活", "健康", "项目", "重要", "关键", "喝", "吃"]
        
        all_keywords = preference_keywords + plan_keywords + high_value_keywords
        
        # 先检查关键词，再检查长度
        if any(kw in content for kw in all_keywords):
            return True
        
        # 或者消息很长（可能包含重要信息）
        if len(content) > 50:
            return True
        
        # 太短的消息可能不重要
        if len(content) < 10:
            return False
        
        return False
    
    async def _process_all_queues(self):
        """处理所有队列（用于关闭时）"""
        for session_key in list(self.message_queues.keys()):
            if self.message_queues[session_key]:
                await self._process_queue(session_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self.stats,
            "queue_sizes": {
                k: len(v) for k, v in self.message_queues.items()
            }
        }
    
    # ========== 队列持久化方法 ==========
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """将对象转换为 JSON 可序列化的格式
        
        Args:
            obj: 任意对象
            
        Returns:
            JSON 可序列化的对象
        """
        from iris_memory.models.emotion_state import EmotionalState
        from datetime import datetime, date
        from collections import deque
        from enum import Enum
        
        if isinstance(obj, EmotionalState):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, deque):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # 对于其他类型，尝试转换为字符串
            try:
                # 检查是否是自定义对象，尝试获取 __dict__
                if hasattr(obj, '__dict__'):
                    return self._make_json_serializable(obj.__dict__)
                elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                    return obj.to_dict()
                else:
                    return str(obj)
            except Exception:
                return str(obj)
    
    async def serialize_queues(self) -> Dict[str, Any]:
        """序列化队列数据用于持久化
        
        Returns:
            Dict[str, Any]: 序列化的队列数据
        """
        from iris_memory.models.emotion_state import EmotionalState
        
        serialized = {
            "queues": {},
            "last_process_time": self.last_process_time.copy(),
            "stats": self.stats.copy()
        }
        
        for session_key, messages in self.message_queues.items():
            serialized["queues"][session_key] = []
            for msg in messages:
                # 处理 context 中的所有可能无法序列化的对象
                context = msg.context.copy() if msg.context else {}
                serializable_context = self._make_json_serializable(context)
                
                # 确保移除任何类引用（保留具体的键名如emotional_state）
                keys_to_remove = [k for k in serializable_context.keys() if k.startswith('_')]
                for key in keys_to_remove:
                    del serializable_context[key]
                    
                serialized["queues"][session_key].append({
                    "content": msg.content,
                    "user_id": msg.user_id,
                    "group_id": msg.group_id,
                    "timestamp": msg.timestamp,
                    "context": serializable_context
                })
        
        return serialized
    
    async def deserialize_queues(self, data: Dict[str, Any]):
        """从持久化数据恢复队列
        
        Args:
            data: 序列化的队列数据
        """
        from iris_memory.models.emotion_state import EmotionalState
        
        if not data:
            return
        
        # 恢复队列
        queues_data = data.get("queues", {})
        for session_key, messages_data in queues_data.items():
            restored_messages = []
            for msg in messages_data:
                context = msg.get("context", {})
                # 将字典转换回 EmotionalState 对象
                if "emotional_state" in context and isinstance(context["emotional_state"], dict):
                    try:
                        context["emotional_state"] = EmotionalState.from_dict(context["emotional_state"])
                    except Exception as e:
                        logger.warning(f"Failed to restore EmotionalState: {e}")
                        context["emotional_state"] = None
                
                restored_messages.append(
                    QueuedMessage(
                        content=msg["content"],
                        user_id=msg["user_id"],
                        group_id=msg["group_id"],
                        timestamp=msg.get("timestamp", time.time()),
                        context=context,
                        umo=msg.get("umo", "")  # 从保存的数据恢复umo
                    )
                )
            self.message_queues[session_key] = restored_messages
        
        # 恢复处理时间
        self.last_process_time = data.get("last_process_time", {})
        
        # 恢复统计
        saved_stats = data.get("stats", {})
        self.stats.update(saved_stats)
        
        total_messages = sum(len(q) for q in self.message_queues.values())
        logger.info(f"Restored batch processor queues: {len(self.message_queues)} sessions, {total_messages} messages")
    
    async def _auto_save_loop(self):
        """自动保存循环 - 定期保存队列状态以防止数据丢失"""
        while self.is_running:
            try:
                await asyncio.sleep(self.AUTO_SAVE_INTERVAL)
                
                # 只有在有未保存的更改时才保存
                if self._dirty:
                    await self._trigger_save()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto save loop error: {e}")
    
    async def _trigger_save(self):
        """触发保存回调"""
        if not self._dirty:
            return
            
        if self.on_save_callback:
            try:
                await self.on_save_callback()
                self._dirty = False
                self._last_save_time = time.time()
                self.stats["auto_saves"] += 1
                logger.debug("Batch processor queues auto-saved")
            except Exception as e:
                logger.error(f"Failed to trigger save callback: {e}")
    
    def set_save_callback(self, callback: callable):
        """设置保存回调函数
        
        Args:
            callback: 异步回调函数，用于持久化队列数据
        """
        self.on_save_callback = callback

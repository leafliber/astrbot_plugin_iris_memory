"""
批量消息处理器 - 支持主动回复

优化措施：
1. 可配置的消息数量阈值触发批量处理
2. 批量消息合并为1个批次，只调用1次LLM
3. 短消息自动合并（连续短消息合并为长消息）
4. 相似消息去重
5. 摘要模式：将多条消息合并为1条摘要记忆

架构：
- 使用组合模式拆分功能模块
- message_merger.py: 消息合并和去重
"""
import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING, Final, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from iris_memory.utils.logger import get_logger
from iris_memory.capture.capture_engine import MemoryCaptureEngine
from iris_memory.capture.message_merger import QueuedMessage, MessageMerger
from iris_memory.models.memory import Memory
from iris_memory.processing.llm_processor import LLMMessageProcessor, LLMSummaryResult
from iris_memory.core.constants import BatchProcessingMode
from iris_memory.core.defaults import DEFAULTS

if TYPE_CHECKING:
    from iris_memory.proactive.proactive_manager import ProactiveReplyManager
    from iris_memory.core.config_manager import ConfigManager

logger = get_logger("batch_processor")


class MessageBatchProcessor:
    """消息批量处理器
    
    核心优化：
    - 阈值：可配置数量的消息触发处理（默认从defaults读取）
    - 合并策略：短消息自动合并，批量只调用1次LLM
    - 摘要模式：将多条消息合并为1条摘要记忆
    """
    
    AUTO_SAVE_INTERVAL: Final[int] = 60
    DEFAULT_THRESHOLD_COUNT: Final[int] = DEFAULTS.message_processing.batch_threshold_count
    DEFAULT_LLM_COOLDOWN: Final[int] = 60
    DEFAULT_SUMMARY_INTERVAL: Final[int] = 300
    
    def __init__(
        self,
        capture_engine: MemoryCaptureEngine,
        llm_processor: Optional[LLMMessageProcessor] = None,
        proactive_manager: Optional['ProactiveReplyManager'] = None,
        threshold_count: int = DEFAULT_THRESHOLD_COUNT,
        threshold_interval: int = 300,
        processing_mode: str = BatchProcessingMode.HYBRID,
        use_llm_summary: bool = False,
        summary_prompt: Optional[str] = None,
        on_save_callback: Optional[Callable[[], Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional['ConfigManager'] = None
    ) -> None:
        self.capture_engine: MemoryCaptureEngine = capture_engine
        self.llm_processor: Optional[LLMMessageProcessor] = llm_processor
        self.proactive_manager: Optional['ProactiveReplyManager'] = proactive_manager
        self.threshold_count: int = threshold_count
        self.threshold_interval: int = threshold_interval
        self.processing_mode: str = processing_mode
        self.use_llm_summary: bool = use_llm_summary
        self.summary_prompt: Optional[str] = summary_prompt
        self.on_save_callback: Optional[Callable[[], Any]] = on_save_callback
        self._config_manager: Optional['ConfigManager'] = config_manager
        
        cfg: Dict[str, Any] = config or {}
        
        self._merger = MessageMerger(
            short_message_threshold=cfg.get("short_message_threshold", 15),
            merge_time_window=cfg.get("merge_time_window", 60),
            max_merge_count=cfg.get("max_merge_count", 5)
        )
        
        self.llm_cooldown_seconds: int = cfg.get("llm_cooldown_seconds", self.DEFAULT_LLM_COOLDOWN)
        self.summary_interval_seconds: int = cfg.get("summary_interval_seconds", self.DEFAULT_SUMMARY_INTERVAL)
        
        self.message_queues: Dict[str, List[QueuedMessage]] = {}
        self.last_process_time: Dict[str, float] = {}
        self._last_llm_call: Dict[str, float] = {}
        self._last_summary_time: Dict[str, float] = {}
        
        self.cleanup_task: Optional[asyncio.Task] = None
        self.auto_save_task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        self._last_save_time: float = time.time()
        self._dirty: bool = False
        
        self.stats: Dict[str, int] = {
            "batches_processed": 0,
            "messages_processed": 0,
            "messages_merged": 0,
            "llm_calls": 0,
            "llm_summaries": 0,
            "local_summaries": 0,
            "auto_saves": 0,
            "llm_calls_skipped": 0,
            "messages_deduped": 0,
        }
    
    async def start(self) -> None:
        """启动处理器"""
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info(
            f"MessageBatchProcessor started (threshold={self.threshold_count}, "
            f"LLM: {self.use_llm_summary})"
        )
    
    async def stop(self) -> None:
        """停止处理器（热更新友好）"""
        logger.info("[Hot-Reload] Stopping MessageBatchProcessor...")
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
        
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
            self.auto_save_task = None
        
        try:
            await self._process_all_queues()
            await self._trigger_save()
        except Exception as e:
            logger.warning(f"[Hot-Reload] Error processing remaining queues during stop: {e}")
        
        logger.info("[Hot-Reload] MessageBatchProcessor stopped")
    
    def _check_llm_cooldown(self, session_key: str) -> bool:
        """检查LLM冷却时间"""
        current_time = time.time()
        last_call = self._last_llm_call.get(session_key, 0)
        return current_time - last_call >= self.llm_cooldown_seconds
    
    def _record_llm_call(self, session_key: str):
        """记录LLM调用时间"""
        self._last_llm_call[session_key] = time.time()
        self.stats["llm_calls"] += 1
    
    def _check_summary_cooldown(self, session_key: str) -> bool:
        """检查摘要生成冷却时间"""
        current_time = time.time()
        last_time = self._last_summary_time.get(session_key, 0)
        return current_time - last_time >= self.summary_interval_seconds
    
    def _record_summary(self, session_key: str):
        """记录摘要生成时间"""
        self._last_summary_time[session_key] = time.time()
    
    async def add_message(
        self,
        content: str,
        user_id: str,
        sender_name: Optional[str] = None,
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
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
            sender_name=sender_name,
            group_id=group_id,
            context=context or {},
            umo=umo
        ))
        
        self._dirty = True
        self.stats["messages_processed"] += 1
        
        logger.debug(
            f"Message queued for {session_key}, "
            f"queue size: {len(self.message_queues[session_key])}"
        )
        
        should_process = await self._check_threshold(session_key)
        if should_process:
            await self._process_queue(session_key)
            return True
        
        return False
    
    def _extract_group_id(self, session_key: str) -> Optional[str]:
        """从 session_key 中提取 group_id"""
        if ":" in session_key:
            parts = session_key.split(":", 1)
            gid = parts[1]
            if gid and gid != "private":
                return gid
        return None
    
    def _get_threshold_count(self, session_key: str) -> int:
        """获取会话的批量处理数量阈值"""
        if self._config_manager:
            group_id = self._extract_group_id(session_key)
            return self._config_manager.get_batch_threshold_count(group_id)
        return self.threshold_count
    
    def _get_threshold_interval(self, session_key: str) -> int:
        """获取会话的批量处理时间间隔"""
        if self._config_manager:
            group_id = self._extract_group_id(session_key)
            return self._config_manager.get_batch_threshold_interval(group_id)
        return self.threshold_interval
    
    async def _check_threshold(self, session_key: str) -> bool:
        """检查是否达到处理阈值"""
        queue = self.message_queues.get(session_key, [])
        last_time = self.last_process_time.get(session_key, 0)
        
        count_threshold = self._get_threshold_count(session_key)
        interval_threshold = self._get_threshold_interval(session_key)
        
        if len(queue) >= count_threshold:
            return True
        
        if time.time() - last_time >= interval_threshold:
            return True
        
        return False
    
    async def _cleanup_loop(self):
        """清理循环"""
        check_interval = max(1, min(60, self.threshold_interval / 2))
        
        while self.is_running:
            try:
                await asyncio.sleep(check_interval)
                
                current_time = time.time()
                for session_key in list(self.message_queues.keys()):
                    interval = self._get_threshold_interval(session_key)
                    last_time = self.last_process_time.get(session_key, 0)
                    if current_time - last_time >= interval:
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
        original_count = len(queue)
        
        logger.info(f"Processing batch for {session_key}, original count: {original_count}")
        
        try:
            queue = self._merger.deduplicate_messages(queue)
            queue = self._merger.merge_short_messages(queue)
            
            merged_count = len(queue)
            logger.info(f"After merge: {merged_count} messages (merged {original_count - merged_count})")
            
            if self.processing_mode == "summary":
                await self._process_summary_mode(session_key, queue)
            elif self.processing_mode == "filter":
                await self._process_filter_mode(session_key, queue)
            else:
                await self._process_hybrid_mode(session_key, queue)
            
            await self._trigger_proactive_reply(queue)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
        finally:
            self.message_queues[session_key] = []
            self.last_process_time[session_key] = time.time()
            
            merger_stats = self._merger.get_stats()
            self.stats["messages_merged"] = merger_stats["messages_merged"]
            self.stats["messages_deduped"] = merger_stats["messages_deduped"]
    
    async def _process_summary_mode(self, session_key: str, queue: List[QueuedMessage]):
        """摘要模式"""
        if len(queue) < 2:
            for msg in queue:
                await self.capture_engine.capture_memory(
                    message=msg.content,
                    user_id=msg.user_id,
                    group_id=msg.group_id,
                    context=msg.context,
                    sender_name=msg.sender_name
                )
            return
        
        can_use_llm = (
            self.use_llm_summary 
            and self.llm_processor 
            and self._check_llm_cooldown(session_key)
        )
        
        if can_use_llm:
            self._record_llm_call(session_key)
            summary_result = await self._generate_llm_summary(queue)
            
            if summary_result:
                await self._capture_summary_memory(
                    session_key, queue, summary_result.summary,
                    source="llm", metadata={
                        "key_points": summary_result.key_points,
                        "preferences": summary_result.user_preferences,
                        "message_count": len(queue),
                        "llm_calls": 1
                    }
                )
                self._record_summary(session_key)
                return
        
        messages = [m.content for m in queue]
        summary = self._generate_local_summary(messages)
        await self._capture_summary_memory(
            session_key, queue, summary,
            source="local", metadata={"message_count": len(queue)}
        )
        self._record_summary(session_key)
    
    async def _process_filter_mode(self, session_key: str, queue: List[QueuedMessage]):
        """筛选模式"""
        can_use_llm = (
            self.use_llm_summary 
            and self.llm_processor 
            and self._check_llm_cooldown(session_key)
        )
        
        if can_use_llm and len(queue) >= 5:
            self._record_llm_call(session_key)
            high_value_indices = await self._batch_classify_with_llm(queue)
            
            for i, msg in enumerate(queue):
                if i in high_value_indices:
                    await self.capture_engine.capture_memory(
                        message=msg.content,
                        user_id=msg.user_id,
                        group_id=msg.group_id,
                        context={**msg.context, "llm_selected": True},
                        sender_name=msg.sender_name
                    )
                elif self._is_high_value_message(msg):
                    await self.capture_engine.capture_memory(
                        message=msg.content,
                        user_id=msg.user_id,
                        group_id=msg.group_id,
                        context=msg.context,
                        sender_name=msg.sender_name
                    )
        else:
            for msg in queue:
                if self._is_high_value_message(msg):
                    await self.capture_engine.capture_memory(
                        message=msg.content,
                        user_id=msg.user_id,
                        group_id=msg.group_id,
                        context=msg.context,
                        sender_name=msg.sender_name
                    )
    
    async def _process_hybrid_mode(self, session_key: str, queue: List[QueuedMessage]):
        """混合模式"""
        can_use_llm = (
            self.use_llm_summary 
            and self.llm_processor 
            and self._check_llm_cooldown(session_key)
        )
        
        high_value_indices = set()
        
        if can_use_llm and len(queue) >= 5:
            self._record_llm_call(session_key)
            high_value_indices = await self._batch_classify_with_llm(queue)
        
        captured_count = 0
        for i, msg in enumerate(queue):
            is_high = i in high_value_indices or self._is_high_value_message(msg)
            
            try:
                memory = await self.capture_engine.capture_memory(
                    message=msg.content,
                    user_id=msg.user_id,
                    group_id=msg.group_id,
                    context={
                        **msg.context, 
                        "batch_processed": True, 
                        "high_value": is_high,
                        "llm_used": i in high_value_indices
                    },
                    sender_name=msg.sender_name
                )
                if memory:
                    captured_count += 1
            except Exception as e:
                logger.warning(f"Failed to capture memory: {e}")
        
        if captured_count > 0:
            logger.info(f"Batch capture: {captured_count}/{len(queue)} messages")
        
        if len(queue) >= 2 and self._check_summary_cooldown(session_key):
            await self._process_summary_mode(session_key, queue)
    
    async def _batch_classify_with_llm(self, queue: List[QueuedMessage]) -> Set[int]:
        """批量分类"""
        if not self.llm_processor or not hasattr(self.llm_processor, '_call_llm'):
            return set()
        
        try:
            messages_text = "\n".join([
                f"[{i}] {msg.content}" for i, msg in enumerate(queue)
            ])
            
            prompt = f"""请分析以下{len(queue)}条消息，判断哪些是高价值消息（需要立即保存）。

消息列表：
{messages_text}

请回复需要立即保存的消息编号（用逗号分隔），如果没有则回复"none"。
高价值消息标准：
- 包含用户偏好（喜欢/讨厌）
- 包含计划/目标
- 包含重要个人信息
- 情感强烈

回复格式：0, 2, 5"""

            response = await self.llm_processor._call_llm(prompt, max_tokens=100)
            
            if not response:
                return set()
            
            response_clean = response.strip().lower()
            if response_clean == "none" or not response_clean:
                return set()
            
            indices: Set[int] = set()
            for num in re.findall(r'\d+', response_clean):
                idx = int(num)
                if 0 <= idx < len(queue):
                    indices.add(idx)
            
            logger.info(f"Batch LLM classified {len(indices)}/{len(queue)} as high value")
            return indices
            
        except Exception as e:
            logger.warning(f"Batch LLM classification failed: {e}")
            return set()
    
    def _is_high_value_message(self, msg: QueuedMessage) -> bool:
        """判断消息是否高价值"""
        content = msg.content
        
        high_value_patterns = [
            r'我(喜欢|讨厌|爱|恨)',
            r'我(想|要|打算|计划)',
            r'记住',
            r'别忘了',
            r'重要',
            r'生日',
            r'电话',
            r'地址',
        ]
        
        for pattern in high_value_patterns:
            if re.search(pattern, content):
                return True
        
        if len(content) > 100:
            return True
        
        return False
    
    async def _generate_llm_summary(self, queue: List[QueuedMessage]) -> Optional[LLMSummaryResult]:
        """使用LLM生成摘要"""
        if not self.llm_processor or not self.llm_processor.is_available():
            return None
        
        messages = [msg.content for msg in queue]
        first_msg = queue[0]
        
        context = {
            "user_persona": first_msg.context.get("user_persona", {}),
        }
        
        return await self.llm_processor.summarize_messages(
            messages=messages,
            context=context,
            custom_prompt=self.summary_prompt
        )
    
    def _generate_local_summary(self, messages: List[str]) -> str:
        """生成本地摘要"""
        if len(messages) == 1:
            return messages[0]
        
        total_len = sum(len(m) for m in messages)
        avg_len = total_len / len(messages)
        
        if avg_len < 20:
            return f"用户连续发送了{len(messages)}条短消息：" + " ".join(messages[:3])
        else:
            return f"用户发送了{len(messages)}条消息，内容涉及：" + messages[0][:50]
    
    async def _capture_summary_memory(
        self,
        session_key: str,
        queue: List[QueuedMessage],
        summary: str,
        source: str,
        metadata: Dict[str, Any]
    ):
        """捕获摘要记忆"""
        first_msg = queue[0]
        
        await self.capture_engine.capture_memory(
            message=summary,
            user_id=first_msg.user_id,
            group_id=first_msg.group_id,
            context={
                **first_msg.context,
                "summary": True,
                "summary_source": source,
                **metadata
            },
            sender_name=first_msg.sender_name
        )
        
        if source == "llm":
            self.stats["llm_summaries"] += 1
        else:
            self.stats["local_summaries"] += 1
    
    async def _trigger_proactive_reply(self, queue: List[QueuedMessage]):
        """触发主动回复检查"""
        if not self.proactive_manager or not queue:
            return
        
        try:
            last_msg = queue[-1]
            await self.proactive_manager.check_and_queue(
                messages=[m.content for m in queue],
                user_id=last_msg.user_id,
                group_id=last_msg.group_id,
                context=last_msg.context,
                umo=last_msg.umo
            )
        except Exception as e:
            logger.warning(f"Proactive reply check failed: {e}")
    
    async def _process_all_queues(self):
        """处理所有队列"""
        for session_key in list(self.message_queues.keys()):
            if self.message_queues[session_key]:
                await self._process_queue(session_key)
    
    async def _auto_save_loop(self):
        """自动保存循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.AUTO_SAVE_INTERVAL)
                
                if self._dirty:
                    await self._trigger_save()
                    self._dirty = False
                    self.stats["auto_saves"] += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto save loop error: {e}")
    
    async def _trigger_save(self):
        """触发保存回调"""
        if self.on_save_callback:
            try:
                result = self.on_save_callback()
                if hasattr(result, '__await__'):
                    await result
            except Exception as e:
                logger.warning(f"Save callback failed: {e}")
    
    async def serialize_queues(self) -> Dict[str, Any]:
        """序列化队列"""
        return {
            "queues": {
                k: [
                    {
                        "content": m.content,
                        "user_id": m.user_id,
                        "sender_name": m.sender_name,
                        "group_id": m.group_id,
                        "timestamp": m.timestamp,
                        "context": m.context,
                        "umo": m.umo,
                        "is_merged": m.is_merged,
                        "original_messages": m.original_messages
                    }
                    for m in v
                ]
                for k, v in self.message_queues.items()
            },
            "last_process_time": self.last_process_time
        }
    
    async def deserialize_queues(self, data: Dict[str, Any]) -> None:
        """反序列化队列"""
        queues = data.get("queues", {})
        for session_key, messages in queues.items():
            self.message_queues[session_key] = [
                QueuedMessage(
                    content=m["content"],
                    user_id=m["user_id"],
                    sender_name=m.get("sender_name"),
                    group_id=m.get("group_id"),
                    timestamp=m.get("timestamp", time.time()),
                    context=m.get("context", {}),
                    umo=m.get("umo", ""),
                    is_merged=m.get("is_merged", False),
                    original_messages=m.get("original_messages", [])
                )
                for m in messages
            ]
        
        self.last_process_time = data.get("last_process_time", {})
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()

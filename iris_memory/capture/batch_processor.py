"""
批量消息处理器 - 支持主动回复

优化措施：
1. 可配置的消息数量阈值触发批量处理
2. 批量消息合并为1个批次，只调用1次LLM
3. 短消息自动合并（连续短消息合并为长消息）
4. 相似消息去重
5. 摘要模式：将多条消息合并为1条摘要记忆
"""
import asyncio
import time
import hashlib
import re
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING, Final, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from iris_memory.utils.logger import get_logger
from iris_memory.capture.capture_engine import MemoryCaptureEngine
from iris_memory.models.memory import Memory
from iris_memory.processing.llm_processor import LLMMessageProcessor, LLMSummaryResult
from iris_memory.core.constants import BatchProcessingMode
from iris_memory.core.defaults import DEFAULTS

if TYPE_CHECKING:
    from iris_memory.proactive.proactive_manager import ProactiveReplyManager

logger = get_logger("batch_processor")


@dataclass
class QueuedMessage:
    """队列中的消息"""
    content: str
    user_id: str
    sender_name: Optional[str]
    group_id: Optional[str]
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    umo: str = ""  # unified_msg_origin
    is_merged: bool = False  # 是否由多条消息合并而成
    original_messages: List[str] = field(default_factory=list)  # 原始消息列表
    
    def __post_init__(self) -> None:
        """初始化后处理"""
        if not isinstance(self.context, dict):
            self.context = {}


class MessageBatchProcessor:
    """消息批量处理器
    
    核心优化：
    - 阈值：可配置数量的消息触发处理（默认从defaults读取）
    - 合并策略：短消息自动合并，批量只调用1次LLM
    - 摘要模式：将多条消息合并为1条摘要记忆
    """
    
    # 类常量
    AUTO_SAVE_INTERVAL: Final[int] = 60  # 自动保存间隔（秒）
    DEFAULT_THRESHOLD_COUNT: Final[int] = DEFAULTS.message_processing.batch_threshold_count  # 默认阈值
    DEFAULT_SHORT_MESSAGE_THRESHOLD: Final[int] = 15  # 短消息长度阈值
    DEFAULT_MERGE_TIME_WINDOW: Final[int] = 60  # 合并时间窗口
    DEFAULT_MAX_MERGE_COUNT: Final[int] = 5  # 最大合并消息数
    DEFAULT_LLM_COOLDOWN: Final[int] = 60  # LLM冷却时间
    DEFAULT_SUMMARY_INTERVAL: Final[int] = 300  # 摘要生成间隔
    DEFAULT_FINGERPRINT_CACHE_SIZE: Final[int] = 1000  # 指纹缓存大小
    DEFAULT_FINGERPRINT_TRIM_SIZE: Final[int] = 500  # 裁剪后大小
    
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
        config: Optional[Dict[str, Any]] = None
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
        
        # 配置合并
        cfg: Dict[str, Any] = config or {}
        self.short_message_threshold: int = cfg.get(
            "short_message_threshold", self.DEFAULT_SHORT_MESSAGE_THRESHOLD
        )
        self.merge_time_window: int = cfg.get(
            "merge_time_window", self.DEFAULT_MERGE_TIME_WINDOW
        )
        self.max_merge_count: int = cfg.get(
            "max_merge_count", self.DEFAULT_MAX_MERGE_COUNT
        )
        self.llm_cooldown_seconds: int = cfg.get(
            "llm_cooldown_seconds", self.DEFAULT_LLM_COOLDOWN
        )
        self.summary_interval_seconds: int = cfg.get(
            "summary_interval_seconds", self.DEFAULT_SUMMARY_INTERVAL
        )
        
        # 状态管理
        self.message_queues: Dict[str, List[QueuedMessage]] = {}
        self.last_process_time: Dict[str, float] = {}
        self._last_llm_call: Dict[str, float] = {}
        self._last_summary_time: Dict[str, float] = {}
        self._processed_fingerprints: Set[str] = set()
        
        # 任务管理
        self.cleanup_task: Optional[asyncio.Task] = None
        self.auto_save_task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        self._last_save_time: float = time.time()
        self._dirty: bool = False
        
        # 统计
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
        """停止处理器"""
        self.is_running = False
        
        # 取消清理任务
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
        
        # 取消自动保存任务
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
            self.auto_save_task = None
        
        # 处理剩余消息
        await self._process_all_queues()
        await self._trigger_save()
        
        logger.info("MessageBatchProcessor stopped")
    
    # ========== 消息合并与去重 ==========
    
    def _get_message_fingerprint(self, content: str) -> str:
        """
        生成消息指纹
        
        Args:
            content: 消息内容
            
        Returns:
            str: 指纹字符串
        """
        simplified = ''.join(c.lower() for c in content if c.isalnum())
        simplified = simplified[:80]
        return hashlib.md5(simplified.encode()).hexdigest()[:12]
    
    def _is_duplicate_message(self, content: str) -> bool:
        """
        检查消息是否重复
        
        Args:
            content: 消息内容
            
        Returns:
            bool: 是否重复
        """
        fingerprint = self._get_message_fingerprint(content)
        
        if fingerprint in self._processed_fingerprints:
            return True
        
        self._processed_fingerprints.add(fingerprint)
        
        # 限制缓存大小
        if len(self._processed_fingerprints) > self.DEFAULT_FINGERPRINT_CACHE_SIZE:
            # 保留后半部分
            self._processed_fingerprints = set(
                list(self._processed_fingerprints)[self.DEFAULT_FINGERPRINT_TRIM_SIZE:]
            )
        
        return False
    
    def _merge_short_messages(self, queue: List[QueuedMessage]) -> List[QueuedMessage]:
        """合并连续短消息
        
        策略：
        1. 连续短消息（<15字符）合并为一条
        2. 合并时间窗口内（60秒）的消息
        3. 最多合并5条
        4. 只有同一用户的消息才合并
        """
        if len(queue) <= 1:
            return queue
        
        merged = []
        current_group: List[QueuedMessage] = []
        
        for msg in queue:
            if not current_group:
                current_group.append(msg)
                continue
            
            last_msg = current_group[-1]
            
            # 判断是否应该合并
            should_merge = (
                len(msg.content) < self.short_message_threshold  # 短消息
                and len(last_msg.content) < self.short_message_threshold * 3  # 前一条也不太長
                and msg.timestamp - last_msg.timestamp < self.merge_time_window  # 时间窗口内
                and msg.user_id == last_msg.user_id  # 同一用户
                and len(current_group) < self.max_merge_count  # 未超过最大合并数
            )
            
            if should_merge:
                current_group.append(msg)
            else:
                # 保存当前组
                if len(current_group) == 1:
                    merged.append(current_group[0])
                else:
                    merged.append(self._combine_message_group(current_group))
                current_group = [msg]
        
        # 处理最后一组
        if current_group:
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                merged.append(self._combine_message_group(current_group))
        
        self.stats["messages_merged"] += len(queue) - len(merged)
        return merged
    
    def _combine_message_group(self, messages: List[QueuedMessage]) -> QueuedMessage:
        """将一组消息合并为一条"""
        if len(messages) == 1:
            return messages[0]
        
        first = messages[0]
        combined_content = " ".join([m.content for m in messages])
        
        return QueuedMessage(
            content=combined_content,
            user_id=first.user_id,
            sender_name=first.sender_name,
            group_id=first.group_id,
            timestamp=first.timestamp,
            context={
                **first.context,
                "merged": True,
                "merge_count": len(messages),
                "time_span": messages[-1].timestamp - first.timestamp
            },
            umo=first.umo,
            is_merged=True,
            original_messages=[m.content for m in messages]
        )
    
    # ========== LLM调用控制 ==========
    
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
    
    # ========== 核心处理流程 ==========
    
    async def add_message(
        self,
        content: str,
        user_id: str,
        sender_name: Optional[str] = None,
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        umo: str = ""
    ) -> bool:
        """
        添加消息到队列
        
        Args:
            content: 消息内容
            user_id: 用户ID
            sender_name: 发送者显示名称
            group_id: 群聊ID
            context: 上下文信息
            umo: 统一消息来源
            
        Returns:
            bool: 是否立即处理
        """
        session_key = f"{user_id}:{group_id or 'private'}"
        
        # 初始化队列
        if session_key not in self.message_queues:
            self.message_queues[session_key] = []
            self.last_process_time[session_key] = time.time()
        
        # 添加消息
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
        
        # 数量阈值检查
        if len(queue) >= self.threshold_count:
            return True
        
        # 时间阈值
        if time.time() - last_time >= self.threshold_interval:
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
                    last_time = self.last_process_time.get(session_key, 0)
                    if current_time - last_time >= self.threshold_interval:
                        if self.message_queues[session_key]:
                            await self._process_queue(session_key)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _process_queue(self, session_key: str):
        """处理指定队列（核心优化：批量消息合并为1次LLM调用）"""
        queue = self.message_queues.get(session_key, [])
        if not queue:
            return
        
        self.stats["batches_processed"] += 1
        original_count = len(queue)
        
        logger.info(f"Processing batch for {session_key}, original count: {original_count}")
        
        try:
            # 步骤1：消息去重
            queue = self._deduplicate_messages(queue)
            
            # 步骤2：合并短消息
            queue = self._merge_short_messages(queue)
            
            merged_count = len(queue)
            logger.info(f"After merge: {merged_count} messages (merged {original_count - merged_count})")
            
            # 步骤3：批量处理
            if self.processing_mode == "summary":
                await self._process_summary_mode(session_key, queue)
            elif self.processing_mode == "filter":
                await self._process_filter_mode(session_key, queue)
            else:  # hybrid
                await self._process_hybrid_mode(session_key, queue)
            
            # 步骤4：触发主动回复
            await self._trigger_proactive_reply(queue)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
        finally:
            self.message_queues[session_key] = []
            self.last_process_time[session_key] = time.time()
    
    def _deduplicate_messages(self, queue: List[QueuedMessage]) -> List[QueuedMessage]:
        """消息去重"""
        unique = []
        for msg in queue:
            if self._is_duplicate_message(msg.content):
                self.stats["messages_deduped"] += 1
                continue
            unique.append(msg)
        return unique
    
    # ========== 处理模式 ==========
    
    async def _process_summary_mode(self, session_key: str, queue: List[QueuedMessage]):
        """摘要模式 - 批量消息合并为1条摘要（只调用1次LLM）"""
        if len(queue) < 2:
            # 消息太少，直接逐条处理
            for msg in queue:
                await self.capture_engine.capture_memory(
                    message=msg.content,
                    user_id=msg.user_id,
                    group_id=msg.group_id,
                    context=msg.context,
                    sender_name=msg.sender_name
                )
            return
        
        # 检查是否可以生成LLM摘要
        can_use_llm = (
            self.use_llm_summary 
            and self.llm_processor 
            and self._check_llm_cooldown(session_key)
        )
        
        if can_use_llm:
            # 批量消息只调用1次LLM生成摘要
            self._record_llm_call(session_key)
            summary_result = await self._generate_llm_summary(queue)
            
            if summary_result:
                await self._capture_summary_memory(
                    session_key, queue, summary_result.summary,
                    source="llm", metadata={
                        "key_points": summary_result.key_points,
                        "preferences": summary_result.user_preferences,
                        "message_count": len(queue),
                        "llm_calls": 1  # 明确记录只调用1次
                    }
                )
                self._record_summary(session_key)
                return
        
        # 使用本地摘要
        messages = [m.content for m in queue]
        summary = self._generate_local_summary(messages)
        await self._capture_summary_memory(
            session_key, queue, summary,
            source="local", metadata={"message_count": len(queue)}
        )
        self._record_summary(session_key)
    
    async def _process_filter_mode(self, session_key: str, queue: List[QueuedMessage]):
        """筛选模式 - 批量消息只调用1次LLM进行批量分类"""
        can_use_llm = (
            self.use_llm_summary 
            and self.llm_processor 
            and self._check_llm_cooldown(session_key)
        )
        
        if can_use_llm and len(queue) >= 5:
            # 批量消息只调用1次LLM进行批量分类
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
            # 本地规则筛选
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
        """混合模式 - 批量消息合并处理，只调用1次LLM"""
        can_use_llm = (
            self.use_llm_summary 
            and self.llm_processor 
            and self._check_llm_cooldown(session_key)
        )
        
        high_value_indices = set()
        
        if can_use_llm and len(queue) >= 5:
            # 批量消息只调用1次LLM进行批量分类
            self._record_llm_call(session_key)
            high_value_indices = await self._batch_classify_with_llm(queue)
        
        # 逐条捕获（使用LLM结果或本地规则）
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
        
        # 生成批量摘要（批量消息只生成1次）
        if len(queue) >= 2 and self._check_summary_cooldown(session_key):
            await self._process_summary_mode(session_key, queue)
    
    async def _batch_classify_with_llm(self, queue: List[QueuedMessage]) -> Set[int]:
        """
        批量分类 - 多条消息只调用1次LLM
        
        Args:
            queue: 消息队列
            
        Returns:
            Set[int]: 高价值消息的索引集合
        """
        if not self.llm_processor or not hasattr(self.llm_processor, '_call_llm'):
            return set()
        
        try:
            # 构建批量分类提示词
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
            
            # 解析响应
            response_clean = response.strip().lower()
            if response_clean == "none" or not response_clean:
                return set()
            
            # 提取数字
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
    
    # ========== 摘要生成 ==========
    
    async def _generate_llm_summary(self, queue: List[QueuedMessage]) -> Optional[LLMSummaryResult]:
        """使用LLM生成摘要（批量消息只调用1次）"""
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
            logger.info(f"LLM summary generated for {len(queue)} messages")
        
        return result
    
    def _generate_local_summary(self, messages: List[str]) -> str:
        """生成本地摘要"""
        self.stats["local_summaries"] += 1
        
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
    
    async def _capture_summary_memory(self, session_key: str, queue: List[QueuedMessage], 
                                     summary: str, source: str, metadata: Dict):
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
                **metadata
            },
            sender_name=first_msg.sender_name
        )
        
        if memory:
            logger.info(f"Created {source} summary memory: {memory.id}")
    
    # ========== 辅助方法 ==========
    
    def _is_high_value_message(self, msg: QueuedMessage) -> bool:
        """本地规则判断高价值消息"""
        content = msg.content
        
        preference_keywords = ["喜欢", "讨厌", "爱", "偏好", " desire", "want", "hate"]
        plan_keywords = ["计划", "目标", "打算", "准备", "要", "will", "plan"]
        high_value_keywords = ["工作", "家庭", "学习", "生活", "健康", "项目", "重要", "关键"]
        
        all_keywords = preference_keywords + plan_keywords + high_value_keywords
        
        if any(kw in content for kw in all_keywords):
            return True
        
        if len(content) > 50:
            return True
        
        if len(content) < 10:
            return False
        
        return False
    
    async def _trigger_proactive_reply(self, queue: List[QueuedMessage]):
        """触发主动回复判断"""
        if not self.proactive_manager or not queue:
            return
        
        try:
            first_msg = queue[0]
            messages = [msg.content for msg in queue]
            
            # 从首条消息的上下文中提取用户画像
            user_persona = {}
            if first_msg.context:
                persona_obj = first_msg.context.get("user_persona")
                if persona_obj is not None:
                    if hasattr(persona_obj, "to_injection_view"):
                        user_persona = persona_obj.to_injection_view()
                    elif isinstance(persona_obj, dict):
                        user_persona = persona_obj
            
            await self.proactive_manager.handle_batch(
                messages=messages,
                user_id=first_msg.user_id,
                group_id=first_msg.group_id,
                context={
                    "time_span": queue[-1].timestamp - queue[0].timestamp if len(queue) > 1 else 0,
                    "message_count": len(queue),
                    "sender_name": first_msg.sender_name or "",
                    "user_persona": user_persona
                },
                umo=first_msg.umo
            )
        except Exception as e:
            logger.error(f"Proactive reply trigger failed: {e}")
    
    async def _process_all_queues(self):
        """处理所有队列"""
        for session_key in list(self.message_queues.keys()):
            if self.message_queues[session_key]:
                await self._process_queue(session_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            **self.stats,
            "queue_sizes": {k: len(v) for k, v in self.message_queues.items()},
            "is_running": self.is_running,
            "total_queued": sum(len(v) for v in self.message_queues.values())
        }
    
    # ========== 持久化方法 ==========
    
    async def serialize_queues(self) -> Dict[str, Any]:
        """序列化队列数据"""
        serialized = {
            "queues": {},
            "last_process_time": self.last_process_time.copy(),
            "stats": self.stats.copy()
        }
        
        for session_key, messages in self.message_queues.items():
            serialized["queues"][session_key] = []
            for msg in messages:
                # 序列化 context，处理 UserPersona 和 EmotionalState 对象
                serialized_context = {}
                if msg.context:
                    for key, value in msg.context.items():
                        if hasattr(value, 'to_dict'):
                            # 如果对象有 to_dict 方法（如 UserPersona, EmotionalState），使用它
                            serialized_context[key] = value.to_dict()
                        else:
                            # 否则直接使用原值
                            serialized_context[key] = value
                
                serialized["queues"][session_key].append({
                    "content": msg.content,
                    "user_id": msg.user_id,
                    "sender_name": msg.sender_name,
                    "group_id": msg.group_id,
                    "timestamp": msg.timestamp,
                    "context": serialized_context,
                    "umo": msg.umo,
                    "is_merged": msg.is_merged,
                    "original_messages": msg.original_messages
                })
        
        return serialized
    
    async def deserialize_queues(self, data: Dict[str, Any]):
        """从持久化数据恢复队列
        
        注意：context 中的 UserPersona 和 EmotionalState 会被反序列化为字典。
        当消息被处理时，会从 memory_service 重新获取最新的对象实例。
        """
        if not data:
            return
        
        queues_data = data.get("queues", {})
        for session_key, messages_data in queues_data.items():
            restored_messages = []
            for msg in messages_data:
                restored_messages.append(
                    QueuedMessage(
                        content=msg["content"],
                        user_id=msg["user_id"],
                        sender_name=msg.get("sender_name"),
                        group_id=msg["group_id"],
                        timestamp=msg.get("timestamp", time.time()),
                        context=msg.get("context", {}),  # 作为字典恢复，处理时会重新获取对象
                        umo=msg.get("umo", ""),
                        is_merged=msg.get("is_merged", False),
                        original_messages=msg.get("original_messages", [])
                    )
                )
            self.message_queues[session_key] = restored_messages
        
        self.last_process_time = data.get("last_process_time", {})
        saved_stats = data.get("stats", {})
        self.stats.update(saved_stats)
        
        total_messages = sum(len(q) for q in self.message_queues.values())
        logger.info(f"Restored batch processor: {len(self.message_queues)} sessions, {total_messages} messages")
    
    async def _auto_save_loop(self):
        """自动保存循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.AUTO_SAVE_INTERVAL)
                if self._dirty:
                    await self._trigger_save()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto save loop error: {e}")
    
    async def _trigger_save(self):
        """触发保存回调"""
        if not self._dirty or not self.on_save_callback:
            return
        
        try:
            await self.on_save_callback()
            self._dirty = False
            self._last_save_time = time.time()
            self.stats["auto_saves"] += 1
        except Exception as e:
            logger.error(f"Failed to trigger save callback: {e}")

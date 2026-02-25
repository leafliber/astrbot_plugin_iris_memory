"""
用户画像批量处理器 — 合并多条消息为单次 LLM 画像提取

通过引入消息队列和双触发器(数量+时间)机制，将多次画像提取合并为单次 LLM 调用，
预计可降低 60%-80% 的画像 LLM 调用量。

架构：
- 生产者: BusinessOperations._update_persona_from_memory() 在记忆捕获后入队
- 消费者: PersonaBatchProcessor 按阈值或定时批量提取
- 降级: 处理器不可用时自动回退到即时提取
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING, Final

from iris_memory.utils.logger import get_logger
from iris_memory.utils.command_utils import SessionKeyBuilder
from iris_memory.analysis.persona.keyword_maps import ExtractionResult

if TYPE_CHECKING:
    from iris_memory.analysis.persona.persona_extractor import PersonaExtractor
    from iris_memory.models.user_persona import UserPersona

logger = get_logger("persona_batch_processor")


# ---------------------------------------------------------------------------
# 队列消息载体
# ---------------------------------------------------------------------------
@dataclass
class PersonaQueuedMessage:
    """待处理的画像提取消息"""

    content: str = ""
    summary: Optional[str] = None
    memory_type: str = ""
    confidence: float = 0.5
    memory_id: Optional[str] = None
    user_id: str = ""
    group_id: Optional[str] = None
    enqueue_time: float = field(default_factory=time.time)

    # 内容自动截断
    MAX_CONTENT_LENGTH: Final[int] = 500

    def __post_init__(self) -> None:
        if self.content and len(self.content) > self.MAX_CONTENT_LENGTH:
            self.content = self.content[: self.MAX_CONTENT_LENGTH]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "summary": self.summary,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "enqueue_time": self.enqueue_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaQueuedMessage":
        return cls(
            content=data.get("content", ""),
            summary=data.get("summary"),
            memory_type=data.get("memory_type", ""),
            confidence=data.get("confidence", 0.5),
            memory_id=data.get("memory_id"),
            user_id=data.get("user_id", ""),
            group_id=data.get("group_id"),
            enqueue_time=data.get("enqueue_time", time.time()),
        )


# ---------------------------------------------------------------------------
# 批量处理器
# ---------------------------------------------------------------------------
class PersonaBatchProcessor:
    """用户画像批量处理器

    核心机制：
    - 数量触发器: 队列消息数达到阈值时立即触发处理
    - 时间触发器: 后台定时任务按固定间隔扫描所有非空队列
    - 会话隔离: 通过 SessionKeyBuilder 构建的复合键隔离不同会话

    与现有组件的关系：
    - 完全复用 PersonaExtractor.extract() 接口
    - 通过回调函数 apply_result_callback 将结果应用到 UserPersona
    - 不改变 PersonaExtractor 和 UserPersona 的任何逻辑
    """

    # 队列容量上限（防止内存膨胀）
    MAX_QUEUE_SIZE: Final[int] = 50

    def __init__(
        self,
        persona_extractor: PersonaExtractor,
        *,
        batch_threshold: int = 5,
        flush_interval: int = 300,
        batch_max_size: int = 10,
        apply_result_callback: Optional[
            Callable[
                [str, str, ExtractionResult, PersonaQueuedMessage],
                Any,
            ]
        ] = None,
    ) -> None:
        """
        初始化画像批量处理器

        Args:
            persona_extractor: 画像提取器实例（复用现有组件）
            batch_threshold: 触发批量处理的消息数量阈值
            flush_interval: 定时刷新的时间间隔（秒）
            batch_max_size: 单次批量处理的最大消息数
            apply_result_callback: 结果应用回调 (user_id, session_key, result, msg)
        """
        self._extractor = persona_extractor
        self._batch_threshold = batch_threshold
        self._flush_interval = flush_interval
        self._batch_max_size = batch_max_size
        self._apply_result_callback = apply_result_callback

        # 会话队列: session_key -> List[PersonaQueuedMessage]
        self._queues: Dict[str, List[PersonaQueuedMessage]] = {}

        # 后台任务
        self._flush_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._lock = asyncio.Lock()

        # 统计
        self._stats = PersonaBatchStats()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动后台定时刷新任务"""
        if self._is_running:
            return
        self._is_running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            f"PersonaBatchProcessor started "
            f"(threshold={self._batch_threshold}, "
            f"interval={self._flush_interval}s, "
            f"max_size={self._batch_max_size})"
        )

    async def stop(self) -> None:
        """停止处理器，处理剩余队列"""
        if not self._is_running:
            return
        logger.info("[Hot-Reload] Stopping PersonaBatchProcessor...")
        self._is_running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # 处理剩余队列
        try:
            await self._flush_all_queues()
        except Exception as e:
            logger.warning(f"Error flushing remaining queues during stop: {e}")

        logger.info(
            f"[Hot-Reload] PersonaBatchProcessor stopped. "
            f"Stats: {self._stats.to_dict()}"
        )

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ------------------------------------------------------------------
    # 消息入队
    # ------------------------------------------------------------------

    async def add_message(
        self,
        content: str,
        user_id: str,
        *,
        group_id: Optional[str] = None,
        summary: Optional[str] = None,
        memory_type: str = "",
        confidence: float = 0.5,
        memory_id: Optional[str] = None,
    ) -> bool:
        """将消息加入画像提取队列

        Args:
            content: 消息内容
            user_id: 用户 ID
            group_id: 群组 ID
            summary: 消息摘要
            memory_type: 记忆类型
            confidence: 置信度
            memory_id: 来源记忆 ID

        Returns:
            True 如果触发了批量处理
        """
        session_key = SessionKeyBuilder.build(user_id, group_id)

        msg = PersonaQueuedMessage(
            content=content,
            summary=summary,
            memory_type=memory_type,
            confidence=confidence,
            memory_id=memory_id,
            user_id=user_id,
            group_id=group_id,
        )

        async with self._lock:
            queue = self._queues.setdefault(session_key, [])

            # 队列容量保护
            if len(queue) >= self.MAX_QUEUE_SIZE:
                logger.warning(
                    f"Queue full for {session_key} "
                    f"({len(queue)}>={self.MAX_QUEUE_SIZE}), "
                    f"forcing flush"
                )
                await self._process_queue(session_key)
                queue = self._queues.setdefault(session_key, [])

            queue.append(msg)
            self._stats.messages_enqueued += 1

            logger.debug(
                f"Persona message queued for {session_key}, "
                f"queue size: {len(queue)}"
            )

            # 数量触发器
            if len(queue) >= self._batch_threshold:
                await self._process_queue(session_key)
                return True

        return False

    # ------------------------------------------------------------------
    # 批量处理核心
    # ------------------------------------------------------------------

    async def _process_queue(self, session_key: str) -> None:
        """处理指定会话的队列

        调用方须在 _lock 内或确保无并发冲突时调用。
        """
        queue = self._queues.pop(session_key, [])
        if not queue:
            return

        # 限制单次处理数量
        if len(queue) > self._batch_max_size:
            # 超出部分放回队列
            self._queues[session_key] = queue[self._batch_max_size:]
            queue = queue[: self._batch_max_size]

        try:
            await self._batch_extract_and_apply(session_key, queue)
        except Exception as e:
            logger.error(
                f"Persona batch processing failed for {session_key}: {e}",
                exc_info=True,
            )

    async def _batch_extract_and_apply(
        self,
        session_key: str,
        messages: List[PersonaQueuedMessage],
    ) -> None:
        """合并消息、调用提取器、应用结果"""
        self._stats.batches_processed += 1
        msg_count = len(messages)

        # 合并消息为格式化文本
        merged_content = self._merge_messages(messages)
        merged_summary = self._merge_summaries(messages)

        logger.info(
            f"Processing persona batch for {session_key}: "
            f"{msg_count} message(s)"
        )

        # 调用现有 PersonaExtractor（复用，不改变接口）
        try:
            result = await self._extractor.extract(
                content=merged_content,
                summary=merged_summary,
            )
            self._stats.llm_calls += 1
        except Exception as e:
            logger.warning(f"Persona extraction failed: {e}")
            self._stats.extraction_errors += 1
            return

        # 检查提取结果是否有效
        if result.confidence <= 0 and not result.interests:
            logger.debug(
                f"Persona extraction returned empty result for {session_key}"
            )
            return

        # 应用结果到每条消息对应的画像
        self._stats.messages_processed += msg_count
        if self._apply_result_callback:
            for msg in messages:
                try:
                    callback_result = self._apply_result_callback(
                        msg.user_id, session_key, result, msg
                    )
                    # 支持同步和异步回调
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                except Exception as e:
                    logger.warning(
                        f"Failed to apply persona result for "
                        f"user={msg.user_id}: {e}"
                    )

    # ------------------------------------------------------------------
    # 消息合并策略
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_messages(messages: List[PersonaQueuedMessage]) -> str:
        """合并多条消息为格式化文本

        每条消息标注序号，保留消息边界信息。
        """
        if len(messages) == 1:
            return messages[0].content

        parts: List[str] = []
        for i, msg in enumerate(messages, 1):
            content = msg.content.strip()
            if content:
                parts.append(f"[{i}] {content}")
        return "\n".join(parts)

    @staticmethod
    def _merge_summaries(messages: List[PersonaQueuedMessage]) -> Optional[str]:
        """合并消息摘要"""
        summaries = [m.summary for m in messages if m.summary]
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]
        return " | ".join(summaries)

    # ------------------------------------------------------------------
    # 定时刷新
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """后台定时扫描所有非空队列"""
        while self._is_running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_all_queues()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Persona flush loop error: {e}", exc_info=True)

    async def _flush_all_queues(self) -> None:
        """处理所有非空队列"""
        async with self._lock:
            non_empty_keys = [
                k for k, v in self._queues.items() if v
            ]
            for key in non_empty_keys:
                await self._process_queue(key)

    # ------------------------------------------------------------------
    # 序列化 / 反序列化（供 KV Store 持久化）
    # ------------------------------------------------------------------

    async def serialize_queues(self) -> Dict[str, Any]:
        """序列化队列状态"""
        async with self._lock:
            return {
                "queues": {
                    k: [m.to_dict() for m in v]
                    for k, v in self._queues.items()
                    if v  # 只序列化非空队列
                },
                "stats": self._stats.to_dict(),
            }

    async def deserialize_and_clear(self, data: Dict[str, Any]) -> None:
        """反序列化队列后立即清空

        重启时加载队列数据但不处理（清空策略），
        避免处理过期的上下文信息。仅恢复统计数据。
        """
        async with self._lock:
            # 恢复统计
            stats_data = data.get("stats", {})
            if stats_data:
                self._stats = PersonaBatchStats.from_dict(stats_data)

            # 记录被丢弃的消息数
            queues_data = data.get("queues", {})
            discarded = sum(len(msgs) for msgs in queues_data.values())
            if discarded > 0:
                logger.info(
                    f"Discarded {discarded} queued persona message(s) "
                    f"from previous session (clear-on-restart policy)"
                )
                self._stats.messages_discarded += discarded

            # 确保队列干净
            self._queues.clear()

    # ------------------------------------------------------------------
    # 统计与健康检查
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.to_dict()
        stats["pending_queues"] = len(
            [k for k, v in self._queues.items() if v]
        )
        stats["total_pending"] = sum(
            len(v) for v in self._queues.values()
        )
        stats["is_running"] = self._is_running
        return stats

    @property
    def pending_count(self) -> int:
        """当前待处理消息总数"""
        return sum(len(v) for v in self._queues.values())


# ---------------------------------------------------------------------------
# 统计数据
# ---------------------------------------------------------------------------
@dataclass
class PersonaBatchStats:
    """批量处理统计"""

    messages_enqueued: int = 0
    messages_processed: int = 0
    messages_discarded: int = 0
    batches_processed: int = 0
    llm_calls: int = 0
    extraction_errors: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "messages_enqueued": self.messages_enqueued,
            "messages_processed": self.messages_processed,
            "messages_discarded": self.messages_discarded,
            "batches_processed": self.batches_processed,
            "llm_calls": self.llm_calls,
            "extraction_errors": self.extraction_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaBatchStats":
        return cls(
            messages_enqueued=data.get("messages_enqueued", 0),
            messages_processed=data.get("messages_processed", 0),
            messages_discarded=data.get("messages_discarded", 0),
            batches_processed=data.get("batches_processed", 0),
            llm_calls=data.get("llm_calls", 0),
            extraction_errors=data.get("extraction_errors", 0),
        )

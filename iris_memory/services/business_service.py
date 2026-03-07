"""
业务操作服务

将业务逻辑从 MemoryService Mixin 中分离为独立的组合式服务类，
通过构造函数显式接收所有依赖，消除隐式耦合。

设计动机：
- 原 BusinessOperations 作为 Mixin 通过 self.xxx 隐式访问 MemoryService 属性
  （如 self.capture_engine, self.retrieval_engine 等），形成循环依赖
- 转为组合模式后，所有依赖通过构造函数显式注入，依赖关系清晰可追踪
- 每个方法的依赖一目了然，便于单独测试
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from datetime import datetime

from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import (
    SessionScope, NumericDefaults, LogTemplates, UNLIMITED_BUDGET
)
from iris_memory.services.context_builder import ContextBuilder

if TYPE_CHECKING:
    from iris_memory.services.modules.storage_module import StorageModule
    from iris_memory.services.modules.analysis_module import AnalysisModule
    from iris_memory.services.modules.llm_enhanced_module import LLMEnhancedModule
    from iris_memory.services.modules.capture_module import CaptureModule
    from iris_memory.services.modules.retrieval_module import RetrievalModule
    from iris_memory.services.modules.kg_module import KnowledgeGraphModule


@dataclass
class BusinessServiceDeps:
    """BusinessService 依赖配置对象

    将相关依赖分组，减少构造函数参数数量，提高可维护性。

    Attributes:
        shared_state: 跨服务共享状态
        cfg: 配置管理器
        core_modules: 核心模块组（storage, analysis, llm_enhanced, capture, retrieval）
        kg: 知识图谱模块
        optional_services: 可选服务组（image_analyzer, member_identity, activity_tracker）
    """
    shared_state: "SharedState"
    cfg: Any
    storage: "StorageModule"
    analysis: "AnalysisModule"
    llm_enhanced: "LLMEnhancedModule"
    capture: "CaptureModule"
    retrieval: "RetrievalModule"
    kg: "KnowledgeGraphModule"
    image_analyzer: Any = None
    member_identity: Any = None
    activity_tracker: Any = None
from iris_memory.core.types import StorageLayer
from iris_memory.utils.command_utils import MessageFilter
from iris_memory.analysis.persona.persona_logger import persona_log
from iris_memory.services.shared_state import SharedState

logger = get_logger("memory_service.business")


class BusinessService:
    """业务操作服务

    通过构造函数注入所有依赖，取代原 Mixin 模式。

    职责：
    1. 记忆捕获、存储、检索
    2. LLM 上下文准备
    3. 图片分析
    4. 消息批处理
    5. 会话管理

    Args:
        deps: 依赖配置对象，包含所有必需的依赖
    """

    def __init__(self, deps: BusinessServiceDeps) -> None:
        self._state = deps.shared_state
        self._cfg = deps.cfg
        self._storage = deps.storage
        self._analysis = deps.analysis
        self._llm_enhanced = deps.llm_enhanced
        self._capture = deps.capture
        self._retrieval = deps.retrieval
        self._kg = deps.kg
        self._image_analyzer = deps.image_analyzer
        self._member_identity = deps.member_identity
        self._activity_tracker = deps.activity_tracker
        self._context_builder = ContextBuilder(
            retrieval=deps.retrieval,
            analysis=deps.analysis,
            storage=deps.storage,
            kg=deps.kg,
            shared_state=deps.shared_state,
            cfg=deps.cfg,
            member_identity=deps.member_identity,
        )

    # ── 可选组件更新（初始化后注入） ──

    def set_image_analyzer(self, analyzer: Any) -> None:
        """更新图片分析器引用"""
        self._image_analyzer = analyzer

    def set_member_identity(self, identity: Any) -> None:
        """更新成员身份服务引用"""
        self._member_identity = identity
        self._context_builder.set_member_identity(identity)

    def set_activity_tracker(self, tracker: Any) -> None:
        """更新群活跃度跟踪器引用"""
        self._activity_tracker = tracker

    # ── 记忆捕获 ──

    async def capture_and_store_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        is_user_requested: bool = False,
        context: Optional[Dict[str, Any]] = None,
        sender_name: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> Optional[Any]:
        """捕获并存储记忆

        画像提取和知识图谱处理作为后台任务异步执行，
        不阻塞主回答流程。
        """
        if not self._capture.capture_engine:
            return None

        try:
            memory = await self._capture.capture_engine.capture_memory(
                message=message,
                user_id=user_id,
                group_id=group_id,
                is_user_requested=is_user_requested,
                context=context,
                sender_name=sender_name,
                persona_id=persona_id,
            )

            if not memory:
                return None

            await self._store_memory_by_layer(memory)

            # 画像提取 + KG 处理作为后台任务，不阻塞主流程
            asyncio.create_task(
                self._post_capture_background(memory, user_id)
            )

            return memory

        except Exception as e:
            logger.error(f"Failed to capture memory: {e}", exc_info=True)
            return None

    async def _post_capture_background(self, memory: Any, user_id: str) -> None:
        """后台执行画像更新和知识图谱提取（不阻塞主回答）"""
        # 画像更新
        try:
            await self._update_persona_from_memory(memory, user_id)
        except Exception as e:
            logger.warning(f"Background persona update failed: {e}")

        # 知识图谱处理
        if self._kg and self._kg.enabled:
            try:
                await self._kg.process_memory(memory)
            except Exception as e:
                logger.debug(f"Background KG processing failed: {e}")

    async def _store_memory_by_layer(self, memory: Any) -> None:
        """根据层级存储记忆"""
        if memory.storage_layer == StorageLayer.WORKING:
            if self._storage.session_manager:
                await self._storage.session_manager.add_working_memory(memory)
        else:
            if self._storage.chroma_manager:
                await self._storage.chroma_manager.add_memory(memory)

    async def _update_persona_from_memory(self, memory: Any, user_id: str) -> None:
        """画像闭环：从记忆更新用户画像并记录 DEBUG 日志

        批量处理策略：
        - emotion 类型消息 → 即时处理（情感时效性要求高）
        - 其他消息 → 若启用 persona_batch_processor 则入队批量处理
        - 批量处理器不可用时 → 自动降级到即时处理

        同时同步 sender_name 到 persona.display_name
        """
        if not self._cfg.get("persona.enabled", True):
            return

        try:
            persona = self._state.get_or_create_user_persona(user_id)
            mem_id = getattr(memory, "id", None)
            persona_log.update_start(user_id, mem_id)

            changes = []
            content = getattr(memory, "content", "") or ""
            summary = getattr(memory, "summary", None)
            mem_type_raw = getattr(memory, "type", None)
            mem_type = mem_type_raw.value if hasattr(mem_type_raw, "value") else str(mem_type_raw)
            confidence = getattr(memory, "confidence", 0.5)
            group_id = getattr(memory, "group_id", None)
            sender_name = getattr(memory, "sender_name", None)

            # 同步 sender_name 到 display_name（如果提供了且当前为空）
            if sender_name and not persona.display_name:
                persona.display_name = sender_name
                logger.debug(f"Updated display_name for user={user_id}: {sender_name}")

            # 情感类记忆保持即时处理
            if mem_type in ("emotion",):
                changes.extend(persona._update_emotional(memory, mem_id, confidence))

            # 尝试走批量处理通道
            persona_batch = self._analysis.persona_batch_processor
            if persona_batch and mem_type not in ("emotion",):
                try:
                    await persona_batch.add_message(
                        content=content,
                        user_id=user_id,
                        group_id=group_id,
                        summary=summary,
                        memory_type=mem_type,
                        confidence=confidence,
                        memory_id=mem_id,
                        sender_name=sender_name,
                    )
                    persona_log.update_skipped(user_id, "queued_for_batch")
                    logger.debug(
                        f"Persona update queued for batch: "
                        f"user={user_id}, memory={mem_id}"
                    )
                    # 仍然更新活跃时段（轻量操作，不需要批量）
                    created = getattr(memory, "created_time", None)
                    if created and isinstance(created, datetime):
                        persona.hourly_distribution[created.hour] += 1.0
                    # 如果有情感变更（emotion 路径），也记录
                    if changes:
                        persona_log.update_applied(
                            user_id,
                            [c.to_dict() for c in changes]
                        )
                    return
                except Exception as e:
                    logger.warning(
                        f"Persona batch enqueue failed, "
                        f"falling back to immediate: {e}"
                    )
                    # 降级到即时处理（下面的逻辑）

            # 即时处理路径（降级或未启用批量处理器）
            persona_extractor = self._analysis.persona_extractor
            if persona_extractor and self._cfg.get("persona.extraction_mode", "rule") != "rule":
                result = await persona_extractor.extract(
                    content=content,
                    summary=summary,
                )
                if result.confidence > 0 or result.interests:
                    ext_changes = persona.apply_extraction_result(
                        result,
                        source_memory_id=mem_id,
                        memory_type=mem_type,
                        base_confidence=confidence,
                    )
                    changes.extend(ext_changes)
                else:
                    changes.extend(persona.update_from_memory(memory))
            else:
                changes.extend(persona.update_from_memory(memory))

            created = getattr(memory, "created_time", None)
            if created and isinstance(created, datetime):
                persona.hourly_distribution[created.hour] += 1.0

            if changes:
                persona_log.update_applied(
                    user_id,
                    [c.to_dict() for c in changes]
                )
                logger.debug(
                    f"Persona updated for user={user_id}: "
                    f"{len(changes)} change(s) from memory={mem_id}"
                )
            else:
                persona_log.update_skipped(user_id, "no_applicable_changes")

        except Exception as e:
            persona_log.update_error(user_id, e)
            logger.warning(f"Failed to update persona from memory: {e}")

    # ── 记忆检索 ──

    async def search_memories(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int = NumericDefaults.TOP_K_SEARCH,
        persona_id: Optional[str] = None,
    ) -> List[Any]:
        """搜索记忆"""
        if not self._retrieval.retrieval_engine:
            return []

        try:
            emotional_state = self._state.get_or_create_emotional_state(user_id)

            memories = await self._retrieval.retrieval_engine.retrieve(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=top_k,
                emotional_state=emotional_state,
                persona_id=persona_id,
            )

            return memories

        except Exception as e:
            logger.warning(f"Failed to search memories: {e}")
            return []

    # ── 记忆删除 ──

    async def clear_memories(self, user_id: str, group_id: Optional[str]) -> bool:
        """清除用户记忆"""
        try:
            if self._storage.chroma_manager:
                await self._storage.chroma_manager.delete_session(user_id, group_id)

            if self._storage.session_manager:
                await self._storage.session_manager.clear_working_memory(user_id, group_id)

            if self._kg and self._kg.enabled:
                await self._kg.delete_user_data(user_id, group_id)

            return True

        except Exception as e:
            logger.warning(f"Failed to clear memories: {e}")
            return False

    async def delete_private_memories(self, user_id: str) -> Tuple[bool, int]:
        """删除用户私聊记忆"""
        try:
            if not self._storage.chroma_manager:
                return False, 0

            success, count = await self._storage.chroma_manager.delete_user_memories(
                user_id, in_private_only=True
            )

            if self._storage.session_manager:
                await self._storage.session_manager.clear_working_memory(user_id, None)

            # 同步删除图谱关联数据（私聊场景: group_id=None）
            if self._kg and self._kg.enabled:
                await self._kg.delete_user_data(user_id, group_id=None)

            return success, count

        except Exception as e:
            logger.warning(f"Failed to delete private memories: {e}")
            return False, 0

    async def delete_group_memories(
        self,
        group_id: str,
        scope_filter: Optional[str],
        user_id: Optional[str] = None
    ) -> Tuple[bool, int]:
        """删除群聊记忆"""
        try:
            if not self._storage.chroma_manager:
                return False, 0

            success, count = await self._storage.chroma_manager.delete_group_memories(
                group_id, scope_filter
            )

            if user_id and scope_filter != SessionScope.GROUP_SHARED:
                if self._storage.session_manager:
                    await self._storage.session_manager.clear_working_memory(user_id, group_id)

            # 同步删除图谱关联数据
            if self._kg and self._kg.enabled:
                if user_id:
                    await self._kg.delete_user_data(user_id, group_id)
                else:
                    # 无特定用户时，删除整个群的图谱数据
                    await self._kg.storage.delete_user_data_by_group(group_id)

            return success, count

        except Exception as e:
            logger.warning(f"Failed to delete group memories: {e}")
            return False, 0

    async def delete_all_memories(self) -> Tuple[bool, int]:
        """删除所有记忆"""
        try:
            if not self._storage.chroma_manager:
                return False, 0

            success, count = await self._storage.chroma_manager.delete_all_memories()

            if self._storage.session_manager:
                from iris_memory.storage.session_manager import SessionManager
                self._storage._session_manager = SessionManager(
                    max_working_memory=self._cfg.get("memory.max_working_memory", 10),
                    max_sessions=self._cfg.get("session.max_sessions", 100),
                    ttl=self._cfg.get("session.session_timeout", 86400),
                    activity_tracker=self._activity_tracker,
                )

            if self._kg and self._kg.enabled:
                await self._kg.delete_all()

            return success, count

        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}", exc_info=True)
            return False, 0

    # ── 统计 ──

    async def get_memory_stats(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> Dict[str, Any]:
        """获取记忆统计"""
        stats: Dict[str, Any] = {
            "working_count": 0,
            "episodic_count": 0,
            "image_analyzed": 0,
            "cache_hits": 0
        }

        try:
            if self._storage.session_manager:
                working_memories = await self._storage.session_manager.get_working_memory(
                    user_id, group_id
                )
                stats["working_count"] = len(working_memories)

            if self._storage.chroma_manager:
                stats["episodic_count"] = await self._storage.chroma_manager.count_memories(
                    user_id=user_id,
                    group_id=group_id
                )

            if self._image_analyzer:
                image_stats = self._image_analyzer.get_statistics()
                stats["image_analyzed"] = image_stats.get('total_analyzed', 0)
                stats["cache_hits"] = image_stats.get('cache_hits', 0)

            if self._kg and self._kg.enabled:
                try:
                    kg_stats = await self._kg.get_stats(user_id, group_id)
                    stats["kg_nodes"] = kg_stats.get("nodes", 0)
                    stats["kg_edges"] = kg_stats.get("edges", 0)
                except Exception:
                    stats["kg_nodes"] = 0
                    stats["kg_edges"] = 0

        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")

        return stats

    # ── LLM 上下文 ──

    async def prepare_llm_context(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        image_context: str = "",
        sender_name: Optional[str] = None,
        reply_context: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> str:
        """准备 LLM 上下文（委托给 ContextBuilder）"""
        return await self._context_builder.build(
            query=query,
            user_id=user_id,
            group_id=group_id,
            image_context=image_context,
            sender_name=sender_name,
            reply_context=reply_context,
            persona_id=persona_id,
        )

    # ── 图片分析 ──

    async def analyze_images(
        self,
        message_chain: List[Any],
        user_id: str,
        group_id: Optional[str],
        context_text: str,
        umo: str,
        session_id: str
    ) -> Tuple[str, str]:
        """分析图片"""
        if not self._image_analyzer:
            return "", ""

        try:
            daily_budget = self._cfg.get("image_analysis.daily_budget", 100)
            effective_daily_budget = daily_budget if daily_budget > 0 else UNLIMITED_BUDGET

            image_results = await self._image_analyzer.analyze_message_images(
                message_chain=message_chain,
                user_id=user_id,
                context_text=context_text,
                umo=umo,
                session_id=session_id,
                daily_analysis_budget=effective_daily_budget,
            )

            if not image_results:
                return "", ""

            llm_context = self._image_analyzer.format_for_llm_context(image_results)
            memory_format = self._image_analyzer.format_for_memory(image_results)

            return llm_context, memory_format

        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return "", ""

    # ── 消息批处理 ──

    async def process_message_batch(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        context: Dict[str, Any],
        umo: str,
        image_description: str = "",
        persona_id: Optional[str] = None,
    ) -> None:
        """处理消息批次"""
        batch_processor = self._capture.batch_processor
        message_classifier = self._capture.message_classifier
        if not batch_processor or not message_classifier:
            return

        try:
            full_message = message
            if image_description:
                full_message = f"{message} {image_description}".strip()
                context["has_image"] = True
                context["image_description"] = image_description

            classification = await message_classifier.classify(full_message, context)

            logger.debug(
                f"Message classified: {classification.layer.value} "
                f"(confidence: {classification.confidence:.2f}, source: {classification.source})"
            )

            if classification.layer.value == "discard":
                return

            if classification.layer.value == "immediate":
                sender_name = context.get("sender_name")
                await self._handle_immediate_memory(
                    full_message, user_id, group_id, classification, sender_name,
                    persona_id=persona_id,
                )
            else:
                sender_name = context.get("sender_name")
                await batch_processor.add_message(
                    content=full_message,
                    user_id=user_id,
                    sender_name=sender_name,
                    group_id=group_id,
                    context=context,
                    umo=umo,
                    persona_id=persona_id or "default",
                )

        except Exception as e:
            logger.warning(f"Failed to process message batch: {e}")

    async def _handle_immediate_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        classification: Any,
        sender_name: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> None:
        """处理立即层级的记忆"""
        memory = await self.capture_and_store_memory(
            message=message,
            user_id=user_id,
            group_id=group_id,
            context={
                "classification": classification.metadata,
                "source": classification.source
            },
            sender_name=sender_name,
            persona_id=persona_id,
        )

        if memory:
            logger.debug(LogTemplates.IMMEDIATE_MEMORY_CAPTURED.format(memory_id=memory.id))

    # ── 聊天记录 ──

    async def record_chat_message(
        self,
        sender_id: str,
        sender_name: Optional[str],
        content: str,
        group_id: Optional[str] = None,
        is_bot: bool = False,
        session_user_id: Optional[str] = None,
        reply_sender_name: Optional[str] = None,
        reply_sender_id: Optional[str] = None,
        reply_content: Optional[str] = None,
    ) -> None:
        """记录一条聊天消息到缓冲区

        Args:
            sender_id: 发送者ID
            sender_name: 发送者昵称
            content: 消息内容
            group_id: 群组ID
            is_bot: 是否为Bot回复
            session_user_id: 会话ID（私聊Bot回复归属）
            reply_sender_name: 被引用消息的发送者昵称
            reply_sender_id: 被引用消息的发送者ID
            reply_content: 被引用消息的内容摘要
        """
        if is_bot and MessageFilter.should_skip_bot_message(content):
            logger.debug("Skipping bot rich-text message from chat history")
            return

        chat_history_buffer = self._storage.chat_history_buffer
        if chat_history_buffer:
            await chat_history_buffer.add_message(
                sender_id=sender_id,
                sender_name=sender_name,
                content=content,
                group_id=group_id,
                is_bot=is_bot,
                session_user_id=session_user_id,
                reply_sender_name=reply_sender_name,
                reply_sender_id=reply_sender_id,
                reply_content=reply_content,
            )

    # ── 会话管理 ──

    def update_session_activity(self, user_id: str, group_id: Optional[str]) -> None:
        """更新会话活动"""
        if self._storage.session_manager:
            self._storage.session_manager.update_session_activity(user_id, group_id)

    async def activate_session(self, user_id: str, group_id: Optional[str]) -> None:
        """激活会话"""
        if self._storage.lifecycle_manager:
            await self._storage.lifecycle_manager.activate_session(user_id, group_id)

    # ── 用户状态代理（供外部直接访问） ──

    def get_or_create_user_persona(self, user_id: str) -> Any:
        """获取或创建用户画像（代理到 SharedState）"""
        return self._state.get_or_create_user_persona(user_id)

    def _get_or_create_emotional_state(self, user_id: str) -> Any:
        """获取或创建用户情感状态（代理到 SharedState）"""
        return self._state.get_or_create_emotional_state(user_id)

    # ── 向后兼容代理（逻辑已移至 ContextBuilder） ──

    async def _build_chat_history_context(
        self, user_id: str, group_id: Optional[str]
    ) -> str:
        """Backward-compat delegate → ContextBuilder._build_chat_history"""
        return await self._context_builder._build_chat_history(user_id, group_id)

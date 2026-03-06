"""LLM 上下文构建器

从 BusinessService 中提取的 prepare_llm_context 及其辅助方法，
负责将记忆、聊天记录、画像、知识图谱、图片描述等拼装为 LLM prompt。
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, TYPE_CHECKING

from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import PersonaStyle, LogTemplates
from iris_memory.utils.command_utils import SessionKeyBuilder
from iris_memory.utils.member_utils import format_member_tag
from iris_memory.analysis.persona.persona_logger import persona_log

if TYPE_CHECKING:
    from iris_memory.services.modules.storage_module import StorageModule
    from iris_memory.services.modules.analysis_module import AnalysisModule
    from iris_memory.services.modules.retrieval_module import RetrievalModule
    from iris_memory.services.modules.kg_module import KnowledgeGraphModule
    from iris_memory.services.shared_state import SharedState

logger = get_logger("memory_service.context_builder")


class ContextBuilder:
    """LLM 上下文构建器

    负责将多源信息组装为完整的 LLM prompt 上下文字符串。

    依赖：
    - retrieval: 记忆检索引擎
    - analysis: 情感分析器
    - storage: 聊天记录缓冲区
    - kg: 知识图谱上下文
    - shared_state: 用户画像和情感状态
    - cfg: 配置管理器
    """

    def __init__(
        self,
        retrieval: "RetrievalModule",
        analysis: "AnalysisModule",
        storage: "StorageModule",
        kg: "KnowledgeGraphModule",
        shared_state: "SharedState",
        cfg: Any,
        member_identity: Any = None,
    ) -> None:
        self._retrieval = retrieval
        self._analysis = analysis
        self._storage = storage
        self._kg = kg
        self._state = shared_state
        self._cfg = cfg
        self._member_identity = member_identity

    def set_member_identity(self, identity: Any) -> None:
        """更新成员身份服务引用"""
        self._member_identity = identity

    # ── 主入口 ──

    async def build(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        image_context: str = "",
        sender_name: Optional[str] = None,
        reply_context: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> str:
        """构建完整的 LLM 上下文

        Args:
            query: 用户查询文本
            user_id: 用户ID
            group_id: 群组ID
            image_context: 图片分析上下文
            sender_name: 发送者名称
            reply_context: 引用消息上下文（已格式化的描述文本）
            persona_id: 人格 ID（非 None 时启用 persona 过滤）
        """
        if not self._retrieval.retrieval_engine:
            return ""

        try:
            emotional_state = self._state.get_or_create_emotional_state(user_id)
            if self._analysis.emotion_analyzer:
                emotion_result = await self._analysis.emotion_analyzer.analyze_emotion(query)
                self._analysis.emotion_analyzer.update_emotional_state(
                    emotional_state,
                    emotion_result["primary"],
                    emotion_result["intensity"],
                    emotion_result["confidence"],
                    emotion_result["secondary"]
                )

            memories = await self._retrieval.retrieval_engine.retrieve(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=self._cfg.get("memory.max_context_memories", 10),
                emotional_state=emotional_state,
                persona_id=persona_id,
            )

            session_key = SessionKeyBuilder.build(user_id, group_id)
            if memories:
                memories = self._state.filter_recently_injected(memories, session_key)

            context_parts: List[str] = []

            chat_context = await self._build_chat_history(user_id, group_id)
            if chat_context:
                context_parts.append(chat_context)

            if memories:
                persona = self._state.get_or_create_user_persona(user_id)
                persona_view = persona.to_injection_view() if persona else None
                if persona_view:
                    persona_log.inject_view(user_id, persona_view)

                memory_context = self._retrieval.retrieval_engine.format_memories_for_llm(
                    memories,
                    persona_style=PersonaStyle.NATURAL,
                    user_persona=persona_view,
                    group_id=group_id,
                    current_sender_name=sender_name
                )
                context_parts.append(memory_context)
                logger.debug(LogTemplates.MEMORY_INJECTED.format(count=len(memories)))

                self._state.track_injected_memories(
                    session_key,
                    [m.id for m in memories]
                )

            member_context = self._build_member_identity(
                memories, group_id, user_id, sender_name
            )
            if member_context:
                context_parts.append(member_context)

            # 知识图谱上下文
            if self._kg and self._kg.enabled:
                try:
                    kg_context = await self._kg.format_graph_context(
                        query=query,
                        user_id=user_id,
                        group_id=group_id,
                        persona_id=persona_id,
                    )
                    if kg_context:
                        context_parts.append(kg_context)
                        logger.debug("Injected knowledge graph context into LLM prompt")
                except Exception as kg_err:
                    logger.debug(f"KG context skipped: {kg_err}")

            if image_context:
                context_parts.append(image_context)
                logger.debug("Injected image context into LLM prompt")

            # 引用消息上下文
            if reply_context:
                context_parts.append(reply_context)
                logger.debug("Injected reply context into LLM prompt")

            behavior_directives = self._build_behavior_directives(group_id, sender_name)
            if behavior_directives:
                context_parts.append(behavior_directives)

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Failed to prepare LLM context: {e}")
            return ""

    # ── 辅助构建方法 ──

    def _build_behavior_directives(
        self,
        group_id: Optional[str],
        sender_name: Optional[str] = None
    ) -> str:
        """构建行为指导，与人格 Prompt 协同工作"""
        directives = [
            "【记忆使用规则】",
            "◆ 禁止重复：不要反复提起同一件事或记忆。如果你刚才已经提到过某个话题，就自然地聊别的，不要翻来覆去说同一件事。",
            "◆ 减少反问：不要频繁反问对方，尤其不要重复问同一个问题。用陈述、共鸣、接话的方式回应，像真人朋友那样自然接话。如果想了解更多，偶尔问一下就够了。",
            "◆ 简短自然：回复尽量简短，像群里随手接话，一行结束。不要写长篇大论，不要列清单式回答日常闲聊。",
        ]

        if group_id:
            directives.append("◆ 知识区分：记忆中标注了「群聊共识」和「个人信息」。群聊共识是大家都知道的事，个人信息是某个人的私事。引用个人信息时要确认是当前对话者的，不要张冠李戴。")
        else:
            directives.append("◆ 这是私聊对话，记忆都是你和对方之间的。")

        return "\n".join(directives)

    def _build_member_identity(
        self,
        memories: List[Any],
        group_id: Optional[str],
        user_id: str,
        sender_name: Optional[str]
    ) -> str:
        """Build a compact member identity hint for group chats."""
        if not group_id:
            return ""

        current_tag = format_member_tag(sender_name, user_id, group_id)
        other_tags: List[str] = []
        seen: set = set()

        for memory in memories:
            tag = format_member_tag(memory.sender_name, memory.user_id, group_id)
            if not tag:
                continue
            if tag == current_tag:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            other_tags.append(tag)

        lines = [
            "【群成员识别】",
            f"当前对话者: {current_tag}。回复时针对这个人，不要混淆成其他群友。",
        ]

        if other_tags:
            lines.append("记忆中涉及成员: " + ", ".join(other_tags[:5]))

        if self._member_identity:
            all_members = self._member_identity.get_group_members(group_id)
            extra_members = [
                m for m in all_members
                if m != current_tag and m not in seen
            ]
            if extra_members:
                lines.append(
                    "群内其他已知成员: " + ", ".join(extra_members[:10])
                )

            history = self._member_identity.get_name_history(user_id)
            if history:
                last_change = history[-1]
                lines.append(
                    f"注意: 当前对话者曾用名 \"{last_change['old_name']}\"，"
                    f"现在叫 \"{last_change['new_name']}\"。"
                )

        lines.append(
            "同名以#后ID区分。不要把A说的话当成B说的，"
            "引用其他人的记忆时要明确说明。"
        )

        return "\n".join(lines)

    async def _build_chat_history(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> str:
        """构建聊天记录上下文"""
        chat_history_buffer = self._storage.chat_history_buffer
        if not chat_history_buffer:
            return ""

        chat_context_count = self._cfg.get("advanced.chat_context_count", 20)
        if chat_context_count <= 0:
            return ""

        if chat_history_buffer.max_messages < chat_context_count:
            chat_history_buffer.set_max_messages(chat_context_count)

        messages = await chat_history_buffer.get_recent_messages(
            user_id=user_id,
            group_id=group_id,
            limit=chat_context_count
        )

        if not messages:
            return ""

        context = chat_history_buffer.format_for_llm(
            messages,
            group_id=group_id
        )

        if context:
            logger.debug(
                f"Injected {len(messages)} chat messages into context "
                f"(group={group_id is not None})"
            )

        return context

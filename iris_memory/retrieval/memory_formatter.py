"""记忆格式化器 — 将记忆列表格式化为 LLM 可注入的文本

从 retrieval_engine.py 分离，符合单一职责原则。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from iris_memory.models.memory import Memory
from iris_memory.core.memory_scope import MemoryScope
from iris_memory.utils.member_utils import format_member_tag
from iris_memory.utils.token_manager import TokenBudget, MemoryCompressor, DynamicMemorySelector
from iris_memory.analysis.persona.persona_coordinator import PersonaCoordinator, CoordinationStrategy
from iris_memory.core.constants import RetrievalDefaults


class MemoryFormatter:
    """记忆格式化器

    负责将检索到的记忆转换为 LLM 上下文注入文本。
    支持多种风格：default / natural / roleplay。
    """

    def __init__(
        self,
        token_budget: Optional[TokenBudget] = None,
        compressor: Optional[MemoryCompressor] = None,
        selector: Optional[DynamicMemorySelector] = None,
        persona_coordinator: Optional[PersonaCoordinator] = None,
    ) -> None:
        self.token_budget = token_budget or TokenBudget(
            total_budget=RetrievalDefaults.TOKEN_BUDGET
        )
        self.memory_compressor = compressor or MemoryCompressor(
            max_summary_length=RetrievalDefaults.MAX_SUMMARY_LENGTH
        )
        self.memory_selector = selector or DynamicMemorySelector(
            token_budget=self.token_budget,
            compressor=self.memory_compressor,
        )
        self.persona_coordinator = persona_coordinator or PersonaCoordinator(
            strategy=CoordinationStrategy.HYBRID
        )

    def format_memories_for_llm(
        self,
        memories: List[Memory],
        use_token_budget: bool = True,
        user_persona: Optional[Dict[str, Any]] = None,
        persona_style: str = "default",
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None,
        max_context_memories: int = 3,
    ) -> str:
        """格式化记忆用于注入到 LLM 上下文

        Args:
            memories: 记忆列表
            use_token_budget: 是否使用 token 预算管理
            user_persona: 用户画像视图
            persona_style: 人格风格 (default/natural/roleplay)
            group_id: 群组ID
            current_sender_name: 当前发言者名称
            max_context_memories: 最大上下文记忆数

        Returns:
            格式化的记忆文本
        """
        if not memories:
            return ""

        if use_token_budget:
            return self.memory_selector.get_memory_context(
                memories,
                target_count=max_context_memories,
                persona_style=persona_style,
                group_id=group_id,
                current_sender_name=current_sender_name,
            )

        if persona_style == "natural":
            formatted = self._format_natural_style(memories, group_id, current_sender_name)
        elif persona_style == "roleplay":
            formatted = self._format_roleplay_style(memories, group_id, current_sender_name)
        else:
            formatted = self._format_default_style(memories, group_id, current_sender_name)

        if user_persona:
            formatted = self.persona_coordinator.format_context_with_persona(
                formatted,
                user_persona,
                bot_persona="friendly",
            )

        return formatted

    def _format_memory_label(self, memory: Memory, group_id: Optional[str] = None) -> str:
        """为记忆生成来源标签"""
        parts = []

        if memory.scope == MemoryScope.GROUP_SHARED:
            parts.append("群聊共识")
        elif memory.scope == MemoryScope.GROUP_PRIVATE:
            sender_tag = self._format_sender_tag(memory, group_id)
            if not sender_tag:
                sender_tag = format_member_tag(None, memory.user_id, group_id)
            if sender_tag:
                parts.append(f"{sender_tag}的个人信息")
            else:
                parts.append("个人信息")
        elif memory.scope == MemoryScope.USER_PRIVATE:
            parts.append("私聊记忆")

        return "｜".join(parts) if parts else ""

    def _format_natural_style(
        self,
        memories: List[Memory],
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None,
    ) -> str:
        """自然群友风格格式化"""
        formatted = "【你记得的事情】\n"
        formatted += "以下是你和群友之间的往事，请用自己的话自然提及，不要暴露'记录'、'数据'等概念：\n"

        if group_id:
            formatted += "（注意区分群共识和个人信息，不要把A的事情说成B的）\n"

        for memory in memories:
            label = self._format_memory_label(memory, group_id)
            sender_tag = self._format_sender_tag(memory, group_id)
            sender = f"（{sender_tag}说的）" if sender_tag else ""
            prefix = f"[{label}]" if label else ""
            formatted += f"- {prefix}{sender}{memory.content}\n"

        return formatted

    def _format_roleplay_style(
        self,
        memories: List[Memory],
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None,
    ) -> str:
        """角色扮演风格格式化"""
        formatted = "【你的记忆】\n"
        formatted += "这些都是你亲身经历的事情，回复时可以自然地说'我记得...'、'你之前说过...'：\n"
        for memory in memories:
            sender_tag = self._format_sender_tag(memory, group_id)
            sender = f"（{sender_tag}）" if sender_tag else ""
            formatted += f"· {sender}{memory.content}\n"
        return formatted

    def _format_default_style(
        self,
        memories: List[Memory],
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None,
    ) -> str:
        """默认格式化"""
        formatted = "【相关记忆】\n"
        for i, memory in enumerate(memories, 1):
            time_str = memory.created_time.strftime("%Y-%m-%d %H:%M")
            if hasattr(memory.type, "value"):
                type_label = memory.type.value.upper()
            else:
                type_label = str(memory.type).upper()

            label = self._format_memory_label(memory, group_id)
            sender_tag = self._format_sender_tag(memory, group_id)
            sender = f" @{sender_tag}" if sender_tag else ""

            formatted += f"{i}. [{type_label}]{sender} {time_str}"
            if label:
                formatted += f" ({label})"
            formatted += f"\n   内容: {memory.content}\n"

            if memory.summary:
                formatted += f"   摘要: {memory.summary}\n"

            if memory.emotional_weight > 0.5:
                formatted += f"   情感强度: {memory.emotional_weight:.2f}\n"

            formatted += "\n"

        return formatted

    @staticmethod
    def _format_sender_tag(memory: Memory, group_id: Optional[str]) -> str:
        """Format a stable sender tag for group disambiguation."""
        if group_id:
            return format_member_tag(memory.sender_name, memory.user_id, group_id)
        return (memory.sender_name or "").strip()

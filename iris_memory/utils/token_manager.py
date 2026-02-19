"""
Token管理器
管理记忆注入的token预算，避免超出LLM限制
"""

from typing import List, Tuple, Optional
from enum import Enum

from iris_memory.utils.logger import get_logger
from iris_memory.utils.member_utils import format_member_tag

# 模块logger
logger = get_logger("token_manager")

# 尝试加载 tiktoken 以获得精确的 token 估算
_tiktoken_encoding = None
try:
    import tiktoken
    _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    logger.debug("tiktoken loaded — using cl100k_base encoding for token estimation")
except ImportError:
    logger.debug("tiktoken not available — using heuristic token estimation")
except Exception as e:
    logger.debug(f"tiktoken failed to initialize: {e} — using heuristic estimation")


class TokenType(str, Enum):
    """Token类型"""
    MEMORY_SUMMARY = "memory_summary"
    MEMORY_FULL = "memory_full"
    PREAMBLE = "preamble"
    POSTAMBLE = "postamble"


class TokenBudget:
    """Token预算管理"""
    
    def __init__(
        self,
        total_budget: int = 512,
        preamble_cost: int = 20,
        postamble_cost: int = 10
    ):
        """初始化Token预算
        
        Args:
            total_budget: 总预算（默认512 tokens）
            preamble_cost: 前导内容消耗（默认20 tokens）
            postamble_cost: 后导内容消耗（默认10 tokens）
        """
        self.total_budget = total_budget
        self.preamble_cost = preamble_cost
        self.postamble_cost = postamble_cost
        self.used_budget = preamble_cost  # 已使用预算（包含前导）
        
        # 启发式估算参数（仅在 tiktoken 不可用时使用）
        self.chars_per_token_cn = 1.5    # 中文：约1.5字符/token
        self.chars_per_token_en = 4.0    # 英文：约4字符/token
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量
        
        优先使用 tiktoken（精确），不可用时使用加权启发式估算。
        
        Args:
            text: 输入文本
            
        Returns:
            int: 估算的token数量
        """
        if not text:
            return 0
        
        # 优先使用 tiktoken 精确计数
        if _tiktoken_encoding is not None:
            return len(_tiktoken_encoding.encode(text))
        
        # 启发式：按中英文字符比例加权估算
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text)
        non_chinese_chars = total_chars - chinese_chars
        
        cn_tokens = chinese_chars / self.chars_per_token_cn
        en_tokens = non_chinese_chars / self.chars_per_token_en
        
        return max(1, int(cn_tokens + en_tokens))
    
    def can_add_memory(
        self,
        memory_text: str,
        as_summary: bool = True
    ) -> bool:
        """判断是否可以添加记忆
        
        Args:
            memory_text: 记忆文本
            as_summary: 是否作为摘要（摘要通常更短）
            
        Returns:
            bool: 是否可以添加
        """
        # 估算所需token
        tokens = self.estimate_tokens(memory_text)
        if as_summary:
            tokens = int(tokens * 0.7)  # 摘要通常节省30% token
        
        # 检查预算
        new_used = self.used_budget + tokens
        return new_used <= self.total_budget
    
    def add_memory(self, memory_text: str, as_summary: bool = True) -> int:
        """添加记忆并返回实际消耗的token
        
        Args:
            memory_text: 记忆文本
            as_summary: 是否作为摘要
            
        Returns:
            int: 实际消耗的token数量
        """
        tokens = self.estimate_tokens(memory_text)
        if as_summary:
            tokens = int(tokens * 0.7)
        
        self.used_budget += tokens
        return tokens
    
    def get_remaining_budget(self) -> int:
        """获取剩余预算
        
        Returns:
            int: 剩余token数量
        """
        return self.total_budget - self.used_budget
    
    def get_utilization(self) -> float:
        """获取预算利用率
        
        Returns:
            float: 利用率（0-1）
        """
        return self.used_budget / self.total_budget
    
    def reset(self):
        """重置预算"""
        self.used_budget = self.preamble_cost
    
    def finalize(self) -> bool:
        """ finalize预算（添加后导内容）
        
        Returns:
            bool: 是否成功（预算足够）
        """
        if self.used_budget + self.postamble_cost <= self.total_budget:
            self.used_budget += self.postamble_cost
            return True
        return False


class MemoryCompressor:
    """记忆压缩器
    
    实现记忆摘要提取，减少token消耗
    """
    
    def __init__(self, max_summary_length: int = 100):
        """初始化压缩器
        
        Args:
            max_summary_length: 最大摘要长度（字符）
        """
        self.max_summary_length = max_summary_length
    
    def compress_memory(
        self,
        content: str,
        summary: Optional[str] = None
    ) -> Tuple[str, bool]:
        """压缩记忆

        Args:
            content: 记忆内容
            summary: 记忆摘要（如果存在）

        Returns:
            Tuple[str, bool]: (压缩后的文本, 是否使用了摘要)
        """
        # 优先使用摘要
        if summary and len(summary) > 0:
            # 确保 max_summary_length 非负
            max_len = max(0, self.max_summary_length)
            compressed = summary[:max_len]
            return compressed, True

        # 提取摘要（简化版：截取前面部分）
        max_len = max(0, self.max_summary_length)
        if len(content) <= max_len:
            return content, False

        # 截取关键部分
        compressed = content[:max_len] + "..."
        return compressed, False
    
    def compress_memories(
        self,
        memories: List[Tuple[str, Optional[str]]]
    ) -> List[Tuple[str, bool]]:
        """批量压缩记忆
        
        Args:
            memories: (content, summary) 元组列表
            
        Returns:
            List[Tuple[str, bool]]: 压缩结果列表
        """
        results = []
        for content, summary in memories:
            compressed, used_summary = self.compress_memory(content, summary)
            results.append((compressed, used_summary))
        
        return results


class DynamicMemorySelector:
    """动态记忆选择器
    
    根据token预算动态选择要注入的记忆
    """
    
    def __init__(
        self,
        token_budget: TokenBudget,
        compressor: Optional[MemoryCompressor] = None
    ):
        """初始化选择器
        
        Args:
            token_budget: Token预算管理器
            compressor: 记忆压缩器（可选）
        """
        self.token_budget = token_budget
        self.compressor = compressor or MemoryCompressor()
    
    def select_memories(
        self,
        memories: List,
        target_count: int
    ) -> Tuple[List, dict]:
        """选择要注入的记忆
        
        Args:
            memories: 候选记忆列表
            target_count: 目标数量
            
        Returns:
            Tuple[List, dict]: (选中的记忆列表, 统计信息)
        """
        # 重置预算
        self.token_budget.reset()
        
        selected = []
        stats = {
            "total_candidates": len(memories),
            "selected_count": 0,
            "used_tokens": 0,
            "skipped_count": 0,
            "summary_used": 0
        }
        
        # 按重要性排序
        sorted_memories = sorted(
            memories,
            key=lambda m: m.rif_score * m.importance_score,
            reverse=True
        )
        
        # 尝试选择记忆
        for memory in sorted_memories[:target_count]:
            # 压缩记忆
            compressed, used_summary = self.compressor.compress_memory(
                memory.content,
                memory.summary
            )
            
            # 检查是否可以添加
            if self.token_budget.can_add_memory(compressed, as_summary=True):
                # 添加记忆
                tokens = self.token_budget.add_memory(compressed, as_summary=True)
                selected.append(memory)
                stats["selected_count"] += 1
                stats["used_tokens"] += tokens
                if used_summary:
                    stats["summary_used"] += 1
            else:
                # 跳过记忆
                stats["skipped_count"] += 1
                # 如果预算不足，停止选择
                break
        
        # Finalize预算
        finalized = self.token_budget.finalize()
        if not finalized:
            stats["postamble_skipped"] = True
        
        return selected, stats
    
    def get_memory_context(
        self,
        memories: List,
        target_count: int = 3,
        persona_style: str = "default",
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None
    ) -> str:
        """生成记忆上下文文本
        
        Args:
            memories: 记忆列表
            target_count: 目标数量
            persona_style: 人格风格 (default/natural/roleplay)
            group_id: 群组ID（用于区分群聊/个人知识）
            current_sender_name: 当前发言者名称
            
        Returns:
            str: 格式化的记忆上下文
        """
        selected, stats = self.select_memories(memories, target_count)
        
        if not selected:
            return ""
        
        # 根据人格风格选择格式化方式
        if persona_style == "natural":
            lines = self._format_natural(selected, group_id, current_sender_name)
        elif persona_style == "roleplay":
            lines = self._format_roleplay(selected, group_id, current_sender_name)
        else:
            lines = self._format_default(selected, group_id, current_sender_name)
        
        context = "\n".join(lines)

        # 记录统计信息
        logger.debug(
            f"Memory context generated: {stats['selected_count']}/{stats['total_candidates']} memories, "
            f"{stats['used_tokens']} tokens, "
            f"{stats['skipped_count']} skipped, "
            f"{stats['summary_used']} summaries used"
        )

        return context

    def _get_scope_label(self, memory) -> str:
        """获取记忆的来源范围标签
        
        Args:
            memory: 记忆对象
            
        Returns:
            str: 来源标签
        """
        try:
            from iris_memory.core.memory_scope import MemoryScope
            if memory.scope == MemoryScope.GROUP_SHARED:
                return "群聊共识"
            elif memory.scope == MemoryScope.GROUP_PRIVATE:
                return "个人信息"
            elif memory.scope == MemoryScope.USER_PRIVATE:
                return "私聊"
            return ""
        except Exception as e:
            logger.debug(f"Failed to get scope label: {e}")
            return ""
    
    def _format_natural(
        self,
        selected: List,
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None
    ) -> List[str]:
        """自然群友风格格式化"""
        lines = ["【你记得的事情】"]
        lines.append("以下是你和群友之间的往事，请用自己的话自然提及，不要暴露'记录'、'数据'等概念：")
        
        if group_id:
            lines.append("（注意：区分群共识和个人信息，不要把A的事情说成B的）")
        
        for i, memory in enumerate(selected, 1):
            compressed, _ = self.compressor.compress_memory(
                memory.content, memory.summary
            )
            
            # 构建标签
            parts = []
            scope_label = self._get_scope_label(memory)
            if scope_label:
                parts.append(scope_label)
            
            sender = ""
            if group_id:
                sender_tag = format_member_tag(
                    getattr(memory, 'sender_name', None),
                    getattr(memory, 'user_id', None),
                    group_id
                )
                sender = f"（{sender_tag}说的）" if sender_tag else ""
            elif getattr(memory, 'sender_name', None):
                sender = f"（{memory.sender_name}说的）"
            
            label = f"[{'｜'.join(parts)}]" if parts else ""
            lines.append(f"- {label}{sender}{compressed}")
        
        return lines
    
    def _format_roleplay(
        self,
        selected: List,
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None
    ) -> List[str]:
        """角色扮演风格格式化"""
        lines = ["【你的记忆】"]
        lines.append("这些都是你亲身经历的事情，回复时可以自然地说'我记得...'、'你之前说过...'：")
        for memory in selected:
            compressed, _ = self.compressor.compress_memory(
                memory.content, memory.summary
            )
            if group_id:
                sender_tag = format_member_tag(
                    getattr(memory, 'sender_name', None),
                    getattr(memory, 'user_id', None),
                    group_id
                )
                sender = f"（{sender_tag}）" if sender_tag else ""
            else:
                sender = f"（{memory.sender_name}）" if getattr(memory, 'sender_name', None) else ""
            lines.append(f"· {sender}{compressed}")
        return lines
    
    def _format_default(
        self,
        selected: List,
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None
    ) -> List[str]:
        """默认格式化"""
        lines = ["【相关记忆】"]
        for i, memory in enumerate(selected, 1):
            compressed, _ = self.compressor.compress_memory(
                memory.content, memory.summary
            )
            
            time_str = memory.created_time.strftime("%m-%d %H:%M")
            if hasattr(memory.type, 'value'):
                type_label = memory.type.value.upper()
            else:
                type_label = str(memory.type).upper()
            
            scope_label = self._get_scope_label(memory)
            if group_id:
                sender_tag = format_member_tag(
                    getattr(memory, 'sender_name', None),
                    getattr(memory, 'user_id', None),
                    group_id
                )
                sender = f" @{sender_tag}" if sender_tag else ""
            else:
                sender = f" @{memory.sender_name}" if getattr(memory, 'sender_name', None) else ""
            scope_tag = f" ({scope_label})" if scope_label else ""
            
            lines.append(f"{i}. [{type_label}]{sender} {time_str}{scope_tag}")
            lines.append(f"   {compressed}")
        
        return lines

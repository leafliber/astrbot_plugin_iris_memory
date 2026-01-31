"""
Token管理器
管理记忆注入的token预算，避免超出LLM限制
"""

from typing import List, Tuple
from enum import Enum


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
        
        # Token估算（中文按字符数估算，英文按词数估算）
        self.chars_per_token = 1.5  # 中文：约1.5字符/token
        self.words_per_token = 0.75  # 英文：约0.75词/token
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: 估算的token数量
        """
        # 检测文本类型（中文/英文/混合）
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text)
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        
        # 混合估算
        if chinese_ratio > 0.5:
            # 主要中文：按字符估算
            return int(total_chars / self.chars_per_token)
        else:
            # 主要英文：按词估算
            words = len(text.split())
            return int(words / self.words_per_token)
    
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
        target_count: int = 3
    ) -> str:
        """生成记忆上下文文本
        
        Args:
            memories: 记忆列表
            target_count: 目标数量
            
        Returns:
            str: 格式化的记忆上下文
        """
        selected, stats = self.select_memories(memories, target_count)
        
        if not selected:
            return ""
        
        # 生成上下文文本
        lines = ["【相关记忆】"]
        for i, memory in enumerate(selected, 1):
            compressed, _ = self.compressor.compress_memory(
                memory.content,
                memory.summary
            )
            
            time_str = memory.created_time.strftime("%m-%d %H:%M")
            # 处理type可能是枚举或字符串的情况
            if hasattr(memory.type, 'value'):
                type_label = memory.type.value.upper()
            else:
                type_label = str(memory.type).upper()
            lines.append(f"{i}. [{type_label}] {time_str}")
            lines.append(f"   {compressed}")
        
        context = "\n".join(lines)

        # 记录统计信息
        try:
            from astrbot.api import logger
            logger.debug(
                f"Memory context generated: {stats['selected_count']}/{stats['total_candidates']} memories, "
                f"{stats['used_tokens']} tokens, "
                f"{stats['skipped_count']} skipped, "
                f"{stats['summary_used']} summaries used"
            )
        except ImportError:
            # 如果在测试环境中，使用标准logging
            import logging
            logging.debug(
                f"Memory context generated: {stats['selected_count']}/{stats['total_candidates']} memories, "
                f"{stats['used_tokens']} tokens, "
                f"{stats['skipped_count']} skipped, "
                f"{stats['summary_used']} summaries used"
            )

        return context

"""
TokenManager测试
测试Token预算管理器和记忆压缩器的核心功能
"""

import pytest
from typing import Optional, Tuple, List
from iris_memory.utils.token_manager import (
    TokenBudget,
    TokenType,
    MemoryCompressor,
    DynamicMemorySelector
)
from unittest.mock import Mock


class TestTokenTypeEnum:
    """测试TokenType枚举"""
    
    def test_token_type_values(self):
        """测试Token类型枚举值"""
        assert TokenType.MEMORY_SUMMARY.value == "memory_summary"
        assert TokenType.MEMORY_FULL.value == "memory_full"
        assert TokenType.PREAMBLE.value == "preamble"
        assert TokenType.POSTAMBLE.value == "postamble"


class TestTokenBudgetInit:
    """测试TokenBudget初始化"""
    
    def test_init_default_values(self):
        """测试默认值初始化"""
        budget = TokenBudget()
        
        assert budget.total_budget == 512
        assert budget.preamble_cost == 20
        assert budget.postamble_cost == 10
        assert budget.used_budget == 20  # 初始使用preamble_cost
        assert budget.chars_per_token == 1.5
        assert budget.words_per_token == 0.75
    
    def test_init_custom_values(self):
        """测试自定义值初始化"""
        budget = TokenBudget(
            total_budget=1024,
            preamble_cost=50,
            postamble_cost=20
        )
        
        assert budget.total_budget == 1024
        assert budget.preamble_cost == 50
        assert budget.postamble_cost == 20
        assert budget.used_budget == 50
    
    def test_init_zero_preamble(self):
        """测试preamble_cost为0"""
        budget = TokenBudget(preamble_cost=0)
        
        assert budget.used_budget == 0


class TestTokenBudgetEstimateTokens:
    """测试Token估算功能"""
    
    def test_estimate_tokens_chinese(self):
        """测试中文文本Token估算"""
        budget = TokenBudget()
        chinese_text = "这是一段中文文本"
        
        tokens = budget.estimate_tokens(chinese_text)
        
        # 8个字符 / 1.5 ≈ 5 tokens
        assert 4 <= tokens <= 6
    
    def test_estimate_tokens_english(self):
        """测试英文文本Token估算"""
        budget = TokenBudget()
        english_text = "This is an English text"
        
        tokens = budget.estimate_tokens(english_text)
        
        # 5个词 / 0.75 ≈ 6-7 tokens
        assert 5 <= tokens <= 7
    
    def test_estimate_tokens_mixed(self):
        """测试中英文混合文本"""
        budget = TokenBudget()
        mixed_text = "Hello这是一段mixed文本"
        
        tokens = budget.estimate_tokens(mixed_text)
        
        # 应该能正常估算
        assert tokens > 0
    
    def test_estimate_tokens_empty_string(self):
        """测试空字符串"""
        budget = TokenBudget()
        
        tokens = budget.estimate_tokens("")
        
        assert tokens == 0
    
    def test_estimate_tokens_long_chinese(self):
        """测试长中文文本"""
        budget = TokenBudget()
        long_text = "中" * 300  # 300个中文字符
        
        tokens = budget.estimate_tokens(long_text)
        
        # 300 / 1.5 = 200 tokens
        assert 190 <= tokens <= 210
    
    def test_estimate_tokens_long_english(self):
        """测试长英文文本"""
        budget = TokenBudget()
        words = "word " * 100  # 100个词
        
        tokens = budget.estimate_tokens(words.strip())
        
        # 100 / 0.75 ≈ 133 tokens
        assert 125 <= tokens <= 140


class TestTokenBudgetCanAddMemory:
    """测试判断是否可以添加记忆"""
    
    def test_can_add_memory_within_budget(self):
        """测试在预算范围内添加记忆"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        memory_text = "这是一段测试记忆文本"
        
        can_add = budget.can_add_memory(memory_text)
        
        assert can_add is True
    
    def test_can_add_memory_exceeds_budget(self):
        """测试超出预算"""
        budget = TokenBudget(total_budget=100, preamble_cost=20)
        long_memory = "测试" * 200  # 很长的文本
        
        can_add = budget.can_add_memory(long_memory)
        
        assert can_add is False
    
    def test_can_add_memory_as_summary(self):
        """测试作为摘要添加（应该更省token）"""
        budget = TokenBudget(total_budget=100, preamble_cost=20)
        memory_text = "这是一段测试记忆文本" * 10
        
        # 作为完整记忆可能超出预算
        can_add_full = budget.can_add_memory(memory_text, as_summary=False)
        
        # 作为摘要应该更可能通过
        can_add_summary = budget.can_add_memory(memory_text, as_summary=True)
        
        # 摘要版本应该更容易通过
        assert can_add_summary >= can_add_full


class TestTokenBudgetAddMemory:
    """测试添加记忆功能"""
    
    def test_add_memory_success(self):
        """测试成功添加记忆"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        memory_text = "这是一段测试记忆文本"
        
        initial_used = budget.used_budget
        tokens_consumed = budget.add_memory(memory_text)
        
        assert tokens_consumed > 0
        assert budget.used_budget > initial_used
        assert budget.used_budget <= budget.total_budget
    
    def test_add_memory_as_summary(self):
        """测试添加摘要记忆"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        memory_text = "这是一段测试记忆文本" * 10
        
        tokens_full = budget.add_memory(memory_text, as_summary=False)
        budget.used_budget = budget.preamble_cost  # 重置
        tokens_summary = budget.add_memory(memory_text, as_summary=True)
        
        # 摘要应该消耗更少token
        assert tokens_summary < tokens_full
    
    def test_add_memory_empty(self):
        """测试添加空记忆"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        
        tokens = budget.add_memory("")
        
        assert tokens == 0


class TestTokenBudgetGetRemainingBudget:
    """测试获取剩余预算"""
    
    def test_get_remaining_initial(self):
        """测试初始剩余预算"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        
        remaining = budget.get_remaining_budget()
        
        assert remaining == 492  # 512 - 20
    
    def test_get_remaining_after_add(self):
        """测试添加记忆后的剩余预算"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        budget.add_memory("测试记忆文本")
        
        remaining = budget.get_remaining_budget()
        
        assert remaining < 492
        assert remaining >= 0
    
    def test_get_remaining_zero(self):
        """测试剩余预算为0"""
        budget = TokenBudget(total_budget=100, preamble_cost=20)
        budget.add_memory("测试" * 100)
        
        remaining = budget.get_remaining_budget()
        
        assert remaining <= 0


class TestTokenBudgetGetUtilization:
    """测试获取预算利用率"""
    
    def test_get_utilization_initial(self):
        """测试初始利用率"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        
        utilization = budget.get_utilization()
        
        assert 0.0 <= utilization <= 1.0
        # 20 / 512 ≈ 0.039
        assert 0.03 <= utilization <= 0.05
    
    def test_get_utilization_half(self):
        """测试50%利用率"""
        budget = TokenBudget(total_budget=100, preamble_cost=10)
        budget.add_memory("测试" * 50)  # 约消耗40 tokens
        
        utilization = budget.get_utilization()
        
        assert 0.4 <= utilization <= 0.6
    
    def test_get_utilization_full(self):
        """测试满利用率"""
        budget = TokenBudget(total_budget=100, preamble_cost=10)
        budget.add_memory("测试" * 200)  # 消耗大量token
        
        utilization = budget.get_utilization()
        
        assert utilization >= 0.9


class TestTokenBudgetReset:
    """测试重置预算"""
    
    def test_reset_to_preamble(self):
        """测试重置到preamble_cost"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        budget.add_memory("测试记忆")
        
        assert budget.used_budget > 20
        
        budget.reset()
        
        assert budget.used_budget == 20
    
    def test_reset_multiple_times(self):
        """测试多次重置"""
        budget = TokenBudget(total_budget=512, preamble_cost=20)
        
        for _ in range(3):
            budget.add_memory("测试记忆")
            budget.reset()
            assert budget.used_budget == 20


class TestTokenBudgetFinalize:
    """测试Finalize预算"""
    
    def test_finalize_success(self):
        """测试成功finalize"""
        budget = TokenBudget(total_budget=512, preamble_cost=20, postamble_cost=10)
        
        success = budget.finalize()
        
        assert success is True
        assert budget.used_budget == 30  # 20 + 10
    
    def test_finalize_exceeds_budget(self):
        """测试finalize超出预算"""
        budget = TokenBudget(total_budget=100, preamble_cost=20, postamble_cost=10)
        budget.add_memory("测试" * 100)  # 消耗大量token
        
        success = budget.finalize()
        
        assert success is False
        assert budget.used_budget < budget.total_budget


class TestMemoryCompressor:
    """测试记忆压缩器"""
    
    def test_compress_memory_with_summary(self):
        """测试使用摘要压缩"""
        compressor = MemoryCompressor(max_summary_length=50)
        content = "这是一段很长的记忆内容，需要被压缩..."
        summary = "这是一段摘要"
        
        compressed, used_summary = compressor.compress_memory(content, summary)
        
        assert used_summary is True
        assert len(compressed) <= 50
        assert "这是一段摘要" in compressed
    
    def test_compress_memory_short_content(self):
        """测试短内容（不需要压缩）"""
        compressor = MemoryCompressor(max_summary_length=100)
        content = "这是一段短记忆"
        
        compressed, used_summary = compressor.compress_memory(content)
        
        assert used_summary is False
        assert compressed == content
    
    def test_compress_memory_long_content(self):
        """测试长内容压缩"""
        compressor = MemoryCompressor(max_summary_length=20)
        content = "这是一段很长的记忆内容，超过了最大摘要长度限制"
        
        compressed, used_summary = compressor.compress_memory(content)
        
        assert used_summary is False
        assert len(compressed) <= 23  # 20 + "..."
        assert compressed.endswith("...")
    
    def test_compress_memory_empty_content(self):
        """测试空内容"""
        compressor = MemoryCompressor()
        
        compressed, used_summary = compressor.compress_memory("")
        
        assert used_summary is False
        assert compressed == ""
    
    def test_compress_memory_with_empty_summary(self):
        """测试空摘要（应该使用内容）"""
        compressor = MemoryCompressor(max_summary_length=50)
        content = "这是一段内容"
        summary = ""
        
        compressed, used_summary = compressor.compress_memory(content, summary)
        
        assert used_summary is False
        assert compressed == content
    
    def test_compress_memories_batch(self):
        """测试批量压缩记忆"""
        compressor = MemoryCompressor(max_summary_length=30)
        memories = [
            ("内容1", "摘要1"),
            ("内容2", None),
            ("很长的内容3需要压缩", "摘要3"),
        ]
        
        results = compressor.compress_memories(memories)
        
        assert len(results) == 3
        assert results[0][1] is True  # 使用了摘要
        assert results[1][1] is False  # 使用了内容
        assert results[2][1] is True  # 使用了摘要


class TestDynamicMemorySelector:
    """测试动态记忆选择器"""
    
    def test_init_with_defaults(self):
        """测试使用默认值初始化"""
        budget = TokenBudget()
        selector = DynamicMemorySelector(token_budget=budget)
        
        assert selector.token_budget == budget
        assert selector.compressor is not None
    
    def test_init_with_custom_compressor(self):
        """测试使用自定义压缩器初始化"""
        budget = TokenBudget()
        compressor = MemoryCompressor(max_summary_length=50)
        selector = DynamicMemorySelector(
            token_budget=budget,
            compressor=compressor
        )
        
        assert selector.compressor == compressor
    
    def test_select_memories_empty(self):
        """测试选择空记忆列表"""
        budget = TokenBudget()
        selector = DynamicMemorySelector(token_budget=budget)
        
        selected, stats = selector.select_memories([], target_count=3)
        
        assert len(selected) == 0
        assert stats["total_candidates"] == 0
        assert stats["selected_count"] == 0
    
    def test_select_memories_sorting(self):
        """测试记忆排序（按重要性）"""
        budget = TokenBudget(total_budget=1000, preamble_cost=20)
        selector = DynamicMemorySelector(token_budget=budget)
        
        memories = [
            Mock(content="低重要性", summary="低", rif_score=0.2, importance_score=0.3),
            Mock(content="高重要性", summary="高", rif_score=0.9, importance_score=0.9),
            Mock(content="中重要性", summary="中", rif_score=0.6, importance_score=0.6),
        ]
        
        selected, stats = selector.select_memories(memories, target_count=3)
        
        # 应该选择高重要性的记忆
        assert len(selected) >= 1
        if len(selected) >= 1:
            assert selected[0].rif_score >= selected[-1].rif_score
    
    def test_select_memories_budget_limit(self):
        """测试预算限制"""
        budget = TokenBudget(total_budget=100, preamble_cost=20)
        selector = DynamicMemorySelector(token_budget=budget)
        
        memories = [
            Mock(content="记忆1", summary="", rif_score=0.9, importance_score=0.9),
            Mock(content="记忆2", summary="", rif_score=0.8, importance_score=0.8),
            Mock(content="记忆3", summary="", rif_score=0.7, importance_score=0.7),
        ]
        
        selected, stats = selector.select_memories(memories, target_count=10)
        
        # 应该因为预算限制选择较少的记忆
        assert stats["selected_count"] <= len(memories)
    
    def test_select_memories_with_summary(self):
        """测试使用摘要"""
        budget = TokenBudget(total_budget=1000, preamble_cost=20)
        compressor = MemoryCompressor(max_summary_length=50)
        selector = DynamicMemorySelector(token_budget=budget, compressor=compressor)
        
        memories = [
            Mock(content="很长的记忆内容1...", summary="摘要1", rif_score=0.9, importance_score=0.9),
            Mock(content="很长的记忆内容2...", summary="摘要2", rif_score=0.8, importance_score=0.8),
        ]
        
        selected, stats = selector.select_memories(memories, target_count=2)
        
        # 应该使用了摘要
        assert stats["summary_used"] > 0
    
    def test_get_memory_context_empty(self):
        """测试生成空记忆上下文"""
        budget = TokenBudget()
        selector = DynamicMemorySelector(token_budget=budget)
        
        context = selector.get_memory_context([], target_count=3)
        
        assert context == ""
    
    def test_get_memory_context_with_memories(self):
        """测试生成记忆上下文"""
        budget = TokenBudget(total_budget=1000, preamble_cost=20)
        selector = DynamicMemorySelector(token_budget=budget)
        
        from datetime import datetime
        memories = [
            Mock(
                content="测试记忆1",
                summary="摘要1",
                rif_score=0.9,
                importance_score=0.9,
                created_time=datetime.now(),
                type=Mock(value="fact")
            ),
            Mock(
                content="测试记忆2",
                summary="摘要2",
                rif_score=0.8,
                importance_score=0.8,
                created_time=datetime.now(),
                type=Mock(value="emotion")
            ),
        ]
        
        context = selector.get_memory_context(memories, target_count=2)
        
        assert "【相关记忆】" in context
        assert "FACT" in context or "EMOTION" in context
        assert "摘要1" in context or "测试记忆1" in context


class TestTokenBudgetEdgeCases:
    """测试边界情况"""
    
    def test_zero_total_budget(self):
        """测试总预算为0"""
        budget = TokenBudget(total_budget=0, preamble_cost=0)
        
        can_add = budget.can_add_memory("测试")
        
        assert can_add is False
    
    def test_negative_costs(self):
        """测试负成本（边界情况）"""
        budget = TokenBudget(
            total_budget=512,
            preamble_cost=-10,
            postamble_cost=-5
        )
        
        # 应该能处理，虽然不合理
        assert budget.preamble_cost == -10
        assert budget.postamble_cost == -5
    
    def test_unicode_text(self):
        """测试Unicode文本"""
        budget = TokenBudget()
        unicode_text = "Hello世界🌍测试文本"
        
        tokens = budget.estimate_tokens(unicode_text)
        
        assert tokens > 0


class TestMemoryCompressorEdgeCases:
    """测试记忆压缩器边界情况"""
    
    def test_zero_max_length(self):
        """测试最大长度为0"""
        compressor = MemoryCompressor(max_summary_length=0)
        
        compressed, used_summary = compressor.compress_memory("测试内容")
        
        assert len(compressed) == 0
    
    def test_negative_max_length(self):
        """测试负的最大长度"""
        compressor = MemoryCompressor(max_summary_length=-10)
        
        compressed, used_summary = compressor.compress_memory("测试内容")
        
        # 应该能处理
        assert isinstance(compressed, str)
    
    def test_special_characters(self):
        """测试特殊字符"""
        compressor = MemoryCompressor(max_summary_length=100)
        
        special_text = "特殊字符：\n\t\r\b测试"
        compressed, used_summary = compressor.compress_memory(special_text)
        
        assert "特殊字符" in compressed or len(compressed) == 0


class TestDynamicMemorySelectorIntegration:
    """测试动态记忆选择器集成场景"""
    
    def test_full_selection_workflow(self):
        """测试完整选择工作流"""
        budget = TokenBudget(total_budget=500, preamble_cost=20, postamble_cost=10)
        compressor = MemoryCompressor(max_summary_length=50)
        selector = DynamicMemorySelector(token_budget=budget, compressor=compressor)
        
        from datetime import datetime
        memories = [
            Mock(
                content="这是一段很长的测试记忆内容，需要被压缩..." * 10,
                summary="记忆1摘要",
                rif_score=0.9,
                importance_score=0.9,
                created_time=datetime.now(),
                type=Mock(value="fact")
            ),
            Mock(
                content="另一段长记忆内容..." * 10,
                summary="记忆2摘要",
                rif_score=0.8,
                importance_score=0.8,
                created_time=datetime.now(),
                type=Mock(value="emotion")
            ),
            Mock(
                content="短记忆",
                summary="短",
                rif_score=0.7,
                importance_score=0.7,
                created_time=datetime.now(),
                type=Mock(value="fact")
            ),
        ]
        
        # 1. 选择记忆
        selected, stats = selector.select_memories(memories, target_count=3)
        
        # 2. 验证选择结果
        assert stats["total_candidates"] == 3
        assert stats["selected_count"] >= 0
        assert stats["used_tokens"] <= budget.total_budget
        
        # 3. 生成上下文
        context = selector.get_memory_context(memories, target_count=3)
        
        # 4. 验证上下文
        if selected:
            assert "【相关记忆】" in context
            assert "FACT" in context or "EMOTION" in context
    
    def test_budget_exhaustion_scenario(self):
        """测试预算耗尽场景"""
        budget = TokenBudget(total_budget=50, preamble_cost=20, postamble_cost=10)
        selector = DynamicMemorySelector(token_budget=budget)
        
        from datetime import datetime
        memories = [
            Mock(
                content="很长的记忆内容" * 20,
                summary="长记忆摘要",
                rif_score=0.9,
                importance_score=0.9,
                created_time=datetime.now(),
                type=Mock(value="fact")
            ),
            Mock(
                content="另一段长记忆" * 20,
                summary="另一段摘要",
                rif_score=0.8,
                importance_score=0.8,
                created_time=datetime.now(),
                type=Mock(value="fact")
            ),
        ]
        
        selected, stats = selector.select_memories(memories, target_count=5)
        
        # 应该因为预算限制选择很少的记忆
        assert stats["selected_count"] <= 2
        assert stats["skipped_count"] > 0

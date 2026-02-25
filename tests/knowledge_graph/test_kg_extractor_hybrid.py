"""
KGExtractor Hybrid 模式单元测试

测试条件触发逻辑、统计追踪、每日限制等 hybrid 增强功能。
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from iris_memory.knowledge_graph.kg_extractor import (
    KGExtractor,
    RULE_TEXT_MAX_LENGTH,
    _QUICK_FILTER_KEYWORDS,
    _RELATIONSHIP_SIGNAL_KEYWORDS,
)
from iris_memory.knowledge_graph.kg_models import (
    KGNodeType,
    KGRelationType,
    KGTriple,
)
from iris_memory.knowledge_graph.kg_storage import KGStorage


def run(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture
def storage():
    """创建 KGStorage"""
    s = KGStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_kg.db"
        run(s.initialize(db_path))
        yield s
        run(s.close())


@pytest.fixture
def hybrid_extractor(storage):
    """创建 hybrid 模式的 KGExtractor"""
    ext = KGExtractor(
        storage=storage,
        mode="hybrid",
        daily_limit=10,
    )
    return ext


@pytest.fixture
def rule_extractor(storage):
    """创建 rule 模式的 KGExtractor"""
    return KGExtractor(storage=storage, mode="rule")


# ==============================================================
# _should_call_llm_hybrid 决策逻辑测试
# ==============================================================

class TestShouldCallLLM:
    """测试 hybrid 模式的 LLM 调用决策"""

    def test_sufficient_high_conf_rules_skip_llm(self, hybrid_extractor):
        """规则已提取到 >=2 个高置信度三元组时跳过 LLM"""
        triples = [
            KGTriple(
                subject="张三", predicate="喜欢", object="编程",
                subject_type=KGNodeType.PERSON, object_type=KGNodeType.CONCEPT,
                relation_type=KGRelationType.LIKES, confidence=0.7,
            ),
            KGTriple(
                subject="张三", predicate="住在", object="北京",
                subject_type=KGNodeType.PERSON, object_type=KGNodeType.LOCATION,
                relation_type=KGRelationType.LIVES_IN, confidence=0.7,
            ),
        ]
        should_call, reason = hybrid_extractor._should_call_llm_hybrid(
            "张三喜欢编程，住在北京", triples
        )
        assert should_call is False
        assert reason == "rule_sufficient"

    def test_text_too_long_triggers_llm(self, hybrid_extractor):
        """超长文本触发 LLM"""
        long_text = "这是一段很长的文本。" * 300  # > 2000 chars
        should_call, reason = hybrid_extractor._should_call_llm_hybrid(
            long_text, []
        )
        assert should_call is True
        assert reason == "text_too_long_for_rules"

    def test_no_rules_but_relationship_signals(self, hybrid_extractor):
        """规则未提取但有关系信号词时触发 LLM"""
        text = "小明毕业于清华大学，现在担任技术总监"
        should_call, reason = hybrid_extractor._should_call_llm_hybrid(text, [])
        assert should_call is True
        assert reason == "relationship_signals_detected"

    def test_low_confidence_rules_trigger_llm(self, hybrid_extractor):
        """全部低置信度三元组时触发 LLM"""
        triples = [
            KGTriple(
                subject="某人", predicate="提到了", object="某物",
                subject_type=KGNodeType.UNKNOWN, object_type=KGNodeType.UNKNOWN,
                relation_type=KGRelationType.RELATED_TO, confidence=0.3,
            ),
        ]
        should_call, reason = hybrid_extractor._should_call_llm_hybrid(
            "某人提到了某物", triples
        )
        assert should_call is True
        assert reason == "low_confidence_rules"

    def test_partial_rules_with_signals(self, hybrid_extractor):
        """1 个高置信度结果 + 关系信号 → 触发 LLM 补充"""
        triples = [
            KGTriple(
                subject="张三", predicate="喜欢", object="编程",
                subject_type=KGNodeType.PERSON, object_type=KGNodeType.CONCEPT,
                relation_type=KGRelationType.LIKES, confidence=0.7,
            ),
        ]
        # 文本包含更多关系信号词
        text = "张三喜欢编程，他住在北京，是一名程序员"
        should_call, reason = hybrid_extractor._should_call_llm_hybrid(
            text, triples
        )
        assert should_call is True
        assert reason == "partial_rules_with_signals"

    def test_no_signal_no_rules(self, hybrid_extractor):
        """无关系信号且无规则结果 → 不调用"""
        text = "今天天气真好啊哈哈"
        should_call, reason = hybrid_extractor._should_call_llm_hybrid(text, [])
        assert should_call is False
        assert reason == "no_signal"

    def test_one_high_conf_no_signal(self, hybrid_extractor):
        """1 个高置信度结果但无额外信号 → 不调用"""
        triples = [
            KGTriple(
                subject="张三", predicate="喜欢", object="音乐",
                subject_type=KGNodeType.PERSON, object_type=KGNodeType.CONCEPT,
                relation_type=KGRelationType.LIKES, confidence=0.7,
            ),
        ]
        text = "ok"  # 无关系信号
        should_call, reason = hybrid_extractor._should_call_llm_hybrid(
            text, triples
        )
        assert should_call is False
        assert reason == "rule_acceptable"


# ==============================================================
# _has_relationship_signals 测试
# ==============================================================

class TestRelationshipSignals:
    """关系信号检测测试"""

    def test_quick_filter_keywords(self):
        """标准关系关键词检测"""
        assert KGExtractor._has_relationship_signals("他是我的朋友") is True
        assert KGExtractor._has_relationship_signals("我喜欢编程") is True
        assert KGExtractor._has_relationship_signals("住在上海") is True

    def test_extended_signal_keywords(self):
        """扩展信号关键词检测"""
        assert KGExtractor._has_relationship_signals("他担任经理") is True
        assert KGExtractor._has_relationship_signals("她毕业于北大") is True
        assert KGExtractor._has_relationship_signals("属于技术部门") is True
        assert KGExtractor._has_relationship_signals("he is a manager") is True

    def test_no_signals(self):
        """无关系信号"""
        assert KGExtractor._has_relationship_signals("今天天气好") is False
        assert KGExtractor._has_relationship_signals("哈哈哈") is False


# ==============================================================
# Stats 统计功能测试
# ==============================================================

class TestHybridStats:
    """Hybrid 模式统计功能"""

    def test_initial_stats(self, hybrid_extractor):
        """初始统计为零"""
        stats = hybrid_extractor.get_stats()
        assert stats["rule_extractions"] == 0
        assert stats["llm_extractions"] == 0
        assert stats["hybrid_decisions"] == 0
        assert stats["llm_skipped_sufficient"] == 0
        assert stats["mode"] == "hybrid"
        assert stats["daily_limit"] == 10

    def test_remaining_calls(self, hybrid_extractor):
        """每日余量"""
        assert hybrid_extractor.remaining_daily_calls == 10

    def test_stats_after_rule_extraction(self, hybrid_extractor):
        """规则提取后统计更新"""
        # 提取有足够高置信度结果的文本
        run(hybrid_extractor.extract_and_store(
            "张三和李四是朋友",
            user_id="u1",
            sender_name="测试",
        ))
        stats = hybrid_extractor.get_stats()
        assert stats["rule_extractions"] >= 1
        assert stats["hybrid_decisions"] >= 1

    def test_stats_no_signal_text(self, hybrid_extractor):
        """无信号文本不触发 LLM"""
        run(hybrid_extractor.extract_and_store(
            "今天天气不错呢",
            user_id="u1",
        ))
        stats = hybrid_extractor.get_stats()
        # 规则和 LLM 都不应被调用（文本太短/无关键词）
        assert stats["llm_extractions"] == 0


# ==============================================================
# Daily Limiter 测试
# ==============================================================

class TestDailyLimiter:
    """每日调用限制测试"""

    def test_daily_limit_initialized(self, hybrid_extractor):
        """每日限制正确初始化"""
        assert hybrid_extractor._limiter._daily_limit == 10

    def test_remaining_decrements(self, hybrid_extractor):
        """手动递减剩余次数"""
        initial = hybrid_extractor.remaining_daily_calls
        hybrid_extractor._limiter.increment()
        assert hybrid_extractor.remaining_daily_calls == initial - 1

    def test_rule_mode_no_limiter_usage(self, rule_extractor):
        """Rule 模式不使用 limiter"""
        run(rule_extractor.extract_and_store(
            "张三喜欢编程",
            user_id="u1",
            sender_name="张三",
        ))
        # Rule 模式默认 daily_limit=100，不应被消耗
        assert rule_extractor.remaining_daily_calls == 100


# ==============================================================
# extract_and_store Hybrid 集成测试
# ==============================================================

class TestHybridExtractAndStore:
    """Hybrid 模式 extract_and_store 集成"""

    def test_hybrid_rule_sufficient_skips_llm(self, hybrid_extractor):
        """规则充足时 hybrid 模式跳过 LLM"""
        # 这个文本应该被规则提取到多个三元组
        text = "张三和李四是朋友，张三住在北京"
        triples = run(hybrid_extractor.extract_and_store(
            text, user_id="u1", sender_name="测试",
        ))
        assert len(triples) >= 1
        stats = hybrid_extractor.get_stats()
        # LLM 不应被调用（规则已足够）
        assert stats["llm_extractions"] == 0

    def test_hybrid_short_text_returns_empty(self, hybrid_extractor):
        """过短文本直接返回空"""
        triples = run(hybrid_extractor.extract_and_store(
            "hi", user_id="u1",
        ))
        assert triples == []

    def test_hybrid_empty_text_returns_empty(self, hybrid_extractor):
        """空文本直接返回空"""
        triples = run(hybrid_extractor.extract_and_store(
            "", user_id="u1",
        ))
        assert triples == []

    def test_rule_mode_no_hybrid_logic(self, rule_extractor):
        """Rule 模式不进入 hybrid 决策逻辑"""
        text = "张三喜欢编程"
        triples = run(rule_extractor.extract_and_store(
            text, user_id="u1", sender_name="张三",
        ))
        stats = rule_extractor.get_stats()
        assert stats["hybrid_decisions"] == 0


# ==============================================================
# 构造函数测试
# ==============================================================

class TestInit:
    """构造函数参数"""

    def test_default_daily_limit(self, storage):
        """默认日限制"""
        ext = KGExtractor(storage=storage, mode="hybrid")
        assert ext._limiter._daily_limit == 100  # _DEFAULT_DAILY_LIMIT

    def test_custom_daily_limit(self, storage):
        """自定义日限制"""
        ext = KGExtractor(storage=storage, mode="hybrid", daily_limit=50)
        assert ext._limiter._daily_limit == 50

    def test_stats_dict_initialized(self, storage):
        """统计字典正确初始化"""
        ext = KGExtractor(storage=storage, mode="rule")
        assert isinstance(ext._stats, dict)
        assert "rule_extractions" in ext._stats
        assert "llm_extractions" in ext._stats

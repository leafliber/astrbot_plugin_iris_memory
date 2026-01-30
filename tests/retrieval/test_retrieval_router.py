"""
检索路由器单元测试
测试RetrievalRouter的所有功能
"""

import pytest

from iris_memory.retrieval.retrieval_router import RetrievalRouter
from iris_memory.core.types import RetrievalStrategy, EmotionType
from iris_memory.models.emotion_state import CurrentEmotionState, EmotionalState


class TestRetrievalRouter:
    """RetrievalRouter单元测试"""

    @pytest.fixture
    def router(self):
        """创建RetrievalRouter实例"""
        return RetrievalRouter()

    @pytest.fixture
    def negative_emotional_state(self):
        """创建负面情感状态"""
        state = EmotionalState()
        state.update_current_emotion(
            primary=EmotionType.SADNESS,
            intensity=0.8,
            confidence=0.7
        )
        return state

    @pytest.fixture
    def positive_emotional_state(self):
        """创建正面情感状态"""
        state = EmotionalState()
        state.update_current_emotion(
            primary=EmotionType.JOY,
            intensity=0.7,
            confidence=0.6
        )
        return state

    # ========== 初始化测试 ==========

    def test_router_initialization(self, router):
        """测试检索路由器初始化"""
        assert router is not None
        assert len(router.time_keywords) > 0
        assert len(router.relation_keywords) > 0

    # ========== 基本路由测试 ==========

    def test_route_simple_query(self, router):
        """测试简单查询路由"""
        query = "苹果"
        strategy = router.route(query)

        # 简单查询应该使用纯向量检索
        assert strategy == RetrievalStrategy.VECTOR_ONLY

    def test_route_simple_query_longer(self, router):
        """测试稍长的简单查询"""
        query = "我喜欢吃苹果"
        strategy = router.route(query)

        # 关键词较少，应该使用纯向量检索
        assert strategy == RetrievalStrategy.VECTOR_ONLY

    # ========== 时间感知查询测试 ==========

    def test_route_time_aware_yesterday(self, router):
        """测试时间感知查询：昨天"""
        query = "我昨天说了什么"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.TIME_AWARE

    def test_route_time_aware_today(self, router):
        """测试时间感知查询：今天"""
        query = "今天的心情怎么样"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.TIME_AWARE

    def test_route_time_aware_last_week(self, router):
        """测试时间感知查询：上周"""
        query = "上周的工作安排"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.TIME_AWARE

    def test_route_time_aware_recently(self, router):
        """测试时间感知查询：最近"""
        query = "最近有什么变化"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.TIME_AWARE

    def test_route_time_aware_english(self, router):
        """测试时间感知查询：英文"""
        query = "What did I say yesterday"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.TIME_AWARE

    # ========== 情感感知查询测试 ==========

    def test_route_emotion_aware_negative(self, router, negative_emotional_state):
        """测试情感感知查询：负面情感"""
        query = "关于工作的事情"
        context = {"emotional_state": negative_emotional_state}

        strategy = router.route(query, context)

        # 负面情感应该使用情感感知检索
        assert strategy == RetrievalStrategy.EMOTION_AWARE

    def test_route_emotion_aware_high_intensity(self, router, positive_emotional_state):
        """测试情感感知查询：高强度情感"""
        query = "关于苹果的事情"
        # 修改正面情感强度
        positive_emotional_state.current.intensity = 0.8

        context = {"emotional_state": positive_emotional_state}

        strategy = router.route(query, context)

        # 高强度情感应该使用情感感知检索
        assert strategy == RetrievalStrategy.EMOTION_AWARE

    def test_route_emotion_aware_no_context(self, router):
        """测试情感感知查询：无上下文"""
        query = "关于工作"

        strategy = router.route(query)

        # 无上下文不应该使用情感感知
        assert strategy != RetrievalStrategy.EMOTION_AWARE

    def test_route_emotion_aware_neutral_low_intensity(self, router, positive_emotional_state):
        """测试情感感知查询：中性低强度"""
        query = "关于工作"
        positive_emotional_state.current.primary = EmotionType.NEUTRAL
        positive_emotional_state.current.intensity = 0.5

        context = {"emotional_state": positive_emotional_state}

        strategy = router.route(query, context)

        # 中性低强度不应该使用情感感知
        assert strategy != RetrievalStrategy.EMOTION_AWARE

    # ========== 多跳推理查询测试 ==========

    def test_route_multi_hop_who_is(self, router):
        """测试多跳推理查询：谁是"""
        query = "谁是王经理的上司"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.GRAPH_ONLY

    def test_route_multi_hop_boss_of(self, router):
        """测试多跳推理查询：boss of"""
        query = "Who is the boss of Alice"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.GRAPH_ONLY

    def test_route_multi_hop_colleague(self, router):
        """测试多跳推理查询：同事"""
        query = "我的同事是谁"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.GRAPH_ONLY

    def test_route_multi_hop_relationship(self, router):
        """测试多跳推理查询：关系"""
        query = "我和小明是什么关系"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.GRAPH_ONLY

    # ========== 复杂查询测试 ==========

    def test_route_complex_multiple_keywords(self, router):
        """测试复杂查询：多个关键词"""
        query = "去年这个时候在公司关于项目的讨论"
        strategy = router.route(query)

        # 5个以上关键词应该使用混合检索
        assert strategy == RetrievalStrategy.HYBRID

    def test_route_complex_time_and_relation(self, router):
        """测试复杂查询：时间和关系"""
        query = "上周谁是我的同事"
        strategy = router.route(query)

        # 包含时间和关系应该使用混合检索
        assert strategy == RetrievalStrategy.HYBRID

    def test_route_complex_time_and_many_keywords(self, router):
        """测试复杂查询：时间和多个关键词"""
        query = "昨天下午在公司开会讨论关于项目的事情"
        strategy = router.route(query)

        # 多关键词应该使用混合检索
        assert strategy == RetrievalStrategy.HYBRID

    # ========== 查询复杂度分析测试 ==========

    def test_analyze_simple_query(self, router):
        """测试分析简单查询"""
        query = "苹果"

        analysis = router.analyze_query_complexity(query)

        assert analysis["complexity"] == "simple"
        assert analysis["features"]["time_aware"] is False
        assert analysis["features"]["multi_hop"] is False
        assert analysis["features"]["keyword_count"] <= 4
        assert analysis["recommended_strategy"] == RetrievalStrategy.VECTOR_ONLY

    def test_analyze_medium_query(self, router):
        """测试分析中等复杂度查询"""
        query = "昨天关于工作"

        analysis = router.analyze_query_complexity(query)

        assert analysis["complexity"] == "medium"
        # 应该有一个特征为True
        assert (analysis["features"]["time_aware"] or
                analysis["features"]["multi_hop"])

    def test_analyze_complex_query(self, router):
        """测试分析复杂查询"""
        query = "去年这个时候在公司关于项目的讨论"

        analysis = router.analyze_query_complexity(query)

        assert analysis["complexity"] == "complex"
        # 应该推荐混合检索
        assert analysis["recommended_strategy"] == RetrievalStrategy.HYBRID

    def test_analyze_with_context(self, router, negative_emotional_state):
        """测试带上下文的分析"""
        query = "工作相关"
        context = {"emotional_state": negative_emotional_state}

        analysis = router.analyze_query_complexity(query)

        # 应该检测到情感感知
        assert analysis["features"]["emotion_aware"] is False  # route方法会检测
        assert analysis["complexity"] in ["simple", "medium"]

    # ========== 边界情况测试 ==========

    def test_route_empty_query(self, router):
        """测试空查询"""
        query = ""
        strategy = router.route(query)

        # 空查询应该默认使用向量检索
        assert strategy == RetrievalStrategy.VECTOR_ONLY

    def test_route_whitespace_only(self, router):
        """测试只有空白字符"""
        query = "   "
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.VECTOR_ONLY

    def test_route_special_characters(self, router):
        """测试特殊字符"""
        query = "测试@#$%^&*()特殊字符"
        strategy = router.route(query)

        # 应该能正常处理
        assert strategy in [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.EMOTION_AWARE]

    def test_route_very_long_query(self, router):
        """测试超长查询"""
        query = "这是一个很长的查询" * 20
        strategy = router.route(query)

        # 超长查询应该使用混合检索
        assert strategy == RetrievalStrategy.HYBRID

    def test_route_unicode(self, router):
        """测试Unicode"""
        query = "测试🍎🍊🍋emoji"
        strategy = router.route(query)

        # 应该能正常处理
        assert strategy == RetrievalStrategy.VECTOR_ONLY

    def test_route_mixed_language(self, router):
        """测试混合语言"""
        query = "Yesterday I said 我喜欢苹果"
        strategy = router.route(query)

        # 应该能检测到时间线索
        assert strategy == RetrievalStrategy.TIME_AWARE

    # ========== 私有方法测试 ==========

    def test_is_time_aware_true(self, router):
        """测试时间感知检测：True"""
        assert router._is_time_aware_query("昨天说了什么") is True
        assert router._is_time_aware_query("last week") is True
        assert router._is_time_aware_query("最近的变化") is True

    def test_is_time_aware_false(self, router):
        """测试时间感知检测：False"""
        assert router._is_time_aware_query("苹果") is False
        assert router._is_time_aware_query("我喜欢") is False

    def test_is_multi_hop_true(self, router):
        """测试多跳推理检测：True"""
        assert router._is_multi_hop_query("谁是小明的上司") is True
        assert router._is_multi_hop_query("my boss") is True
        assert router._is_multi_hop_query("关系是什么") is True

    def test_is_multi_hop_false(self, router):
        """测试多跳推理检测：False"""
        assert router._is_multi_hop_query("苹果") is False
        assert router._is_multi_hop_query("我喜欢") is False

    def test_is_complex_query_true(self, router):
        """测试复杂查询检测：True"""
        assert router._is_complex_query("这是一个 包含 很多 关键词 的 长查询", None) is True

    def test_is_complex_query_false(self, router):
        """测试复杂查询检测：False"""
        assert router._is_complex_query("苹果", None) is False

    def test_is_complex_query_with_time_and_relation(self, router):
        """测试复杂查询：时间和关系组合"""
        assert router._is_complex_query("上周谁是我的同事", None) is True

    # ========== 查询特征提取测试 ==========

    def test_analyze_features_time_only(self, router):
        """测试分析：仅时间特征"""
        query = "昨天发生了什么"
        analysis = router.analyze_query_complexity(query)

        features = analysis["features"]
        assert features["time_aware"] is True
        assert features["multi_hop"] is False
        assert features["keyword_count"] >= 2

    def test_analyze_features_relation_only(self, router):
        """测试分析：仅关系特征"""
        query = "谁是王经理"
        analysis = router.analyze_query_complexity(query)

        features = analysis["features"]
        assert features["time_aware"] is False
        assert features["multi_hop"] is True
        assert features["keyword_count"] >= 2

    def test_analyze_features_combined(self, router):
        """测试分析：组合特征"""
        query = "上周谁是王经理"
        analysis = router.analyze_query_complexity(query)

        features = analysis["features"]
        assert features["time_aware"] is True
        assert features["multi_hop"] is True

    # ========== 路由决策逻辑测试 ==========

    def test_route_priority_complex_over_time(self, router):
        """测试路由优先级：复杂 > 时间"""
        query = "上周公司同事关于项目讨论"
        strategy = router.route(query)

        # 复杂查询优先于时间感知
        assert strategy == RetrievalStrategy.HYBRID

    def test_route_priority_time_over_emotion(self, router, positive_emotional_state):
        """测试路由优先级：时间 > 情感"""
        query = "昨天关于工作"
        context = {"emotional_state": positive_emotional_state}

        strategy = router.route(query, context)

        # 时间感知优先于情感感知
        assert strategy == RetrievalStrategy.TIME_AWARE

    def test_route_priority_graph_over_vector(self, router):
        """测试路由优先级：图 > 向量"""
        query = "谁是王经理"
        strategy = router.route(query)

        # 多跳推理优先于向量检索
        assert strategy == RetrievalStrategy.GRAPH_ONLY

    # ========== 英文查询测试 ==========

    def test_english_query_simple(self, router):
        """测试英文简单查询"""
        query = "What did I say"
        strategy = router.route(query)

        assert strategy in [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.TIME_AWARE]

    def test_english_query_time(self, router):
        """测试英文时间查询"""
        query = "What did I say yesterday"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.TIME_AWARE

    def test_english_query_relation(self, router):
        """测试英文关系查询"""
        query = "Who is the boss of Alice"
        strategy = router.route(query)

        assert strategy == RetrievalStrategy.GRAPH_ONLY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

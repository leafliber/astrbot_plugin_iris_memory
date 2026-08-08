"""暗语学习与采集测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.learning.collector import LearningCollector, clean_text
from iris_memory.learning.jargon import INFER_THRESHOLDS, JargonLearner
from iris_memory.learning.reviewer import LearningReviewer


def _make_learner(storage):
    """构造 JargonLearner + 依赖的 collector/reviewer"""
    jargon = JargonLearner(storage)
    reviewer = LearningReviewer(storage)
    collector = LearningCollector(storage, jargon, reviewer)
    return jargon, collector


class TestCleanText:
    """图片占位符剥离"""

    def test_strip_image_placeholder(self):
        assert clean_text("看这个[图:一只猫]怎么样") == "看这个怎么样"
        assert clean_text("[IMG:photo.jpg]") == ""

    def test_plain_text_unchanged(self):
        assert clean_text("  你好呀  ") == "你好呀"

    def test_none_safe(self):
        assert clean_text(None) == ""


class TestCollectorOnMessage:
    """消息采集过滤"""

    def _event(self, self_id="bot"):
        event = MagicMock()
        event.get_self_id.return_value = self_id
        return event

    def test_self_message_filtered(self, storage):
        jargon, collector = _make_learner(storage)
        with patch(
            "iris_memory.learning.collector.get_adapter"
        ) as mock_adapter:
            mock_adapter.return_value.get_group_id.return_value = "g1"
            collector.on_message(self._event("u1"), "sess1", "u1", "绝绝子")
        assert jargon._counts == {}

    def test_normal_message_counted(self, storage):
        jargon, collector = _make_learner(storage)
        with patch(
            "iris_memory.learning.collector.get_adapter"
        ) as mock_adapter:
            mock_adapter.return_value.get_group_id.return_value = "g1"
            collector.on_message(self._event(), "sess1", "u1", "绝绝子")
        assert any(term == "绝绝子" for (_, term) in jargon._counts)

    def test_image_only_message_skipped(self, storage):
        jargon, collector = _make_learner(storage)
        with patch(
            "iris_memory.learning.collector.get_adapter"
        ) as mock_adapter:
            mock_adapter.return_value.get_group_id.return_value = "g1"
            collector.on_message(self._event(), "sess1", "u1", "[图:一只猫]")
        assert jargon._counts == {}


class TestNGramCounts:
    """n-gram 计数与停用词"""

    def test_ngram_terms_counted(self, storage):
        jargon, _ = _make_learner(storage)
        jargon.update_counts("g1", "绝绝子")
        terms = {term for (_, term) in jargon._counts}
        # 2/3/4 字滑窗
        assert "绝绝" in terms
        assert "绝子" in terms
        assert "绝绝子" in terms

    def test_stop_words_filtered(self, storage):
        jargon, _ = _make_learner(storage)
        jargon.update_counts("g1", "我们")
        terms = {term for (_, term) in jargon._counts}
        assert "我们" not in terms

    def test_pure_tone_noise_filtered(self, storage):
        jargon, _ = _make_learner(storage)
        jargon.update_counts("g1", "了吗")
        terms = {term for (_, term) in jargon._counts}
        assert "了吗" not in terms

    def test_group_isolation(self, storage):
        jargon, _ = _make_learner(storage)
        jargon.update_counts("g1", "绝绝子")
        jargon.update_counts("g2", "绝绝子")
        assert jargon._counts[("g1", "绝绝子")] == 1
        assert jargon._counts[("g2", "绝绝子")] == 1


class TestFlush:
    """增量刷盘"""

    def test_flush_delta_not_losing_counts(self, storage):
        jargon, _ = _make_learner(storage)
        jargon.update_counts("g1", "绝绝子")
        jargon.flush()
        assert storage.upsert_jargon_count("g1", "绝绝子", 0) == 1
        # 再次计数后刷盘，只写增量
        jargon.update_counts("g1", "绝绝子")
        jargon.flush()
        assert storage.upsert_jargon_count("g1", "绝绝子", 0) == 2

    def test_flush_empty_noop(self, storage):
        jargon, _ = _make_learner(storage)
        jargon.flush()  # 无脏数据不报错

    def test_load_counts(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 5)
        jargon, _ = _make_learner(storage)
        jargon.load_counts()
        assert jargon._counts[("g1", "yyds")] == 5


class TestInferTier:
    """阈值档位判定"""

    def test_never_inferred_included(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 3)
        jargon, _ = _make_learner(storage)
        terms = jargon.get_terms_for_inference()
        assert [t["term"] for t in terms] == ["yyds"]

    def test_below_min_threshold_excluded(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 2)
        jargon, _ = _make_learner(storage)
        assert jargon.get_terms_for_inference() == []

    def test_same_tier_not_reinferred(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 4)
        term = storage.get_jargon_terms_for_inference(INFER_THRESHOLDS)[0]
        storage.mark_jargon_inferred(term["id"], "永远的神", 0.9)
        jargon, _ = _make_learner(storage)
        jargon._inferred_tier[("g1", "yyds")] = 3  # 上次推断档位为 3
        # count=4 仍在 3 档，不重推
        assert jargon.get_terms_for_inference() == []

    def test_cross_tier_reinferred(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 6)
        term = storage.get_jargon_terms_for_inference(INFER_THRESHOLDS)[0]
        storage.mark_jargon_inferred(term["id"], "永远的神", 0.9)
        jargon, _ = _make_learner(storage)
        jargon._inferred_tier[("g1", "yyds")] = 3  # 上次推断档位为 3
        # count=6 跨入 6 档，需重推
        terms = jargon.get_terms_for_inference()
        assert [t["term"] for t in terms] == ["yyds"]

    def test_restart_initializes_tier_from_count(self, storage):
        """重启后按当前 count 初始化推断档位，不立即重推"""
        storage.upsert_jargon_count("g1", "yyds", 6)
        term = storage.get_jargon_terms_for_inference(INFER_THRESHOLDS)[0]
        storage.mark_jargon_inferred(term["id"], "永远的神", 0.9)
        jargon, _ = _make_learner(storage)
        jargon.load_counts()
        assert jargon._inferred_tier[("g1", "yyds")] == 6
        assert jargon.get_terms_for_inference() == []


class TestInfer:
    """LLM 含义推断（mock）"""

    def _term_info(self, storage, term="yyds", count=5):
        storage.upsert_jargon_count("g1", term, count)
        return storage.get_jargon_terms_for_inference(INFER_THRESHOLDS)[0]

    @pytest.mark.asyncio
    async def test_infer_success(self, config, storage):
        jargon, _ = _make_learner(storage)
        info = self._term_info(storage)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value='{"meaning": "永远的神", "confidence": 0.9}'
        )
        ok = await jargon.infer(info, llm)
        assert ok == ("永远的神", 0.9)
        # 推断本身不落库（写库由组件层在锁内完成）
        rows = storage.list_by_group("jargon", "g1")
        assert rows[0]["meaning"] is None
        # 调用参数走 learning_review 模块
        _, kwargs = llm.generate_direct.call_args
        assert kwargs["module"] == "learning_review"

    @pytest.mark.asyncio
    async def test_infer_malformed_json(self, config, storage):
        jargon, _ = _make_learner(storage)
        info = self._term_info(storage)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(return_value="这不是 JSON")
        ok = await jargon.infer(info, llm)
        assert ok is None
        assert storage.list_by_group("jargon", "g1")[0]["meaning"] is None

    @pytest.mark.asyncio
    async def test_infer_low_confidence(self, config, storage):
        jargon, _ = _make_learner(storage)
        info = self._term_info(storage)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value='{"meaning": "不确定", "confidence": 0.3}'
        )
        ok = await jargon.infer(info, llm)
        assert ok is None
        assert storage.list_by_group("jargon", "g1")[0]["meaning"] is None

    @pytest.mark.asyncio
    async def test_infer_llm_exception(self, config, storage):
        jargon, _ = _make_learner(storage)
        info = self._term_info(storage)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(side_effect=RuntimeError("LLM 挂了"))
        ok = await jargon.infer(info, llm)
        assert ok is None

    @pytest.mark.asyncio
    async def test_infer_use_l1_context(self, config, storage):
        """jargon_infer_use_l1 开启：从 L1 取含词消息拼上下文"""
        config._user_config["learning"]["jargon_infer_use_l1"] = True
        jargon, _ = _make_learner(storage)
        info = self._term_info(storage)
        l1 = MagicMock()
        msg_hit = MagicMock()
        msg_hit.content = "他这波操作真的 yyds"
        msg_miss = MagicMock()
        msg_miss.content = "无关消息"
        l1.get_context.return_value = [msg_hit, msg_miss]
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value='{"meaning": "永远的神", "confidence": 0.9}'
        )
        ok = await jargon.infer(info, llm, l1_buffer=l1, session_id="g1")
        assert ok == ("永远的神", 0.9)
        prompt = llm.generate_direct.call_args.kwargs["prompt"]
        assert "他这波操作真的 yyds" in prompt
        assert "无关消息" not in prompt

    @pytest.mark.asyncio
    async def test_infer_without_l1(self, config, storage):
        """jargon_infer_use_l1 关闭：仅凭词条推断，不访问 L1"""
        config._user_config["learning"]["jargon_infer_use_l1"] = False
        jargon, _ = _make_learner(storage)
        info = self._term_info(storage)
        l1 = MagicMock()
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value='{"meaning": "永远的神", "confidence": 0.9}'
        )
        ok = await jargon.infer(info, llm, l1_buffer=l1, session_id="g1")
        assert ok == ("永远的神", 0.9)
        l1.get_context.assert_not_called()
        prompt = llm.generate_direct.call_args.kwargs["prompt"]
        assert "yyds" in prompt


class TestMatchTerms:
    """注入侧命中匹配"""

    def test_match_inferred_terms(self, storage):
        storage.upsert_jargon_count("g1", "yyds", 5)
        term = storage.get_jargon_terms_for_inference(INFER_THRESHOLDS)[0]
        storage.mark_jargon_inferred(term["id"], "永远的神", 0.9)
        jargon, _ = _make_learner(storage)
        hits = jargon.match_terms("g1", "这波 yyds")
        assert len(hits) == 1
        assert hits[0]["meaning"] == "永远的神"
        assert jargon.match_terms("g1", "没有命中") == []

    def test_match_max_items(self, storage):
        jargon, _ = _make_learner(storage)
        for i in range(8):
            storage.upsert_jargon_count("g1", f"词{i}呀", 5)
        for info in storage.get_jargon_terms_for_inference(INFER_THRESHOLDS):
            storage.mark_jargon_inferred(info["id"], "含义", 0.9)
        text = " ".join(f"词{i}呀" for i in range(8))
        assert len(jargon.match_terms("g1", text, max_items=5)) == 5

"""V2 暗语漏斗回归测试。"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.learning.collector import LearningCollector, clean_text
from iris_memory.learning.jargon import CandidateExtractor, JargonLearner
from iris_memory.learning.jargon.models import CandidateCluster
from iris_memory.learning.jargon.reviewer import JargonReviewer
from iris_memory.learning.reviewer import LearningReviewer


def _make(storage):
    jargon = JargonLearner(storage)
    return jargon, LearningCollector(storage, jargon, LearningReviewer(storage))


class TestExtractor:
    def test_command_is_skipped_before_slash_removed(self):
        assert CandidateExtractor.should_skip("/今日运势", is_group=True)

    def test_private_and_bot_are_skipped(self):
        assert CandidateExtractor.should_skip("绝绝子", is_group=False)
        assert CandidateExtractor.should_skip("绝绝子", is_group=True, is_bot=True)

    def test_repeated_character_terms_removed(self):
        result = CandidateExtractor().extract("哼" * 75)
        assert result.observations == []

    def test_ascii_kept_whole_without_fragments(self):
        terms = {o["term"] for o in CandidateExtractor().extract("这波 YYDS").observations}
        assert "yyds" in terms
        assert "yy" not in terms and "yyd" not in terms

    def test_terms_unique_inside_one_message(self):
        terms = [o["term"] for o in CandidateExtractor().extract("绝绝子绝绝子").observations]
        assert len(terms) == len(set(terms))


class TestCollector:
    def _event(self, self_id="bot"):
        event = MagicMock()
        event.get_self_id.return_value = self_id
        return event

    def test_self_message_filtered(self, config, storage):
        jargon, collector = _make(storage)
        with patch("iris_memory.learning.collector.get_adapter") as adapter:
            adapter.return_value.get_group_id.return_value = "g1"
            collector.on_message(self._event("u1"), "g1", "u1", "绝绝子")
        assert storage.get_jargon_candidate_snapshots(14) == []

    def test_command_and_repetition_create_no_candidate(self, config, storage):
        jargon, _ = _make(storage)
        assert jargon.record_message("g", "u", "/今日运势", is_group=True) == 0
        assert jargon.record_message("g", "u", "哼" * 75, is_group=True) == 0
        assert storage.get_jargon_candidate_snapshots(14) == []

    def test_same_message_counts_each_term_once(self, config, storage):
        jargon, _ = _make(storage)
        jargon.record_message("g", "u", "绝绝子绝绝子", is_group=True, now=time.time())
        target = next(s for s in storage.get_jargon_candidate_snapshots(14) if s["term"] == "绝绝子")
        assert target["message_count"] == 1

    def test_user_cooldown_blocks_repeat(self, config, storage):
        jargon, _ = _make(storage)
        now = time.time()
        jargon.record_message("g", "u", "绝绝子", is_group=True, now=now)
        jargon.record_message("g", "u", "绝绝子", is_group=True, now=now + 60)
        target = next(s for s in storage.get_jargon_candidate_snapshots(14) if s["term"] == "绝绝子")
        assert target["message_count"] == 1


def _seed_eligible(jargon, text="绝绝子", start=None):
    start = start or time.time() - 8 * 3600
    for i, user in enumerate(("u1", "u2", "u1", "u2", "u1", "u2")):
        jargon.record_message("g", user, text, is_group=True, now=start + i * 1.6 * 3600)


class TestScoringAndReview:
    def test_single_sender_bot_template_never_queued(self, config, storage):
        config._hidden.set("learning_jargon_user_cooldown_minutes", 0)
        jargon, _ = _make(storage)
        start = time.time() - 10 * 3600
        for i in range(20):
            jargon.record_message("g", "other_bot", "内容仅供娱乐", is_group=True, now=start + i * 1800)
        assert jargon.prepare_review() == []

    def test_multi_user_candidate_is_batched(self, config, storage):
        config._hidden.set("learning_jargon_llm_trigger_size", 1)
        config._hidden.set("learning_jargon_llm_min_interval_hours", 0)
        jargon, _ = _make(storage)
        _seed_eligible(jargon)
        clusters = jargon.prepare_review()
        assert len(clusters) == 1
        assert "绝绝子" in clusters[0].terms
        assert clusters[0].review_token

    @pytest.mark.asyncio
    async def test_batch_reviewer_protocol(self):
        cluster = CandidateCluster("g:1", "g", [1], ["绝绝子"], "绝绝子", 6, 2, 8, 0.8)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(return_value=json.dumps({"items": [{
            "cluster_id": "g:1", "decision": "approve", "category": "slang",
            "canonical_term": "绝绝子", "aliases": [], "meaning": "非常棒",
            "confidence": 0.91, "reason": "多用户稳定使用",
        }]}))
        verdicts = await JargonReviewer().review([cluster], llm)
        assert verdicts and verdicts[0].decision == "approve"
        assert llm.generate_direct.call_count == 1
        assert llm.generate_direct.call_args.kwargs["module"] == "learning_jargon_review"

    def test_invalid_canonical_is_deferred(self):
        cluster = CandidateCluster("g:1", "g", [1], ["绝绝子"], "绝绝子", 6, 2, 8, 0.8)
        raw = json.dumps({"items": [{
            "cluster_id": "g:1", "decision": "approve", "category": "slang",
            "canonical_term": "模型发明词", "confidence": 0.99,
        }]})
        verdict = JargonReviewer.parse(raw, [cluster])[0]
        assert verdict.decision == "defer"
        assert verdict.canonical_term == "绝绝子"

    def test_substring_alias_is_rejected(self):
        cluster = CandidateCluster("g:1", "g", [1, 2], ["绝绝子", "绝绝"], "绝绝子", 6, 2, 8, 0.8)
        raw = json.dumps({"items": [{
            "cluster_id": "g:1", "decision": "approve", "category": "slang",
            "canonical_term": "绝绝子", "aliases": ["绝绝"], "meaning": "非常棒",
            "confidence": 0.9,
        }]})
        assert JargonReviewer.parse(raw, [cluster])[0].aliases == []

    def test_approve_promotes_formal_jargon(self, config, storage):
        config._hidden.set("learning_jargon_llm_trigger_size", 1)
        config._hidden.set("learning_jargon_llm_min_interval_hours", 0)
        jargon, _ = _make(storage)
        _seed_eligible(jargon)
        clusters = jargon.prepare_review()
        verdicts = JargonReviewer.parse(json.dumps({"items": [{
            "cluster_id": clusters[0].cluster_id, "decision": "approve",
            "category": "slang", "canonical_term": "绝绝子", "aliases": [],
            "meaning": "非常棒", "confidence": 0.91, "reason": "多用户使用",
        }]}), clusters)
        jargon.apply_verdicts(clusters, verdicts)
        active = storage.get_active_jargon("g")
        assert len(active) == 1 and active[0]["term"] == "绝绝子"

    def test_failed_llm_is_deferred_and_attempt_counted(self, config, storage):
        config._hidden.set("learning_jargon_llm_trigger_size", 1)
        config._hidden.set("learning_jargon_llm_min_interval_hours", 0)
        jargon, _ = _make(storage)
        _seed_eligible(jargon)
        clusters = jargon.prepare_review()
        jargon.apply_verdicts(clusters, None)
        rows = storage.get_jargon_candidate_snapshots(14)
        assert rows and all(r["state"] == "deferred" and r["llm_attempts"] == 1 for r in rows)


class TestMatcher:
    def test_longest_non_overlapping_and_alias_dedup(self, config, storage):
        now = time.time()
        storage._db.execute(
            "INSERT INTO jargon(group_id,term,aliases_json,meaning,confidence,status,category,"
            "evidence_count,approved_at,last_seen_at,created_at,updated_at)"
            " VALUES(?,?,?,?,?,'active','slang',?,?,?,?,?)",
            ("g", "绝绝子", '["绝绝"]', "非常棒", 0.9, 8, now, now, now, now),
        )
        storage._db.commit()
        jargon, _ = _make(storage)
        hits = jargon.match_terms("g", "今天真的绝绝子")
        assert len(hits) == 1 and hits[0]["term"] == "绝绝子"

    def test_dormant_term_reactivates_after_multi_user_evidence(self, config, storage):
        storage.insert_jargon("g", "绝绝子", "非常棒", 0.9)
        storage.set_jargon_status("g", "绝绝子", "dormant")
        jargon, _ = _make(storage)
        now = time.time()
        jargon.record_message("g", "u1", "这波绝绝子", is_group=True, now=now)
        jargon.record_message("g", "u2", "确实绝绝子", is_group=True, now=now + 60)
        assert storage.get_active_jargon("g") == []
        jargon.record_message("g", "u1", "还是绝绝子", is_group=True, now=now + 1900)
        assert storage.get_active_jargon("g")[0]["term"] == "绝绝子"


class TestCleanText:
    def test_image_placeholder(self):
        assert clean_text("看这个[图:猫]") == "看这个"

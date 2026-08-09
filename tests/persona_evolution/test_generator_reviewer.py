"""阶段 B 候选生成与阶段 C 独立审查测试"""

import json

import pytest

from iris_memory.persona_evolution.generator import (
    CandidateGenerator,
    parse_generation,
)
from iris_memory.persona_evolution.models import ErrorCode
from iris_memory.persona_evolution.reviewer import (
    DEFAULT_THRESHOLDS,
    PromptReviewer,
    parse_review,
    review_passed,
)

from .fakes import (
    GENERATION_MODULE,
    REVIEW_MODULE,
    FakeLLMManager,
    good_generation_json,
    good_review_json,
)


class TestParseGeneration:
    def test_good(self):
        result = parse_generation(json.loads(good_generation_json("候选")))
        assert result is not None
        assert result["candidate_prompt"] == "候选"
        assert result["change_summary"] == ["回复更短", "更常追问"]

    def test_missing_candidate(self):
        data = json.loads(good_generation_json("候选"))
        del data["candidate_prompt"]
        assert parse_generation(data) is None

    def test_empty_candidate(self):
        assert parse_generation(json.loads(good_generation_json("   "))) is None

    def test_bad_summary_type(self):
        data = json.loads(good_generation_json("候选"))
        data["change_summary"] = "不是列表"
        assert parse_generation(data) is None

    def test_bad_confidence(self):
        data = json.loads(good_generation_json("候选"))
        data["confidence"] = None
        assert parse_generation(data) is None


class TestCandidateGenerator:
    @pytest.mark.asyncio
    async def test_success(self):
        llm = FakeLLMManager()
        llm.set_default(GENERATION_MODULE, good_generation_json("新候选"))
        generator = CandidateGenerator()
        result, err = await generator.generate(
            llm,
            current_prompt="旧",
            edit_mode="managed_block",
            goal_snapshot={"text": "目标"},
            style_profile={"tone": []},
            protected_fragments=[],
        )
        assert err is None
        assert result["candidate_prompt"] == "新候选"
        assert llm.calls[0]["module"] == GENERATION_MODULE

    @pytest.mark.asyncio
    async def test_bad_json(self):
        llm = FakeLLMManager()
        llm.set_default(GENERATION_MODULE, "没有 JSON")
        generator = CandidateGenerator()
        _, err = await generator.generate(
            llm,
            current_prompt="旧",
            edit_mode="managed_block",
            goal_snapshot={},
            style_profile={},
            protected_fragments=[],
        )
        assert err == ErrorCode.GENERATION_PARSE_FAILED


class TestParseReview:
    def test_good(self):
        result = parse_review(json.loads(good_review_json()))
        assert result is not None and result["pass"] is True

    def test_score_out_of_range(self):
        assert parse_review(json.loads(good_review_json(privacy_safety=1.2))) is None

    def test_injection_not_bool(self):
        assert (
            parse_review(json.loads(good_review_json(prompt_injection_suspected="no")))
            is None
        )


class TestReviewPassed:
    def test_all_pass(self):
        assert review_passed(json.loads(good_review_json())) is True

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("identity_consistency", 0.79),
            ("constraint_preservation", 0.89),
            ("goal_alignment", 0.69),
            ("privacy_safety", 0.89),
        ],
    )
    def test_below_threshold_fails(self, field, bad_value):
        review = json.loads(good_review_json(**{field: bad_value}))
        assert review_passed(review) is False

    @pytest.mark.parametrize(
        "field",
        [
            "identity_consistency",
            "constraint_preservation",
            "goal_alignment",
            "privacy_safety",
        ],
    )
    def test_at_threshold_passes(self, field):
        review = json.loads(good_review_json(**{field: DEFAULT_THRESHOLDS[field]}))
        assert review_passed(review) is True

    def test_injection_suspected_fails(self):
        review = json.loads(good_review_json(prompt_injection_suspected=True))
        assert review_passed(review) is False

    def test_pass_false_fails(self):
        review = json.loads(good_review_json(**{"pass": False}))
        assert review_passed(review) is False


class TestPromptReviewer:
    @pytest.mark.asyncio
    async def test_success(self):
        llm = FakeLLMManager()
        llm.set_default(REVIEW_MODULE, good_review_json())
        reviewer = PromptReviewer()
        result, err = await reviewer.review(
            llm,
            base_prompt="旧",
            candidate_prompt="新",
            goal_snapshot={},
            style_profile={},
        )
        assert err is None and result["pass"] is True
        assert llm.calls[0]["module"] == REVIEW_MODULE

    @pytest.mark.asyncio
    async def test_review_failed_returns_result(self):
        llm = FakeLLMManager()
        llm.set_default(REVIEW_MODULE, good_review_json(identity_consistency=0.5))
        reviewer = PromptReviewer()
        result, err = await reviewer.review(
            llm, base_prompt="旧", candidate_prompt="新",
            goal_snapshot={}, style_profile={},
        )
        assert err == ErrorCode.REVIEW_FAILED
        assert result is not None  # 失败结果保留供 Revision 快照

    @pytest.mark.asyncio
    async def test_parse_failed(self):
        llm = FakeLLMManager()
        llm.set_default(REVIEW_MODULE, "不是 JSON")
        reviewer = PromptReviewer()
        result, err = await reviewer.review(
            llm, base_prompt="旧", candidate_prompt="新",
            goal_snapshot={}, style_profile={},
        )
        assert result is None
        assert err == ErrorCode.REVIEW_PARSE_FAILED

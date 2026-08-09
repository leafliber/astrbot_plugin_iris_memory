"""阶段 A 风格归纳测试：JSON 容错与置信度停轮"""

import asyncio

import pytest

from iris_memory.persona_evolution.analyzer import (
    StyleAnalyzer,
    extract_json_object,
    parse_style_profile,
)
from iris_memory.persona_evolution.models import ErrorCode

from .fakes import ANALYSIS_MODULE, FakeLLMManager, good_analysis_json


class TestExtractJsonObject:
    def test_plain_json(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_json_wrapped_in_prose(self):
        raw = '好的，分析结果如下：\n```json\n{"a": 1}\n```\n以上。'
        assert extract_json_object(raw) == {"a": 1}

    def test_not_json(self):
        assert extract_json_object("完全不是 JSON") is None

    def test_json_array_rejected(self):
        assert extract_json_object("[1, 2, 3]") is None

    def test_empty(self):
        assert extract_json_object("") is None


class TestParseStyleProfile:
    def test_good(self):
        import json

        profile = parse_style_profile(json.loads(good_analysis_json()))
        assert profile is not None
        assert profile["verbosity"] == "short"
        assert profile["confidence"] == 0.85

    def test_bad_enum_rejected(self):
        import json

        data = json.loads(good_analysis_json())
        data["verbosity"] = "verbose"
        assert parse_style_profile(data) is None
        data = json.loads(good_analysis_json())
        data["emoji_style"] = "sparkling"
        assert parse_style_profile(data) is None

    def test_bad_confidence_rejected(self):
        import json

        data = json.loads(good_analysis_json())
        data["confidence"] = "high"
        assert parse_style_profile(data) is None
        data["confidence"] = 1.5
        assert parse_style_profile(data) is None


class TestStyleAnalyzer:
    @pytest.mark.asyncio
    async def test_success(self):
        llm = FakeLLMManager()
        llm.set_default(ANALYSIS_MODULE, good_analysis_json(0.9))
        analyzer = StyleAnalyzer(min_confidence=0.65)
        profile, err = await analyzer.analyze(llm, ["语料一"], "目标")
        assert err is None
        assert profile["confidence"] == 0.9
        assert llm.calls[0]["module"] == ANALYSIS_MODULE

    @pytest.mark.asyncio
    async def test_low_confidence_stops_round(self):
        llm = FakeLLMManager()
        llm.set_default(ANALYSIS_MODULE, good_analysis_json(0.64))
        analyzer = StyleAnalyzer(min_confidence=0.65)
        profile, err = await analyzer.analyze(llm, ["语料"], "目标")
        assert profile is None
        assert err == ErrorCode.ANALYSIS_LOW_CONFIDENCE

    @pytest.mark.asyncio
    async def test_confidence_boundary_passes(self):
        llm = FakeLLMManager()
        llm.set_default(ANALYSIS_MODULE, good_analysis_json(0.65))
        analyzer = StyleAnalyzer(min_confidence=0.65)
        profile, err = await analyzer.analyze(llm, ["语料"], "目标")
        assert err is None and profile is not None

    @pytest.mark.asyncio
    async def test_bad_json_parse_failed(self):
        llm = FakeLLMManager()
        llm.set_default(ANALYSIS_MODULE, "这不是 JSON 输出")
        analyzer = StyleAnalyzer()
        profile, err = await analyzer.analyze(llm, ["语料"], "目标")
        assert profile is None
        assert err == ErrorCode.ANALYSIS_PARSE_FAILED

    @pytest.mark.asyncio
    async def test_schema_invalid_parse_failed(self):
        llm = FakeLLMManager()
        llm.set_default(ANALYSIS_MODULE, '{"verbosity": "wrong", "emoji_style": "low"}')
        analyzer = StyleAnalyzer()
        _, err = await analyzer.analyze(llm, ["语料"], "目标")
        assert err == ErrorCode.ANALYSIS_PARSE_FAILED

    @pytest.mark.asyncio
    async def test_provider_error(self):
        llm = FakeLLMManager()
        llm.push(ANALYSIS_MODULE, RuntimeError("provider down"))
        analyzer = StyleAnalyzer()
        _, err = await analyzer.analyze(llm, ["语料"], "目标")
        assert err == ErrorCode.PROVIDER_ERROR

    @pytest.mark.asyncio
    async def test_provider_timeout(self):
        llm = FakeLLMManager()
        llm.push(ANALYSIS_MODULE, asyncio.TimeoutError())
        analyzer = StyleAnalyzer()
        _, err = await analyzer.analyze(llm, ["语料"], "目标")
        assert err == ErrorCode.PROVIDER_TIMEOUT

"""批量画像分析的失败语义与超时配置测试"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.profile.analyzer import (
    ProfileAnalysisError,
    ProfileAnalyzer,
)

ANALYZER_CONFIG = "astrbot_plugin_iris_memory.iris_memory.profile.analyzer.get_config"
GROUP = {"id": "g1", "tier": "mid", "messages": ["m"], "profile": {}}
USER = {"id": "u1", "tier": "mid", "messages": ["m"], "profile": {}}


def _analyzer(response="{}", side_effect=None) -> ProfileAnalyzer:
    llm = Mock()
    llm.generate_direct = AsyncMock(return_value=response, side_effect=side_effect)
    return ProfileAnalyzer(llm)


def _config(timeout_ms) -> Mock:
    config = Mock()
    config.get = Mock(
        side_effect=lambda key, default=None: {"profile.timeout_ms": timeout_ms}.get(
            key, default
        )
    )
    return config


@pytest.mark.asyncio
async def test_nothing_to_analyze_skips_llm_call():
    analyzer = _analyzer()

    result = await analyzer.analyze_profiles_batch(group=None, users=[])

    assert result == {"group": {}, "users": {}}
    analyzer._llm_manager.generate_direct.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_only_prompt_has_no_users_template():
    analyzer = _analyzer(json.dumps({"group": None}))

    await analyzer.analyze_profiles_batch(group=GROUP, users=[])

    prompt = analyzer._llm_manager.generate_direct.await_args.kwargs["prompt"]
    assert '"users": {' not in prompt
    assert "原样用户ID" not in prompt


@pytest.mark.asyncio
async def test_user_prompt_keeps_users_template():
    analyzer = _analyzer(json.dumps({"group": None, "users": {}}))

    await analyzer.analyze_profiles_batch(group=None, users=[USER])

    prompt = analyzer._llm_manager.generate_direct.await_args.kwargs["prompt"]
    assert '"users": {' in prompt


@pytest.mark.asyncio
async def test_timeout_keeps_real_reason():
    analyzer = _analyzer(side_effect=TimeoutError())

    with pytest.raises(ProfileAnalysisError, match="TimeoutError"):
        await analyzer.analyze_profiles_batch(group=GROUP, users=[])


@pytest.mark.asyncio
async def test_unparsable_response_is_a_failure():
    analyzer = _analyzer("not json")

    with pytest.raises(ProfileAnalysisError, match="可解析"):
        await analyzer.analyze_profiles_batch(group=GROUP, users=[])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured,expected",
    [(0, None), (None, None), (90000, 90.0)],
)
async def test_profile_timeout_is_forwarded(configured, expected):
    analyzer = _analyzer(json.dumps({"group": None, "users": {}}))

    with patch(ANALYZER_CONFIG, return_value=_config(configured)):
        await analyzer.analyze_profiles_batch(group=None, users=[USER])

    kwargs = analyzer._llm_manager.generate_direct.await_args.kwargs
    assert kwargs["timeout"] == expected

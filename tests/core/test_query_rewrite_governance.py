import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from iris_memory.core.llm_request_hook import (
    _QUERY_REWRITE_CACHE,
    _QUERY_REWRITE_INFLIGHT,
    _rewrite_query_for_retrieval,
)


def _config(**overrides):
    values = {
        "l2_query_rewrite_enable": True,
        "l2_query_rewrite_timeout_ms": 3000,
        "l2_query_rewrite_queue_timeout_ms": 800,
        "l2_query_rewrite_cache_size": 256,
        **overrides,
    }
    return SimpleNamespace(get=lambda key, default=None: values.get(key, default))


@pytest.fixture(autouse=True)
def clear_rewrite_state():
    _QUERY_REWRITE_CACHE.clear()
    _QUERY_REWRITE_INFLIGHT.clear()
    yield
    _QUERY_REWRITE_CACHE.clear()
    _QUERY_REWRITE_INFLIGHT.clear()


@pytest.mark.asyncio
async def test_common_preference_question_uses_local_rewrite():
    component_manager = Mock()

    with patch("iris_memory.config.get_config", return_value=_config()):
        result = await _rewrite_query_for_retrieval(
            "你还记得我喜欢什么吗？", component_manager, "persona-a"
        )

    assert result == "用户 偏好 喜好"
    component_manager.get_component.assert_not_called()


@pytest.mark.asyncio
async def test_identical_queries_join_singleflight_and_cache():
    async def generate(**kwargs):
        await asyncio.sleep(0.02)
        return "项目 截止日期"

    llm = SimpleNamespace(is_available=True, generate_direct=AsyncMock(side_effect=generate))
    component_manager = Mock()
    component_manager.get_component.return_value = llm
    query = "你还记得我上次提到的项目截止日期吗？"

    with patch("iris_memory.config.get_config", return_value=_config()):
        results = await asyncio.gather(
            *(
                _rewrite_query_for_retrieval(query, component_manager, "persona-a")
                for _ in range(50)
            )
        )
        cached = await _rewrite_query_for_retrieval(
            query, component_manager, "persona-a"
        )

    assert results == ["项目 截止日期"] * 50
    assert cached == "项目 截止日期"
    assert llm.generate_direct.await_count == 1
    assert llm.generate_direct.await_args.kwargs["queue_timeout"] == 0.8

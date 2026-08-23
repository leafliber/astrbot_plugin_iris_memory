import asyncio

import pytest

from iris_memory.llm.budget import LLMCallBudget, LLMCallBudgetExceeded


@pytest.mark.asyncio
async def test_budget_caps_concurrent_call_attempts():
    budget = LLMCallBudget(max_calls=20, max_runtime_seconds=60)

    async def reserve():
        try:
            await budget.before_call("dream_consolidation")
            return True
        except LLMCallBudgetExceeded:
            return False

    results = await asyncio.gather(*(reserve() for _ in range(100)))

    assert sum(results) == 20
    assert budget.calls == 20
    assert budget.exhausted


@pytest.mark.asyncio
async def test_budget_enforces_minimum_start_interval():
    loop = asyncio.get_running_loop()
    budget = LLMCallBudget(
        max_calls=2,
        max_runtime_seconds=60,
        min_call_interval_seconds=0.02,
    )
    starts = []
    for _ in range(2):
        await budget.before_call("dream_consolidation")
        starts.append(loop.time())

    assert starts[1] - starts[0] >= 0.018

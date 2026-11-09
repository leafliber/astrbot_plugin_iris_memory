"""总结后更新画像的失败语义测试"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.l1_buffer.buffer import L1Buffer
from astrbot_plugin_iris_memory.iris_memory.l1_buffer.models import ContextMessage
from astrbot_plugin_iris_memory.iris_memory.llm.manager import LLMManager
from astrbot_plugin_iris_memory.iris_memory.profile.analyzer import (
    ProfileAnalysisError,
    ProfileAnalyzer,
)
from astrbot_plugin_iris_memory.iris_memory.profile.models import GroupProfile, UserProfile
from astrbot_plugin_iris_memory.iris_memory.profile.storage import ProfileStorage

PACKAGE = "astrbot_plugin_iris_memory.iris_memory"


def _buffer() -> L1Buffer:
    buffer = L1Buffer()
    buffer._is_available = True
    profile_storage = MagicMock(spec=ProfileStorage)
    profile_storage.is_available = True
    llm_manager = MagicMock(spec=LLMManager)
    llm_manager.is_available = True
    component_manager = MagicMock()
    component_manager.get_component.side_effect = lambda name: {
        "profile": profile_storage,
        "llm_manager": llm_manager,
    }.get(name)
    buffer._component_manager = component_manager
    return buffer


def _group_manager() -> MagicMock:
    manager = MagicMock()
    manager.increment_summary_count = AsyncMock()
    manager.get_or_create = AsyncMock(return_value=GroupProfile(group_id="g1"))
    manager.should_update_mid.return_value = True
    manager.should_update_long.return_value = False
    manager.update_from_analysis = AsyncMock()
    manager.update_long_term_from_analysis = AsyncMock()
    return manager


def _user_manager(update: bool) -> MagicMock:
    manager = MagicMock()
    manager.increment_summary_count = AsyncMock()
    manager.get_or_create = AsyncMock(
        side_effect=lambda user_id, group_id, persona_id: UserProfile(user_id=user_id)
    )
    manager.should_update_mid.return_value = update
    manager.should_update_long.return_value = False
    manager.update_from_analysis = AsyncMock()
    manager.update_long_term_from_analysis = AsyncMock()
    return manager


def _messages(count: int) -> list[ContextMessage]:
    return [
        ContextMessage(
            role="user",
            content=f"message-{index}",
            timestamp=datetime.now(),
            token_count=1,
            source=f"u{index}",
        )
        for index in range(count)
    ]


async def _run(analyze, group_manager, user_manager, messages, raise_errors):
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "profile.enable": True,
        "isolation_config.enable_group_isolation": True,
        "profile_batch_size": 1,
    }.get(key, default)
    with (
        patch(f"{PACKAGE}.l1_buffer.buffer.get_config", return_value=config),
        patch(f"{PACKAGE}.profile.GroupProfileManager", return_value=group_manager),
        patch(f"{PACKAGE}.profile.UserProfileManager", return_value=user_manager),
        patch.object(ProfileAnalyzer, "analyze_profiles_batch", analyze),
    ):
        await _buffer()._update_profile_after_summary(
            "g1", messages, "summary", raise_errors=raise_errors
        )


@pytest.mark.asyncio
async def test_null_group_result_is_not_a_failure():
    group_manager = _group_manager()
    analyze = AsyncMock(return_value={"group": None, "users": {}})

    await _run(analyze, group_manager, _user_manager(False), _messages(1), True)

    group_manager.update_from_analysis.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_failure_reason_reaches_outbox():
    analyze = AsyncMock(side_effect=ProfileAnalysisError("TimeoutError: "))

    with pytest.raises(ProfileAnalysisError, match="TimeoutError"):
        await _run(analyze, _group_manager(), _user_manager(False), _messages(1), True)


@pytest.mark.asyncio
async def test_failed_batch_does_not_stop_remaining_batches():
    user_manager = _user_manager(True)
    analyze = AsyncMock(
        side_effect=[
            ProfileAnalysisError("TimeoutError: "),
            {"group": {}, "users": {"u1": {"interests": ["topic"]}}},
        ]
    )

    await _run(analyze, _group_manager(), user_manager, _messages(2), False)

    assert analyze.await_count == 2
    user_manager.update_from_analysis.assert_awaited_once()

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.l1_buffer.buffer import L1Buffer
from iris_memory.l1_buffer.models import ContextMessage
from iris_memory.llm.manager import LLMManager
from iris_memory.profile.analyzer import ProfileAnalyzer
from iris_memory.profile.models import GroupProfile, UserProfile
from iris_memory.profile.storage import ProfileStorage


@pytest.mark.asyncio
async def test_forty_users_are_analyzed_in_five_profile_batches():
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

    group_manager = MagicMock()
    group_manager.increment_summary_count = AsyncMock()
    group_manager.get_or_create = AsyncMock(return_value=GroupProfile(group_id="g1"))
    group_manager.should_update_mid.return_value = True
    group_manager.should_update_long.return_value = False
    group_manager.update_from_analysis = AsyncMock()
    group_manager.update_long_term_from_analysis = AsyncMock()

    user_manager = MagicMock()
    user_manager.increment_summary_count = AsyncMock()
    user_manager.get_or_create = AsyncMock(
        side_effect=lambda user_id, group_id, persona_id: UserProfile(user_id=user_id)
    )
    user_manager.should_update_mid.return_value = True
    user_manager.should_update_long.return_value = False
    user_manager.update_from_analysis = AsyncMock()
    user_manager.update_long_term_from_analysis = AsyncMock()

    messages = [
        ContextMessage(
            role="user",
            content=f"message-{index}",
            timestamp=datetime.now(),
            token_count=1,
            source=f"u{index}",
        )
        for index in range(40)
    ]

    async def batch_result(*, group, users):
        return {
            "group": {"interests": ["topic"]} if group else {},
            "users": {
                item["id"]: {"interests": [f"interest-{item['id']}"]}
                for item in users
            },
        }

    analyze = AsyncMock(side_effect=batch_result)
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "profile.enable": True,
        "isolation_config.enable_group_isolation": True,
        "profile_batch_size": 8,
    }.get(key, default)

    with (
        patch("iris_memory.l1_buffer.buffer.get_config", return_value=config),
        patch("iris_memory.profile.GroupProfileManager", return_value=group_manager),
        patch("iris_memory.profile.UserProfileManager", return_value=user_manager),
        patch.object(ProfileAnalyzer, "analyze_profiles_batch", analyze),
    ):
        await buffer._update_profile_after_summary(
            "g1", messages, "summary", raise_errors=True
        )

    assert analyze.await_count == 5
    assert user_manager.update_from_analysis.await_count == 40

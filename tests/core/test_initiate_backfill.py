"""initiate 直发消息 L1 回填测试

验证 handle_initiate_backfill 的 persona_id 继承逻辑：
- 队列尾部消息的 persona_id 被继承（人格隔离场景防污染）
- 队列为空 / 组件不可用时静默跳过
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime

from iris_memory.core.initiate_backfill import handle_initiate_backfill
from iris_memory.l1_buffer.models import ContextMessage


class FakeL1Buffer:
    """模拟 L1Buffer（内存队列）"""

    def __init__(self, messages=None):
        self.is_available = True
        self._messages = list(messages or [])
        self.added = []

    def get_context(self, group_id, max_length=None):
        return list(self._messages)

    async def add_message(self, group_id, role, content, source, persona_id="default", metadata=None):
        self.added.append(
            {
                "group_id": group_id,
                "role": role,
                "content": content,
                "source": source,
                "persona_id": persona_id,
            }
        )


class FakeComponentManager:
    def __init__(self, buffer):
        self._buffer = buffer

    def get_available_component(self, name):
        return self._buffer if name == "l1_buffer" else None


def _msg(persona_id: str, content: str = "hello") -> ContextMessage:
    return ContextMessage(
        role="user",
        content=content,
        timestamp=datetime.now(),
        token_count=5,
        source="u1",
        persona_id=persona_id,
    )


@pytest.mark.asyncio
async def test_backfill_inherits_last_message_persona():
    """人格隔离启用时，回填消息应继承队列尾部消息的 persona_id"""
    buffer = FakeL1Buffer(messages=[_msg("xyz"), _msg("xyz", "latest")])
    cm = FakeComponentManager(buffer)

    await handle_initiate_backfill("g1", "bot 发起的话题", cm)

    assert len(buffer.added) == 1
    assert buffer.added[0]["persona_id"] == "xyz"
    assert buffer.added[0]["role"] == "assistant"
    assert buffer.added[0]["group_id"] == "g1"


@pytest.mark.asyncio
async def test_backfill_default_persona_when_isolation_off():
    """人格隔离未启用时，队列消息 persona 均为 default，行为与旧版一致"""
    buffer = FakeL1Buffer(messages=[_msg("default")])
    cm = FakeComponentManager(buffer)

    await handle_initiate_backfill("g1", "bot 发起的话题", cm)

    assert len(buffer.added) == 1
    assert buffer.added[0]["persona_id"] == "default"


@pytest.mark.asyncio
async def test_backfill_skips_empty_queue():
    """群尚无 L1 缓冲内容时静默跳过"""
    buffer = FakeL1Buffer(messages=[])
    cm = FakeComponentManager(buffer)

    await handle_initiate_backfill("g1", "bot 发起的话题", cm)

    assert buffer.added == []


@pytest.mark.asyncio
async def test_backfill_skips_unavailable_component():
    """L1 组件不可用时静默跳过"""
    cm = FakeComponentManager(None)

    await handle_initiate_backfill("g1", "bot 发起的话题", cm)


@pytest.mark.asyncio
async def test_backfill_skips_empty_args():
    """空 group_id 或空 text 直接返回"""
    buffer = FakeL1Buffer(messages=[_msg("xyz")])
    cm = FakeComponentManager(buffer)

    await handle_initiate_backfill("", "text", cm)
    await handle_initiate_backfill("g1", "", cm)

    assert buffer.added == []

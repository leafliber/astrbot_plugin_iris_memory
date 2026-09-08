import logging
from dataclasses import replace

import pytest

from iris_memory.identity import Identity


class FakeHost:
    logger = logging.getLogger("iris-tests")

    def __init__(self):
        self.sent = []
        self.adopted = {}
        self.shared_closed = False

    async def providers(self, control):
        return None, None

    async def check_persona_support(self):
        pass

    async def adopt_persona(self, control, identity, persona):
        self.adopted[identity.key] = persona

    async def release_personas(self, control, only_umo=None):
        self.adopted.clear()

    async def generate(self, control, prompt, **kwargs):
        await control.budget.reserve(100)
        return "你好，记得休息。"

    async def send(self, umo, text):
        self.sent.append((umo, text))
        return True

    def provider_list(self):
        return {"chat": [], "embedding": []}


@pytest.fixture
def host():
    return FakeHost()


@pytest.fixture
def identity():
    return Identity(
        "chat:test",
        "session:test",
        "test",
        "test-realm",
        "user",
        "User",
        "test:FriendMessage:user",
        "message-1",
        False,
    )


@pytest.fixture
def other(identity):
    return replace(
        identity,
        key="chat:other",
        session_key="session:other",
        user="other",
        umo="test:FriendMessage:other",
    )

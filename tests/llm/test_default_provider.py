"""模块未配置 Provider 时回退到 AstrBot 当前对话 Provider 的测试"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from astrbot.core.provider.provider import AbstractProvider
from astrbot.core.star.context import Context

from astrbot_plugin_iris_memory.iris_memory.llm.manager import LLMManager

MANAGER_CONFIG = "astrbot_plugin_iris_memory.iris_memory.llm.manager.get_config"


class FakeChatProvider:
    def meta(self):
        return SimpleNamespace(id="current_chat")


class FakeContext:
    """只暴露 AstrBot Context 上真实存在的接口。"""

    def __init__(self, provider=None, error: Exception | None = None):
        self._provider = provider
        self._error = error

    def get_using_provider(self, umo=None):
        if self._error:
            raise self._error
        return self._provider


def _manager(context) -> LLMManager:
    return LLMManager(context, AsyncMock())


def _config(values: dict) -> Mock:
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: values.get(key, default))
    return config


def test_astrbot_exposes_the_interfaces_we_rely_on():
    assert callable(Context.get_using_provider)
    assert callable(AbstractProvider.meta)


def test_default_is_astrbot_current_chat_provider():
    assert _manager(FakeContext(FakeChatProvider()))._get_default_provider() == "current_chat"


def test_no_default_without_chat_provider():
    assert _manager(FakeContext())._get_default_provider() is None


def test_no_default_when_astrbot_rejects_lookup():
    context = FakeContext(error=ValueError("wrong provider type"))
    assert _manager(context)._get_default_provider() is None


@pytest.mark.asyncio
async def test_blank_module_provider_falls_back_to_current():
    manager = _manager(FakeContext(FakeChatProvider()))

    with patch(MANAGER_CONFIG, return_value=_config({"l1_buffer.summary_provider": ""})):
        assert await manager._resolve_provider("l1_summarizer", None) == "current_chat"


@pytest.mark.asyncio
async def test_configured_module_provider_wins():
    manager = _manager(FakeContext(FakeChatProvider()))

    with patch(MANAGER_CONFIG, return_value=_config({"l1_buffer.summary_provider": "chosen"})):
        assert await manager._resolve_provider("l1_summarizer", None) == "chosen"

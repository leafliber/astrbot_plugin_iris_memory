"""LearningComponent 组件测试"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iris_memory.config import init_config
from iris_memory.config.config import reset_config
from iris_memory.learning import LearningComponent


@pytest.fixture
def disabled_config(tmp_path: Path):
    """learning.enable=false 的全局配置"""
    cfg = init_config({}, tmp_path)
    yield cfg
    reset_config()


def _event():
    event = MagicMock()
    event.message_str = "今晚吃啥？"
    event.get_self_id.return_value = "bot"
    event.message_obj = None
    return event


def _resp():
    resp = MagicMock()
    resp.completion_text = "随便吃点呀"
    return resp


def _fake_adapter():
    adapter = MagicMock()
    adapter.get_session_id.return_value = "sess1"
    adapter.get_group_id.return_value = "g1"
    adapter.get_user_id.return_value = "u1"
    adapter.is_group_message.return_value = True
    adapter.get_raw_message.return_value = {}
    return adapter


class TestInitialize:
    """初始化与禁用语义"""

    @pytest.mark.asyncio
    async def test_disabled_when_config_off(self, disabled_config):
        comp = LearningComponent()
        await comp.initialize()
        assert comp.is_available is False
        assert comp.is_disabled is True
        assert "未启用" in comp.init_error

    @pytest.mark.asyncio
    async def test_initialize_success(self, config):
        comp = LearningComponent()
        await comp.initialize()
        assert comp.is_available is True
        assert comp.storage is not None
        assert comp.name == "learning"
        await comp.shutdown()


class TestFaultIsolation:
    """公开方法内部异常不抛出"""

    @pytest.mark.asyncio
    async def test_on_message_exception_swallowed(self, config):
        comp = LearningComponent()
        await comp.initialize()
        with patch(
            "iris_memory.learning.collector.get_adapter",
            side_effect=RuntimeError("适配层炸了"),
        ):
            await comp.on_message(_event())  # 不抛出
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_on_response_exception_swallowed(self, config):
        comp = LearningComponent()
        await comp.initialize()
        with patch(
            "iris_memory.learning.collector.get_adapter",
            side_effect=RuntimeError("适配层炸了"),
        ):
            await comp.on_response(_event(), _resp())  # 不抛出
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_build_context_exception_swallowed(self, config):
        comp = LearningComponent()
        await comp.initialize()
        meta = {}
        with patch(
            "iris_memory.learning.injector.get_adapter",
            side_effect=RuntimeError("适配层炸了"),
        ):
            text = await comp.build_context(_event(), meta)
        assert text == ""
        assert "error" in meta
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_methods_noop_when_unavailable(self, disabled_config):
        comp = LearningComponent()
        await comp.initialize()  # 未启用
        await comp.on_message(_event())
        await comp.on_response(_event(), _resp())
        assert await comp.build_context(_event(), {}) == ""
        await comp.run_review()
        await comp.run_jargon_scan()
        await comp.run_decay()


class TestCollection:
    """采集通路"""

    @pytest.mark.asyncio
    async def test_on_response_pairs_and_patterns(self, config):
        comp = LearningComponent()
        await comp.initialize()
        with patch(
            "iris_memory.learning.collector.get_adapter",
            return_value=_fake_adapter(),
        ):
            await comp.on_response(_event(), _resp())
        pairs = comp.storage.get_pending_pairs(10)
        assert len(pairs) == 1
        assert pairs[0]["user_text"] == "今晚吃啥？"
        assert pairs[0]["bot_text"] == "随便吃点呀"
        patterns = comp.storage.get_pending_patterns(10)
        assert len(patterns) >= 1
        # 待审队列已登记
        assert comp._reviewer.queue_size >= 1
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_on_message_updates_jargon(self, config):
        comp = LearningComponent()
        await comp.initialize()
        # 组件层与采集层各自顶层绑定了 get_adapter，需同时 patch
        with (
            patch(
                "iris_memory.learning.component.get_adapter",
                return_value=_fake_adapter(),
            ),
            patch(
                "iris_memory.learning.collector.get_adapter",
                return_value=_fake_adapter(),
            ),
        ):
            await comp.on_message(_event())
        assert comp.storage.get_stats()["jargon_candidate"]["total"] > 0
        await comp.shutdown()


class TestShutdown:
    """关闭语义"""

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, config):
        comp = LearningComponent()
        await comp.initialize()
        await comp.shutdown()
        assert comp.is_available is False
        assert comp.storage is None
        # 再次关闭不报错
        await comp.shutdown()

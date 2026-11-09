"""后台组件初始化失败后的自愈测试"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.core import components as components_module
from astrbot_plugin_iris_memory.iris_memory.core.components import (
    Component,
    ComponentManager,
    InitMode,
)
from astrbot_plugin_iris_memory.iris_memory.core.lifecycle import (
    _start_scheduled_tasks_deferred,
)


class FlakyComponent(Component):
    """前 failures 次初始化时依赖尚未就绪。"""

    def __init__(self, failures: int, retryable: bool = True):
        super().__init__()
        self._failures = failures
        self._retryable = retryable
        self._waiting = False
        self.attempts = 0
        self._init_mode = InitMode.BACKGROUND

    @property
    def name(self) -> str:
        return "flaky"

    @property
    def should_retry_init(self) -> bool:
        return self._waiting

    async def initialize(self) -> None:
        self.attempts += 1
        if self.attempts <= self._failures:
            self._waiting = self._retryable
            self._is_available = False
            self._init_error = "dependency not ready"
            return
        self._waiting = False
        self._is_available = True

    async def shutdown(self) -> None:
        self._is_available = False


async def _until(predicate, timeout: float = 2.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


@pytest.fixture(autouse=True)
def fast_retry(monkeypatch):
    monkeypatch.setattr(components_module, "_INIT_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(components_module, "_INIT_RETRY_MAX_SECONDS", 0.02)


@pytest.mark.asyncio
async def test_failed_background_component_recovers():
    component = FlakyComponent(failures=3)
    manager = ComponentManager((component,))

    await manager.initialize_all()
    await _until(lambda: component.is_available)
    await _until(lambda: manager.status.is_module_available("flaky"))

    assert component.attempts == 4
    assert component.init_error is None
    assert manager.get_background_init_results()["flaky"].success
    await manager.shutdown_all()


@pytest.mark.asyncio
async def test_non_retryable_failure_is_not_retried():
    component = FlakyComponent(failures=1, retryable=False)
    manager = ComponentManager((component,))

    await manager.initialize_all()
    await manager.wait_for_background_init(timeout=1)
    await asyncio.sleep(0.05)

    assert component.attempts == 1
    assert not component.is_available
    await manager.shutdown_all()


@pytest.mark.asyncio
async def test_retry_does_not_block_background_wait():
    component = FlakyComponent(failures=10_000)
    manager = ComponentManager((component,))

    await manager.initialize_all()
    await asyncio.wait_for(manager.wait_for_background_init(), timeout=1)

    assert manager.all_background_init_done
    await manager.shutdown_all()


@pytest.mark.asyncio
async def test_shutdown_stops_retrying():
    component = FlakyComponent(failures=10_000)
    manager = ComponentManager((component,))

    await manager.initialize_all()
    await _until(lambda: component.attempts >= 2)
    await manager.shutdown_all()
    attempts = component.attempts
    await asyncio.sleep(0.05)

    assert component.attempts == attempts


def _deferred_manager(l2):
    scheduler = Mock()
    scheduler.is_available = True
    manager = Mock()
    manager.get_component = Mock(
        side_effect=lambda name: {"scheduler": scheduler, "l2_memory": l2}.get(name)
    )
    manager.check_component = Mock(return_value="unavailable")
    config = Mock()
    config.get = Mock(
        side_effect=lambda key, default=None: {
            "scheduled_tasks.enable_dream": True,
            "dream_task_interval_hours": 24,
        }.get(key, default)
    )
    return manager, scheduler, config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "available,waiting,registered",
    [(True, False, True), (False, True, True), (False, False, False)],
)
async def test_dream_registration_follows_l2_recoverability(
    available, waiting, registered
):
    l2 = Mock(is_available=available, should_retry_init=waiting)
    manager, scheduler, config = _deferred_manager(l2)

    with patch(
        "astrbot_plugin_iris_memory.iris_memory.config.get_config", return_value=config
    ):
        await _start_scheduled_tasks_deferred(manager)

    assert scheduler.register_periodic_task.called is registered

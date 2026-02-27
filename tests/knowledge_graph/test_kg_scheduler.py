"""
KGScheduler 定时维护调度器单元测试
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from iris_memory.services.modules.kg_module import KnowledgeGraphModule, KGScheduler


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def kg_module():
    """创建完整初始化的 KG 模块（不启动调度器）"""
    module = KnowledgeGraphModule()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(module.initialize(
            plugin_data_path=Path(tmpdir),
            kg_mode="rule",
            enabled=True,
            auto_maintenance=False,  # 不启动调度器
        ))
        yield module
        run(module.close())


class TestKGSchedulerLifecycle:
    """调度器生命周期测试"""

    def test_scheduler_start_stop(self, kg_module):
        """调度器可以正常启动和停止"""
        scheduler = KGScheduler(kg_module=kg_module, interval=3600)
        run(scheduler.start())
        assert scheduler._is_running is True
        assert scheduler._task is not None

        run(scheduler.stop())
        assert scheduler._is_running is False
        assert scheduler._task is None

    def test_scheduler_double_start(self, kg_module):
        """重复启动不会创建多个任务"""
        scheduler = KGScheduler(kg_module=kg_module, interval=3600)
        run(scheduler.start())
        task1 = scheduler._task

        run(scheduler.start())
        task2 = scheduler._task
        assert task1 is task2

        run(scheduler.stop())

    def test_scheduler_stop_idempotent(self, kg_module):
        """停止操作是幂等的"""
        scheduler = KGScheduler(kg_module=kg_module, interval=3600)
        run(scheduler.stop())  # 未启动时停止不报错

    def test_scheduler_run_once(self, kg_module):
        """手动触发一次维护"""
        scheduler = KGScheduler(kg_module=kg_module, interval=3600)
        # run_once 不需要启动调度循环
        run(scheduler.run_once())  # 应该不报错


class TestKGSchedulerIntegration:
    """调度器集成到 KG 模块测试"""

    def test_module_init_with_auto_maintenance(self):
        """模块启用自动维护时创建调度器"""
        module = KnowledgeGraphModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            run(module.initialize(
                plugin_data_path=Path(tmpdir),
                kg_mode="rule",
                enabled=True,
                auto_maintenance=True,
                maintenance_interval=3600,
            ))
            assert module._scheduler is not None
            assert module._scheduler._is_running is True
            run(module.close())

    def test_module_init_without_auto_maintenance(self):
        """模块禁用自动维护时不创建调度器"""
        module = KnowledgeGraphModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            run(module.initialize(
                plugin_data_path=Path(tmpdir),
                kg_mode="rule",
                enabled=True,
                auto_maintenance=False,
            ))
            assert module._scheduler is None
            run(module.close())

    def test_module_close_stops_scheduler(self):
        """模块关闭时停止调度器"""
        module = KnowledgeGraphModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            run(module.initialize(
                plugin_data_path=Path(tmpdir),
                kg_mode="rule",
                enabled=True,
                auto_maintenance=True,
                maintenance_interval=3600,
            ))
            scheduler = module._scheduler
            assert scheduler._is_running is True

            run(module.close())
            assert scheduler._is_running is False


class TestKGSchedulerConfig:
    """调度器配置参数测试"""

    def test_scheduler_config_passed(self, kg_module):
        """配置参数正确传递到调度器"""
        scheduler = KGScheduler(
            kg_module=kg_module,
            interval=7200,
            auto_cleanup_orphans=False,
            auto_cleanup_low_confidence=True,
            low_confidence_threshold=0.3,
            staleness_days=60,
        )
        assert scheduler._interval == 7200
        assert scheduler._auto_cleanup_orphans is False
        assert scheduler._auto_cleanup_low_confidence is True
        assert scheduler._low_confidence_threshold == 0.3
        assert scheduler._staleness_days == 60

"""PersonaEvolutionComponent 组件测试"""

from pathlib import Path
from unittest.mock import patch

import pytest

from iris_memory.persona_evolution import EvolutionJob, PersonaEvolutionComponent

from .conftest import insert_sample, make_adapter, make_event

ADAPTER_PATH = "iris_memory.persona_evolution.collector.get_adapter"


class TestInitialize:
    """初始化与禁用语义"""

    @pytest.mark.asyncio
    async def test_disabled_when_config_off(self, disabled_config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        assert comp.is_available is False
        assert comp.is_disabled is True
        assert "未启用" in comp.init_error

    @pytest.mark.asyncio
    async def test_initialize_success(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        assert comp.is_available is True
        assert comp.storage is not None
        assert comp.name == "persona_evolution"
        # 数据库落在 config.data_dir/persona_evolution/ 下
        db_path = Path(config.data_dir) / "persona_evolution" / "persona_evolution.db"
        assert db_path.exists()
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_init_failure_degrades(self, config, monkeypatch):
        """初始化失败置 _init_error 降级，不抛出"""
        monkeypatch.setattr(
            "iris_memory.persona_evolution.component.PersonaEvolutionStorage.init_schema",
            lambda self: (_ for _ in ()).throw(RuntimeError("磁盘只读")),
        )
        comp = PersonaEvolutionComponent()
        await comp.initialize()  # 不抛出
        assert comp.is_available is False
        assert "磁盘只读" in comp.init_error
        await comp.shutdown()


class TestFaultIsolation:
    """公开方法内部异常不抛出"""

    @pytest.mark.asyncio
    async def test_on_message_exception_swallowed(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        with patch(ADAPTER_PATH, side_effect=RuntimeError("适配层炸了")):
            await comp.on_message(make_event())  # 不抛出
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_methods_noop_when_unavailable(self, disabled_config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()  # 未启用
        await comp.on_message(make_event())
        assert await comp.count_scoped_samples() == 0
        assert await comp.count_job_samples(1) == 0
        assert await comp.run_sample_prune() == 0
        await comp.shutdown()


class TestCollection:
    """采集通路"""

    @pytest.mark.asyncio
    async def test_on_message_inserts_sample(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        with patch(ADAPTER_PATH, return_value=make_adapter()):
            await comp.on_message(make_event())
        assert comp.storage.count_samples() == 1
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_private_message_not_collected(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        with patch(ADAPTER_PATH, return_value=make_adapter(is_group=False)):
            await comp.on_message(make_event())
        assert comp.storage.count_samples() == 0
        await comp.shutdown()


class TestSampleCountQuery:
    """Job 语料计数查询接口（供触发用）"""

    @pytest.mark.asyncio
    async def test_count_job_samples_scoped(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        storage = comp.storage
        insert_sample(storage, group_id="g1", user_id="u1", text="范围内的消息")
        insert_sample(storage, group_id="g2", user_id="u2", text="范围外的消息")
        job_id = storage.create_job(
            EvolutionJob(persona_id="p1", source_group_ids=["g1"])
        )
        assert await comp.count_job_samples(job_id, since_cursor=False) == 1
        assert await comp.count_job_samples(9999) == 0
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_count_job_samples_since_cursor(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        storage = comp.storage
        insert_sample(storage, text="基线前的消息")
        job_id = storage.create_job(
            EvolutionJob(persona_id="p1", last_sample_cursor=storage.get_latest_sample_id())
        )
        insert_sample(storage, text="基线后的消息")
        assert await comp.count_job_samples(job_id) == 1
        assert await comp.count_job_samples(job_id, since_cursor=False) == 2
        await comp.shutdown()


class TestPrune:
    """语料保留清理任务"""

    @pytest.mark.asyncio
    async def test_run_sample_prune(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        import time

        insert_sample(
            comp.storage, text="过期消息", created_at=time.time() - 31 * 86400
        )
        insert_sample(comp.storage, text="保留消息")
        removed = await comp.run_sample_prune()
        assert removed == 1
        assert comp.storage.count_samples() == 1
        await comp.shutdown()


class TestShutdown:
    """关闭语义"""

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        await comp.shutdown()
        assert comp.is_available is False
        assert comp.storage is None
        # 再次关闭不报错
        await comp.shutdown()


class _FakeScheduler:
    """TaskScheduler fake：记录 schedule_task 调用"""

    is_available = True

    def __init__(self, running: bool = False):
        self._running = running
        self.scheduled = []

    def is_task_running(self, name: str) -> bool:
        return self._running

    async def schedule_task(self, name, coro_func) -> None:
        self.scheduled.append(name)


class TestTriggerSchedule:
    """消息计数触发：只更新本地计数，满足后投递一次性任务"""

    @pytest.mark.asyncio
    async def test_threshold_reached_schedules_once(self, config, monkeypatch):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        comp.storage.create_job(
            EvolutionJob(persona_id="p1", trigger_sample_count=3)
        )
        scheduler = _FakeScheduler()
        monkeypatch.setattr(comp, "_get_scheduler", lambda: scheduler)

        with patch(ADAPTER_PATH, return_value=make_adapter()):
            for i in range(3):
                await comp.on_message(
                    make_event(text=f"今晚吃点什么好呢 {i}", message_id=f"m{i}")
                )
        assert scheduler.scheduled == ["persona_evolution_trigger_scan"]
        # 计数已重置，再发一条不重复调度
        with patch(ADAPTER_PATH, return_value=make_adapter()):
            await comp.on_message(
                make_event(text="还没凑够新的门槛", message_id="m9")
            )
        assert len(scheduler.scheduled) == 1
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_below_threshold_no_schedule(self, config, monkeypatch):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        comp.storage.create_job(
            EvolutionJob(persona_id="p1", trigger_sample_count=5)
        )
        scheduler = _FakeScheduler()
        monkeypatch.setattr(comp, "_get_scheduler", lambda: scheduler)

        with patch(ADAPTER_PATH, return_value=make_adapter()):
            for i in range(2):
                await comp.on_message(
                    make_event(text=f"今晚吃点什么好呢 {i}", message_id=f"m{i}")
                )
        assert scheduler.scheduled == []
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_no_jobs_no_schedule(self, config, monkeypatch):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        scheduler = _FakeScheduler()
        monkeypatch.setattr(comp, "_get_scheduler", lambda: scheduler)

        with patch(ADAPTER_PATH, return_value=make_adapter()):
            for i in range(10):
                await comp.on_message(
                    make_event(text=f"今晚吃点什么好呢 {i}", message_id=f"m{i}")
                )
        assert scheduler.scheduled == []
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_scan_running_no_reschedule(self, config, monkeypatch):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        comp.storage.create_job(
            EvolutionJob(persona_id="p1", trigger_sample_count=1)
        )
        scheduler = _FakeScheduler(running=True)
        monkeypatch.setattr(comp, "_get_scheduler", lambda: scheduler)

        with patch(ADAPTER_PATH, return_value=make_adapter()):
            await comp.on_message(make_event(text="今晚吃点什么好呢"))
        assert scheduler.scheduled == []  # 防重入：扫描进行中不重复投递
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_run_job_unavailable_component(self, disabled_config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        result = await comp.run_job(1, "manual")
        assert result["ok"] is False
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_run_trigger_scan_unavailable(self, disabled_config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        assert await comp.run_trigger_scan() == 0
        await comp.shutdown()

    @pytest.mark.asyncio
    async def test_service_created_on_initialize(self, config):
        comp = PersonaEvolutionComponent()
        await comp.initialize()
        assert comp.service is not None
        await comp.shutdown()

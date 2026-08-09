"""发布器测试：发布流程、回读验证、PersonaManager 降级与调用参数"""

import pytest

from iris_memory.persona_evolution.models import (
    ErrorCode,
    PersonaRevision,
    RevisionStatus,
)
from iris_memory.persona_evolution.publisher import (
    PersonaPublisher,
    persona_hash,
)

from .conftest import make_job
from .fakes import FakeContext, FakePersonaManager

BASE = "你是 Iris。\n\n<!-- IRIS_EVOLUTION:BEGIN v1 -->\n旧\n<!-- IRIS_EVOLUTION:END -->\n"
CANDIDATE = BASE.replace("旧", "新")


def _make_revision(storage, job_id: int, status=RevisionStatus.CANDIDATE) -> PersonaRevision:
    revision = PersonaRevision(
        job_id=job_id,
        version=0,
        status=status.value,
        base_prompt=BASE,
        result_prompt=CANDIDATE,
        base_hash=persona_hash(BASE),
        result_hash=persona_hash(CANDIDATE),
    )
    storage.create_revision(revision)
    return revision


class TestPublishHappyPath:
    @pytest.mark.asyncio
    async def test_publish_applied(self, storage, persona_manager: FakePersonaManager):
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")
        job = storage.get_job(job_id)
        revision = _make_revision(storage, job_id)

        publisher = PersonaPublisher(FakeContext(persona_manager), storage)
        err = await publisher.publish(job, revision)

        assert err is None
        assert persona_manager.get_prompt("p1") == CANDIDATE
        updated = storage.get_revision(revision.id)
        assert updated.status == RevisionStatus.APPLIED.value
        assert updated.applied_at is not None
        job_after = storage.get_job(job_id)
        assert job_after.last_applied_revision_id == revision.id
        assert job_after.last_success_at is not None

    @pytest.mark.asyncio
    async def test_update_persona_called_with_exactly_two_params(
        self, storage, persona_manager: FakePersonaManager
    ):
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")
        revision = _make_revision(storage, job_id)

        publisher = PersonaPublisher(FakeContext(persona_manager), storage)
        await publisher.publish(storage.get_job(job_id), revision)

        assert len(persona_manager.update_calls) == 1
        args, kwargs = persona_manager.update_calls[0]
        assert args == ("p1",)  # 位置参数只有 persona_id
        assert set(kwargs) == {"system_prompt"}  # 关键字参数只有 system_prompt
        assert kwargs["system_prompt"] == CANDIDATE

    @pytest.mark.asyncio
    async def test_revision_intent_recorded_before_publish(
        self, storage, persona_manager: FakePersonaManager
    ):
        # 发布意图（publishing）先于 PersonaManager 调用落库
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")
        revision = _make_revision(storage, job_id)
        seen_status = {}

        original_update = persona_manager.update_persona

        async def spy_update(*args, **kwargs):
            seen_status["at_call"] = storage.get_revision(revision.id).status
            return await original_update(*args, **kwargs)

        persona_manager.update_persona = spy_update
        publisher = PersonaPublisher(FakeContext(persona_manager), storage)
        await publisher.publish(storage.get_job(job_id), revision)
        assert seen_status["at_call"] == RevisionStatus.PUBLISHING.value


class TestPublishFailures:
    @pytest.mark.asyncio
    async def test_base_hash_mismatch_no_overwrite(
        self, storage, persona_manager: FakePersonaManager
    ):
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")
        revision = _make_revision(storage, job_id)
        persona_manager.external_edit("p1", BASE + "外部编辑")

        publisher = PersonaPublisher(FakeContext(persona_manager), storage)
        err = await publisher.publish(storage.get_job(job_id), revision)

        assert err == ErrorCode.BASE_HASH_MISMATCH
        assert persona_manager.get_prompt("p1") == BASE + "外部编辑"
        assert persona_manager.update_calls == []
        assert (
            storage.get_revision(revision.id).status
            == RevisionStatus.PUBLISH_FAILED.value
        )

    @pytest.mark.asyncio
    async def test_update_raises(self, storage, persona_manager: FakePersonaManager):
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")
        revision = _make_revision(storage, job_id)

        async def boom(*args, **kwargs):
            raise RuntimeError("db locked")

        persona_manager.update_persona = boom
        publisher = PersonaPublisher(FakeContext(persona_manager), storage)
        err = await publisher.publish(storage.get_job(job_id), revision)
        assert err == ErrorCode.PUBLISH_FAILED
        assert (
            storage.get_revision(revision.id).status
            == RevisionStatus.PUBLISH_FAILED.value
        )

    @pytest.mark.asyncio
    async def test_readback_mismatch(self, storage, persona_manager: FakePersonaManager):
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")
        revision = _make_revision(storage, job_id)

        original_get = persona_manager.get_persona
        calls = {"n": 0}

        async def flaky_get(persona_id):
            calls["n"] += 1
            persona = await original_get(persona_id)
            if calls["n"] >= 2 and persona is not None:  # 回读时返回错误内容
                persona.system_prompt = "被篡改"
            return persona

        persona_manager.get_persona = flaky_get
        publisher = PersonaPublisher(FakeContext(persona_manager), storage)
        err = await publisher.publish(storage.get_job(job_id), revision)
        assert err == ErrorCode.PUBLISH_FAILED

    @pytest.mark.asyncio
    async def test_persona_not_found(self, storage, persona_manager: FakePersonaManager):
        job_id = make_job(storage, "ghost")
        revision = _make_revision(storage, job_id)
        publisher = PersonaPublisher(FakeContext(persona_manager), storage)
        err = await publisher.publish(storage.get_job(job_id), revision)
        assert err == ErrorCode.PERSONA_NOT_FOUND

    @pytest.mark.asyncio
    async def test_no_persona_manager(self, storage):
        job_id = make_job(storage, "p1")
        revision = _make_revision(storage, job_id)
        publisher = PersonaPublisher(FakeContext(None), storage)
        err = await publisher.publish(storage.get_job(job_id), revision)
        assert err == ErrorCode.PUBLISH_FAILED
        assert (
            storage.get_revision(revision.id).status
            == RevisionStatus.PUBLISH_FAILED.value
        )

    @pytest.mark.asyncio
    async def test_no_update_persona_method(
        self, storage, persona_manager: FakePersonaManager
    ):
        # 老版本 PersonaManager 没有 update_persona：getattr 探测降级
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")
        revision = _make_revision(storage, job_id)

        class LegacyManager:
            async def get_persona(self, persona_id):
                return await persona_manager.get_persona(persona_id)

        publisher = PersonaPublisher(FakeContext(LegacyManager()), storage)
        err = await publisher.publish(storage.get_job(job_id), revision)
        assert err == ErrorCode.PUBLISH_FAILED

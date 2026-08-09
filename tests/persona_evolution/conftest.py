"""人格自迭代测试共享 fixture"""

from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from iris_memory.config import init_config
from iris_memory.config.config import reset_config
from iris_memory.persona_evolution import PersonaEvolutionStorage
from iris_memory.persona_evolution.collector import PersonaCollector
from iris_memory.persona_evolution.models import EvolutionJob
from iris_memory.persona_evolution.service import PersonaEvolutionService

from .fakes import FakeContext, FakeLLMManager, FakePersonaManager


@pytest.fixture
def config(tmp_path: Path):
    """初始化全局配置单例（persona_evolution.enable=true），测试后重置

    get_config 是全局单例，persona_evolution 模块在模块顶层绑定了
    get_config 符号，因此这里用真实 init_config 而不是 patch。
    """
    cfg = init_config({"persona_evolution": {"enable": True}}, tmp_path)
    yield cfg
    reset_config()


@pytest.fixture
def disabled_config(tmp_path: Path):
    """persona_evolution.enable=false 的全局配置（默认值）"""
    cfg = init_config({}, tmp_path)
    yield cfg
    reset_config()


@pytest.fixture
def storage(tmp_path: Path):
    """tmp_path 隔离的 PersonaEvolutionStorage 实例"""
    s = PersonaEvolutionStorage(tmp_path / "persona_evolution.db")
    s.init_schema()
    yield s
    s.close()


@pytest.fixture
def collector(storage: PersonaEvolutionStorage):
    """持有真实 storage 的采集器"""
    return PersonaCollector(storage)


def make_event(
    text: str = "今晚吃什么好呢",
    platform: str = "aiocqhttp",
    self_id: str = "bot999",
    message_id: str | None = "m1",
):
    """构造 mock 消息事件（参照 tests/learning/test_component.py 的做法）"""
    event = MagicMock()
    event.message_str = text
    event.get_platform_name.return_value = platform
    event.get_self_id.return_value = self_id
    if message_id is None:
        event.message_obj = None
    else:
        event.message_obj = MagicMock(message_id=message_id)
    return event


def make_adapter(
    group_id: str = "g1",
    user_id: str = "u1",
    is_group: bool = True,
    forwards: list | None = None,
):
    """构造 mock 平台适配器（get_forward_messages 为异步方法）"""
    adapter = MagicMock()
    adapter.is_group_message.return_value = is_group
    adapter.get_group_id.return_value = group_id
    adapter.get_user_id.return_value = user_id
    adapter.get_user_name.return_value = "张三"
    adapter.get_group_name.return_value = "测试群"
    adapter.get_forward_messages = AsyncMock(return_value=forwards or [])
    return adapter


def insert_sample(
    storage: PersonaEvolutionStorage,
    *,
    group_id: str = "g1",
    user_id: str = "u1",
    text: str = "样本",
    dedupe_hash: str | None = None,
    message_id: str | None = None,
    created_at: float | None = None,
) -> int | None:
    """测试辅助：直接插入一条语料"""
    import hashlib

    if dedupe_hash is None:
        dedupe_hash = hashlib.sha256(
            f"{group_id}|{user_id}|{text}|{created_at}".encode()
        ).hexdigest()
    return storage.insert_sample(
        platform="aiocqhttp",
        group_id=group_id,
        group_name=f"群{group_id}",
        user_id=user_id,
        user_name=f"用户{user_id}",
        normalized_text=text,
        dedupe_hash=dedupe_hash,
        message_id=message_id,
        created_at=created_at,
    )


def make_job(
    storage: PersonaEvolutionStorage,
    persona_id: str = "p1",
    **overrides: Any,
) -> int:
    """测试辅助：创建一个 Job 并返回 id"""
    fields: dict = {"persona_id": persona_id, "name": f"Job-{persona_id}"}
    fields.update(overrides)
    return storage.create_job(EvolutionJob(**fields))


def seed_samples(
    storage: PersonaEvolutionStorage,
    count: int,
    *,
    group_id: str = "g1",
    user_id: str = "u1",
    text_prefix: str = "今晚去哪里吃好吃的呢",
    start: int = 0,
) -> None:
    """测试辅助：批量插入互不重复的有效语料（start 为序号偏移）"""
    for i in range(start, start + count):
        insert_sample(
            storage,
            group_id=group_id,
            user_id=user_id,
            text=f"{text_prefix} 第{i}条",
        )


@pytest.fixture
def persona_manager() -> FakePersonaManager:
    return FakePersonaManager()


@pytest.fixture
def llm() -> FakeLLMManager:
    return FakeLLMManager()


@pytest.fixture
def service(
    config,
    storage: PersonaEvolutionStorage,
    persona_manager: FakePersonaManager,
    llm: FakeLLMManager,
):
    """装配好的 PersonaEvolutionService（真实 storage + fake 外部依赖）"""

    def factory(
        persona_id: str = "p1",
        initial_prompt: Optional[str] = None,
    ) -> PersonaEvolutionService:
        if initial_prompt is not None:
            persona_manager.add_persona(persona_id, initial_prompt)
        return PersonaEvolutionService(
            storage,
            FakeContext(persona_manager),
            llm_manager=llm,
        )

    return factory

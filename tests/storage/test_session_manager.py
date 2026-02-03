"""
测试会话管理器
测试 iris_memory.storage.session_manager 中的 SessionManager 类
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta

from iris_memory.storage.session_manager import SessionManager
from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer
from iris_memory.core.defaults import DEFAULTS


@pytest.fixture
def session_manager():
    """创建会话管理器实例"""
    return SessionManager()


@pytest.fixture
def sample_memory():
    """创建示例记忆"""
    return Memory(
        id="mem_1",
        user_id="user_1",
        content="测试记忆",
        type="fact",
        storage_layer=StorageLayer.WORKING,
        created_time=datetime.now()
    )


@pytest.fixture
def sample_memory_with_group():
    """创建带群组的示例记忆"""
    return Memory(
        id="mem_2",
        user_id="user_1",
        group_id="group_1",
        content="群组记忆",
        type="fact",
        storage_layer=StorageLayer.WORKING,
        created_time=datetime.now()
    )


class TestSessionManagerInit:
    """测试SessionManager初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        manager = SessionManager()

        assert manager.working_memory_cache == {}
        assert manager.session_metadata == {}
        assert manager.max_working_memory == DEFAULTS.memory.max_working_memory

    def test_init_custom_max(self):
        """测试自定义最大工作记忆数"""
        manager = SessionManager()
        manager.max_working_memory = 5

        assert manager.max_working_memory == 5


class TestGetSessionKey:
    """测试get_session_key方法"""

    def test_get_session_key_private(self, session_manager):
        """测试私聊会话键生成"""
        key = session_manager.get_session_key("user_1", None)

        assert key == "user_1:private"

    def test_get_session_key_group(self, session_manager):
        """测试群聊会话键生成"""
        key = session_manager.get_session_key("user_1", "group_1")

        assert key == "user_1:group_1"

    def test_get_session_key_consistency(self, session_manager):
        """测试会话键一致性"""
        key1 = session_manager.get_session_key("user_1", "group_1")
        key2 = session_manager.get_session_key("user_1", "group_1")

        assert key1 == key2


class TestCreateSession:
    """测试create_session方法"""

    def test_create_private_session(self, session_manager):
        """测试创建私聊会话"""
        key = session_manager.create_session("user_1", None)

        assert key == "user_1:private"
        assert key in session_manager.session_metadata
        assert key in session_manager.working_memory_cache

    def test_create_group_session(self, session_manager):
        """测试创建群聊会话"""
        key = session_manager.create_session("user_1", "group_1")

        assert key == "user_1:group_1"
        assert key in session_manager.session_metadata

    def test_create_session_with_initial_data(self, session_manager):
        """测试带初始数据创建会话"""
        initial_data = {"custom_field": "value"}
        key = session_manager.create_session("user_1", None, initial_data)

        metadata = session_manager.session_metadata[key]
        assert metadata["custom_field"] == "value"

    def test_create_session_metadata(self, session_manager):
        """测试创建会话的元数据"""
        key = session_manager.create_session("user_1", None)

        metadata = session_manager.session_metadata[key]
        assert metadata["user_id"] == "user_1"
        assert metadata["group_id"] is None
        assert "created_at" in metadata
        assert "last_active" in metadata
        assert metadata["message_count"] == 0

    def test_create_session_existing(self, session_manager):
        """测试创建已存在的会话"""
        key1 = session_manager.create_session("user_1", None)
        key2 = session_manager.create_session("user_1", None)

        # 应该返回相同的键
        assert key1 == key2

        # 工作记忆缓存应该保持不变
        assert len(session_manager.working_memory_cache[key1]) == 0


class TestGetSession:
    """测试get_session方法"""

    def test_get_existing_session(self, session_manager):
        """测试获取已存在的会话"""
        key = session_manager.create_session("user_1", None)
        metadata = session_manager.get_session(key)

        assert metadata is not None
        assert metadata["user_id"] == "user_1"

    def test_get_nonexistent_session(self, session_manager):
        """测试获取不存在的会话"""
        metadata = session_manager.get_session("user_999:private")

        assert metadata is None


class TestUpdateSessionActivity:
    """测试update_session_activity方法"""

    def test_update_private_session(self, session_manager):
        """测试更新私聊会话活动"""
        key = session_manager.create_session("user_1", None)
        original_time = session_manager.session_metadata[key]["last_active"]
        original_count = session_manager.session_metadata[key]["message_count"]

        # 等待一小段时间确保时间戳变化
        import time
        time.sleep(0.01)

        session_manager.update_session_activity("user_1", None)

        updated_time = session_manager.session_metadata[key]["last_active"]
        updated_count = session_manager.session_metadata[key]["message_count"]

        # 时间和消息计数应该都更新了
        assert updated_time != original_time
        assert updated_count == original_count + 1

    def test_update_group_session(self, session_manager):
        """测试更新群聊会话活动"""
        key = session_manager.create_session("user_1", "group_1")
        original_count = session_manager.session_metadata[key]["message_count"]

        session_manager.update_session_activity("user_1", "group_1")

        updated_count = session_manager.session_metadata[key]["message_count"]
        assert updated_count == original_count + 1

    def test_update_nonexistent_session(self, session_manager):
        """测试更新不存在的会话（应该不报错）"""
        # 不应该抛出异常
        session_manager.update_session_activity("user_999", None)

        # 验证会话没有被创建
        assert session_manager.get_session_count() == 0


class TestAddWorkingMemory:
    """测试add_working_memory方法"""

    @pytest.mark.asyncio
    async def test_add_to_private_session(self, session_manager, sample_memory):
        """测试添加到私聊会话"""
        await session_manager.add_working_memory(sample_memory)

        key = "user_1:private"
        memories = await session_manager.get_working_memory("user_1", None)

        assert len(memories) == 1
        assert memories[0].id == "mem_1"

    @pytest.mark.asyncio
    async def test_add_to_group_session(self, session_manager, sample_memory_with_group):
        """测试添加到群聊会话"""
        await session_manager.add_working_memory(sample_memory_with_group)

        key = "user_1:group_1"
        memories = await session_manager.get_working_memory("user_1", "group_1")

        assert len(memories) == 1
        assert memories[0].id == "mem_2"

    @pytest.mark.asyncio
    async def test_add_auto_create_session(self, session_manager, sample_memory):
        """测试自动创建会话"""
        # 会话不存在
        assert "user_1:private" not in session_manager.session_metadata

        await session_manager.add_working_memory(sample_memory)

        # 会话应该被创建
        assert "user_1:private" in session_manager.session_metadata
        assert "user_1:private" in session_manager.working_memory_cache

    @pytest.mark.asyncio
    async def test_add_multiple_memories(self, session_manager, sample_memory):
        """测试添加多个记忆"""
        for i in range(3):
            memory = Memory(
                id=f"mem_{i}",
                user_id="user_1",
                content=f"记忆{i}",
                type="fact",
                storage_layer=StorageLayer.WORKING,
                created_time=datetime.now()
            )
            await session_manager.add_working_memory(memory)

        memories = await session_manager.get_working_memory("user_1", None)
        assert len(memories) == 3


class TestGetWorkingMemory:
    """测试get_working_memory方法"""

    @pytest.mark.asyncio
    async def test_get_from_private_session(self, session_manager, sample_memory):
        """测试获取私聊会话工作记忆"""
        await session_manager.add_working_memory(sample_memory)

        memories = await session_manager.get_working_memory("user_1", None)

        assert len(memories) == 1
        assert memories[0].id == "mem_1"

    @pytest.mark.asyncio
    async def test_get_from_group_session(self, session_manager, sample_memory_with_group):
        """测试获取群聊会话工作记忆"""
        await session_manager.add_working_memory(sample_memory_with_group)

        memories = await session_manager.get_working_memory("user_1", "group_1")

        assert len(memories) == 1
        assert memories[0].id == "mem_2"

    @pytest.mark.asyncio
    async def test_get_empty_session(self, session_manager):
        """测试获取空会话的工作记忆"""
        memories = await session_manager.get_working_memory("user_1", None)

        assert memories == []

    @pytest.mark.asyncio
    async def test_get_different_users(self, session_manager):
        """测试获取不同用户的工作记忆"""
        mem1 = Memory(
            id="mem_1",
            user_id="user_1",
            content="用户1的记忆",
            type="fact",
            storage_layer=StorageLayer.WORKING,
            created_time=datetime.now()
        )
        mem2 = Memory(
            id="mem_2",
            user_id="user_2",
            content="用户2的记忆",
            type="fact",
            storage_layer=StorageLayer.WORKING,
            created_time=datetime.now()
        )

        await session_manager.add_working_memory(mem1)
        await session_manager.add_working_memory(mem2)

        memories1 = await session_manager.get_working_memory("user_1", None)
        memories2 = await session_manager.get_working_memory("user_2", None)

        assert len(memories1) == 1
        assert len(memories2) == 1
        assert memories1[0].id == "mem_1"
        assert memories2[0].id == "mem_2"


class TestClearWorkingMemory:
    """测试clear_working_memory方法"""

    @pytest.mark.asyncio
    async def test_clear_private_session(self, session_manager, sample_memory):
        """测试清除私聊会话工作记忆"""
        await session_manager.add_working_memory(sample_memory)
        assert len(await session_manager.get_working_memory("user_1", None)) == 1

        await session_manager.clear_working_memory("user_1", None)

        assert len(await session_manager.get_working_memory("user_1", None)) == 0

    @pytest.mark.asyncio
    async def test_clear_group_session(self, session_manager, sample_memory_with_group):
        """测试清除群聊会话工作记忆"""
        await session_manager.add_working_memory(sample_memory_with_group)
        assert len(await session_manager.get_working_memory("user_1", "group_1")) == 1

        await session_manager.clear_working_memory("user_1", "group_1")

        assert len(await session_manager.get_working_memory("user_1", "group_1")) == 0

    @pytest.mark.asyncio
    async def test_clear_nonexistent_session(self, session_manager):
        """测试清除不存在的会话（应该不报错）"""
        # 不应该抛出异常
        await session_manager.clear_working_memory("user_999", None)


class TestDeleteSession:
    """测试delete_session方法"""

    @pytest.mark.asyncio
    async def test_delete_existing_session(self, session_manager, sample_memory):
        """测试删除已存在的会话"""
        await session_manager.add_working_memory(sample_memory)
        key = "user_1:private"

        assert key in session_manager.session_metadata
        assert key in session_manager.working_memory_cache

        result = session_manager.delete_session(key)

        assert result is True
        assert key not in session_manager.session_metadata
        assert key not in session_manager.working_memory_cache

    def test_delete_nonexistent_session(self, session_manager):
        """测试删除不存在的会话"""
        result = session_manager.delete_session("user_999:private")

        assert result is False


class TestGetAllSessions:
    """测试get_all_sessions方法"""

    def test_get_all_empty(self, session_manager):
        """测试获取所有空会话"""
        sessions = session_manager.get_all_sessions()

        assert sessions == {}

    def test_get_all_multiple(self, session_manager):
        """测试获取多个会话"""
        session_manager.create_session("user_1", None)
        session_manager.create_session("user_2", None)
        session_manager.create_session("user_1", "group_1")

        sessions = session_manager.get_all_sessions()

        assert len(sessions) == 3
        assert all(isinstance(v, dict) for v in sessions.values())
        # 验证返回结构包含 metadata 和 working_memories
        for session_key, session_data in sessions.items():
            assert "metadata" in session_data
            assert "working_memories" in session_data


class TestGetSessionCount:
    """测试get_session_count方法"""

    def test_count_empty(self, session_manager):
        """测试空会话计数"""
        count = session_manager.get_session_count()

        assert count == 0

    def test_count_multiple(self, session_manager):
        """测试多个会话计数"""
        session_manager.create_session("user_1", None)
        session_manager.create_session("user_2", None)
        session_manager.create_session("user_3", "group_1")

        count = session_manager.get_session_count()

        assert count == 3


class TestLRUWorkingMemory:
    """测试工作记忆LRU机制"""

    @pytest.mark.asyncio
    async def test_lru_exceeded_max(self, session_manager):
        """测试超过最大数量时的LRU淘汰"""
        manager = SessionManager()
        manager.set_max_working_memory(3)

        # 添加5个记忆
        for i in range(5):
            memory = Memory(
                id=f"mem_{i}",
                user_id="user_1",
                content=f"记忆{i}",
                type="fact",
                storage_layer=StorageLayer.WORKING,
                created_time=datetime.now()
            )
            await manager.add_working_memory(memory)

        memories = await manager.get_working_memory("user_1", None)

        # 应该只保留最后3个
        assert len(memories) == 3
        assert memories[0].id == "mem_2"
        assert memories[1].id == "mem_3"
        assert memories[2].id == "mem_4"


class TestSerialization:
    """测试序列化和反序列化"""

    @pytest.mark.asyncio
    async def test_serialize_empty(self, session_manager):
        """测试序列化空会话"""
        data = await session_manager.serialize_for_kv_storage()

        assert "sessions" in data
        assert "working_memory" in data
        assert data["sessions"] == {}
        assert data["working_memory"] == {}

    @pytest.mark.asyncio
    async def test_serialize_with_data(self, session_manager, sample_memory):
        """测试序列化带数据的会话"""
        await session_manager.add_working_memory(sample_memory)

        data = await session_manager.serialize_for_kv_storage()

        assert "user_1:private" in data["sessions"]
        assert "user_1:private" in data["working_memory"]

    @pytest.mark.asyncio
    async def test_deserialize_empty(self, session_manager):
        """测试反序列化空数据"""
        data = {"sessions": {}, "working_memory": {}}

        await session_manager.deserialize_from_kv_storage(data)

        assert session_manager.get_session_count() == 0

    @pytest.mark.asyncio
    async def test_deserialize_with_data(self, session_manager):
        """测试反序列化带数据的会话"""
        # 构建序列化数据
        data = {
            "sessions": {
                "user_1:private": {
                    "user_id": "user_1",
                    "group_id": None,
                    "created_at": datetime.now().isoformat(),
                    "last_active": datetime.now().isoformat(),
                    "message_count": 5
                }
            },
            "working_memory": {
                "user_1:private": [
                    Memory(
                        id="mem_1",
                        user_id="user_1",
                        content="测试记忆",
                        type="fact",
                        storage_layer=StorageLayer.WORKING,
                        created_time=datetime.now()
                    ).to_dict()
                ]
            }
        }

        await session_manager.deserialize_from_kv_storage(data)

        # 验证反序列化结果
        assert session_manager.get_session_count() == 1
        memories = await session_manager.get_working_memory("user_1", None)
        assert len(memories) == 1
        assert memories[0].id == "mem_1"


class TestSetMaxWorkingMemory:
    """测试set_max_working_memory方法"""

    def test_set_max(self, session_manager):
        """测试设置最大工作记忆数"""
        session_manager.set_max_working_memory(20)

        assert session_manager.max_working_memory == 20


class TestCleanExpiredWorkingMemory:
    """测试clean_expired_working_memory方法"""

    def test_clean_empty(self, session_manager):
        """测试清理空会话"""
        session_manager.clean_expired_working_memory()

        # 不应该报错
        assert session_manager.get_session_count() == 0

    @pytest.mark.asyncio
    async def test_clean_no_expired(self, session_manager, sample_memory):
        """测试清理没有过期的记忆"""
        await session_manager.add_working_memory(sample_memory)

        # 清理0小时前（清理所有过期记忆）
        session_manager.clean_expired_working_memory(hours=0)

        # 所有记忆都应该被保留（因为是刚创建的）
        memories = await session_manager.get_working_memory("user_1", None)
        assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_clean_expired(self, session_manager):
        """测试清理过期记忆"""
        # 添加一个旧记忆
        old_time = datetime.now() - timedelta(hours=25)
        old_memory = Memory(
            id="old_mem",
            user_id="user_1",
            content="旧记忆",
            type="fact",
            storage_layer=StorageLayer.WORKING,
            created_time=old_time
        )

        # 添加一个新记忆
        new_memory = Memory(
            id="new_mem",
            user_id="user_1",
            content="新记忆",
            type="fact",
            storage_layer=StorageLayer.WORKING,
            created_time=datetime.now()
        )

        await session_manager.add_working_memory(old_memory)
        await session_manager.add_working_memory(new_memory)

        # 清理24小时前的记忆
        session_manager.clean_expired_working_memory(hours=24)

        memories = await session_manager.get_working_memory("user_1", None)

        # 应该只保留新记忆
        assert len(memories) == 1
        assert memories[0].id == "new_mem"


class TestSessionIsolation:
    """测试会话隔离"""

    @pytest.mark.asyncio
    async def test_private_and_group_isolated(
        self, session_manager, sample_memory, sample_memory_with_group
    ):
        """测试私聊和群聊会话隔离"""
        await session_manager.add_working_memory(sample_memory)
        await session_manager.add_working_memory(sample_memory_with_group)

        private_memories = await session_manager.get_working_memory("user_1", None)
        group_memories = await session_manager.get_working_memory("user_1", "group_1")

        assert len(private_memories) == 1
        assert len(group_memories) == 1
        assert private_memories[0].id == "mem_1"
        assert group_memories[0].id == "mem_2"

    @pytest.mark.asyncio
    async def test_user_isolation(self, session_manager):
        """测试用户隔离"""
        mem1 = Memory(
            id="mem_1",
            user_id="user_1",
            content="用户1的记忆",
            type="fact",
            storage_layer=StorageLayer.WORKING,
            created_time=datetime.now()
        )
        mem2 = Memory(
            id="mem_2",
            user_id="user_2",
            content="用户2的记忆",
            type="fact",
            storage_layer=StorageLayer.WORKING,
            created_time=datetime.now()
        )

        await session_manager.add_working_memory(mem1)
        await session_manager.add_working_memory(mem2)

        memories1 = await session_manager.get_working_memory("user_1", None)
        memories2 = await session_manager.get_working_memory("user_2", None)

        # 验证用户隔离
        assert len(memories1) == 1
        assert len(memories2) == 1
        assert memories1[0].user_id == "user_1"
        assert memories2[0].user_id == "user_2"

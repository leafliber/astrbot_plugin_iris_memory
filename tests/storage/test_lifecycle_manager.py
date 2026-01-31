"""
LifecycleManager测试
测试会话生命周期管理器的核心功能
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from iris_memory.storage.lifecycle_manager import (
    SessionLifecycleManager,
    SessionState,
    LifecycleManager
)
from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer


@pytest.fixture
def mock_session_manager():
    """模拟会话管理器"""
    manager = Mock()
    manager.get_session_key = Mock(side_effect=lambda user_id, group_id: f"{user_id}:{group_id or 'private'}")
    manager.update_session_activity = Mock()
    manager.get_working_memory = Mock(return_value=[])
    manager.clear_working_memory = Mock()
    manager.delete_session = Mock()
    return manager


@pytest.fixture
def lifecycle_manager(mock_session_manager):
    """LifecycleManager实例"""
    return SessionLifecycleManager(
        session_manager=mock_session_manager,
        cleanup_interval=10,  # 10秒用于快速测试
        session_timeout=60,  # 60秒
        inactive_timeout=30  # 30秒
    )


@pytest.mark.asyncio
async def test_session_state_enum():
    """测试会话状态枚举"""
    assert SessionState.ACTIVE.value == "active"
    assert SessionState.INACTIVE.value == "inactive"
    assert SessionState.CLOSED.value == "closed"
    assert SessionState.ARCHIVED.value == "archived"


class TestLifecycleManagerInit:
    """测试初始化功能"""
    
    def test_init_basic(self, lifecycle_manager, mock_session_manager):
        """测试基本初始化"""
        assert lifecycle_manager.session_manager == mock_session_manager
        assert lifecycle_manager.cleanup_interval == 10
        assert lifecycle_manager.session_timeout == 60
        assert lifecycle_manager.inactive_timeout == 30
        assert lifecycle_manager.is_running is False
        assert lifecycle_manager.cleanup_task is None
        assert len(lifecycle_manager.session_states) == 0
    
    def test_init_default_values(self, mock_session_manager):
        """测试默认值"""
        manager = SessionLifecycleManager(
            session_manager=mock_session_manager
        )
        
        assert manager.cleanup_interval == 3600
        assert manager.session_timeout == 86400
        assert manager.inactive_timeout == 1800


class TestLifecycleManagerStartStop:
    """测试启动和停止功能"""
    
    @pytest.mark.asyncio
    async def test_start_success(self, lifecycle_manager):
        """测试成功启动"""
        await lifecycle_manager.start()
        
        assert lifecycle_manager.is_running is True
        assert lifecycle_manager.cleanup_task is not None
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, lifecycle_manager):
        """测试重复启动"""
        await lifecycle_manager.start()
        
        # 第二次启动应该被忽略
        await lifecycle_manager.start()
        
        assert lifecycle_manager.is_running is True
    
    @pytest.mark.asyncio
    async def test_stop_success(self, lifecycle_manager):
        """测试成功停止"""
        await lifecycle_manager.start()
        
        await lifecycle_manager.stop()
        
        assert lifecycle_manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_stop_without_start(self, lifecycle_manager):
        """测试未启动就停止"""
        await lifecycle_manager.stop()
        
        assert lifecycle_manager.is_running is False
        assert lifecycle_manager.cleanup_task is None


class TestLifecycleManagerCleanupLoop:
    """测试清理循环功能"""
    
    @pytest.mark.asyncio
    async def test_cleanup_loop_basic(self, lifecycle_manager):
        """测试基本清理循环"""
        # 添加一个过期的ACTIVE会话
        session_key = "user_123:private"
        lifecycle_manager.session_states[session_key] = {
            'state': SessionState.ACTIVE,
            'last_active': datetime.now() - timedelta(seconds=35),  # 超过inactive_timeout
            'last_updated': datetime.now()
        }
        
        await lifecycle_manager._cleanup_expired_sessions()
        
        # 应该转换为INACTIVE
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.INACTIVE
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_to_archived(self, lifecycle_manager):
        """测试INACTIVE到ARCHIVED的转换"""
        # 添加一个需要归档的会话
        session_key = "user_123:private"
        lifecycle_manager.session_states[session_key] = {
            'state': SessionState.INACTIVE,
            'last_active': datetime.now() - timedelta(seconds=65),  # 超过session_timeout
            'last_updated': datetime.now()
        }
        
        # Mock归档返回True
        lifecycle_manager._archive_session = AsyncMock(return_value=True)
        
        await lifecycle_manager._cleanup_expired_sessions()
        
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.ARCHIVED
        lifecycle_manager._archive_session.assert_called_once_with(session_key)
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_to_closed(self, lifecycle_manager):
        """测试INACTIVE到CLOSED的转换（归档失败）"""
        session_key = "user_123:private"
        lifecycle_manager.session_states[session_key] = {
            'state': SessionState.INACTIVE,
            'last_active': datetime.now() - timedelta(seconds=65),
            'last_updated': datetime.now()
        }
        
        # Mock归档返回False
        lifecycle_manager._archive_session = AsyncMock(return_value=False)
        lifecycle_manager._cleanup_session_data = AsyncMock()
        
        await lifecycle_manager._cleanup_expired_sessions()
        
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.CLOSED
        lifecycle_manager._archive_session.assert_called_once_with(session_key)
        lifecycle_manager._cleanup_session_data.assert_called_once_with(session_key)
    
    @pytest.mark.asyncio
    async def test_cleanup_closed_deleted(self, lifecycle_manager):
        """测试CLOSED会话被删除"""
        session_key = "user_123:private"
        lifecycle_manager.session_states[session_key] = {
            'state': SessionState.CLOSED,
            'last_active': datetime.now() - timedelta(seconds=130),  # 超过session_timeout * 2
            'last_updated': datetime.now()
        }
        
        lifecycle_manager._delete_session = Mock()
        
        await lifecycle_manager._cleanup_expired_sessions()
        
        # 会话应该从状态字典中删除
        assert session_key not in lifecycle_manager.session_states
        lifecycle_manager._delete_session.assert_called_once_with(session_key)
    
    @pytest.mark.asyncio
    async def test_cleanup_archived_deleted(self, lifecycle_manager):
        """测试ARCHIVED会话被删除"""
        session_key = "user_123:private"
        lifecycle_manager.session_states[session_key] = {
            'state': SessionState.ARCHIVED,
            'last_active': datetime.now() - timedelta(seconds=130),
            'last_updated': datetime.now()
        }
        
        lifecycle_manager._delete_session = Mock()
        
        await lifecycle_manager._cleanup_expired_sessions()
        
        assert session_key not in lifecycle_manager.session_states
        lifecycle_manager._delete_session.assert_called_once_with(session_key)
    
    @pytest.mark.asyncio
    async def test_cleanup_multiple_sessions(self, lifecycle_manager):
        """测试清理多个会话"""
        # 添加多个不同状态的会话
        lifecycle_manager.session_states["user_1:private"] = {
            'state': SessionState.ACTIVE,
            'last_active': datetime.now() - timedelta(seconds=35),
            'last_updated': datetime.now()
        }
        lifecycle_manager.session_states["user_2:private"] = {
            'state': SessionState.INACTIVE,
            'last_active': datetime.now() - timedelta(seconds=65),
            'last_updated': datetime.now()
        }
        lifecycle_manager.session_states["user_3:private"] = {
            'state': SessionState.CLOSED,
            'last_active': datetime.now() - timedelta(seconds=130),
            'last_updated': datetime.now()
        }
        
        lifecycle_manager._archive_session = AsyncMock(return_value=False)
        lifecycle_manager._cleanup_session_data = AsyncMock()
        lifecycle_manager._delete_session = Mock()
        
        await lifecycle_manager._cleanup_expired_sessions()
        
        assert lifecycle_manager.session_states["user_1:private"]['state'] == SessionState.INACTIVE
        assert lifecycle_manager.session_states["user_2:private"]['state'] == SessionState.CLOSED
        assert "user_3:private" not in lifecycle_manager.session_states


class TestLifecycleManagerUpdateSessionState:
    """测试更新会话状态功能"""
    
    def test_update_session_state_new(self, lifecycle_manager):
        """测试更新新会话状态"""
        session_key = "user_123:private"
        
        lifecycle_manager._update_session_state(session_key, SessionState.ACTIVE)
        
        assert session_key in lifecycle_manager.session_states
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.ACTIVE
        assert 'last_updated' in lifecycle_manager.session_states[session_key]
    
    def test_update_session_state_existing(self, lifecycle_manager):
        """测试更新现有会话状态"""
        session_key = "user_123:private"
        lifecycle_manager.session_states[session_key] = {
            'state': SessionState.INACTIVE,
            'last_active': datetime.now(),
            'last_updated': datetime.now()
        }
        
        lifecycle_manager._update_session_state(session_key, SessionState.ACTIVE)
        
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.ACTIVE
        assert isinstance(lifecycle_manager.session_states[session_key]['last_updated'], datetime)


class TestLifecycleManagerArchiveSession:
    """测试归档会话功能"""
    
    @pytest.mark.asyncio
    async def test_archive_session_success(self, lifecycle_manager):
        """测试成功归档会话"""
        # Mock工作记忆
        from iris_memory.core.types import StorageLayer
        from unittest.mock import Mock, MagicMock
        
        memory1 = Mock()
        memory1.should_upgrade_to_episodic = Mock(return_value=True)
        memory1.storage_layer = StorageLayer.WORKING
        memory1.id = "mem_1"
        
        memory2 = Mock()
        memory2.should_upgrade_to_episodic = Mock(return_value=False)
        
        lifecycle_manager.session_manager.get_working_memory = Mock(return_value=[memory1, memory2])
        
        session_key = "user_123:group_456"
        result = await lifecycle_manager._archive_session(session_key)
        
        assert result is True
        # 验证memory1的storage_layer被修改为EPISODIC
        assert memory1.storage_layer == StorageLayer.EPISODIC
        # 验证memory2的storage_layer未被修改
        assert memory2.should_upgrade_to_episodic.called
    
    @pytest.mark.asyncio
    async def test_archive_session_no_upgrades(self, lifecycle_manager):
        """测试没有记忆升级的归档"""
        memory = Mock()
        memory.should_upgrade_to_episodic = Mock(return_value=False)
        
        lifecycle_manager.session_manager.get_working_memory = Mock(return_value=[memory])
        
        session_key = "user_123:group_456"
        result = await lifecycle_manager._archive_session(session_key)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_archive_session_invalid_key(self, lifecycle_manager):
        """测试无效的会话键"""
        session_key = "invalid_key"
        result = await lifecycle_manager._archive_session(session_key)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_archive_session_private_chat(self, lifecycle_manager):
        """测试私聊会话归档"""
        memory = Mock()
        memory.should_upgrade_to_episodic = Mock(return_value=True)
        
        lifecycle_manager.session_manager.get_working_memory = Mock(return_value=[memory])
        
        session_key = "user_123:private"
        result = await lifecycle_manager._archive_session(session_key)
        
        assert result is True
        lifecycle_manager.session_manager.get_working_memory.assert_called_once_with("user_123", None)


class TestLifecycleManagerCleanupSessionData:
    """测试清理会话数据功能"""
    
    @pytest.mark.asyncio
    async def test_cleanup_session_data_group_chat(self, lifecycle_manager):
        """测试清理群聊会话数据"""
        session_key = "user_123:group_456"
        
        await lifecycle_manager._cleanup_session_data(session_key)
        
        lifecycle_manager.session_manager.clear_working_memory.assert_called_once_with("user_123", "group_456")
    
    @pytest.mark.asyncio
    async def test_cleanup_session_data_private_chat(self, lifecycle_manager):
        """测试清理私聊会话数据"""
        session_key = "user_123:private"
        
        await lifecycle_manager._cleanup_session_data(session_key)
        
        lifecycle_manager.session_manager.clear_working_memory.assert_called_once_with("user_123", None)


class TestLifecycleManagerDeleteSession:
    """测试删除会话功能"""
    
    def test_delete_session_group_chat(self, lifecycle_manager):
        """测试删除群聊会话"""
        session_key = "user_123:group_456"
        
        lifecycle_manager._delete_session(session_key)
        
        lifecycle_manager.session_manager.delete_session.assert_called_once_with("user_123", "group_456")
    
    def test_delete_session_private_chat(self, lifecycle_manager):
        """测试删除私聊会话"""
        session_key = "user_123:private"
        
        lifecycle_manager._delete_session(session_key)
        
        lifecycle_manager.session_manager.delete_session.assert_called_once_with("user_123", None)


class TestLifecycleManagerActivateSession:
    """测试激活会话功能"""
    
    @pytest.mark.asyncio
    async def test_activate_session_group_chat(self, lifecycle_manager):
        """测试激活群聊会话"""
        session_key = "user_123:group_456"
        
        await lifecycle_manager.activate_session("user_123", "group_456")
        
        lifecycle_manager.session_manager.update_session_activity.assert_called_once_with("user_123", "group_456")
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.ACTIVE
    
    @pytest.mark.asyncio
    async def test_activate_session_private_chat(self, lifecycle_manager):
        """测试激活私聊会话"""
        session_key = "user_123:private"
        
        await lifecycle_manager.activate_session("user_123", None)
        
        lifecycle_manager.session_manager.update_session_activity.assert_called_once_with("user_123", None)
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.ACTIVE


class TestLifecycleManagerDeactivateSession:
    """测试停用会话功能"""
    
    @pytest.mark.asyncio
    async def test_deactivate_session(self, lifecycle_manager):
        """测试停用会话"""
        session_key = "user_123:group_456"
        
        await lifecycle_manager.deactivate_session("user_123", "group_456")
        
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.INACTIVE


class TestLifecycleManagerCloseSession:
    """测试关闭会话功能"""
    
    @pytest.mark.asyncio
    async def test_close_session(self, lifecycle_manager):
        """测试关闭会话"""
        session_key = "user_123:group_456"
        
        lifecycle_manager._cleanup_session_data = AsyncMock()
        
        await lifecycle_manager.close_session("user_123", "group_456")
        
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.CLOSED
        lifecycle_manager._cleanup_session_data.assert_called_once_with(session_key)


class TestLifecycleManagerGetSessionState:
    """测试获取会话状态功能"""
    
    def test_get_session_state_active(self, lifecycle_manager):
        """测试获取活跃会话状态"""
        session_key = "user_123:group_456"
        lifecycle_manager.session_states[session_key] = {
            'state': SessionState.ACTIVE,
            'last_active': datetime.now(),
            'last_updated': datetime.now()
        }
        
        state = lifecycle_manager.get_session_state("user_123", "group_456")
        
        assert state == SessionState.ACTIVE
    
    def test_get_session_state_not_found(self, lifecycle_manager):
        """测试获取不存在的会话状态"""
        state = lifecycle_manager.get_session_state("user_999", "group_999")
        
        assert state is None


class TestLifecycleManagerGetStatistics:
    """测试获取统计信息功能"""
    
    def test_get_statistics_empty(self, lifecycle_manager):
        """测试空会话的统计"""
        stats = lifecycle_manager.get_session_statistics()
        
        assert stats['total_sessions'] == 0
        assert stats['active_sessions'] == 0
        assert stats['inactive_sessions'] == 0
        assert stats['closed_sessions'] == 0
        assert stats['archived_sessions'] == 0
    
    def test_get_statistics_mixed(self, lifecycle_manager):
        """测试混合状态的统计"""
        lifecycle_manager.session_states["user_1:private"] = {
            'state': SessionState.ACTIVE
        }
        lifecycle_manager.session_states["user_2:private"] = {
            'state': SessionState.ACTIVE
        }
        lifecycle_manager.session_states["user_3:private"] = {
            'state': SessionState.INACTIVE
        }
        lifecycle_manager.session_states["user_4:private"] = {
            'state': SessionState.CLOSED
        }
        lifecycle_manager.session_states["user_5:private"] = {
            'state': SessionState.ARCHIVED
        }
        
        stats = lifecycle_manager.get_session_statistics()
        
        assert stats['total_sessions'] == 5
        assert stats['active_sessions'] == 2
        assert stats['inactive_sessions'] == 1
        assert stats['closed_sessions'] == 1
        assert stats['archived_sessions'] == 1


class TestLifecycleManagerSerializeState:
    """测试序列化状态功能"""
    
    @pytest.mark.asyncio
    async def test_serialize_state(self, lifecycle_manager):
        """测试序列化状态"""
        session_key = "user_123:group_456"
        now = datetime.now()
        lifecycle_manager.session_states[session_key] = {
            'state': SessionState.ACTIVE,
            'last_active': now,
            'last_updated': now
        }
        
        serialized = await lifecycle_manager.serialize_state()
        
        assert session_key in serialized
        assert serialized[session_key]['state'] == 'active'
        assert 'last_active' in serialized[session_key]
        assert 'last_updated' in serialized[session_key]


class TestLifecycleManagerDeserializeState:
    """测试反序列化状态功能"""
    
    @pytest.mark.asyncio
    async def test_deserialize_state(self, lifecycle_manager):
        """测试反序列化状态"""
        data = {
            "user_123:group_456": {
                "state": "active",
                "last_active": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "user_456:private": {
                "state": "inactive",
                "last_active": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        await lifecycle_manager.deserialize_state(data)
        
        assert len(lifecycle_manager.session_states) == 2
        assert lifecycle_manager.session_states["user_123:group_456"]['state'] == SessionState.ACTIVE
        assert lifecycle_manager.session_states["user_456:private"]['state'] == SessionState.INACTIVE
    
    @pytest.mark.asyncio
    async def test_deserialize_state_with_defaults(self, lifecycle_manager):
        """测试带默认值的反序列化"""
        data = {
            "user_123:group_456": {
                "state": "invalid_state",  # 无效状态
                "last_active": "invalid_date"  # 无效日期
            }
        }
        
        # 不应该抛出异常，使用默认值
        await lifecycle_manager.deserialize_state(data)
        
        assert "user_123:group_456" in lifecycle_manager.session_states


class TestLifecycleManagerBackwardCompatibility:
    """测试向后兼容性"""
    
    def test_lifecycle_manager_alias(self):
        """测试LifecycleManager别名"""
        assert LifecycleManager == SessionLifecycleManager
    
    @pytest.mark.asyncio
    async def test_create_with_alias(self, mock_session_manager):
        """测试使用别名创建"""
        manager = LifecycleManager(
            session_manager=mock_session_manager,
            cleanup_interval=10
        )
        
        assert isinstance(manager, SessionLifecycleManager)
        assert manager.cleanup_interval == 10


class TestLifecycleManagerIntegration:
    """测试集成场景"""
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_workflow(self, lifecycle_manager):
        """测试完整的生命周期工作流"""
        # 1. 激活会话
        await lifecycle_manager.activate_session("user_123", "group_456")
        session_key = "user_123:group_456"
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.ACTIVE
        
        # 2. 更新活动时间（模拟继续活跃）
        lifecycle_manager.session_states[session_key]['last_active'] = datetime.now()
        
        # 3. 停用会话
        await lifecycle_manager.deactivate_session("user_123", "group_456")
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.INACTIVE
        
        # 4. 模拟超时后清理
        lifecycle_manager.session_states[session_key]['last_active'] = \
            datetime.now() - timedelta(seconds=70)  # 超过session_timeout
        
        lifecycle_manager._archive_session = AsyncMock(return_value=False)
        lifecycle_manager._cleanup_session_data = AsyncMock()
        
        await lifecycle_manager._cleanup_expired_sessions()
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.CLOSED
        
        # 5. 删除会话
        lifecycle_manager._delete_session = Mock()
        lifecycle_manager.session_states[session_key]['last_active'] = \
            datetime.now() - timedelta(seconds=140)  # 超过session_timeout * 2
        
        await lifecycle_manager._cleanup_expired_sessions()
        assert session_key not in lifecycle_manager.session_states

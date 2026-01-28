"""
会话生命周期管理器
管理会话的状态转换和同步
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any

from iris_memory.utils.logger import logger


class SessionState(str, Enum):
    """会话状态"""
    
    # 活跃状态：会话正在进行中
    ACTIVE = "active"
    
    # 非活跃状态：会话存在但暂时未活动
    INACTIVE = "inactive"
    
    # 关闭状态：会话已结束
    CLOSED = "closed"
    
    # 归档状态：会话数据已归档
    ARCHIVED = "archived"


class SessionLifecycleManager:
    """会话生命周期管理器
    
    管理会话的状态转换、清理和持久化
    """
    
    def __init__(
        self,
        session_manager,
        cleanup_interval: int = 3600,  # 清理间隔（秒），默认1小时
        session_timeout: int = 86400,  # 会话超时（秒），默认24小时
        inactive_timeout: int = 1800  # 非活跃超时（秒），默认30分钟
    ):
        """初始化生命周期管理器
        
        Args:
            session_manager: 会话管理器实例
            cleanup_interval: 定时清理间隔
            session_timeout: 会话超时时间
            inactive_timeout: 非活跃超时时间
        """
        self.session_manager = session_manager
        self.cleanup_interval = cleanup_interval
        self.session_timeout = session_timeout
        self.inactive_timeout = inactive_timeout
        
        # 会话状态缓存：{session_key: {"state": SessionState, "last_active": datetime}}
        self.session_states: Dict[str, Dict[str, Any]] = {}
        
        # 定时任务
        self.cleanup_task = None
        self.is_running = False
    
    async def start(self):
        """启动生命周期管理器"""
        if self.is_running:
            logger.warning("SessionLifecycleManager is already running")
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SessionLifecycleManager started")
    
    async def stop(self):
        """停止生命周期管理器"""
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SessionLifecycleManager stopped")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        now = datetime.now()
        cleaned_count = 0
        archived_count = 0
        
        # 遍历所有会话状态
        for session_key, state_info in self.session_states.copy().items():
            session_state = state_info.get("state", SessionState.INACTIVE)
            last_active = state_info.get("last_active", now)
            
            # 计算非活跃时间
            inactive_duration = (now - last_active).total_seconds()
            
            # 状态转换逻辑
            if session_state == SessionState.ACTIVE:
                # 检查是否转为非活跃
                if inactive_duration > self.inactive_timeout:
                    self._update_session_state(
                        session_key, 
                        SessionState.INACTIVE
                    )
                    logger.debug(
                        f"Session {session_key} changed from ACTIVE to INACTIVE"
                    )
            
            elif session_state == SessionState.INACTIVE:
                # 检查是否应该关闭或归档
                if inactive_duration > self.session_timeout:
                    # 尝试归档工作记忆
                    if await self._archive_session(session_key):
                        self._update_session_state(
                            session_key, 
                            SessionState.ARCHIVED
                        )
                        archived_count += 1
                    else:
                        # 无法归档，直接关闭
                        self._update_session_state(
                            session_key, 
                            SessionState.CLOSED
                        )
                        await self._cleanup_session_data(session_key)
                        cleaned_count += 1
                    
                    logger.debug(
                        f"Session {session_key} changed from INACTIVE to "
                        f"{self.session_states[session_key]['state'].value}"
                    )
            
            elif session_state in [SessionState.CLOSED, SessionState.ARCHIVED]:
                # 检查是否应该完全删除
                if inactive_duration > self.session_timeout * 2:  # 48小时后删除
                    self._delete_session(session_key)
                    del self.session_states[session_key]
                    cleaned_count += 1
        
        if cleaned_count > 0 or archived_count > 0:
            logger.info(
                f"Session cleanup completed: {cleaned_count} closed, "
                f"{archived_count} archived"
            )
    
    def _update_session_state(
        self,
        session_key: str,
        new_state: SessionState
    ):
        """更新会话状态
        
        Args:
            session_key: 会话标识符
            new_state: 新状态
        """
        if session_key not in self.session_states:
            self.session_states[session_key] = {}
        
        self.session_states[session_key]["state"] = new_state
        self.session_states[session_key]["last_updated"] = datetime.now()
    
    async def _archive_session(self, session_key: str) -> bool:
        """归档会话
        
        将工作记忆提升到情景记忆或删除
        
        Args:
            session_key: 会话标识符
            
        Returns:
            bool: 是否归档成功
        """
        # 获取工作记忆
        try:
            # 解析session_key获取user_id和group_id
            parts = session_key.split(":")
            if len(parts) == 2:
                user_id, group_id_str = parts
                group_id = group_id_str if group_id_str != "private" else None
            else:
                return False
            
            # 获取工作记忆
            working_memories = self.session_manager.get_working_memory(
                user_id, group_id
            )
            
            # 尝试将重要记忆提升到情景记忆
            upgraded_count = 0
            for memory in working_memories:
                if memory.should_upgrade_to_episodic():
                    # 更改存储层
                    memory.storage_layer = memory.storage_layer.__class__("EPISODIC")
                    upgraded_count += 1
                    logger.debug(
                        f"Memory {memory.id} upgraded from WORKING to EPISODIC"
                    )
            
            if upgraded_count > 0:
                logger.info(
                    f"Archived session {session_key}: {upgraded_count} memories upgraded"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to archive session {session_key}: {e}")
            return False
    
    async def _cleanup_session_data(self, session_key: str):
        """清理会话数据
        
        Args:
            session_key: 会话标识符
        """
        try:
            # 解析session_key获取user_id和group_id
            parts = session_key.split(":")
            if len(parts) == 2:
                user_id, group_id_str = parts
                group_id = group_id_str if group_id_str != "private" else None
                
                # 清除工作记忆
                self.session_manager.clear_working_memory(user_id, group_id)
                
                logger.debug(f"Cleaned up session data: {session_key}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup session {session_key}: {e}")
    
    def _delete_session(self, session_key: str):
        """删除会话
        
        Args:
            session_key: 会话标识符
        """
        try:
            # 解析session_key获取user_id和group_id
            parts = session_key.split(":")
            if len(parts) == 2:
                user_id, group_id_str = parts
                group_id = group_id_str if group_id_str != "private" else None
                
                # 删除会话
                self.session_manager.delete_session(user_id, group_id)
                
                logger.debug(f"Deleted session: {session_key}")
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_key}: {e}")
    
    async def activate_session(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ):
        """激活会话
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        session_key = self.session_manager.get_session_key(user_id, group_id)
        
        # 更新活动时间
        self.session_manager.update_session_activity(user_id, group_id)
        
        # 更新状态
        self._update_session_state(session_key, SessionState.ACTIVE)
        
        logger.debug(f"Session activated: {session_key}")
    
    async def deactivate_session(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ):
        """停用会话（转为非活跃）
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        session_key = self.session_manager.get_session_key(user_id, group_id)
        
        # 仅更新状态为非活跃，不清理数据
        self._update_session_state(session_key, SessionState.INACTIVE)
        
        logger.debug(f"Session deactivated: {session_key}")
    
    async def close_session(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ):
        """关闭会话
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        session_key = self.session_manager.get_session_key(user_id, group_id)
        
        # 清理工作记忆
        await self._cleanup_session_data(session_key)
        
        # 更新状态为关闭
        self._update_session_state(session_key, SessionState.CLOSED)
        
        logger.debug(f"Session closed: {session_key}")
    
    def get_session_state(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ) -> Optional[SessionState]:
        """获取会话状态
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            
        Returns:
            Optional[SessionState]: 会话状态
        """
        session_key = self.session_manager.get_session_key(user_id, group_id)
        state_info = self.session_states.get(session_key)
        
        if state_info:
            return state_info.get("state")
        
        return None
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """获取会话统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            "total_sessions": len(self.session_states),
            "active_sessions": 0,
            "inactive_sessions": 0,
            "closed_sessions": 0,
            "archived_sessions": 0
        }
        
        for state_info in self.session_states.values():
            state = state_info.get("state", SessionState.INACTIVE)
            if state == SessionState.ACTIVE:
                stats["active_sessions"] += 1
            elif state == SessionState.INACTIVE:
                stats["inactive_sessions"] += 1
            elif state == SessionState.CLOSED:
                stats["closed_sessions"] += 1
            elif state == SessionState.ARCHIVED:
                stats["archived_sessions"] += 1
        
        return stats
    
    async def serialize_state(self) -> Dict[str, Any]:
        """序列化会话状态用于持久化
        
        Returns:
            Dict[str, Any]: 序列化的状态数据
        """
        serialized = {}
        for session_key, state_info in self.session_states.items():
            serialized[session_key] = {
                "state": state_info["state"].value,
                "last_active": state_info["last_active"].isoformat(),
                "last_updated": state_info["last_updated"].isoformat()
            }
        
        return serialized
    
    async def deserialize_state(self, data: Dict[str, Any]):
        """反序列化会话状态
        
        Args:
            data: 序列化的状态数据
        """
        for session_key, state_data in data.items():
            self.session_states[session_key] = {
                "state": SessionState(state_data.get("state", "inactive")),
                "last_active": datetime.fromisoformat(
                    state_data.get("last_active", datetime.now().isoformat())
                ),
                "last_updated": datetime.fromisoformat(
                    state_data.get("last_updated", datetime.now().isoformat())
                )
            }
        
        logger.info(f"Loaded {len(self.session_states)} session states")


# 别名，保持向后兼容性
LifecycleManager = SessionLifecycleManager

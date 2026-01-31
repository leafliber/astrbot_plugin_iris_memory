"""
会话管理器
管理私聊和群聊的会话隔离，支持KV存储持久化
"""

from typing import Optional, Dict, Any
from datetime import datetime

from iris_memory.utils.logger import logger

from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer


class SessionManager:
    """会话管理器
    
    管理私聊和群聊的会话隔离：
    - 基于user_id和group_id的双重隔离机制
    - 支持工作记忆缓存
    - 使用KV存储持久化会话状态
    """
    
    def __init__(self):
        """初始化会话管理器"""
        # 工作记忆缓存：{session_key: [Memory]}
        self.working_memory_cache: Dict[str, list[Memory]] = {}
        
        # 会话元数据：{session_key: metadata}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 最大工作记忆数量（可配置）
        self.max_working_memory = 10
    
    def get_session_key(self, user_id: str, group_id: Optional[str] = None) -> str:
        """生成会话标识符
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（私聊时为None）
            
        Returns:
            str: 会话标识符（格式：user_id:group_id 或 user_id:private）
        """
        if group_id:
            return f"{user_id}:{group_id}"
        else:
            return f"{user_id}:private"
    
    def create_session(self, user_id: str, group_id: Optional[str] = None, initial_data: Optional[Dict[str, Any]] = None) -> str:
        """创建新会话

        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            initial_data: 初始会话数据（可选）

        Returns:
            str: 会话标识符
        """
        session_key = self.get_session_key(user_id, group_id)

        if session_key not in self.session_metadata:
            metadata = {
                "user_id": user_id,
                "group_id": group_id,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "message_count": 0,
            }

            # 合并初始数据
            if initial_data:
                metadata.update(initial_data)

            self.session_metadata[session_key] = metadata

            # 初始化工作记忆缓存
            self.working_memory_cache[session_key] = []

            logger.debug(f"Session created: {session_key}")

        return session_key

    def get_session(self, session_key: str) -> Optional[Dict[str, Any]]:
        """获取会话信息

        Args:
            session_key: 会话键

        Returns:
            Dict[str, Any]: 会话元数据，如果不存在则返回None
        """
        return self.session_metadata.get(session_key)
    
    def update_session_activity(self, user_id: str, group_id: Optional[str] = None):
        """更新会话活动时间
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        session_key = self.get_session_key(user_id, group_id)
        
        if session_key in self.session_metadata:
            self.session_metadata[session_key]["last_active"] = datetime.now().isoformat()
            self.session_metadata[session_key]["message_count"] += 1
    
    def add_working_memory(self, memory: Memory):
        """添加工作记忆
        
        Args:
            memory: 记忆对象
        """
        session_key = self.get_session_key(memory.user_id, memory.group_id)
        
        # 确保会话存在
        if session_key not in self.working_memory_cache:
            self.create_session(memory.user_id, memory.group_id)
        
        # 添加到工作记忆
        self.working_memory_cache[session_key].append(memory)
        
        # 限制最大数量（LRU策略：移除最旧的）
        if len(self.working_memory_cache[session_key]) > self.max_working_memory:
            removed = self.working_memory_cache[session_key].pop(0)
            logger.debug(f"Working memory LRU removed: {removed.id}")
    
    def get_working_memory(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ) -> list[Memory]:
        """获取工作记忆
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            
        Returns:
            list[Memory]: 工作记忆列表
        """
        session_key = self.get_session_key(user_id, group_id)
        return self.working_memory_cache.get(session_key, [])
    
    def clear_working_memory(self, user_id: str, group_id: Optional[str] = None):
        """清除工作记忆
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        session_key = self.get_session_key(user_id, group_id)
        if session_key in self.working_memory_cache:
            self.working_memory_cache[session_key] = []
            logger.debug(f"Working memory cleared for session: {session_key}")
    
    def delete_session(self, session_key: str) -> bool:
        """删除会话

        Args:
            session_key: 会话键

        Returns:
            bool: 是否成功删除
        """
        # 检查会话是否存在
        if session_key not in self.session_metadata:
            return False

        # 清除缓存
        if session_key in self.working_memory_cache:
            del self.working_memory_cache[session_key]

        # 清除元数据
        if session_key in self.session_metadata:
            del self.session_metadata[session_key]

        logger.info(f"Session deleted: {session_key}")
        return True
    
    def get_all_sessions(self) -> list[Dict[str, Any]]:
        """获取所有会话信息
        
        Returns:
            list[Dict[str, Any]]: 所有会话的元数据列表
        """
        return list(self.session_metadata.values())
    
    def get_session_count(self) -> int:
        """获取会话数量
        
        Returns:
            int: 会话数量
        """
        return len(self.session_metadata)
    
    async def serialize_for_kv_storage(self) -> Dict[str, Any]:
        """序列化会话数据用于KV存储
        
        Returns:
            Dict[str, Any]: 序列化的会话数据
        """
        return {
            "sessions": self.session_metadata,
            "working_memory": {
                session_key: [m.to_dict() for m in memories]
                for session_key, memories in self.working_memory_cache.items()
            }
        }
    
    async def deserialize_from_kv_storage(self, data: Dict[str, Any]):
        """从KV存储反序列化会话数据
        
        Args:
            data: 序列化的会话数据
        """
        if "sessions" in data:
            self.session_metadata = data["sessions"]
        
        if "working_memory" in data:
            self.working_memory_cache = {}
            from iris_memory.models.memory import Memory
            for session_key, memories_data in data["working_memory"].items():
                self.working_memory_cache[session_key] = [
                    Memory.from_dict(m) for m in memories_data
                ]
        
        logger.info(f"Session data deserialized: {len(self.session_metadata)} sessions")
    
    def set_max_working_memory(self, max_count: int):
        """设置最大工作记忆数量
        
        Args:
            max_count: 最大数量
        """
        self.max_working_memory = max_count
        logger.debug(f"Max working memory set to: {max_count}")
    
    def clean_expired_working_memory(self, hours: int = 24):
        """清理过期的工作记忆
        
        Args:
            hours: 过期时间（小时）
        """
        from datetime import datetime, timedelta
        now = datetime.now()
        cutoff_time = now - timedelta(hours=hours)
        
        cleaned_count = 0
        for session_key, memories in self.working_memory_cache.items():
            # 过滤掉过期的记忆
            valid_memories = [
                m for m in memories
                if m.created_time >= cutoff_time and not m.should_delete_working()
            ]
            removed = len(memories) - len(valid_memories)
            self.working_memory_cache[session_key] = valid_memories
            cleaned_count += removed
        
        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} expired working memories")

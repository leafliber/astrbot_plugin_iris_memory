"""
会话管理器
管理私聊和群聊的会话隔离，支持KV存储持久化
整合工作记忆缓存功能，提供统一的工作记忆管理
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio

from iris_memory.utils.logger import get_logger

from iris_memory.models.memory import Memory
from iris_memory.core.defaults import DEFAULTS

# 模块logger
logger = get_logger("session_manager")


class SessionManager:
    """会话管理器（统一版）
    
    整合原 SessionManager 和 WorkingMemoryCache 的功能：
    - 基于user_id和group_id的双重隔离机制
    - 支持工作记忆缓存（带TTL）
    - 使用KV存储持久化会话状态
    - 提供缓存统计信息
    """
    
    def __init__(
        self,
        max_working_memory: int = None,
        max_sessions: int = None,
        ttl: int = None
    ):
        """初始化会话管理器
        
        Args:
            max_working_memory: 每个会话最大工作记忆数量
            max_sessions: 最大会话数量
            ttl: 工作记忆生存时间（秒），默认24小时
        """
        # 工作记忆缓存：{session_key: [Memory]}
        self.working_memory_cache: Dict[str, List[Memory]] = {}
        
        # 会话元数据：{session_key: metadata}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 配置
        self.max_working_memory = max_working_memory or DEFAULTS.memory.max_working_memory
        self.max_sessions = max_sessions or DEFAULTS.session.max_sessions
        self.ttl = ttl or DEFAULTS.cache.working_cache_ttl
        
        # 会话访问顺序（用于LRU淘汰）
        self._session_order: List[str] = []
        
        # 统计信息
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memories_added": 0,
            "memories_removed": 0
        }
        
        # 异步锁（用于并发安全）
        self._lock = asyncio.Lock()
    
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
    
    async def add_working_memory(self, memory: Memory):
        """添加工作记忆（线程安全）
        
        Args:
            memory: 记忆对象
        """
        async with self._lock:
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
    
    async def get_working_memory(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        update_access: bool = True
    ) -> List[Memory]:
        """获取工作记忆（线程安全）

        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            update_access: 是否更新访问计数（默认True）

        Returns:
            List[Memory]: 工作记忆列表
        """
        async with self._lock:
            session_key = self.get_session_key(user_id, group_id)
            memories = list(self.working_memory_cache.get(session_key, []))

            # 更新每条工作记忆的访问计数
            if update_access and memories:
                for memory in memories:
                    memory.update_access()
                    logger.debug(f"Updated access count for memory {memory.id}: {memory.access_count}")

            return memories
    
    async def clear_working_memory(self, user_id: str, group_id: Optional[str] = None):
        """清除工作记忆（线程安全）
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        async with self._lock:
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
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有会话信息（包含工作记忆）
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有会话的完整数据，包含 working_memories
        """
        result = {}
        for session_key in self.session_metadata:
            result[session_key] = {
                "metadata": self.session_metadata.get(session_key, {}),
                "working_memories": list(self.working_memory_cache.get(session_key, []))
            }
        return result
    
    async def remove_working_memory(self, user_id: str, group_id: Optional[str], memory_id: str) -> bool:
        """从工作记忆中移除指定记忆（线程安全）
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            memory_id: 记忆ID
            
        Returns:
            bool: 是否成功移除
        """
        async with self._lock:
            session_key = self.get_session_key(user_id, group_id)
            if session_key not in self.working_memory_cache:
                return False
            
            memories = self.working_memory_cache[session_key]
            original_count = len(memories)
            
            # 过滤掉指定ID的记忆
            self.working_memory_cache[session_key] = [
                m for m in memories if m.id != memory_id
            ]
            
            removed = len(memories) > len(self.working_memory_cache[session_key])
            if removed:
                logger.debug(f"Removed working memory {memory_id} from session {session_key}")
            
            return removed
    
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
        for session_key, memories in list(self.working_memory_cache.items()):
            # 过滤掉过期的记忆
            if hours == 0:
                # hours=0表示不清理任何记忆
                valid_memories = memories
            else:
                valid_memories = [
                    m for m in memories
                    if m.created_time >= cutoff_time and not m.should_delete_working()
                ]
                removed = len(memories) - len(valid_memories)
                self.working_memory_cache[session_key] = valid_memories
                cleaned_count += removed
                self._stats["memories_removed"] += removed

        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} expired working memories")
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """获取内存使用统计
        
        Returns:
            Dict[str, Any]: 内存使用统计信息
        """
        import sys
        
        total_memories = sum(len(mems) for mems in self.working_memory_cache.values())
        total_sessions = len(self.session_metadata)
        
        # 估算内存使用（粗略估计）
        try:
            # 获取工作记忆缓存的内存占用
            wm_size = sys.getsizeof(self.working_memory_cache)
            for session_key, memories in self.working_memory_cache.items():
                wm_size += sys.getsizeof(session_key)
                for mem in memories:
                    wm_size += sys.getsizeof(mem)
            
            # 获取元数据的内存占用
            meta_size = sys.getsizeof(self.session_metadata)
            for key, val in self.session_metadata.items():
                meta_size += sys.getsizeof(key) + sys.getsizeof(val)
            
            estimated_memory_mb = (wm_size + meta_size) / (1024 * 1024)
        except:
            estimated_memory_mb = 0
        
        return {
            "total_working_memories": total_memories,
            "total_sessions": total_sessions,
            "estimated_memory_mb": round(estimated_memory_mb, 2),
            "max_working_memory": self.max_working_memory,
            "max_sessions": self.max_sessions,
            "cache_hits": self._stats["hits"],
            "cache_misses": self._stats["misses"],
            "evictions": self._stats["evictions"]
        }
    
    async def perform_maintenance(self):
        """执行维护任务
        
        - 清理过期工作记忆
        - 清理过期会话
        - 强制LRU淘汰（如果超过限制）
        """
        logger.info("Starting session manager maintenance...")
        
        # 1. 清理过期工作记忆
        self.clean_expired_working_memory(hours=24)
        
        # 2. 清理过期会话（30天未活动）
        from datetime import datetime, timedelta
        now = datetime.now()
        cutoff = now - timedelta(days=30)
        
        expired_sessions = []
        for session_key, metadata in self.session_metadata.items():
            last_active_str = metadata.get("last_active")
            try:
                last_active = datetime.fromisoformat(last_active_str) if last_active_str else now
                if last_active < cutoff:
                    expired_sessions.append(session_key)
            except:
                pass
        
        for session_key in expired_sessions:
            self.delete_session(session_key)
        
        if expired_sessions:
            logger.info(f"Removed {len(expired_sessions)} expired sessions")
        
        # 3. 强制LRU淘汰（如果超过限制）
        while len(self._session_order) > self.max_sessions:
            oldest_key = self._session_order.pop(0)
            if oldest_key in self.working_memory_cache:
                del self.working_memory_cache[oldest_key]
            if oldest_key in self.session_metadata:
                del self.session_metadata[oldest_key]
            self._stats["evictions"] += 1
            logger.debug(f"LRU evicted session: {oldest_key}")
        
        stats = self.get_memory_usage_stats()
        logger.info(f"Maintenance complete: {stats['total_sessions']} sessions, "
                   f"{stats['total_working_memories']} memories, "
                   f"~{stats['estimated_memory_mb']}MB estimated")
    
    # ========== 异步方法（兼容 WorkingMemoryCache 接口）==========
    
    async def add_memory_async(
        self,
        user_id: str,
        group_id: Optional[str],
        memory: Memory
    ) -> bool:
        """异步添加记忆到工作记忆缓存
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            memory: 记忆对象
            
        Returns:
            bool: 是否添加成功
        """
        async with self._lock:
            await self.add_working_memory(memory)
            self._stats["memories_added"] += 1
            return True
    
    async def get_recent_memories(
        self,
        user_id: str,
        group_id: Optional[str],
        limit: int = 10
    ) -> List[Memory]:
        """获取最近的工作记忆
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            limit: 最大返回数量
            
        Returns:
            List[Memory]: 记忆列表（按时间倒序）
        """
        session_key = self.get_session_key(user_id, group_id)
        memories = self.working_memory_cache.get(session_key, [])
        
        # 按创建时间倒序排序
        sorted_memories = sorted(
            memories,
            key=lambda m: m.created_time,
            reverse=True
        )
        
        if sorted_memories:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
        
        return sorted_memories[:limit]
    
    async def clear_session_async(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> bool:
        """异步清除会话的工作记忆
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            
        Returns:
            bool: 是否成功
        """
        async with self._lock:
            self.clear_working_memory(user_id, group_id)
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_memories = sum(
            len(memories) for memories in self.working_memory_cache.values()
        )
        
        return {
            **self._stats,
            "total_sessions": len(self.session_metadata),
            "total_working_memories": total_memories,
            "max_sessions": self.max_sessions,
            "max_working_memory": self.max_working_memory,
            "ttl": self.ttl
        }
    
    def _update_session_order(self, session_key: str):
        """更新会话访问顺序（LRU）
        
        Args:
            session_key: 会话键
        """
        if session_key in self._session_order:
            self._session_order.remove(session_key)
        self._session_order.append(session_key)
        
        # 如果超过最大会话数，淘汰最旧的
        while len(self._session_order) > self.max_sessions:
            oldest_key = self._session_order.pop(0)
            if oldest_key in self.working_memory_cache:
                del self.working_memory_cache[oldest_key]
            if oldest_key in self.session_metadata:
                del self.session_metadata[oldest_key]
            self._stats["evictions"] += 1
            logger.debug(f"Evicted oldest session: {oldest_key}")

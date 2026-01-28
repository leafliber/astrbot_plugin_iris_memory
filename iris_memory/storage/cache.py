"""
缓存模块
为工作记忆和嵌入向量提供高效缓存
支持LRU/LFU策略，可选Redis持久化
"""

import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import asyncio


class CacheStrategy(str, Enum):
    """缓存策略"""
    LRU = "lru"      # 最近最少使用（Least Recently Used）
    LFU = "lfu"      # 最不经常使用（Least Frequently Used）
    FIFO = "fifo"    # 先进先出（First In First Out）


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # 生存时间（秒），None表示永不过期
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_time).total_seconds() > self.ttl
    
    def touch(self):
        """更新访问时间和访问次数"""
        self.last_access_time = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size': self.size,
            'max_size': self.max_size,
            'hit_rate': self.hit_rate
        }


class BaseCache(ABC):
    """缓存基类"""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 默认生存时间（秒），None表示永不过期
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self.stats = CacheStats(max_size=max_size)
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或过期则返回None
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），None表示使用默认值
            
        Returns:
            是否设置成功
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空缓存
        
        Returns:
            是否清空成功
        """
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """获取缓存大小
        
        Returns:
            当前缓存条目数
        """
        pass
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        self.stats.size = self.get_size()
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = CacheStats(max_size=self.max_size)


class LRUCache(BaseCache):
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 默认生存时间（秒）
        """
        super().__init__(max_size, ttl)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self._cache:
            self.stats.misses += 1
            return None
        
        entry = self._cache[key]
        
        # 检查是否过期
        if entry.is_expired():
            del self._cache[key]
            self.stats.misses += 1
            return None
        
        # 更新访问时间和移到末尾（最近使用）
        entry.touch()
        self._cache.move_to_end(key)
        self.stats.hits += 1
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        # 如果键已存在，更新值并移到末尾
        if key in self._cache:
            self._cache[key].value = value
            if ttl is not None:
                self._cache[key].ttl = ttl
            self._cache.move_to_end(key)
            return True
        
        # 检查是否达到最大容量，执行LRU淘汰
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        # 创建新条目
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl if ttl is not None else self.default_ttl
        )
        self._cache[key] = entry
        
        return True
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> bool:
        """清空缓存"""
        self._cache.clear()
        return True
    
    def get_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    def _evict_lru(self):
        """执行LRU淘汰（移除最久未使用的条目）"""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats.evictions += 1


class LFUCache(BaseCache):
    """LFU缓存实现"""

    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """初始化LFU缓存

        Args:
            max_size: 最大缓存条目数
            ttl: 默认生存时间（秒）
        """
        super().__init__(max_size, ttl)
        self._cache: Dict[str, CacheEntry] = {}
        self._min_freq = 1  # 最小访问频率
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self._cache:
            self.stats.misses += 1
            return None

        entry = self._cache[key]

        # 检查是否过期
        if entry.is_expired():
            del self._cache[key]
            self.stats.misses += 1
            return None

        # 更新访问次数
        old_count = entry.access_count
        entry.touch()
        self.stats.hits += 1
        
        # 如果当前条目是唯一的最小频率条目，更新最小频率
        if old_count == self._min_freq:
            # 检查是否还有其他条目也是这个频率
            remaining = [v for v in self._cache.values() if v.access_count == old_count]
            if not remaining:
                self._min_freq = entry.access_count

        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        # 如果键已存在，更新值
        if key in self._cache:
            entry = self._cache[key]
            old_count = entry.access_count
            entry.value = value
            if ttl is not None:
                entry.ttl = ttl
            entry.touch()
            
            # 更新最小频率
            if old_count == self._min_freq:
                remaining = [v for v in self._cache.values() if v.access_count == old_count]
                if not remaining:
                    self._min_freq = entry.access_count
            
            return True

        # 检查是否达到最大容量，执行LFU淘汰
        if len(self._cache) >= self.max_size:
            self._evict_lfu()

        # 创建新条目
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl if ttl is not None else self.default_ttl
        )
        self._cache[key] = entry

        return True

    def delete(self, key: str) -> bool:
        """删除缓存"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> bool:
        """清空缓存"""
        self._cache.clear()
        self._min_freq = 0
        return True

    def get_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)

    def _evict_lfu(self):
        """执行LFU淘汰（移除访问次数最少的条目）"""
        if not self._cache:
            return

        # 使用记录的最小频率
        candidates = [k for k, v in self._cache.items() if v.access_count == self._min_freq]
        
        # 如果没有找到（可能频率变化了），重新计算最小频率
        if not candidates:
            min_access_count = min(entry.access_count for entry in self._cache.values())
            candidates = [k for k, v in self._cache.items() if v.access_count == min_access_count]
            self._min_freq = min_access_count

        # 如果有多个，选择最老的（按创建时间）
        lfu_key = min(candidates, key=lambda k: self._cache[k].created_time)

        del self._cache[lfu_key]
        self.stats.evictions += 1


class EmbeddingCache:
    """嵌入向量缓存
    
    功能：
    1. 缓存文本的嵌入向量，避免重复计算
    2. 使用MD5哈希作为缓存键
    3. 支持LRU/LFU策略
    4. 提供命中率统计
    """
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        """初始化嵌入缓存
        
        Args:
            max_size: 最大缓存条目数
            strategy: 缓存策略（LRU/LFU）
        """
        self.max_size = max_size
        self.strategy = strategy
        
        # 创建底层缓存
        if strategy == CacheStrategy.LRU:
            self._cache = LRUCache(max_size)
        elif strategy == CacheStrategy.LFU:
            self._cache = LFUCache(max_size)
        else:
            self._cache = LRUCache(max_size)
    
    def _compute_hash(self, text: str) -> str:
        """计算文本的MD5哈希
        
        Args:
            text: 文本内容
            
        Returns:
            MD5哈希值
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """获取文本的嵌入向量
        
        Args:
            text: 文本内容
            
        Returns:
            嵌入向量，如果缓存未命中则返回None
        """
        key = self._compute_hash(text)
        return self._cache.get(key)
    
    def set(self, text: str, embedding: List[float]) -> bool:
        """缓存文本的嵌入向量
        
        Args:
            text: 文本内容
            embedding: 嵌入向量
            
        Returns:
            是否设置成功
        """
        key = self._compute_hash(text)
        return self._cache.set(key, embedding)
    
    def delete(self, text: str) -> bool:
        """删除文本的缓存
        
        Args:
            text: 文本内容
            
        Returns:
            是否删除成功
        """
        key = self._compute_hash(text)
        return self._cache.delete(key)
    
    def clear(self):
        """清空所有缓存"""
        self._cache.clear()
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        return self._cache.get_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self._cache.reset_stats()


class WorkingMemoryCache:
    """工作记忆缓存
    
    功能：
    1. 缓存会话内的最近对话
    2. 按会话隔离存储
    3. 支持快速查询最近N条记忆
    4. 自动清理过期记忆
    """
    
    def __init__(self, max_sessions: int = 100, max_memories_per_session: int = 50, ttl: int = 86400):
        """初始化工作记忆缓存
        
        Args:
            max_sessions: 最大会话数
            max_memories_per_session: 每个会话最多缓存记忆数
            ttl: 记忆生存时间（秒），默认24小时
        """
        self.max_sessions = max_sessions
        self.max_memories_per_session = max_memories_per_session
        self.ttl = ttl
        
        # {session_key: LRUCache}
        self._sessions: Dict[str, LRUCache] = {}
        self._session_order: List[str] = []  # 维护会话访问顺序
        self._lock = asyncio.Lock()
    
    def _generate_session_key(self, user_id: str, group_id: Optional[str] = None) -> str:
        """生成会话键
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            
        Returns:
            会话键
        """
        return f"{user_id}:{group_id if group_id else 'private'}"
    
    async def add_memory(self, user_id: str, group_id: Optional[str], memory_id: str, memory: Any) -> bool:
        """添加记忆到工作记忆缓存
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            memory_id: 记忆ID
            memory: 记忆对象
            
        Returns:
            是否添加成功
        """
        session_key = self._generate_session_key(user_id, group_id)
        
        async with self._lock:
            # 如果会话不存在，创建新会话
            if session_key not in self._sessions:
                # 检查是否超过最大会话数
                if len(self._sessions) >= self.max_sessions:
                    # 删除最旧的会话
                    oldest_key = self._session_order.pop(0)
                    del self._sessions[oldest_key]
                
                self._sessions[session_key] = LRUCache(self.max_memories_per_session, self.ttl)
            
            # 更新会话访问顺序
            if session_key in self._session_order:
                self._session_order.remove(session_key)
            self._session_order.append(session_key)
            
            # 添加记忆
            return self._sessions[session_key].set(memory_id, memory)
    
    async def get_memory(self, user_id: str, group_id: Optional[str], memory_id: str) -> Optional[Any]:
        """获取工作记忆
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            memory_id: 记忆ID
            
        Returns:
            记忆对象，如果不存在则返回None
        """
        session_key = self._generate_session_key(user_id, group_id)
        
        async with self._lock:
            if session_key not in self._sessions:
                return None
            
            memory = self._sessions[session_key].get(memory_id)
            
            # 更新会话访问顺序
            if memory is not None:
                if session_key in self._session_order:
                    self._session_order.remove(session_key)
                self._session_order.append(session_key)
            
            return memory
    
    async def get_recent_memories(self, user_id: str, group_id: Optional[str], limit: int = 10) -> List[Any]:
        """获取最近N条记忆
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            limit: 返回的最大数量
            
        Returns:
            记忆列表（按访问时间倒序）
        """
        session_key = self._generate_session_key(user_id, group_id)
        
        async with self._lock:
            if session_key not in self._sessions:
                return []
            
            # 获取所有缓存条目
            cache = self._sessions[session_key]
            all_memories = []
            
            for key in reversed(cache._cache.keys()):  # 最近使用的在最后
                entry = cache._cache[key]
                if not entry.is_expired():
                    all_memories.append(entry.value)
                    if len(all_memories) >= limit:
                        break
            
            return all_memories
    
    async def clear_session(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """清空会话缓存
        
        Args:
            user_id: 用户ID
            group_id: 群组ID，如果为None则清空该用户的所有会话
            
        Returns:
            是否清空成功
        """
        async with self._lock:
            if group_id is None:
                # 清空用户的所有会话
                keys_to_delete = [k for k in self._sessions.keys() if k.startswith(f"{user_id}:")]
                for key in keys_to_delete:
                    del self._sessions[key]
                    if key in self._session_order:
                        self._session_order.remove(key)
                return len(keys_to_delete) > 0
            else:
                # 清空指定会话
                session_key = self._generate_session_key(user_id, group_id)
                if session_key in self._sessions:
                    del self._sessions[session_key]
                    if session_key in self._session_order:
                        self._session_order.remove(session_key)
                    return True
                return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        async with self._lock:
            total_memories = sum(cache.get_size() for cache in self._sessions.values())
            
            return {
                'total_sessions': len(self._sessions),
                'total_memories': total_memories,
                'max_sessions': self.max_sessions,
                'max_memories_per_session': self.max_memories_per_session,
                'sessions': {
                    key: cache.get_stats().to_dict()
                    for key, cache in self._sessions.items()
                }
            }


class MemoryCompressor:
    """记忆压缩器
    
    功能：
    1. 提取记忆摘要，压缩内容
    2. 保留关键实体和情感信息
    3. 降低存储空间占用
    """
    
    def __init__(self, max_length: int = 200):
        """初始化记忆压缩器
        
        Args:
            max_length: 压缩后的最大长度
        """
        self.max_length = max_length
    
    def compress_memory(self, content: str, entities: Optional[List[Any]] = None) -> str:
        """压缩记忆内容
        
        Args:
            content: 原始内容
            entities: 实体列表（可选）
            
        Returns:
            压缩后的内容
        """
        # 如果内容已经很短，直接返回
        if len(content) <= self.max_length:
            return content
        
        # 简单压缩策略：保留前N个字符
        compressed = content[:self.max_length - 3] + "..."
        
        # TODO: 更智能的压缩策略
        # 1. 保留关键句子
        # 2. 保留实体上下文
        # 3. 使用NLP生成摘要
        
        return compressed
    
    def extract_keywords(self, content: str, top_k: int = 5) -> List[str]:
        """提取关键词
        
        Args:
            content: 内容文本
            top_k: 返回的关键词数量
            
        Returns:
            关键词列表
        """
        # TODO: 实现更智能的关键词提取
        # 当前使用简单的词频统计
        
        import re
        from collections import Counter
        
        # 分词（简单实现）
        words = re.findall(r'\w+', content.lower())
        word_freq = Counter(words)
        
        # 过滤停用词（简化版）
        stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', 'the', 'and', 'is', 'to', 'of', 'a', 'in', 'that', 'it', 'for', 'on', 'with', 'as', 'this', 'was', 'at'}
        
        keywords = [word for word, freq in word_freq.most_common(top_k * 2) if len(word) > 1 and word not in stopwords]
        
        return keywords[:top_k]


class CacheManager:
    """缓存管理器
    
    统一管理所有缓存：
    - 嵌入向量缓存
    - 工作记忆缓存
    - 提供缓存预热功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化缓存管理器
        
        Args:
            config: 配置字典
        """
        # 嵌入缓存配置
        embedding_config = config.get('embedding_cache', {})
        self.embedding_cache = EmbeddingCache(
            max_size=embedding_config.get('max_size', 1000),
            strategy=CacheStrategy(embedding_config.get('strategy', 'lru'))
        )
        
        # 工作记忆缓存配置
        working_config = config.get('working_cache', {})
        self.working_cache = WorkingMemoryCache(
            max_sessions=working_config.get('max_sessions', 100),
            max_memories_per_session=working_config.get('max_memories_per_session', 50),
            ttl=working_config.get('ttl', 86400)
        )
        
        # 记忆压缩器
        compression_config = config.get('compression', {})
        self.compressor = MemoryCompressor(
            max_length=compression_config.get('max_length', 200)
        )
    
    def get_embedding_cache(self) -> EmbeddingCache:
        """获取嵌入缓存"""
        return self.embedding_cache
    
    def get_working_cache(self) -> WorkingMemoryCache:
        """获取工作记忆缓存"""
        return self.working_cache
    
    def get_compressor(self) -> MemoryCompressor:
        """获取记忆压缩器"""
        return self.compressor
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取所有缓存的统计信息"""
        embedding_stats = self.embedding_cache.get_stats().to_dict()
        working_stats = await self.working_cache.get_stats()
        
        return {
            'embedding_cache': embedding_stats,
            'working_cache': working_stats
        }
    
    async def clear_all(self):
        """清空所有缓存"""
        self.embedding_cache.clear()
        # 工作记忆缓存需要逐会话清理
        async with self.working_cache._lock:
            self.working_cache._sessions.clear()
            self.working_cache._session_order.clear()

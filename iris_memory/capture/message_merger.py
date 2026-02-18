"""
消息合并模块

将消息合并和去重逻辑从 BatchProcessor 中拆分出来，提高代码可维护性。
"""

import hashlib
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from iris_memory.utils.logger import get_logger

logger = get_logger("message_merger")


@dataclass
class QueuedMessage:
    """队列中的消息"""
    content: str
    user_id: str
    sender_name: Optional[str]
    group_id: Optional[str]
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    umo: str = ""
    is_merged: bool = False
    original_messages: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if not isinstance(self.context, dict):
            self.context = {}


class MessageMerger:
    """消息合并器
    
    职责：
    1. 短消息合并
    2. 消息去重
    3. 消息指纹生成
    """
    
    DEFAULT_SHORT_MESSAGE_THRESHOLD: int = 15
    DEFAULT_MERGE_TIME_WINDOW: int = 60
    DEFAULT_MAX_MERGE_COUNT: int = 5
    DEFAULT_FINGERPRINT_CACHE_SIZE: int = 1000
    DEFAULT_FINGERPRINT_TRIM_SIZE: int = 500
    
    def __init__(
        self,
        short_message_threshold: int = DEFAULT_SHORT_MESSAGE_THRESHOLD,
        merge_time_window: int = DEFAULT_MERGE_TIME_WINDOW,
        max_merge_count: int = DEFAULT_MAX_MERGE_COUNT,
        fingerprint_cache_size: int = DEFAULT_FINGERPRINT_CACHE_SIZE,
        fingerprint_trim_size: int = DEFAULT_FINGERPRINT_TRIM_SIZE
    ):
        """初始化消息合并器
        
        Args:
            short_message_threshold: 短消息长度阈值
            merge_time_window: 合并时间窗口（秒）
            max_merge_count: 最大合并消息数
            fingerprint_cache_size: 指纹缓存大小
            fingerprint_trim_size: 裁剪后大小
        """
        self.short_message_threshold = short_message_threshold
        self.merge_time_window = merge_time_window
        self.max_merge_count = max_merge_count
        self.fingerprint_cache_size = fingerprint_cache_size
        self.fingerprint_trim_size = fingerprint_trim_size
        
        self._processed_fingerprints: Set[str] = set()
        
        self.stats: Dict[str, int] = {
            "messages_merged": 0,
            "messages_deduped": 0,
        }
    
    def get_message_fingerprint(self, content: str) -> str:
        """生成消息指纹
        
        Args:
            content: 消息内容
            
        Returns:
            str: 指纹字符串
        """
        simplified = ''.join(c.lower() for c in content if c.isalnum())
        simplified = simplified[:80]
        return hashlib.md5(simplified.encode()).hexdigest()[:12]
    
    def is_duplicate_message(self, content: str) -> bool:
        """检查消息是否重复
        
        Args:
            content: 消息内容
            
        Returns:
            bool: 是否重复
        """
        fingerprint = self.get_message_fingerprint(content)
        
        if fingerprint in self._processed_fingerprints:
            return True
        
        self._processed_fingerprints.add(fingerprint)
        
        if len(self._processed_fingerprints) > self.fingerprint_cache_size:
            self._processed_fingerprints = set(
                list(self._processed_fingerprints)[self.fingerprint_trim_size:]
            )
        
        return False
    
    def deduplicate_messages(self, queue: List[QueuedMessage]) -> List[QueuedMessage]:
        """消息去重
        
        Args:
            queue: 消息队列
            
        Returns:
            List[QueuedMessage]: 去重后的消息队列
        """
        unique = []
        for msg in queue:
            if self.is_duplicate_message(msg.content):
                self.stats["messages_deduped"] += 1
                continue
            unique.append(msg)
        return unique
    
    def merge_short_messages(self, queue: List[QueuedMessage]) -> List[QueuedMessage]:
        """合并连续短消息
        
        策略：
        1. 连续短消息（<15字符）合并为一条
        2. 合并时间窗口内（60秒）的消息
        3. 最多合并5条
        4. 只有同一用户的消息才合并
        
        Args:
            queue: 消息队列
            
        Returns:
            List[QueuedMessage]: 合并后的消息队列
        """
        if len(queue) <= 1:
            return queue
        
        merged = []
        current_group: List[QueuedMessage] = []
        
        for msg in queue:
            if not current_group:
                current_group.append(msg)
                continue
            
            last_msg = current_group[-1]
            
            should_merge = (
                len(msg.content) < self.short_message_threshold
                and len(last_msg.content) < self.short_message_threshold * 3
                and msg.timestamp - last_msg.timestamp < self.merge_time_window
                and msg.user_id == last_msg.user_id
                and len(current_group) < self.max_merge_count
            )
            
            if should_merge:
                current_group.append(msg)
            else:
                if len(current_group) == 1:
                    merged.append(current_group[0])
                else:
                    merged.append(self._combine_message_group(current_group))
                current_group = [msg]
        
        if current_group:
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                merged.append(self._combine_message_group(current_group))
        
        self.stats["messages_merged"] += len(queue) - len(merged)
        return merged
    
    def _combine_message_group(self, messages: List[QueuedMessage]) -> QueuedMessage:
        """将一组消息合并为一条
        
        Args:
            messages: 消息列表
            
        Returns:
            QueuedMessage: 合并后的消息
        """
        if len(messages) == 1:
            return messages[0]
        
        first = messages[0]
        combined_content = " ".join([m.content for m in messages])
        
        return QueuedMessage(
            content=combined_content,
            user_id=first.user_id,
            sender_name=first.sender_name,
            group_id=first.group_id,
            timestamp=first.timestamp,
            context={
                **first.context,
                "merged": True,
                "merge_count": len(messages),
                "time_span": messages[-1].timestamp - first.timestamp
            },
            umo=first.umo,
            is_merged=True,
            original_messages=[m.content for m in messages]
        )
    
    def clear_fingerprints(self) -> None:
        """清空指纹缓存"""
        self._processed_fingerprints.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()

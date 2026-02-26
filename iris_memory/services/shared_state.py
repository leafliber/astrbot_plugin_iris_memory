"""
共享状态管理

集中管理跨服务共享的用户状态数据（情感状态、用户画像、注入记录等），
避免分散在多个 Mixin 中导致隐式耦合。

设计动机：
- 原 Mixin 模式下，BusinessOperations 和 PersistenceOperations 通过 self 隐式共享状态
- 提取为独立类后，依赖关系显式化，可独立测试
"""

from typing import List

from iris_memory.models.emotion_state import EmotionalState
from iris_memory.models.user_persona import UserPersona
from iris_memory.utils.bounded_dict import BoundedDict


class SharedState:
    """跨服务共享状态

    持有所有 Service 层共享的用户粒度状态。
    使用 BoundedDict 实现 LRU 淘汰，防止内存无限增长。

    Args:
        max_size: 各 BoundedDict 的最大容量
        max_recent_track: 每个 session 追踪的最近注入记忆数量上限
    """

    def __init__(self, max_size: int = 2000, max_recent_track: int = 20) -> None:
        self._user_emotional_states: BoundedDict[str, EmotionalState] = BoundedDict(max_size=max_size)
        self._user_personas: BoundedDict[str, UserPersona] = BoundedDict(max_size=max_size)
        self._recently_injected: BoundedDict[str, List[str]] = BoundedDict(max_size=max_size)
        self._max_recent_track: int = max_recent_track

    # ── 用户状态访问 ──

    def get_or_create_emotional_state(self, user_id: str) -> EmotionalState:
        """获取或创建用户情感状态"""
        if user_id not in self._user_emotional_states:
            self._user_emotional_states[user_id] = EmotionalState()
        return self._user_emotional_states[user_id]

    def get_or_create_user_persona(self, user_id: str) -> UserPersona:
        """获取或创建用户画像"""
        if user_id not in self._user_personas:
            self._user_personas[user_id] = UserPersona(user_id=user_id)
        return self._user_personas[user_id]

    # ── 记忆注入追踪 ──

    def filter_recently_injected(self, memories: list, session_key: str) -> list:
        """过滤最近已注入过的记忆，避免重复提及同一件事

        Args:
            memories: 待过滤的记忆列表
            session_key: 会话标识

        Returns:
            过滤后的记忆列表
        """
        recent_ids = set(self._recently_injected.get(session_key, []))
        if not recent_ids:
            return memories

        filtered = [m for m in memories if m.id not in recent_ids]

        if not filtered:
            return memories[:max(1, len(memories) // 2)]

        return filtered

    def track_injected_memories(self, session_key: str, memory_ids: List[str]) -> None:
        """记录本次注入的记忆ID

        Args:
            session_key: 会话标识
            memory_ids: 本次注入的记忆ID列表
        """
        if session_key not in self._recently_injected:
            self._recently_injected[session_key] = []

        self._recently_injected[session_key].extend(memory_ids)

        if len(self._recently_injected[session_key]) > self._max_recent_track:
            self._recently_injected[session_key] = (
                self._recently_injected[session_key][-self._max_recent_track:]
            )

    # ── 向后兼容属性 ──

    @property
    def user_personas(self) -> BoundedDict[str, UserPersona]:
        """用户画像字典（向后兼容 main.py 直接访问）"""
        return self._user_personas

    @property
    def user_emotional_states(self) -> BoundedDict[str, EmotionalState]:
        """用户情感状态字典（向后兼容 main.py 直接访问）"""
        return self._user_emotional_states

    @property
    def recently_injected(self) -> BoundedDict[str, List[str]]:
        """最近注入记忆字典"""
        return self._recently_injected

    @property
    def max_recent_track(self) -> int:
        """单 session 追踪上限"""
        return self._max_recent_track

"""
记忆可见性类型 - 定义记忆的作用域和可见性
"""

from enum import Enum


class MemoryScope(str, Enum):
    """记忆可见性范围
    
    定义记忆的可见性和共享级别。
    """
    
    USER_PRIVATE = "user_private"  # 用户私有记忆，仅创建者可见
    GROUP_SHARED = "group_shared"  # 群组共享记忆，群内所有人可见
    GROUP_PRIVATE = "group_private"  # 群组内个人记忆，仅创建者在该群组可见
    GLOBAL = "global"  # 全局记忆，所有用户可见（慎用）

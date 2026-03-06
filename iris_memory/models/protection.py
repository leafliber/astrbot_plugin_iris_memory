"""
记忆保护标记系统

基于位掩码的保护标记，防止重要记忆被误删或降级。
"""

import re
from enum import IntEnum
from typing import ClassVar, List, TYPE_CHECKING

if TYPE_CHECKING:
    from iris_memory.models.memory import Memory


class ProtectionFlag(IntEnum):
    """保护标记位定义"""
    CORE_IDENTITY = 0x01      # 核心身份信息（姓名/生日/年龄）
    USER_PINNED = 0x02        # 用户手动钉选
    HIGH_EMOTION = 0x04       # 高情感价值记忆
    ANNIVERSARY = 0x08        # 纪念日/特殊日期关联
    RELATIONSHIP_KEY = 0x10   # 关系定义性记忆


class ProtectionMixin:
    """Memory 保护操作混入

    为 Memory dataclass 提供保护标记的位操作方法。
    要求宿主类具有 ``protection_flags: int`` 字段。
    """

    def add_protection(self, flag: ProtectionFlag) -> None:
        self.protection_flags |= flag.value

    def remove_protection(self, flag: ProtectionFlag) -> None:
        self.protection_flags &= ~flag.value

    def has_protection(self, flag: ProtectionFlag) -> bool:
        return bool(self.protection_flags & flag.value)

    @property
    def is_protected(self) -> bool:
        """任何保护标记生效时返回 True"""
        return self.protection_flags != 0

    @property
    def is_deletable(self) -> bool:
        """综合判断是否可被删除"""
        if self.is_protected:
            return False
        if self.is_user_requested:
            return False
        from iris_memory.core.types import QualityLevel
        if self.quality_level == QualityLevel.CONFIRMED:
            return False
        return True


class ProtectionRules:
    """保护标记自动判定规则"""

    CORE_IDENTITY_PATTERNS: ClassVar[List[str]] = [
        r"(?:我|俺|本人)(?:叫|是|的?名字(?:是|叫))",
        r"(?:我|俺)(?:今年|的?年龄|岁)",
        r"(?:我|俺)的?(?:生日|出生)",
        r"(?:my\s+name\s+is|i\s+am|i'm)\s+",
    ]

    RELATIONSHIP_KEY_PATTERNS: ClassVar[List[str]] = [
        r"(?:你|我们)是(?:朋友|闺蜜|兄弟|姐妹|情侣|家人)",
        r"(?:我把你当|你对我来说是|我们的关系)",
    ]

    ANNIVERSARY_PATTERNS: ClassVar[List[str]] = [
        r"(?:纪念日|周年|生日|第一次)",
        r"(?:anniversary|birthday|first\s+time)",
    ]

    @classmethod
    def evaluate(cls, memory: "Memory") -> int:
        """评估记忆应获得的保护标记

        Args:
            memory: 待评估的记忆对象

        Returns:
            int: 保护标记位掩码
        """
        from iris_memory.core.types import MemoryType

        flags = 0
        content = memory.content.lower() if memory.content else ""

        # 核心身份
        if memory.type == MemoryType.FACT and memory.confidence >= 0.85:
            for pattern in cls.CORE_IDENTITY_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    flags |= ProtectionFlag.CORE_IDENTITY
                    break

        # 高情感价值
        if memory.emotional_weight >= 0.85 and memory.type == MemoryType.EMOTION:
            flags |= ProtectionFlag.HIGH_EMOTION

        # 关系定义
        if memory.type == MemoryType.RELATIONSHIP and memory.confidence >= 0.8:
            for pattern in cls.RELATIONSHIP_KEY_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    flags |= ProtectionFlag.RELATIONSHIP_KEY
                    break

        # 纪念日
        for pattern in cls.ANNIVERSARY_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                flags |= ProtectionFlag.ANNIVERSARY
                break

        # 用户明确请求
        if memory.is_user_requested:
            flags |= ProtectionFlag.USER_PINNED

        return flags

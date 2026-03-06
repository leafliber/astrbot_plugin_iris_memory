"""
快速通道评估器

高置信核心信息绕过逐级升级，直达 SEMANTIC 层。
"""

import re
from typing import ClassVar, List, Optional

from iris_memory.core.types import MemoryType, QualityLevel, StorageLayer
from iris_memory.models.memory import Memory
from iris_memory.models.protection import ProtectionFlag, ProtectionRules
from iris_memory.utils.logger import get_logger

logger = get_logger("fast_track")


class FastTrackEvaluator:
    """快速通道评估器

    准入条件（满足任一）:
    1. FACT 类型 + confidence >= 0.9 + 命中核心身份关键词
    2. 用户明确请求保存 + confidence >= 0.85
    3. quality_level == CONFIRMED
    4. 有 CORE_IDENTITY 保护标记 + confidence >= 0.85
    """

    IDENTITY_KEYWORDS: ClassVar[List[str]] = [
        "名字", "姓名", "叫", "姓", "生日", "出生", "年龄", "岁",
        "性别", "身份证", "电话", "手机号",
    ]

    def __init__(self, confidence_threshold: float = 0.9):
        self._confidence_threshold = confidence_threshold

    def evaluate(self, memory: Memory, is_user_requested: bool) -> Optional[StorageLayer]:
        """快速通道评估

        Returns:
            StorageLayer.SEMANTIC 如果满足快速通道条件，否则 None（走常规通道）
        """
        # 条件 1: 核心身份信息
        if (
            memory.type == MemoryType.FACT
            and memory.confidence >= self._confidence_threshold
            and self._contains_identity_keyword(memory.content)
        ):
            memory.add_protection(ProtectionFlag.CORE_IDENTITY)
            logger.debug(f"Fast-track: core identity → SEMANTIC (memory {memory.id[:8]})")
            return StorageLayer.SEMANTIC

        # 条件 2: 用户请求 + 高置信
        if is_user_requested and memory.confidence >= 0.85:
            memory.add_protection(ProtectionFlag.USER_PINNED)
            logger.debug(f"Fast-track: user requested → SEMANTIC (memory {memory.id[:8]})")
            return StorageLayer.SEMANTIC

        # 条件 3: 已确认信息
        if memory.quality_level == QualityLevel.CONFIRMED:
            logger.debug(f"Fast-track: CONFIRMED quality → SEMANTIC (memory {memory.id[:8]})")
            return StorageLayer.SEMANTIC

        # 条件 4: 已有核心身份标记
        if (
            memory.has_protection(ProtectionFlag.CORE_IDENTITY)
            and memory.confidence >= 0.85
        ):
            logger.debug(f"Fast-track: existing CORE_IDENTITY flag → SEMANTIC (memory {memory.id[:8]})")
            return StorageLayer.SEMANTIC

        return None

    def _contains_identity_keyword(self, content: str) -> bool:
        """检查内容是否包含身份关键词"""
        if not content:
            return False
        content_lower = content.lower()
        return any(kw in content_lower for kw in self.IDENTITY_KEYWORDS)

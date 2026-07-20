"""
Iris Chat Memory - L2 记忆数据模型

定义 L2 记忆库的数据结构，包括记忆条目和检索结果。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ============================================================================
# 数据类定义
# ============================================================================


@dataclass
class MemoryEntry:
    """记忆条目数据类

    存储单条记忆的完整信息，包括内容、向量、元数据等。

    Attributes:
        id: 记忆唯一标识符
        content: 记忆内容文本
        embedding: 向量表示（可选，ChromaDB 内部管理）
        metadata: 元数据字典，包含：
            - group_id: 群聊ID
            - user_id: 用户ID（可选）
            - timestamp: 创建时间戳
            - access_count: 访问次数
            - last_access_time: 最近访问时间
            - confidence: 置信度
            - source: 来源（summary/tool）
        persona_id: 人格ID，用于人格命名空间隔离（SQLite 独立列，
            不属于 metadata）。导出/导入必须透传此字段，否则模型迁移
            的「导出→删库→重导入」回合会将所有记忆塌缩为 "default"，
            永久破坏人格隔离。

    Examples:
        >>> entry = MemoryEntry(
        ...     id="mem_001",
        ...     content="用户喜欢吃苹果",
        ...     metadata={
        ...         "group_id": "group_123",
        ...         "timestamp": datetime.now().isoformat(),
        ...         "access_count": 1,
        ...         "confidence": 0.85
        ...     },
        ...     persona_id="yuki",
        ... )
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    persona_id: str = "default"
    embedding: Optional[list[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含所有字段的字典（不含 embedding），persona_id 单独列出
            以便导出/导入回合透传人格归属
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "persona_id": self.persona_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从字典创建实例

        Args:
            data: 包含记忆数据的字典

        Returns:
            MemoryEntry 实例

        Note:
            旧版导出文件无 persona_id 字段时回退为 "default"，
            与历史行为兼容。
        """
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            persona_id=data.get("persona_id", "default"),
            embedding=data.get("embedding"),
        )

    @property
    def group_id(self) -> Optional[str]:
        """获取群聊ID

        Returns:
            群聊ID，不存在时返回 None
        """
        return self.metadata.get("group_id")

    @property
    def timestamp(self) -> Optional[str]:
        """获取创建时间戳

        Returns:
            ISO 格式时间戳字符串，不存在时返回 None
        """
        return self.metadata.get("timestamp")

    @property
    def access_count(self) -> int:
        """获取访问次数

        Returns:
            访问次数，默认为 0
        """
        return self.metadata.get("access_count", 0)

    @property
    def last_access_time(self) -> Optional[str]:
        """获取最近访问时间

        Returns:
            ISO 格式时间戳字符串，不存在时返回 None
        """
        return self.metadata.get("last_access_time")

    @property
    def confidence(self) -> float:
        """获取置信度

        Returns:
            置信度分数，默认为 0.5
        """
        return self.metadata.get("confidence", 0.5)

    @property
    def kg_processed(self) -> bool:
        """获取知识图谱处理状态

        Returns:
            是否已处理，默认为 False
        """
        return self.metadata.get("kg_processed", False)


@dataclass
class MemorySearchResult:
    """记忆检索结果数据类

    存储单条检索结果，包括记忆条目、相似度分数和距离。

    Attributes:
        entry: 记忆条目
        score: 相似度分数（越高越相似，范围 [0, 1]）
        distance: 向量距离（越低越相似）

    Examples:
        >>> entry = MemoryEntry(id="mem_001", content="测试")
        >>> result = MemorySearchResult(entry=entry, score=0.85, distance=0.15)
        >>> result.score
        0.85
    """

    entry: MemoryEntry
    score: float
    distance: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含条目信息和分数的字典
        """
        return {
            "id": self.entry.id,
            "content": self.entry.content,
            "score": self.score,
            "distance": self.distance,
            "metadata": self.entry.metadata,
        }

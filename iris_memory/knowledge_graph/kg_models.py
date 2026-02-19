"""
知识图谱数据模型

定义知识图谱中的节点、边和三元组数据结构。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class KGNodeType(str, Enum):
    """知识图谱节点类型"""
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    OBJECT = "object"            # 物品/事物
    EVENT = "event"              # 事件
    CONCEPT = "concept"          # 抽象概念（兴趣/技能/习惯等）
    TIME = "time"                # 时间节点
    UNKNOWN = "unknown"


class KGRelationType(str, Enum):
    """知识图谱关系类型"""
    # 人际关系
    FRIEND_OF = "friend_of"
    COLLEAGUE_OF = "colleague_of"
    FAMILY_OF = "family_of"
    BOSS_OF = "boss_of"
    SUBORDINATE_OF = "subordinate_of"
    KNOWS = "knows"

    # 属性关系
    LIVES_IN = "lives_in"
    WORKS_AT = "works_at"
    STUDIES_AT = "studies_at"
    BELONGS_TO = "belongs_to"
    OWNS = "owns"

    # 行为/状态关系
    LIKES = "likes"
    DISLIKES = "dislikes"
    DOES = "does"                # 做某事（习惯/行为）
    IS = "is"                    # 是（属性描述）
    HAS = "has"                  # 拥有（属性/物品）
    WANTS = "wants"              # 想要

    # 事件关系
    PARTICIPATED_IN = "participated_in"
    HAPPENED_AT = "happened_at"
    CAUSED_BY = "caused_by"

    # 通用
    RELATED_TO = "related_to"


@dataclass
class KGNode:
    """知识图谱节点"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""                           # 规范化名称
    display_name: str = ""                   # 显示名称（原始文本）
    node_type: KGNodeType = KGNodeType.UNKNOWN
    user_id: str = ""                        # 来源用户
    group_id: Optional[str] = None           # 来源群组
    aliases: List[str] = field(default_factory=list)  # 别名列表
    properties: Dict[str, Any] = field(default_factory=dict)
    mention_count: int = 1                   # 提及次数
    confidence: float = 0.5
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        import json
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "node_type": self.node_type.value,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "aliases": json.dumps(self.aliases, ensure_ascii=False),
            "properties": json.dumps(self.properties, ensure_ascii=False),
            "mention_count": self.mention_count,
            "confidence": self.confidence,
            "created_time": self.created_time.isoformat(),
            "updated_time": self.updated_time.isoformat(),
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "KGNode":
        """从数据库行恢复"""
        import json
        return cls(
            id=row["id"],
            name=row["name"],
            display_name=row.get("display_name", row["name"]),
            node_type=KGNodeType(row.get("node_type", "unknown")),
            user_id=row.get("user_id", ""),
            group_id=row.get("group_id"),
            aliases=json.loads(row.get("aliases", "[]")),
            properties=json.loads(row.get("properties", "{}")),
            mention_count=row.get("mention_count", 1),
            confidence=row.get("confidence", 0.5),
            created_time=datetime.fromisoformat(row["created_time"]) if isinstance(row.get("created_time"), str) else row.get("created_time", datetime.now()),
            updated_time=datetime.fromisoformat(row["updated_time"]) if isinstance(row.get("updated_time"), str) else row.get("updated_time", datetime.now()),
        )


@dataclass
class KGEdge:
    """知识图谱边（关系）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""                      # 源节点 ID
    target_id: str = ""                      # 目标节点 ID
    relation_type: KGRelationType = KGRelationType.RELATED_TO
    relation_label: str = ""                 # 自由文本关系标签（补充 relation_type）
    memory_id: Optional[str] = None          # 来源记忆 ID
    user_id: str = ""
    group_id: Optional[str] = None
    confidence: float = 0.5
    weight: float = 1.0                      # 边权重（频率/重要性）
    properties: Dict[str, Any] = field(default_factory=dict)
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        import json
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "relation_label": self.relation_label,
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "confidence": self.confidence,
            "weight": self.weight,
            "properties": json.dumps(self.properties, ensure_ascii=False),
            "created_time": self.created_time.isoformat(),
            "updated_time": self.updated_time.isoformat(),
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "KGEdge":
        """从数据库行恢复"""
        import json
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=KGRelationType(row.get("relation_type", "related_to")),
            relation_label=row.get("relation_label", ""),
            memory_id=row.get("memory_id"),
            user_id=row.get("user_id", ""),
            group_id=row.get("group_id"),
            confidence=row.get("confidence", 0.5),
            weight=row.get("weight", 1.0),
            properties=json.loads(row.get("properties", "{}")),
            created_time=datetime.fromisoformat(row["created_time"]) if isinstance(row.get("created_time"), str) else row.get("created_time", datetime.now()),
            updated_time=datetime.fromisoformat(row["updated_time"]) if isinstance(row.get("updated_time"), str) else row.get("updated_time", datetime.now()),
        )


@dataclass
class KGTriple:
    """三元组：(主语, 谓语, 宾语)

    用于从文本中提取的原始关系（尚未映射到图节点/边）
    """
    subject: str = ""                        # 主语文本
    predicate: str = ""                      # 谓语/关系文本
    object: str = ""                         # 宾语文本
    subject_type: KGNodeType = KGNodeType.UNKNOWN
    object_type: KGNodeType = KGNodeType.UNKNOWN
    relation_type: KGRelationType = KGRelationType.RELATED_TO
    confidence: float = 0.5
    source_text: str = ""                    # 原始文本

    def __repr__(self) -> str:
        return f"({self.subject} --[{self.predicate}]--> {self.object})"


@dataclass
class KGPath:
    """多跳推理路径"""
    nodes: List[KGNode] = field(default_factory=list)
    edges: List[KGEdge] = field(default_factory=list)
    total_confidence: float = 0.0
    hop_count: int = 0

    def to_text(self) -> str:
        """将路径转为可读文本"""
        if not self.nodes:
            return ""
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(node.display_name or node.name)
            if i < len(self.edges):
                edge = self.edges[i]
                label = edge.relation_label or edge.relation_type.value
                parts.append(f" --[{label}]--> ")
        return "".join(parts)

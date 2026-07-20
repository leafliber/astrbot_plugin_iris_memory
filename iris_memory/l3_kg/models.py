"""L3 知识图谱数据模型"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib

NODE_TYPE_WHITELIST = {
    "Person",
    "Preference",
    "Skill",
    "Trait",
    "Goal",
    "Belief",
    "Event",
    "Concept",
    "Location",
    "Item",
    "Topic",
    "Group",
}

NODE_TYPE_DESCRIPTIONS = {
    "Person": "人物——具有稳定身份的个体",
    "Preference": "偏好——某人持续倾向的选择或喜好",
    "Skill": "技能——某人掌握或正在学习的能力",
    "Trait": "性格特征——某人稳定的性格、行为模式",
    "Goal": "目标——某人正在追求的计划或意图",
    "Belief": "信念——某人持有的观点、价值观或立场",
    "Event": "事件——具有时间跨度的重大事件（非日常对话）",
    "Concept": "概念——抽象知识或理论",
    "Location": "地点——地理位置或场所",
    "Item": "物品——具体物件或工具",
    "Topic": "话题——反复出现的讨论主题",
    "Group": "群体——具有共同特征的群体或组织",
}

RELATION_TYPE_WHITELIST = {
    "KNOWS",
    "HAS_PREFERENCE",
    "HAS_SKILL",
    "HAS_TRAIT",
    "HAS_GOAL",
    "HAS_BELIEF",
    "PARTICIPATED_IN",
    "LOCATED_AT",
    "HAPPENED_AT",
    "PART_OF",
    "LEADS_TO",
    "CONTRADICTS",
    "SUPPORTS",
    "RELATED_TO",
}

RELATION_TYPE_DESCRIPTIONS = {
    "KNOWS": "认识——人与人之间的相识关系",
    "HAS_PREFERENCE": "偏好——某人对某事物有持续倾向",
    "HAS_SKILL": "掌握——某人拥有某项技能",
    "HAS_TRAIT": "具有——某人具有某种性格特征",
    "HAS_GOAL": "追求——某人正在追求某个目标",
    "HAS_BELIEF": "相信——某人持有某种信念或观点",
    "PARTICIPATED_IN": "参与——某人参与了某事件或活动",
    "LOCATED_AT": "位于——某事物位于某地点",
    "HAPPENED_AT": "发生在——某事件发生在某地点或时间",
    "PART_OF": "属于——某事物是更大整体的一部分",
    "LEADS_TO": "导致——某事物导致另一事物",
    "CONTRADICTS": "矛盾——某事物与另一事物相矛盾",
    "SUPPORTS": "支持——某事物支持或印证另一事物",
    "RELATED_TO": "相关——无法归类的弱关联（最后手段）",
}


@dataclass
class GraphNode:
    """图谱节点

    Attributes:
        id: 节点唯一ID（基于内容hash生成）
        label: 节点类型标签（动态，优先使用白名单类型）
        name: 实体名称
        content: 完整描述内容
        confidence: 置信度 [0.3, 1.0]
        access_count: 访问次数
        last_access_time: 最后访问时间
        created_time: 创建时间
        source_memory_id: 来源记忆ID
        group_id: 群聊ID（用于隔离）
        properties: 扩展属性（存储为 MAP<STRING, STRING>）
    """

    id: str
    label: str
    name: str
    content: str
    confidence: float = 1.0
    access_count: int = 0
    last_access_time: Optional[datetime] = None
    created_time: datetime = field(default_factory=datetime.now)
    source_memory_id: Optional[str] = None
    group_id: Optional[str] = None
    properties: dict[str, str] = field(default_factory=dict)

    def generate_id(self) -> str:
        """基于实体名称生成唯一ID

        使用 label、name 的组合进行 MD5 hash，
        同一 label+name 的实体始终生成相同 ID，确保去重合并。
        生成格式：{label_lower}_{hash_prefix}

        Returns:
            节点唯一ID
        """
        content_hash = hashlib.md5(f"{self.label}:{self.name}".encode()).hexdigest()
        return f"{self.label.lower()}_{content_hash[:12]}"

    def to_dict(self) -> dict:
        """转换为字典格式（用于 KuzuDB 存储）

        Returns:
            包含所有字段的字典
        """
        return {
            "id": self.id,
            "label": self.label,
            "name": self.name,
            "content": self.content,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "created_time": self.created_time,
            "source_memory_id": self.source_memory_id,
            "group_id": self.group_id,
            "properties": self.properties,
        }


@dataclass
class GraphEdge:
    """图谱边

    Attributes:
        source_id: 源节点ID
        target_id: 目标节点ID
        relation_type: 关系类型（动态，优先使用白名单类型）
        weight: 边权重 [0.0, 1.0]
        confidence: 置信度
        access_count: 访问次数
        last_access_time: 最后访问时间
        created_time: 创建时间
        source_memory_id: 来源记忆ID
        properties: 扩展属性（存储为 MAP<STRING, STRING>）
    """

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    confidence: float = 1.0
    access_count: int = 0
    last_access_time: Optional[datetime] = None
    created_time: datetime = field(default_factory=datetime.now)
    source_memory_id: Optional[str] = None
    properties: dict[str, str] = field(default_factory=dict)

    def generate_id(self) -> str:
        """生成边唯一标识

        格式：{source_id}_{relation_type}_{target_id}

        Returns:
            边唯一标识
        """
        return f"{self.source_id}_{self.relation_type}_{self.target_id}"

    def to_dict(self) -> dict:
        """转换为字典格式（用于 KuzuDB 存储）

        Returns:
            包含所有字段的字典
        """
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "created_time": self.created_time,
            "source_memory_id": self.source_memory_id,
            "properties": self.properties,
        }


@dataclass
class ExtractionResult:
    """实体提取结果

    Attributes:
        nodes: 提取的节点列表
        edges: 提取的边列表
        extraction_confidence: 提取置信度
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    extraction_confidence: float = 1.0

    def is_empty(self) -> bool:
        """检查结果是否为空

        Returns:
            如果没有节点和边则返回 True
        """
        return len(self.nodes) == 0 and len(self.edges) == 0

    def to_dict(self) -> dict:
        """转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "extraction_confidence": self.extraction_confidence,
        }

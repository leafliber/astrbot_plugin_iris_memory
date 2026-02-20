"""
Memory数据模型
根据companion-memory框架文档定义的完整Memory数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import numpy as np

from iris_memory.core.types import (
    MemoryType,
    ModalityType,
    QualityLevel,
    SensitivityLevel,
    StorageLayer,
    VerificationMethod,
    DecayRate
)
from iris_memory.core.memory_scope import MemoryScope


@dataclass
class Memory:
    """记忆数据模型
    
    完整实现companion-memory框架中定义的Memory数据结构
    """
    
    # ========== 标识 ==========
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    sender_name: Optional[str] = None  # 发送者显示名称（群聊中用于区分成员）
    group_id: Optional[str] = None  # 群聊ID，私聊为None
    scope: MemoryScope = MemoryScope.GROUP_PRIVATE  # 记忆可见性范围
    type: MemoryType = MemoryType.FACT
    subtype: Optional[str] = None  # 子类型（如emotion中的具体情绪）
    modality: ModalityType = ModalityType.TEXT
    
    # ========== 内容（多模态）==========
    content: str = ""
    text: Optional[str] = None
    audio: Optional[Dict[str, Any]] = None  # 存储音频相关信息
    image: Optional[Dict[str, Any]] = None  # 存储图像相关信息: {url, description, analysis_level}
    video: Optional[Dict[str, Any]] = None  # 存储视频相关信息
    summary: Optional[str] = None
    embedding: Optional[np.ndarray] = None  # 向量嵌入
    keywords: List[str] = field(default_factory=list)
    
    # ========== 图片分析扩展 ==========
    image_description: Optional[str] = None  # 图片智能分析描述
    has_image: bool = False  # 是否包含图片
    
    # ========== 质量 ==========
    quality_level: QualityLevel = QualityLevel.MODERATE
    confidence: float = 0.5  # 0-1
    verification_method: VerificationMethod = VerificationMethod.UNVERIFIED
    consistency_score: float = 0.5  # 0-1, 与已有记忆的一致性
    rif_score: float = 0.5  # 0-1, RIF评分
    
    # ========== 敏感度 ==========
    sensitivity_level: SensitivityLevel = SensitivityLevel.PUBLIC
    detected_entities: List[str] = field(default_factory=list)  # 检测到的敏感实体
    encrypted: bool = False
    encryption_method: Optional[str] = None
    
    # ========== 重要性 ==========
    importance_score: float = 0.5  # 0-1
    is_user_requested: bool = False  # 用户是否显式请求保存
    emotional_weight: float = 0.5  # 0-1, 情感权重
    access_frequency: float = 0.0  # 访问频率
    time_weight: float = 0.0  # 时间权重
    
    # ========== 访问统计 ==========
    access_count: int = 0
    last_access_time: datetime = field(default_factory=datetime.now)
    created_time: datetime = field(default_factory=datetime.now)
    personalized_decay: float = 0.0  # 个性化衰减常数
    base_decay: float = DecayRate.HABIT  # 基础衰减常数
    
    # ========== 存储 ==========
    storage_layer: StorageLayer = StorageLayer.EPISODIC
    expires_at: Optional[datetime] = None  # 过期时间
    version: int = 1  # 版本号，用于追踪修改
    
    # ========== 关联 ==========
    related_memories: List[str] = field(default_factory=list)  # 相关记忆ID
    conflicting_memories: List[str] = field(default_factory=list)  # 冲突记忆ID
    supporting_memories: List[str] = field(default_factory=list)  # 支持记忆ID
    graph_nodes: List[str] = field(default_factory=list)  # 图节点ID
    graph_edges: List[str] = field(default_factory=list)  # 图边ID
    
    # ========== 元数据 ==========
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        data = {}
        
        # 处理所有基本字段
        for key, value in self.__dict__.items():
            if key == 'embedding':
                # 特殊处理numpy数组
                data[key] = value.tolist() if value is not None else None
            elif isinstance(value, datetime):
                # 转换datetime为ISO格式字符串
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                # 转换Enum为字符串值
                data[key] = value.value
            else:
                data[key] = value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """从字典创建Memory对象"""
        # 防御性拷贝，避免修改调用方原始字典
        data = data.copy()
        
        # 处理特殊字段
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        
        if 'created_time' in data and isinstance(data['created_time'], str):
            data['created_time'] = datetime.fromisoformat(data['created_time'])
        
        if 'last_access_time' in data and isinstance(data['last_access_time'], str):
            data['last_access_time'] = datetime.fromisoformat(data['last_access_time'])
        
        if 'expires_at' in data and data['expires_at'] is not None:
            if isinstance(data['expires_at'], str):
                data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        
        # 转换字符串为Enum
        enum_mappings = {
            'scope': MemoryScope,
            'type': MemoryType,
            'modality': ModalityType,
            'quality_level': QualityLevel,
            'sensitivity_level': SensitivityLevel,
            'storage_layer': StorageLayer,
            'verification_method': VerificationMethod,
        }
        
        for field_name, enum_class in enum_mappings.items():
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = enum_class(data[field_name])
        
        return cls(**data)
    
    def update_access(self):
        """更新访问统计"""
        self.access_count += 1
        self.last_access_time = datetime.now()
        self.access_frequency = self.access_count / max(1, (datetime.now() - self.created_time).days)
    
    def should_upgrade_to_episodic(self) -> bool:
        """判断是否应该从工作记忆升级到情景记忆

        触发条件（放宽）：
        - 访问>=1次 且 重要性>0.5
        - 或 情感强度>0.6
        - 或 置信度>=0.7
        - 或 用户主动请求的记忆
        """
        if self.storage_layer != StorageLayer.WORKING:
            return False

        condition1 = self.access_count >= 1 and self.importance_score > 0.5
        condition2 = self.emotional_weight > 0.6
        condition3 = self.confidence >= 0.7
        condition4 = self.is_user_requested

        return condition1 or condition2 or condition3 or condition4
    
    def should_upgrade_to_semantic(self) -> bool:
        """判断是否应该从情景记忆升级到语义记忆
        
        触发条件：
        - 访问>=10次 且 置信度>0.8
        - 或 质量=CONFIRMED
        """
        if self.storage_layer != StorageLayer.EPISODIC:
            return False
        
        condition1 = self.access_count >= 10 and self.confidence > 0.8
        condition2 = self.quality_level == QualityLevel.CONFIRMED
        
        return condition1 or condition2
    
    def should_archive(self, rif_threshold: float = 0.4) -> bool:
        """判断是否应该归档
        
        触发条件：
        - RIF评分<阈值 且 30天未访问
        """
        if self.storage_layer != StorageLayer.EPISODIC:
            return False
        
        days_since_access = (datetime.now() - self.last_access_time).days
        return self.rif_score < rif_threshold and days_since_access > 30
    
    def should_delete_working(self) -> bool:
        """判断是否应该清除工作记忆
        
        触发条件：
        - 会话结束24小时后
        """
        if self.storage_layer != StorageLayer.WORKING:
            return False
        
        hours_since_creation = (datetime.now() - self.created_time).total_seconds() / 3600
        return hours_since_creation > 24
    
    def calculate_time_weight(self) -> float:
        """计算时间权重
        
        根据时近性计算权重：
        - 7天内：1.2
        - 7-30天：1.0
        - 30-90天：0.8
        - >90天：0.6
        """
        days = (datetime.now() - self.last_access_time).days
        
        if days < 7:
            return 1.2
        elif days < 30:
            return 1.0
        elif days < 90:
            return 0.8
        else:
            return 0.6

    def calculate_time_score(self, use_created_time: bool = False) -> float:
        """计算归一化的时间得分（0~1）

        统一的时间评分算法，同时服务于检索引擎和重排序器。
        基于指数衰减：越新的记忆得分越高。

        Args:
            use_created_time: True 使用创建时间（适用于按新旧排序），
                              False 使用最后访问时间（适用于重排序）

        Returns:
            float: 时间得分 (0-1)
        """
        ref_time = self.created_time if use_created_time else self.last_access_time
        days_ago = (datetime.now() - ref_time).total_seconds() / 86400

        if days_ago < 7:
            return 1.0 - (days_ago / 7) * 0.05       # 7天内: 0.95-1.0
        elif days_ago < 30:
            return 0.95 - ((days_ago - 7) / 23) * 0.1  # 7-30天: 0.85-0.95
        elif days_ago < 90:
            return 0.85 - ((days_ago - 30) / 60) * 0.2  # 30-90天: 0.65-0.85
        elif days_ago < 365:
            return 0.65 - ((days_ago - 90) / 275) * 0.3  # 90-365天: 0.35-0.65
        else:
            return max(0.0, 0.35 - ((days_ago - 365) / 365) * 0.35)  # >365天: 0-0.35
    
    def add_conflict(self, other_memory_id: str):
        """添加冲突记忆"""
        if other_memory_id not in self.conflicting_memories:
            self.conflicting_memories.append(other_memory_id)
    
    def add_relation(self, other_memory_id: str):
        """添加相关记忆"""
        if other_memory_id not in self.related_memories:
            self.related_memories.append(other_memory_id)

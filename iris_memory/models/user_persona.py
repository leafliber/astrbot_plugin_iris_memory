"""
UserPersona数据模型
根据companion-memory框架文档定义的完整用户画像数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from iris_memory.core.types import DecayRate


@dataclass
class UserPersona:
    """用户画像数据模型
    
    多维度画像，记录用户的特征、偏好、情感状态等
    """
    
    # ========== 基础信息 ==========
    user_id: str = ""
    version: int = 1
    last_updated: datetime = field(default_factory=datetime.now)
    
    # ========== 工作维度 ==========
    work_style: Optional[str] = None  # 工作风格：如严谨、创新、高效
    work_goals: List[str] = field(default_factory=list)  # 工作目标
    work_challenges: List[str] = field(default_factory=list)  # 工作挑战
    work_preferences: Dict[str, Any] = field(default_factory=dict)  # 工作偏好
    
    # ========== 生活维度 ==========
    lifestyle: Optional[str] = None  # 生活方式：如忙碌、悠闲、规律
    interests: Dict[str, float] = field(default_factory=dict)  # 兴趣领域及权重
    habits: List[str] = field(default_factory=list)  # 习惯
    life_preferences: Dict[str, Any] = field(default_factory=dict)  # 生活偏好
    
    # ========== 情感维度 ==========
    emotional_baseline: str = "neutral"  # 情感基线：joy, sadness, anger, neutral等
    emotional_volatility: float = 0.5  # 情感波动性：0-1
    emotional_triggers: List[str] = field(default_factory=list)  # 情感触发器
    emotional_soothers: Dict[str, Any] = field(default_factory=dict)  # 情感缓解因素
    emotional_patterns: Dict[str, int] = field(default_factory=dict)  # 情感模式统计
    emotional_trajectory: Optional[str] = None  # 情感趋势：improving, deteriorating, stable, volatile
    negative_ratio: float = 0.3  # 负面情感占比
    
    # ========== 关系维度 ==========
    social_style: Optional[str] = None  # 社交风格：如外向、内向、温和
    social_boundaries: Dict[str, Any] = field(default_factory=dict)  # 社交边界
    trust_level: float = 0.5  # 信任等级：0-1
    intimacy_level: float = 0.5  # 亲密程度：0-1
    
    # ========== 人格维度（Big Five）==========
    personality_openness: float = 0.5  # 开放性：0-1
    personality_conscientiousness: float = 0.5  # 尽责性：0-1
    personality_extraversion: float = 0.5  # 外向性：0-1
    personality_agreeableness: float = 0.5  # 宜人性：0-1
    personality_neuroticism: float = 0.5  # 神经质：0-1
    confidence_decay: float = DecayRate.PERSONALITY  # 人格衰减常数
    
    # ========== 沟通维度 ==========
    communication_formality: float = 0.5  # 正式程度：0-1
    communication_directness: float = 0.5  # 直接程度：0-1
    communication_humor: float = 0.5  # 幽默感：0-1
    communication_empathy: float = 0.5  # 共情能力：0-1
    
    # ========== 行为模式 ==========
    hourly_distribution: List[float] = field(default_factory=lambda: [0.0]*24)  # 24小时活跃度分布
    topic_sequences: List[str] = field(default_factory=list)  # 话题转换序列
    memory_cooccurrence: Dict[str, List[str]] = field(default_factory=dict)  # 记忆共现关系
    
    # ========== 证据追踪 ==========
    evidence_confirmed: List[str] = field(default_factory=list)  # 已确认的证据记忆ID
    evidence_inferred: List[str] = field(default_factory=list)  # 推断的证据记忆ID
    evidence_contested: List[str] = field(default_factory=list)  # 有争议的证据记忆ID
    
    # ========== 元数据 ==========
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        data = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPersona':
        """从字典创建UserPersona对象"""
        # 处理datetime字段
        if 'last_updated' in data and isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        
        return cls(**data)
    
    def add_memory_evidence(self, memory_id: str, evidence_type: str = "confirmed"):
        """添加记忆证据
        
        Args:
            memory_id: 记忆ID
            evidence_type: 证据类型：confirmed, inferred, contested
        """
        if evidence_type == "confirmed" and memory_id not in self.evidence_confirmed:
            self.evidence_confirmed.append(memory_id)
        elif evidence_type == "inferred" and memory_id not in self.evidence_inferred:
            self.evidence_inferred.append(memory_id)
        elif evidence_type == "contested" and memory_id not in self.evidence_contested:
            self.evidence_contested.append(memory_id)
    
    def update_from_memory(self, memory):
        """从记忆更新画像"""
        self.last_updated = datetime.now()
        
        # 根据记忆类型更新不同维度
        if memory.type == "emotion":
            self._update_emotional_from_memory(memory)
        elif memory.type == "fact":
            self._update_facts_from_memory(memory)
        elif memory.type == "relationship":
            self._update_social_from_memory(memory)
    
    def _update_emotional_from_memory(self, memory):
        """从情感记忆更新情感维度"""
        # 更新情感基线（如果强度足够）
        if memory.emotional_weight > 0.7:
            self.emotional_baseline = memory.subtype
        
        # 更新情感模式统计
        if memory.subtype:
            self.emotional_patterns[memory.subtype] = \
                self.emotional_patterns.get(memory.subtype, 0) + 1
    
    def _update_facts_from_memory(self, memory):
        """从事实记忆更新事实维度"""
        # 根据内容识别并更新工作或生活维度
        content_lower = memory.content.lower()
        
        # 工作相关
        work_keywords = ['工作', '公司', '项目', '同事', '老板', '职业', '事业']
        if any(kw in content_lower for kw in work_keywords):
            if memory.summary and memory.summary not in self.work_goals:
                self.work_goals.append(memory.summary)
        
        # 生活相关
        life_keywords = ['喜欢', '爱好', '兴趣', '习惯', '运动', '娱乐']
        if any(kw in content_lower for kw in life_keywords):
            if memory.summary and memory.summary not in self.habits:
                self.habits.append(memory.summary)
    
    def _update_social_from_memory(self, memory):
        """从关系记忆更新社交维度"""
        # 更新关系相关的信息
        if memory.summary:
            if "信任" in memory.summary:
                self.trust_level = min(1.0, self.trust_level + 0.1)
            elif "亲密" in memory.summary:
                self.intimacy_level = min(1.0, self.intimacy_level + 0.1)

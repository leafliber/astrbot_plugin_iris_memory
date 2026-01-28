"""
EmotionalState数据模型
根据companion-memory框架文档定义的完整情感状态数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from iris_memory.core.types import EmotionType


class TrendType(str, Enum):
    """情感趋势"""
    IMPROVING = "improving"        # 改善
    DETERIORATING = "deteriorating"  # 恶化
    STABLE = "stable"              # 稳定
    VOLATILE = "volatile"          # 波动


@dataclass
class EmotionContext:
    """情感上下文"""
    recent_topics: List[str] = field(default_factory=list)
    active_session: Optional[str] = None  # 当前活动会话ID
    user_situation: Optional[str] = None  # 用户当前处境


@dataclass
class EmotionConfig:
    """情感配置"""
    history_size: int = 100  # 历史记录最大数量
    window_size: int = 7  # 分析窗口大小（天）
    min_confidence: float = 0.3  # 最小置信度阈值


@dataclass
class CurrentEmotionState:
    """当前情感状态"""
    primary: EmotionType = EmotionType.NEUTRAL
    secondary: List[EmotionType] = field(default_factory=list)  # 次要情感
    intensity: float = 0.5  # 情感强度：0-1
    confidence: float = 0.5  # 置信度：0-1
    detected_at: datetime = field(default_factory=datetime.now)
    contextual_correction: bool = False  # 是否经过上下文修正


@dataclass
class EmotionalTrajectory:
    """情感轨迹分析"""
    trend: TrendType = TrendType.STABLE
    volatility: float = 0.5  # 波动性：0-1
    anomaly_detected: bool = False  # 是否检测到异常
    needs_intervention: bool = False  # 是否需要干预
    last_intervention: Optional[datetime] = None  # 上次干预时间


@dataclass
class EmotionalState:
    """情感状态数据模型
    
    记录用户的情感状态、历史轨迹和模式
    """
    
    # ========== 当前状态 ==========
    current: CurrentEmotionState = field(default_factory=CurrentEmotionState)
    
    # ========== 历史记录 ==========
    history: List[CurrentEmotionState] = field(default_factory=list)
    
    # ========== 时序分析 ==========
    trajectory: EmotionalTrajectory = field(default_factory=EmotionalTrajectory)
    
    # ========== 模式和触发器 ==========
    patterns: Dict[str, int] = field(default_factory=dict)  # 情感出现次数统计
    triggers: List[Dict[str, Any]] = field(default_factory=list)  # 情感触发器列表
    soothers: List[Dict[str, Any]] = field(default_factory=list)  # 情感缓解因素列表
    
    # ========== 上下文 ==========
    context: EmotionContext = field(default_factory=EmotionContext)
    
    # ========== 配置 ==========
    config: EmotionConfig = field(default_factory=EmotionConfig)
    
    # ========== 元数据 ==========
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def update_current_emotion(
        self,
        primary: EmotionType,
        intensity: float,
        confidence: float,
        secondary: List[EmotionType] = None
    ):
        """更新当前情感状态"""
        # 保存到历史
        self.history.append(self.current)
        
        # 限制历史记录大小
        if len(self.history) > self.config.history_size:
            self.history.pop(0)
        
        # 更新当前状态
        self.current = CurrentEmotionState(
            primary=primary,
            secondary=secondary or [],
            intensity=intensity,
            confidence=confidence,
            detected_at=datetime.now()
        )
        
        # 更新模式统计
        self.patterns[primary.value] = self.patterns.get(primary.value, 0) + 1
        
        # 分析情感轨迹
        self._analyze_trajectory()
    
    def _analyze_trajectory(self):
        """分析情感轨迹"""
        if len(self.history) < self.config.window_size:
            return
        
        # 获取最近N天的情感历史
        recent_emotions = self.history[-self.config.window_size:]
        
        # 计算情感变化
        positive_count = sum(1 for e in recent_emotions if e.primary in [EmotionType.JOY, EmotionType.EXCITEMENT, EmotionType.CALM])
        negative_count = sum(1 for e in recent_emotions if e.primary in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.ANXIETY, EmotionType.FEAR, EmotionType.DISGUST])
        
        # 计算波动性
        intensity_values = [e.intensity for e in recent_emotions]
        avg_intensity = sum(intensity_values) / len(intensity_values)
        variance = sum((x - avg_intensity) ** 2 for x in intensity_values) / len(intensity_values)
        self.trajectory.volatility = min(1.0, variance)  # 归一化到0-1
        
        # 判断趋势
        if len(self.history) >= 2:
            prev_positive = sum(1 for e in self.history[-self.config.window_size*2:-self.config.window_size] 
                              if e.primary in [EmotionType.JOY, EmotionType.EXCITEMENT])
            prev_negative = sum(1 for e in self.history[-self.config.window_size*2:-self.config.window_size] 
                              if e.primary in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.ANXIETY])
            
            if positive_count > prev_positive and negative_count < prev_negative:
                self.trajectory.trend = TrendType.IMPROVING
            elif positive_count < prev_positive and negative_count > prev_negative:
                self.trajectory.trend = TrendType.DETERIORATING
            elif self.trajectory.volatility > 0.6:
                self.trajectory.trend = TrendType.VOLATILE
            else:
                self.trajectory.trend = TrendType.STABLE
        
        # 检测异常
        if negative_count / len(recent_emotions) > 0.6:
            self.trajectory.anomaly_detected = True
            self.trajectory.needs_intervention = True
        elif self.current.intensity > 0.8 and self.current.primary in [EmotionType.ANGER, EmotionType.ANXIETY]:
            self.trajectory.anomaly_detected = True
            self.trajectory.needs_intervention = True
        else:
            self.trajectory.anomaly_detected = False
    
    def add_trigger(self, trigger_type: str, description: str, emotion: EmotionType):
        """添加情感触发器
        
        Args:
            trigger_type: 触发器类型
            description: 描述
            emotion: 触发的情感
        """
        trigger = {
            "type": trigger_type,
            "description": description,
            "emotion": emotion.value,
            "detected_at": datetime.now().isoformat()
        }
        self.triggers.append(trigger)
    
    def add_soothe(self, soothe_type: str, description: str, emotion: EmotionType):
        """添加情感缓解因素
        
        Args:
            soothe_type: 缓解类型
            description: 描述
            emotion: 缓解的情感
        """
        soothe = {
            "type": soothe_type,
            "description": description,
            "emotion": emotion.value,
            "detected_at": datetime.now().isoformat()
        }
        self.soothers.append(soothe)
    
    def get_negative_ratio(self) -> float:
        """计算负面情感占比"""
        total = sum(self.patterns.values())
        if total == 0:
            return 0.0
        
        negative_emotions = [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.ANXIETY, 
                            EmotionType.FEAR, EmotionType.DISGUST]
        negative_count = sum(count for emotion, count in self.patterns.items() 
                            if emotion in [e.value for e in negative_emotions])
        
        return negative_count / total
    
    def should_filter_positive(self) -> bool:
        """判断是否应该过滤高强度正面记忆
        
        当用户心情不好时，避免检索高强度正面记忆
        """
        negative_emotions = [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.ANXIETY]
        return (self.current.primary in negative_emotions and 
                self.current.intensity > 0.6)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        data = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, CurrentEmotionState):
                data[key] = {
                    "primary": value.primary.value,
                    "secondary": [e.value for e in value.secondary],
                    "intensity": value.intensity,
                    "confidence": value.confidence,
                    "detected_at": value.detected_at.isoformat(),
                    "contextual_correction": value.contextual_correction
                }
            elif isinstance(value, EmotionalTrajectory):
                data[key] = {
                    "trend": value.trend.value,
                    "volatility": value.volatility,
                    "anomaly_detected": value.anomaly_detected,
                    "needs_intervention": value.needs_intervention,
                    "last_intervention": value.last_intervention.isoformat() if value.last_intervention else None
                }
            elif isinstance(value, EmotionContext):
                data[key] = {
                    "recent_topics": value.recent_topics,
                    "active_session": value.active_session,
                    "user_situation": value.user_situation
                }
            elif isinstance(value, EmotionConfig):
                data[key] = {
                    "history_size": value.history_size,
                    "window_size": value.window_size,
                    "min_confidence": value.min_confidence
                }
            elif isinstance(value, list) and value and isinstance(value[0], CurrentEmotionState):
                data[key] = [
                    {
                        "primary": e.primary.value,
                        "secondary": [s.value for s in e.secondary],
                        "intensity": e.intensity,
                        "confidence": e.confidence,
                        "detected_at": e.detected_at.isoformat(),
                        "contextual_correction": e.contextual_correction
                    }
                    for e in value
                ]
            else:
                data[key] = value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalState':
        """从字典创建EmotionalState对象"""
        # 处理嵌套对象
        if 'current' in data:
            current_data = data['current']
            data['current'] = CurrentEmotionState(
                primary=EmotionType(current_data.get('primary', 'neutral')),
                secondary=[EmotionType(s) for s in current_data.get('secondary', [])],
                intensity=current_data.get('intensity', 0.5),
                confidence=current_data.get('confidence', 0.5),
                detected_at=datetime.fromisoformat(current_data.get('detected_at', datetime.now().isoformat())),
                contextual_correction=current_data.get('contextual_correction', False)
            )
        
        if 'history' in data and data['history']:
            data['history'] = [
                CurrentEmotionState(
                    primary=EmotionType(e.get('primary', 'neutral')),
                    secondary=[EmotionType(s) for s in e.get('secondary', [])],
                    intensity=e.get('intensity', 0.5),
                    confidence=e.get('confidence', 0.5),
                    detected_at=datetime.fromisoformat(e.get('detected_at', datetime.now().isoformat())),
                    contextual_correction=e.get('contextual_correction', False)
                )
                for e in data['history']
            ]
        
        if 'trajectory' in data:
            traj_data = data['trajectory']
            data['trajectory'] = EmotionalTrajectory(
                trend=TrendType(traj_data.get('trend', 'stable')),
                volatility=traj_data.get('volatility', 0.5),
                anomaly_detected=traj_data.get('anomaly_detected', False),
                needs_intervention=traj_data.get('needs_intervention', False),
                last_intervention=datetime.fromisoformat(traj_data['last_intervention']) 
                    if traj_data.get('last_intervention') else None
            )
        
        if 'context' in data:
            ctx_data = data['context']
            data['context'] = EmotionContext(
                recent_topics=ctx_data.get('recent_topics', []),
                active_session=ctx_data.get('active_session'),
                user_situation=ctx_data.get('user_situation')
            )
        
        if 'config' in data:
            cfg_data = data['config']
            data['config'] = EmotionConfig(
                history_size=cfg_data.get('history_size', 100),
                window_size=cfg_data.get('window_size', 7),
                min_confidence=cfg_data.get('min_confidence', 0.3)
            )
        
        return cls(**data)

"""
人格管理器
管理Bot人格与用户画像的协调和冲突检测
"""

from enum import Enum
from typing import List, Dict, Any, Optional

from iris_memory.utils.logger import logger


class ConflictType(str, Enum):
    """冲突类型"""
    
    # 风格冲突：用户偏好与Bot风格不符
    STYLE_CONFLICT = "style_conflict"
    
    # 回复频率冲突：用户希望少回复，但Bot主动
    FREQUENCY_CONFLICT = "frequency_conflict"
    
    # 情感冲突：用户当前情感与Bot风格冲突
    EMOTION_CONFLICT = "emotion_conflict"
    
    # 内容冲突：用户画像与Bot角色设定冲突
    CONTENT_CONFLICT = "content_conflict"


class CoordinationStrategy(str, Enum):
    """协调策略"""
    
    # Bot优先：Bot人格优先
    BOT_PRIORITY = "bot_priority"
    
    # 用户优先：用户画像优先
    USER_PRIORITY = "user_priority"
    
    # 混合：平衡两者
    HYBRID = "hybrid"
    
    # 动态：根据情况动态选择
    DYNAMIC = "dynamic"


class PersonaConflictDetector:
    """人格冲突检测器
    
    检测用户画像与Bot人格的潜在冲突
    """
    
    def __init__(self):
        """初始化冲突检测器"""
        # Bot人格关键词（示例，应从配置加载）
        self.bot_persona_keywords = {
            "friendly": ["热情", "友好", "亲切", "关心"],
            "professional": ["专业", "礼貌", "正式", "客观"],
            "humorous": ["幽默", "风趣", "轻松", "搞笑"],
            "calm": ["平和", "冷静", "理性", "稳重"]
        }
        
        # 用户偏好关键词
        self.user_preference_keywords = {
            "less_talkative": ["不喜欢说太多", "安静", "简洁", "不爱说话"],
            "not_enthusiastic": ["不要热情", "冷淡", "正式", "严肃"],
            "not_friendly": ["不要亲切", "疏远", "专业", "正式"]
        }
    
    def detect_conflicts(
        self,
        user_persona: Dict[str, Any],
        bot_persona: str = "friendly"
    ) -> List[Dict[str, Any]]:
        """检测人格冲突
        
        Args:
            user_persona: 用户画像数据
            bot_persona: Bot人格类型
            
        Returns:
            List[Dict]: 冲突列表
            [{"type": ConflictType, "description": str, "severity": int}]
        """
        conflicts = []
        
        # 检查风格冲突
        style_conflicts = self._detect_style_conflicts(user_persona, bot_persona)
        conflicts.extend(style_conflicts)
        
        # 检查情感冲突
        emotion_conflicts = self._detect_emotion_conflicts(user_persona)
        conflicts.extend(emotion_conflicts)
        
        return conflicts
    
    def _detect_style_conflicts(
        self,
        user_persona: Dict[str, Any],
        bot_persona: str
    ) -> List[Dict[str, Any]]:
        """检测风格冲突
        
        Args:
            user_persona: 用户画像
            bot_persona: Bot人格
            
        Returns:
            List[Dict]: 冲突列表
        """
        conflicts = []
        
        # 从用户画像中提取偏好关键词
        user_preferences = user_persona.get("preferences", {})
        
        # 检查每个偏好
        for preference_type, preference_value in user_preferences.items():
            # 检查是否是负面偏好
            for pref_key, keywords in self.user_preference_keywords.items():
                if pref_key in str(preference_value).lower():
                    # 找到负面偏好关键词
                    conflicts.append({
                        "type": ConflictType.STYLE_CONFLICT,
                        "description": f"用户偏好'{preference_value}'可能与Bot风格冲突",
                        "preference": preference_value,
                        "bot_style": bot_persona,
                        "severity": 2  # 中等严重性
                    })
        
        return conflicts
    
    def _detect_emotion_conflicts(
        self,
        user_persona: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """检测情感冲突
        
        Args:
            user_persona: 用户画像
            
        Returns:
            List[Dict]: 冲突列表
        """
        conflicts = []
        
        # 检查情感轨迹
        emotional_state = user_persona.get("emotional", {})
        trajectory = emotional_state.get("trajectory", "")
        
        # 如果用户情感持续负面，需要注意回复风格
        if trajectory in ["deteriorating", "volatile"]:
            conflicts.append({
                "type": ConflictType.EMOTION_CONFLICT,
                "description": "用户情感持续负面，建议调整回复风格",
                "trajectory": trajectory,
                "severity": 1  # 低严重性
            })
        
        return conflicts
    
    def get_resolution_suggestions(
        self,
        conflicts: List[Dict[str, Any]]
    ) -> List[str]:
        """获取冲突解决建议
        
        Args:
            conflicts: 冲突列表
            
        Returns:
            List[str]: 建议列表
        """
        suggestions = []
        
        for conflict in conflicts:
            conflict_type = conflict["type"]
            
            if conflict_type == ConflictType.STYLE_CONFLICT:
                suggestions.append(
                    f"建议：{conflict['description']}"
                )
                if conflict["bot_style"] == "friendly":
                    suggestions.append("考虑临时调整为更专业的风格")
                elif conflict["bot_style"] == "professional":
                    suggestions.append("考虑适当增加亲和力")
            
            elif conflict_type == ConflictType.EMOTION_CONFLICT:
                suggestions.append(
                    f"建议：{conflict['description']}"
                )
                suggestions.append("优先倾听和支持，减少主动输出")
                suggestions.append("使用温和的回复风格")
        
        # 去重
        return list(set(suggestions))


class PersonaCoordinator:
    """人格协调器
    
    协调Bot人格与用户画像，确保回复的一致性
    """
    
    def __init__(self, strategy: CoordinationStrategy = CoordinationStrategy.HYBRID):
        """初始化协调器
        
        Args:
            strategy: 协调策略
        """
        self.strategy = strategy
        self.conflict_detector = PersonaConflictDetector()
    
    def coordinate_persona(
        self,
        user_persona: Dict[str, Any],
        bot_persona: str = "friendly",
        memory_context: str = ""
    ) -> str:
        """协调人格并生成prompt提示
        
        Args:
            user_persona: 用户画像
            bot_persona: Bot人格
            memory_context: 记忆上下文
            
        Returns:
            str: 人格协调提示
        """
        # 检测冲突
        conflicts = self.conflict_detector.detect_conflicts(user_persona, bot_persona)
        
        # 根据策略生成提示
        if self.strategy == CoordinationStrategy.BOT_PRIORITY:
            return self._bot_priority_prompt(bot_persona, memory_context)
        elif self.strategy == CoordinationStrategy.USER_PRIORITY:
            return self._user_priority_prompt(user_persona, memory_context, conflicts)
        elif self.strategy == CoordinationStrategy.HYBRID:
            return self._hybrid_prompt(bot_persona, user_persona, memory_context, conflicts)
        else:  # DYNAMIC
            return self._dynamic_prompt(bot_persona, user_persona, memory_context, conflicts)
    
    def _bot_priority_prompt(
        self,
        bot_persona: str,
        memory_context: str
    ) -> str:
        """Bot优先策略：Bot人格优先"""
        
        prompt_parts = []
        
        # Bot人格描述
        if bot_persona == "friendly":
            prompt_parts.append("保持友好热情的风格")
        elif bot_persona == "professional":
            prompt_parts.append("保持专业客观的风格")
        elif bot_persona == "humorous":
            prompt_parts.append("保持幽默风趣的风格")
        elif bot_persona == "calm":
            prompt_parts.append("保持平和理性的风格")
        
        # 记忆上下文处理
        if memory_context:
            prompt_parts.append(
                "同时参考用户画像中的偏好，但在风格上保持一致性"
            )
        
        return "\n".join(prompt_parts)
    
    def _user_priority_prompt(
        self,
        user_persona: Dict[str, Any],
        memory_context: str,
        conflicts: List[Dict[str, Any]]
    ) -> str:
        """用户优先策略：用户画像优先"""
        
        prompt_parts = []
        
        # 提取用户偏好
        preferences = user_persona.get("preferences", {})
        
        # 根据偏好生成提示
        if preferences.get("style") == "less_talkative":
            prompt_parts.append("回复要简洁，不要过于热情")
        elif preferences.get("style") == "not_enthusiastic":
            prompt_parts.append("回复要冷静克制，不要过于热情")
        
        # 情感相关提示
        emotional_state = user_persona.get("emotional", {})
        trajectory = emotional_state.get("trajectory", "")
        if trajectory == "deteriorating":
            prompt_parts.append("用户情感状态不佳，需要更多的支持和理解")
        elif trajectory == "volatile":
            prompt_parts.append("用户情绪波动较大，保持回复的稳定性")
        
        # 记忆上下文
        if memory_context:
            prompt_parts.append(
                "优先响应用户画像中的偏好，即使这与默认风格不同"
            )
        
        # 冲突处理
        if conflicts:
            suggestions = self.conflict_detector.get_resolution_suggestions(conflicts)
            if suggestions:
                prompt_parts.append("\n注意事项：")
                prompt_parts.extend([f"- {s}" for s in suggestions[:3]])
        
        return "\n".join(prompt_parts)
    
    def _hybrid_prompt(
        self,
        bot_persona: str,
        user_persona: Dict[str, Any],
        memory_context: str,
        conflicts: List[Dict[str, Any]]
    ) -> str:
        """混合策略：平衡Bot人格和用户画像"""
        
        prompt_parts = []
        
        # 基础人格保持
        if bot_persona == "friendly":
            prompt_parts.append("保持基本的友好热情风格")
        elif bot_persona == "professional":
            prompt_parts.append("保持基本的专业客观风格")
        
        # 用户画像调整
        prompt_parts.append(
            "同时根据用户画像调整回复细节和语气"
        )
        
        # 特定偏好处理
        preferences = user_persona.get("preferences", {})
        if preferences.get("style"):
            prompt_parts.append(f"注意用户的风格偏好：{preferences['style']}")
        
        # 情感协调
        emotional_state = user_persona.get("emotional", {})
        trajectory = emotional_state.get("trajectory", "")
        if trajectory in ["deteriorating", "volatile"]:
            prompt_parts.append(
                "当前用户情感状态不佳，适当调整回复风格，更加温和支持"
            )
        
        # 记忆上下文
        if memory_context:
            prompt_parts.append(
                "参考记忆上下文中的用户偏好，但不要被其限制"
            )
        
        # 冲突处理
        if conflicts:
            suggestions = self.conflict_detector.get_resolution_suggestions(conflicts)
            if suggestions:
                prompt_parts.append("\n注意事项：")
                prompt_parts.extend([f"- {s}" for s in suggestions[:3]])
        
        return "\n".join(prompt_parts)
    
    def _dynamic_prompt(
        self,
        bot_persona: str,
        user_persona: Dict[str, Any],
        memory_context: str,
        conflicts: List[Dict[str, Any]]
    ) -> str:
        """动态策略：根据情况动态选择"""
        
        prompt_parts = []
        
        # 检查冲突严重性
        has_high_severity = any(
            c.get("severity", 0) >= 2 for c in conflicts
        )
        
        if has_high_severity:
            # 有高严重冲突，用户优先
            return self._user_priority_prompt(user_persona, memory_context, conflicts)
        
        # 检查情感状态
        emotional_state = user_persona.get("emotional", {})
        trajectory = emotional_state.get("trajectory", "")
        
        if trajectory == "deteriorating":
            # 情感恶化，用户优先
            return self._user_priority_prompt(user_persona, memory_context, conflicts)
        elif trajectory == "volatile":
            # 情感波动，混合策略
            return self._hybrid_prompt(bot_persona, user_persona, memory_context, conflicts)
        
        # 默认：混合策略
        return self._hybrid_prompt(bot_persona, user_persona, memory_context, conflicts)
    
    def format_context_with_persona(
        self,
        memory_context: str,
        user_persona: Dict[str, Any],
        bot_persona: str = "friendly"
    ) -> str:
        """格式化记忆上下文，包含人格协调信息
        
        Args:
            memory_context: 记忆上下文
            user_persona: 用户画像
            bot_persona: Bot人格
            
        Returns:
            str: 格式化的上下文
        """
        # 检测冲突
        conflicts = self.conflict_detector.detect_conflicts(user_persona, bot_persona)
        
        # 生成协调提示
        coordination_hint = self.coordinate_persona(
            user_persona, bot_persona, memory_context
        )
        
        # 构建最终上下文
        formatted_context = memory_context
        
        if conflicts and coordination_hint:
            formatted_context += f"\n\n【人格协调提示】\n{coordination_hint}"
        
        return formatted_context
    
    def set_strategy(self, strategy: CoordinationStrategy):
        """设置协调策略
        
        Args:
            strategy: 协调策略
        """
        self.strategy = strategy
        logger.info(f"Persona coordination strategy set to: {strategy.value}")

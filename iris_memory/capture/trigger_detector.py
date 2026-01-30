"""
触发器检测器
检测消息中是否包含记忆捕获的触发器
"""

import re
from typing import List, Dict, Any, Optional

from iris_memory.core.types import TriggerType


class TriggerDetector:
    """触发器检测器
    
    根据companion-memory框架检测记忆捕获触发器：
    - 显式触发器：记住、重要、关键等
    - 偏好触发器：喜欢、讨厌、偏好、想要
    - 情感触发器：觉得、感到、感觉、心情
    - 关系触发器：我们是、我们算是、你对我来说是
    - 事实触发器：我是、我有、我做
    - 边界触发器：不要、不想、隐私、秘密
    """
    
    def __init__(self):
        """初始化触发器检测器"""
        self._init_triggers()
    
    def _init_triggers(self):
        """初始化触发器模式"""
        self.triggers = {
            TriggerType.EXPLICIT: [
                r"记住", r"重要", r"关键", r"记住这个", r"别忘了",
                r"要记住", r"mark", r"important", r"remember"
            ],
            TriggerType.PREFERENCE: [
                r"喜欢", r"讨厌", r"爱", r"恨", r"偏好", r"想要",
                r"想要的是", r"不喜欢", r"不喜欢的是", r"偏好是",
                r"like", r"love", r"hate", r"prefer", r"want"
            ],
            TriggerType.EMOTION: [
                r"觉得", r"感到", r"感觉", r"心情", r"情绪",
                r"感到很", r"觉得很", r"感觉像", r"心情好", r"心情不好",
                r"feel", r"feeling", r"mood"
            ],
            TriggerType.RELATIONSHIP: [
                r"我们是", r"我们算是", r"你对我来说是", r"我们关系是",
                r"我们是朋友", r"我们是家人", r"关系", r"对我来说",
                r"we are", r"you're like", r"relationship"
            ],
            TriggerType.FACT: [
                r"我是", r"我有", r"我做", r"我在", r"我叫",
                r"我的工作是", r"我住", r"我来自", r"我的爱好是",
                r"出生于", r"出生在", r"生日是",
                r"i am", r"i have", r"i do", r"i work as", r"i live in"
            ],
            TriggerType.BOUNDARY: [
                r"不要", r"不想", r"隐私", r"秘密", r"不要问",
                r"不许", r"别", r"不喜欢别人知道", r"不要告诉别人",
                r"don't", r"never", r"private", r"secret", r"don't ask"
            ]
        }
        
        # 负样本模式（不应捕获的闲聊）
        self.negative_patterns = [
            r"^天气.*[？?]?$",  # 关于天气的简单问句
            r"^在吗",  # "在吗"问候
            r"^你好",  # 问候语
            r"^[嗯哦好]$|^好的?$|^嗯$",  # 简单确认
            r"^(哈哈|呵呵|嘻嘻)$",  # 笑声
            r"^(谢谢|感谢)$",  # 感谢
        ]
    
    def detect_triggers(self, text: str) -> List[Dict[str, Any]]:
        """检测文本中的触发器
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict[str, Any]]: 检测到的触发器列表
            [{
                "type": TriggerType,
                "pattern": str,
                "confidence": float,
                "position": int
            }]
        """
        if not text:
            return []
        
        # 检查负样本
        if self._is_negative_sample(text):
            return []
        
        triggers = []
        
        for trigger_type, patterns in self.triggers.items():
            for pattern in patterns:
                # 查找所有匹配
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    triggers.append({
                        "type": trigger_type,
                        "pattern": pattern,
                        "confidence": self._calculate_confidence(text, pattern, trigger_type),
                        "position": match.start()
                    })
        
        return triggers
    
    def _is_negative_sample(self, text: str) -> bool:
        """判断是否为负样本（不应捕获）"""
        text = text.strip()
        
        for pattern in self.negative_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # 长度过滤（太短的消息可能不包含有用信息）
        if len(text) < 3:
            return True
        
        return False
    
    def _calculate_confidence(
        self,
        text: str,
        pattern: str,
        trigger_type: TriggerType
    ) -> float:
        """计算触发器置信度
        
        Args:
            text: 输入文本
            pattern: 匹配的模式
            trigger_type: 触发器类型
            
        Returns:
            float: 置信度（0-1）
        """
        # 显式触发器置信度最高
        if trigger_type == TriggerType.EXPLICIT:
            return 0.95
        
        # 边界触发器置信度较高
        if trigger_type == TriggerType.BOUNDARY:
            return 0.9
        
        # 事实和偏好触发器置信度中等
        if trigger_type in [TriggerType.FACT, TriggerType.PREFERENCE]:
            return 0.8
        
        # 情感和关系触发器置信度稍低
        if trigger_type in [TriggerType.EMOTION, TriggerType.RELATIONSHIP]:
            return 0.7
        
        return 0.6
    
    def has_trigger(self, text: str) -> bool:
        """判断文本是否包含任何触发器
        
        Args:
            text: 输入文本
            
        Returns:
            bool: 是否包含触发器
        """
        triggers = self.detect_triggers(text)
        return len(triggers) > 0
    
    def get_trigger_types(self, text: str) -> List[TriggerType]:
        """获取文本中的所有触发器类型
        
        Args:
            text: 输入文本
            
        Returns:
            List[TriggerType]: 触发器类型列表（去重）
        """
        triggers = self.detect_triggers(text)
        return list(set([t["type"] for t in triggers]))
    
    def get_highest_confidence_trigger(self, text: str) -> Optional[Dict[str, Any]]:
        """获取置信度最高的触发器
        
        Args:
            text: 输入文本
            
        Returns:
            Optional[Dict[str, Any]]: 置信度最高的触发器，如果没有则返回None
        """
        triggers = self.detect_triggers(text)
        
        if not triggers:
            return None
        
        # 按置信度排序
        triggers.sort(key=lambda x: x["confidence"], reverse=True)
        
        return triggers[0]

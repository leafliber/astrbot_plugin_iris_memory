"""
通用检测结果基类

提供所有LLM增强检测器共用的结果处理工具：
- 置信度限制
- 列表类型转换
- 枚举映射
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar('T')


@dataclass
class BaseDetectionResult:
    """通用检测结果基类
    
    所有LLM增强检测器的结果类型都应继承此基类。
    提供通用字段和工具方法。
    """
    confidence: float
    source: str  # "rule" | "llm" | "hybrid"
    reason: str

    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """限制数值在有效范围内
        
        Args:
            value: 待限制的值
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            限制后的值
        """
        return max(min_val, min(max_val, value))

    @staticmethod
    def ensure_list(value: Any) -> List[Any]:
        """统一列表类型转换
        
        将字符串或其他类型统一转为列表。
        
        Args:
            value: 待转换的值
            
        Returns:
            列表
        """
        if isinstance(value, str):
            return [value]
        return value if isinstance(value, list) else []

    @staticmethod
    def parse_confidence(data: Dict[str, Any], key: str = "confidence", default: float = 0.5) -> float:
        """从字典中解析并限制置信度
        
        Args:
            data: 数据字典
            key: 置信度键名
            default: 默认值
            
        Returns:
            限制在 [0.0, 1.0] 范围内的置信度
        """
        return BaseDetectionResult.clamp(float(data.get(key, default)))

    @staticmethod
    def map_enum(value: str, enum_map: Dict[str, Any], default: Any = None) -> Any:
        """将字符串映射到枚举值
        
        Args:
            value: 待映射的字符串
            enum_map: 枚举映射字典
            default: 默认值
            
        Returns:
            枚举值
        """
        return enum_map.get(value.lower() if isinstance(value, str) else "", default)

    @staticmethod
    def validate_string(value: str, allowed: List[str], default: str) -> str:
        """验证字符串是否在允许值列表中
        
        Args:
            value: 待验证的字符串
            allowed: 允许的值列表
            default: 默认值
            
        Returns:
            验证后的字符串
        """
        value_lower = value.lower() if isinstance(value, str) else ""
        return value_lower if value_lower in allowed else default

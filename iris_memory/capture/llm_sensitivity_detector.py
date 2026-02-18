"""
LLM增强敏感度检测器
使用LLM进行语义层面的敏感信息检测
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core.types import SensitivityLevel
from iris_memory.capture.sensitivity_detector import SensitivityDetector
from iris_memory.processing.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedBase,
    LLMCallResult,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("llm_sensitivity_detector")


SENSITIVITY_DETECTION_PROMPT = """分析以下文本是否包含敏感信息。

## 敏感度等级定义
- CRITICAL（极度敏感）：身份证号、银行卡号、密码、手机号、邮箱、详细住址等可直接识别个人身份的信息
- SENSITIVE（敏感）：健康状况、财务信息、工作单位、学校等私人信息
- PRIVATE（私人）：家庭关系、个人经历、社交关系等
- PERSONAL（个人偏好）：生活方式、消费习惯、兴趣爱好等
- PUBLIC（公开）：普通对话内容，不涉及隐私

## 上下文理解
请识别：
1. 隐含敏感信息（如"我在医院工作"暗示医疗背景）
2. 上下文区分（"我的密码是123456" vs "密码设置建议"）
3. 是否真的需要保护

## 待分析文本
{text}

## 输出格式
请以JSON格式返回：
```json
{{
  "level": "CRITICAL|SENSITIVE|PRIVATE|PERSONAL|PUBLIC",
  "confidence": 0.0-1.0,
  "detected_entities": ["检测到的敏感实体"],
  "reason": "简短的判断理由",
  "implicit_info": ["隐含的敏感信息，如有"]
}}
```

仅返回JSON，不要有其他文字。"""


@dataclass
class SensitivityDetectionResult:
    """敏感度检测结果"""
    level: SensitivityLevel
    entities: List[str]
    confidence: float
    reason: str
    implicit_info: List[str]
    source: str  # "rule" | "llm" | "hybrid"


class LLMSensitivityDetector(LLMEnhancedBase):
    """LLM增强敏感度检测器
    
    支持三种模式：
    - rule: 仅使用规则检测（快速）
    - llm: 仅使用LLM检测（准确）
    - hybrid: 规则预筛 + LLM确认（推荐）
    """
    
    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        mode: DetectionMode = DetectionMode.HYBRID,
        confidence_threshold: float = 0.7,
        daily_limit: int = 0,
    ):
        super().__init__(
            astrbot_context=astrbot_context,
            provider_id=provider_id,
            mode=mode,
            daily_limit=daily_limit,
            max_tokens=200,
        )
        self._confidence_threshold = confidence_threshold
        self._rule_detector = SensitivityDetector()
    
    async def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SensitivityDetectionResult:
        """检测文本敏感度
        
        Args:
            text: 输入文本
            context: 上下文信息
            
        Returns:
            SensitivityDetectionResult: 检测结果
        """
        if not text:
            return SensitivityDetectionResult(
                level=SensitivityLevel.PUBLIC,
                entities=[],
                confidence=1.0,
                reason="空文本",
                implicit_info=[],
                source="rule"
            )
        
        if self._mode == DetectionMode.RULE:
            return self._rule_detect(text, context)
        elif self._mode == DetectionMode.LLM:
            return await self._llm_detect(text, context)
        else:
            return await self._hybrid_detect(text, context)
    
    def _rule_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SensitivityDetectionResult:
        """规则检测"""
        level, entities = self._rule_detector.detect_sensitivity(text, context)
        return SensitivityDetectionResult(
            level=level,
            entities=entities,
            confidence=0.8,
            reason="规则匹配",
            implicit_info=[],
            source="rule"
        )
    
    async def _llm_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SensitivityDetectionResult:
        """LLM检测"""
        prompt = SENSITIVITY_DETECTION_PROMPT.format(text=text[:500])
        result = await self._call_llm(prompt)
        
        if not result.success or not result.parsed_json:
            logger.debug(f"LLM sensitivity detection failed, falling back to rule")
            return self._rule_detect(text, context)
        
        return self._parse_llm_result(result.parsed_json)
    
    async def _hybrid_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SensitivityDetectionResult:
        """混合检测：规则预筛 + LLM确认"""
        rule_result = self._rule_detect(text, context)
        
        if rule_result.level == SensitivityLevel.PUBLIC:
            if not self._has_potential_sensitive_keywords(text):
                return rule_result
        
        if rule_result.level.value >= SensitivityLevel.SENSITIVE.value:
            llm_result = await self._llm_detect(text, context)
            if llm_result.confidence >= self._confidence_threshold:
                llm_result.source = "hybrid"
                return llm_result
        
        if rule_result.level.value >= SensitivityLevel.PRIVATE.value:
            llm_result = await self._llm_detect(text, context)
            if llm_result.confidence >= self._confidence_threshold:
                llm_result.source = "hybrid"
                return llm_result
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _has_potential_sensitive_keywords(self, text: str) -> bool:
        """检查是否有潜在的敏感关键词"""
        potential_keywords = [
            "密码", "账号", "银行卡", "信用卡", "身份证",
            "住址", "地址", "工资", "收入", "病历", "诊断",
            "password", "account", "credit card", "address"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in potential_keywords)
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> SensitivityDetectionResult:
        """解析LLM结果"""
        level_str = data.get("level", "PUBLIC").upper()
        level_map = {
            "CRITICAL": SensitivityLevel.CRITICAL,
            "SENSITIVE": SensitivityLevel.SENSITIVE,
            "PRIVATE": SensitivityLevel.PRIVATE,
            "PERSONAL": SensitivityLevel.PERSONAL,
            "PUBLIC": SensitivityLevel.PUBLIC,
        }
        level = level_map.get(level_str, SensitivityLevel.PUBLIC)
        
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        
        entities = data.get("detected_entities", [])
        if isinstance(entities, str):
            entities = [entities]
        
        implicit_info = data.get("implicit_info", [])
        if isinstance(implicit_info, str):
            implicit_info = [implicit_info]
        
        return SensitivityDetectionResult(
            level=level,
            entities=entities,
            confidence=confidence,
            reason=data.get("reason", "LLM判断"),
            implicit_info=implicit_info,
            source="llm"
        )
    
    def should_filter(self, result: SensitivityDetectionResult) -> bool:
        """判断是否应该过滤"""
        return result.level == SensitivityLevel.CRITICAL
    
    def get_encryption_required(self, result: SensitivityDetectionResult) -> bool:
        """判断是否需要加密"""
        return result.level.value >= SensitivityLevel.PRIVATE.value

"""
敏感度检测器
自动检测消息中的敏感信息
"""

import re
from typing import List, Tuple, Optional

from iris_memory.core.types import SensitivityLevel


class SensitivityDetector:
    """敏感度检测器
    
    自动检测敏感信息：
    - 关键词匹配：身份证号、银行卡号、密码等
    - 上下文关联：医院工作 → 隐含医疗信息
    - 组合检测：姓名+身份证号 → CRITICAL
    - 模式检测：18位数字→身份证，16位数字→银行卡
    - 语义理解：健康状况、财务状况、关系隐私
    """
    
    def __init__(self):
        """初始化敏感度检测器"""
        self._init_patterns()
    
    def _init_patterns(self):
        """初始化敏感信息模式"""
        # CRITICAL（极度敏感）
        self.critical_patterns = [
            # 身份证号（18位数字，最后一位可能是X）
            # 使用 lookaround 来确保匹配独立的数字序列
            r'(?<![0-9])\d{17}[\dXx](?![0-9])',
            # 银行卡号（16-19位数字）
            r'(?<![0-9])\d{16,19}(?![0-9])',
            # 密码相关
            r'密码[:：是]\S+', r'password[:：]\S+',
            # 手机号（11位）
            r'(?<![0-9])1[3-9]\d{9}(?![0-9])',
            # 邮箱（不使用 \b 以支持中文环境）
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        ]

        # SENSITIVE（敏感）
        self.sensitive_patterns = [
            # 健康状况
            r'病情|疾病|治疗|医院|医生|药|症状|诊断',
            # 财务信息
            r'工资|收入|存款|贷款|信用卡|债务|财务|资产|房贷',
            # 住址
            r'地址|住在|居住地|家庭地址|住址',
        ]
        
        # PRIVATE（私人）
        self.private_patterns = [
            # 工作单位
            r'公司|单位|上班|同事|老板|工作单位',
            # 学校
            r'学校|大学|中学|小学|班级|同学',
            # 家庭关系
            r'家人|父母|孩子|配偶|丈夫|妻子|父母亲',
        ]
        
        # PERSONAL（个人偏好）
        self.personal_patterns = [
            # 生活方式
            r'作息|起床|睡觉|习惯|生活方式',
            # 消费习惯
            r'消费|购物|喜欢买|经常买|消费习惯',
        ]
    
    def detect_sensitivity(self, text: str, context: Optional[dict] = None) -> Tuple[SensitivityLevel, List[str]]:
        """检测文本敏感度
        
        Args:
            text: 输入文本
            context: 上下文信息（可选）
            
        Returns:
            Tuple[SensitivityLevel, List[str]]: (敏感度等级, 检测到的敏感实体列表)
        """
        if not text:
            return SensitivityLevel.PUBLIC, []
        
        detected_entities = []
        max_level = SensitivityLevel.PUBLIC
        
        # 检测CRITICAL级别
        critical_matches = self._detect_patterns(text, self.critical_patterns)
        if critical_matches:
            for match in critical_matches:
                detected_entities.append(f"CRITICAL: {match}")
            max_level = SensitivityLevel.CRITICAL
        
        # 检测SENSITIVE级别
        sensitive_matches = self._detect_patterns(text, self.sensitive_patterns)
        if sensitive_matches:
            for match in sensitive_matches:
                detected_entities.append(f"SENSITIVE: {match}")
            if max_level.value < SensitivityLevel.SENSITIVE.value:
                max_level = SensitivityLevel.SENSITIVE
        
        # 检测PRIVATE级别
        private_matches = self._detect_patterns(text, self.private_patterns)
        if private_matches:
            for match in private_matches:
                detected_entities.append(f"PRIVATE: {match}")
            if max_level.value < SensitivityLevel.PRIVATE.value:
                max_level = SensitivityLevel.PRIVATE
        
        # 检测PERSONAL级别
        personal_matches = self._detect_patterns(text, self.personal_patterns)
        if personal_matches:
            for match in personal_matches:
                detected_entities.append(f"PERSONAL: {match}")
            if max_level.value < SensitivityLevel.PERSONAL.value:
                max_level = SensitivityLevel.PERSONAL
        
        # 上下文关联检测
        context_level = self._detect_contextual_sensitivity(context)
        if context_level and context_level.value > max_level.value:
            max_level = context_level
        
        return max_level, detected_entities
    
    def _detect_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """检测模式匹配
        
        Args:
            text: 输入文本
            patterns: 正则表达式模式列表
            
        Returns:
            List[str]: 匹配的文本片段列表
        """
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text)
            matches.extend(found)
        return matches
    
    def _detect_contextual_sensitivity(self, context: Optional[dict]) -> Optional[SensitivityLevel]:
        """基于上下文检测敏感度
        
        Args:
            context: 上下文信息
            
        Returns:
            Optional[SensitivityLevel]: 检测到的敏感度，如果没有则返回None
        """
        if not context:
            return None
        
        # 检查上下文中的关键词
        text = " ".join([str(v) for v in context.values()])
        
        # 如果上下文中提到医院、医生等，隐含医疗信息
        if any(kw in text for kw in ["医院", "医生", "治疗", "看病"]):
            return SensitivityLevel.SENSITIVE
        
        # 如果上下文中提到银行、金融等，隐含财务信息
        if any(kw in text for kw in ["银行", "贷款", "投资", "理财"]):
            return SensitivityLevel.SENSITIVE
        
        return None
    
    def should_filter(self, sensitivity_level: SensitivityLevel) -> bool:
        """判断是否应该过滤（不存储）该级别的信息
        
        Args:
            sensitivity_level: 敏感度等级
            
        Returns:
            bool: 是否应该过滤
        """
        # CRITICAL级别的信息默认不存储，需要用户明确确认
        return sensitivity_level == SensitivityLevel.CRITICAL
    
    def get_encryption_required(self, sensitivity_level: SensitivityLevel) -> bool:
        """判断是否需要加密

        Args:
            sensitivity_level: 敏感度等级

        Returns:
            bool: 是否需要加密
        """
        # PRIVATE及以上级别需要加密（私人信息和敏感信息都需要保护）
        return sensitivity_level.value >= SensitivityLevel.PRIVATE.value

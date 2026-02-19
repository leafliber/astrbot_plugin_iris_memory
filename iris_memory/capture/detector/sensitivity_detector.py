"""
敏感度检测器
自动检测消息中的敏感信息
"""

import re
from typing import List, Tuple, Optional

from iris_memory.core.types import SensitivityLevel
from iris_memory.utils.logger import get_logger

logger = get_logger("sensitivity_detector")


def _mask_sensitive(value: str) -> str:
    """对敏感信息进行脱敏处理
    
    规则：
    - 长度 <= 4: 全部替换为 *
    - 长度 <= 8: 保留首尾各1字符
    - 长度 > 8: 保留首尾各4字符
    """
    length = len(value)
    if length <= 4:
        return "*" * length
    if length <= 8:
        return value[0] + "*" * (length - 2) + value[-1]
    return value[:4] + "*" * (length - 8) + value[-4:]


def _validate_china_id(digits: str) -> bool:
    """验证中国身份证号校验位（GB 11643-1999）"""
    if len(digits) != 18:
        return False
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_chars = '10X98765432'
    try:
        total = sum(int(digits[i]) * weights[i] for i in range(17))
        return check_chars[total % 11].upper() == digits[17].upper()
    except (ValueError, IndexError):
        return False


def _validate_bank_card(digits: str) -> bool:
    """Luhn 算法验证银行卡号"""
    if not digits.isdigit() or len(digits) < 16 or len(digits) > 19:
        return False
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


class SensitivityDetector:
    """敏感度检测器
    
    自动检测敏感信息：
    - 身份证号（18位 + 校验位验证）
    - 银行卡号（16-19位 + Luhn 校验）
    - 手机号、密码、邮箱
    - 健康、财务、住址等敏感话题
    """
    
    def __init__(self):
        """初始化敏感度检测器"""
        self._init_patterns()
    
    def _init_patterns(self):
        """初始化敏感信息模式"""
        # 身份证号候选：18位数字（最后一位可能是X）
        self._id_card_pattern = re.compile(r'(?<![0-9])\d{17}[\dXx](?![0-9])')
        # 银行卡号候选：16-19位纯数字
        self._bank_card_pattern = re.compile(r'(?<![0-9])\d{16,19}(?![0-9])')
        
        # CRITICAL（极度敏感）— 不再包含身份证/银行卡的宽泛正则
        self.critical_patterns = [
            # 密码相关（支持多种格式：冒号、等号、空格分隔）
            r'密码[\s]*[:：=是][\s]*\S+',
            r'(?:password|passwd|pwd|pass)[\s]*[:：=][\s]*\S+',
            # 手机号（11位）
            r'(?<![0-9])1[3-9]\d{9}(?![0-9])',
            # 邮箱
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
            # API 密钥 / Token 格式（常见前缀 + 长随机串，不区分大小写）
            r'(?i)(?:sk|pk|api[_-]?key|token|secret|access[_-]?key|private[_-]?key)[_-]?[:\s=]+\S{16,}',
            r'(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}',  # GitHub token
            r'(?:Bearer|Basic)\s+[A-Za-z0-9+/=_-]{20,}',    # Auth header
            r'(?i)(?:aws|azure|gcp)[_-]?(?:key|secret|token)[_-]?[:\s=]+\S{16,}',  # 云服务密钥
            # 社保号 / SSN 格式（美式，可选）
            r'\b\d{3}-\d{2}-\d{4}\b',
        ]

        # SENSITIVE（敏感）
        self.sensitive_patterns = [
            r'病情|疾病|治疗|医院|医生|药|症状|诊断',
            r'工资|收入|存款|贷款|信用卡|债务|财务|资产|房贷',
            r'地址|住在|居住地|家庭地址|住址',
        ]
        
        # PRIVATE（私人）
        self.private_patterns = [
            r'公司|单位|上班|同事|老板|工作单位',
            r'学校|大学|中学|小学|班级|同学',
            r'家人|父母|孩子|配偶|丈夫|妻子|父母亲',
        ]
        
        # PERSONAL（个人偏好）
        self.personal_patterns = [
            r'作息|起床|睡觉|习惯|生活方式',
            r'消费|购物|喜欢买|经常买|消费习惯',
        ]
    
    def _detect_validated_credentials(self, text: str) -> List[str]:
        """检测经过校验的身份证号和银行卡号"""
        results: List[str] = []
        
        # 身份证号检测 + 校验位验证
        for m in self._id_card_pattern.finditer(text):
            candidate = m.group()
            if _validate_china_id(candidate):
                results.append(candidate)
        
        # 银行卡号检测 + Luhn 校验
        for m in self._bank_card_pattern.finditer(text):
            candidate = m.group()
            # 排除已被识别为身份证的子串
            if any(candidate in r for r in results):
                continue
            if _validate_bank_card(candidate):
                results.append(candidate)
        
        return results
    
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
        
        detected_entities: List[str] = []
        max_level = SensitivityLevel.PUBLIC
        
        # 检测经过校验的身份证/银行卡号
        validated_creds = self._detect_validated_credentials(text)
        if validated_creds:
            for cred in validated_creds:
                detected_entities.append(f"CRITICAL: {_mask_sensitive(cred)}")
            max_level = SensitivityLevel.CRITICAL
        
        # 检测其他 CRITICAL 级别模式
        critical_matches = self._detect_patterns(text, self.critical_patterns)
        if critical_matches:
            for match in critical_matches:
                detected_entities.append(f"CRITICAL: {_mask_sensitive(match)}")
            max_level = SensitivityLevel.CRITICAL
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

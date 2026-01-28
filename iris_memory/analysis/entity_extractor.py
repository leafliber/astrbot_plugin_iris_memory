"""
实体提取器模块
从对话文本中提取结构化实体（人名、地点、时间、组织等）
支持中英文混合文本处理
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import hashlib


class EntityType(str, Enum):
    """实体类型枚举"""
    PERSON = "person"                  # 人名：小王、张三、Alice
    LOCATION = "location"              # 地点：北京、Shanghai、会议室
    TIME = "time"                      # 时间：明天、下周三、3pm
    DATE = "date"                      # 日期：2025-01-27、1月27号
    ORGANIZATION = "organization"      # 组织：阿里、腾讯、Google
    PHONE = "phone"                    # 电话号码
    EMAIL = "email"                    # 邮箱地址
    URL = "url"                        # URL链接
    MONEY = "money"                    # 金额：100元、$50
    QUANTITY = "quantity"              # 数量：3个、5kg
    UNKNOWN = "unknown"                # 未知类型


@dataclass
class Entity:
    """实体数据结构"""
    entity_type: EntityType          # 实体类型
    text: str                         # 实体原始文本
    start_pos: int                    # 起始位置
    end_pos: int                      # 结束位置
    confidence: float = 0.5            # 置信度 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外信息


class EntityExtractor:
    """实体提取器
    
    功能：
    1. 基于规则和正则表达式提取常见实体
    2. 支持中英文混合文本
    3. 提供实体标准化（时间、地点等）
    4. 计算实体置信度
    """
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """初始化实体提取器
        
        Args:
            reference_date: 参考日期（用于时间标准化），默认为当前时间
        """
        self.reference_date = reference_date or datetime.now()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译所有正则表达式模式"""
        # 中文人名模式（简单规则：2-4个汉字，常见姓氏）
        self.chinese_surnames = [
            '王', '李', '张', '刘', '陈', '杨', '黄', '赵', '吴', '周',
            '徐', '孙', '马', '朱', '胡', '郭', '何', '高', '林', '罗',
            '郑', '梁', '谢', '宋', '唐', '许', '韩', '冯', '邓', '曹'
        ]
        chinese_name_pattern = f"[{''.join(self.chinese_surnames)}][\\u4e00-\\u9fa5]{{1,2}}"

        # 中文昵称前缀模式：小/老/阿 + 1个汉字
        chinese_nickname_pattern = f"[{'小老阿'}][\\u4e00-\\u9fa5]"
        
        # 英文人名模式（首字母大写，可选多个单词）
        english_name_pattern = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"
        
        # 地点模式
        location_cn_pattern = r"(?:北京|上海|广州|深圳|杭州|成都|重庆|武汉|西安|南京|天津|苏州|长沙|青岛|大连|厦门|无锡|宁波|郑州|哈尔滨|济南|石家庄|长春|合肥|福州|昆明|沈阳|南昌|南宁|贵阳|兰州|太原|呼和浩特|乌鲁木齐|西宁|银川|海口|拉萨|清华)(?:大学|市|省|区|县|路|街|图书馆)?"
        location_en_pattern = r"(?:[A-Z][a-z]+\s+){1,2}(?:City|University|Hospital|Airport|Station|Park|Museum|Library|Street|Road|Avenue|Square|Center|Building)"
        
        # 时间模式（相对时间）
        time_relative_pattern = r"(?:今天|明天|后天|昨天|前天|今天|今早|今晚|下周|上周|这周|本周|下月|上月|本月|今年|明年|去年)(?:的)?(?:上午|下午|晚上|早上|中午|夜里|凌晨)?(?:\d{1,2}点|\d{1,2}:\d{2})?"
        time_en_pattern = r"(?:today|tomorrow|yesterday|this morning|this afternoon|this evening|tonight|tomorrow morning|tomorrow afternoon|tomorrow evening)(?: at \d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))?"
        
        # 日期模式（绝对日期）
        date_cn_pattern = r"\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日(?:周[一二三四五六日天])?|\d{4}-\d{1,2}-\d{1,2}"
        date_en_pattern = r"\d{4}-\d{1,2}-\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4}"
        
        # 组织模式
        org_cn_pattern = r"(?:阿里|腾讯|百度|字节|华为|小米|京东|美团|滴滴|网易|新浪|搜狐|360|OPPO|vivo|联想|海尔|格力|美的|比亚迪|长城|吉利)(?:公司|集团|科技|网络|信息|电子|汽车|控股)?"
        org_en_pattern = r"(?:Google|Microsoft|Apple|Amazon|Meta|Tesla|Netflix|Twitter|LinkedIn|Intel|AMD|NVIDIA|IBM|Oracle|SAP|Salesforce|Adobe|Tencent|Alibaba|Baidu|ByteDance|Huawei|Xiaomi|JD|Meituan|Didi|NetEase)(?: Inc\.?| Corp\.?| Ltd\.?| Group)?"
        
        # 电话号码模式
        phone_cn_pattern = r"1[3-9]\d{9}"
        phone_intl_pattern = r"\+\d{1,3}[-\s]?\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}"
        
        # 邮箱模式
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        
        # URL模式
        url_pattern = r"https?://[^\s，。！？!?;；:()\[\]]+"
        
        # 金额模式
        money_cn_pattern = r"\d+(?:\.\d+)?(?:元|块|万|亿|千|百)"
        money_en_pattern = r"\$\d+(?:,\d{3})*(?:\.\d{2})?|USD\s*\d+(?:,\d{3})*(?:\.\d{2})?"
        
        # 数量模式
        quantity_cn_pattern = r"\d+(?:\.\d+)?(?:个|只|条|件|本|台|次|遍|趟|公斤|kg|千克|克|g|升|升|L|毫升|ml|米|m|厘米|cm|毫米|mm)"
        quantity_en_pattern = r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:pieces|pcs|items|kg|g|L|ml|m|cm|mm|times|times)"
        
        # 编译所有模式
        self.patterns = {
            EntityType.PERSON: [
                re.compile(chinese_name_pattern),
                re.compile(chinese_nickname_pattern),
                re.compile(english_name_pattern)
            ],
            EntityType.LOCATION: [
                re.compile(location_cn_pattern),
                re.compile(location_en_pattern)
            ],
            EntityType.TIME: [
                re.compile(time_relative_pattern, re.IGNORECASE)
            ],
            EntityType.DATE: [
                re.compile(date_cn_pattern),
                re.compile(date_en_pattern, re.IGNORECASE)
            ],
            EntityType.ORGANIZATION: [
                re.compile(org_cn_pattern),
                re.compile(org_en_pattern, re.IGNORECASE)
            ],
            EntityType.PHONE: [
                re.compile(phone_cn_pattern),
                re.compile(phone_intl_pattern)
            ],
            EntityType.EMAIL: [
                re.compile(email_pattern)
            ],
            EntityType.URL: [
                re.compile(url_pattern)
            ],
            EntityType.MONEY: [
                re.compile(money_cn_pattern),
                re.compile(money_en_pattern, re.IGNORECASE)
            ],
            EntityType.QUANTITY: [
                re.compile(quantity_cn_pattern),
                re.compile(quantity_en_pattern, re.IGNORECASE)
            ]
        }
        
        # 置信度权重
        self.confidence_weights = {
            EntityType.EMAIL: 0.95,
            EntityType.URL: 0.95,
            EntityType.PHONE: 0.90,
            EntityType.MONEY: 0.85,
            EntityType.DATE: 0.80,
            EntityType.ORGANIZATION: 0.75,
            EntityType.LOCATION: 0.70,
            EntityType.TIME: 0.65,
            EntityType.QUANTITY: 0.60,
            EntityType.PERSON: 0.50,
            EntityType.UNKNOWN: 0.30
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """从文本中提取所有实体
        
        Args:
            text: 待分析的文本
            
        Returns:
            提取到的实体列表
        """
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity_text = match.group()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # 计算置信度
                    confidence = self.confidence_weights.get(entity_type, 0.5)
                    
                    # 创建实体
                    entity = Entity(
                        entity_type=entity_type,
                        text=entity_text,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence
                    )
                    
                    # 标准化实体
                    self._normalize_entity(entity)
                    
                    entities.append(entity)
        
        # 去重（基于位置）
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _normalize_entity(self, entity: Entity):
        """标准化实体信息
        
        Args:
            entity: 待标准化的实体（会原地修改）
        """
        if entity.entity_type == EntityType.TIME:
            entity.metadata['normalized_time'] = self._normalize_time(entity.text)
        
        elif entity.entity_type == EntityType.DATE:
            entity.metadata['normalized_date'] = self._normalize_date(entity.text)
        
        elif entity.entity_type == EntityType.LOCATION:
            entity.metadata['normalized_location'] = entity.text.strip()
        
        elif entity.entity_type == EntityType.MONEY:
            entity.metadata['normalized_money'] = self._normalize_money(entity.text)
        
        elif entity.entity_type == EntityType.QUANTITY:
            entity.metadata['normalized_quantity'] = self._normalize_quantity(entity.text)
        
        elif entity.entity_type == EntityType.PHONE:
            entity.metadata['normalized_phone'] = self._normalize_phone(entity.text)
        
        elif entity.entity_type == EntityType.EMAIL:
            entity.metadata['normalized_email'] = entity.text.lower().strip()
        
        elif entity.entity_type == EntityType.URL:
            entity.metadata['normalized_url'] = entity.text.strip()
    
    def _normalize_time(self, text: str) -> Optional[datetime]:
        """标准化时间实体
        
        Args:
            text: 时间文本
            
        Returns:
            标准化后的datetime对象，如果无法解析则返回None
        """
        text = text.lower()
        now = self.reference_date
        
        try:
            # 相对时间
            if '今天' in text or 'today' in text:
                base_date = now
            elif '明天' in text or 'tomorrow' in text:
                base_date = now + timedelta(days=1)
            elif '后天' in text:
                base_date = now + timedelta(days=2)
            elif '昨天' in text or 'yesterday' in text:
                base_date = now - timedelta(days=1)
            elif '前天' in text:
                base_date = now - timedelta(days=2)
            else:
                # 尝试解析下周/上周
                if '下周' in text or 'next week' in text:
                    base_date = now + timedelta(weeks=1)
                elif '上周' in text or 'last week' in text:
                    base_date = now - timedelta(weeks=1)
                elif '这周' in text or '本周' in text or 'this week' in text:
                    base_date = now
                else:
                    return None
            
            # 解析小时
            hour_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(?:am|pm|上午|下午|晚上|早上|中午|夜里|凌晨)?', text)
            if hour_match:
                hour = int(hour_match.group(1))
                minute = int(hour_match.group(2)) if hour_match.group(2) else 0
                
                # 处理12小时制
                if 'pm' in text or '下午' in text or '晚上' in text or '夜里' in text:
                    if hour != 12:
                        hour += 12
                elif 'am' in text or '上午' in text or '早上' in text or '凌晨' in text:
                    if hour == 12:
                        hour = 0
                
                return datetime(base_date.year, base_date.month, base_date.day, hour, minute)
            else:
                # 默认时间
                if '晚上' in text or 'evening' in text or '夜里' in text or 'tonight' in text:
                    return datetime(base_date.year, base_date.month, base_date.day, 20, 0)
                elif '下午' in text or 'afternoon' in text:
                    return datetime(base_date.year, base_date.month, base_date.day, 14, 0)
                elif '早上' in text or '今早' in text or 'morning' in text:
                    return datetime(base_date.year, base_date.month, base_date.day, 9, 0)
                elif '中午' in text:
                    return datetime(base_date.year, base_date.month, base_date.day, 12, 0)
                else:
                    return datetime(base_date.year, base_date.month, base_date.day, 10, 0)
        
        except Exception:
            return None
    
    def _normalize_date(self, text: str) -> Optional[datetime]:
        """标准化日期实体
        
        Args:
            text: 日期文本
            
        Returns:
            标准化后的datetime对象，如果无法解析则返回None
        """
        try:
            # 格式1：YYYY年MM月DD日
            match = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
            if match:
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            
            # 格式2：MM月DD日
            match = re.match(r'(\d{1,2})月(\d{1,2})日', text)
            if match:
                year = self.reference_date.year
                return datetime(year, int(match.group(1)), int(match.group(2)))
            
            # 格式3：YYYY-MM-DD
            match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', text)
            if match:
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            
            # 格式4：英文日期（Jan 27, 2025）
            match = re.match(r'(\w{3})\w*\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})', text, re.IGNORECASE)
            if match:
                months = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                month = months.get(match.group(1).lower())
                if month:
                    return datetime(int(match.group(3)), month, int(match.group(2)))
            
            return None
        
        except Exception:
            return None
    
    def _normalize_money(self, text: str) -> Optional[float]:
        """标准化金额实体
        
        Args:
            text: 金额文本
            
        Returns:
            标准化后的金额（单位：元），如果无法解析则返回None
        """
        try:
            # 中文金额
            match = re.match(r'([\d.]+)(?:元|块)', text)
            if match:
                return float(match.group(1))
            
            # 万、亿单位
            match = re.match(r'([\d.]+)(万|亿)', text)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if unit == '万':
                    return value * 10000
                elif unit == '亿':
                    return value * 100000000
            
            # 英文金额（$）
            match = re.match(r'\$([\d,]+(?:\.\d{2})?)', text)
            if match:
                value_str = match.group(1).replace(',', '')
                return float(value_str) * 7  # 简单汇率估算：1美元=7人民币
            
            # USD格式
            match = re.match(r'USD\s*([\d,]+(?:\.\d{2})?)', text)
            if match:
                value_str = match.group(1).replace(',', '')
                return float(value_str) * 7
            
            return None
        
        except Exception:
            return None
    
    def _normalize_quantity(self, text: str) -> Optional[Dict[str, Any]]:
        """标准化数量实体
        
        Args:
            text: 数量文本
            
        Returns:
            标准化后的数量信息 {'value': float, 'unit': str}
        """
        try:
            match = re.match(r'([\d.]+)\s*([个只条件本台次遍趟公斤kg千克克升升L毫升ml米m厘米cm毫米mmpiecespcsitems]+)', text, re.IGNORECASE)
            if match:
                return {
                    'value': float(match.group(1)),
                    'unit': match.group(2).lower()
                }
            return None
        except Exception:
            return None
    
    def _normalize_phone(self, text: str) -> Optional[str]:
        """标准化电话号码
        
        Args:
            text: 电话号码文本
            
        Returns:
            标准化后的电话号码
        """
        # 移除所有非数字字符
        cleaned = re.sub(r'[^\d+]', '', text)
        return cleaned if len(cleaned) >= 7 else None
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重实体（基于位置重叠）
        
        Args:
            entities: 原始实体列表
            
        Returns:
            去重后的实体列表
        """
        # 按起始位置排序
        entities.sort(key=lambda e: e.start_pos)
        
        unique_entities = []
        for entity in entities:
            # 检查是否与已有实体重叠
            is_duplicate = False
            for existing in unique_entities:
                # 如果位置有重叠
                if not (entity.end_pos <= existing.start_pos or entity.start_pos >= existing.end_pos):
                    # 保留置信度更高的
                    if entity.confidence > existing.confidence:
                        unique_entities.remove(existing)
                        unique_entities.append(entity)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities
    
    def get_entities_by_type(self, text: str, entity_type: EntityType) -> List[Entity]:
        """提取特定类型的实体
        
        Args:
            text: 待分析的文本
            entity_type: 实体类型
            
        Returns:
            提取到的实体列表
        """
        all_entities = self.extract_entities(text)
        return [e for e in all_entities if e.entity_type == entity_type]
    
    def get_entity_summary(self, text: str) -> Dict[str, List[Entity]]:
        """获取所有实体的摘要
        
        Args:
            text: 待分析的文本
            
        Returns:
            按类型分组的实体字典
        """
        entities = self.extract_entities(text)
        summary = {}
        
        for entity in entities:
            type_name = entity.entity_type.value
            if type_name not in summary:
                summary[type_name] = []
            summary[type_name].append(entity)
        
        return summary


# 便捷函数
def extract_entities(text: str, reference_date: Optional[datetime] = None) -> List[Entity]:
    """便捷函数：从文本中提取所有实体
    
    Args:
        text: 待分析的文本
        reference_date: 参考日期
        
    Returns:
        提取到的实体列表
    """
    extractor = EntityExtractor(reference_date)
    return extractor.extract_entities(text)


def get_entity_summary(text: str) -> Dict[str, List[Entity]]:
    """便捷函数：获取实体摘要
    
    Args:
        text: 待分析的文本
        
    Returns:
        按类型分组的实体字典
    """
    extractor = EntityExtractor()
    return extractor.get_entity_summary(text)

"""
画像提取结果与关键词配置管理

ExtractionResult: 画像提取结果数据类
KeywordMaps: 外置关键词配置管理
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from iris_memory.utils.logger import get_logger

logger = get_logger("persona_extractor")


# ---------------------------------------------------------------------------
# 提取结果
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """画像提取结果"""
    interests: Dict[str, float] = field(default_factory=dict)
    social_style: Optional[str] = None
    reply_style_preference: Optional[str] = None
    formality_adjustment: float = 0.0
    topic_blacklist: List[str] = field(default_factory=list)
    work_info: Optional[str] = None
    life_info: Optional[str] = None
    trust_delta: float = 0.0
    intimacy_delta: float = 0.0
    confidence: float = 0.0
    source: str = "rule"  # "rule" | "llm" | "hybrid"


# ---------------------------------------------------------------------------
# LLM 提取 Prompt
# ---------------------------------------------------------------------------
PERSONA_EXTRACTION_PROMPT = """分析以下用户消息，提取用户的画像特征。

重点关注：
1. 兴趣爱好（如有提及）
2. 社交风格倾向
3. 沟通偏好（简洁/详细、正式/随意）
4. 话题偏好或排斥

用户消息：
{content}

请以JSON格式返回（仅包含能从消息中提取到的信息，无法判断的字段设为null）：
{{
  "interests": {{"兴趣名": 0.0到1.0的置信度}},
  "social_style": "外向|内向|温和|null",
  "reply_preference": "brief|detailed|null",
  "formality": -1.0到1.0的变化量或null,
  "topic_blacklist": ["排斥话题"],
  "work_info": "工作相关信息摘要或null",
  "life_info": "生活相关信息摘要或null",
  "confidence": 0.0到1.0
}}"""


# ---------------------------------------------------------------------------
# 关键词加载
# ---------------------------------------------------------------------------
class KeywordMaps:
    """外置关键词配置管理"""

    def __init__(self, yaml_path: Optional[Path] = None):
        self._yaml_path = yaml_path
        self._data: Dict[str, Any] = {}
        self._loaded = False

    def load(self) -> None:
        """加载关键词配置"""
        if self._yaml_path and self._yaml_path.exists():
            try:
                with open(self._yaml_path, "r", encoding="utf-8") as f:
                    self._data = yaml.safe_load(f) or {}
                self._loaded = True
                logger.info(f"Keyword maps loaded from {self._yaml_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load keyword maps from {self._yaml_path}: {e}")

        # 内置默认值（兜底）
        self._data = self._builtin_defaults()
        self._loaded = True
        logger.info("Using builtin default keyword maps")

    @staticmethod
    def _builtin_defaults() -> Dict[str, Any]:
        """内置默认关键词（与旧硬编码一致）"""
        return {
            "interests": {
                "编程": ["编程", "代码", "开发", "程序"],
                "阅读": ["阅读", "读书", "看书", "书"],
                "运动": ["运动", "跑步", "健身", "锻炼"],
                "音乐": ["音乐", "歌", "唱"],
                "游戏": ["游戏", "打游戏", "玩游戏"],
                "美食": ["吃", "美食", "餐厅", "做饭"],
                "旅行": ["旅行", "旅游", "出游"],
            },
            "social_styles": {
                "外向": ["外向", "活泼", "爱交际"],
                "内向": ["内向", "安静", "独处"],
                "温和": ["温和", "和善", "温柔"],
            },
            "work_keywords": ["工作", "公司", "项目", "同事", "老板", "职业", "事业", "上班"],
            "life_keywords": ["喜欢", "爱好", "兴趣", "习惯", "运动", "娱乐", "爱吃", "讨厌"],
            "reply_style": {
                "brief": ["简短", "简洁", "不要太多"],
                "detailed": ["详细", "具体", "展开说"],
            },
            "formality": {
                "formal": ["正式", "礼貌", "敬语"],
                "casual": ["随意", "口语", "不用客气"],
            },
            "trust_keywords": ["信任"],
            "intimacy_keywords": ["亲密"],
        }

    # -- 便捷访问 --
    @property
    def interests(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("interests", {})

    @property
    def social_styles(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("social_styles", {})

    @property
    def work_keywords(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._data.get("work_keywords", [])

    @property
    def life_keywords(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._data.get("life_keywords", [])

    @property
    def reply_style(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("reply_style", {})

    @property
    def formality(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("formality", {})

    @property
    def trust_keywords(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._data.get("trust_keywords", [])

    @property
    def intimacy_keywords(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._data.get("intimacy_keywords", [])

    def reload(self) -> None:
        """热重载关键词配置"""
        self._loaded = False
        self.load()

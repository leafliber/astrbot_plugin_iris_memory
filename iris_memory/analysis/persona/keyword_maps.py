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

    # ── v2 新增维度 ──

    # 工作维度
    work_style: Optional[str] = None           # 工作风格：远程/坐班/自由
    work_challenge: Optional[str] = None       # 工作挑战（追加到 work_challenges 列表）
    work_preferences: Dict[str, Any] = field(default_factory=dict)  # 工作偏好

    # 生活维度
    lifestyle: Optional[str] = None            # 生活方式：夜猫子/早起/宅
    life_preferences: Dict[str, Any] = field(default_factory=dict)  # 生活偏好

    # 情感维度
    emotional_triggers: List[str] = field(default_factory=list)     # 情感触发器
    emotional_soothers: Dict[str, Any] = field(default_factory=dict)  # 情感安慰物

    # 社交维度
    social_boundaries: Dict[str, Any] = field(default_factory=dict)  # 社交边界

    # 人格维度（Big Five 增量，范围 -0.1~+0.1）
    personality_openness_delta: float = 0.0
    personality_conscientiousness_delta: float = 0.0
    personality_extraversion_delta: float = 0.0
    personality_agreeableness_delta: float = 0.0
    personality_neuroticism_delta: float = 0.0

    # 沟通维度
    directness_adjustment: float = 0.0    # 沟通直接度调整量
    humor_adjustment: float = 0.0         # 幽默度调整量
    empathy_adjustment: float = 0.0       # 共情度调整量

    # 交互偏好
    proactive_reply_delta: float = 0.0    # 主动回复偏好调整量


# ---------------------------------------------------------------------------
# LLM 提取 Prompt
# ---------------------------------------------------------------------------
PERSONA_EXTRACTION_PROMPT = """分析以下用户消息，提取用户的画像特征。

重点关注：
1. 兴趣爱好（如有提及）
2. 社交风格倾向
3. 沟通偏好（简洁/详细、正式/随意、直接/委婉、幽默程度）
4. 话题偏好或排斥
5. 工作风格与挑战
6. 生活方式
7. 情感触发因素与安慰方式
8. 社交边界
9. 人格特质倾向（开放性、尽责性、外向性、亲和性、神经质）
10. 对被主动联系的态度

用户消息：
{content}

请以JSON格式返回（仅包含能从消息中提取到的信息，无法判断的字段设为null或0）：
{{
  "interests": {{"兴趣名": 0.0到1.0的置信度}},
  "social_style": "外向|内向|温和|null",
  "reply_preference": "brief|detailed|null",
  "formality": -1.0到1.0的变化量或null,
  "directness": -1.0到1.0的变化量或null,
  "humor": -1.0到1.0的变化量或null,
  "empathy": -1.0到1.0的变化量或null,
  "topic_blacklist": ["排斥话题"],
  "work_info": "工作相关信息摘要或null",
  "work_style": "远程|坐班|自由|null",
  "work_challenge": "工作挑战描述或null",
  "life_info": "生活相关信息摘要或null",
  "lifestyle": "夜猫子|早起|宅|null",
  "emotional_triggers": ["引起负面情绪的事物"],
  "emotional_soothers": {{"安慰方式": "描述"}},
  "social_boundaries": {{"边界类型": "描述"}},
  "personality": {{
    "openness": -0.1到0.1的调整量或0,
    "conscientiousness": -0.1到0.1的调整量或0,
    "extraversion": -0.1到0.1的调整量或0,
    "agreeableness": -0.1到0.1的调整量或0,
    "neuroticism": -0.1到0.1的调整量或0
  }},
  "proactive_reply": -0.1到0.1的调整量或0,
  "confidence": 0.0到1.0
}}"""


# ---------------------------------------------------------------------------
# 关键词加载
# ---------------------------------------------------------------------------
class KeywordMaps:
    """外置关键词配置管理"""

    def __init__(self, yaml_path: Optional[Path] = None):
        if yaml_path is not None:
            self._yaml_path = yaml_path
        else:
            default_yaml = Path(__file__).resolve().parent / "keyword_maps.yaml"
            self._yaml_path = default_yaml if default_yaml.exists() else None
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
            # ── v2 新增 ──
            "work_styles": {
                "远程": ["远程", "在家办公", "remote"],
                "坐班": ["坐班", "朝九晚五", "打卡"],
                "自由": ["自由职业", "弹性", "灵活"],
            },
            "work_challenge_keywords": ["压力", "加班", "困难", "挑战", "焦虑", "紧急"],
            "lifestyles": {
                "夜猫子": ["夜猫子", "熬夜", "晚睡"],
                "早起": ["早起", "早睡早起", "晨跑"],
                "宅": ["宅", "宅家", "不出门"],
            },
            "emotional_trigger_keywords": ["怕", "害怕", "讨厌", "受不了", "崩溃", "最烦"],
            "emotional_soother_keywords": ["放松", "治愈", "安慰", "解压"],
            "social_boundary_keywords": ["别聊", "不想说", "不讨论", "别问", "不说"],
            "directness": {
                "direct": ["直说", "别绕弯", "说重点", "直接"],
                "indirect": ["委婉", "含蓄", "暗示"],
            },
            "humor": {
                "high": ["哈哈", "233", "笑死", "段子", "幽默", "lol", "搞笑"],
                "low": ["严肃", "认真", "正经"],
            },
            "empathy": {
                "high": ["理解", "共情", "体谅", "安慰", "换位思考"],
                "low": ["冷漠", "无所谓", "别矫情", "不关心"],
            },
            "proactive_preference": {
                "welcome": ["多聊聊", "常来", "找我聊", "欢迎"],
                "unwanted": ["别打扰", "别找我", "不用管我", "少说话"],
            },
            "personality": {
                "openness": {
                    "high": ["新鲜", "创意", "创新", "尝试", "探索", "好奇"],
                    "low": ["传统", "保守", "不变", "墨守成规"],
                },
                "conscientiousness": {
                    "high": ["计划", "规律", "条理", "准时", "认真", "仔细"],
                    "low": ["随性", "拖延", "随便", "懒"],
                },
                "extraversion": {
                    "high": ["外向", "社交", "聚会", "热闹", "活泼"],
                    "low": ["内向", "独处", "安静", "一个人"],
                },
                "agreeableness": {
                    "high": ["温和", "随和", "配合", "善良", "体贴"],
                    "low": ["坚持", "固执", "不妥协", "竞争"],
                },
                "neuroticism": {
                    "high": ["紧张", "焦虑", "担心", "多虑", "敏感"],
                    "low": ["淡定", "冷静", "平和", "稳重"],
                },
            },
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

    # ── v2 新增属性 ──

    @property
    def work_styles(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("work_styles", {})

    @property
    def work_challenge_keywords(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._data.get("work_challenge_keywords", [])

    @property
    def lifestyles(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("lifestyles", {})

    @property
    def emotional_trigger_keywords(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._data.get("emotional_trigger_keywords", [])

    @property
    def emotional_soother_keywords(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._data.get("emotional_soother_keywords", [])

    @property
    def social_boundary_keywords(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._data.get("social_boundary_keywords", [])

    @property
    def directness(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("directness", {})

    @property
    def humor(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("humor", {})

    @property
    def empathy(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("empathy", {})

    @property
    def proactive_preference(self) -> Dict[str, List[str]]:
        if not self._loaded:
            self.load()
        return self._data.get("proactive_preference", {})

    @property
    def personality(self) -> Dict[str, Dict[str, List[str]]]:
        if not self._loaded:
            self.load()
        return self._data.get("personality", {})

    def reload(self) -> None:
        """热重载关键词配置"""
        self._loaded = False
        self.load()

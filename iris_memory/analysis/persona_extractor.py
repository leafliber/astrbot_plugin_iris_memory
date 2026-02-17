"""
画像提取器 - 统一入口模块

支持三种提取模式：
- rule: 纯规则匹配（快速，零成本，覆盖有限）
- llm: 纯 LLM 提取（准确，有 token 成本）
- hybrid: 混合模式（规则优先 + LLM 补充）

关键词配置外置到 data/keyword_maps.yaml，支持用户自定义扩展。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from iris_memory.utils.logger import get_logger
from iris_memory.utils.provider_utils import (
    extract_provider_id,
    get_default_provider,
    get_provider_by_id,
    normalize_provider_id,
)

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


# ---------------------------------------------------------------------------
# 规则提取器
# ---------------------------------------------------------------------------
class RuleBasedExtractor:
    """基于关键词规则的画像提取"""

    def __init__(self, keyword_maps: KeywordMaps):
        self._kw = keyword_maps

    def extract(self, content: str, summary: Optional[str] = None) -> ExtractionResult:
        """从文本中基于关键词提取画像信息"""
        result = ExtractionResult(source="rule")
        content_lower = content.lower()
        text = content + (summary or "")

        # 兴趣提取
        for interest, keywords in self._kw.interests.items():
            if any(kw in content_lower for kw in keywords):
                result.interests[interest] = 0.1  # 权重增量

        # 社交风格
        for style, keywords in self._kw.social_styles.items():
            if any(kw in text for kw in keywords):
                result.social_style = style
                break

        # 回复风格偏好
        for style, keywords in self._kw.reply_style.items():
            if any(kw in content_lower for kw in keywords):
                result.reply_style_preference = style
                break

        # 沟通正式度
        for direction, keywords in self._kw.formality.items():
            if any(kw in content_lower for kw in keywords):
                result.formality_adjustment = 0.1 if direction == "formal" else -0.1
                break

        # 工作/生活维度
        if any(kw in content_lower for kw in self._kw.work_keywords) and summary:
            result.work_info = summary
        if any(kw in content_lower for kw in self._kw.life_keywords) and summary:
            result.life_info = summary

        # 信任 & 亲密度
        if any(kw in text for kw in self._kw.trust_keywords):
            result.trust_delta = 0.1
        if any(kw in text for kw in self._kw.intimacy_keywords):
            result.intimacy_delta = 0.1

        # 计算置信度（匹配的维度数越多 -> 越高）
        hit_count = sum([
            bool(result.interests),
            result.social_style is not None,
            result.reply_style_preference is not None,
            result.formality_adjustment != 0.0,
            result.work_info is not None,
            result.life_info is not None,
            result.trust_delta > 0,
            result.intimacy_delta > 0,
        ])
        result.confidence = min(1.0, hit_count * 0.2) if hit_count else 0.0
        return result


# ---------------------------------------------------------------------------
# LLM 提取器
# ---------------------------------------------------------------------------
class LLMExtractor:
    """基于 LLM 的画像提取"""

    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        max_tokens: int = 300,
        daily_limit: int = 50,
    ):
        self._astrbot_context = astrbot_context
        self._provider_id = normalize_provider_id(provider_id)  # "default" 或具体 provider_id
        self._max_tokens = max_tokens
        self._daily_limit = daily_limit

        # 每日调用计数
        self._call_date: Optional[date] = None
        self._call_count: int = 0

        # 缓存 provider
        self._resolved_provider = None
        self._resolved_provider_id: Optional[str] = None

    def _reset_daily_counter(self) -> None:
        """日期翻转时重置计数器"""
        today = date.today()
        if self._call_date != today:
            self._call_date = today
            self._call_count = 0

    def _is_within_limit(self) -> bool:
        """检查是否在每日限制内"""
        self._reset_daily_counter()
        return self._call_count < self._daily_limit

    async def _resolve_provider(self) -> bool:
        """解析 LLM 提供者

        支持：
        - provider_id == "default" 或 None → 使用 AstrBot 默认提供者
        - provider_id == 具体 ID → 查找并使用该提供者
        """
        if self._resolved_provider is not None:
            return True
        if not self._astrbot_context:
            return False

        try:
            # 指定了具体 provider_id → 尝试匹配
            if self._provider_id and self._provider_id != "default":
                provider, resolved_id = get_provider_by_id(self._astrbot_context, self._provider_id)
                if provider:
                    self._resolved_provider = provider
                    self._resolved_provider_id = resolved_id
                    logger.info(f"Persona LLM provider resolved: {resolved_id}")
                    return True
                logger.warning(
                    f"Persona LLM provider '{self._provider_id}' not found, "
                    f"falling back to default"
                )

            # 默认提供者
            provider, provider_id = get_default_provider(self._astrbot_context)
            if provider:
                self._resolved_provider = provider
                self._resolved_provider_id = provider_id or extract_provider_id(provider)
                logger.info(f"Persona LLM provider (default): {self._resolved_provider_id}")
                return True
        except Exception as e:
            logger.debug(f"Failed to resolve persona LLM provider: {e}")
        return False

    async def extract(
        self,
        content: str,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """使用 LLM 从文本中提取画像信息"""
        result = ExtractionResult(source="llm")

        if not self._is_within_limit():
            logger.debug("Persona LLM daily limit reached, skipping")
            return result

        if not await self._resolve_provider():
            logger.debug("Persona LLM provider not available")
            return result

        try:
            prompt = PERSONA_EXTRACTION_PROMPT.format(content=content[:1000])
            response = await self._call_llm(prompt)
            if not response:
                return result

            self._call_count += 1
            parsed = self._parse_response(response)
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"Persona LLM extraction failed: {e}")

        return result

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM"""
        ctx = self._astrbot_context
        pid = self._resolved_provider_id

        # 优先使用 llm_generate
        if ctx and hasattr(ctx, "llm_generate") and pid:
            try:
                resp = await ctx.llm_generate(
                    chat_provider_id=pid,
                    prompt=prompt,
                )
                if resp and hasattr(resp, "completion_text"):
                    return (resp.completion_text or "").strip()
            except Exception as e:
                logger.debug(f"llm_generate failed for persona: {e}")

        # 回退: text_chat
        provider = self._resolved_provider
        if provider and hasattr(provider, "text_chat"):
            try:
                resp = await provider.text_chat(prompt=prompt, context=[])
                if hasattr(resp, "completion_text"):
                    return (resp.completion_text or "").strip()
                if isinstance(resp, dict):
                    return (resp.get("text", "") or resp.get("content", "")).strip()
                return str(resp).strip() if resp else None
            except Exception as e:
                logger.debug(f"text_chat failed for persona: {e}")

        return None

    @staticmethod
    def _parse_response(response: str) -> Optional[ExtractionResult]:
        """解析 LLM JSON 响应为 ExtractionResult"""
        import re

        raw: Optional[Dict] = None
        # 直接 JSON
        try:
            raw = json.loads(response)
        except json.JSONDecodeError:
            pass

        if raw is None:
            # code-fenced JSON
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if m:
                try:
                    raw = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if raw is None:
            # 裸 {...}
            m = re.search(r"(\{[\s\S]*\})", response)
            if m:
                try:
                    raw = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if not raw or not isinstance(raw, dict):
            return None

        result = ExtractionResult(source="llm")

        # interests
        interests = raw.get("interests")
        if isinstance(interests, dict):
            result.interests = {
                k: max(0.0, min(1.0, float(v)))
                for k, v in interests.items()
                if isinstance(v, (int, float))
            }

        # social_style
        ss = raw.get("social_style")
        if ss and isinstance(ss, str) and ss.lower() != "null":
            result.social_style = ss

        # reply_preference
        rp = raw.get("reply_preference")
        if rp and isinstance(rp, str) and rp.lower() in ("brief", "detailed"):
            result.reply_style_preference = rp.lower()

        # formality
        fm = raw.get("formality")
        if fm is not None and isinstance(fm, (int, float)):
            result.formality_adjustment = max(-1.0, min(1.0, float(fm)))

        # topic_blacklist
        tb = raw.get("topic_blacklist")
        if isinstance(tb, list):
            result.topic_blacklist = [str(t) for t in tb if t]

        # work / life info
        wi = raw.get("work_info")
        if wi and isinstance(wi, str) and wi.lower() != "null":
            result.work_info = wi
        li = raw.get("life_info")
        if li and isinstance(li, str) and li.lower() != "null":
            result.life_info = li

        # confidence
        conf = raw.get("confidence")
        if conf is not None and isinstance(conf, (int, float)):
            result.confidence = max(0.0, min(1.0, float(conf)))

        return result

    @property
    def remaining_daily_calls(self) -> int:
        self._reset_daily_counter()
        return max(0, self._daily_limit - self._call_count)


# ---------------------------------------------------------------------------
# 统一提取器入口
# ---------------------------------------------------------------------------
class PersonaExtractor:
    """画像提取器 - 统一入口

    根据配置的 extraction_mode 选择提取策略：
    - rule:   纯规则  （快速，零成本）
    - llm:    纯 LLM  （准确但消耗 token）
    - hybrid: 混合    （规则优先 + LLM 补充）
    """

    def __init__(
        self,
        extraction_mode: str = "rule",
        keyword_maps: Optional[KeywordMaps] = None,
        astrbot_context=None,
        llm_provider_id: Optional[str] = None,
        llm_max_tokens: int = 300,
        llm_daily_limit: int = 50,
        enable_interest: bool = True,
        enable_style: bool = True,
        enable_preference: bool = True,
        fallback_to_rule: bool = True,
    ):
        self._mode = extraction_mode  # "rule" | "llm" | "hybrid"
        self._fallback_to_rule = fallback_to_rule
        self._enable_interest = enable_interest
        self._enable_style = enable_style
        self._enable_preference = enable_preference

        # 规则提取器（所有模式均可用于兜底）
        self._kw = keyword_maps or KeywordMaps()
        self._rule_extractor = RuleBasedExtractor(self._kw)

        # LLM 提取器
        self._llm_extractor: Optional[LLMExtractor] = None
        if extraction_mode in ("llm", "hybrid") and astrbot_context:
            self._llm_extractor = LLMExtractor(
                astrbot_context=astrbot_context,
                provider_id=llm_provider_id,
                max_tokens=llm_max_tokens,
                daily_limit=llm_daily_limit,
            )

    # -- 公共接口 --

    async def extract(
        self,
        content: str,
        summary: Optional[str] = None,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """从内容中提取画像信息"""
        if self._mode == "rule":
            return self._rule_extract(content, summary)
        elif self._mode == "llm":
            return await self._llm_extract(content, memory_context)
        elif self._mode == "hybrid":
            return await self._hybrid_extract(content, summary, memory_context)
        else:
            logger.warning(f"Unknown extraction mode '{self._mode}', falling back to rule")
            return self._rule_extract(content, summary)

    # -- 策略实现 --

    def _rule_extract(self, content: str, summary: Optional[str] = None) -> ExtractionResult:
        """纯规则提取"""
        result = self._rule_extractor.extract(content, summary)
        return self._apply_feature_gates(result)

    async def _llm_extract(
        self,
        content: str,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """纯 LLM 提取（失败时可回退到规则）"""
        if self._llm_extractor:
            result = await self._llm_extractor.extract(content, memory_context)
            if result.confidence > 0:
                return self._apply_feature_gates(result)

        # 回退
        if self._fallback_to_rule:
            logger.debug("LLM extraction failed/empty, falling back to rule")
            return self._rule_extract(content)

        return ExtractionResult(source="llm")

    async def _hybrid_extract(
        self,
        content: str,
        summary: Optional[str] = None,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """混合提取：先规则，再 LLM 补充"""
        rule_result = self._rule_extractor.extract(content, summary)

        # 如果规则已经提取到足够信息（置信度 >= 0.6），不调用 LLM
        if rule_result.confidence >= 0.6:
            rule_result.source = "hybrid"
            return self._apply_feature_gates(rule_result)

        # LLM 补充
        if self._llm_extractor:
            llm_result = await self._llm_extractor.extract(content, memory_context)
            if llm_result.confidence > 0:
                merged = self._merge_results(rule_result, llm_result)
                return self._apply_feature_gates(merged)

        rule_result.source = "hybrid"
        return self._apply_feature_gates(rule_result)

    # -- 工具方法 --

    @staticmethod
    def _merge_results(
        rule: ExtractionResult,
        llm: ExtractionResult,
    ) -> ExtractionResult:
        """合并规则和 LLM 提取结果（LLM 优先，规则补充）"""
        merged = ExtractionResult(source="hybrid")

        # 兴趣：合并，LLM 权重更高
        merged.interests = dict(rule.interests)
        for k, v in llm.interests.items():
            merged.interests[k] = max(merged.interests.get(k, 0.0), v)

        # 社交风格：LLM 优先
        merged.social_style = llm.social_style or rule.social_style
        # 回复偏好：LLM 优先
        merged.reply_style_preference = llm.reply_style_preference or rule.reply_style_preference
        # 正式度：LLM 优先（非零时）
        merged.formality_adjustment = (
            llm.formality_adjustment if llm.formality_adjustment != 0.0
            else rule.formality_adjustment
        )
        # 黑名单：合并去重
        merged.topic_blacklist = list(set(rule.topic_blacklist + llm.topic_blacklist))
        # 工作/生活：LLM 优先
        merged.work_info = llm.work_info or rule.work_info
        merged.life_info = llm.life_info or rule.life_info
        # 信任/亲密度
        merged.trust_delta = max(rule.trust_delta, llm.trust_delta) if llm.trust_delta else rule.trust_delta
        merged.intimacy_delta = max(rule.intimacy_delta, llm.intimacy_delta) if llm.intimacy_delta else rule.intimacy_delta
        # 置信度取最大
        merged.confidence = max(rule.confidence, llm.confidence)

        return merged

    def _apply_feature_gates(self, result: ExtractionResult) -> ExtractionResult:
        """根据功能开关过滤结果"""
        if not self._enable_interest:
            result.interests = {}
        if not self._enable_style:
            result.social_style = None
        if not self._enable_preference:
            result.reply_style_preference = None
            result.formality_adjustment = 0.0
        return result

    def reload_keywords(self) -> None:
        """热重载关键词配置"""
        self._kw.reload()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def llm_remaining_calls(self) -> int:
        if self._llm_extractor:
            return self._llm_extractor.remaining_daily_calls
        return 0

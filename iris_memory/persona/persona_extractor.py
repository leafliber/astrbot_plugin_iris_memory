"""
画像提取器 - 统一入口模块

支持三种提取模式：
- rule: 纯规则匹配（快速，零成本，覆盖有限）
- llm: 纯 LLM 提取（准确，有 token 成本）
- hybrid: 混合模式（规则优先 + LLM 补充）

关键词配置默认位于 iris_memory/analysis/persona/keyword_maps.yaml，支持外部覆盖与自定义扩展。
"""

from typing import Dict, Any, Optional
import re

from iris_memory.utils.logger import get_logger
from iris_memory.analysis.persona.keyword_maps import ExtractionResult, KeywordMaps
from iris_memory.analysis.persona.rule_extractor import RuleExtractor
from iris_memory.analysis.persona.llm_extractor import LLMExtractor

logger = get_logger("persona_extractor")


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
        self._rule_extractor = RuleExtractor(self._kw)

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

        # 动态阈值：根据内容价值信号调整
        threshold = self._compute_dynamic_threshold(content, rule_result)

        # 如果规则已经提取到足够信息，不调用 LLM
        if rule_result.confidence >= threshold:
            rule_result.source = "hybrid"
            logger.debug(
                f"Hybrid skip LLM: rule confidence {rule_result.confidence:.2f} "
                f">= threshold {threshold:.2f}"
            )
            return self._apply_feature_gates(rule_result)

        # LLM 补充
        if self._llm_extractor:
            llm_result = await self._llm_extractor.extract(content, memory_context)
            if llm_result.confidence > 0:
                merged = self._merge_results(rule_result, llm_result)
                logger.debug(
                    f"Hybrid LLM triggered: threshold={threshold:.2f}, "
                    f"rule_conf={rule_result.confidence:.2f}, "
                    f"llm_conf={llm_result.confidence:.2f}"
                )
                return self._apply_feature_gates(merged)

        rule_result.source = "hybrid"
        return self._apply_feature_gates(rule_result)

    @staticmethod
    def _compute_dynamic_threshold(
        content: str,
        rule_result: ExtractionResult,
    ) -> float:
        """动态计算 LLM 触发阈值

        高价值内容降低阈值（更容易触发 LLM），
        低价值内容提高阈值（节省 LLM 调用）。

        基准阈值: 0.6
        调整范围: [0.4, 0.75]
        """
        base_threshold = 0.6
        value_signals = 0

        # 1. 包含第一人称（自述信息更有价值）
        first_person = ["我", "本人", "俺", "咱", "吃"]
        if any(p in content for p in first_person):
            value_signals += 1

        # 2. 包含具体数字、地名等信息
        if re.search(r'\d+', content):
            value_signals += 1

        # 3. 内容较长（更可能包含价值信息）
        if len(content) > 50:
            value_signals += 1

        # 4. 规则提取到了部分而非完整信息
        has_partial = (
            (rule_result.interests and len(rule_result.interests) == 1)
            or (rule_result.social_style and not rule_result.reply_style_preference)
            or (rule_result.work_info and not rule_result.life_info)
            or (rule_result.work_style and not rule_result.work_challenge)
            or (rule_result.directness_adjustment != 0.0 and rule_result.humor_adjustment == 0.0)
        )
        if has_partial:
            value_signals += 1

        # 5. 包含明确的属性描述词（但规则可能未完全捕捉）
        attribute_signals = [
            "喜欢", "讨厌", "擅长", "不喜欢", "对…感兴趣",
            "需要", "希望", "认为", "觉得",
        ]
        if any(s in content for s in attribute_signals):
            value_signals += 1

        # 根据价值信号调整阈值
        # 更多信号 → 更低阈值 → 更容易触发 LLM
        adjustment = min(value_signals * 0.05, 0.20)
        return max(0.4, min(0.75, base_threshold - adjustment))

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

        # ── v2 新增维度合并 ──

        # 工作维度
        merged.work_style = llm.work_style or rule.work_style
        merged.work_challenge = llm.work_challenge or rule.work_challenge
        merged.work_preferences = {**rule.work_preferences, **llm.work_preferences}

        # 生活维度
        merged.lifestyle = llm.lifestyle or rule.lifestyle
        merged.life_preferences = {**rule.life_preferences, **llm.life_preferences}

        # 情感维度
        merged.emotional_triggers = list(set(rule.emotional_triggers + llm.emotional_triggers))
        merged.emotional_soothers = {**rule.emotional_soothers, **llm.emotional_soothers}

        # 社交边界
        merged.social_boundaries = {**rule.social_boundaries, **llm.social_boundaries}

        # 人格特质（LLM 优先，非零时）
        for trait in ("openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"):
            attr = f"personality_{trait}_delta"
            llm_val = getattr(llm, attr, 0.0)
            rule_val = getattr(rule, attr, 0.0)
            setattr(merged, attr, llm_val if llm_val != 0.0 else rule_val)

        # 沟通维度（LLM 优先，非零时）
        merged.directness_adjustment = (
            llm.directness_adjustment if llm.directness_adjustment != 0.0
            else rule.directness_adjustment
        )
        merged.humor_adjustment = (
            llm.humor_adjustment if llm.humor_adjustment != 0.0
            else rule.humor_adjustment
        )
        merged.empathy_adjustment = (
            llm.empathy_adjustment if llm.empathy_adjustment != 0.0
            else rule.empathy_adjustment
        )

        # 主动回复偏好
        merged.proactive_reply_delta = (
            llm.proactive_reply_delta if llm.proactive_reply_delta != 0.0
            else rule.proactive_reply_delta
        )

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

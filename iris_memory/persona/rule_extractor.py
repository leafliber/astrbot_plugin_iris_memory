"""
规则提取器 - 基于关键词规则的画像提取
"""

import re
from typing import Optional

from iris_memory.analysis.persona.keyword_maps import ExtractionResult, KeywordMaps


class RuleExtractor:
    """基于关键词规则的画像提取"""

    def __init__(self, keyword_maps: KeywordMaps):
        self._kw = keyword_maps

    @staticmethod
    def _contains_any(text: str, keywords) -> bool:
        """安全关键词匹配：兼容 YAML 中的数字等非字符串标量。"""
        return any(str(kw).lower() in text for kw in keywords if kw is not None)

    def extract(self, content: str, summary: Optional[str] = None) -> ExtractionResult:
        """从文本中基于关键词提取画像信息"""
        result = ExtractionResult(source="rule")
        content_lower = content.lower()
        text = content + (summary or "")

        # 兴趣提取
        for interest, keywords in self._kw.interests.items():
            if self._contains_any(content_lower, keywords):
                result.interests[interest] = 0.1  # 权重增量

        # 社交风格
        for style, keywords in self._kw.social_styles.items():
            if self._contains_any(text.lower(), keywords):
                result.social_style = style
                break

        # 回复风格偏好
        for style, keywords in self._kw.reply_style.items():
            if self._contains_any(content_lower, keywords):
                result.reply_style_preference = style
                break

        # 沟通正式度
        for direction, keywords in self._kw.formality.items():
            if self._contains_any(content_lower, keywords):
                result.formality_adjustment = 0.1 if direction == "formal" else -0.1
                break

        # 工作/生活维度
        if self._contains_any(content_lower, self._kw.work_keywords) and summary:
            result.work_info = summary
        if self._contains_any(content_lower, self._kw.life_keywords) and summary:
            result.life_info = summary

        # 信任 & 亲密度
        if self._contains_any(text.lower(), self._kw.trust_keywords):
            result.trust_delta = 0.1
        if self._contains_any(text.lower(), self._kw.intimacy_keywords):
            result.intimacy_delta = 0.1

        # ── v2 新增维度提取 ──

        # 工作风格
        for style, keywords in self._kw.work_styles.items():
            if self._contains_any(content_lower, keywords):
                result.work_style = style
                break

        # 工作挑战
        if self._contains_any(content_lower, self._kw.work_challenge_keywords) and summary:
            result.work_challenge = summary

        # 生活方式
        for style, keywords in self._kw.lifestyles.items():
            if self._contains_any(content_lower, keywords):
                result.lifestyle = style
                break

        # 情感触发器：提取「怕/讨厌/受不了 + 内容」模式
        for kw in self._kw.emotional_trigger_keywords:
            if kw in content:
                # 尝试提取触发器上下文（关键词后最多20字）
                idx = content.index(kw)
                trigger_ctx = content[idx:idx + 25].strip()
                if trigger_ctx and trigger_ctx not in result.emotional_triggers:
                    result.emotional_triggers.append(trigger_ctx)

        # 情感安慰：提取「放松/治愈 + 内容」模式
        for kw in self._kw.emotional_soother_keywords:
            if kw in content:
                idx = content.index(kw)
                # 向前回溯尝试提取安慰物（如「听音乐能放松」→ 音乐: 放松）
                soother_ctx = content[max(0, idx - 15):idx + len(kw) + 10].strip()
                if soother_ctx:
                    result.emotional_soothers[kw] = soother_ctx

        # 社交边界：提取「别聊/不讨论 + 话题」模式
        for kw in self._kw.social_boundary_keywords:
            if kw in content:
                idx = content.index(kw)
                boundary_ctx = content[idx:idx + 20].strip()
                if boundary_ctx:
                    result.social_boundaries[kw] = boundary_ctx

        # 沟通直接度
        for direction, keywords in self._kw.directness.items():
            if self._contains_any(content_lower, keywords):
                result.directness_adjustment = 0.1 if direction == "direct" else -0.1
                break

        # 幽默度
        for level, keywords in self._kw.humor.items():
            if self._contains_any(content_lower, keywords):
                result.humor_adjustment = 0.1 if level == "high" else -0.1
                break

        # 共情度
        for level, keywords in self._kw.empathy.items():
            if self._contains_any(content_lower, keywords):
                result.empathy_adjustment = 0.1 if level == "high" else -0.1
                break

        # 主动回复偏好
        for pref, keywords in self._kw.proactive_preference.items():
            if self._contains_any(content_lower, keywords):
                result.proactive_reply_delta = 0.1 if pref == "welcome" else -0.1
                break

        # 人格特质（Big Five）
        personality_config = self._kw.personality
        for trait, directions in personality_config.items():
            high_kws = directions.get("high", [])
            low_kws = directions.get("low", [])
            delta = 0.0
            if self._contains_any(content_lower, high_kws):
                delta = 0.05
            elif self._contains_any(content_lower, low_kws):
                delta = -0.05
            if delta != 0.0:
                attr_name = f"personality_{trait}_delta"
                if hasattr(result, attr_name):
                    setattr(result, attr_name, delta)

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
            result.work_style is not None,
            result.work_challenge is not None,
            result.lifestyle is not None,
            bool(result.emotional_triggers),
            bool(result.emotional_soothers),
            bool(result.social_boundaries),
            result.directness_adjustment != 0.0,
            result.humor_adjustment != 0.0,
            result.empathy_adjustment != 0.0,
            result.proactive_reply_delta != 0.0,
            any(getattr(result, f"personality_{t}_delta", 0.0) != 0.0
                for t in ("openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism")),
        ])
        result.confidence = min(1.0, hit_count * 0.15) if hit_count else 0.0
        return result

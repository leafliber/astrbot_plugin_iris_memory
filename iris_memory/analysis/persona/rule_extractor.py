"""
规则提取器 - 基于关键词规则的画像提取
"""

from typing import Optional

from iris_memory.analysis.persona.keyword_maps import ExtractionResult, KeywordMaps


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

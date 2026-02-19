"""
分析模块 — 聚合情感分析、RIF评分、画像提取等分析能力

包含：EmotionAnalyzer, RIFScorer, PersonaExtractor
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
    from iris_memory.analysis.rif_scorer import RIFScorer
    from iris_memory.analysis.persona.persona_extractor import PersonaExtractor

logger = get_logger("module.analysis")


class AnalysisModule:
    """分析模块

    聚合情感分析器、RIF评分器、画像提取器三个分析组件。
    """

    def __init__(self) -> None:
        self._emotion_analyzer: Optional[EmotionAnalyzer] = None
        self._rif_scorer: Optional[RIFScorer] = None
        self._persona_extractor: Optional[PersonaExtractor] = None

    @property
    def emotion_analyzer(self) -> Optional[EmotionAnalyzer]:
        return self._emotion_analyzer

    @property
    def rif_scorer(self) -> Optional[RIFScorer]:
        return self._rif_scorer

    @property
    def persona_extractor(self) -> Optional[PersonaExtractor]:
        return self._persona_extractor

    def initialize(self, config: Any) -> None:
        """初始化核心分析组件（同步）"""
        from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
        from iris_memory.analysis.rif_scorer import RIFScorer

        self._emotion_analyzer = EmotionAnalyzer(config)
        self._rif_scorer = RIFScorer()
        logger.info("AnalysisModule core initialized (emotion + rif)")

    async def init_persona_extractor(
        self,
        cfg: Any,
        plugin_data_path: Path,
        context: Any,
    ) -> None:
        """初始化画像提取器"""
        from iris_memory.analysis.persona.persona_extractor import PersonaExtractor
        from iris_memory.analysis.persona.keyword_maps import KeywordMaps

        mode = cfg.persona_extraction_mode
        logger.info(f"Initializing persona extractor (mode={mode})")

        keyword_yaml = plugin_data_path.parent / "data" / "keyword_maps.yaml"
        if not keyword_yaml.exists():
            keyword_yaml = Path(__file__).resolve().parent.parent.parent / "data" / "keyword_maps.yaml"
        kw_maps = KeywordMaps(yaml_path=keyword_yaml if keyword_yaml.exists() else None)

        self._persona_extractor = PersonaExtractor(
            extraction_mode=mode,
            keyword_maps=kw_maps,
            astrbot_context=context if mode in ("llm", "hybrid") else None,
            llm_provider_id=cfg.persona_llm_provider,
            llm_max_tokens=cfg.persona_llm_max_tokens,
            llm_daily_limit=cfg.persona_llm_daily_limit,
            enable_interest=cfg.persona_enable_interest,
            enable_style=cfg.persona_enable_style,
            enable_preference=cfg.persona_enable_preference,
            fallback_to_rule=cfg.persona_fallback_to_rule,
        )
        logger.info(f"Persona extractor ready (mode={mode})")

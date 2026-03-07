"""
PersonaView - 画像注入视图生成 + Dict 兼容接口

从 user_persona.py 提取 to_injection_view 逻辑。
提供 build_injection_view() 纯函数，无需访问 self。
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from iris_memory.core.constants import DEFAULT_EMOTION

if TYPE_CHECKING:
    from iris_memory.models.user_persona import UserPersona


def build_injection_view(persona: "UserPersona") -> Dict[str, Any]:
    """生成精简的画像视图（供 prompt 注入使用，不含审计日志）

    情感字段通过委托访问器从 EmotionalState 读取（如果已绑定）。

    Args:
        persona: UserPersona 实例

    Returns:
        精简字典视图
    """
    view: Dict[str, Any] = {}

    emotional_baseline = persona.get_emotional_baseline()
    emotional_trajectory = persona.get_emotional_trajectory()
    emotional_volatility = persona.get_emotional_volatility()
    negative_ratio = persona.get_negative_ratio()

    if emotional_baseline != DEFAULT_EMOTION or emotional_trajectory:
        emotional: Dict[str, Any] = {
            "baseline": emotional_baseline,
            "trajectory": emotional_trajectory,
            "volatility": round(emotional_volatility, 2),
            "negative_ratio": round(negative_ratio, 3),
        }
        if persona.emotional_triggers:
            emotional["triggers"] = persona.emotional_triggers[:5]
        if persona.emotional_soothers:
            emotional["soothers"] = dict(list(persona.emotional_soothers.items())[:3])
        view["emotional"] = emotional

    # 兴趣 / 习惯
    if persona.interests:
        top = sorted(persona.interests.items(), key=lambda x: x[1], reverse=True)[:5]
        view["interests"] = {k: round(v, 2) for k, v in top}
    if persona.habits:
        view["habits"] = persona.habits[:10]

    # 工作维度
    work: Dict[str, Any] = {}
    if persona.work_style:
        work["style"] = persona.work_style
    if persona.work_goals:
        work["goals"] = persona.work_goals[:5]
    if persona.work_challenges:
        work["challenges"] = persona.work_challenges[:5]
    if persona.work_preferences:
        work["preferences"] = dict(list(persona.work_preferences.items())[:5])
    if work:
        view["work"] = work

    # 生活维度
    life: Dict[str, Any] = {}
    if persona.lifestyle:
        life["style"] = persona.lifestyle
    if persona.life_preferences:
        life["preferences"] = dict(list(persona.life_preferences.items())[:5])
    if life:
        view["life"] = life

    # 沟通偏好
    view["communication"] = {
        "formality": round(persona.communication_formality, 2),
        "directness": round(persona.communication_directness, 2),
        "humor": round(persona.communication_humor, 2),
        "empathy": round(persona.communication_empathy, 2),
    }

    # 人格特质（Big Five）— 仅展示偏离默认值(0.5)的维度
    personality: Dict[str, float] = {}
    for trait in ("openness", "conscientiousness", "extraversion",
                  "agreeableness", "neuroticism"):
        val = getattr(persona, f"personality_{trait}", 0.5)
        if abs(val - 0.5) > 0.05:
            personality[trait] = round(val, 2)
    if personality:
        view["personality"] = personality

    # 交互偏好
    view["preferences"] = {}
    if persona.preferred_reply_style:
        view["preferences"]["style"] = persona.preferred_reply_style
    view["preferences"]["proactive_reply"] = round(persona.proactive_reply_preference, 2)
    if persona.topic_blacklist:
        view["preferences"]["topic_blacklist"] = persona.topic_blacklist[:10]

    # 关系
    relationship: Dict[str, Any] = {
        "trust": round(persona.trust_level, 2),
        "intimacy": round(persona.intimacy_level, 2),
    }
    if persona.social_style:
        relationship["social_style"] = persona.social_style
    if persona.social_boundaries:
        relationship["boundaries"] = dict(list(persona.social_boundaries.items())[:5])
    view["relationship"] = relationship

    return view

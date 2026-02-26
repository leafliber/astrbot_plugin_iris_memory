"""用户画像和情感状态仓库实现

实现 PersonaRepository 和 EmotionRepository 接口。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("persona_repo")


class PersonaRepositoryImpl:
    """用户画像仓库实现"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    async def list_all(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页列出用户画像"""
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        result: Dict[str, Any] = {"items": [], "total": 0, "page": page, "page_size": page_size}

        try:
            personas = self._service._user_personas
            all_items: List[Dict[str, Any]] = []

            for uid, persona in personas.items():
                interests = getattr(persona, "interests", {}) or {}
                top_interests = dict(sorted(interests.items(), key=lambda x: x[1], reverse=True)[:5])

                emotional_baseline = getattr(persona, "emotional_baseline", "neutral") or "neutral"

                current_emotion = "neutral"
                try:
                    states = self._service._user_emotional_states
                    state = states.get(uid)
                    if state:
                        current_emotion = getattr(state, "current", {}).get("primary", "neutral")
                except Exception:
                    pass

                all_items.append({
                    "user_id": uid,
                    "display_name": getattr(persona, "display_name", None),
                    "update_count": getattr(persona, "update_count", 0),
                    "last_updated": persona.last_updated.isoformat() if hasattr(persona.last_updated, "isoformat") else str(persona.last_updated),
                    "interests": top_interests,
                    "all_interests_count": len(interests),
                    "trust_level": getattr(persona, "trust_level", 0.5),
                    "intimacy_level": getattr(persona, "intimacy_level", 0.5),
                    "emotional_baseline": emotional_baseline,
                    "current_emotion": current_emotion,
                    "work_style": getattr(persona, "work_style", None),
                    "lifestyle": getattr(persona, "lifestyle", None),
                    "work_goals": list(getattr(persona, "work_goals", [])[:3]),
                    "habits": list(getattr(persona, "habits", [])[:3]),
                    "preferred_reply_style": getattr(persona, "preferred_reply_style", None),
                    "proactive_reply_preference": getattr(persona, "proactive_reply_preference", 0.5),
                    "personality_openness": getattr(persona, "personality_openness", 0.5),
                    "personality_conscientiousness": getattr(persona, "personality_conscientiousness", 0.5),
                    "personality_extraversion": getattr(persona, "personality_extraversion", 0.5),
                    "personality_agreeableness": getattr(persona, "personality_agreeableness", 0.5),
                    "personality_neuroticism": getattr(persona, "personality_neuroticism", 0.5),
                })

            all_items.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            result["total"] = len(all_items)
            start = (page - 1) * page_size
            result["items"] = all_items[start:start + page_size]

        except Exception as e:
            logger.warning(f"List personas error: {e}")

        return result

    async def get_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据用户 ID 获取画像"""
        try:
            personas = self._service._user_personas
            persona = personas.get(user_id)
            if not persona:
                return None
            return persona.to_dict()
        except Exception as e:
            logger.warning(f"Get persona by user_id error: {e}")
            return None

    async def search_personas(
        self,
        query: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """搜索用户画像

        支持按用户ID、兴趣、工作风格、生活方式等字段搜索。

        Args:
            query: 搜索关键词
            page: 页码
            page_size: 每页数量

        Returns:
            {items: [...], total: N, page: N, page_size: N}
        """
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        result: Dict[str, Any] = {"items": [], "total": 0, "page": page, "page_size": page_size}

        try:
            personas = self._service._user_personas
            all_items: List[Dict[str, Any]] = []

            query_lower = query.lower().strip() if query else ""

            for uid, persona in personas.items():
                interests = getattr(persona, "interests", {}) or {}
                top_interests = dict(sorted(interests.items(), key=lambda x: x[1], reverse=True)[:5])

                emotional_baseline = getattr(persona, "emotional_baseline", "neutral") or "neutral"

                current_emotion = "neutral"
                try:
                    states = self._service._user_emotional_states
                    state = states.get(uid)
                    if state:
                        current_emotion = getattr(state, "current", {}).get("primary", "neutral")
                except Exception:
                    pass

                # 构建画像数据
                persona_data = {
                    "user_id": uid,
                    "display_name": getattr(persona, "display_name", None),
                    "update_count": getattr(persona, "update_count", 0),
                    "last_updated": persona.last_updated.isoformat() if hasattr(persona.last_updated, "isoformat") else str(persona.last_updated),
                    "interests": top_interests,
                    "all_interests_count": len(interests),
                    "trust_level": getattr(persona, "trust_level", 0.5),
                    "intimacy_level": getattr(persona, "intimacy_level", 0.5),
                    "emotional_baseline": emotional_baseline,
                    "current_emotion": current_emotion,
                    "work_style": getattr(persona, "work_style", None),
                    "lifestyle": getattr(persona, "lifestyle", None),
                    "work_goals": list(getattr(persona, "work_goals", [])[:3]),
                    "habits": list(getattr(persona, "habits", [])[:3]),
                    "preferred_reply_style": getattr(persona, "preferred_reply_style", None),
                    "proactive_reply_preference": getattr(persona, "proactive_reply_preference", 0.5),
                    "personality_openness": getattr(persona, "personality_openness", 0.5),
                    "personality_conscientiousness": getattr(persona, "personality_conscientiousness", 0.5),
                    "personality_extraversion": getattr(persona, "personality_extraversion", 0.5),
                    "personality_agreeableness": getattr(persona, "personality_agreeableness", 0.5),
                    "personality_neuroticism": getattr(persona, "personality_neuroticism", 0.5),
                    "social_style": getattr(persona, "social_style", None),
                }

                # 如果有搜索词，进行过滤
                if query_lower:
                    match = False
                    # 搜索用户ID
                    if query_lower in uid.lower():
                        match = True
                    # 搜索显示名称
                    elif persona_data["display_name"] and query_lower in persona_data["display_name"].lower():
                        match = True
                    # 搜索兴趣
                    elif any(query_lower in k.lower() for k in interests.keys()):
                        match = True
                    # 搜索工作风格
                    elif persona_data["work_style"] and query_lower in persona_data["work_style"].lower():
                        match = True
                    # 搜索生活方式
                    elif persona_data["lifestyle"] and query_lower in persona_data["lifestyle"].lower():
                        match = True
                    # 搜索工作目标
                    elif any(query_lower in g.lower() for g in (getattr(persona, "work_goals", []) or [])):
                        match = True
                    # 搜索习惯
                    elif any(query_lower in h.lower() for h in (getattr(persona, "habits", []) or [])):
                        match = True
                    # 搜索社交风格
                    elif persona_data["social_style"] and query_lower in persona_data["social_style"].lower():
                        match = True

                    if not match:
                        continue

                all_items.append(persona_data)

            all_items.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            result["total"] = len(all_items)
            start = (page - 1) * page_size
            result["items"] = all_items[start:start + page_size]

        except Exception as e:
            logger.warning(f"Search personas error: {e}")

        return result

    async def get_all_user_ids(self) -> List[str]:
        """获取所有用户 ID"""
        try:
            personas = self._service._user_personas
            return list(personas.keys())
        except Exception:
            return []


class EmotionRepositoryImpl:
    """情感状态仓库实现"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    async def get_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据用户 ID 获取情感状态"""
        try:
            if not user_id:
                return None
            states = self._service._user_emotional_states
            state = states.get(user_id)
            if not state:
                return None
            return state.to_dict()
        except Exception as e:
            logger.warning(f"Get emotion state error: {e}")
            return None

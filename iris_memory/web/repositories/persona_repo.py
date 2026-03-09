"""用户画像和情感状态仓库"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from iris_memory.utils.logger import get_logger

logger = get_logger("web.persona_repo")


class PersonaRepository:
    """用户画像仓库"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    def _get_personas(self) -> Dict[str, Any]:
        try:
            return self._service._user_personas or {}
        except Exception:
            return {}

    def _get_emotional_states(self) -> Dict[str, Any]:
        try:
            return self._service._user_emotional_states or {}
        except Exception:
            return {}

    def _build_persona_data(self, uid: str, persona: Any) -> Dict[str, Any]:
        interests = getattr(persona, "interests", {}) or {}
        top_interests = dict(sorted(interests.items(), key=lambda x: x[1], reverse=True)[:5])

        emotional_baseline = getattr(persona, "emotional_baseline", "neutral") or "neutral"

        current_emotion = "neutral"
        try:
            states = self._get_emotional_states()
            state = states.get(uid)
            if state:
                current_emotion = getattr(state, "current", {}).get("primary", "neutral")
        except Exception:
            pass

        return {
            "user_id": uid,
            "display_name": getattr(persona, "display_name", None),
            "version": getattr(persona, "version", 3),
            "update_count": getattr(persona, "update_count", 0),
            "last_updated": (
                persona.last_updated.isoformat()
                if hasattr(persona.last_updated, "isoformat")
                else str(persona.last_updated)
            ),
            "interests": top_interests,
            "all_interests_count": len(interests),
            "trust_level": getattr(persona, "trust_level", 0.5),
            "intimacy_level": getattr(persona, "intimacy_level", 0.5),
            "emotional_baseline": emotional_baseline,
            "emotional_volatility": getattr(persona, "emotional_volatility", 0.5),
            "current_emotion": current_emotion,
            "work_style": getattr(persona, "work_style", None),
            "lifestyle": getattr(persona, "lifestyle", None),
            "work_goals": list(getattr(persona, "work_goals", [])[:3]),
            "work_challenges": list(getattr(persona, "work_challenges", [])[:3]),
            "habits": list(getattr(persona, "habits", [])[:3]),
            "social_style": getattr(persona, "social_style", None),
            "preferred_reply_style": getattr(persona, "preferred_reply_style", None),
            "proactive_reply_preference": getattr(persona, "proactive_reply_preference", 0.5),
            "personality": {
                "openness": getattr(persona, "personality_openness", 0.5),
                "conscientiousness": getattr(persona, "personality_conscientiousness", 0.5),
                "extraversion": getattr(persona, "personality_extraversion", 0.5),
                "agreeableness": getattr(persona, "personality_agreeableness", 0.5),
                "neuroticism": getattr(persona, "personality_neuroticism", 0.5),
            },
            "communication_style": {
                "formality": getattr(persona, "communication_formality", 0.5),
                "directness": getattr(persona, "communication_directness", 0.5),
                "humor": getattr(persona, "communication_humor", 0.5),
                "empathy": getattr(persona, "communication_empathy", 0.5),
            },
        }

    async def list_all(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """分页列出用户画像"""
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        result: Dict[str, Any] = {"items": [], "total": 0, "page": page, "page_size": page_size}

        try:
            personas = self._get_personas()
            all_items = [self._build_persona_data(uid, p) for uid, p in personas.items()]
            all_items.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            result["total"] = len(all_items)
            start = (page - 1) * page_size
            result["items"] = all_items[start : start + page_size]
        except Exception as e:
            logger.warning(f"List personas error: {e}")

        return result

    async def get_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据用户 ID 获取画像"""
        try:
            personas = self._get_personas()
            persona = personas.get(user_id)
            if not persona:
                return None
            return persona.to_dict()
        except Exception as e:
            logger.warning(f"Get persona error: {e}")
            return None

    async def search(self, query: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """搜索用户画像"""
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        result: Dict[str, Any] = {"items": [], "total": 0, "page": page, "page_size": page_size}

        try:
            personas = self._get_personas()
            query_lower = query.lower().strip()
            all_items = []

            for uid, persona in personas.items():
                data = self._build_persona_data(uid, persona)
                if self._matches_query(uid, persona, data, query_lower):
                    all_items.append(data)

            all_items.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            result["total"] = len(all_items)
            start = (page - 1) * page_size
            result["items"] = all_items[start : start + page_size]
        except Exception as e:
            logger.warning(f"Search personas error: {e}")

        return result

    @staticmethod
    def _matches_query(uid: str, persona: Any, data: Dict[str, Any], query_lower: str) -> bool:
        if query_lower in uid.lower():
            return True
        if data.get("display_name") and query_lower in data["display_name"].lower():
            return True
        interests = getattr(persona, "interests", {}) or {}
        if any(query_lower in k.lower() for k in interests):
            return True
        if data.get("work_style") and query_lower in data["work_style"].lower():
            return True
        if data.get("lifestyle") and query_lower in data["lifestyle"].lower():
            return True
        for goal in getattr(persona, "work_goals", []) or []:
            if query_lower in goal.lower():
                return True
        for habit in getattr(persona, "habits", []) or []:
            if query_lower in habit.lower():
                return True
        return False

    async def delete_by_user_id(self, user_id: str) -> Tuple[bool, str]:
        """删除指定用户的画像"""
        try:
            personas = self._get_personas()
            if user_id not in personas:
                return False, "用户画像不存在"
            del personas[user_id]
            return True, "删除成功"
        except Exception as e:
            logger.error(f"Delete persona error: {e}")
            return False, f"删除失败: {e}"

    async def clear_all(self) -> Tuple[bool, str, int]:
        """清空所有用户画像"""
        try:
            personas = self._get_personas()
            count = len(personas)
            personas.clear()
            return True, f"已清空 {count} 个画像", count
        except Exception as e:
            logger.error(f"Clear personas error: {e}")
            return False, f"清空失败: {e}", 0


class EmotionRepository:
    """情感状态仓库"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    async def get_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据用户 ID 获取情感状态"""
        try:
            states = self._service._user_emotional_states
            state = states.get(user_id)
            if not state:
                return None

            if hasattr(state, "to_dict"):
                return state.to_dict()

            return {
                "user_id": user_id,
                "current": getattr(state, "current", {}),
                "history": getattr(state, "history", []),
            }
        except Exception as e:
            logger.warning(f"Get emotion state error: {e}")
            return None

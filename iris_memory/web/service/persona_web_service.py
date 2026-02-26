"""Web 用户画像与情感状态服务

封装面向 Web 的用户画像和情感状态查询。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from iris_memory.web.data.persona_repo import EmotionRepositoryImpl, PersonaRepositoryImpl
from iris_memory.utils.logger import get_logger

logger = get_logger("persona_web_service")


class PersonaWebService:
    """Web 端用户画像与情感状态服务"""

    def __init__(self, memory_service: Any) -> None:
        self._persona_repo = PersonaRepositoryImpl(memory_service)
        self._emotion_repo = EmotionRepositoryImpl(memory_service)

    async def list_personas(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """获取用户画像分页列表"""
        return await self._persona_repo.list_all(page=page, page_size=page_size)

    async def get_persona_detail(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取指定用户的画像详情"""
        return await self._persona_repo.get_by_user_id(user_id)

    async def get_emotion_state(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """获取指定用户的情感状态"""
        if not user_id:
            return None
        return await self._emotion_repo.get_by_user_id(user_id)

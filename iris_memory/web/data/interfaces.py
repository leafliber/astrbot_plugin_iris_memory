"""数据访问层接口定义

定义仓库接口（Repository Pattern），隔离数据访问逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class MemoryRepository(ABC):
    """记忆数据仓库接口"""

    @abstractmethod
    async def search(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        memory_type: Optional[str] = None,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """向量搜索记忆"""
        pass

    @abstractmethod
    async def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取记忆"""
        pass

    @abstractmethod
    async def list_all(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页列出记忆"""
        pass

    @abstractmethod
    async def create(self, memory_data: Dict[str, Any]) -> Tuple[bool, str]:
        """创建记忆"""
        pass

    @abstractmethod
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """更新记忆"""
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> Tuple[bool, str]:
        """删除记忆"""
        pass

    @abstractmethod
    async def batch_delete(self, memory_ids: List[str]) -> Dict[str, Any]:
        """批量删除"""
        pass

    @abstractmethod
    async def count_by_layer(self) -> Dict[str, int]:
        """按层级统计记忆数量"""
        pass

    @abstractmethod
    async def get_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取创建趋势"""
        pass


class PersonaRepository(ABC):
    """用户画像仓库接口"""

    @abstractmethod
    async def list_all(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页列出用户画像"""
        pass

    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据用户 ID 获取画像"""
        pass

    @abstractmethod
    async def get_all_user_ids(self) -> List[str]:
        """获取所有用户 ID"""
        pass


class EmotionRepository(ABC):
    """情感状态仓库接口"""

    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据用户 ID 获取情感状态"""
        pass


class KnowledgeGraphRepository(ABC):
    """知识图谱仓库接口"""

    @abstractmethod
    async def list_nodes(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """列出节点"""
        pass

    @abstractmethod
    async def search_nodes(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """搜索节点"""
        pass

    @abstractmethod
    async def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取单个节点"""
        pass

    @abstractmethod
    async def delete_node(self, node_id: str) -> Tuple[bool, str]:
        """删除节点"""
        pass

    @abstractmethod
    async def list_edges(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        node_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """列出边"""
        pass

    @abstractmethod
    async def delete_edge(self, edge_id: str) -> Tuple[bool, str]:
        """删除边"""
        pass

    @abstractmethod
    async def get_graph_data(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        center_node_id: Optional[str] = None,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """获取图数据"""
        pass


class SessionRepository(ABC):
    """会话仓库接口"""

    @abstractmethod
    async def get_all_sessions(self) -> Dict[str, Any]:
        """获取所有会话"""
        pass

    @abstractmethod
    async def get_session_stats(self) -> Dict[str, int]:
        """获取会话统计"""
        pass

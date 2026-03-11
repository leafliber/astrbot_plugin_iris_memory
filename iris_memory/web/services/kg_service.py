"""Web 知识图谱管理服务"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from iris_memory.web.audit import audit_log
from iris_memory.web.dto.converters import edge_to_web_dict, node_to_web_dict
from iris_memory.utils.logger import get_logger

logger = get_logger("web.kg_svc")


class KgWebService:
    """Web 端知识图谱管理服务"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service
        # Repo lazily created since it needs memory_service
        self._repo: Any = None

    def _get_repo(self) -> Any:
        if self._repo is None:
            from iris_memory.web.repositories.kg_repo import KnowledgeGraphRepository
            self._repo = KnowledgeGraphRepository(self._service)
        return self._repo

    async def search_kg_nodes(
        self,
        query: str = "",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        repo = self._get_repo()
        nodes, total = await repo.search_nodes(
            query=query,
            user_id=user_id,
            group_id=group_id,
            node_type=node_type,
            page=page,
            page_size=page_size,
        )
        return {"items": [node_to_web_dict(n) for n in nodes], "total": total}

    async def list_kg_edges(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        node_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        repo = self._get_repo()
        edges, node_names, total = await repo.list_edges(
            user_id=user_id,
            group_id=group_id,
            relation_type=relation_type,
            node_id=node_id,
            page=page,
            page_size=page_size,
        )
        return {"items": [edge_to_web_dict(e, node_names) for e in edges], "total": total}

    async def delete_kg_node(self, node_id: str) -> Tuple[bool, str]:
        repo = self._get_repo()
        success, msg = await repo.delete_node(node_id)
        if success:
            audit_log("delete_kg_node", f"id={node_id}")
        return success, msg

    async def delete_kg_edge(self, edge_id: str) -> Tuple[bool, str]:
        repo = self._get_repo()
        success, msg = await repo.delete_edge(edge_id)
        if success:
            audit_log("delete_kg_edge", f"id={edge_id}")
        return success, msg

    async def get_kg_graph_data(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        center_node_id: Optional[str] = None,
        depth: int = 2,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        repo = self._get_repo()
        return await repo.get_graph_data(
            user_id=user_id,
            group_id=group_id,
            center_node_id=center_node_id,
            depth=depth,
            max_nodes=max_nodes,
        )

    async def run_maintenance(self) -> Dict[str, Any]:
        repo = self._get_repo()
        result = await repo.run_maintenance()
        if "error" not in result:
            audit_log("kg_maintenance", "completed")
        return result

    async def check_consistency(self) -> Dict[str, Any]:
        return await self._get_repo().check_consistency()

    async def get_quality_report(self) -> Dict[str, Any]:
        return await self._get_repo().get_quality_report()

"""知识图谱仓库实现

实现 KnowledgeGraphRepository 接口。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from iris_memory.knowledge_graph.kg_models import KGNode, KGEdge, KGNodeType, KGRelationType
from iris_memory.utils.logger import get_logger

logger = get_logger("kg_repo")


class KnowledgeGraphRepositoryImpl:
    """知识图谱仓库实现"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    async def list_nodes(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """列出节点"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return []

            storage = kg.storage
            async with storage._lock:
                assert storage._conn

                sql = "SELECT * FROM kg_nodes"
                conditions: List[str] = []
                params: List[Any] = []

                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if group_id:
                    conditions.append("(group_id = ? OR group_id IS NULL)")
                    params.append(group_id)
                if node_type:
                    conditions.append("node_type = ?")
                    params.append(node_type)

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += f" ORDER BY created_time DESC LIMIT {limit}"

                rows = storage._conn.execute(sql, params).fetchall()
                return [self._node_to_dict(KGNode.from_row(dict(r))) for r in rows]

        except Exception as e:
            logger.warning(f"List KG nodes error: {e}")
            return []

    async def search_nodes(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """搜索节点"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return []

            nodes = await kg.search_nodes(query, user_id, limit)
            return [self._node_to_dict(n) for n in nodes]

        except Exception as e:
            logger.warning(f"Search KG nodes error: {e}")
            return []

    async def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取单个节点"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return None

            node = await kg.get_node(node_id)
            if not node:
                return None
            return self._node_to_dict(node)

        except Exception as e:
            logger.warning(f"Get KG node error: {e}")
            return None

    async def delete_node(self, node_id: str) -> Tuple[bool, str]:
        """删除节点"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return False, "知识图谱未启用"

            success = await kg.delete_node(node_id)
            if success:
                return True, "删除成功"
            return False, "节点不存在"

        except Exception as e:
            logger.error(f"Delete KG node error: {e}")
            return False, f"删除失败: {e}"

    async def list_edges(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        node_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """列出边"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return []

            storage = kg.storage
            async with storage._lock:
                assert storage._conn

                sql = "SELECT * FROM kg_edges"
                conditions: List[str] = []
                params: List[Any] = []

                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if group_id:
                    conditions.append("(group_id = ? OR group_id IS NULL)")
                    params.append(group_id)
                if relation_type:
                    conditions.append("relation_type = ?")
                    params.append(relation_type)
                if node_id:
                    conditions.append("(source_id = ? OR target_id = ?)")
                    params.extend([node_id, node_id])

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += f" ORDER BY created_time DESC LIMIT {limit}"

                rows = storage._conn.execute(sql, params).fetchall()
                edges = [KGEdge.from_row(dict(r)) for r in rows]

                node_ids = set()
                for e in edges:
                    node_ids.add(e.source_id)
                    node_ids.add(e.target_id)

                node_names: Dict[str, str] = {}
                if node_ids:
                    placeholders = ",".join(["?"] * len(node_ids))
                    nrows = storage._conn.execute(
                        f"SELECT id, display_name, name FROM kg_nodes WHERE id IN ({placeholders})",
                        list(node_ids),
                    ).fetchall()
                    for nr in nrows:
                        nrd = dict(nr)
                        node_names[nrd["id"]] = nrd.get("display_name") or nrd.get("name") or nrd["id"]

                return [self._edge_to_dict(e, node_names) for e in edges]

        except Exception as e:
            logger.warning(f"List KG edges error: {e}")
            return []

    async def delete_edge(self, edge_id: str) -> Tuple[bool, str]:
        """删除边"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return False, "知识图谱未启用"

            success = await kg.delete_edge(edge_id)
            if success:
                return True, "删除成功"
            return False, "边不存在"

        except Exception as e:
            logger.error(f"Delete KG edge error: {e}")
            return False, f"删除失败: {e}"

    async def get_graph_data(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        center_node_id: Optional[str] = None,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """获取图数据"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return {"nodes": [], "edges": []}

            if center_node_id:
                return await self._get_graph_from_center(kg, center_node_id, depth)
            else:
                nodes = await self.list_nodes(user_id, group_id, None, 100)
                edges = await self.list_edges(user_id, group_id, None, None, 100)
                return {"nodes": nodes, "edges": edges}

        except Exception as e:
            logger.warning(f"Get graph data error: {e}")
            return {"nodes": [], "edges": []}

    async def _get_graph_from_center(
        self,
        kg: Any,
        center_node_id: str,
        depth: int,
    ) -> Dict[str, Any]:
        """从中心节点获取图数据"""
        try:
            center_node = await kg.get_node(center_node_id)
            if not center_node:
                return {"nodes": [], "edges": []}

            nodes: Dict[str, Dict[str, Any]] = {center_node_id: self._node_to_graph_dict(center_node)}
            edges: List[Dict[str, Any]] = []

            storage = kg.storage
            async with storage._lock:
                for d in range(depth):
                    current_node_ids = list(nodes.keys())
                    if not current_node_ids:
                        break

                    placeholders = ",".join(["?"] * len(current_node_ids))
                    edge_rows = storage._conn.execute(
                        f"SELECT * FROM kg_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                        current_node_ids * 2,
                    ).fetchall()

                    for er in edge_rows:
                        edge = KGEdge.from_row(dict(er))
                        source_id = edge.source_id
                        target_id = edge.target_id

                        if source_id not in nodes:
                            node = await kg.get_node(source_id)
                            if node:
                                nodes[source_id] = self._node_to_graph_dict(node)
                        if target_id not in nodes:
                            node = await kg.get_node(target_id)
                            if node:
                                nodes[target_id] = self._node_to_graph_dict(node)

                        edges.append(self._edge_to_graph_dict(edge))

            return {
                "nodes": list(nodes.values()),
                "edges": edges,
            }

        except Exception as e:
            logger.warning(f"Get graph from center error: {e}")
            return {"nodes": [], "edges": []}

    def _node_to_dict(self, node: KGNode) -> Dict[str, Any]:
        """节点转字典"""
        return {
            "id": node.id,
            "name": node.name,
            "display_name": node.display_name or node.name,
            "node_type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
            "description": node.description or "",
            "properties": node.properties or {},
            "user_id": node.user_id,
            "group_id": node.group_id,
            "confidence": node.confidence,
            "created_time": node.created_time.isoformat() if hasattr(node.created_time, "isoformat") else str(node.created_time),
        }

    def _node_to_graph_dict(self, node: KGNode) -> Dict[str, Any]:
        """节点转图数据字典"""
        return {
            "id": node.id,
            "label": node.display_name or node.name,
            "node_type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
            "group_id": node.group_id,
        }

    def _edge_to_dict(self, edge: KGEdge, node_names: Dict[str, str]) -> Dict[str, Any]:
        """边转字典"""
        return {
            "id": edge.id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "source_name": node_names.get(edge.source_id, edge.source_id),
            "target_name": node_names.get(edge.target_id, edge.target_id),
            "relation_type": edge.relation_type.value if hasattr(edge.relation_type, "value") else str(edge.relation_type),
            "relation_label": edge.relation_label or edge.relation_type.value if hasattr(edge.relation_type, "value") else str(edge.relation_type),
            "user_id": edge.user_id,
            "group_id": edge.group_id,
            "confidence": edge.confidence,
            "weight": edge.weight,
            "created_time": edge.created_time.isoformat() if hasattr(edge.created_time, "isoformat") else str(edge.created_time),
        }

    def _edge_to_graph_dict(self, edge: KGEdge) -> Dict[str, Any]:
        """边转图数据字典"""
        return {
            "id": edge.id,
            "source": edge.source_id,
            "target": edge.target_id,
            "label": edge.relation_label or edge.relation_type.value if hasattr(edge.relation_type, "value") else str(edge.relation_type),
            "weight": edge.weight,
        }

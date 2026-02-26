"""
Web 服务层 - 向后兼容门面（Facade）

保留原 WebService 的公共 API 签名以兼容现有测试和外部调用方，
内部逻辑已迁移到各领域子服务：
  - DashboardService   (统计面板)
  - MemoryWebService   (记忆 CRUD)
  - KgWebService       (知识图谱)
  - PersonaWebService  (用户画像/情感)
  - IoService          (数据导入导出)

新代码应直接使用子服务，不应继续扩展此文件。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from iris_memory.web.service.audit import audit_log
from iris_memory.web.service.dashboard_service import DashboardService
from iris_memory.web.service.dto.converters import (
    edge_to_graph_dict,
    memory_to_web_dict,
    node_to_graph_dict,
    node_to_web_dict,
)
from iris_memory.web.service.io_service import IoService
from iris_memory.web.service.kg_web_service import KgWebService
from iris_memory.web.service.memory_web_service import MemoryWebService
from iris_memory.web.service.persona_web_service import PersonaWebService
from iris_memory.utils.logger import get_logger

logger = get_logger("web_service")


class WebService:
    """Web 业务服务层（向后兼容门面）

    封装所有面向 Web 前端的查询和操作逻辑。
    内部委托给领域子服务，保持 API 签名不变。
    """

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service
        self._dashboard = DashboardService(memory_service)
        self._memory = MemoryWebService(memory_service)
        self._kg = KgWebService(memory_service)
        self._persona = PersonaWebService(memory_service)
        self._io = IoService(memory_service)

    # ================================================================
    # 统计面板
    # ================================================================

    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """获取仪表盘统计信息"""
        return await self._dashboard.get_dashboard_stats()

    async def _get_system_stats(self) -> Dict[str, Any]:
        """系统级统计（兼容旧测试）"""
        return await self._dashboard._get_system_stats()

    async def _get_memory_overview(self) -> Dict[str, Any]:
        """记忆总览统计（兼容旧测试）"""
        return await self._dashboard._get_memory_overview()

    async def _get_kg_overview(self) -> Dict[str, Any]:
        """知识图谱总览（兼容旧测试）"""
        return await self._dashboard._get_kg_overview()

    async def get_memory_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取记忆创建趋势数据"""
        return await self._dashboard.get_memory_trend(days)

    # ================================================================
    # 记忆管理
    # ================================================================

    async def search_memories_web(
        self,
        query: str = "",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Web 端记忆检索"""
        return await self._memory.search_memories_web(
            query=query,
            user_id=user_id,
            group_id=group_id,
            storage_layer=storage_layer,
            memory_type=memory_type,
            page=page,
            page_size=page_size,
        )

    async def get_memory_detail(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单条记忆的完整详情"""
        return await self._memory.get_memory_detail(memory_id)

    async def update_memory_by_id(
        self,
        memory_id: str,
        updates: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """更新单条记忆的内容/元数据"""
        return await self._memory.update_memory_by_id(memory_id, updates)

    async def delete_memory_by_id(self, memory_id: str) -> Tuple[bool, str]:
        """删除单条记忆"""
        return await self._memory.delete_memory_by_id(memory_id)

    async def batch_delete_memories(self, memory_ids: List[str]) -> Dict[str, Any]:
        """批量删除记忆"""
        return await self._memory.batch_delete_memories(memory_ids)

    # ================================================================
    # 知识图谱
    # ================================================================

    async def search_kg_nodes(
        self,
        query: str = "",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """搜索知识图谱节点"""
        return await self._kg.search_kg_nodes(
            query=query,
            user_id=user_id,
            group_id=group_id,
            node_type=node_type,
            limit=limit,
        )

    async def list_kg_edges(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        node_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """列出知识图谱边"""
        return await self._kg.list_kg_edges(
            user_id=user_id,
            group_id=group_id,
            relation_type=relation_type,
            node_id=node_id,
            limit=limit,
        )

    async def get_kg_graph_data(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        center_node_id: Optional[str] = None,
        depth: int = 2,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """获取知识图谱可视化数据"""
        return await self._kg.get_kg_graph_data(
            user_id=user_id,
            group_id=group_id,
            center_node_id=center_node_id,
            depth=depth,
            max_nodes=max_nodes,
        )

    async def delete_kg_node(self, node_id: str) -> Tuple[bool, str]:
        """删除知识图谱节点及关联边"""
        return await self._kg.delete_kg_node(node_id)

    async def delete_kg_edge(self, edge_id: str) -> Tuple[bool, str]:
        """删除知识图谱边"""
        return await self._kg.delete_kg_edge(edge_id)

    # ================================================================
    # 用户画像与情感状态
    # ================================================================

    async def list_personas(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """获取用户画像分页列表"""
        return await self._persona.list_personas(page=page, page_size=page_size)

    async def get_persona_detail(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取指定用户的画像详情"""
        return await self._persona.get_persona_detail(user_id)

    async def get_emotion_state(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """获取指定用户的情感状态"""
        return await self._persona.get_emotion_state(user_id, group_id)

    # ================================================================
    # 数据导入导出
    # ================================================================

    async def export_memories(
        self,
        fmt: str = "json",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, str, str]:
        """导出记忆数据"""
        fmt = kwargs.get("format", fmt)
        return await self._io.export_memories(
            fmt=fmt, user_id=user_id, group_id=group_id, storage_layer=storage_layer,
        )

    async def export_kg(
        self,
        fmt: str = "json",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, str, str]:
        """导出知识图谱数据"""
        fmt = kwargs.get("format", fmt)
        return await self._io.export_kg(fmt=fmt, user_id=user_id, group_id=group_id)

    async def import_memories(
        self,
        data: str = "",
        fmt: str = "json",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """导入记忆数据"""
        fmt = kwargs.get("format", fmt)
        return await self._io.import_memories(data=data, fmt=fmt)

    async def import_kg(
        self,
        data: str = "",
        fmt: str = "json",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """导入知识图谱数据"""
        fmt = kwargs.get("format", fmt)
        return await self._io.import_kg(data=data, fmt=fmt)

    async def preview_import_data(
        self,
        data: str,
        fmt: str,
        import_type: str,
    ) -> Dict[str, Any]:
        """预览导入数据（不实际导入）"""
        return await self._io.preview_import_data(data=data, fmt=fmt, import_type=import_type)

    # ================================================================
    # 内部工具方法（兼容旧测试直接调用）
    # ================================================================

    def _audit_log(self, action: str, detail: str = "") -> None:
        """记录审计日志"""
        audit_log(action, detail)

    def _memory_to_web_dict(self, memory: Any) -> Dict[str, Any]:
        """将 Memory 对象转换为前端展示字典"""
        return memory_to_web_dict(memory)

    def _node_to_web_dict(self, node: Any) -> Dict[str, Any]:
        """节点转前端展示字典"""
        return node_to_web_dict(node)

    def _node_to_graph_dict(self, node: Any) -> Dict[str, Any]:
        """节点转图谱可视化字典"""
        return node_to_graph_dict(node)

    def _edge_to_graph_dict(self, edge: Any) -> Dict[str, Any]:
        """边转图谱可视化字典"""
        return edge_to_graph_dict(edge)

    def _memories_to_csv(self, items: List[Dict]) -> str:
        """将记忆列表转为 CSV 字符串"""
        return self._io._memories_to_csv(items)

    def _kg_nodes_to_csv(self, nodes: List[Dict]) -> str:
        """节点列表转 CSV"""
        return self._io._kg_nodes_to_csv(nodes)

    def _kg_edges_to_csv(self, edges: List[Dict]) -> str:
        """边列表转 CSV"""
        return self._io._kg_edges_to_csv(edges)

    def _parse_json_memories(self, data: str) -> List[Dict]:
        """解析 JSON 格式的记忆数据"""
        return self._io._parse_json_memories(data)

    def _parse_csv_memories(self, data: str) -> List[Dict]:
        """解析 CSV 格式的记忆数据"""
        return self._io._parse_csv_memories(data)

    def _parse_json_kg(self, data: str) -> Tuple[List[Dict], List[Dict]]:
        """解析 JSON 格式的 KG 数据"""
        return self._io._parse_json_kg(data)

    def _parse_csv_kg(self, data: str) -> Tuple[List[Dict], List[Dict]]:
        """解析 CSV 格式的 KG 数据"""
        return self._io._parse_csv_kg(data)

    def _validate_memory_import(self, item: Dict) -> Tuple[bool, str]:
        """验证导入记忆数据的合法性"""
        return self._io._validate_memory_import(item)

    def _validate_kg_node_import(self, nd: Dict) -> Tuple[bool, str]:
        """验证 KG 节点导入数据"""
        return self._io._validate_kg_node_import(nd)

    def _validate_kg_edge_import(self, ed: Dict) -> Tuple[bool, str]:
        """验证 KG 边导入数据"""
        return self._io._validate_kg_edge_import(ed)

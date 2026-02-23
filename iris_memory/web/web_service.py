"""
Web 服务层 - 封装 Web 模块的业务逻辑

为 API 路由提供数据查询、转换、验证等服务。
不直接依赖 HTTP 框架，纯业务逻辑。
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core.types import MemoryType, StorageLayer, QualityLevel
from iris_memory.knowledge_graph.kg_models import KGNode, KGEdge, KGNodeType, KGRelationType
from iris_memory.utils.logger import get_logger

logger = get_logger("web_service")

# 审计日志 logger（独立通道）
audit_logger = get_logger("web_audit")

# 导出限制
_EXPORT_MAX_MEMORIES = 10000
_EXPORT_MAX_KG_ITEMS = 50000
_IMPORT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


class WebService:
    """Web 业务服务层

    封装所有面向 Web 前端的查询和操作逻辑，
    上层 API 路由只负责 HTTP 协议转换。
    """

    def __init__(self, memory_service: Any) -> None:
        """
        Args:
            memory_service: MemoryService 实例
        """
        self._service = memory_service

    # ================================================================
    # 统计面板
    # ================================================================

    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """获取仪表盘统计信息

        Returns:
            包含系统概览、记忆统计、KG 统计的字典
        """
        stats: Dict[str, Any] = {
            "system": await self._get_system_stats(),
            "memories": await self._get_memory_overview(),
            "knowledge_graph": await self._get_kg_overview(),
            "health": {},
        }

        # 系统健康状态
        try:
            stats["health"] = self._service.health_check()
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            stats["health"] = {"status": "unknown"}

        return stats

    async def _get_system_stats(self) -> Dict[str, Any]:
        """系统级统计"""
        result: Dict[str, Any] = {
            "total_users": 0,
            "total_sessions": 0,
            "active_sessions": 0,
        }

        try:
            if self._service.session_manager:
                sessions = self._service.session_manager.get_all_sessions()
                result["total_sessions"] = len(sessions)
                # 统计活跃会话（有工作记忆的）
                active = 0
                user_ids = set()
                for key, meta in sessions.items():
                    user_ids.add(key.split(":")[0] if ":" in key else key)
                    wm = self._service.session_manager.working_memory_cache.get(key, [])
                    if wm:
                        active += 1
                result["active_sessions"] = active
                result["total_users"] = len(user_ids)
        except Exception as e:
            logger.debug(f"Session stats error: {e}")

        # 用户画像数量
        try:
            result["total_personas"] = len(self._service._user_personas)
        except Exception:
            result["total_personas"] = 0

        return result

    async def _get_memory_overview(self) -> Dict[str, Any]:
        """记忆总览统计"""
        result: Dict[str, Any] = {
            "total_count": 0,
            "by_layer": {"working": 0, "episodic": 0, "semantic": 0},
            "by_type": {},
        }

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return result

            # 总数
            collection = chroma.collection
            total = collection.count()
            result["total_count"] = total

            # 按层级统计
            for layer in StorageLayer:
                try:
                    res = collection.get(
                        where={"storage_layer": layer.value},
                        include=[]
                    )
                    result["by_layer"][layer.value] = len(res["ids"]) if res["ids"] else 0
                except Exception:
                    pass

            # 按类型统计
            for mtype in MemoryType:
                try:
                    res = collection.get(
                        where={"type": mtype.value},
                        include=[]
                    )
                    count = len(res["ids"]) if res["ids"] else 0
                    if count > 0:
                        result["by_type"][mtype.value] = count
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Memory overview error: {e}")

        return result

    async def _get_kg_overview(self) -> Dict[str, Any]:
        """知识图谱总览"""
        result: Dict[str, Any] = {"nodes": 0, "edges": 0, "enabled": False}

        try:
            kg = self._service.kg
            if kg and kg.enabled:
                result["enabled"] = True
                stats = await kg.get_stats()
                result["nodes"] = stats.get("nodes", 0)
                result["edges"] = stats.get("edges", 0)
        except Exception as e:
            logger.debug(f"KG overview error: {e}")

        return result

    async def get_memory_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取记忆创建趋势数据

        Args:
            days: 回溯天数

        Returns:
            按日期分组的记忆创建数统计列表
        """
        trend: List[Dict[str, Any]] = []

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return trend

            # 获取所有记忆的创建时间
            collection = chroma.collection
            results = collection.get(include=["metadatas"])

            if not results["ids"]:
                return trend

            # 按日期分组统计
            date_counts: Dict[str, int] = {}
            cutoff = datetime.now() - timedelta(days=days)

            for meta in results["metadatas"]:
                created = meta.get("created_time", "")
                if not created:
                    continue
                try:
                    dt = datetime.fromisoformat(created)
                    if dt >= cutoff:
                        date_key = dt.strftime("%Y-%m-%d")
                        date_counts[date_key] = date_counts.get(date_key, 0) + 1
                except (ValueError, TypeError):
                    pass

            # 填充空白日期
            for i in range(days):
                d = (datetime.now() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
                trend.append({"date": d, "count": date_counts.get(d, 0)})

        except Exception as e:
            logger.warning(f"Memory trend error: {e}")

        return trend

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
        """Web 端记忆检索

        支持向量搜索和条件过滤，带分页。

        Returns:
            {items: [...], total: N, page: N, page_size: N}
        """
        result: Dict[str, Any] = {"items": [], "total": 0, "page": page, "page_size": page_size}

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return result

            # 有搜索词时做向量搜索
            if query and user_id:
                sl = StorageLayer(storage_layer) if storage_layer else None
                memories = await chroma.query_memories(
                    query_text=query,
                    user_id=user_id,
                    group_id=group_id,
                    top_k=100,
                    storage_layer=sl,
                )
                items = [self._memory_to_web_dict(m) for m in memories]
                if memory_type:
                    items = [it for it in items if it.get("type") == memory_type]
                result["total"] = len(items)
                start = (page - 1) * page_size
                result["items"] = items[start:start + page_size]
                return result

            # 无搜索词时列出所有记忆（分页通过客户端获取全量后截取）
            collection = chroma.collection
            where_clause: Dict[str, Any] = {}

            if user_id:
                where_clause["user_id"] = user_id
            if group_id:
                where_clause["group_id"] = group_id
            if storage_layer:
                where_clause["storage_layer"] = storage_layer
            if memory_type:
                where_clause["type"] = memory_type

            if where_clause:
                built = chroma._build_where_clause(where_clause)
                res = collection.get(
                    where=built,
                    include=["documents", "metadatas"]
                )
            else:
                res = collection.get(include=["documents", "metadatas"])

            if res["ids"]:
                all_items = []
                for i in range(len(res["ids"])):
                    item = {
                        "id": res["ids"][i],
                        "content": res["documents"][i] if res.get("documents") else "",
                    }
                    if res.get("metadatas") and i < len(res["metadatas"]):
                        meta = res["metadatas"][i]
                        item.update({
                            "user_id": meta.get("user_id", ""),
                            "group_id": meta.get("group_id", ""),
                            "sender_name": meta.get("sender_name", ""),
                            "type": meta.get("type", ""),
                            "storage_layer": meta.get("storage_layer", ""),
                            "scope": meta.get("scope", ""),
                            "confidence": meta.get("confidence", 0),
                            "importance_score": meta.get("importance_score", 0),
                            "created_time": meta.get("created_time", ""),
                            "summary": meta.get("summary", ""),
                        })
                    all_items.append(item)

                # 按创建时间倒序
                all_items.sort(
                    key=lambda x: x.get("created_time", ""),
                    reverse=True
                )
                result["total"] = len(all_items)
                start = (page - 1) * page_size
                result["items"] = all_items[start:start + page_size]

        except Exception as e:
            logger.warning(f"Web search memories error: {e}")

        return result

    async def delete_memory_by_id(self, memory_id: str) -> Tuple[bool, str]:
        """删除单条记忆

        Returns:
            (success, message)
        """
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return False, "存储服务未就绪"

            success = await chroma.delete_memory(memory_id)
            if success:
                # 同步删除关联的 KG 边
                kg = self._service.kg
                if kg and kg.enabled:
                    try:
                        await kg.storage.delete_by_memory_id(memory_id)
                    except Exception:
                        pass
                self._audit_log("delete_memory", f"id={memory_id}")
                return True, "删除成功"
            return False, "记忆不存在或删除失败"

        except Exception as e:
            logger.error(f"Delete memory error: {e}")
            return False, f"删除失败: {e}"

    async def batch_delete_memories(self, memory_ids: List[str]) -> Dict[str, Any]:
        """批量删除记忆

        Returns:
            {success_count, fail_count, errors}
        """
        success_count = 0
        fail_count = 0
        errors: List[str] = []

        for mid in memory_ids:
            ok, msg = await self.delete_memory_by_id(mid)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                errors.append(f"{mid}: {msg}")

        return {
            "success_count": success_count,
            "fail_count": fail_count,
            "errors": errors[:10],  # 最多返回 10 条错误
        }

    def _audit_log(self, action: str, detail: str = "") -> None:
        """记录审计日志

        Args:
            action: 操作类型（如 delete_memory, import, export 等）
            detail: 操作详情
        """
        audit_logger.info(f"[AUDIT] action={action} detail={detail} time={datetime.now().isoformat()}")

    async def update_memory_by_id(
        self,
        memory_id: str,
        updates: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """更新单条记忆的内容/元数据

        Args:
            memory_id: 记忆 ID
            updates: 要更新的字段字典 (content, type, storage_layer, confidence, summary 等)

        Returns:
            (success, message)
        """
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return False, "存储服务未就绪"

            # 先获取原记忆
            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if not res["ids"]:
                return False, "记忆不存在"

            from iris_memory.models.memory import Memory

            # 构建 Memory 对象
            doc = res["documents"][0] if res.get("documents") else ""
            meta = res["metadatas"][0] if res.get("metadatas") else {}

            memory = Memory(
                id=memory_id,
                content=updates.get("content", doc),
                user_id=meta.get("user_id", ""),
                sender_name=meta.get("sender_name", ""),
                group_id=meta.get("group_id") or None,
                storage_layer=StorageLayer(updates.get("storage_layer", meta.get("storage_layer", "episodic"))),
                created_time=datetime.fromisoformat(meta["created_time"]) if meta.get("created_time") else datetime.now(),
            )

            # 更新可选字段
            if updates.get("type") or meta.get("type"):
                try:
                    memory.type = MemoryType(updates.get("type", meta.get("type")))
                except ValueError:
                    pass
            memory.confidence = float(updates.get("confidence", meta.get("confidence", 0.5)))
            memory.importance_score = float(updates.get("importance_score", meta.get("importance_score", 0.5)))
            if updates.get("summary") is not None:
                memory.summary = updates["summary"]
            elif meta.get("summary"):
                memory.summary = meta["summary"]

            success = await chroma.update_memory(memory)
            if success:
                self._audit_log("update_memory", f"id={memory_id} fields={list(updates.keys())}")
                return True, "更新成功"
            return False, "更新失败"

        except Exception as e:
            logger.error(f"Update memory error: {e}")
            return False, f"更新失败: {e}"

    async def get_memory_detail(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单条记忆的完整详情

        Args:
            memory_id: 记忆 ID

        Returns:
            完整记忆字典，不截断 content
        """
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return None

            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if not res["ids"]:
                return None

            item: Dict[str, Any] = {
                "id": res["ids"][0],
                "content": res["documents"][0] if res.get("documents") else "",
            }
            if res.get("metadatas") and len(res["metadatas"]) > 0:
                meta = res["metadatas"][0]
                item.update({
                    "user_id": meta.get("user_id", ""),
                    "group_id": meta.get("group_id", ""),
                    "sender_name": meta.get("sender_name", ""),
                    "type": meta.get("type", ""),
                    "storage_layer": meta.get("storage_layer", ""),
                    "scope": meta.get("scope", ""),
                    "confidence": meta.get("confidence", 0),
                    "importance_score": meta.get("importance_score", 0),
                    "created_time": meta.get("created_time", ""),
                    "summary": meta.get("summary", ""),
                    "keywords": meta.get("keywords", ""),
                    "quality_level": meta.get("quality_level", ""),
                    "access_count": meta.get("access_count", 0),
                    "rif_score": meta.get("rif_score", 0),
                })
            return item

        except Exception as e:
            logger.warning(f"Get memory detail error: {e}")
            return None

    async def preview_import_data(
        self,
        data: str,
        format: str,
        import_type: str,
    ) -> Dict[str, Any]:
        """预览导入数据（不实际导入）

        Args:
            data: 文件内容
            format: 'json' 或 'csv'
            import_type: 'memories' 或 'kg'

        Returns:
            {total, valid, invalid, preview_items, errors}
        """
        result: Dict[str, Any] = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "preview_items": [],
            "errors": [],
        }

        try:
            if import_type == "memories":
                if format == "csv":
                    items = self._parse_csv_memories(data)
                else:
                    items = self._parse_json_memories(data)

                result["total"] = len(items)
                for item in items:
                    valid, err = self._validate_memory_import(item)
                    if valid:
                        result["valid"] += 1
                    else:
                        result["invalid"] += 1
                        if len(result["errors"]) < 10:
                            result["errors"].append(err)

                # 返回前 20 条预览
                for item in items[:20]:
                    result["preview_items"].append({
                        "content": (item.get("content") or "")[:200],
                        "user_id": item.get("user_id", ""),
                        "type": item.get("type", ""),
                        "storage_layer": item.get("storage_layer", ""),
                    })

            elif import_type == "kg":
                if format == "csv":
                    nodes_data, edges_data = self._parse_csv_kg(data)
                else:
                    nodes_data, edges_data = self._parse_json_kg(data)

                result["total"] = len(nodes_data) + len(edges_data)
                node_valid = 0
                edge_valid = 0

                for nd in nodes_data:
                    valid, err = self._validate_kg_node_import(nd)
                    if valid:
                        node_valid += 1
                    else:
                        result["invalid"] += 1
                        if len(result["errors"]) < 10:
                            result["errors"].append(f"节点: {err}")

                for ed in edges_data:
                    valid, err = self._validate_kg_edge_import(ed)
                    if valid:
                        edge_valid += 1
                    else:
                        result["invalid"] += 1
                        if len(result["errors"]) < 10:
                            result["errors"].append(f"边: {err}")

                result["valid"] = node_valid + edge_valid

                # 预览
                for nd in nodes_data[:10]:
                    result["preview_items"].append({
                        "item_type": "node",
                        "name": nd.get("name", ""),
                        "node_type": nd.get("node_type", ""),
                        "user_id": nd.get("user_id", ""),
                    })
                for ed in edges_data[:10]:
                    result["preview_items"].append({
                        "item_type": "edge",
                        "source_id": ed.get("source_id", ""),
                        "target_id": ed.get("target_id", ""),
                        "relation_type": ed.get("relation_type", ""),
                    })

        except Exception as e:
            result["errors"].append(f"解析失败: {e}")

        return result

    # ================================================================
    # 用户画像与情感状态
    # ================================================================

    async def get_user_personas_list(self) -> List[Dict[str, Any]]:
        """获取所有用户画像摘要列表"""
        result: List[Dict[str, Any]] = []
        try:
            personas = self._service._user_personas
            for uid, persona in personas.items():
                result.append({
                    "user_id": uid,
                    "update_count": getattr(persona, "update_count", 0),
                    "last_updated": persona.last_updated.isoformat() if hasattr(persona.last_updated, "isoformat") else str(persona.last_updated),
                    "interests": dict(list(getattr(persona, "interests", {}).items())[:5]),
                    "trust_level": getattr(persona, "trust_level", 0.5),
                    "intimacy_level": getattr(persona, "intimacy_level", 0.5),
                    "emotional_baseline": getattr(persona, "emotional_baseline", "neutral"),
                    "work_style": getattr(persona, "work_style", None),
                    "lifestyle": getattr(persona, "lifestyle", None),
                })
        except Exception as e:
            logger.warning(f"Get personas list error: {e}")
        return result

    async def get_user_persona_detail(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取指定用户的画像详情"""
        try:
            personas = self._service._user_personas
            persona = personas.get(user_id)
            if not persona:
                return None
            return persona.to_dict()
        except Exception as e:
            logger.warning(f"Get persona detail error: {e}")
            return None

    async def get_emotion_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取指定用户的情感状态"""
        try:
            states = self._service._user_emotional_states
            state = states.get(user_id)
            if not state:
                return None
            return state.to_dict()
        except Exception as e:
            logger.warning(f"Get emotion state error: {e}")
            return None

    # ================================================================
    # KG 边管理
    # ================================================================

    async def list_kg_edges(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        node_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """列出知识图谱边

        Args:
            user_id: 用户 ID 过滤
            group_id: 群组 ID 过滤
            relation_type: 关系类型过滤
            node_id: 关联节点 ID 过滤
            limit: 最大返回数

        Returns:
            边信息列表（含源/目标节点名称）
        """
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

                # 获取节点名称映射
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

            result: List[Dict[str, Any]] = []
            for e in edges:
                ed = {
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "source_name": node_names.get(e.source_id, e.source_id),
                    "target_name": node_names.get(e.target_id, e.target_id),
                    "relation_type": e.relation_type.value,
                    "relation_label": e.relation_label or e.relation_type.value,
                    "user_id": e.user_id,
                    "group_id": e.group_id,
                    "confidence": e.confidence,
                    "weight": e.weight,
                    "created_time": e.created_time.isoformat() if isinstance(e.created_time, datetime) else str(e.created_time),
                }
                result.append(ed)
            return result

        except Exception as e:
            logger.warning(f"List KG edges error: {e}")
            return []

    def _memory_to_web_dict(self, memory: Any) -> Dict[str, Any]:
        """将 Memory 对象转换为前端展示字典"""
        return {
            "id": getattr(memory, "id", ""),
            "content": getattr(memory, "content", ""),
            "summary": getattr(memory, "summary", ""),
            "user_id": getattr(memory, "user_id", ""),
            "sender_name": getattr(memory, "sender_name", ""),
            "group_id": getattr(memory, "group_id", ""),
            "type": getattr(memory, "type", "").value if hasattr(getattr(memory, "type", None), "value") else str(getattr(memory, "type", "")),
            "storage_layer": getattr(memory, "storage_layer", "").value if hasattr(getattr(memory, "storage_layer", None), "value") else str(getattr(memory, "storage_layer", "")),
            "scope": getattr(memory, "scope", "").value if hasattr(getattr(memory, "scope", None), "value") else str(getattr(memory, "scope", "")),
            "confidence": getattr(memory, "confidence", 0),
            "importance_score": getattr(memory, "importance_score", 0),
            "created_time": getattr(memory, "created_time", "").isoformat() if hasattr(getattr(memory, "created_time", None), "isoformat") else str(getattr(memory, "created_time", "")),
            "keywords": getattr(memory, "keywords", []),
        }

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
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return []

            nt = KGNodeType(node_type) if node_type else None

            if query:
                nodes = await kg.storage.search_nodes(
                    query=query,
                    user_id=user_id,
                    group_id=group_id,
                    node_type=nt,
                    limit=limit,
                )
            else:
                # 无查询词，列出所有节点
                nodes = await self._list_all_kg_nodes(
                    user_id=user_id,
                    group_id=group_id,
                    node_type=nt,
                    limit=limit,
                )

            return [self._node_to_web_dict(n) for n in nodes]

        except Exception as e:
            logger.warning(f"Search KG nodes error: {e}")
            return []

    async def _list_all_kg_nodes(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[KGNodeType] = None,
        limit: int = 50,
    ) -> List[KGNode]:
        """列出所有节点（带过滤）"""
        kg = self._service.kg
        if not kg or not kg.enabled:
            return []

        storage = kg.storage
        async with storage._lock:
            assert storage._conn
            sql = "SELECT * FROM kg_nodes"
            conditions = []
            params: List[Any] = []

            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            if group_id:
                conditions.append("(group_id = ? OR group_id IS NULL)")
                params.append(group_id)
            if node_type:
                conditions.append("node_type = ?")
                params.append(node_type.value)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += f" ORDER BY mention_count DESC LIMIT {limit}"

            rows = storage._conn.execute(sql, params).fetchall()
            return [KGNode.from_row(dict(r)) for r in rows]

    async def get_kg_graph_data(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        center_node_id: Optional[str] = None,
        depth: int = 2,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """获取知识图谱可视化数据

        Returns:
            {nodes: [...], edges: [...]}
        """
        result: Dict[str, Any] = {"nodes": [], "edges": []}

        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return result

            storage = kg.storage

            if center_node_id:
                # 从中心节点扩展
                return await self._get_graph_from_center(storage, center_node_id, depth, max_nodes)

            # 获取所有节点（限制数量）
            nodes = await self._list_all_kg_nodes(
                user_id=user_id,
                group_id=group_id,
                limit=max_nodes,
            )

            if not nodes:
                return result

            node_ids = {n.id for n in nodes}
            result["nodes"] = [self._node_to_graph_dict(n) for n in nodes]

            # 获取这些节点间的边
            async with storage._lock:
                assert storage._conn
                if len(node_ids) > 0:
                    placeholders = ",".join(["?"] * len(node_ids))
                    ids_list = list(node_ids)
                    rows = storage._conn.execute(
                        f"""SELECT * FROM kg_edges
                            WHERE source_id IN ({placeholders})
                            OR target_id IN ({placeholders})
                            LIMIT 500""",
                        ids_list + ids_list,
                    ).fetchall()

                    for r in rows:
                        edge = KGEdge.from_row(dict(r))
                        if edge.source_id in node_ids and edge.target_id in node_ids:
                            result["edges"].append(self._edge_to_graph_dict(edge))

        except Exception as e:
            logger.warning(f"Get KG graph data error: {e}")

        return result

    async def _get_graph_from_center(
        self,
        storage: Any,
        center_node_id: str,
        depth: int,
        max_nodes: int,
    ) -> Dict[str, Any]:
        """从中心节点出发获取子图"""
        visited_nodes: Dict[str, KGNode] = {}
        edges_list: List[KGEdge] = []
        to_visit = [center_node_id]

        for _ in range(depth):
            if not to_visit or len(visited_nodes) >= max_nodes:
                break

            next_visit = []
            for nid in to_visit:
                if nid in visited_nodes:
                    continue
                node = await storage.get_node(nid)
                if node:
                    visited_nodes[nid] = node
                neighbors = await storage.get_neighbors(nid, limit=20)
                for edge, neighbor in neighbors:
                    edges_list.append(edge)
                    if neighbor.id not in visited_nodes and len(visited_nodes) < max_nodes:
                        next_visit.append(neighbor.id)
                        visited_nodes[neighbor.id] = neighbor

            to_visit = next_visit

        # 去重边
        seen_edges = set()
        unique_edges = []
        for e in edges_list:
            if e.id not in seen_edges:
                seen_edges.add(e.id)
                if e.source_id in visited_nodes and e.target_id in visited_nodes:
                    unique_edges.append(e)

        return {
            "nodes": [self._node_to_graph_dict(n) for n in visited_nodes.values()],
            "edges": [self._edge_to_graph_dict(e) for e in unique_edges],
        }

    async def delete_kg_node(self, node_id: str) -> Tuple[bool, str]:
        """删除知识图谱节点及关联边"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return False, "知识图谱未启用"

            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                # 先删除关联边
                with storage._tx() as cur:
                    cur.execute(
                        "DELETE FROM kg_edges WHERE source_id = ? OR target_id = ?",
                        (node_id, node_id),
                    )
                    edge_count = cur.rowcount
                    cur.execute("DELETE FROM kg_nodes WHERE id = ?", (node_id,))
                    node_count = cur.rowcount

                storage._invalidate_cache()

                if node_count > 0:
                    self._audit_log("delete_kg_node", f"id={node_id} edges_removed={edge_count}")
                    return True, f"已删除节点及 {edge_count} 条关联边"
                return False, "节点不存在"

        except Exception as e:
            logger.error(f"Delete KG node error: {e}")
            return False, f"删除失败: {e}"

    async def delete_kg_edge(self, edge_id: str) -> Tuple[bool, str]:
        """删除知识图谱边"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return False, "知识图谱未启用"

            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                with storage._tx() as cur:
                    cur.execute("DELETE FROM kg_edges WHERE id = ?", (edge_id,))
                    if cur.rowcount > 0:
                        self._audit_log("delete_kg_edge", f"id={edge_id}")
                        return True, "删除成功"
                    return False, "边不存在"

        except Exception as e:
            logger.error(f"Delete KG edge error: {e}")
            return False, f"删除失败: {e}"

    def _node_to_web_dict(self, node: KGNode) -> Dict[str, Any]:
        """节点转前端展示字典"""
        return {
            "id": node.id,
            "name": node.name,
            "display_name": node.display_name,
            "node_type": node.node_type.value,
            "user_id": node.user_id,
            "group_id": node.group_id,
            "aliases": node.aliases,
            "mention_count": node.mention_count,
            "confidence": node.confidence,
            "created_time": node.created_time.isoformat() if isinstance(node.created_time, datetime) else str(node.created_time),
        }

    def _node_to_graph_dict(self, node: KGNode) -> Dict[str, Any]:
        """节点转图谱可视化字典"""
        return {
            "id": node.id,
            "label": node.display_name or node.name,
            "type": node.node_type.value,
            "size": min(30, 10 + node.mention_count * 2),
            "confidence": node.confidence,
        }

    def _edge_to_graph_dict(self, edge: KGEdge) -> Dict[str, Any]:
        """边转图谱可视化字典"""
        return {
            "id": edge.id,
            "source": edge.source_id,
            "target": edge.target_id,
            "label": edge.relation_label or edge.relation_type.value,
            "relation_type": edge.relation_type.value,
            "weight": edge.weight,
            "confidence": edge.confidence,
        }

    # ================================================================
    # 数据导入导出 - 记忆
    # ================================================================

    async def export_memories(
        self,
        format: str = "json",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """导出记忆数据

        Args:
            format: 导出格式 'json' 或 'csv'
            user_id: 仅导出指定用户
            group_id: 仅导出指定群组
            storage_layer: 仅导出指定层级

        Returns:
            (data_string, content_type, filename)
        """
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return "", "text/plain", "error.txt"

            # 构建查询条件
            collection = chroma.collection
            where_clause: Dict[str, Any] = {}
            if user_id:
                where_clause["user_id"] = user_id
            if group_id:
                where_clause["group_id"] = group_id
            if storage_layer:
                where_clause["storage_layer"] = storage_layer

            if where_clause:
                built = chroma._build_where_clause(where_clause)
                res = collection.get(where=built, include=["documents", "metadatas"])
            else:
                res = collection.get(include=["documents", "metadatas"])

            if not res["ids"]:
                if format == "csv":
                    return "id,content,user_id,type,storage_layer,created_time\n", "text/csv", "memories_empty.csv"
                return json.dumps({"memories": [], "exported_at": datetime.now().isoformat()}, ensure_ascii=False), "application/json", "memories_empty.json"

            items = []
            for i in range(min(len(res["ids"]), _EXPORT_MAX_MEMORIES)):
                item = {"id": res["ids"][i], "content": res["documents"][i] if res.get("documents") else ""}
                if res.get("metadatas") and i < len(res["metadatas"]):
                    item.update(res["metadatas"][i])
                items.append(item)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format == "csv":
                return self._memories_to_csv(items), "text/csv", f"memories_{timestamp}.csv"

            export_data = {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "total_count": len(items),
                "filters": {
                    "user_id": user_id,
                    "group_id": group_id,
                    "storage_layer": storage_layer,
                },
                "memories": items,
            }
            return json.dumps(export_data, ensure_ascii=False, indent=2), "application/json", f"memories_{timestamp}.json"

        except Exception as e:
            logger.error(f"Export memories error: {e}")
            return json.dumps({"error": str(e)}), "application/json", "error.json"

    def _memories_to_csv(self, items: List[Dict]) -> str:
        """将记忆列表转为 CSV 字符串"""
        if not items:
            return "id,content\n"

        output = io.StringIO()
        # 收集所有可能的字段
        all_keys = set()
        for item in items:
            all_keys.update(item.keys())

        # 固定字段顺序
        priority_keys = ["id", "content", "user_id", "sender_name", "group_id", "type",
                         "storage_layer", "scope", "confidence", "importance_score", "created_time", "summary"]
        ordered_keys = [k for k in priority_keys if k in all_keys]
        ordered_keys.extend(sorted(all_keys - set(ordered_keys)))

        writer = csv.DictWriter(output, fieldnames=ordered_keys, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            # 确保所有值都是字符串
            row = {}
            for k in ordered_keys:
                v = item.get(k, "")
                if isinstance(v, (list, dict)):
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = str(v) if v is not None else ""
            writer.writerow(row)

        return output.getvalue()

    async def import_memories(
        self,
        data: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """导入记忆数据

        Args:
            data: 文件内容字符串
            format: 格式 'json' 或 'csv'

        Returns:
            {success_count, fail_count, errors, skipped}
        """
        result = {"success_count": 0, "fail_count": 0, "errors": [], "skipped": 0}

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                result["errors"].append("存储服务未就绪")
                return result

            if len(data) > _IMPORT_MAX_FILE_SIZE:
                result["errors"].append(f"文件过大，最大支持 {_IMPORT_MAX_FILE_SIZE // 1024 // 1024}MB")
                return result

            if format == "csv":
                items = self._parse_csv_memories(data)
            else:
                items = self._parse_json_memories(data)

            if not items:
                result["errors"].append("未解析到有效记忆数据")
                return result

            for item in items:
                try:
                    valid, err = self._validate_memory_import(item)
                    if not valid:
                        result["fail_count"] += 1
                        if len(result["errors"]) < 10:
                            result["errors"].append(err)
                        continue

                    # 构建 Memory 对象
                    from iris_memory.models.memory import Memory

                    memory = Memory(
                        id=item.get("id", str(uuid.uuid4())),
                        content=item["content"],
                        user_id=item.get("user_id", "imported"),
                        sender_name=item.get("sender_name", ""),
                        group_id=item.get("group_id"),
                        storage_layer=StorageLayer(item.get("storage_layer", "episodic")),
                        created_time=datetime.fromisoformat(item["created_time"]) if item.get("created_time") else datetime.now(),
                    )

                    # 设置可选字段
                    if item.get("type"):
                        try:
                            memory.type = MemoryType(item["type"])
                        except ValueError:
                            pass
                    if item.get("confidence"):
                        memory.confidence = float(item["confidence"])
                    if item.get("importance_score"):
                        memory.importance_score = float(item["importance_score"])
                    if item.get("summary"):
                        memory.summary = item["summary"]

                    await chroma.add_memory(memory)
                    result["success_count"] += 1

                except Exception as e:
                    result["fail_count"] += 1
                    if len(result["errors"]) < 10:
                        result["errors"].append(f"导入失败: {e}")

        except Exception as e:
            logger.error(f"Import memories error: {e}")
            result["errors"].append(f"导入失败: {e}")

        self._audit_log("import_memories", f"success={result['success_count']} fail={result['fail_count']}")
        return result

    def _parse_json_memories(self, data: str) -> List[Dict]:
        """解析 JSON 格式的记忆数据"""
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "memories" in parsed:
                return parsed["memories"]
            return []
        except json.JSONDecodeError:
            return []

    def _parse_csv_memories(self, data: str) -> List[Dict]:
        """解析 CSV 格式的记忆数据"""
        try:
            reader = csv.DictReader(io.StringIO(data))
            return list(reader)
        except Exception:
            return []

    def _validate_memory_import(self, item: Dict) -> Tuple[bool, str]:
        """验证导入记忆数据的合法性"""
        if not item.get("content"):
            return False, "缺少 content 字段"
        if len(item["content"]) > 10000:
            return False, f"content 过长: {len(item['content'])} 字符"
        if item.get("storage_layer"):
            try:
                StorageLayer(item["storage_layer"])
            except ValueError:
                return False, f"无效的 storage_layer: {item['storage_layer']}"
        if item.get("type"):
            try:
                MemoryType(item["type"])
            except ValueError:
                return False, f"无效的 type: {item['type']}"
        return True, ""

    # ================================================================
    # 数据导入导出 - 知识图谱
    # ================================================================

    async def export_kg(
        self,
        format: str = "json",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """导出知识图谱数据

        Returns:
            (data_string, content_type, filename)
        """
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return json.dumps({"error": "知识图谱未启用"}), "application/json", "error.json"

            storage = kg.storage

            # 获取节点
            async with storage._lock:
                assert storage._conn

                node_sql = "SELECT * FROM kg_nodes"
                edge_sql = "SELECT * FROM kg_edges"
                params: List[Any] = []
                conditions = []

                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if group_id:
                    conditions.append("(group_id = ? OR group_id IS NULL)")
                    params.append(group_id)

                if conditions:
                    where = " WHERE " + " AND ".join(conditions)
                    node_sql += where
                    edge_sql += where

                node_sql += f" LIMIT {_EXPORT_MAX_KG_ITEMS}"
                edge_sql += f" LIMIT {_EXPORT_MAX_KG_ITEMS}"

                node_rows = storage._conn.execute(node_sql, params).fetchall()
                edge_rows = storage._conn.execute(edge_sql, params).fetchall()

            nodes = [dict(r) for r in node_rows]
            edges = [dict(r) for r in edge_rows]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format == "csv":
                nodes_csv = self._kg_nodes_to_csv(nodes)
                edges_csv = self._kg_edges_to_csv(edges)
                # 合并为一个文件，用分隔行区分
                combined = f"# NODES\n{nodes_csv}\n# EDGES\n{edges_csv}"
                return combined, "text/csv", f"knowledge_graph_{timestamp}.csv"

            export_data = {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "nodes_count": len(nodes),
                "edges_count": len(edges),
                "filters": {"user_id": user_id, "group_id": group_id},
                "nodes": nodes,
                "edges": edges,
            }
            return json.dumps(export_data, ensure_ascii=False, indent=2), "application/json", f"knowledge_graph_{timestamp}.json"

        except Exception as e:
            logger.error(f"Export KG error: {e}")
            return json.dumps({"error": str(e)}), "application/json", "error.json"

    def _kg_nodes_to_csv(self, nodes: List[Dict]) -> str:
        """节点列表转 CSV"""
        if not nodes:
            return "id,name,display_name,node_type,user_id,group_id\n"

        output = io.StringIO()
        keys = ["id", "name", "display_name", "node_type", "user_id", "group_id",
                "aliases", "properties", "mention_count", "confidence", "created_time", "updated_time"]
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for node in nodes:
            writer.writerow({k: str(node.get(k, "")) for k in keys})
        return output.getvalue()

    def _kg_edges_to_csv(self, edges: List[Dict]) -> str:
        """边列表转 CSV"""
        if not edges:
            return "id,source_id,target_id,relation_type,relation_label\n"

        output = io.StringIO()
        keys = ["id", "source_id", "target_id", "relation_type", "relation_label",
                "memory_id", "user_id", "group_id", "confidence", "weight",
                "properties", "created_time", "updated_time"]
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for edge in edges:
            writer.writerow({k: str(edge.get(k, "")) for k in keys})
        return output.getvalue()

    async def import_kg(
        self,
        data: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """导入知识图谱数据

        Returns:
            {nodes_imported, edges_imported, errors}
        """
        result = {"nodes_imported": 0, "edges_imported": 0, "errors": []}

        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                result["errors"].append("知识图谱未启用")
                return result

            if len(data) > _IMPORT_MAX_FILE_SIZE:
                result["errors"].append(f"文件过大")
                return result

            if format == "csv":
                nodes_data, edges_data = self._parse_csv_kg(data)
            else:
                nodes_data, edges_data = self._parse_json_kg(data)

            storage = kg.storage

            # 导入节点
            for nd in nodes_data:
                try:
                    valid, err = self._validate_kg_node_import(nd)
                    if not valid:
                        if len(result["errors"]) < 10:
                            result["errors"].append(f"节点: {err}")
                        continue

                    node = KGNode(
                        id=nd.get("id", str(uuid.uuid4())),
                        name=nd.get("name", ""),
                        display_name=nd.get("display_name", nd.get("name", "")),
                        node_type=KGNodeType(nd.get("node_type", "unknown")),
                        user_id=nd.get("user_id", "imported"),
                        group_id=nd.get("group_id"),
                        mention_count=int(nd.get("mention_count", 1)),
                        confidence=float(nd.get("confidence", 0.5)),
                    )

                    if nd.get("aliases"):
                        if isinstance(nd["aliases"], str):
                            try:
                                node.aliases = json.loads(nd["aliases"])
                            except json.JSONDecodeError:
                                node.aliases = [nd["aliases"]]
                        elif isinstance(nd["aliases"], list):
                            node.aliases = nd["aliases"]

                    await storage.upsert_node(node)
                    result["nodes_imported"] += 1

                except Exception as e:
                    if len(result["errors"]) < 10:
                        result["errors"].append(f"节点导入失败: {e}")

            # 导入边
            for ed in edges_data:
                try:
                    valid, err = self._validate_kg_edge_import(ed)
                    if not valid:
                        if len(result["errors"]) < 10:
                            result["errors"].append(f"边: {err}")
                        continue

                    edge = KGEdge(
                        id=ed.get("id", str(uuid.uuid4())),
                        source_id=ed["source_id"],
                        target_id=ed["target_id"],
                        relation_type=KGRelationType(ed.get("relation_type", "related_to")),
                        relation_label=ed.get("relation_label", ""),
                        memory_id=ed.get("memory_id"),
                        user_id=ed.get("user_id", "imported"),
                        group_id=ed.get("group_id"),
                        confidence=float(ed.get("confidence", 0.5)),
                        weight=float(ed.get("weight", 1.0)),
                    )

                    await storage.upsert_edge(edge)
                    result["edges_imported"] += 1

                except Exception as e:
                    if len(result["errors"]) < 10:
                        result["errors"].append(f"边导入失败: {e}")

        except Exception as e:
            logger.error(f"Import KG error: {e}")
            result["errors"].append(f"导入失败: {e}")

        self._audit_log("import_kg", f"nodes={result['nodes_imported']} edges={result['edges_imported']}")
        return result

    def _parse_json_kg(self, data: str) -> Tuple[List[Dict], List[Dict]]:
        """解析 JSON 格式的 KG 数据"""
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return parsed.get("nodes", []), parsed.get("edges", [])
            return [], []
        except json.JSONDecodeError:
            return [], []

    def _parse_csv_kg(self, data: str) -> Tuple[List[Dict], List[Dict]]:
        """解析 CSV 格式的 KG 数据（用 # NODES / # EDGES 分隔）"""
        nodes: List[Dict] = []
        edges: List[Dict] = []

        try:
            sections = data.split("# EDGES")
            nodes_section = sections[0].replace("# NODES", "").strip()
            edges_section = sections[1].strip() if len(sections) > 1 else ""

            if nodes_section:
                nodes = list(csv.DictReader(io.StringIO(nodes_section)))
            if edges_section:
                edges = list(csv.DictReader(io.StringIO(edges_section)))

        except Exception:
            # 尝试作为纯节点 CSV 解析
            try:
                nodes = list(csv.DictReader(io.StringIO(data)))
            except Exception:
                pass

        return nodes, edges

    def _validate_kg_node_import(self, nd: Dict) -> Tuple[bool, str]:
        """验证 KG 节点导入数据"""
        if not nd.get("name"):
            return False, "缺少 name 字段"
        if len(nd["name"]) > 500:
            return False, "name 过长"
        if nd.get("node_type"):
            try:
                KGNodeType(nd["node_type"])
            except ValueError:
                return False, f"无效的 node_type: {nd['node_type']}"
        return True, ""

    def _validate_kg_edge_import(self, ed: Dict) -> Tuple[bool, str]:
        """验证 KG 边导入数据"""
        if not ed.get("source_id"):
            return False, "缺少 source_id"
        if not ed.get("target_id"):
            return False, "缺少 target_id"
        if ed.get("relation_type"):
            try:
                KGRelationType(ed["relation_type"])
            except ValueError:
                return False, f"无效的 relation_type: {ed['relation_type']}"
        return True, ""

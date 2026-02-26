"""数据导入导出服务

封装记忆和知识图谱的导入导出逻辑，
包括 JSON/CSV 解析、序列化、数据验证。
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.knowledge_graph.kg_models import KGEdge, KGNode, KGNodeType, KGRelationType
from iris_memory.web.service.audit import audit_log
from iris_memory.utils.logger import get_logger

logger = get_logger("io_service")

# 导出限制
_EXPORT_MAX_MEMORIES = 10000
_EXPORT_MAX_KG_ITEMS = 50000
_IMPORT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


class IoService:
    """数据导入导出服务"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    # ================================================================
    # 记忆导出
    # ================================================================

    async def export_memories(
        self,
        fmt: str = "json",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """导出记忆数据

        Returns:
            (data_string, content_type, filename)
        """
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return "", "text/plain", "error.txt"

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
                if fmt == "csv":
                    return "id,content,user_id,type,storage_layer,created_time\n", "text/csv", "memories_empty.csv"
                return json.dumps(
                    {"memories": [], "exported_at": datetime.now().isoformat()},
                    ensure_ascii=False,
                ), "application/json", "memories_empty.json"

            items = []
            for i in range(min(len(res["ids"]), _EXPORT_MAX_MEMORIES)):
                item: Dict[str, Any] = {
                    "id": res["ids"][i],
                    "content": res["documents"][i] if res.get("documents") else "",
                }
                if res.get("metadatas") and i < len(res["metadatas"]):
                    item.update(res["metadatas"][i])
                items.append(item)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if fmt == "csv":
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
            return (
                json.dumps(export_data, ensure_ascii=False, indent=2),
                "application/json",
                f"memories_{timestamp}.json",
            )

        except Exception as e:
            logger.error(f"Export memories error: {e}")
            return json.dumps({"error": str(e)}), "application/json", "error.json"

    # ================================================================
    # 记忆导入
    # ================================================================

    async def import_memories(
        self,
        data: str,
        fmt: str = "json",
    ) -> Dict[str, Any]:
        """导入记忆数据

        Returns:
            {success_count, fail_count, errors, skipped}
        """
        result: Dict[str, Any] = {"success_count": 0, "fail_count": 0, "errors": [], "skipped": 0}

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                result["errors"].append("存储服务未就绪")
                return result

            if len(data) > _IMPORT_MAX_FILE_SIZE:
                result["errors"].append(f"文件过大，最大支持 {_IMPORT_MAX_FILE_SIZE // 1024 // 1024}MB")
                return result

            items = self._parse_csv_memories(data) if fmt == "csv" else self._parse_json_memories(data)

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

        audit_log("import_memories", f"success={result['success_count']} fail={result['fail_count']}")
        return result

    # ================================================================
    # KG 导出
    # ================================================================

    async def export_kg(
        self,
        fmt: str = "json",
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

            async with storage._lock:
                assert storage._conn

                node_sql = "SELECT * FROM kg_nodes"
                edge_sql = "SELECT * FROM kg_edges"
                params: List[Any] = []
                conditions: List[str] = []

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

            if fmt == "csv":
                nodes_csv = self._kg_nodes_to_csv(nodes)
                edges_csv = self._kg_edges_to_csv(edges)
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
            return (
                json.dumps(export_data, ensure_ascii=False, indent=2),
                "application/json",
                f"knowledge_graph_{timestamp}.json",
            )

        except Exception as e:
            logger.error(f"Export KG error: {e}")
            return json.dumps({"error": str(e)}), "application/json", "error.json"

    # ================================================================
    # KG 导入
    # ================================================================

    async def import_kg(
        self,
        data: str,
        fmt: str = "json",
    ) -> Dict[str, Any]:
        """导入知识图谱数据

        Returns:
            {nodes_imported, edges_imported, errors}
        """
        result: Dict[str, Any] = {"nodes_imported": 0, "edges_imported": 0, "errors": []}

        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                result["errors"].append("知识图谱未启用")
                return result

            if len(data) > _IMPORT_MAX_FILE_SIZE:
                result["errors"].append("文件过大")
                return result

            if fmt == "csv":
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

        audit_log("import_kg", f"nodes={result['nodes_imported']} edges={result['edges_imported']}")
        return result

    # ================================================================
    # 导入预览
    # ================================================================

    async def preview_import_data(
        self,
        data: str,
        fmt: str,
        import_type: str,
    ) -> Dict[str, Any]:
        """预览导入数据（不实际导入）

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
                items = self._parse_csv_memories(data) if fmt == "csv" else self._parse_json_memories(data)
                result["total"] = len(items)

                for item in items:
                    valid, err = self._validate_memory_import(item)
                    if valid:
                        result["valid"] += 1
                    else:
                        result["invalid"] += 1
                        if len(result["errors"]) < 10:
                            result["errors"].append(err)

                for item in items[:20]:
                    result["preview_items"].append({
                        "content": (item.get("content") or "")[:200],
                        "user_id": item.get("user_id", ""),
                        "type": item.get("type", ""),
                        "storage_layer": item.get("storage_layer", ""),
                    })

            elif import_type == "kg":
                if fmt == "csv":
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
    # CSV 序列化
    # ================================================================

    def _memories_to_csv(self, items: List[Dict]) -> str:
        """将记忆列表转为 CSV 字符串"""
        if not items:
            return "id,content\n"

        output = io.StringIO()
        all_keys: set[str] = set()
        for item in items:
            all_keys.update(item.keys())

        priority_keys = [
            "id", "content", "user_id", "sender_name", "group_id", "type",
            "storage_layer", "scope", "confidence", "importance_score",
            "created_time", "summary",
        ]
        ordered_keys = [k for k in priority_keys if k in all_keys]
        ordered_keys.extend(sorted(all_keys - set(ordered_keys)))

        writer = csv.DictWriter(output, fieldnames=ordered_keys, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            row = {}
            for k in ordered_keys:
                v = item.get(k, "")
                if isinstance(v, (list, dict)):
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = str(v) if v is not None else ""
            writer.writerow(row)

        return output.getvalue()

    def _kg_nodes_to_csv(self, nodes: List[Dict]) -> str:
        """节点列表转 CSV"""
        if not nodes:
            return "id,name,display_name,node_type,user_id,group_id\n"

        output = io.StringIO()
        keys = [
            "id", "name", "display_name", "node_type", "user_id", "group_id",
            "aliases", "properties", "mention_count", "confidence",
            "created_time", "updated_time",
        ]
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
        keys = [
            "id", "source_id", "target_id", "relation_type", "relation_label",
            "memory_id", "user_id", "group_id", "confidence", "weight",
            "properties", "created_time", "updated_time",
        ]
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for edge in edges:
            writer.writerow({k: str(edge.get(k, "")) for k in keys})
        return output.getvalue()

    # ================================================================
    # 解析器
    # ================================================================

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
            try:
                nodes = list(csv.DictReader(io.StringIO(data)))
            except Exception:
                pass

        return nodes, edges

    # ================================================================
    # 验证器
    # ================================================================

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

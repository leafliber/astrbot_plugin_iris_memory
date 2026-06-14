"""数据导入导出服务"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.web.audit import audit_log
from iris_memory.utils.logger import get_logger

logger = get_logger("web.io_svc")

_EXPORT_MAX_MEMORIES = 10000
_EXPORT_MAX_KG_ITEMS = 50000
_EXPORT_MAX_PERSONAS = 10000
_IMPORT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


class IoService:
    """数据导入导出服务"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    # ── 记忆导出 ──

    async def export_memories(
        self,
        fmt: str = "json",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """导出记忆数据，返回 (data_string, content_type, filename)"""
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

    # ── 导出为 iris_chat_memory（新版）导入格式 ──

    async def export_to_iris_chat_memory(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """导出记忆为 iris_chat_memory（新版）导入格式。

        输出符合新版 ``l2_memory/io.py`` 的 ``MemoryExport`` 结构，
        可直接通过新版 Web UI「数据管理 → 导入 L2 记忆」或
        ``MemoryImporter.import_from_file`` 导入。

        字段映射（老版 chroma metadata → 新版 entry.metadata）::

            created_time      → timestamp       （新版时间筛选依赖）
            group_id          → group_id
            user_id           → user_id
            last_access_time  → last_access_time
            access_count      → access_count
            confidence        → confidence
            summarized        → source = "summary" / "tool"
            其余字段原样保留，并标记 migrated_from="iris_memory"
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

            ids = res.get("ids", []) or []
            documents = res.get("documents", []) or []
            metadatas = res.get("metadatas", []) or []

            def _to_int(v: Any, default: int = 0) -> int:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return default

            def _to_float(v: Any, default: float = 0.5) -> float:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            entries: List[Dict[str, Any]] = []
            total = min(len(ids), _EXPORT_MAX_MEMORIES)
            skipped = 0

            for i in range(total):
                content = documents[i] if i < len(documents) else ""
                if not content:
                    skipped += 1
                    continue

                meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
                timestamp = meta.get("created_time") or datetime.now().isoformat()

                entry_meta: Dict[str, Any] = {
                    # 新版标准字段
                    "group_id": meta.get("group_id") or "",
                    "user_id": meta.get("user_id") or "",
                    "timestamp": timestamp,
                    "access_count": _to_int(meta.get("access_count"), 0),
                    "confidence": _to_float(meta.get("confidence"), 0.5),
                    "source": "summary" if meta.get("summarized") else "tool",
                    # 保留的老版信息（新版会原样存入 metadata，便于回溯）
                    "sender_name": meta.get("sender_name") or "",
                    "type": meta.get("type"),
                    "persona_id": meta.get("persona_id") or "default",
                    "original_storage_layer": meta.get("storage_layer"),
                    "importance_score": _to_float(meta.get("importance_score"), 0.5),
                    "rif_score": _to_float(meta.get("rif_score"), 0.5),
                    "migrated_from": "iris_memory",
                }
                last_access = meta.get("last_access_time")
                if last_access:
                    entry_meta["last_access_time"] = last_access
                # 剔除 None，避免新版 metadata 写入 None
                entry_meta = {k: v for k, v in entry_meta.items() if v is not None}

                entries.append({"id": ids[i], "content": content, "metadata": entry_meta})

            exported = len(entries)
            export_data = {
                "version": "1.0",
                "persona_id": "default",
                "export_time": datetime.now().isoformat(),
                "entries": entries,
                "stats": {
                    "total_count": total,
                    "exported_count": exported,
                    "skipped_count": skipped,
                },
            }

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"iris_chat_memory_l2_{timestamp_str}.json"
            logger.info(
                f"Export to iris_chat_memory: total={total} "
                f"exported={exported} skipped={skipped}"
            )
            return (
                json.dumps(export_data, ensure_ascii=False, indent=2),
                "application/json",
                filename,
            )

        except Exception as e:
            logger.error(f"Export to iris_chat_memory error: {e}")
            return json.dumps({"error": str(e)}), "application/json", "error.json"

    # ── 记忆导入 ──

    async def import_memories(self, data: str, fmt: str = "json") -> Dict[str, Any]:
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
                        created_time=(
                            datetime.fromisoformat(item["created_time"])
                            if item.get("created_time")
                            else datetime.now()
                        ),
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

    # ── KG 导出 ──

    async def export_kg(
        self,
        fmt: str = "json",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return json.dumps({"error": "知识图谱未启用"}), "application/json", "error.json"

            storage = kg.storage
            nodes_data: List[Dict[str, Any]] = []
            edges_data: List[Dict[str, Any]] = []

            async with storage._lock:
                assert storage._conn

                sql_nodes = "SELECT * FROM kg_nodes"
                params: List[Any] = []
                conditions: List[str] = []
                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if group_id:
                    conditions.append("(group_id = ? OR group_id IS NULL)")
                    params.append(group_id)
                if conditions:
                    sql_nodes += " WHERE " + " AND ".join(conditions)
                sql_nodes += f" LIMIT {_EXPORT_MAX_KG_ITEMS}"

                for row in storage._conn.execute(sql_nodes, params).fetchall():
                    nodes_data.append(dict(row))

                sql_edges = "SELECT * FROM kg_edges"
                if conditions:
                    sql_edges += " WHERE " + " AND ".join(conditions)
                sql_edges += f" LIMIT {_EXPORT_MAX_KG_ITEMS}"

                for row in storage._conn.execute(sql_edges, params).fetchall():
                    edges_data.append(dict(row))

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if fmt == "csv":
                return self._kg_to_csv(nodes_data, edges_data), "text/csv", f"kg_{timestamp}.csv"

            export_data = {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "nodes": nodes_data,
                "edges": edges_data,
            }
            return (
                json.dumps(export_data, ensure_ascii=False, indent=2, default=str),
                "application/json",
                f"kg_{timestamp}.json",
            )

        except Exception as e:
            logger.error(f"Export KG error: {e}")
            return json.dumps({"error": str(e)}), "application/json", "error.json"

    # ── KG 导入 ──

    async def import_kg(self, data: str, fmt: str = "json") -> Dict[str, Any]:
        result: Dict[str, Any] = {"success_count": 0, "fail_count": 0, "errors": []}

        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                result["errors"].append("知识图谱未启用")
                return result

            if len(data) > _IMPORT_MAX_FILE_SIZE:
                result["errors"].append(f"文件过大，最大支持 {_IMPORT_MAX_FILE_SIZE // 1024 // 1024}MB")
                return result

            parsed = json.loads(data) if isinstance(data, str) else data
            nodes = parsed.get("nodes", [])
            edges = parsed.get("edges", [])

            from iris_memory.knowledge_graph.kg_models import KGNode, KGEdge, KGNodeType, KGRelationType

            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                with storage._tx() as cur:
                    for node_data in nodes:
                        try:
                            cur.execute(
                                "INSERT OR REPLACE INTO kg_nodes (id, name, display_name, node_type, user_id, group_id, created_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (
                                    node_data.get("id", str(uuid.uuid4())),
                                    node_data.get("name", ""),
                                    node_data.get("display_name", ""),
                                    node_data.get("node_type", "concept"),
                                    node_data.get("user_id", "imported"),
                                    node_data.get("group_id"),
                                    node_data.get("created_time", datetime.now().isoformat()),
                                ),
                            )
                            result["success_count"] += 1
                        except Exception as e:
                            result["fail_count"] += 1
                            if len(result["errors"]) < 10:
                                result["errors"].append(f"节点导入失败: {e}")

                    for edge_data in edges:
                        try:
                            cur.execute(
                                "INSERT OR REPLACE INTO kg_edges (id, source_id, target_id, relation_type, user_id, group_id, created_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (
                                    edge_data.get("id", str(uuid.uuid4())),
                                    edge_data.get("source_id", ""),
                                    edge_data.get("target_id", ""),
                                    edge_data.get("relation_type", "related_to"),
                                    edge_data.get("user_id", "imported"),
                                    edge_data.get("group_id"),
                                    edge_data.get("created_time", datetime.now().isoformat()),
                                ),
                            )
                            result["success_count"] += 1
                        except Exception as e:
                            result["fail_count"] += 1
                            if len(result["errors"]) < 10:
                                result["errors"].append(f"边导入失败: {e}")

                storage._invalidate_cache()

        except json.JSONDecodeError as e:
            result["errors"].append(f"JSON 解析失败: {e}")
        except Exception as e:
            logger.error(f"Import KG error: {e}")
            result["errors"].append(f"导入失败: {e}")

        audit_log("import_kg", f"success={result['success_count']} fail={result['fail_count']}")
        return result

    # ── 画像导出 ──

    async def export_personas(
        self,
        fmt: str = "json",
        user_id: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """导出用户画像数据，返回 (data_string, content_type, filename)"""
        try:
            personas = self._service._user_personas
            if not personas:
                if fmt == "csv":
                    return "user_id,display_name,trust_level,intimacy_level,emotional_baseline,last_updated\n", "text/csv", "personas_empty.csv"
                return json.dumps(
                    {"personas": [], "exported_at": datetime.now().isoformat()},
                    ensure_ascii=False,
                ), "application/json", "personas_empty.json"

            items = []
            for uid, persona in personas.items():
                if user_id and uid != user_id:
                    continue
                try:
                    items.append({
                        "user_id": uid,
                        "persona_data": persona.to_dict(),
                    })
                except Exception as e:
                    logger.warning(f"Export persona {uid} failed: {e}")
                    continue

                if len(items) >= _EXPORT_MAX_PERSONAS:
                    break

            if not items:
                if fmt == "csv":
                    return "user_id,display_name,trust_level,intimacy_level,emotional_baseline,last_updated\n", "text/csv", "personas_empty.csv"
                return json.dumps(
                    {"personas": [], "exported_at": datetime.now().isoformat()},
                    ensure_ascii=False,
                ), "application/json", "personas_empty.json"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if fmt == "csv":
                return self._personas_to_csv(items), "text/csv", f"personas_{timestamp}.csv"

            export_data = {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "total_count": len(items),
                "filters": {"user_id": user_id},
                "personas": items,
            }
            return (
                json.dumps(export_data, ensure_ascii=False, indent=2, default=str),
                "application/json",
                f"personas_{timestamp}.json",
            )

        except Exception as e:
            logger.error(f"Export personas error: {e}")
            return json.dumps({"error": str(e)}), "application/json", "error.json"

    # ── 画像导入 ──

    async def import_personas(self, data: str, fmt: str = "json") -> Dict[str, Any]:
        result: Dict[str, Any] = {"success_count": 0, "fail_count": 0, "errors": [], "skipped": 0}

        try:
            if len(data) > _IMPORT_MAX_FILE_SIZE:
                result["errors"].append(f"文件过大，最大支持 {_IMPORT_MAX_FILE_SIZE // 1024 // 1024}MB")
                return result

            items = self._parse_csv_personas(data) if fmt == "csv" else self._parse_json_personas(data)

            if not items:
                result["errors"].append("未解析到有效画像数据")
                return result

            from iris_memory.models.user_persona import UserPersona

            personas = self._service._user_personas

            for item in items:
                try:
                    user_id = item.get("user_id")
                    persona_data = item.get("persona_data", item)

                    if not user_id:
                        result["skipped"] += 1
                        continue

                    valid, err = self._validate_persona_import(persona_data)
                    if not valid:
                        result["fail_count"] += 1
                        if len(result["errors"]) < 10:
                            result["errors"].append(f"{user_id}: {err}")
                        continue

                    persona = UserPersona.from_dict(persona_data)
                    persona.user_id = user_id

                    if user_id in personas:
                        existing = personas[user_id]
                        if persona.update_count <= existing.update_count:
                            result["skipped"] += 1
                            continue

                    personas[user_id] = persona
                    result["success_count"] += 1

                except Exception as e:
                    result["fail_count"] += 1
                    if len(result["errors"]) < 10:
                        result["errors"].append(f"导入失败: {e}")

        except Exception as e:
            logger.error(f"Import personas error: {e}")
            result["errors"].append(f"导入失败: {e}")

        audit_log("import_personas", f"success={result['success_count']} fail={result['fail_count']} skipped={result['skipped']}")
        return result

    # ── 导入预览 ──

    async def preview_import_data(
        self,
        data: str,
        fmt: str = "json",
        import_type: str = "memories",
    ) -> Dict[str, Any]:
        try:
            if import_type == "memories":
                items = self._parse_csv_memories(data) if fmt == "csv" else self._parse_json_memories(data)
                return {
                    "type": "memories",
                    "total": len(items),
                    "preview": items[:10],
                    "fields": list(items[0].keys()) if items else [],
                }
            elif import_type == "personas":
                items = self._parse_csv_personas(data) if fmt == "csv" else self._parse_json_personas(data)
                return {
                    "type": "personas",
                    "total": len(items),
                    "preview": items[:10],
                    "fields": list(items[0].keys()) if items else [],
                }
            else:
                parsed = json.loads(data) if isinstance(data, str) else data
                nodes = parsed.get("nodes", [])
                edges = parsed.get("edges", [])
                return {
                    "type": "kg",
                    "nodes_count": len(nodes),
                    "edges_count": len(edges),
                    "preview_nodes": nodes[:5],
                    "preview_edges": edges[:5],
                }
        except Exception as e:
            return {"error": str(e)}

    # ── 内部方法 ──

    @staticmethod
    def _memories_to_csv(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "id,content,user_id,type,storage_layer,created_time\n"

        output = io.StringIO()
        fieldnames = ["id", "content", "user_id", "group_id", "sender_name", "type", "storage_layer", "confidence", "importance_score", "created_time", "summary"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            writer.writerow(item)
        return output.getvalue()

    @staticmethod
    def _kg_to_csv(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
        output = io.StringIO()
        output.write("# NODES\n")
        if nodes:
            writer = csv.DictWriter(output, fieldnames=list(nodes[0].keys()))
            writer.writeheader()
            for n in nodes:
                writer.writerow(n)
        output.write("\n# EDGES\n")
        if edges:
            writer = csv.DictWriter(output, fieldnames=list(edges[0].keys()))
            writer.writeheader()
            for e in edges:
                writer.writerow(e)
        return output.getvalue()

    @staticmethod
    def _parse_json_memories(data: str) -> List[Dict[str, Any]]:
        try:
            parsed = json.loads(data) if isinstance(data, str) else data
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return parsed.get("memories", [])
            return []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _parse_csv_memories(data: str) -> List[Dict[str, Any]]:
        try:
            reader = csv.DictReader(io.StringIO(data))
            return [dict(row) for row in reader]
        except Exception:
            return []

    @staticmethod
    def _validate_memory_import(item: Dict[str, Any]) -> Tuple[bool, str]:
        if not item.get("content"):
            return False, "缺少 content 字段"
        if len(item["content"]) > 10000:
            return False, "content 长度超过 10000"
        return True, ""

    @staticmethod
    def _personas_to_csv(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "user_id,display_name,trust_level,intimacy_level,emotional_baseline,last_updated\n"

        output = io.StringIO()
        fieldnames = ["user_id", "display_name", "trust_level", "intimacy_level", "emotional_baseline", "emotional_volatility", "work_style", "lifestyle", "social_style", "proactive_reply_preference", "last_updated", "update_count"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            persona_data = item.get("persona_data", item)
            row = {"user_id": item.get("user_id", "")}
            row.update(persona_data)
            writer.writerow(row)
        return output.getvalue()

    @staticmethod
    def _parse_json_personas(data: str) -> List[Dict[str, Any]]:
        try:
            parsed = json.loads(data) if isinstance(data, str) else data
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return parsed.get("personas", [])
            return []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _parse_csv_personas(data: str) -> List[Dict[str, Any]]:
        try:
            reader = csv.DictReader(io.StringIO(data))
            items = []
            for row in reader:
                user_id = row.get("user_id", "")
                if not user_id:
                    continue
                persona_data = dict(row)
                persona_data.pop("user_id", None)
                for key in ["trust_level", "intimacy_level", "emotional_volatility", "proactive_reply_preference"]:
                    if key in persona_data:
                        try:
                            persona_data[key] = float(persona_data[key])
                        except (ValueError, TypeError):
                            pass
                for key in ["update_count"]:
                    if key in persona_data:
                        try:
                            persona_data[key] = int(persona_data[key])
                        except (ValueError, TypeError):
                            pass
                items.append({"user_id": user_id, "persona_data": persona_data})
            return items
        except Exception:
            return []

    @staticmethod
    def _validate_persona_import(item: Dict[str, Any]) -> Tuple[bool, str]:
        if not item.get("user_id"):
            return False, "缺少 user_id 字段"
        return True, ""

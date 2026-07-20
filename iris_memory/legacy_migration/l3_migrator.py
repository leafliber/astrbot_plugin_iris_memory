"""
旧版（v2.x）知识图谱（knowledge_graph.db）→ 新版 L3（SQLite 图谱）迁移器

节点映射（kg_nodes → nodes / GraphNode）：
    node_type       → label（经 NODE_TYPE_MAP 映射到新版白名单类型，
                      未收录的类型首字母大写作为动态 label）
    display_name    → name（缺省回退 name）
    properties.description / name → content（旧库无描述字段）
    confidence      → confidence
    mention_count   → access_count
    created_time    → created_time
    updated_time    → last_access_time
    group_id        → group_id
    id              → properties.legacy_node_id（新 id 按 label+name 哈希重新生成，
                      使后续梦境提取的同名实体能与迁移节点合并去重）
    user_id/persona_id → properties.legacy_user_id / legacy_persona_id
                      （新版 nodes 表无此列）
    aliases/properties → 合并入 properties（值统一转为字符串）

边映射（kg_edges → edges / GraphEdge）：
    source_id/target_id → 经旧→新节点 id 映射表重指向；端点缺失的边跳过
    relation_type   → 经 RELATION_TYPE_MAP 映射到新版白名单关系，
                      未收录的归入 RELATED_TO，原类型保留在
                      properties.legacy_relation_type
    relation_label  → properties.relation_label
    memory_id       → properties.legacy_memory_id（旧记忆 id 在新 L2 中
                      已重新生成，直接引用必然悬空，故不写入 source_memory_id）
    weight/confidence → weight/confidence
    created_time    → created_time
    updated_time    → last_access_time
    user_id/persona_id → properties.legacy_user_id / legacy_persona_id
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from iris_memory.core import get_logger
from iris_memory.l3_kg.models import GraphEdge, GraphNode

logger = get_logger("legacy_migration")

#: 每批读取的行数
BATCH_SIZE = 500

#: 旧节点类型 → 新节点 label（白名单）
NODE_TYPE_MAP: Dict[str, str] = {
    "person": "Person",
    "location": "Location",
    "organization": "Group",
    "object": "Item",
    "event": "Event",
    "concept": "Concept",
    "time": "Concept",
    "unknown": "Concept",
}

#: 旧关系类型 → 新关系类型（白名单）
RELATION_TYPE_MAP: Dict[str, str] = {
    # 人际
    "friend_of": "KNOWS",
    "colleague_of": "KNOWS",
    "family_of": "KNOWS",
    "boss_of": "KNOWS",
    "subordinate_of": "KNOWS",
    "knows": "KNOWS",
    # 偏好
    "likes": "HAS_PREFERENCE",
    "dislikes": "HAS_PREFERENCE",
    # 属性
    "is": "HAS_TRAIT",
    "wants": "HAS_GOAL",
    "lives_in": "LOCATED_AT",
    "works_at": "PART_OF",
    "studies_at": "PART_OF",
    "belongs_to": "PART_OF",
    # 事件
    "participated_in": "PARTICIPATED_IN",
    "happened_at": "HAPPENED_AT",
    "caused_by": "LEADS_TO",
    # 通用
    "does": "RELATED_TO",
    "has": "RELATED_TO",
    "owns": "RELATED_TO",
    "related_to": "RELATED_TO",
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_dt(value: Any) -> Optional[datetime]:
    """解析 ISO 时间字符串为 datetime；失败返回 None"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _parse_json_obj(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _stringify_properties(props: Dict[str, Any]) -> Dict[str, str]:
    """properties 值统一转字符串（新模型 properties 为 dict[str, str]）"""
    result: Dict[str, str] = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, str):
            result[str(key)] = value
        else:
            result[str(key)] = json.dumps(value, ensure_ascii=False)
    return result


def _map_label(legacy_type: str) -> str:
    mapped = NODE_TYPE_MAP.get(legacy_type.lower())
    if mapped:
        return mapped
    # 未收录类型：首字母大写作为动态 label（新版支持动态 label）
    return legacy_type.strip().capitalize() or "Concept"


def map_legacy_node(row: Dict[str, Any]) -> Optional[GraphNode]:
    """旧 kg_nodes 行 → 新 GraphNode；名称缺失返回 None"""
    name = (row.get("display_name") or row.get("name") or "").strip()
    if not name:
        return None

    legacy_type = (row.get("node_type") or "unknown").strip()
    label = _map_label(legacy_type)

    old_props = _parse_json_obj(row.get("properties"))
    description = old_props.get("description")
    content = description if isinstance(description, str) and description else name

    properties = _stringify_properties(old_props)
    properties["legacy_node_id"] = str(row.get("id") or "")
    properties["legacy_node_type"] = legacy_type
    if row.get("user_id"):
        properties["legacy_user_id"] = str(row["user_id"])
    if row.get("persona_id") and row["persona_id"] != "default":
        properties["legacy_persona_id"] = str(row["persona_id"])
    aliases = row.get("aliases")
    if isinstance(aliases, str) and aliases and aliases != "[]":
        properties["aliases"] = aliases
    properties["migrated_from"] = "iris_memory_v2"

    node = GraphNode(
        id="",
        label=label,
        name=name,
        content=content,
        confidence=_to_float(row.get("confidence"), 0.5),
        access_count=_to_int(row.get("mention_count"), 0),
        last_access_time=_parse_dt(row.get("updated_time")),
        created_time=_parse_dt(row.get("created_time")) or datetime.now(),
        source_memory_id=None,
        group_id=row.get("group_id") or None,
        properties=properties,
    )
    node.id = node.generate_id()
    return node


def map_legacy_edge(
    row: Dict[str, Any], id_map: Dict[str, str]
) -> Tuple[Optional[GraphEdge], bool]:
    """旧 kg_edges 行 → 新 GraphEdge

    Returns:
        (边对象, 是否悬空)：端点不在 id_map 中返回 (None, True)，
        其他失败返回 (None, False)
    """
    old_source = str(row.get("source_id") or "")
    old_target = str(row.get("target_id") or "")
    new_source = id_map.get(old_source)
    new_target = id_map.get(old_target)
    if not new_source or not new_target:
        return None, True

    legacy_rel = (row.get("relation_type") or "related_to").strip()
    relation_type = RELATION_TYPE_MAP.get(legacy_rel.lower(), "RELATED_TO")

    old_props = _parse_json_obj(row.get("properties"))
    properties = _stringify_properties(old_props)
    properties["legacy_relation_type"] = legacy_rel
    if row.get("relation_label"):
        properties["relation_label"] = str(row["relation_label"])
    if row.get("memory_id"):
        properties["legacy_memory_id"] = str(row["memory_id"])
    if row.get("user_id"):
        properties["legacy_user_id"] = str(row["user_id"])
    if row.get("persona_id") and row["persona_id"] != "default":
        properties["legacy_persona_id"] = str(row["persona_id"])
    if legacy_rel.lower() == "dislikes":
        properties["polarity"] = "dislike"
    properties["migrated_from"] = "iris_memory_v2"

    edge = GraphEdge(
        source_id=new_source,
        target_id=new_target,
        relation_type=relation_type,
        weight=_to_float(row.get("weight"), 1.0),
        confidence=_to_float(row.get("confidence"), 0.5),
        access_count=0,
        last_access_time=_parse_dt(row.get("updated_time")),
        created_time=_parse_dt(row.get("created_time")) or datetime.now(),
        source_memory_id=None,
        properties=properties,
    )
    return edge, False


def _open_legacy_db(db_path: Any) -> sqlite3.Connection:
    """打开旧图谱数据库：优先只读模式，失败回退普通连接（只执行 SELECT）"""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        logger.warning(f"只读模式打开 {db_path} 失败，回退普通连接（仅读取）")
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _get_available_adapter(component_manager: Any) -> Optional[Any]:
    try:
        adapter = component_manager.get_component("l3_kg")
    except Exception as e:
        logger.warning(f"获取 L3 组件失败：{e}")
        return None
    if adapter is None or not getattr(adapter, "is_available", False):
        return None
    return adapter


async def migrate_l3(detection: Any, component_manager: Any) -> Dict[str, Any]:
    """迁移旧知识图谱到新版 L3

    Args:
        detection: LegacyDetection 检测结果
        component_manager: 组件管理器

    Returns:
        统计信息
    """
    stats: Dict[str, Any] = {
        "status": "ok",
        "nodes_total": 0,
        "nodes_imported": 0,
        "edges_total": 0,
        "edges_imported": 0,
        "edges_dangling": 0,
        "errors": 0,
    }

    if detection.kg_db_path is None:
        stats["status"] = "skipped_no_data"
        return stats

    adapter = _get_available_adapter(component_manager)
    if adapter is None:
        stats["status"] = "skipped_adapter_unavailable"
        logger.warning("L3 知识图谱组件不可用，跳过旧图谱迁移")
        return stats

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _open_legacy_db(detection.kg_db_path)

        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        if "kg_nodes" not in tables:
            stats["status"] = "skipped_no_tables"
            logger.warning("旧数据库中未找到 kg_nodes 表，跳过图谱迁移")
            return stats

        if "kg_meta" in tables:
            try:
                row = conn.execute(
                    "SELECT value FROM kg_meta WHERE key = 'schema_version'"
                ).fetchone()
                if row:
                    logger.info(f"旧图谱 schema 版本：{row[0]}")
            except Exception:
                pass

        # ── 节点迁移 ──
        id_map: Dict[str, str] = {}
        offset = 0
        while True:
            rows = conn.execute(
                "SELECT * FROM kg_nodes ORDER BY rowid LIMIT ? OFFSET ?",
                (BATCH_SIZE, offset),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                stats["nodes_total"] += 1
                row_dict = dict(row)
                try:
                    node = map_legacy_node(row_dict)
                    if node is None:
                        stats["errors"] += 1
                        continue
                    # 同名节点坍缩到同一新 id，由 add_node 的合并逻辑去重
                    id_map[str(row_dict.get("id") or "")] = node.id
                    if await adapter.add_node(node):
                        stats["nodes_imported"] += 1
                    else:
                        stats["errors"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    logger.warning(f"迁移节点失败（{row_dict.get('id')}）：{e}")
            offset += len(rows)
            logger.info(
                f"L3 节点迁移进度：{offset} 已扫描，"
                f"导入 {stats['nodes_imported']}，错误 {stats['errors']}"
            )

        # ── 边迁移 ──
        offset = 0
        while True:
            rows = conn.execute(
                "SELECT * FROM kg_edges ORDER BY rowid LIMIT ? OFFSET ?",
                (BATCH_SIZE, offset),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                stats["edges_total"] += 1
                row_dict = dict(row)
                try:
                    edge, dangling = map_legacy_edge(row_dict, id_map)
                    if dangling:
                        stats["edges_dangling"] += 1
                        continue
                    if edge is None:
                        stats["errors"] += 1
                        continue
                    if await adapter.add_edge(edge):
                        stats["edges_imported"] += 1
                    else:
                        stats["errors"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    logger.warning(f"迁移边失败（{row_dict.get('id')}）：{e}")
            offset += len(rows)
            logger.info(
                f"L3 边迁移进度：{offset} 已扫描，"
                f"导入 {stats['edges_imported']}，悬空 {stats['edges_dangling']}"
            )

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        logger.error(f"旧知识图谱迁移失败：{e}", exc_info=True)
        return stats
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    logger.info(
        f"L3 迁移完成：节点 {stats['nodes_imported']}/{stats['nodes_total']}，"
        f"边 {stats['edges_imported']}/{stats['edges_total']}"
        f"（悬空跳过 {stats['edges_dangling']}），错误 {stats['errors']}"
    )
    return stats

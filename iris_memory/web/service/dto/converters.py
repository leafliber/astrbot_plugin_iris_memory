"""数据转换器（DTO Converters）

将内部模型转换为 Web 前端展示所需的字典格式。
集中管理所有序列化逻辑，避免各处重复。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from iris_memory.knowledge_graph.kg_models import KGEdge, KGNode


def memory_to_web_dict(memory: Any) -> Dict[str, Any]:
    """将 Memory 对象转换为前端展示字典（列表视图，content 可截断）"""
    return {
        "id": getattr(memory, "id", ""),
        "content": getattr(memory, "content", ""),
        "summary": getattr(memory, "summary", ""),
        "user_id": getattr(memory, "user_id", ""),
        "sender_name": getattr(memory, "sender_name", ""),
        "group_id": getattr(memory, "group_id", ""),
        "type": _enum_val(getattr(memory, "type", "")),
        "storage_layer": _enum_val(getattr(memory, "storage_layer", "")),
        "scope": _enum_val(getattr(memory, "scope", "")),
        "confidence": getattr(memory, "confidence", 0),
        "importance_score": getattr(memory, "importance_score", 0),
        "created_time": _isoformat(getattr(memory, "created_time", "")),
        "keywords": getattr(memory, "keywords", []),
    }


def memory_detail_from_chroma(
    res: Dict[str, Any],
    index: int,
    *,
    full: bool = False,
) -> Dict[str, Any]:
    """从 ChromaDB 结果构建记忆字典

    Args:
        res: ChromaDB collection.get() 的返回结果
        index: 结果中的索引
        full: 是否包含完整字段（详情视图）

    Returns:
        记忆字典
    """
    meta = res["metadatas"][index] if res.get("metadatas") and index < len(res["metadatas"]) else {}
    item: Dict[str, Any] = {
        "id": res["ids"][index],
        "content": res["documents"][index] if res.get("documents") else "",
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
    }
    if full:
        item.update({
            "keywords": meta.get("keywords", ""),
            "quality_level": meta.get("quality_level", ""),
            "access_count": meta.get("access_count", 0),
            "rif_score": meta.get("rif_score", 0),
        })
    return item


def node_to_web_dict(node: KGNode) -> Dict[str, Any]:
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
        "created_time": _isoformat(node.created_time),
    }


def node_to_graph_dict(node: KGNode) -> Dict[str, Any]:
    """节点转图谱可视化字典"""
    return {
        "id": node.id,
        "label": node.display_name or node.name,
        "type": node.node_type.value,
        "size": min(30, 10 + node.mention_count * 2),
        "confidence": node.confidence,
    }


def edge_to_web_dict(
    edge: KGEdge,
    node_names: Dict[str, str],
) -> Dict[str, Any]:
    """边转前端展示字典（含源/目标节点名称）"""
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "source_name": node_names.get(edge.source_id, edge.source_id),
        "target_name": node_names.get(edge.target_id, edge.target_id),
        "relation_type": edge.relation_type.value,
        "relation_label": edge.relation_label or edge.relation_type.value,
        "user_id": edge.user_id,
        "group_id": edge.group_id,
        "confidence": edge.confidence,
        "weight": edge.weight,
        "created_time": _isoformat(edge.created_time),
    }


def edge_to_graph_dict(edge: KGEdge) -> Dict[str, Any]:
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


# ── 内部工具 ──


def _enum_val(obj: Any) -> str:
    """安全提取枚举 value"""
    if hasattr(obj, "value"):
        return obj.value
    return str(obj) if obj else ""


def _isoformat(obj: Any) -> str:
    """安全转 ISO 格式时间字符串"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj) if obj else ""

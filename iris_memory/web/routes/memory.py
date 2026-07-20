"""
记忆相关 API 路由

提供L1/L2/L3三层记忆的访问接口：
- L1 Buffer: 消息缓冲列表
- L2 Memory: 记忆搜索
- L3 KG: 知识图谱数据
"""

from quart import jsonify, request
from iris_memory.core import get_component_manager, get_logger
from iris_memory.l1_buffer.buffer import L1Buffer
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.platform.base import PRIVATE_SESSION_PREFIX
from iris_memory.profile.storage import ProfileStorage

logger = get_logger("web.memory")

PLUGIN_NAME = "astrbot_plugin_iris_memory"


def get_private_queue_display(l1_buffer: L1Buffer, queue_key: str) -> dict:
    """提取私聊队列的展示信息（用户 ID 与昵称）

    昵称取自队列中最近一条携带 user_name 元数据的用户消息，
    无需访问数据库；无昵称记录时返回空字符串，由前端回退显示用户 ID。

    Args:
        l1_buffer: L1 Buffer 实例
        queue_key: 私聊队列键（private:{user_id}）

    Returns:
        {"is_private": True, "user_id": str, "group_name": str}
    """
    user_id = queue_key[len(PRIVATE_SESSION_PREFIX) :]
    user_name = ""
    for msg in reversed(l1_buffer.get_context(queue_key)):
        if msg.role == "user" and msg.metadata:
            name = msg.metadata.get("user_name")
            if name:
                user_name = name
                break
    return {"is_private": True, "user_id": user_id, "group_name": user_name}


async def search_l2_memory():
    data = await request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求正文为空或格式错误"}), 400
    query = data.get("query", "")
    group_id = data.get("group_id")
    top_k = data.get("top_k", 10)
    persona_id = data.get("persona", request.args.get("persona", "default"))

    if not query:
        return jsonify({"success": False, "error": "搜索关键词不能为空"}), 400

    if not isinstance(top_k, int) or top_k < 1:
        top_k = 10
    top_k = min(top_k, 100)

    manager = get_component_manager()
    l2_retriever = manager.get_component("l2_memory", L2MemoryAdapter)

    if not l2_retriever or not l2_retriever.is_available:
        return jsonify({"success": False, "error": "L2 记忆库不可用"}), 503

    results = await l2_retriever.retrieve(query, group_id, top_k, persona_id)

    formatted_results = [
        {
            "id": r.entry.id,
            "content": r.entry.content,
            "score": r.score,
            "metadata": r.entry.metadata,
            "timestamp": r.entry.metadata.get("timestamp"),
            "access_count": r.entry.metadata.get("access_count", 0),
            "last_access_time": r.entry.metadata.get("last_access_time"),
            "confidence": r.entry.metadata.get("confidence", 0.5),
            "source": r.entry.metadata.get("source"),
            "group_id": r.entry.metadata.get("group_id"),
        }
        for r in results
    ]

    logger.info(f"搜索L2记忆成功：查询='{query[:20]}...', 结果数={len(results)}")

    return jsonify({"success": True, "results": formatted_results})


async def get_latest_l2_memories():
    limit = request.args.get("limit", default=20, type=int)
    offset = request.args.get("offset", default=0, type=int)
    group_id = request.args.get("group_id")
    persona_id = request.args.get("persona", "default")
    sort_by = request.args.get("sort_by", default="timestamp")
    sort_order = request.args.get("sort_order", default="desc")

    valid_limits = [10, 20, 50, 100]
    if limit not in valid_limits:
        limit = 20

    if offset < 0:
        offset = 0

    valid_sort_fields = [
        "timestamp",
        "access_count",
        "confidence",
        "last_access_time",
    ]
    if sort_by not in valid_sort_fields:
        sort_by = "timestamp"

    if sort_order not in ("asc", "desc"):
        sort_order = "desc"

    manager = get_component_manager()
    l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)

    if not l2_adapter or not l2_adapter.is_available:
        return jsonify({"success": False, "error": "L2 记忆库不可用"}), 503

    if group_id:
        all_entries = await l2_adapter.get_entries_by_group(group_id, persona_id)
    else:
        all_entries = await l2_adapter.get_all_entries(persona_id)

    raw_entries = []
    for entry in all_entries:
        meta = entry.metadata
        raw_entries.append(
            {
                "id": entry.id,
                "content": entry.content,
                "score": 1.0,
                "metadata": meta,
                "timestamp": meta.get("timestamp"),
                "access_count": meta.get("access_count", 0),
                "last_access_time": meta.get("last_access_time"),
                "confidence": meta.get("confidence", 0.5),
                "source": meta.get("source"),
                "group_id": meta.get("group_id"),
            }
        )

    def sort_key(entry):
        val = entry.get(sort_by)
        if val is None:
            if sort_by in ("access_count", "confidence"):
                val = 0
            else:
                val = ""
        if sort_by in ("access_count", "confidence") and isinstance(val, str):
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0
        return val

    raw_entries.sort(key=sort_key, reverse=(sort_order == "desc"))

    total_count = len(raw_entries)
    formatted_results = raw_entries[offset : offset + limit]

    logger.info(
        f"获取最新L2记忆成功：limit={limit}, offset={offset}, sort_by={sort_by}, "
        f"sort_order={sort_order}, 总数={total_count}, 返回={len(formatted_results)}"
    )

    return jsonify(
        {
            "success": True,
            "results": formatted_results,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
        }
    )


async def list_l1_buffer():
    group_id = request.args.get("group_id")

    manager = get_component_manager()
    l1_buffer = manager.get_component("l1_buffer", L1Buffer)

    if not l1_buffer or not l1_buffer.is_available:
        return jsonify({"success": False, "error": "L1 缓冲不可用"}), 503

    messages = l1_buffer.get_context(group_id)

    formatted_messages = [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            "user_id": msg.source if hasattr(msg, "source") else None,
            "user_name": msg.metadata.get("user_name")
            if hasattr(msg, "metadata") and msg.metadata
            else None,
        }
        for msg in messages
    ]

    logger.info(f"获取L1缓冲成功：群聊={group_id}, 消息数={len(messages)}")

    return jsonify(
        {
            "success": True,
            "messages": formatted_messages,
            "count": len(formatted_messages),
        }
    )


async def list_l1_queues():
    manager = get_component_manager()
    l1_buffer = manager.get_component("l1_buffer", L1Buffer)

    if not l1_buffer or not l1_buffer.is_available:
        return jsonify({"success": False, "error": "L1 缓冲不可用"}), 503

    queues = l1_buffer.get_all_queues_stats()

    # 私聊队列（private:{user_id}）单独补充展示信息：用户 ID 与昵称
    for q in queues:
        gid = q.get("group_id", "")
        if gid.startswith(PRIVATE_SESSION_PREFIX):
            q.update(get_private_queue_display(l1_buffer, gid))

    group_names: dict[str, str] = {}
    profile_storage = manager.get_component("profile", ProfileStorage)
    if profile_storage and profile_storage.is_available:
        try:
            from iris_memory.profile import GroupProfileManager

            group_manager = GroupProfileManager(profile_storage)
            for q in queues:
                gid = q.get("group_id", "")
                if (
                    gid
                    and not gid.startswith(PRIVATE_SESSION_PREFIX)
                    and gid not in group_names
                ):
                    try:
                        profile = await group_manager._storage.get_group_profile(gid)
                        group_names[gid] = (
                            profile.group_name if profile and profile.group_name else ""
                        )
                    except Exception:
                        group_names[gid] = ""
        except Exception as e:
            logger.debug(f"获取群聊名称失败: {e}")

    for q in queues:
        gid = q.get("group_id", "")
        if not gid.startswith(PRIVATE_SESSION_PREFIX):
            q["group_name"] = group_names.get(gid, "")

    return jsonify({"success": True, "queues": queues})


async def get_l3_graph():
    node_id = request.args.get("node_id")
    depth = request.args.get("depth", default=1, type=int)
    max_nodes = request.args.get("max_nodes", default=20, type=int)
    max_edges = request.args.get("max_edges", default=100, type=int)
    group_id = request.args.get("group_id")

    depth = max(1, min(depth, 5))
    max_nodes = max(1, min(max_nodes, 500))
    max_edges = max(1, min(max_edges, 1000))

    manager = get_component_manager()
    l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

    if not l3_adapter or not l3_adapter.is_available:
        return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

    if not node_id:
        random_node = await l3_adapter.get_random_person_node()
        if random_node:
            node_id = random_node["id"]
        else:
            return jsonify(
                {
                    "success": True,
                    "start_node": None,
                    "nodes": [],
                    "edges": [],
                    "message": "图谱中没有 Person 类型节点",
                }
            )

    nodes, edges = await l3_adapter.expand_from_node(
        node_id=node_id,
        depth=depth,
        max_nodes=max_nodes,
        max_edges=max_edges,
        group_id=group_id,
    )

    start_node = None
    for node in nodes:
        if node["id"] == node_id:
            start_node = node
            break

    formatted_nodes = [
        {
            "id": node["id"],
            "label": node.get("label", "Entity"),
            "name": node.get("name", node["id"]),
            "content": node.get("content", ""),
            "confidence": node.get("confidence", 0.5),
            "access_count": node.get("access_count", 0),
            "last_access_time": node.get("last_access_time"),
            "created_time": node.get("created_time"),
            "source_memory_id": node.get("source_memory_id"),
            "group_id": node.get("group_id"),
            "properties": node.get("properties", {}),
        }
        for node in nodes
    ]

    formatted_edges = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "relation": edge.get("relation", edge.get("relation_type", "RELATED")),
            "weight": edge.get("weight", 1.0),
            "confidence": edge.get("confidence", 0.5),
            "access_count": edge.get("access_count", 0),
            "created_time": edge.get("created_time"),
        }
        for edge in edges
    ]

    logger.info(
        f"获取L3图谱成功：起始={node_id}, 深度={depth}, 节点={len(formatted_nodes)}, 边={len(formatted_edges)}"
    )

    return jsonify(
        {
            "success": True,
            "start_node": start_node,
            "nodes": formatted_nodes,
            "edges": formatted_edges,
        }
    )


async def get_l2_stats():
    manager = get_component_manager()
    l2_retriever = manager.get_component("l2_memory", L2MemoryAdapter)

    if not l2_retriever or not l2_retriever.is_available:
        return jsonify({"success": False, "error": "L2 记忆库不可用"}), 503

    stats = await l2_retriever.get_stats()

    return jsonify({"success": True, "stats": stats})


async def get_l3_stats():
    """获取 L3 图谱全局统计（节点/边总数、类型分布）"""
    manager = get_component_manager()
    l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

    if not l3_adapter or not l3_adapter.is_available:
        return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

    stats = await l3_adapter.get_stats()
    return jsonify({"success": True, "stats": stats})


async def search_l3_nodes():
    keyword = request.args.get("keyword", "")
    limit = request.args.get("limit", default=20, type=int)

    if not keyword:
        return jsonify({"success": False, "error": "搜索关键词不能为空"}), 400

    limit = max(1, min(limit, 100))

    manager = get_component_manager()
    l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

    if not l3_adapter or not l3_adapter.is_available:
        return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

    nodes = await l3_adapter.search_nodes(keyword, limit)

    logger.info(f"搜索L3节点成功：关键词='{keyword}', 结果数={len(nodes)}")

    return jsonify({"success": True, "nodes": nodes})


async def search_l3_edges():
    keyword = request.args.get("keyword", "")
    limit = request.args.get("limit", default=20, type=int)

    if not keyword:
        return jsonify({"success": False, "error": "搜索关键词不能为空"}), 400

    limit = max(1, min(limit, 100))

    manager = get_component_manager()
    l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

    if not l3_adapter or not l3_adapter.is_available:
        return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

    edges = await l3_adapter.search_edges(keyword, limit)

    logger.info(f"搜索L3边成功：关键词='{keyword}', 结果数={len(edges)}")

    return jsonify({"success": True, "edges": edges})


async def delete_l2_entries():
    data = await request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求正文为空或格式错误"}), 400
    ids = data.get("ids", [])

    if not ids or not isinstance(ids, list):
        return jsonify({"success": False, "error": "请提供要删除的记忆 ID 列表"}), 400

    if len(ids) > 100:
        return jsonify({"success": False, "error": "单次最多删除 100 条记忆"}), 400

    manager = get_component_manager()
    l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)

    if not l2_adapter or not l2_adapter.is_available:
        return jsonify({"success": False, "error": "L2 记忆库不可用"}), 503

    success = await l2_adapter.delete_entries(ids)

    if success:
        logger.info(f"已删除 {len(ids)} 条 L2 记忆")
        return jsonify({"success": True, "deleted_count": len(ids)})
    else:
        return jsonify({"success": False, "error": "删除失败"}), 500


async def update_l2_entry():
    data = await request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求正文为空或格式错误"}), 400
    memory_id = data.get("id", "")
    new_content = data.get("content", "")

    if not memory_id or not new_content:
        return jsonify({"success": False, "error": "请提供记忆 ID 和新内容"}), 400

    manager = get_component_manager()
    l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)

    if not l2_adapter or not l2_adapter.is_available:
        return jsonify({"success": False, "error": "L2 记忆库不可用"}), 503

    success = await l2_adapter.update_content(memory_id, new_content)

    if success:
        logger.info(f"已更新 L2 记忆：{memory_id}")
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "更新失败，记忆可能不存在"}), 500


async def list_l3_nodes():
    limit = request.args.get("limit", default=100, type=int)
    # 钳制 limit 到合理范围，防止恶意大值导致内存/DB 过载
    limit = max(1, min(limit, 500))
    keyword = request.args.get("keyword", "")
    group_id = request.args.get("group_id")

    manager = get_component_manager()
    l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

    if not l3_adapter or not l3_adapter.is_available:
        return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

    if keyword:
        nodes = await l3_adapter.search_nodes(keyword, limit, group_id=group_id)
    else:
        nodes = await l3_adapter.get_all_nodes(limit, group_id=group_id)

    logger.info(f"获取L3节点列表成功：关键词='{keyword}', group_id={group_id}, 结果数={len(nodes)}")

    return jsonify({"success": True, "nodes": nodes})


async def list_l3_edges():
    limit = request.args.get("limit", default=100, type=int)
    # 钳制 limit 到合理范围，防止恶意大值导致内存/DB 过载
    limit = max(1, min(limit, 500))
    keyword = request.args.get("keyword", "")
    group_id = request.args.get("group_id")

    manager = get_component_manager()
    l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

    if not l3_adapter or not l3_adapter.is_available:
        return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

    if keyword:
        edges = await l3_adapter.search_edges(keyword, limit)
    else:
        edges = await l3_adapter.get_all_edges(limit, group_id=group_id)

    logger.info(f"获取L3关系列表成功：关键词='{keyword}', 结果数={len(edges)}")

    return jsonify({"success": True, "edges": edges})


async def delete_l3_nodes():
    data = await request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求正文为空或格式错误"}), 400
    ids = data.get("ids", [])

    if not ids or not isinstance(ids, list):
        return jsonify({"success": False, "error": "请提供要删除的节点 ID 列表"}), 400

    if len(ids) > 100:
        return jsonify({"success": False, "error": "单次最多删除 100 个节点"}), 400

    manager = get_component_manager()
    l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

    if not l3_adapter or not l3_adapter.is_available:
        return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

    deleted_count = await l3_adapter.evict_nodes(ids)

    logger.info(f"已删除 {deleted_count} 个 L3 节点")

    return jsonify({"success": True, "deleted_count": deleted_count})


async def delete_l3_edge():
    data = await request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求正文为空或格式错误"}), 400
    source_id = data.get("source_id", "")
    target_id = data.get("target_id", "")
    relation = data.get("relation", "")

    if not source_id or not target_id or not relation:
        return jsonify(
            {"success": False, "error": "请提供源节点ID、目标节点ID和关系类型"}
        ), 400

    manager = get_component_manager()
    l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

    if not l3_adapter or not l3_adapter.is_available:
        return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

    success = await l3_adapter.delete_edge(source_id, target_id, relation)

    if success:
        logger.info(f"已删除 L3 关系：{source_id} -[{relation}]-> {target_id}")
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "删除失败，关系可能不存在"}), 500


def register_memory_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/memory"

    routes = [
        (f"{prefix}/l2/search", search_l2_memory, ["POST"], "搜索 L2 记忆"),
        (f"{prefix}/l2/latest", get_latest_l2_memories, ["GET"], "获取最新 L2 记忆"),
        (f"{prefix}/l2/stats", get_l2_stats, ["GET"], "获取 L2 统计"),
        (f"{prefix}/l2/delete", delete_l2_entries, ["POST"], "删除 L2 记忆条目"),
        (f"{prefix}/l2/update", update_l2_entry, ["POST"], "更新 L2 记忆条目"),
        (f"{prefix}/l1/list", list_l1_buffer, ["GET"], "获取 L1 缓冲列表"),
        (f"{prefix}/l1/queues", list_l1_queues, ["GET"], "获取 L1 队列列表"),
        (f"{prefix}/l3/graph", get_l3_graph, ["GET"], "获取 L3 图谱数据"),
        (f"{prefix}/l3/stats", get_l3_stats, ["GET"], "获取 L3 图谱统计"),
        (f"{prefix}/l3/search/nodes", search_l3_nodes, ["GET"], "搜索 L3 节点"),
        (f"{prefix}/l3/search/edges", search_l3_edges, ["GET"], "搜索 L3 边"),
        (f"{prefix}/l3/nodes", list_l3_nodes, ["GET"], "获取 L3 节点列表"),
        (f"{prefix}/l3/edges", list_l3_edges, ["GET"], "获取 L3 关系列表"),
        (f"{prefix}/l3/nodes/delete", delete_l3_nodes, ["POST"], "删除 L3 节点"),
        (f"{prefix}/l3/edges/delete", delete_l3_edge, ["POST"], "删除 L3 关系"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)

"""
统计相关 API 路由

提供各类统计数据：
- Token使用统计
- 记忆统计
- 知识图谱统计
- 组件状态追踪
"""

from quart import jsonify
from iris_memory.core import get_component_manager, get_logger, get_uptime
from iris_memory.llm.manager import LLMManager
from iris_memory.l1_buffer.buffer import L1Buffer
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from typing import Dict, Any

logger = get_logger("web.stats")

PLUGIN_NAME = "astrbot_plugin_iris_memory"


def _get_uptime() -> int:
    try:
        return get_uptime()
    except Exception as e:
        logger.warning(f"获取运行时间失败：{e}")
        return 0


async def get_token_stats():
    try:
        manager = get_component_manager()
        llm_manager = manager.get_component("llm_manager", LLMManager)

        if not llm_manager or not llm_manager.is_available:
            return jsonify({"success": False, "error": "LLM 管理器不可用"}), 503

        all_stats = await llm_manager.get_all_token_stats()

        formatted_stats = {}
        for module, stat in all_stats.items():
            formatted_stats[module] = {
                "total_input_tokens": stat.total_input_tokens
                if hasattr(stat, "total_input_tokens")
                else stat.get("total_input_tokens", 0),
                "total_output_tokens": stat.total_output_tokens
                if hasattr(stat, "total_output_tokens")
                else stat.get("total_output_tokens", 0),
                "total_calls": stat.total_calls
                if hasattr(stat, "total_calls")
                else stat.get("total_calls", 0),
            }

        logger.info("获取Token统计成功")

        return jsonify({"success": True, "stats": formatted_stats})

    except Exception as e:
        logger.error(f"获取 Token 统计失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def get_memory_stats():
    try:
        manager = get_component_manager()

        stats: Dict[str, Any] = {"l1": {}, "l2": {}, "l3": {}}

        l1_buffer = manager.get_component("l1_buffer", L1Buffer)
        if l1_buffer and l1_buffer.is_available:
            try:
                stats["l1"] = l1_buffer.get_stats()
            except Exception as e:
                logger.warning(f"获取L1统计失败：{e}")
                stats["l1"] = {}

        l2_memory = manager.get_component("l2_memory", L2MemoryAdapter)
        if l2_memory and l2_memory.is_available:
            try:
                stats["l2"] = await l2_memory.get_stats()
            except Exception as e:
                logger.warning(f"获取L2统计失败：{e}")
                stats["l2"] = {}

        l3_kg = manager.get_component("l3_kg", L3KGAdapter)
        if l3_kg and l3_kg.is_available:
            try:
                kg_stats = await l3_kg.get_stats()
                stats["l3"] = kg_stats
            except Exception as e:
                logger.warning(f"获取L3统计失败：{e}")
                stats["l3"] = {}

        logger.info("获取记忆统计成功")

        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        logger.error(f"获取记忆统计失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def get_kg_stats():
    try:
        manager = get_component_manager()
        l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

        if not l3_adapter or not l3_adapter.is_available:
            return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

        stats = await l3_adapter.get_stats()

        logger.info("获取图谱统计成功")

        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        logger.error(f"获取图谱统计失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def get_system_stats():
    try:
        manager = get_component_manager()

        component_states = manager.get_all_states()

        global_status = manager.status.global_status.value

        stats = {
            "components": component_states,
            "global_status": global_status,
            "uptime": _get_uptime(),
        }

        logger.info("获取系统统计成功")

        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        logger.error(f"获取系统统计失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def get_isolation_status():
    """返回三类隔离开关的当前值，供前端展示状态徽章"""
    try:
        from iris_memory.config import get_config

        config = get_config()
        status = {
            "enable_group_memory_isolation": bool(
                config.get("isolation_config.enable_group_memory_isolation")
            ),
            "enable_group_isolation": bool(
                config.get("isolation_config.enable_group_isolation")
            ),
            "enable_persona_isolation": bool(
                config.get("isolation_config.enable_persona_isolation")
            ),
        }
        return jsonify({"success": True, "status": status})
    except Exception as e:
        logger.error(f"获取隔离状态失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def get_all_stats():
    try:
        manager = get_component_manager()

        memory_stats: Dict[str, Any] = {"l1": {}, "l2": {}, "l3": {}}

        l1_buffer = manager.get_component("l1_buffer", L1Buffer)
        if l1_buffer and l1_buffer.is_available:
            try:
                memory_stats["l1"] = l1_buffer.get_stats()
            except Exception as e:
                logger.warning(f"获取L1统计失败：{e}")

        l2_memory = manager.get_component("l2_memory", L2MemoryAdapter)
        if l2_memory and l2_memory.is_available:
            try:
                memory_stats["l2"] = await l2_memory.get_stats()
            except Exception as e:
                logger.warning(f"获取L2统计失败：{e}")

        l3_kg = manager.get_component("l3_kg", L3KGAdapter)
        if l3_kg and l3_kg.is_available:
            try:
                memory_stats["l3"] = await l3_kg.get_stats()
            except Exception as e:
                logger.warning(f"获取L3统计失败：{e}")

        token_stats: Dict[str, Any] = {
            "global": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
            }
        }
        llm_manager = manager.get_component("llm_manager", LLMManager)
        if llm_manager and llm_manager.is_available:
            try:
                all_stats = await llm_manager.get_all_token_stats()
                for module, stat in all_stats.items():
                    token_stats[module] = {
                        "total_input_tokens": stat.total_input_tokens
                        if hasattr(stat, "total_input_tokens")
                        else stat.get("total_input_tokens", 0),
                        "total_output_tokens": stat.total_output_tokens
                        if hasattr(stat, "total_output_tokens")
                        else stat.get("total_output_tokens", 0),
                        "total_calls": stat.total_calls
                        if hasattr(stat, "total_calls")
                        else stat.get("total_calls", 0),
                    }
            except Exception as e:
                logger.warning(f"获取Token统计失败：{e}")

        kg_stats: Dict[str, Any] = {
            "node_count": 0,
            "edge_count": 0,
            "node_types": {},
            "relation_types": {},
        }
        if l3_kg and l3_kg.is_available:
            try:
                kg_stats = await l3_kg.get_stats()
            except Exception as e:
                logger.warning(f"获取图谱统计失败：{e}")

        component_states = manager.get_all_states()
        global_status = manager.status.global_status.value

        system_stats = {
            "components": component_states,
            "global_status": global_status,
            "uptime": _get_uptime(),
        }

        logger.info("获取所有统计成功")

        return jsonify(
            {
                "success": True,
                "memory": memory_stats,
                "token": token_stats,
                "kg": kg_stats,
                "system": system_stats,
            }
        )

    except Exception as e:
        logger.error(f"获取所有统计失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


def register_stats_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/stats"

    routes = [
        (f"{prefix}/token", get_token_stats, ["GET"], "获取 Token 统计"),
        (f"{prefix}/memory", get_memory_stats, ["GET"], "获取记忆统计"),
        (f"{prefix}/kg", get_kg_stats, ["GET"], "获取图谱统计"),
        (f"{prefix}/system", get_system_stats, ["GET"], "获取系统统计"),
        (f"{prefix}/isolation", get_isolation_status, ["GET"], "获取隔离状态"),
        (f"{prefix}/all", get_all_stats, ["GET"], "获取所有统计"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)

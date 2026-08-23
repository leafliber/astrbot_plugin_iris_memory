"""
统计相关 API 路由

提供各类统计数据：
- Token使用统计
- 记忆统计
- 知识图谱统计
- 组件状态追踪
"""

from quart import jsonify, request
from iris_memory.core import get_component_manager, get_logger, get_uptime
from iris_memory.llm.manager import LLMManager
from iris_memory.l1_buffer.buffer import L1Buffer
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from typing import Dict, Any
from datetime import datetime
import time

logger = get_logger("web.stats")

PLUGIN_NAME = "astrbot_plugin_iris_memory"
TOKEN_STATS_DAYS = {1, 7, 30}


def _get_token_stats_days() -> int:
    """读取 Token 统计范围；后台页面默认展示最近 7 日。"""
    raw_days = request.args.get("days", "7")
    try:
        days = int(raw_days)
    except (TypeError, ValueError):
        days = 7
    return days if days in TOKEN_STATS_DAYS else 7


def _get_uptime() -> int:
    try:
        return get_uptime()
    except Exception as e:
        logger.warning(f"获取运行时间失败：{e}")
        return 0


# 可选组件名 → 启用配置键（配置关闭时组件不注册，补报为 disabled）
_OPTIONAL_COMPONENT_CONFIG = {
    "l1_buffer": "l1_buffer.enable",
    "learning": "learning.enable",
    "l2_memory": "l2_memory.enable",
    "l3_kg": "l3_kg.enable",
    "profile": "profile.enable",
    "image_quota": "l1_buffer.image_parsing.enable",
    "image_cache": "l1_buffer.image_parsing.enable",
}


def _augment_disabled_components(component_states: Dict[str, Any]) -> Dict[str, Any]:
    """把配置禁用而未注册的组件补报为 unavailable/disabled 状态

    前端 ComponentDisabled 依赖 error_type='disabled' 展示"已禁用"；
    未注册组件此前在状态中缺失，页面会停在"等待初始化"遮罩。
    """
    try:
        from iris_memory.config import get_config

        config = get_config()
    except Exception:
        return component_states
    for name, key in _OPTIONAL_COMPONENT_CONFIG.items():
        if name not in component_states and not config.get(key):
            component_states[name] = {
                "status": "unavailable",
                "error": "组件未启用",
                "error_type": "disabled",
            }
    return component_states


async def get_token_stats():
    try:
        manager = get_component_manager()
        llm_manager = manager.get_component("llm_manager", LLMManager)

        if not llm_manager or not llm_manager.is_available:
            return jsonify({"success": False, "error": "LLM 管理器不可用"}), 503

        days = _get_token_stats_days()
        all_stats = await llm_manager.get_token_stats_for_days(days)

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
                "successful_calls": stat.successful_calls
                if hasattr(stat, "successful_calls")
                else stat.get("successful_calls", stat.get("total_calls", 0)),
                "failed_calls": stat.failed_calls
                if hasattr(stat, "failed_calls")
                else stat.get("failed_calls", 0),
                "pending_calls": stat.pending_calls
                if hasattr(stat, "pending_calls")
                else stat.get("pending_calls", 0),
            }

        logger.info("获取Token统计成功")

        return jsonify({"success": True, "days": days, "stats": formatted_stats})

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

        component_states = _augment_disabled_components(manager.get_all_states())

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


def _governor_alerts(metrics: Dict[str, Any], recent_calls: list[dict]) -> list[dict]:
    """Build a read-only alert snapshot from current Governor/call-log state."""

    from iris_memory.config import get_config

    alerts: list[dict] = []
    rpm_limit = int(get_config().get("llm_provider_rpm", 30) or 0)
    if rpm_limit > 0:
        for provider_id, modules in metrics.get("calls_per_minute", {}).items():
            total = sum(int(value) for value in modules.values())
            if total >= rpm_limit * 0.8:
                alerts.append(
                    {
                        "type": "provider_rpm_high",
                        "provider_id": provider_id,
                        "value": total,
                        "threshold": rpm_limit * 0.8,
                    }
                )

    queue_limit = max(1, int(metrics.get("queue_limit", 500)))
    if int(metrics.get("queue_depth", 0)) >= queue_limit * 0.7:
        alerts.append(
            {
                "type": "queue_depth_high",
                "value": int(metrics.get("queue_depth", 0)),
                "threshold": queue_limit * 0.7,
            }
        )
    if int(metrics.get("queue_wait_p95_ms", 0)) > 1000:
        alerts.append(
            {
                "type": "queue_wait_high",
                "value": int(metrics.get("queue_wait_p95_ms", 0)),
                "threshold": 1000,
            }
        )

    streaks: dict[str, int] = {}
    closed: set[str] = set()
    for call in reversed(recent_calls):
        module = str(call.get("module") or "default")
        if module in closed:
            continue
        if call.get("success"):
            closed.add(module)
            continue
        streaks[module] = streaks.get(module, 0) + 1
    for module, count in streaks.items():
        if count >= 5:
            alerts.append(
                {
                    "type": "module_failure_streak",
                    "module": module,
                    "value": count,
                    "threshold": 5,
                }
            )

    # 图片 singleflight/claim 的最终验收信号：同一 hash 在一分钟窗口内若
    # 真正进入 Provider 两次，即使其中一次失败也应告警。排队超时、本地缓存
    # 命中等没有 provider_call_started 的日志不会误报。
    image_calls: dict[str, list[dict]] = {}
    cutoff = time.time() - 60.0
    for call in recent_calls:
        if str(call.get("module") or "") != "image_parsing":
            continue
        metadata = call.get("metadata") or {}
        if not metadata.get("provider_call_started"):
            continue
        image_hash = str(metadata.get("image_hash") or "")
        if not image_hash:
            continue
        try:
            timestamp = datetime.fromisoformat(str(call.get("timestamp"))).timestamp()
        except (TypeError, ValueError):
            continue
        if timestamp >= cutoff:
            image_calls.setdefault(image_hash, []).append(call)
    for image_hash, calls in image_calls.items():
        if len(calls) >= 2:
            alerts.append(
                {
                    "type": "duplicate_image_provider_call",
                    "image_hash": image_hash,
                    "value": len(calls),
                    "threshold": 1,
                    "provider_ids": sorted(
                        {str(call.get("provider_id") or "") for call in calls}
                    ),
                }
            )
    return alerts


async def get_llm_governance_stats():
    try:
        manager = get_component_manager()
        llm_manager = manager.get_component("llm_manager", LLMManager)
        if not llm_manager or not llm_manager.is_available:
            return jsonify({"success": False, "error": "LLM 管理器不可用"}), 503
        metrics = llm_manager.get_governor_metrics()
        recent_calls = llm_manager.get_recent_call_logs(limit=100)
        return jsonify(
            {
                "success": True,
                "metrics": metrics,
                "alerts": _governor_alerts(metrics, recent_calls),
                "recent_calls": recent_calls,
            }
        )
    except Exception as e:
        logger.error(f"获取 LLM 治理统计失败：{e}", exc_info=True)
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
        token_days = _get_token_stats_days()

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
                "successful_calls": 0,
                "failed_calls": 0,
                "pending_calls": 0,
            }
        }
        llm_manager = manager.get_component("llm_manager", LLMManager)
        if llm_manager and llm_manager.is_available:
            try:
                all_stats = await llm_manager.get_token_stats_for_days(token_days)
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
                        "successful_calls": stat.successful_calls
                        if hasattr(stat, "successful_calls")
                        else stat.get("successful_calls", stat.get("total_calls", 0)),
                        "failed_calls": stat.failed_calls
                        if hasattr(stat, "failed_calls")
                        else stat.get("failed_calls", 0),
                        "pending_calls": stat.pending_calls
                        if hasattr(stat, "pending_calls")
                        else stat.get("pending_calls", 0),
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

        component_states = _augment_disabled_components(manager.get_all_states())
        global_status = manager.status.global_status.value

        system_stats = {
            "components": component_states,
            "global_status": global_status,
            "uptime": _get_uptime(),
        }
        llm_governance: Dict[str, Any] = {"metrics": {}, "alerts": []}
        if llm_manager and llm_manager.is_available:
            try:
                governance_metrics = llm_manager.get_governor_metrics()
                governance_calls = llm_manager.get_recent_call_logs(limit=100)
                llm_governance = {
                    "metrics": governance_metrics,
                    "alerts": _governor_alerts(governance_metrics, governance_calls),
                }
            except Exception as e:
                logger.warning(f"获取 LLM 治理统计失败：{e}")

        logger.info("获取所有统计成功")

        return jsonify(
            {
                "success": True,
                "memory": memory_stats,
                "token": token_stats,
                "token_days": token_days,
                "kg": kg_stats,
                "system": system_stats,
                "llm_governance": llm_governance,
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
        (f"{prefix}/llm-governance", get_llm_governance_stats, ["GET"], "获取 LLM 治理统计"),
        (f"{prefix}/isolation", get_isolation_status, ["GET"], "获取隔离状态"),
        (f"{prefix}/all", get_all_stats, ["GET"], "获取所有统计"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)

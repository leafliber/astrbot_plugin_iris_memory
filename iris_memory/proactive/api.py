from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .config import ConfigManager
from .perception import SlidingWindow
from .prompts import resolve_level
from .state import StateManager
from .stats import StatsCollector

if TYPE_CHECKING:
    from astrbot.api.star import Context

KvSaveFn = Callable[[str, Any], Awaitable[None]]


def group_state_summary(state: StateManager, group_id: str) -> dict:
    """构建群状态摘要字典（stats 同步与白名单列表共用）。"""
    data = state.get_state(group_id)
    effective_n, effective_t = state.get_effective_thresholds(group_id)
    return {
        "state": data.state.value,
        "willingness": data.willingness,
        "msg_count": data.msg_count,
        "effective_n": effective_n,
        "effective_t": effective_t,
        "backoff_level": data.backoff_level,
        "consecutive_replies": data.consecutive_replies,
        "initiate_daily_count": data.initiate_daily_count,
    }


def sync_stats_group_state(state: StateManager, stats: StatsCollector) -> None:
    """把白名单内各群的实时状态同步到统计收集器。"""
    if not stats.enabled:
        return
    for gid in state.get_whitelist():
        stats.update_group_state(
            group_id=gid,
            **group_state_summary(state, gid),
        )


def register_web_apis(
    *,
    context: Context,
    plugin_name: str,
    config: ConfigManager,
    state: StateManager,
    stats: StatsCollector,
    window: SlidingWindow,
    kv_save: KvSaveFn,
) -> None:
    """注册插件的统计 / 白名单 / 群管理 / 配置 Web API。"""
    from quart import jsonify
    from quart import request as qrequest

    async def _json_body() -> dict:
        return await qrequest.get_json(force=True, silent=True) or {}

    def _body_group_id(body: dict) -> str:
        return str(body.get("group_id") or "")

    # ---- 统计 ----

    async def _stats_status():
        return jsonify({"enabled": stats.enabled})

    async def _stats_groups():
        return jsonify(stats.get_group_summaries())

    async def _stats_logs():
        group_id = qrequest.args.get("group_id", "")
        try:
            limit = int(qrequest.args.get("limit", "50"))
            offset = int(qrequest.args.get("offset", "0"))
        except (ValueError, TypeError):
            limit, offset = 50, 0
        limit = max(1, min(200, limit))
        offset = max(0, offset)
        return jsonify(stats.get_llm_logs(
            group_id=group_id or None,
            limit=limit,
            offset=offset,
        ))

    async def _stats_group_detail(group_id: str):
        detail = stats.get_group_detail(group_id)
        if detail is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(detail)

    async def _stats_clear():
        stats.clear_logs()
        return jsonify({"ok": True})

    # ---- 白名单 ----

    async def _whitelist_list():
        groups = []
        for gid in state.get_whitelist():
            data = state.get_state(gid)
            anchor = data.anchor
            entry = {"group_id": gid, **group_state_summary(state, gid)}
            entry["anchor_kind"] = anchor.kind
            entry["anchor_users"] = sorted(anchor.participants)
            entry["anchor_keywords"] = sorted(anchor.keywords)
            entry["anchor_reason"] = anchor.reason
            entry["anchor_bot_message"] = anchor.bot_message
            entry["initiate_pending"] = data.initiate_pending_since > 0
            entry["initiate_no_reply_streak"] = data.initiate_no_reply_streak
            groups.append(entry)
        return jsonify(groups)

    async def _whitelist_enable():
        body = await _json_body()
        group_id = _body_group_id(body)
        if not group_id:
            return jsonify({"error": "group_id required"}), 400
        state.add_to_whitelist(group_id)
        await state.save_dirty(kv_save)
        return jsonify({"ok": True, "group_id": group_id})

    async def _whitelist_disable():
        body = await _json_body()
        group_id = _body_group_id(body)
        if not group_id:
            return jsonify({"error": "group_id required"}), 400
        state.remove_from_whitelist(group_id)
        window.remove_group(group_id)
        state.remove_group_lock(group_id)
        await state.save_dirty(kv_save)
        return jsonify({"ok": True, "group_id": group_id})

    # ---- 群管理 ----

    async def _group_set_willingness():
        body = await _json_body()
        group_id = _body_group_id(body)
        level = str(body.get("willingness") or "")
        if not group_id or not level:
            return jsonify({"error": "group_id and willingness required"}), 400
        resolved = resolve_level(level)
        if not resolved:
            return jsonify({"error": "invalid willingness level"}), 400
        state.set_willingness(group_id, resolved)
        await state.save_dirty(kv_save)
        return jsonify({"ok": True, "group_id": group_id, "willingness": resolved})

    async def _group_reset():
        body = await _json_body()
        group_id = _body_group_id(body)
        if not group_id:
            return jsonify({"error": "group_id required"}), 400
        state.reset_group(group_id)
        await state.save_dirty(kv_save)
        return jsonify({"ok": True, "group_id": group_id})

    # ---- 配置 ----

    async def _config_get():
        return jsonify({
            "values": config.get_all_page_config(),
            "meta": ConfigManager.get_page_config_meta(),
        })

    async def _config_set():
        body = await _json_body()
        for key, value in body.items():
            config.set_override(key, value)
        await kv_save("iris_reply:config_overrides", config.get_overrides())
        return jsonify({"ok": True, "values": config.get_all_page_config()})

    prefix = f"/{plugin_name}/reply/stats"
    context.register_web_api(f"{prefix}/status", _stats_status, ["GET"], "Stats status")
    context.register_web_api(f"{prefix}/groups", _stats_groups, ["GET"], "Stats groups")
    context.register_web_api(f"{prefix}/logs", _stats_logs, ["GET"], "Stats logs")
    context.register_web_api(f"{prefix}/group/<group_id>", _stats_group_detail, ["GET"], "Stats group detail")
    context.register_web_api(f"{prefix}/clear", _stats_clear, ["POST"], "Stats clear")

    wl_prefix = f"/{plugin_name}/reply/whitelist"
    context.register_web_api(f"{wl_prefix}/list", _whitelist_list, ["GET"], "Whitelist list")
    context.register_web_api(f"{wl_prefix}/enable", _whitelist_enable, ["POST"], "Whitelist enable")
    context.register_web_api(f"{wl_prefix}/disable", _whitelist_disable, ["POST"], "Whitelist disable")

    grp_prefix = f"/{plugin_name}/reply/group"
    context.register_web_api(f"{grp_prefix}/set_willingness", _group_set_willingness, ["POST"], "Group set willingness")
    context.register_web_api(f"{grp_prefix}/reset", _group_reset, ["POST"], "Group reset")

    cfg_prefix = f"/{plugin_name}/reply/config"
    context.register_web_api(f"{cfg_prefix}/get", _config_get, ["GET"], "Config get")
    context.register_web_api(f"{cfg_prefix}/set", _config_set, ["POST"], "Config set")

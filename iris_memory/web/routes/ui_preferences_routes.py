"""
UI 偏好 API 路由

提供前端 UI 偏好的持久化存储（如主题模式）。

背景：AstrBot 插件页面以 iframe 形式嵌入 Dashboard，受浏览器安全策略限制
localStorage 不可用（sandbox/第三方存储分区）。因此将 UI 偏好存储在后端
JSON 文件中，前端通过 bridge API 读写。
"""

import json
from pathlib import Path
from typing import Any, Dict

from quart import jsonify, request
from iris_memory.config import get_config
from iris_memory.core import get_logger

logger = get_logger("web.ui_preferences")

PLUGIN_NAME = "astrbot_plugin_iris_memory"
_PREFS_FILENAME = "ui_preferences.json"

# 允许存储的键及其默认值
_DEFAULTS: Dict[str, Any] = {
    "dark_mode": True,
}


def _get_prefs_path() -> Path:
    """获取 UI 偏好文件路径"""
    return get_config().data_dir / _PREFS_FILENAME


def _load_prefs() -> Dict[str, Any]:
    """从文件加载 UI 偏好，合并默认值"""
    prefs = dict(_DEFAULTS)
    path = _get_prefs_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in _DEFAULTS:
                    if key in data:
                        prefs[key] = data[key]
        except Exception as e:
            logger.warning(f"加载 UI 偏好失败，使用默认值: {e}")
    return prefs


def _save_prefs(prefs: Dict[str, Any]) -> None:
    """保存 UI 偏好到文件"""
    path = _get_prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, ensure_ascii=False, indent=2)


async def get_ui_preferences():
    try:
        prefs = _load_prefs()
        return jsonify({"success": True, "preferences": prefs})
    except Exception as e:
        logger.error(f"获取 UI 偏好失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def update_ui_preferences():
    try:
        data = await request.get_json(silent=True) or {}
        updates = data.get("updates", {})

        if not updates:
            return jsonify({"success": False, "error": "未提供更新内容"}), 400

        invalid_keys = [k for k in updates if k not in _DEFAULTS]
        if invalid_keys:
            return jsonify(
                {"success": False, "error": f"无效的偏好键: {', '.join(invalid_keys)}"}
            ), 400

        prefs = _load_prefs()
        prefs.update(updates)
        _save_prefs(prefs)

        logger.debug(f"UI 偏好已更新: {list(updates.keys())}")
        return jsonify({"success": True, "preferences": prefs})
    except Exception as e:
        logger.error(f"更新 UI 偏好失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


def register_ui_preferences_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/ui-preferences"

    routes = [
        (f"{prefix}", get_ui_preferences, ["GET"], "获取 UI 偏好"),
        (f"{prefix}/update", update_ui_preferences, ["POST"], "更新 UI 偏好"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)

"""
隐藏配置 API 路由

提供隐藏参数的查看和修改功能：
- 获取所有隐藏配置（含默认值、类型、描述）
- 批量更新隐藏配置
- 删除单个隐藏配置项（恢复默认值）
- 重置所有隐藏配置为默认值

描述和分组信息从 HiddenConfig 的 field(metadata) 自动提取，
增删字段时只需修改 dataclass，无需手动同步本文件。
"""

import re
from collections import OrderedDict
from dataclasses import asdict, fields
from typing import Dict, Any, List

from quart import jsonify, request
from iris_memory.config import get_config
from iris_memory.config.defaults import HiddenConfig
from iris_memory.core import get_logger

logger = get_logger("web.hidden_config")

PLUGIN_NAME = "astrbot_plugin_iris_memory"


def _get_field_type(field_obj) -> str:
    type_name = str(field_obj.type)
    if "int" in type_name:
        return "int"
    elif "float" in type_name:
        return "float"
    elif "bool" in type_name:
        return "bool"
    elif "Literal" in type_name:
        return "literal"
    elif "str" in type_name:
        return "string"
    return "string"


def _get_literal_options(field_obj) -> list:
    type_str = str(field_obj.type)
    if "Literal" not in type_str:
        return []
    matches = re.findall(r"'([^']*)'", type_str)
    return matches


def _build_hidden_config_meta() -> tuple[Dict[str, Dict[str, Any]], OrderedDict]:
    """从 HiddenConfig 的 field(metadata) 自动提取描述和分组信息

    Returns:
        (field_meta, groups)
        field_meta: {key: {"description": ..., "group": ...}}
        groups: OrderedDict {group_name: [key1, key2, ...]}  保持声明顺序
    """
    field_meta: Dict[str, Dict[str, Any]] = {}
    groups: OrderedDict = OrderedDict()

    for f in fields(HiddenConfig):
        meta = f.metadata
        description = meta.get("description", "")
        group = meta.get("group", "其他")

        field_meta[f.name] = {"description": description, "group": group}

        if group not in groups:
            groups[group] = []
        groups[group].append(f.name)

    return field_meta, groups


_FIELD_META, _GROUPS = _build_hidden_config_meta()


async def get_hidden_config():
    try:
        config = get_config()
        all_values = config.get_all_hidden()
        defaults = asdict(HiddenConfig())
        field_map = {f.name: f for f in fields(HiddenConfig)}

        items: List[Dict[str, Any]] = []
        for key, current_value in all_values.items():
            field_obj = field_map.get(key)
            if field_obj is None:
                continue

            meta = _FIELD_META.get(key, {})

            items.append(
                {
                    "key": key,
                    "value": current_value,
                    "default": defaults.get(key),
                    "type": _get_field_type(field_obj),
                    "description": meta.get("description", ""),
                    "group": meta.get("group", "其他"),
                    "options": _get_literal_options(field_obj),
                }
            )

        groups = [{"name": name, "keys": keys} for name, keys in _GROUPS.items()]

        return jsonify({"success": True, "items": items, "groups": groups})

    except Exception as e:
        logger.error(f"获取隐藏配置失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def update_hidden_config():
    try:
        config = get_config()
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求正文为空或格式错误"}), 400
        updates = data.get("updates", {})

        if not updates:
            return jsonify({"success": False, "error": "未提供更新内容"}), 400

        # 构建字段名→类型映射，校验键名和值类型
        field_types = {}
        for f in fields(HiddenConfig):
            type_name = (
                f.type
                if isinstance(f.type, str)
                else getattr(f.type, "__name__", str(f.type))
            )
            # 解析 "int" / "float" / "str" / "bool" 类型注解
            if "int" in type_name:
                field_types[f.name] = (int, "整数")
            elif "float" in type_name:
                field_types[f.name] = (float, "浮点数")
            elif "str" in type_name:
                field_types[f.name] = (str, "字符串")
            elif "bool" in type_name:
                field_types[f.name] = (bool, "布尔值")

        invalid_keys = [k for k in updates if k not in field_types]
        if invalid_keys:
            return jsonify(
                {"success": False, "error": f"无效的配置键: {', '.join(invalid_keys)}"}
            ), 400

        # 校验值类型：字符串污染下游算术/wait_for
        type_errors = []
        for key, value in updates.items():
            expected_type, type_label = field_types[key]
            # bool 是 int 的子类，需单独检查：不允许用 int 给 bool 字段
            if expected_type is bool and not isinstance(value, bool):
                type_errors.append(
                    f"{key} 需要{type_label}，实际 {type(value).__name__}"
                )
            elif expected_type is int and isinstance(value, bool):
                type_errors.append(f"{key} 需要整数，实际布尔值")
            elif expected_type is int and not isinstance(value, int):
                type_errors.append(
                    f"{key} 需要{type_label}，实际 {type(value).__name__}"
                )
            elif expected_type is float and not isinstance(value, (int, float)):
                type_errors.append(
                    f"{key} 需要{type_label}，实际 {type(value).__name__}"
                )
            elif expected_type is str and not isinstance(value, str):
                type_errors.append(
                    f"{key} 需要{type_label}，实际 {type(value).__name__}"
                )

        if type_errors:
            return jsonify(
                {"success": False, "error": "类型错误: " + "; ".join(type_errors)}
            ), 400

        config.update_hidden(updates)

        logger.info(f"隐藏配置已批量更新: {list(updates.keys())}")

        return jsonify({"success": True, "updated_keys": list(updates.keys())})

    except Exception as e:
        logger.error(f"更新隐藏配置失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def delete_hidden_config_item():
    try:
        key = request.args.get("key")
        if not key:
            return jsonify({"success": False, "error": "缺少 key 参数"}), 400

        config = get_config()

        valid_keys = {f.name for f in fields(HiddenConfig)}
        if key not in valid_keys:
            return jsonify({"success": False, "error": f"无效的配置键: {key}"}), 400

        deleted = config.delete_hidden(key)
        if deleted:
            logger.info(f"已删除隐藏配置项: {key}，将使用默认值")
            return jsonify({"success": True, "message": f"配置项 {key} 已恢复为默认值"})
        else:
            return jsonify(
                {"success": True, "message": f"配置项 {key} 未被修改，已是默认值"}
            )

    except Exception as e:
        logger.error(f"删除隐藏配置项失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def reset_hidden_config():
    try:
        config = get_config()
        config.reset_hidden_to_defaults()

        logger.info("已重置所有隐藏配置为默认值")

        return jsonify({"success": True, "message": "所有隐藏配置已重置为默认值"})

    except Exception as e:
        logger.error(f"重置隐藏配置失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


def register_hidden_config_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/hidden-config"

    routes = [
        (f"{prefix}", get_hidden_config, ["GET"], "获取隐藏配置"),
        (f"{prefix}/update", update_hidden_config, ["POST"], "更新隐藏配置"),
        (f"{prefix}/delete", delete_hidden_config_item, ["POST"], "删除隐藏配置项"),
        (f"{prefix}/reset", reset_hidden_config, ["POST"], "重置隐藏配置"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)

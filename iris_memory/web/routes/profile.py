"""
画像相关 API 路由

提供群聊画像和用户画像的管理接口：
- 群聊画像：查看和编辑
- 用户画像：查看和编辑
"""

from quart import jsonify, request
from iris_memory.core import get_component_manager, get_logger
from iris_memory.profile.models import profile_to_dict
from iris_memory.profile.storage import ProfileStorage
from typing import Any, Optional, Tuple
import os

logger = get_logger("web.profile")

PLUGIN_NAME = "astrbot_plugin_iris_memory"

DEBUG_MODE = os.environ.get("IRIS_DEBUG", "").lower() in ("true", "1", "yes")


def get_profile_storage() -> Tuple[Optional[Any], Optional[Tuple]]:
    manager = get_component_manager()
    storage = manager.get_component("profile", ProfileStorage)

    if not storage or not storage.is_available:
        return None, (jsonify({"success": False, "error": "画像系统不可用"}), 503)

    return storage, None


def handle_exception(e: Exception, operation: str) -> Tuple[Any, int]:
    logger.error(f"{operation}失败：{e}", exc_info=True)

    if DEBUG_MODE:
        error_msg = str(e)
    else:
        error_msg = "服务器内部错误"

    return jsonify({"success": False, "error": error_msg}), 500


async def get_group_profile():
    try:
        group_id = request.args.get("group_id")
        if not group_id:
            return jsonify({"success": False, "error": "缺少 group_id 参数"}), 400

        persona_id = request.args.get("persona", "default")
        profile_storage, error = get_profile_storage()
        if error:
            return error

        profile = await profile_storage.get_group_profile(group_id, persona_id)

        if not profile:
            return jsonify({"success": True, "profile": {}})

        profile_data = profile_to_dict(profile)

        logger.info(f"获取群聊画像成功：group_id={group_id}")

        return jsonify({"success": True, "profile": profile_data})

    except Exception as e:
        return handle_exception(e, "获取群聊画像")


async def update_group_profile():
    try:
        data = await request.get_json()
        group_id = data.get("group_id") or request.args.get("group_id")

        if not group_id:
            return jsonify({"success": False, "error": "缺少 group_id 参数"}), 400

        if not data:
            return jsonify({"success": False, "error": "请求体不能为空"}), 400

        persona_id = request.args.get("persona", data.get("persona", "default"))
        profile_storage, error = get_profile_storage()
        if error:
            return error

        success = await profile_storage.update_group_profile(group_id, data, persona_id)

        if success:
            logger.info(f"更新群聊画像成功：group_id={group_id}")
            return jsonify({"success": True, "message": "画像已更新"})
        else:
            return jsonify({"success": False, "error": "更新失败"}), 500

    except Exception as e:
        return handle_exception(e, "更新群聊画像")


async def get_user_profile():
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "缺少 user_id 参数"}), 400

        group_id = request.args.get("group_id")

        persona_id = request.args.get("persona", "default")
        profile_storage, error = get_profile_storage()
        if error:
            return error

        profile = await profile_storage.get_user_profile(user_id, group_id, persona_id)

        if not profile:
            return jsonify({"success": True, "profile": {}})

        profile_data = profile_to_dict(profile)

        logger.info(f"获取用户画像成功：user_id={user_id}, group_id={group_id}")

        return jsonify({"success": True, "profile": profile_data})

    except Exception as e:
        return handle_exception(e, "获取用户画像")


async def update_user_profile():
    try:
        data = await request.get_json()
        user_id = request.args.get("user_id") or (data or {}).get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "缺少 user_id 参数"}), 400

        group_id = request.args.get("group_id") or (data or {}).get("group_id")

        if not data:
            return jsonify({"success": False, "error": "请求体不能为空"}), 400

        persona_id = request.args.get("persona", data.get("persona", "default"))
        profile_storage, error = get_profile_storage()
        if error:
            return error

        success = await profile_storage.update_user_profile(
            user_id=user_id,
            group_id=group_id or "default",
            updates=data,
            persona_id=persona_id,
        )

        if success:
            logger.info(f"更新用户画像成功：user_id={user_id}, group_id={group_id}")
            return jsonify({"success": True, "message": "画像已更新"})
        else:
            return jsonify({"success": False, "error": "更新失败"}), 500

    except Exception as e:
        return handle_exception(e, "更新用户画像")


async def delete_group_profile():
    try:
        group_id = request.args.get("group_id") or (
            await request.get_json(silent=True) or {}
        ).get("group_id")
        if not group_id:
            return jsonify({"success": False, "error": "缺少 group_id 参数"}), 400
        persona_id = request.args.get("persona", "default")
        profile_storage, error = get_profile_storage()
        if error:
            return error

        success = await profile_storage.delete_group_profile(group_id, persona_id)

        if success:
            logger.info(f"删除群聊画像成功：group_id={group_id}")
            return jsonify({"success": True, "message": "群聊画像已删除"})
        else:
            return jsonify({"success": False, "error": "删除失败"}), 500

    except Exception as e:
        return handle_exception(e, "删除群聊画像")


async def delete_user_profile():
    try:
        body = await request.get_json(silent=True) or {}
        user_id = request.args.get("user_id") or body.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "缺少 user_id 参数"}), 400

        # 同时从 query args 和 POST body 读取 group_id，
        # 前端 apiPost 发送 JSON body，此处此前只读 args 导致恒为 "default"
        group_id = request.args.get("group_id") or body.get("group_id") or "default"
        persona_id = request.args.get("persona", body.get("persona", "default"))

        profile_storage, error = get_profile_storage()
        if error:
            return error

        success = await profile_storage.delete_user_profile(
            user_id, group_id, persona_id
        )

        if success:
            logger.info(f"删除用户画像成功：user_id={user_id}, group_id={group_id}")
            return jsonify({"success": True, "message": "用户画像已删除"})
        else:
            return jsonify({"success": False, "error": "删除失败"}), 500

    except Exception as e:
        return handle_exception(e, "删除用户画像")


async def list_group_profiles():
    try:
        persona_id = request.args.get("persona", "default")
        profile_storage, error = get_profile_storage()
        if error:
            return error

        groups = await profile_storage.list_groups(persona_id)

        return jsonify({"success": True, "groups": groups})

    except Exception as e:
        return handle_exception(e, "获取群聊列表")


async def list_user_profiles():
    try:
        group_id = request.args.get("group_id")
        persona_id = request.args.get("persona", "default")

        profile_storage, error = get_profile_storage()
        if error:
            return error
        assert profile_storage is not None

        # 未指定 group_id 时返回所有群聊的用户（遍历 user_group_index），
        # 避免恒返回 "default" 群的用户导致隔离开启后看不到真实群聊的用户
        if group_id:
            users = await profile_storage.list_users(group_id, persona_id)
        else:
            users = await profile_storage.list_all_users(persona_id)

        return jsonify({"success": True, "users": users})

    except Exception as e:
        return handle_exception(e, "获取用户列表")


def register_profile_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/profile"

    routes = [
        (f"{prefix}/group", get_group_profile, ["GET"], "获取群聊画像"),
        (f"{prefix}/group/update", update_group_profile, ["POST"], "更新群聊画像"),
        (f"{prefix}/group/delete", delete_group_profile, ["POST"], "删除群聊画像"),
        (f"{prefix}/user", get_user_profile, ["GET"], "获取用户画像"),
        (f"{prefix}/user/update", update_user_profile, ["POST"], "更新用户画像"),
        (f"{prefix}/user/delete", delete_user_profile, ["POST"], "删除用户画像"),
        (f"{prefix}/groups", list_group_profiles, ["GET"], "获取群聊列表"),
        (f"{prefix}/users", list_user_profiles, ["GET"], "获取用户列表"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)

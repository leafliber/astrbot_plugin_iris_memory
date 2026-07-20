"""
管理操作 API 路由

提供数据管理功能：
- L1 缓冲清空
- L2 记忆删除
- L3 知识图谱删除/合并
- 画像删除
- 任务手动触发
"""

from quart import jsonify, request
from iris_memory.core import get_component_manager, get_logger
from iris_memory.l1_buffer.buffer import L1Buffer
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.profile.storage import ProfileStorage
from iris_memory.tasks.scheduler import TaskScheduler

logger = get_logger("web.manage")

PLUGIN_NAME = "astrbot_plugin_iris_memory"


async def clear_l1_buffer():
    try:
        manager = get_component_manager()
        l1_buffer = manager.get_component("l1_buffer", L1Buffer)

        if not l1_buffer or not l1_buffer.is_available:
            return jsonify({"success": False, "error": "L1 缓冲不可用"}), 503

        data = await request.get_json(silent=True) or {}
        group_id = data.get("group_id")

        # 区分"未传 group_id"（清空全部）与"显式传空字符串"
        # （清空遗留的空键队列——修复前私聊消息混入的 "" 队列）
        if group_id is not None:
            count = l1_buffer.clear_by_group(group_id)
            logger.info(f"已清空 L1 缓冲：会话={group_id!r}，{count} 条消息")
        else:
            count = l1_buffer.clear_all()
            logger.info(f"已清空所有 L1 缓冲：{count} 条消息")

        return jsonify({"success": True, "cleared_count": count})

    except Exception as e:
        logger.error(f"清空 L1 缓冲失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def delete_l2_memory():
    try:
        manager = get_component_manager()
        l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)

        if not l2_adapter or not l2_adapter.is_available:
            return jsonify({"success": False, "error": "L2 记忆库不可用"}), 503

        data = await request.get_json()
        scope = data.get("scope", "all")
        persona_id = data.get("persona", "default")

        if scope == "group":
            group_id = data.get("group_id")
            if not group_id:
                return jsonify(
                    {"success": False, "error": "scope=group 时必须提供 group_id"}
                ), 400
            count = await l2_adapter.delete_by_group(group_id, persona_id)
            logger.info(
                f"已删除群聊 {group_id} (persona {persona_id}) 的 L2 记忆：{count} 条"
            )
        elif scope == "all":
            count = await l2_adapter.delete_all(persona_id)
            logger.info(f"已删除 persona {persona_id} 的 L2 记忆：{count} 条")
        else:
            return jsonify(
                {
                    "success": False,
                    "error": f"无效的 scope：{scope}，可选值：all, group",
                }
            ), 400

        return jsonify({"success": True, "deleted_count": count})

    except Exception as e:
        logger.error(f"删除 L2 记忆失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def delete_l3_kg():
    try:
        manager = get_component_manager()
        l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

        if not l3_adapter or not l3_adapter.is_available:
            return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

        data = await request.get_json()
        scope = data.get("scope", "all")

        if scope == "group":
            group_id = data.get("group_id")
            if not group_id:
                return jsonify(
                    {"success": False, "error": "scope=group 时必须提供 group_id"}
                ), 400
            count = await l3_adapter.delete_by_group(group_id)
            logger.info(f"已删除群聊 {group_id} 的 L3 图谱：{count} 节点")
        elif scope == "all":
            count = await l3_adapter.delete_all()
            logger.info(f"已删除所有 L3 图谱：{count} 节点")
        else:
            return jsonify(
                {
                    "success": False,
                    "error": f"无效的 scope：{scope}，可选值：all, group",
                }
            ), 400

        return jsonify({"success": True, "deleted_count": count})

    except Exception as e:
        logger.error(f"删除 L3 图谱失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def merge_l3_duplicates():
    try:
        manager = get_component_manager()
        l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

        if not l3_adapter or not l3_adapter.is_available:
            return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

        merged_count, deleted_count = await l3_adapter.merge_duplicate_nodes()

        logger.info(
            f"合并重复节点完成：合并 {merged_count} 组，删除 {deleted_count} 个"
        )

        return jsonify(
            {
                "success": True,
                "merged_count": merged_count,
                "deleted_count": deleted_count,
            }
        )

    except Exception as e:
        logger.error(f"合并重复节点失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def delete_profile():
    try:
        manager = get_component_manager()
        profile_storage = manager.get_component("profile", ProfileStorage)

        if not profile_storage or not profile_storage.is_available:
            return jsonify({"success": False, "error": "画像系统不可用"}), 503

        data = await request.get_json()
        scope = data.get("scope", "all")

        if scope == "group":
            group_id = data.get("group_id")
            if not group_id:
                return jsonify(
                    {"success": False, "error": "scope=group 时必须提供 group_id"}
                ), 400
            success = await profile_storage.delete_group_profile(group_id)
            logger.info(f"删除群聊画像：{group_id}，结果={success}")
            return jsonify(
                {"success": success, "deleted": "group_profile", "group_id": group_id}
            )

        elif scope == "user":
            user_id = data.get("user_id")
            group_id = data.get("group_id", "default")
            if not user_id:
                return jsonify(
                    {"success": False, "error": "scope=user 时必须提供 user_id"}
                ), 400
            success = await profile_storage.delete_user_profile(user_id, group_id)
            logger.info(f"删除用户画像：{user_id}@{group_id}，结果={success}")
            return jsonify(
                {
                    "success": success,
                    "deleted": "user_profile",
                    "user_id": user_id,
                    "group_id": group_id,
                }
            )

        elif scope == "all":
            result = await profile_storage.delete_all_profiles()
            logger.info(f"删除所有画像：{result}")
            return jsonify({"success": True, "result": result})

        else:
            return jsonify(
                {
                    "success": False,
                    "error": f"无效的 scope：{scope}，可选值：group, user, all",
                }
            ), 400

    except Exception as e:
        logger.error(f"删除画像失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def trigger_task():
    try:
        manager = get_component_manager()
        scheduler = manager.get_component("scheduler", TaskScheduler)

        if not scheduler or not scheduler.is_available:
            return jsonify({"success": False, "error": "任务调度器不可用"}), 503

        data = await request.get_json()
        task_name = data.get("task")

        if not task_name:
            return jsonify({"success": False, "error": "必须提供 task 参数"}), 400

        valid_tasks = {
            "dream": "iris_memory.dream",
            "cache_cleanup": "iris_memory.tasks.cache_cleanup_task",
        }

        if task_name not in valid_tasks:
            return jsonify(
                {
                    "success": False,
                    "error": f"无效的任务名：{task_name}，可选值：{', '.join(valid_tasks.keys())}",
                }
            ), 400

        if scheduler.is_task_running(task_name):
            return jsonify(
                {
                    "success": False,
                    "error": f"任务 {task_name} 正在执行中，请稍后再试",
                }
            ), 409

        async def _run_task():
            if task_name == "dream":
                from iris_memory.dream import DreamTask

                task = DreamTask(manager)
                await task.execute()
            elif task_name == "cache_cleanup":
                from iris_memory.tasks import ImageCacheCleanupTask

                task = ImageCacheCleanupTask(manager)
                await task.execute()

        await scheduler.schedule_task(task_name, _run_task)

        logger.info(f"已调度任务：{task_name}")

        return jsonify({"success": True, "message": f"任务 {task_name} 已加入调度队列"})

    except Exception as e:
        logger.error(f"触发任务失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def get_tasks_status():
    try:
        manager = get_component_manager()
        scheduler = manager.get_component("scheduler", TaskScheduler)

        if not scheduler or not scheduler.is_available:
            return jsonify({"success": False, "error": "任务调度器不可用"}), 503

        task_names = ["dream", "cache_cleanup"]
        tasks_status = {}
        for name in task_names:
            tasks_status[name] = {
                "running": scheduler.is_task_running(name),
            }

        return jsonify({"success": True, "tasks": tasks_status})

    except Exception as e:
        logger.error(f"获取任务状态失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


def register_manage_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/manage"

    routes = [
        (f"{prefix}/l1/clear", clear_l1_buffer, ["POST"], "清空 L1 缓冲"),
        (f"{prefix}/l2/delete", delete_l2_memory, ["POST"], "删除 L2 记忆"),
        (f"{prefix}/l3/delete", delete_l3_kg, ["POST"], "删除 L3 图谱"),
        (
            f"{prefix}/l3/merge-duplicates",
            merge_l3_duplicates,
            ["POST"],
            "合并 L3 重复节点",
        ),
        (f"{prefix}/profile/delete", delete_profile, ["POST"], "删除画像"),
        (f"{prefix}/tasks/trigger", trigger_task, ["POST"], "手动触发任务"),
        (f"{prefix}/tasks/status", get_tasks_status, ["GET"], "获取任务状态"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)

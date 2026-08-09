"""
数据导入导出 API 路由

提供各数据模块的导入导出功能：
- L2 记忆导出/导入
- L3 知识图谱导出/导入
- 画像导出/导入
- 学习模块导出/导入
- 人格自迭代导出/导入
- 全量导出/导入
"""

import json
from datetime import datetime

from quart import jsonify, request, Response
from iris_memory.core import get_component_manager, get_logger
from iris_memory.l2_memory.io import MemoryExporter, MemoryImporter
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.profile.storage import ProfileStorage
from iris_memory.learning import LearningComponent
from iris_memory.persona_evolution import PersonaEvolutionComponent

logger = get_logger("web.data")

PLUGIN_NAME = "astrbot_plugin_iris_memory"


async def export_l2_memory():
    try:
        manager = get_component_manager()
        l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)

        if not l2_adapter or not l2_adapter.is_available:
            return jsonify({"success": False, "error": "L2 记忆库不可用"}), 503

        group_id = request.args.get("group_id")

        exporter = MemoryExporter(l2_adapter)
        export_data = exporter.export_to_json(await l2_adapter.get_all_entries())

        if group_id:
            data = json.loads(export_data)
            filtered = [
                e
                for e in data.get("entries", [])
                if e.get("metadata", {}).get("group_id") == group_id
            ]
            data["entries"] = filtered
            data["stats"] = {
                "total_count": len(filtered),
                "exported_count": len(filtered),
                "skipped_count": 0,
            }
            export_data = json.dumps(data, ensure_ascii=False, indent=2)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iris_l2_memory_{timestamp}.json"

        response = Response(
            export_data,
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

        logger.info("导出 L2 记忆成功")
        return response

    except Exception as e:
        logger.error(f"导出 L2 记忆失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def import_l2_memory():
    try:
        manager = get_component_manager()
        l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)

        if not l2_adapter or not l2_adapter.is_available:
            return jsonify({"success": False, "error": "L2 记忆库不可用"}), 503

        content_type = request.content_type or ""
        skip_duplicates = True

        if "multipart/form-data" in content_type:
            files = await request.files
            if "file" not in files:
                return jsonify({"success": False, "error": "未找到上传文件"}), 400

            file = files["file"]
            file_content = file.read().decode("utf-8")
            import_data = json.loads(file_content)
        else:
            body = await request.get_json()
            if not body or "data" not in body:
                return jsonify({"success": False, "error": "请求体缺少 data 字段"}), 400
            import_data = body["data"]
            skip_duplicates = body.get("skip_duplicates", True)

        importer = MemoryImporter(l2_adapter)

        if isinstance(import_data, dict) and "entries" in import_data:
            entries_data = import_data["entries"]
        elif isinstance(import_data, list):
            entries_data = import_data
        else:
            return jsonify({"success": False, "error": "无法识别的导入数据格式"}), 400

        from iris_memory.l2_memory.models import MemoryEntry

        entries = []
        for entry_data in entries_data:
            entries.append(MemoryEntry.from_dict(entry_data))

        stats = await importer.import_entries(entries, skip_duplicates=skip_duplicates)

        logger.info(f"导入 L2 记忆成功：{stats.imported_count} 条")

        return jsonify(
            {
                "success": True,
                "stats": {
                    "total_count": stats.total_count,
                    "imported_count": stats.imported_count,
                    "skipped_count": stats.skipped_count,
                    "error_count": stats.error_count,
                },
            }
        )

    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"JSON 解析失败：{e}"}), 400
    except Exception as e:
        logger.error(f"导入 L2 记忆失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def export_l3_kg():
    try:
        manager = get_component_manager()
        l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

        if not l3_adapter or not l3_adapter.is_available:
            return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

        export_data = await l3_adapter.export_all()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iris_l3_kg_{timestamp}.json"

        response = Response(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

        logger.info("导出 L3 知识图谱成功")
        return response

    except Exception as e:
        logger.error(f"导出 L3 知识图谱失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def import_l3_kg():
    try:
        manager = get_component_manager()
        l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

        if not l3_adapter or not l3_adapter.is_available:
            return jsonify({"success": False, "error": "L3 知识图谱不可用"}), 503

        content_type = request.content_type or ""

        if "multipart/form-data" in content_type:
            files = await request.files
            if "file" not in files:
                return jsonify({"success": False, "error": "未找到上传文件"}), 400

            file = files["file"]
            file_content = file.read().decode("utf-8")
            import_data = json.loads(file_content)
            skip_duplicates = True
        else:
            body = await request.get_json()
            if not body or "data" not in body:
                return jsonify({"success": False, "error": "请求体缺少 data 字段"}), 400
            import_data = body["data"]
            skip_duplicates = body.get("skip_duplicates", True)

        stats = await l3_adapter.import_from_data(
            import_data, skip_duplicates=skip_duplicates
        )

        logger.info(
            f"导入 L3 知识图谱成功：{stats['imported_nodes']} 节点，{stats['imported_edges']} 边"
        )

        return jsonify({"success": True, "stats": stats})

    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"JSON 解析失败：{e}"}), 400
    except Exception as e:
        logger.error(f"导入 L3 知识图谱失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def export_profiles():
    try:
        manager = get_component_manager()
        profile_storage = manager.get_component("profile", ProfileStorage)

        if not profile_storage or not profile_storage.is_available:
            return jsonify({"success": False, "error": "画像系统不可用"}), 503

        export_data = await profile_storage.export_all()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iris_profiles_{timestamp}.json"

        response = Response(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

        logger.info("导出画像成功")
        return response

    except Exception as e:
        logger.error(f"导出画像失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def import_profiles():
    try:
        manager = get_component_manager()
        profile_storage = manager.get_component("profile", ProfileStorage)

        if not profile_storage or not profile_storage.is_available:
            return jsonify({"success": False, "error": "画像系统不可用"}), 503

        content_type = request.content_type or ""

        if "multipart/form-data" in content_type:
            files = await request.files
            if "file" not in files:
                return jsonify({"success": False, "error": "未找到上传文件"}), 400

            file = files["file"]
            file_content = file.read().decode("utf-8")
            import_data = json.loads(file_content)
            skip_duplicates = True
        else:
            body = await request.get_json()
            if not body or "data" not in body:
                return jsonify({"success": False, "error": "请求体缺少 data 字段"}), 400
            import_data = body["data"]
            skip_duplicates = body.get("skip_duplicates", True)

        stats = await profile_storage.import_from_data(
            import_data, skip_duplicates=skip_duplicates
        )

        logger.info(
            f"导入画像成功：{stats['imported_groups']} 群聊，{stats['imported_users']} 用户"
        )

        return jsonify({"success": True, "stats": stats})

    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"JSON 解析失败：{e}"}), 400
    except Exception as e:
        logger.error(f"导入画像失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def export_learning():
    try:
        manager = get_component_manager()
        component = manager.get_component("learning", LearningComponent)
        if not component or not component.is_available or not component.storage:
            return jsonify({"success": False, "error": "学习模块不可用或未启用"}), 503

        export_data = component.storage.export_all()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iris_learning_{timestamp}.json"
        logger.info("导出学习模块数据成功")
        return Response(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"导出学习模块数据失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def import_learning():
    try:
        body = await request.get_json(silent=True)
        if not body or "data" not in body:
            return jsonify({"success": False, "error": "请求体缺少 data 字段"}), 400

        manager = get_component_manager()
        component = manager.get_component("learning", LearningComponent)
        if not component or not component.is_available or not component.storage:
            return jsonify({"success": False, "error": "学习模块不可用或未启用"}), 503

        try:
            stats = component.storage.import_from_data(
                body["data"], skip_duplicates=bool(body.get("skip_duplicates", True))
            )
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        logger.info(f"导入学习模块数据成功：{stats}")
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        logger.error(f"导入学习模块数据失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def export_persona_evolution():
    try:
        manager = get_component_manager()
        component = manager.get_component(
            "persona_evolution", PersonaEvolutionComponent
        )
        if not component or not component.is_available or not component.storage:
            return jsonify({"success": False, "error": "人格自迭代模块不可用或未启用"}), 503

        include_samples = (request.args.get("include_samples") or "").lower() in (
            "true", "1", "yes"
        )
        export_data = component.storage.export_all(include_samples=include_samples)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iris_persona_evolution_{timestamp}.json"
        logger.info(f"导出人格自迭代数据成功（include_samples={include_samples}）")
        return Response(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"导出人格自迭代数据失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def import_persona_evolution():
    try:
        body = await request.get_json(silent=True)
        if not body or "data" not in body:
            return jsonify({"success": False, "error": "请求体缺少 data 字段"}), 400

        manager = get_component_manager()
        component = manager.get_component(
            "persona_evolution", PersonaEvolutionComponent
        )
        if not component or not component.is_available or not component.storage:
            return jsonify({"success": False, "error": "人格自迭代模块不可用或未启用"}), 503

        try:
            stats = component.storage.import_from_data(
                body["data"], skip_duplicates=bool(body.get("skip_duplicates", True))
            )
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        logger.info(f"导入人格自迭代数据成功：{stats}")
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        logger.error(f"导入人格自迭代数据失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def export_all():
    try:
        manager = get_component_manager()
        result = {
            "version": "1.1",
            "export_time": datetime.now().isoformat(),
            "l2_memory": None,
            "l3_kg": None,
            "profiles": None,
            "learning": None,
            "persona_evolution": None,
        }

        l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)
        if l2_adapter and l2_adapter.is_available:
            try:
                exporter = MemoryExporter(l2_adapter)
                all_entries = await l2_adapter.get_all_entries()
                export_json = exporter.export_to_json(all_entries)
                result["l2_memory"] = json.loads(export_json)
            except Exception as e:
                logger.warning(f"全量导出 L2 失败：{e}")
                result["l2_memory"] = {"error": "内部错误，详见服务日志"}

        l3_adapter = manager.get_component("l3_kg", L3KGAdapter)
        if l3_adapter and l3_adapter.is_available:
            try:
                result["l3_kg"] = await l3_adapter.export_all()
            except Exception as e:
                logger.warning(f"全量导出 L3 失败：{e}")
                result["l3_kg"] = {"error": "内部错误，详见服务日志"}

        profile_storage = manager.get_component("profile", ProfileStorage)
        if profile_storage and profile_storage.is_available:
            try:
                result["profiles"] = await profile_storage.export_all()
            except Exception as e:
                logger.warning(f"全量导出画像失败：{e}")
                result["profiles"] = {"error": "内部错误，详见服务日志"}

        learning_component = manager.get_component("learning", LearningComponent)
        if (
            learning_component
            and learning_component.is_available
            and learning_component.storage
        ):
            try:
                result["learning"] = learning_component.storage.export_all()
            except Exception as e:
                logger.warning(f"全量导出学习模块失败：{e}")
                result["learning"] = {"error": "内部错误，详见服务日志"}

        pe_component = manager.get_component("persona_evolution", PersonaEvolutionComponent)
        if pe_component and pe_component.is_available and pe_component.storage:
            try:
                result["persona_evolution"] = pe_component.storage.export_all()
            except Exception as e:
                logger.warning(f"全量导出人格自迭代失败：{e}")
                result["persona_evolution"] = {"error": "内部错误，详见服务日志"}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iris_full_backup_{timestamp}.json"

        response = Response(
            json.dumps(result, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

        logger.info("全量导出成功")
        return response

    except Exception as e:
        logger.error(f"全量导出失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def import_all():
    try:
        manager = get_component_manager()
        body = await request.get_json()

        if not body or "data" not in body:
            return jsonify({"success": False, "error": "请求体缺少 data 字段"}), 400

        import_data = body["data"]
        skip_duplicates = body.get("skip_duplicates", True)

        result = {
            "l2_memory": None,
            "l3_kg": None,
            "profiles": None,
            "learning": None,
            "persona_evolution": None,
        }

        l2_data = import_data.get("l2_memory")
        if l2_data:
            l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)
            if l2_adapter and l2_adapter.is_available:
                try:
                    importer = MemoryImporter(l2_adapter)
                    entries_data = (
                        l2_data.get("entries", [])
                        if isinstance(l2_data, dict)
                        else l2_data
                    )
                    from iris_memory.l2_memory.models import MemoryEntry

                    entries = [MemoryEntry.from_dict(e) for e in entries_data]
                    stats = await importer.import_entries(
                        entries, skip_duplicates=skip_duplicates
                    )
                    result["l2_memory"] = {
                        "total_count": stats.total_count,
                        "imported_count": stats.imported_count,
                        "skipped_count": stats.skipped_count,
                        "error_count": stats.error_count,
                    }
                except Exception as e:
                    logger.warning(f"全量导入 L2 记忆失败：{e}")
                    result["l2_memory"] = {"error": "内部错误，详见服务日志"}
            else:
                result["l2_memory"] = {"error": "L2 记忆库不可用"}

        l3_data = import_data.get("l3_kg")
        if l3_data:
            l3_adapter = manager.get_component("l3_kg", L3KGAdapter)
            if l3_adapter and l3_adapter.is_available:
                try:
                    stats = await l3_adapter.import_from_data(
                        l3_data, skip_duplicates=skip_duplicates
                    )
                    result["l3_kg"] = stats
                except Exception as e:
                    logger.warning(f"全量导入 L3 图谱失败：{e}")
                    result["l3_kg"] = {"error": "内部错误，详见服务日志"}
            else:
                result["l3_kg"] = {"error": "L3 知识图谱不可用"}

        profile_data = import_data.get("profiles")
        if profile_data:
            profile_storage = manager.get_component("profile", ProfileStorage)
            if profile_storage and profile_storage.is_available:
                try:
                    stats = await profile_storage.import_from_data(
                        profile_data, skip_duplicates=skip_duplicates
                    )
                    result["profiles"] = stats
                except Exception as e:
                    logger.warning(f"全量导入画像失败：{e}")
                    result["profiles"] = {"error": "内部错误，详见服务日志"}
            else:
                result["profiles"] = {"error": "画像系统不可用"}

        learning_data = import_data.get("learning")
        if learning_data:
            learning_component = manager.get_component("learning", LearningComponent)
            if (
                learning_component
                and learning_component.is_available
                and learning_component.storage
            ):
                try:
                    result["learning"] = learning_component.storage.import_from_data(
                        learning_data, skip_duplicates=skip_duplicates
                    )
                except Exception as e:
                    logger.warning(f"全量导入学习模块失败：{e}")
                    result["learning"] = {"error": "内部错误，详见服务日志"}
            else:
                result["learning"] = {"error": "学习模块不可用"}

        # 旧版 1.0 备份无 persona_evolution 字段，.get() 天然跳过（§19）
        pe_data = import_data.get("persona_evolution")
        if pe_data:
            pe_component = manager.get_component(
                "persona_evolution", PersonaEvolutionComponent
            )
            if pe_component and pe_component.is_available and pe_component.storage:
                try:
                    stats = pe_component.storage.import_from_data(
                        pe_data, skip_duplicates=skip_duplicates
                    )
                    result["persona_evolution"] = stats
                except Exception as e:
                    logger.warning(f"全量导入人格自迭代失败：{e}")
                    result["persona_evolution"] = {"error": "内部错误，详见服务日志"}
            else:
                result["persona_evolution"] = {"error": "人格自迭代不可用"}

        logger.info(f"全量导入完成：{result}")

        return jsonify({"success": True, "result": result})

    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"JSON 解析失败：{e}"}), 400
    except Exception as e:
        logger.error(f"全量导入失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


def register_data_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/data"

    routes = [
        (f"{prefix}/l2/export", export_l2_memory, ["GET"], "导出 L2 记忆"),
        (f"{prefix}/l2/import", import_l2_memory, ["POST"], "导入 L2 记忆"),
        (f"{prefix}/l3/export", export_l3_kg, ["GET"], "导出 L3 知识图谱"),
        (f"{prefix}/l3/import", import_l3_kg, ["POST"], "导入 L3 知识图谱"),
        (f"{prefix}/profile/export", export_profiles, ["GET"], "导出画像"),
        (f"{prefix}/profile/import", import_profiles, ["POST"], "导入画像"),
        (f"{prefix}/learning/export", export_learning, ["GET"], "导出学习模块数据"),
        (f"{prefix}/learning/import", import_learning, ["POST"], "导入学习模块数据"),
        (
            f"{prefix}/persona-evolution/export",
            export_persona_evolution,
            ["GET"],
            "导出人格自迭代数据",
        ),
        (
            f"{prefix}/persona-evolution/import",
            import_persona_evolution,
            ["POST"],
            "导入人格自迭代数据",
        ),
        (f"{prefix}/all/export", export_all, ["GET"], "全量导出"),
        (f"{prefix}/all/import", import_all, ["POST"], "全量导入"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)

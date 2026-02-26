"""数据导入导出路由蓝图"""

from __future__ import annotations

from typing import Any

from iris_memory.web.api.response import error_response, success_response
from iris_memory.utils.logger import get_logger

logger = get_logger("route_io")


def register_io_routes(app: Any, io_service: Any) -> None:
    """注册导入导出路由

    Args:
        app: Quart 应用实例
        io_service: IoService 实例
    """
    from quart import request, Response

    @app.route("/api/export/memories", methods=["GET"])
    async def export_memories():
        try:
            fmt = request.args.get("format", "json")
            if fmt not in ("json", "csv"):
                return error_response("format 必须是 json 或 csv")

            data, content_type, filename = await io_service.export_memories(
                fmt=fmt,
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                storage_layer=request.args.get("layer"),
            )

            return Response(
                data,
                content_type=f"{content_type}; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as e:
            logger.error(f"Export memories error: {e}")
            return error_response("导出失败", 500)

    @app.route("/api/export/kg", methods=["GET"])
    async def export_kg():
        try:
            fmt = request.args.get("format", "json")
            if fmt not in ("json", "csv"):
                return error_response("format 必须是 json 或 csv")

            data, content_type, filename = await io_service.export_kg(
                fmt=fmt,
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
            )

            return Response(
                data,
                content_type=f"{content_type}; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as e:
            logger.error(f"Export KG error: {e}")
            return error_response("导出失败", 500)

    @app.route("/api/import/memories", methods=["POST"])
    async def import_memories():
        try:
            body = await request.get_json()
            if not body or not body.get("data"):
                return error_response("缺少导入数据")

            fmt = body.get("format", "json")
            if fmt not in ("json", "csv"):
                return error_response("format 必须是 json 或 csv")

            result = await io_service.import_memories(data=body["data"], fmt=fmt)
            return success_response(result)
        except Exception as e:
            logger.error(f"Import memories error: {e}")
            return error_response("导入失败", 500)

    @app.route("/api/import/kg", methods=["POST"])
    async def import_kg():
        try:
            body = await request.get_json()
            if not body or not body.get("data"):
                return error_response("缺少导入数据")

            fmt = body.get("format", "json")
            if fmt not in ("json", "csv"):
                return error_response("format 必须是 json 或 csv")

            result = await io_service.import_kg(data=body["data"], fmt=fmt)
            return success_response(result)
        except Exception as e:
            logger.error(f"Import KG error: {e}")
            return error_response("导入失败", 500)

    @app.route("/api/import/preview", methods=["POST"])
    async def import_preview():
        try:
            body = await request.get_json()
            if not body or not body.get("data"):
                return error_response("缺少导入数据")

            fmt = body.get("format", "json")
            import_type = body.get("type", "memories")

            result = await io_service.preview_import_data(
                data=body["data"],
                fmt=fmt,
                import_type=import_type,
            )
            return success_response(result)
        except Exception as e:
            logger.error(f"Import preview error: {e}")
            return error_response("预览失败", 500)

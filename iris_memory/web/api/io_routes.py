"""导入导出路由 /api/v1/io"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request, Response

from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_io_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/io/export/memories", methods=["GET"])
    async def api_export_memories():
        svc = container.get("io_service")
        data_str, content_type, filename = await svc.export_memories(
            fmt=request.args.get("format", "json"),
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
            storage_layer=request.args.get("storage_layer"),
        )
        return Response(
            data_str,
            mimetype=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.route("/api/v1/io/export/iris_chat_memory", methods=["GET"])
    async def api_export_iris_chat_memory():
        """导出记忆为 iris_chat_memory（新版）导入格式。"""
        svc = container.get("io_service")
        data_str, content_type, filename = await svc.export_to_iris_chat_memory(
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
            storage_layer=request.args.get("storage_layer"),
        )
        return Response(
            data_str,
            mimetype=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.route("/api/v1/io/import/memories", methods=["POST"])
    async def api_import_memories():
        fmt = request.args.get("format", "json")

        files = await request.files
        if "file" in files:
            file = files["file"]
            data = (await file.read()).decode("utf-8")
        else:
            data = (await request.get_data()).decode("utf-8")

        if not data:
            return error_response("未收到数据")

        svc = container.get("io_service")
        result = await svc.import_memories(data, fmt=fmt)
        return success_response(result)

    @app.route("/api/v1/io/export/kg", methods=["GET"])
    async def api_export_kg():
        svc = container.get("io_service")
        data_str, content_type, filename = await svc.export_kg(
            fmt=request.args.get("format", "json"),
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
        )
        return Response(
            data_str,
            mimetype=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.route("/api/v1/io/export/personas", methods=["GET"])
    async def api_export_personas():
        svc = container.get("io_service")
        data_str, content_type, filename = await svc.export_personas(
            fmt=request.args.get("format", "json"),
            user_id=request.args.get("user_id"),
        )
        return Response(
            data_str,
            mimetype=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.route("/api/v1/io/import/kg", methods=["POST"])
    async def api_import_kg():
        fmt = request.args.get("format", "json")

        files = await request.files
        if "file" in files:
            file = files["file"]
            data = (await file.read()).decode("utf-8")
        else:
            data = (await request.get_data()).decode("utf-8")

        if not data:
            return error_response("未收到数据")

        svc = container.get("io_service")
        result = await svc.import_kg(data, fmt=fmt)
        return success_response(result)

    @app.route("/api/v1/io/import/personas", methods=["POST"])
    async def api_import_personas():
        fmt = request.args.get("format", "json")

        files = await request.files
        if "file" in files:
            file = files["file"]
            data = (await file.read()).decode("utf-8")
        else:
            data = (await request.get_data()).decode("utf-8")

        if not data:
            return error_response("未收到数据")

        svc = container.get("io_service")
        result = await svc.import_personas(data, fmt=fmt)
        return success_response(result)

    @app.route("/api/v1/io/preview", methods=["POST"])
    async def api_import_preview():
        fmt = request.args.get("format", "json")
        import_type = request.args.get("type", "memories")

        files = await request.files
        if "file" in files:
            file = files["file"]
            data = (await file.read()).decode("utf-8")
        else:
            data = (await request.get_data()).decode("utf-8")

        if not data:
            return error_response("未收到数据")

        svc = container.get("io_service")
        result = await svc.preview_import_data(data, fmt=fmt, import_type=import_type)
        if "error" in result:
            return error_response(result["error"])
        return success_response(result)

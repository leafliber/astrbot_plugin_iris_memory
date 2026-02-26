"""API 响应工具

集中管理 HTTP JSON 响应的构建方式。
"""

from __future__ import annotations

from typing import Any


def json_response(data: Any, status: int = 200) -> Any:
    """构建 JSON 响应"""
    from quart import jsonify
    return jsonify(data), status


def error_response(message: str, status: int = 400) -> Any:
    """构建错误响应"""
    return json_response({"status": "error", "message": message}, status)


def success_response(data: Any = None, message: str = "success") -> Any:
    """构建成功响应"""
    result: dict[str, Any] = {"status": "ok", "message": message}
    if data is not None:
        result["data"] = data
    return json_response(result)

"""审计日志服务

提供统一的审计日志记录能力。
"""

from __future__ import annotations

from datetime import datetime

from iris_memory.utils.logger import get_logger

audit_logger = get_logger("web_audit")


def audit_log(action: str, detail: str = "") -> None:
    """记录审计日志

    Args:
        action: 操作类型（如 delete_memory, import, export 等）
        detail: 操作详情
    """
    audit_logger.info(
        f"[AUDIT] action={action} detail={detail} time={datetime.now().isoformat()}"
    )

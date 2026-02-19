"""Provider 相关工具函数。

注意：规范模块已迁移到 iris_memory.core.provider_utils，
本文件为向后兼容的 re-export。
"""

from iris_memory.core.provider_utils import (  # noqa: F401
    normalize_provider_id,
    extract_provider_id,
    get_provider_by_id,
    get_default_provider,
)

__all__ = [
    "normalize_provider_id",
    "extract_provider_id",
    "get_provider_by_id",
    "get_default_provider",
]

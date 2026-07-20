"""
Iris Chat Memory - 图片解析模块

提供图片解析、配额管理、缓存管理和消息集成功能。
"""

from .models import (
    ImageInfo,
    ParseResult,
    QuotaStatus,
    MessageImages,
    ImageQueueItem,
    ImageParseCache,
    ImageParseStatus,
)
from .quota_manager import ImageQuotaManager
from .cache_manager import ImageCacheManager
from .parser import ImageParser
from .recorder_bridge import (
    MessageRecorderBridge,
    init_recorder_bridge,
    get_recorder_bridge,
)
from .image_utils import (
    compute_phash,
    hamming_distance,
    is_similar_image,
    check_invalid_image,
    compute_url_hash,
    compute_image_hash,
    detect_image_extension,
)

__all__ = [
    "ImageInfo",
    "ParseResult",
    "QuotaStatus",
    "MessageImages",
    "ImageQueueItem",
    "ImageParseCache",
    "ImageParseStatus",
    "ImageQuotaManager",
    "ImageCacheManager",
    "ImageParser",
    "MessageRecorderBridge",
    "init_recorder_bridge",
    "get_recorder_bridge",
    "compute_phash",
    "hamming_distance",
    "is_similar_image",
    "check_invalid_image",
    "compute_url_hash",
    "compute_image_hash",
    "detect_image_extension",
]

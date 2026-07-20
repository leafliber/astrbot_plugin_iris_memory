"""
Iris Chat Memory - Token 计数工具

优先使用 tiktoken 计算精确 Token 数。
若 tiktoken 不可用或编码器下载失败，降级为字符估算。
"""

from iris_memory.core import get_logger

logger = get_logger("token_counter")

_TIKTOKEN_AVAILABLE = False
tiktoken = None

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.debug("tiktoken 未安装，Token 计数将使用字符估算")


# ============================================================================
# 编码器缓存（单例模式）
# ============================================================================

_encoder_cache: dict = {}


def _estimate_tokens(text: str) -> int:
    """字符估算 Token 数

    中文约 2 字符/token，英文约 4 字符/token。
    采用保守估算：平均 2 字符/token。
    """
    return len(text) // 2 + 1


def _try_get_encoder(encoding_name: str = "cl100k_base"):
    """尝试获取编码器，下载失败时降级

    tiktoken 首次使用时会从远程下载编码器文件，
    网络不可用时捕获异常并永久降级为字符估算。
    """
    if not _TIKTOKEN_AVAILABLE:
        return None

    if encoding_name in _encoder_cache:
        return _encoder_cache.get(encoding_name)

    try:
        logger.debug(f"初始化编码器：{encoding_name}")
        enc = tiktoken.get_encoding(encoding_name)
        _encoder_cache[encoding_name] = enc
        logger.debug(f"编码器 {encoding_name} 已缓存")
        return enc
    except Exception as e:
        logger.warning(
            f"tiktoken 编码器 {encoding_name} 初始化失败：{e}，降级为字符估算"
        )
        # 缓存 None 表示已降级，避免反复重试
        _encoder_cache[encoding_name] = None
        return None


def get_encoder(encoding_name: str = "cl100k_base"):
    """获取编码器实例（单例模式）

    若 tiktoken 不可用或编码器下载失败则返回 None。
    """
    return _try_get_encoder(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """计算文本的 Token 数量

    优先使用 tiktoken 编码器计算，若不可用则降级为字符估算。
    """
    if not text:
        return 0

    encoder = _try_get_encoder(encoding_name)
    if encoder is not None:
        return len(encoder.encode(text))

    return _estimate_tokens(text)


def count_messages_tokens(
    messages: list[dict], encoding_name: str = "cl100k_base"
) -> int:
    """计算消息列表的总 Token 数

    适用于 OpenAI Chat API 格式的消息列表。
    """
    if not messages:
        return 0

    encoder = _try_get_encoder(encoding_name)

    total_tokens = 0

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if encoder is not None:
            total_tokens += len(encoder.encode(role))
            total_tokens += len(encoder.encode(content))
        else:
            total_tokens += _estimate_tokens(role)
            total_tokens += _estimate_tokens(content)

        total_tokens += 4

    total_tokens += 2

    return total_tokens

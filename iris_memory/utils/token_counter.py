"""
Iris Chat Memory - Token 计数工具

优先使用 tiktoken 计算精确 Token 数。
若 tiktoken 不可用或编码器下载失败，降级为字符估算。

tiktoken 首次使用需从网络同步下载 BPE 文件（约 1-2MB、无内置超时），
直接发生在事件循环线程会冻结整个 bot。插件启动时应调用
``warm_up_encoders_async`` 在后台线程完成下载；预热期间计数
临时使用字符估算，完成后自动恢复精确计数。
"""

import asyncio
import threading

from ..core import get_logger

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
_warmup_state_lock = threading.Lock()
_warmup_pending = False


def _is_warmup_pending() -> bool:
    with _warmup_state_lock:
        return _warmup_pending


def _set_warmup_pending(value: bool) -> None:
    global _warmup_pending
    with _warmup_state_lock:
        _warmup_pending = value


async def warm_up_encoders_async(
    encodings: tuple[str, ...] = ("cl100k_base",),
) -> None:
    """在后台线程预热 tiktoken 编码器（含首次网络下载）。

    下载完成后（无论成败）恢复正常计数语义；预热期间
    ``count_tokens`` 使用字符估算，避免事件循环被同步下载阻塞。
    """
    _set_warmup_pending(True)
    try:
        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    _try_get_encoder,
                    name,
                    allow_during_warmup=True,
                )
                for name in encodings
            ),
            return_exceptions=True,
        )
        for name, result in zip(encodings, results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"tiktoken 编码器 {name} 预热失败：{result}，继续使用字符估算"
                )
            elif result is not None:
                logger.debug(f"tiktoken 编码器 {name} 预热完成")
    finally:
        _set_warmup_pending(False)


def _estimate_tokens(text: str) -> int:
    """字符估算 Token 数

    中文约 2 字符/token，英文约 4 字符/token。
    采用保守估算：平均 2 字符/token。
    """
    return len(text) // 2 + 1


def _try_get_encoder(
    encoding_name: str = "cl100k_base",
    *,
    allow_during_warmup: bool = False,
):
    """尝试获取编码器，下载失败时降级

    tiktoken 首次使用时会从远程下载编码器文件，
    网络不可用时捕获异常并永久降级为字符估算。
    """
    if not _TIKTOKEN_AVAILABLE:
        return None

    if encoding_name in _encoder_cache:
        return _encoder_cache.get(encoding_name)

    if _is_warmup_pending() and not allow_during_warmup:
        # 预热进行中：同步下载可能阻塞事件循环数十秒，先用字符估算过渡。
        # 预热线程自身通过 allow_during_warmup 绕过此分支；普通调用不缓存
        # None，预热完成后此键恢复正常的加载/降级语义。
        return None

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

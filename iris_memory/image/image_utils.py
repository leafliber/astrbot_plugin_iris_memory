"""
Iris Chat Memory - 图片工具模块

提供 pHash 感知哈希去重和无效图过滤功能。
"""

import hashlib
import io
from typing import Optional, Tuple

from iris_memory.core import get_logger

logger = get_logger("image.utils")

_PIL_AVAILABLE: Optional[bool] = None


def _check_pil() -> bool:
    """检查 PIL 是否可用（延迟检测，结果缓存）"""
    global _PIL_AVAILABLE
    if _PIL_AVAILABLE is None:
        try:
            from PIL import Image  # noqa: F401

            _PIL_AVAILABLE = True
        except ImportError:
            _PIL_AVAILABLE = False
            logger.warning("Pillow 未安装，图片 pHash 去重和无效图过滤不可用")
    return _PIL_AVAILABLE


async def compute_phash(image_data: bytes, hash_size: int = 8) -> Optional[str]:
    """计算图片的感知哈希（pHash）

    使用 DCT（离散余弦变换）算法计算感知哈希，
    相似图片的哈希值汉明距离较小。

    Args:
        image_data: 图片二进制数据
        hash_size: 哈希大小（默认 8，生成 64 位哈希）

    Returns:
        十六进制哈希字符串，失败返回 None
    """
    if not _check_pil():
        return None

    try:
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(image_data))
        img = img.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)

        pixels = np.array(img, dtype=np.float64)

        dct_result = _dct2d(pixels)

        dct_low = dct_result[:hash_size, :hash_size]

        med = np.median(dct_low)
        hash_bits = (dct_low > med).flatten()

        hash_int = 0
        for bit in hash_bits:
            hash_int = (hash_int << 1) | int(bit)

        return format(hash_int, f"0{hash_size * hash_size // 4}x")

    except Exception as e:
        logger.debug(f"计算 pHash 失败：{e}")
        return None


def _dct2d(matrix):
    """简化的 2D DCT 变换

    使用类型-II DCT，仅计算低频部分以提高性能。

    Args:
        matrix: 2D 数组

    Returns:
        DCT 变换结果
    """
    import numpy as np

    N = matrix.shape[0]
    M = matrix.shape[1]

    n_idx = np.arange(N)
    k_idx = np.arange(N)
    cos_n = np.cos(np.pi * (2 * n_idx[:, None] + 1) * k_idx[None, :] / (2 * N))
    dct_rows = matrix @ cos_n

    m_idx = np.arange(M)
    l_idx = np.arange(M)
    cos_m = np.cos(np.pi * (2 * m_idx[:, None] + 1) * l_idx[None, :] / (2 * M))
    dct_result = cos_m.T @ dct_rows @ cos_m

    return dct_result


def hamming_distance(hash1: str, hash2: str) -> int:
    """计算两个哈希值的汉明距离

    Args:
        hash1: 第一个哈希值
        hash2: 第二个哈希值

    Returns:
        汉明距离（不同位的数量）
    """
    hash1 = hash1.removeprefix("ph:")
    hash2 = hash2.removeprefix("ph:")
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 999

    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    xor = val1 ^ val2
    return bin(xor).count("1")


def is_similar_image(hash1: str, hash2: str, threshold: int = 10) -> bool:
    """判断两个图片哈希是否相似

    Args:
        hash1: 第一个哈希值
        hash2: 第二个哈希值
        threshold: 汉明距离阈值（默认 10，越小越严格）

    Returns:
        是否相似
    """
    return hamming_distance(hash1, hash2) <= threshold


async def check_invalid_image(
    image_data: bytes,
    min_size: int = 16,
    std_threshold: float = 5.0,
) -> Tuple[bool, str]:
    """检查图片是否为无效图（纯色/过小）

    Args:
        image_data: 图片二进制数据
        min_size: 最小图片尺寸（像素），小于此值视为无效
        std_threshold: 纯色检测标准差阈值，低于此值视为纯色

    Returns:
        (是否无效, 原因描述)
    """
    if not _check_pil():
        return False, ""

    try:
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(image_data))

        if img.width < min_size or img.height < min_size:
            return True, f"图片过小：{img.width}x{img.height}"

        gray = img.convert("L")
        pixels = np.array(gray, dtype=np.float64)

        std_dev = np.std(pixels)
        if std_dev < std_threshold:
            return True, f"图片接近纯色：标准差={std_dev:.1f}"

        return False, ""

    except Exception as e:
        logger.debug(f"无效图检查失败：{e}")
        return False, ""


def compute_url_hash(url: str) -> str:
    """计算 URL 的 MD5 哈希（用于快速去重，作为 pHash 的降级方案）

    Args:
        url: 图片 URL

    Returns:
        MD5 哈希字符串
    """
    return hashlib.md5(url.encode()).hexdigest()


def detect_image_extension(data: bytes, url: str = "") -> str:
    """从图片数据或 URL 推断文件扩展名

    优先通过图片魔数（magic bytes）检测，回退到 URL 后缀。

    Args:
        data: 图片二进制数据
        url: 图片 URL（魔数检测失败时的回退）

    Returns:
        文件扩展名（含点号），如 ".jpg"、".png"，默认 ".jpg"
    """
    # 魔数检测
    if len(data) >= 12:
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return ".gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"

    # URL 后缀回退
    if url:
        from urllib.parse import urlparse

        path = urlparse(url).path.lower()
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            if path.endswith(ext):
                return ext

    return ".jpg"


async def compute_image_hash(
    image_data: Optional[bytes] = None,
    url: Optional[str] = None,
    use_phash: bool = True,
) -> str:
    """计算图片哈希（优先 pHash，降级为 URL hash）

    Args:
        image_data: 图片二进制数据（可选）
        url: 图片 URL（可选）
        use_phash: 是否尝试使用 pHash

    Returns:
        哈希字符串
    """
    if use_phash and image_data:
        phash = await compute_phash(image_data)
        if phash:
            return f"ph:{phash}"

    if url:
        return compute_url_hash(url)

    return ""

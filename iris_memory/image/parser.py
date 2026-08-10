"""
Iris Chat Memory - 图片解析器

使用 LLM Vision 模型解析图片内容。
优先通过 message_recorder 插件获取本地图片，避免链接过期。
"""

import asyncio
import base64
import re
from typing import List, Optional, TYPE_CHECKING

from iris_memory.core import get_logger
from .models import ImageInfo, ParseResult
from iris_memory.llm_modules import IMAGE_PARSING
from .recorder_bridge import MessageRecorderBridge
from .security import (
    fetch_safe_image_bytes,
    is_safe_remote_url,
    local_image_to_data_url,
)

if TYPE_CHECKING:
    from iris_memory.llm.manager import LLMManager

logger = get_logger("image")


class ImageParser:
    """图片解析器

    使用支持 Vision 能力的 LLM 模型解析图片内容。
    优先通过 MessageRecorderBridge 获取本地图片文件，
    转为 base64 data URL 传给 LLM，避免网络链接过期。

    Attributes:
        _llm_manager: LLM 调用管理器
        _provider: Provider ID（可选）
        _recorder_bridge: MessageRecorder 桥接器（可选）

    Examples:
        >>> parser = ImageParser(llm_manager, recorder_bridge=bridge)
        >>> result = await parser.parse(image_info)
        >>> print(result.content)
    """

    def __init__(
        self,
        llm_manager: "LLMManager",
        provider: str = "",
        recorder_bridge: Optional[MessageRecorderBridge] = None,
    ):
        """初始化图片解析器

        Args:
            llm_manager: LLM 调用管理器
            provider: Provider ID（留空使用配置或默认）
            recorder_bridge: MessageRecorder 桥接器（可选）
        """
        self._llm_manager = llm_manager
        self._provider = provider
        self._recorder_bridge = recorder_bridge

    async def _resolve_image_url(self, image_info: ImageInfo) -> Optional[str]:
        """解析图片 URL，优先使用本地文件

        优先级：
        1. 通过 MessageRecorderBridge 获取本地图片 → data URL
        2. 使用 ImageInfo 中的 file_path → data URL
        3. 回退到网络 URL

        Args:
            image_info: 图片信息

        Returns:
            可用的图片 URL（HTTP URL 或 data URL），不可用返回 None
        """
        if self._recorder_bridge and image_info.message_id:
            local_path = await self._recorder_bridge.get_local_image_path(
                message_id=image_info.message_id,
                image_url=image_info.url,
            )
            if local_path:
                # 路径由 MessageRecorder API 返回；仍强制图片魔数与大小上限。
                data_url = local_image_to_data_url(local_path)
                if data_url:
                    logger.debug(f"使用 MessageRecorder 本地图片：{local_path.name}")
                    return data_url

        if image_info.has_file_path and image_info.file_path:
            from pathlib import Path

            file_path = Path(image_info.file_path)
            if file_path.is_absolute():
                # 平台消息中的绝对路径不可信，只允许读取本插件生成的图片缓存。
                from iris_memory.config import get_config

                try:
                    cache_root = get_config().data_dir / "image_cache"
                    data_url = local_image_to_data_url(
                        file_path, allowed_root=cache_root
                    )
                except RuntimeError:
                    data_url = None
                if data_url:
                    logger.debug(f"使用本地文件图片：{file_path.name}")
                    return data_url
                logger.warning(f"本地图片文件无法读取：{image_info.file_path}")
            elif self._recorder_bridge:
                abs_path = self._recorder_bridge.resolve_relative_path(
                    image_info.file_path
                )
                if abs_path:
                    data_url = local_image_to_data_url(abs_path)
                    if data_url:
                        logger.debug(
                            f"通过 MessageRecorder 解析本地图片：{abs_path.name}"
                        )
                        return data_url

        if image_info.has_url:
            data_url = await self._fetch_image_data_url(image_info.url)
            if data_url:
                return data_url
            logger.info(
                f"图片 URL 不可访问或主机不安全，跳过 LLM 解析：{image_info.url[:80]}"
            )
            return None

        return None

    async def _is_safe_url(self, url: str) -> bool:
        """校验 URL 是否安全（防 SSRF）。

        仅允许 http/https；对主机的全部 DNS 解析地址要求为全局可达地址，
        拒绝任何私网、环回、链路本地、云元数据、保留、组播、未指定等地址。
        """
        return await is_safe_remote_url(url)

    async def _check_url_accessible(self, url: str) -> bool:
        """检查网络图片 URL 是否可达且有内容

        使用统一安全下载器验证 URL 返回了受支持且未超限的图片，避免 LLM
        调用因图片不可下载而浪费 token；初始地址和每一跳重定向均执行 SSRF
        校验。

        Args:
            url: 图片 URL

        Returns:
            URL 是否可访问且有内容
        """
        return await fetch_safe_image_bytes(url) is not None

    async def _fetch_image_data_url(self, url: str) -> Optional[str]:
        """下载网络图片并以 base64 data URL 返回。

        统一下载器逐跳校验 SSRF、禁止自动重定向、限制跳数和流式响应大小，
        并在请求交给网络栈前再次解析目标主机。图片转为 data URL 后，LLM
        provider 不再直接访问用户提供的外链。

        Args:
            url: 图片 URL

        Returns:
            base64 data URL；主机不安全、不可达或超限时返回 None
        """
        downloaded = await fetch_safe_image_bytes(url)
        if downloaded:
            content, mime = downloaded
            b64 = base64.b64encode(content).decode("ascii")
            return f"data:{mime};base64,{b64}"
        return None

    async def parse(self, image_info: ImageInfo) -> ParseResult:
        """解析单张图片

        优先使用本地图片（避免链接过期），回退到网络 URL。

        Args:
            image_info: 图片信息

        Returns:
            解析结果
        """
        image_url = await self._resolve_image_url(image_info)

        if not image_url:
            return ParseResult(
                image_info=image_info, success=False, error_message="图片信息无效"
            )

        prompt = self._build_parse_prompt()

        try:
            response = await self._llm_manager.generate_with_images(
                prompt=prompt,
                image_urls=[image_url],
                module=IMAGE_PARSING,
                provider_id=self._provider if self._provider else None,
            )

            if self._is_unable_to_describe(response):
                logger.info(f"LLM 无法识别图片内容，视为解析失败：{response[:80]}")
                return ParseResult(
                    image_info=image_info,
                    success=False,
                    error_message="LLM 无法识别图片内容",
                )

            return ParseResult(image_info=image_info, content=response, success=True)

        except Exception as e:
            logger.error(f"图片解析失败：{e}")
            return ParseResult(
                image_info=image_info, success=False, error_message=str(e)
            )

    async def parse_batch(
        self, images: List[ImageInfo], max_concurrent: int = 3
    ) -> List[ParseResult]:
        """批量解析图片

        Args:
            images: 图片信息列表
            max_concurrent: 最大并发数

        Returns:
            解析结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _parse_with_semaphore(image: ImageInfo) -> ParseResult:
            async with semaphore:
                return await self.parse(image)

        tasks = [_parse_with_semaphore(img) for img in images]
        return list(await asyncio.gather(*tasks))

    def _build_parse_prompt(self) -> str:
        """构建图片解析提示词

        Returns:
            解析提示词
        """
        return "简要描述图片内容，重点写文字和关键物体，不超过80字。"

    _UNABLE_PATTERNS = re.compile(
        r"无法[查查看].*图|不能.*[查查看].*图|"
        r"没有.*图片|未.*上传|图片.*失败|"
        r"无法.*分析|不能.*分析|"
        r"无法.*识别|不能.*识别|"
        r"无法.*获取|不能.*获取|"
        r"抱歉.*图|sorry.*image",
        re.IGNORECASE,
    )

    def _is_unable_to_describe(self, content: str) -> bool:
        """检测 LLM 返回的内容是否表示无法描述图片

        当图片加载失败或链接过期时，LLM 不会抛异常，
        而是返回"抱歉，我无法查看或分析图片"之类的文本。
        这些内容不应作为图片描述写入 L1 Buffer。

        Args:
            content: LLM 返回的文本

        Returns:
            是否为"无法描述"类回复
        """
        if not content:
            return False

        stripped = content.strip()
        if len(stripped) < 10:
            return False

        return bool(self._UNABLE_PATTERNS.search(stripped))

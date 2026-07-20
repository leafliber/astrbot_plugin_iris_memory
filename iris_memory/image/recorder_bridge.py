"""
Iris Chat Memory - Message Recorder 桥接模块

通过 astrbot_plugin_message_recorder 插件获取本地图片路径，
避免图片链接过期导致无法解析。
"""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from iris_memory.core import get_logger

if TYPE_CHECKING:
    from astrbot.api.star import Context

logger = get_logger("image.recorder_bridge")

RECORDER_PLUGIN_NAME = "astrbot_plugin_message_recorder"


class MessageRecorderBridge:
    """Message Recorder 桥接器

    通过 AstrBot Context 获取 message_recorder 插件的 API，
    查询消息记录中的本地图片路径。

    优先级：
    1. 通过 message_id 查询消息记录，提取本地图片路径
    2. 通过图片 URL 匹配消息链中的 Image 组件
    3. 获取本地文件的绝对路径
    """

    def __init__(self, context: Optional["Context"] = None):
        self._context = context
        self._api: Any = None
        self._checked = False

    def _ensure_api(self) -> bool:
        if self._checked:
            return self._api is not None

        self._checked = True

        if not self._context:
            logger.debug("Context 未设置，无法获取 MessageRecorder API")
            return False

        try:
            star_meta = self._context.get_registered_star(RECORDER_PLUGIN_NAME)
            if not star_meta:
                logger.debug(f"未找到插件 {RECORDER_PLUGIN_NAME}，本地图片获取不可用")
                return False

            star_instance = getattr(star_meta, "star_instance", None) or getattr(
                star_meta, "instance", None
            )
            if not star_instance:
                logger.debug(f"插件 {RECORDER_PLUGIN_NAME} 无实例，本地图片获取不可用")
                return False

            get_api = getattr(star_instance, "get_api", None)
            if not callable(get_api):
                logger.debug(f"插件 {RECORDER_PLUGIN_NAME} 无 get_api() 方法")
                return False

            self._api = get_api()
            if self._api is None:
                logger.debug(f"插件 {RECORDER_PLUGIN_NAME} get_api() 返回 None")
                return False

            logger.info("已连接 MessageRecorder API，本地图片获取可用")
            return True

        except Exception as e:
            logger.debug(f"获取 MessageRecorder API 失败：{e}")
            return False

    async def get_local_image_path(
        self, message_id: str, image_url: Optional[str] = None
    ) -> Optional[Path]:
        """获取消息中图片的本地文件路径

        通过 message_id 查询 message_recorder 中的消息记录，
        从消息链中提取 Image 组件的 local_path。

        Args:
            message_id: 平台消息 ID
            image_url: 图片 URL（用于匹配消息链中的具体图片组件）

        Returns:
            本地文件绝对路径，获取失败返回 None
        """
        if not self._ensure_api():
            return None

        if not message_id:
            return None

        try:
            record = await self._api.get_by_platform_message_id(message_id)
            if not record:
                logger.debug(f"MessageRecorder 中未找到消息：message_id={message_id}")
                return None

            chain_list = record.get_message_chain_list()
            if not chain_list:
                return None

            image_components = [
                comp
                for comp in chain_list
                if isinstance(comp, dict) and comp.get("type") == "Image"
            ]

            if not image_components:
                return None

            target_comp = None
            if image_url and len(image_components) > 1:
                for comp in image_components:
                    comp_url = comp.get("url", "")
                    if comp_url and comp_url == image_url:
                        target_comp = comp
                        break

            if target_comp is None:
                target_comp = image_components[0]

            local_path = target_comp.get("local_path")
            if not local_path:
                logger.debug(f"消息 {message_id} 中的图片未下载到本地")
                return None

            abs_path = self._api.get_media_absolute_path(local_path)
            if abs_path and abs_path.exists():
                logger.debug(f"从 MessageRecorder 获取本地图片：{abs_path}")
                return abs_path

            logger.debug(f"本地图片文件不存在：{local_path}")
            return None

        except Exception as e:
            logger.debug(f"从 MessageRecorder 获取本地图片失败：{e}")
            return None

    def resolve_relative_path(self, rel_path: str) -> Optional[Path]:
        """通过 MessageRecorder API 将相对路径解析为绝对路径

        利用 message_recorder 插件的 get_media_absolute_path() 方法，
        将平台提供的相对文件名（如 QQ 的 CC00AC7C...jpg）解析为本地绝对路径。

        Args:
            rel_path: 相对文件路径或平台文件名

        Returns:
            解析后的绝对路径，失败返回 None
        """
        if not self._ensure_api():
            return None

        try:
            abs_path = self._api.get_media_absolute_path(rel_path)
            if abs_path and abs_path.exists():
                logger.debug(f"通过 MessageRecorder 解析路径：{rel_path} -> {abs_path}")
                return abs_path

            return None

        except Exception as e:
            logger.debug(f"解析相对路径失败：{e}")
            return None

    @staticmethod
    def image_to_data_url(file_path: Path) -> Optional[str]:
        """将本地图片文件转为 base64 data URL

        Args:
            file_path: 图片文件绝对路径

        Returns:
            data URL 字符串（如 data:image/jpeg;base64,...），
            失败返回 None
        """
        try:
            if not file_path.exists() or not file_path.is_file():
                return None

            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = "image/jpeg"

            with open(file_path, "rb") as f:
                image_data = f.read()

            b64_data = base64.b64encode(image_data).decode("utf-8")
            return f"data:{mime_type};base64,{b64_data}"

        except Exception as e:
            logger.debug(f"图片转 data URL 失败：{e}")
            return None


_recorder_bridge: Optional[MessageRecorderBridge] = None


def init_recorder_bridge(context: Optional["Context"] = None) -> MessageRecorderBridge:
    """初始化全局 MessageRecorderBridge 实例

    Args:
        context: AstrBot Context 对象

    Returns:
        MessageRecorderBridge 实例
    """
    global _recorder_bridge
    _recorder_bridge = MessageRecorderBridge(context)
    return _recorder_bridge


def get_recorder_bridge() -> Optional[MessageRecorderBridge]:
    """获取全局 MessageRecorderBridge 实例

    Returns:
        MessageRecorderBridge 实例，未初始化返回 None
    """
    return _recorder_bridge

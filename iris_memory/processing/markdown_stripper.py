"""
Markdown 格式去除器

在消息发送前，根据 AstrBot 的 t2i 配置，智能决定是否去除 Markdown 格式标记。

工作原理：
- 当 AstrBot 启用文本转图片（t2i）时，长文本会渲染为图片（Markdown 正常展示），
  短文本以纯文本发送（Markdown 标记原样暴露）。
- 本模块在消息发送前检测这种情况，自动去除短文本中的 Markdown 格式标记。
- 当 t2i 全局禁用时，所有文本均以纯文本发送，同样执行去除。

配置说明：
- 用户可见配置：enable 开关（通过 AstrBot 管理界面）
- 内部默认配置：preserve_code_blocks、preserve_links、threshold_offset、
  strip_headers、strip_lists（通过 defaults.py 设置，默认全部开启去除）
"""

from __future__ import annotations

import re
import threading
import time
from re import Pattern
from typing import Any, Callable, Optional, TYPE_CHECKING

from iris_memory.core.defaults import DEFAULTS
from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from astrbot.api.star import Context
    from iris_memory.core.config_manager import ConfigManager

logger = get_logger("markdown_stripper")


class T2IConfigReader:
    """AstrBot 文转图配置读取器

    从 AstrBot 主配置读取 t2i 相关设置。
    内置配置缓存，避免频繁读取配置对象。

    Args:
        context: AstrBot Context 对象
        cache_ttl: 配置缓存 TTL（秒），默认 60 秒
    """

    def __init__(self, context: Context, cache_ttl: float = 60.0) -> None:
        self._context = context
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get_t2i_enabled(self) -> bool:
        """获取文转图开关状态

        Returns:
            是否启用文转图
        """
        return self._get_cached("t2i_enabled", self._read_t2i_enabled)

    def get_t2i_threshold(self) -> int:
        """获取文转图字数阈值

        Returns:
            触发转图的最小字数
        """
        return self._get_cached("t2i_threshold", self._read_t2i_threshold)

    def invalidate_cache(self) -> None:
        """清除配置缓存"""
        with self._lock:
            self._cache.clear()

    def _get_cached(self, key: str, reader: Callable[[], Any]) -> Any:
        """带缓存的配置读取"""
        with self._lock:
            now = time.time()
            if key in self._cache:
                value, expire_time = self._cache[key]
                if now < expire_time:
                    return value

            value = reader()
            self._cache[key] = (value, now + self._cache_ttl)
            return value

    def _read_t2i_enabled(self) -> bool:
        """从 AstrBot 主配置读取 t2i 开关"""
        try:
            config = self._context.get_config()
            return bool(config.get("t2i", False))
        except Exception:
            return False

    def _read_t2i_threshold(self) -> int:
        """从 AstrBot 主配置读取 t2i 阈值"""
        try:
            config = self._context.get_config()
            return int(config.get("t2i_word_threshold", 150))
        except Exception:
            return 150


class MarkdownStripper:
    """Markdown 格式去除器

    在消息发送前，根据 AstrBot 的 t2i 配置和消息级 use_t2i_ 覆盖，
    智能决定是否去除 Markdown 格式标记。

    决策逻辑：
    1. 功能总开关关闭 → 跳过
    2. 文本不含 Markdown → 跳过
    3. 消息级 use_t2i_=True → 跳过（强制转图）
    4. 消息级 use_t2i_=False → 执行去除（强制纯文本）
    5. t2i 全局禁用 → 执行去除（全部文本为纯文本）
    6. t2i 全局启用 → 文本长度 < 阈值时执行去除

    Args:
        context: AstrBot Context 对象
        config: 插件配置管理器
    """

    # 编译后的快速检测正则：匹配常见 Markdown 标记
    _MARKDOWN_QUICK_RE: Pattern[str] = re.compile(
        r"\*{1,3}"   # *斜体* **粗体** ***粗斜体***
        r"|__"        # __粗体__
        r"|~~"        # ~~删除线~~
        r"|`"         # `代码` 或 ```代码块```
        r"|#{1,6}\s"  # # 标题
        r"|^>\s"      # > 引用
        r"|\[.+?\]\(" # [链接](url)
        r"|!\["       # ![图片](url)
        r"|^[-*+]\s"  # - 列表
        r"|^\d+\.\s", # 1. 有序列表
        re.MULTILINE,
    )

    # 转义字符正则
    _ESCAPE_RE: Pattern[str] = re.compile(r"\\([*_~`#>\[\]!\\])")
    # 占位符前缀（使用 NUL 字符序列，在正常文本中不会出现）
    _ESC_PLACEHOLDER = "\x00ESC:"

    def __init__(
        self,
        context: Context,
        config: ConfigManager,
    ) -> None:
        self._context = context
        self._config = config
        self._t2i_reader = T2IConfigReader(context)
        self._strip_rules = self._build_strip_rules()

    def should_process(self, event: Any) -> bool:
        """检查是否需要处理该事件

        Args:
            event: 消息事件对象

        Returns:
            是否需要处理
        """
        return bool(self._config.markdown_stripper_enabled)

    def process_result(self, result: Any) -> None:
        """处理消息结果，按需去除 Markdown 格式

        对 chain 中的 Plain 文本组件执行原地修改，保留图片、At 等非文本组件。

        Args:
            result: 消息结果对象（MessageEventResult）
        """
        if not result:
            return

        text = self._get_result_plain_text(result)
        if not text:
            return

        use_t2i = getattr(result, "use_t2i_", None)
        if not self.should_strip(text, use_t2i):
            return

        self._strip_chain_plain_texts(result)

    def should_strip(self, text: str, use_t2i: Optional[bool] = None) -> bool:
        """判断是否需要去除 Markdown

        Args:
            text: 待检测文本
            use_t2i: 消息级 use_t2i_ 覆盖（None 跟随全局设置）

        Returns:
            是否需要去除
        """
        if not self._config.markdown_stripper_enabled:
            return False

        if not self.has_markdown(text):
            return False

        # 消息级 t2i 覆盖
        if use_t2i is True:
            return False  # 强制转图，Markdown 会被正常渲染
        if use_t2i is False:
            return True  # 强制纯文本，需要去除

        # use_t2i is None: 跟随全局 t2i 设置
        t2i_enabled = self._t2i_reader.get_t2i_enabled()

        if not t2i_enabled:
            # t2i 全局禁用 → 所有文本均为纯文本 → 去除
            return True

        # t2i 全局启用 → 短文本纯文本发送，长文本转图
        threshold = self._t2i_reader.get_t2i_threshold()
        offset = DEFAULTS.markdown_stripper.threshold_offset
        effective_threshold = max(1, threshold + offset)

        return len(text) < effective_threshold

    def has_markdown(self, text: str) -> bool:
        """快速检测文本是否包含 Markdown 标记

        使用编译后的正则进行快速预检，避免对纯文本执行不必要的替换。

        Args:
            text: 待检测文本

        Returns:
            是否包含 Markdown 标记
        """
        return bool(self._MARKDOWN_QUICK_RE.search(text))

    # 代码块占位符前缀
    _CODE_PLACEHOLDER = "\x00CODE:"  # noqa: RUF001

    def strip(self, text: str) -> str:
        """执行 Markdown 格式去除

        处理流程：
        1. 保护代码块内容（提取内部内容，替换为占位符）
        2. 保护转义字符（替换为占位符）
        3. 按规则顺序处理所有 Markdown 标记
        4. 恢复转义字符
        5. 恢复代码块内容（已去除标记）
        6. 清理多余空白

        Args:
            text: 原始文本

        Returns:
            去除格式后的文本
        """
        result = text
        code_placeholders: list[str] = []
        escape_placeholders: list[str] = []

        # 1. 保护代码块内容：提取内部内容，替换为占位符
        def _replace_code_block(m: re.Match) -> str:
            idx = len(code_placeholders)
            # 提取代码块内部内容（去除 ``` 和语言标记）
            code_content = m.group(1) if m.lastindex else m.group(0)
            # 去除围栏标记后的内容
            code_content = re.sub(r"^```(?:\w*)\n?", "", code_content)
            code_content = re.sub(r"```$", "", code_content)
            # 去除行内代码的反引号
            code_content = re.sub(r"^`(.+)`$", r"\1", code_content)
            code_placeholders.append(code_content)
            return f"{self._CODE_PLACEHOLDER}{idx}\x00"

        # 保护围栏代码块（匹配 ```...```）
        result = re.sub(r"```(?:\w*)\n?([\s\S]*?)```", _replace_code_block, result)
        # 保护行内代码（匹配 `...`）
        result = re.sub(r"`([^`]+)`", _replace_code_block, result)

        # 2. 保护转义字符：\* → 占位符
        def _replace_escape(m: re.Match) -> str:
            idx = len(escape_placeholders)
            escape_placeholders.append(m.group(1))
            return f"{self._ESC_PLACEHOLDER}{idx}\x00"

        result = self._ESCAPE_RE.sub(_replace_escape, result)

        # 3. 执行 Markdown 去除
        for pattern, replacement in self._strip_rules:
            result = pattern.sub(replacement, result)

        # 4. 恢复转义字符
        for idx, char in enumerate(escape_placeholders):
            result = result.replace(f"{self._ESC_PLACEHOLDER}{idx}\x00", char)

        # 5. 恢复代码块内容（已去除标记）
        for idx, code in enumerate(code_placeholders):
            result = result.replace(f"{self._CODE_PLACEHOLDER}{idx}\x00", code)

        # 6. 清理多余空白
        result = self._cleanup_whitespace(result)
        return result

    # ── 内部方法 ──

    def _build_strip_rules(self) -> list[tuple[Pattern[str], str]]:
        """根据配置构建去除规则

        规则按优先级排序：转义 > 代码块 > 行内格式 > 链接 > 块级格式。
        转义字符使用占位符机制保护，在所有替换完成后恢复。

        内部配置从 defaults.py 读取，用户无需配置。

        Returns:
            (编译后正则, 替换字符串) 元组列表
        """
        rules: list[tuple[Pattern[str], str]] = []

        # ── 代码块与行内代码（优先处理） ──
        if not DEFAULTS.markdown_stripper.preserve_code_blocks:
            # 围栏代码块：提取内容，去除围栏标记
            rules.append(
                (re.compile(r"```(?:\w*)\n?([\s\S]*?)```"), r"\1")
            )
            # 行内代码：提取内容
            rules.append(
                (re.compile(r"`([^`]+)`"), r"\1")
            )

        # ── 图片（在链接之前处理） ──
        rules.append(
            (re.compile(r"!\[([^\]]*)\]\([^)]+\)"), r"[\1]" if r"\1" else "[图片]")
        )

        # ── 链接 ──
        if not DEFAULTS.markdown_stripper.preserve_links:
            rules.append(
                (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1")
            )

        # ── 分隔线（在粗体/斜体之前处理，避免 *** 被误匹配） ──
        rules.append((re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE), ""))

        # ── 粗斜体组合 ──
        rules.append((re.compile(r"\*\*\*(.+?)\*\*\*"), r"\1"))
        rules.append((re.compile(r"___(.+?)___"), r"\1"))

        # ── 粗体 ──
        rules.append((re.compile(r"\*\*(.+?)\*\*"), r"\1"))
        rules.append((re.compile(r"__(.+?)__"), r"\1"))

        # ── 斜体 ──
        # 修复：避免匹配数学表达式（如 3*4*5 或 a*b*c）
        # 策略：斜体标记要求内容以字母/中文开头和结尾，或两侧有空格
        # 匹配 *斜体*、*italic* 但不匹配 *4*、*b*（单字母变量）
        rules.append((re.compile(r"\*(?=[a-zA-Z\u4e00-\u9fa5])([a-zA-Z\u4e00-\u9fa5].*?[a-zA-Z\u4e00-\u9fa5])\*"), r"\1"))
        rules.append((re.compile(r"(?<!\w)_(.+?)_(?!\w)"), r"\1"))

        # ── 删除线 ──
        rules.append((re.compile(r"~~(.+?)~~"), r"\1"))

        # ── 标题 ──
        if DEFAULTS.markdown_stripper.strip_headers:
            rules.append((re.compile(r"^#{1,6}\s+", re.MULTILINE), ""))

        # ── 引用 ──
        rules.append((re.compile(r"^>\s*", re.MULTILINE), ""))

        # ── 列表 ──
        if DEFAULTS.markdown_stripper.strip_lists:
            rules.append((re.compile(r"^[*\-+]\s+", re.MULTILINE), ""))
            rules.append((re.compile(r"^\d+\.\s+", re.MULTILINE), ""))

        return rules

    def _strip_chain_plain_texts(self, result: Any) -> None:
        """修改 chain 中 Plain 组件的文本

        原地修改 Plain 组件的 text 属性，保留 chain 中的非文本组件。
        如果无法导入 Plain 类，则回退到 chain.clear() + message() 模式。
        """
        try:
            from astrbot.api.message_components import Plain
        except ImportError:
            # 回退：整体替换
            text = self._get_result_plain_text(result)
            stripped = self.strip(text)
            if stripped != text:
                result.chain.clear()
                result.message(stripped)
            return

        modified = False
        for comp in result.chain:
            if isinstance(comp, Plain) and comp.text:
                stripped = self.strip(comp.text)
                if stripped != comp.text:
                    comp.text = stripped
                    modified = True

        if modified:
            logger.debug("已去除消息中的 Markdown 格式标记")

    @staticmethod
    def _cleanup_whitespace(text: str) -> str:
        """清理多余空白

        - 合并连续空行（3 行以上 → 2 行）
        - 去除首尾空白
        """
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _get_result_plain_text(result: Any) -> str:
        """获取消息结果的纯文本内容

        Args:
            result: 消息结果对象

        Returns:
            纯文本内容，无法获取时返回空字符串
        """
        if hasattr(result, "get_plain_text"):
            return result.get_plain_text() or ""
        return ""

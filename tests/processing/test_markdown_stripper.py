"""
Markdown 去除器测试

覆盖：
- T2IConfigReader：配置缓存与读取
- MarkdownStripper：决策逻辑、格式去除规则、chain 组件处理
"""

from __future__ import annotations

import time
from unittest.mock import Mock, MagicMock, patch

import pytest

from iris_memory.core.test_utils import setup_test_config, reset_config_manager
from iris_memory.processing.markdown_stripper import (
    T2IConfigReader,
    MarkdownStripper,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def mock_context():
    """创建模拟的 AstrBot Context"""
    ctx = Mock()
    ctx.get_config.return_value = {"t2i": True, "t2i_word_threshold": 150}
    return ctx


@pytest.fixture()
def mock_config():
    """创建启用了 markdown_stripper 的插件配置"""
    setup_test_config({
        "markdown_stripper": {
            "enable": True,
        },
        "error_friendly": {"enable": True},
    })
    from iris_memory.core.config_manager import get_config_manager
    return get_config_manager()


@pytest.fixture()
def reader(mock_context):
    """创建 T2IConfigReader 实例"""
    return T2IConfigReader(mock_context, cache_ttl=60.0)


@pytest.fixture()
def stripper(mock_context, mock_config):
    """创建 MarkdownStripper 实例（默认配置）"""
    return MarkdownStripper(context=mock_context, config=mock_config)


def _make_result(text: str, use_t2i: bool | None = None) -> Mock:
    """构造模拟的 MessageEventResult"""
    from astrbot.api.message_components import Plain

    result = Mock()
    plain = Plain(text)
    result.chain = [plain]
    result.use_t2i_ = use_t2i
    result.get_plain_text.return_value = text
    result.message = Mock(return_value=result)
    return result


# =========================================================================
# T2IConfigReader 测试
# =========================================================================

class TestT2IConfigReader:
    """T2I 配置读取器测试"""

    def test_get_t2i_enabled_true(self, mock_context):
        """t2i 启用时返回 True"""
        mock_context.get_config.return_value = {"t2i": True}
        reader = T2IConfigReader(mock_context)
        assert reader.get_t2i_enabled() is True

    def test_get_t2i_enabled_false(self, mock_context):
        """t2i 未启用时返回 False"""
        mock_context.get_config.return_value = {"t2i": False}
        reader = T2IConfigReader(mock_context)
        assert reader.get_t2i_enabled() is False

    def test_get_t2i_enabled_missing(self, mock_context):
        """配置缺失时返回 False"""
        mock_context.get_config.return_value = {}
        reader = T2IConfigReader(mock_context)
        assert reader.get_t2i_enabled() is False

    def test_get_t2i_threshold_default(self, mock_context):
        """配置缺失时返回默认阈值 150"""
        mock_context.get_config.return_value = {}
        reader = T2IConfigReader(mock_context)
        assert reader.get_t2i_threshold() == 150

    def test_get_t2i_threshold_custom(self, mock_context):
        """返回自定义阈值"""
        mock_context.get_config.return_value = {"t2i_word_threshold": 200}
        reader = T2IConfigReader(mock_context)
        assert reader.get_t2i_threshold() == 200

    def test_cache_hit(self, mock_context):
        """缓存命中时不再调用 get_config"""
        mock_context.get_config.return_value = {"t2i": True}
        reader = T2IConfigReader(mock_context, cache_ttl=60.0)

        reader.get_t2i_enabled()
        reader.get_t2i_enabled()

        # 只调用一次（第二次命中缓存）
        assert mock_context.get_config.call_count == 1

    def test_cache_expired(self, mock_context):
        """缓存过期后重新读取"""
        mock_context.get_config.return_value = {"t2i": True}
        reader = T2IConfigReader(mock_context, cache_ttl=0.01)

        reader.get_t2i_enabled()
        time.sleep(0.02)
        reader.get_t2i_enabled()

        assert mock_context.get_config.call_count == 2

    def test_invalidate_cache(self, mock_context):
        """手动清除缓存"""
        mock_context.get_config.return_value = {"t2i": True}
        reader = T2IConfigReader(mock_context, cache_ttl=60.0)

        reader.get_t2i_enabled()
        reader.invalidate_cache()
        reader.get_t2i_enabled()

        assert mock_context.get_config.call_count == 2

    def test_exception_returns_defaults(self, mock_context):
        """get_config 异常时返回安全默认值"""
        mock_context.get_config.side_effect = RuntimeError("boom")
        reader = T2IConfigReader(mock_context)

        assert reader.get_t2i_enabled() is False
        assert reader.get_t2i_threshold() == 150


# =========================================================================
# MarkdownStripper - 决策逻辑测试
# =========================================================================

class TestMarkdownStripperShouldStrip:
    """should_strip 决策逻辑测试"""

    def test_disabled_returns_false(self, mock_context):
        """功能未启用时不处理"""
        setup_test_config({
            "markdown_stripper": {"enable": False},
        })
        from iris_memory.core.config_manager import get_config_manager
        config = get_config_manager()
        stripper = MarkdownStripper(mock_context, config)
        assert stripper.should_strip("**粗体**") is False

    def test_no_markdown_returns_false(self, stripper):
        """纯文本不处理"""
        assert stripper.should_strip("这是一段普通文本没有格式") is False

    def test_use_t2i_true_returns_false(self, stripper):
        """消息级 use_t2i=True 时不处理"""
        assert stripper.should_strip("**粗体文本**", use_t2i=True) is False

    def test_use_t2i_false_returns_true(self, stripper):
        """消息级 use_t2i=False 时强制处理"""
        assert stripper.should_strip("**粗体文本**", use_t2i=False) is True

    def test_t2i_disabled_always_strip(self, mock_context, mock_config):
        """t2i 全局禁用时：所有 Markdown 文本都需要去除"""
        mock_context.get_config.return_value = {"t2i": False}
        stripper = MarkdownStripper(mock_context, mock_config)
        # 即使文本很长也要去除
        assert stripper.should_strip("**粗体**" + "a" * 500) is True

    def test_t2i_enabled_short_text_strip(self, stripper):
        """t2i 启用 + 短文本 → 去除"""
        short_text = "**粗体**" + "a" * 50  # 远低于阈值 150
        assert stripper.should_strip(short_text) is True

    def test_t2i_enabled_long_text_skip(self, stripper):
        """t2i 启用 + 长文本 → 跳过（会转图）"""
        long_text = "**粗体**" + "a" * 200  # 超过阈值 150
        assert stripper.should_strip(long_text) is False


# =========================================================================
# MarkdownStripper - has_markdown 测试
# =========================================================================

class TestHasMarkdown:
    """Markdown 标记检测测试"""

    @pytest.mark.parametrize("text", [
        "**粗体**",
        "*斜体*",
        "__粗体__",
        "~~删除线~~",
        "`代码`",
        "```python\nprint()```",
        "# 标题",
        "## 二级标题",
        "> 引用",
        "[链接](https://example.com)",
        "![图片](url)",
        "- 列表项",
        "1. 有序列表",
    ])
    def test_detect_markdown(self, stripper, text):
        """能检测各种 Markdown 标记"""
        assert stripper.has_markdown(text) is True

    @pytest.mark.parametrize("text", [
        "这是一段普通文本",
        "Hello World 123",
        "价格是100元",
        "没有任何格式的中文句子。",
    ])
    def test_plain_text_not_detected(self, stripper, text):
        """纯文本不会被误检"""
        assert stripper.has_markdown(text) is False


# =========================================================================
# MarkdownStripper - strip 规则测试
# =========================================================================

class TestMarkdownStripperStrip:
    """Markdown 格式去除规则测试"""

    # ── 行内格式 ──

    def test_strip_bold_asterisk(self, stripper):
        """去除 ** 粗体"""
        assert stripper.strip("**粗体文本**") == "粗体文本"

    def test_strip_bold_underscore(self, stripper):
        """去除 __ 粗体"""
        assert stripper.strip("__粗体文本__") == "粗体文本"

    def test_strip_italic_asterisk(self, stripper):
        """去除 * 斜体"""
        assert stripper.strip("*斜体文本*") == "斜体文本"

    def test_strip_italic_underscore(self, stripper):
        """去除 _ 斜体（不在单词内）"""
        assert stripper.strip("_斜体_") == "斜体"

    def test_strip_bold_italic(self, stripper):
        """去除 *** 粗斜体"""
        assert stripper.strip("***粗斜体***") == "粗斜体"

    def test_strip_strikethrough(self, stripper):
        """去除 ~~ 删除线"""
        assert stripper.strip("~~删除线~~") == "删除线"

    def test_strip_inline_code(self, stripper):
        """去除行内代码反引号"""
        assert stripper.strip("`代码`") == "代码"

    # ── 代码块 ──

    def test_strip_code_block(self, stripper):
        """去除围栏代码块，保留内容"""
        text = "```python\nprint('hello')\n```"
        result = stripper.strip(text)
        assert "print('hello')" in result
        assert "```" not in result

    def test_strip_code_block_no_language(self, stripper):
        """去除无语言标记的代码块"""
        text = "```\nsome code\n```"
        result = stripper.strip(text)
        assert "some code" in result
        assert "```" not in result

    # ── 链接与图片 ──

    def test_strip_link(self, stripper):
        """去除链接，保留文本"""
        assert stripper.strip("[点击这里](https://example.com)") == "点击这里"

    def test_strip_image(self, stripper):
        """替换图片为 [alt]"""
        result = stripper.strip("![示例图片](https://example.com/img.png)")
        assert result == "[示例图片]"

    def test_strip_image_empty_alt(self, stripper):
        """空 alt 的图片替换为 []"""
        result = stripper.strip("![](https://example.com/img.png)")
        assert result == "[]"

    # ── 标题 ──

    def test_strip_headers(self, stripper):
        """去除标题标记"""
        assert stripper.strip("# 一级标题") == "一级标题"
        assert stripper.strip("## 二级标题") == "二级标题"
        assert stripper.strip("### 三级标题") == "三级标题"

    # ── 引用 ──

    def test_strip_blockquote(self, stripper):
        """去除引用标记"""
        assert stripper.strip("> 引用文本") == "引用文本"

    def test_strip_nested_blockquote(self, stripper):
        """去除多行引用"""
        text = "> 第一行\n> 第二行"
        result = stripper.strip(text)
        assert "第一行" in result
        assert "第二行" in result
        assert ">" not in result

    # ── 列表 ──

    def test_strip_unordered_list(self, stripper):
        """去除无序列表标记"""
        text = "- 项目一\n- 项目二\n- 项目三"
        result = stripper.strip(text)
        assert "项目一" in result
        assert "- " not in result

    def test_strip_ordered_list(self, stripper):
        """去除有序列表标记"""
        text = "1. 第一\n2. 第二\n3. 第三"
        result = stripper.strip(text)
        assert "第一" in result
        assert "1." not in result

    # ── 分隔线 ──

    def test_strip_horizontal_rule(self, stripper):
        """去除分隔线"""
        assert stripper.strip("---") == ""
        assert stripper.strip("***") == ""
        assert stripper.strip("___") == ""

    # ── 转义字符 ──

    def test_preserve_escaped(self, stripper):
        """保留转义后的字符"""
        assert stripper.strip(r"\*不是斜体\*") == "*不是斜体*"
        assert stripper.strip(r"\# 不是标题") == "# 不是标题"

    # ── 组合场景 ──

    def test_strip_combined(self, stripper):
        """组合格式去除"""
        text = "这是**粗体**和*斜体*以及`代码`"
        expected = "这是粗体和斜体以及代码"
        assert stripper.strip(text) == expected

    def test_strip_complex_message(self, stripper):
        """复杂消息格式去除"""
        text = (
            "# 今日推荐\n"
            "\n"
            "这是一条**重要消息**，包含以下内容：\n"
            "\n"
            "- 第一点：*很重要*\n"
            "- 第二点：~~已取消~~\n"
            "\n"
            "> 引用一段话\n"
            "\n"
            "[详情链接](https://example.com)"
        )
        result = stripper.strip(text)
        assert "**" not in result
        assert "*很重要*" not in result
        assert "~~" not in result
        assert "#" not in result
        assert "- " not in result
        assert "> " not in result
        assert "[详情链接](https://example.com)" not in result
        assert "今日推荐" in result
        assert "重要消息" in result

    def test_preserve_plain_text(self, stripper):
        """纯文本保持不变"""
        text = "这是一段普通文本，没有任何格式。"
        assert stripper.strip(text) == text

    def test_cleanup_excess_newlines(self, stripper):
        """合并多余空行"""
        text = "第一行\n\n\n\n第二行"
        result = stripper.strip(text)
        assert result == "第一行\n\n第二行"


# =========================================================================
# MarkdownStripper - process_result / should_process 测试
# =========================================================================

class TestProcessResult:
    """process_result 集成测试"""

    def test_should_process_enabled(self, stripper):
        """功能启用时 should_process 返回 True"""
        event = Mock()
        assert stripper.should_process(event) is True

    def test_should_process_disabled(self, mock_context):
        """功能禁用时 should_process 返回 False"""
        setup_test_config({"markdown_stripper": {"enable": False}})
        from iris_memory.core.config_manager import get_config_manager
        config = get_config_manager()
        s = MarkdownStripper(mock_context, config)

        event = Mock()
        assert s.should_process(event) is False

    def test_process_result_strips_plain_component(self, stripper):
        """process_result 修改 Plain 组件的 text"""
        result = _make_result("**粗体文本**")
        stripper.process_result(result)

        from astrbot.api.message_components import Plain
        plain_comp = result.chain[0]
        assert isinstance(plain_comp, Plain)
        assert plain_comp.text == "粗体文本"

    def test_process_result_preserves_non_text_components(self, stripper):
        """process_result 不影响非文本组件"""
        from astrbot.api.message_components import Plain

        image_mock = Mock()
        image_mock.__class__ = type("Image", (), {})  # 非 Plain 类型
        plain = Plain("**粗体**")

        result = Mock()
        result.chain = [plain, image_mock]
        result.use_t2i_ = None
        result.get_plain_text.return_value = "**粗体**"

        stripper.process_result(result)

        # Plain 被修改
        assert plain.text == "粗体"
        # 非 Plain 组件不受影响
        assert result.chain[1] is image_mock

    def test_process_result_skip_no_markdown(self, stripper):
        """纯文本结果不被修改"""
        result = _make_result("这是普通文本")
        original_text = result.chain[0].text
        stripper.process_result(result)
        assert result.chain[0].text == original_text

    def test_process_result_skip_use_t2i_true(self, stripper):
        """use_t2i_=True 时不处理"""
        result = _make_result("**粗体**", use_t2i=True)
        stripper.process_result(result)
        assert result.chain[0].text == "**粗体**"

    def test_process_result_none_result(self, stripper):
        """result 为 None 时安全跳过"""
        stripper.process_result(None)  # 不应抛异常

    def test_process_result_empty_text(self, stripper):
        """空文本安全跳过"""
        result = _make_result("")
        result.get_plain_text.return_value = ""
        stripper.process_result(result)  # 不应抛异常


# =========================================================================
# 边界情况测试
# =========================================================================

class TestEdgeCases:
    """边界情况"""

    def test_empty_string(self, stripper):
        """空字符串"""
        assert stripper.strip("") == ""

    def test_only_markdown_markers(self, stripper):
        """仅包含 Markdown 标记"""
        assert stripper.strip("**粗体**") == "粗体"
        assert stripper.strip("***") == ""  # 分隔线
        assert stripper.strip("---") == ""

    def test_nested_bold_italic(self, stripper):
        """嵌套粗斜体"""
        assert stripper.strip("***粗斜体***") == "粗斜体"

    def test_multiple_paragraphs(self, stripper):
        """多段落文本"""
        text = "**段落一**\n\n*段落二*\n\n`段落三`"
        result = stripper.strip(text)
        assert "段落一" in result
        assert "段落二" in result
        assert "段落三" in result
        assert "**" not in result
        assert "*" not in result or result.count("*") == 0

    def test_url_only_not_markdown(self, stripper):
        """纯 URL 不被当作 Markdown"""
        text = "https://example.com"
        assert stripper.strip(text) == text

    def test_code_block_with_markdown_inside(self, stripper):
        """代码块内的 Markdown 语法也被提取（代码块本身被去除）"""
        text = "```\n**bold** in code\n```"
        result = stripper.strip(text)
        # 代码块去除后，内容中的 **bold** 会被进一步处理
        assert "```" not in result

    def test_multiple_inline_code(self, stripper):
        """多个行内代码"""
        text = "`foo` 和 `bar` 和 `baz`"
        result = stripper.strip(text)
        assert result == "foo 和 bar 和 baz"

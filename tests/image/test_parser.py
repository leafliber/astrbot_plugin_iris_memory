"""图片解析器测试"""

import pytest
from unittest.mock import Mock, AsyncMock

from iris_memory.image import ImageParser, ImageInfo


class TestImageParser:
    """ImageParser 测试"""

    @pytest.fixture
    def mock_llm_manager(self):
        """模拟 LLM Manager"""
        manager = Mock()
        manager.generate_with_images = AsyncMock(
            return_value="这是一张风景图片，显示蓝天白云"
        )
        return manager

    @pytest.fixture
    def parser(self, mock_llm_manager):
        """创建解析器实例"""
        return ImageParser(mock_llm_manager, provider="test_provider")

    @pytest.fixture
    def parser_with_url_check(self, parser):
        """创建并 mock 网络图片下载为成功（返回 data URL）"""
        parser._fetch_image_data_url = AsyncMock(
            return_value="data:image/jpeg;base64,bW9jaw=="
        )
        return parser

    @pytest.mark.asyncio
    async def test_parse_with_url(self, parser_with_url_check, mock_llm_manager):
        """测试使用 URL 解析图片"""
        image_info = ImageInfo(url="https://example.com/image.jpg", format="jpg")

        result = await parser_with_url_check.parse(image_info)

        assert result.success
        assert result.content == "这是一张风景图片，显示蓝天白云"
        assert result.image_info == image_info

        # 验证调用参数：网络图片应已转为 data URL，不再以原始外链传给 LLM
        mock_llm_manager.generate_with_images.assert_called_once()
        call_args = mock_llm_manager.generate_with_images.call_args
        assert len(call_args[1]["image_urls"]) == 1
        assert call_args[1]["image_urls"][0].startswith("data:")
        assert call_args[1]["module"] == "image_parsing"
        assert call_args[1]["provider_id"] == "test_provider"

    @pytest.mark.asyncio
    async def test_parse_with_file_path(self, parser):
        """测试使用文件路径解析图片（文件不存在时回退到 URL 检查）"""
        image_info = ImageInfo(file_path="/path/to/image.jpg", format="jpg")

        result = await parser.parse(image_info)

        assert not result.success
        assert "图片信息无效" in result.error_message

    @pytest.mark.asyncio
    async def test_parse_with_invalid_info(self, parser):
        """测试使用无效信息解析"""
        image_info = ImageInfo()

        result = await parser.parse(image_info)

        assert not result.success
        assert "图片信息无效" in result.error_message

    @pytest.mark.asyncio
    async def test_parse_with_llm_error(self, parser_with_url_check, mock_llm_manager):
        """测试 LLM 调用失败"""
        mock_llm_manager.generate_with_images.side_effect = Exception("网络错误")

        image_info = ImageInfo(url="https://example.com/image.jpg")
        result = await parser_with_url_check.parse(image_info)

        assert not result.success
        assert "网络错误" in result.error_message

    @pytest.mark.asyncio
    async def test_parse_batch(self, parser_with_url_check, mock_llm_manager):
        """测试批量解析"""
        images = [
            ImageInfo(url="https://example.com/1.jpg"),
            ImageInfo(url="https://example.com/2.jpg"),
            ImageInfo(url="https://example.com/3.jpg"),
        ]

        results = await parser_with_url_check.parse_batch(images)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert mock_llm_manager.generate_with_images.call_count == 3

    @pytest.mark.asyncio
    async def test_parse_with_default_provider(self, mock_llm_manager):
        """测试使用默认 provider"""
        parser = ImageParser(mock_llm_manager)  # 不指定 provider
        parser._fetch_image_data_url = AsyncMock(
            return_value="data:image/jpeg;base64,bW9jaw=="
        )

        image_info = ImageInfo(url="https://example.com/image.jpg")
        result = await parser.parse(image_info)

        assert result.success

        # 验证调用参数
        call_args = mock_llm_manager.generate_with_images.call_args
        assert call_args[1]["provider_id"] is None

    def test_build_parse_prompt(self, parser):
        """测试构建解析提示词"""
        prompt = parser._build_parse_prompt()

        assert "简要描述图片内容" in prompt
        assert "不超过80字" in prompt

    @pytest.mark.asyncio
    async def test_parse_with_unable_to_describe_response(
        self, parser_with_url_check, mock_llm_manager
    ):
        """测试 LLM 返回无法识别图片内容时的处理"""
        mock_llm_manager.generate_with_images.return_value = (
            "抱歉，我无法查看或分析图片，因为目前没有图片附件上传成功。"
        )

        image_info = ImageInfo(url="https://example.com/broken.jpg")
        result = await parser_with_url_check.parse(image_info)

        assert not result.success
        assert "无法识别" in result.error_message

    @pytest.mark.asyncio
    async def test_parse_with_unable_to_analyze_response(self, mock_llm_manager):
        """测试 LLM 返回无法分析图片时的处理"""
        manager = Mock()
        manager.generate_with_images = AsyncMock(
            return_value="我无法分析这张图片，因为这里没有图片显示。"
        )
        p = ImageParser(manager)
        p._fetch_image_data_url = AsyncMock(
            return_value="data:image/jpeg;base64,bW9jaw=="
        )

        image_info = ImageInfo(url="https://example.com/missing.jpg")
        result = await p.parse(image_info)

        assert not result.success

    @pytest.mark.asyncio
    async def test_parse_with_url_inaccessible(self, parser):
        """测试图片 URL 不可达时跳过 LLM 调用"""
        parser._fetch_image_data_url = AsyncMock(return_value=None)

        image_info = ImageInfo(url="https://example.com/expired.jpg")
        result = await parser.parse(image_info)

        assert not result.success
        assert "图片信息无效" in result.error_message

    def test_is_unable_to_describe_positive(self, parser):
        """测试检测无法描述图片的回复（正向用例）"""
        assert parser._is_unable_to_describe(
            "抱歉，我无法查看或分析图片，因为目前没有图片附件上传成功。"
        )
        assert parser._is_unable_to_describe(
            "我无法分析这张图片，因为这里没有图片显示。"
        )
        assert parser._is_unable_to_describe("无法识别图片中的内容")
        assert parser._is_unable_to_describe("没有图片内容可以查看")
        assert parser._is_unable_to_describe("图片加载失败，无法获取图片信息")

    def test_is_unable_to_describe_negative(self, parser):
        """测试正常图片描述不被误判（反向用例）"""
        assert not parser._is_unable_to_describe(
            "图中展示了一个紫色卡通玩偶，带有黑色猫耳和白色骷髅头装饰"
        )
        assert not parser._is_unable_to_describe(
            "图片显示一份每日任务清单，包含背诵古诗、数学练习等"
        )
        assert not parser._is_unable_to_describe("")
        assert not parser._is_unable_to_describe("短文本")

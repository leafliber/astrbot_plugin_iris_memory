"""图片解析数据模型测试"""

from iris_memory.image.models import ImageInfo, ParseResult, QuotaStatus, MessageImages


class TestImageInfo:
    """ImageInfo 数据类测试"""

    def test_create_with_url(self):
        """测试使用 URL 创建"""
        info = ImageInfo(url="https://example.com/image.jpg", format="jpg", size_kb=256)

        assert info.url == "https://example.com/image.jpg"
        assert info.format == "jpg"
        assert info.size_kb == 256
        assert info.source == "user"
        assert info.message_id == ""

    def test_create_with_file_path(self):
        """测试使用文件路径创建"""
        info = ImageInfo(file_path="/path/to/image.png", format="png", size_kb=512)

        assert info.file_path == "/path/to/image.png"
        assert info.format == "png"
        assert info.has_file_path
        assert not info.has_url

    def test_to_dict(self):
        """测试转换为字典"""
        info = ImageInfo(
            url="https://example.com/image.jpg",
            format="jpg",
            size_kb=256,
            source="forward",
            message_id="msg_123",
        )

        data = info.to_dict()

        assert data["url"] == "https://example.com/image.jpg"
        assert data["format"] == "jpg"
        assert data["size_kb"] == 256
        assert data["source"] == "forward"
        assert data["message_id"] == "msg_123"

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "url": "https://example.com/image.jpg",
            "format": "jpg",
            "size_kb": 256,
            "source": "user",
            "message_id": "",
        }

        info = ImageInfo.from_dict(data)

        assert info.url == "https://example.com/image.jpg"
        assert info.format == "jpg"
        assert info.size_kb == 256

    def test_has_url(self):
        """测试 URL 检查"""
        info_with_url = ImageInfo(url="https://example.com/image.jpg")
        info_without_url = ImageInfo()

        assert info_with_url.has_url
        assert not info_without_url.has_url

    def test_has_file_path(self):
        """测试文件路径检查"""
        info_with_path = ImageInfo(file_path="/path/to/image.jpg")
        info_without_path = ImageInfo()

        assert info_with_path.has_file_path
        assert not info_without_path.has_file_path


class TestParseResult:
    """ParseResult 数据类测试"""

    def test_create_success_result(self):
        """测试创建成功结果"""
        image_info = ImageInfo(url="https://example.com/image.jpg")
        result = ParseResult(
            image_info=image_info,
            content="这是一张风景图片",
            input_tokens=150,
            output_tokens=30,
            success=True,
        )

        assert result.success
        assert result.content == "这是一张风景图片"
        assert result.total_tokens == 180
        assert result.error_message == ""

    def test_create_failure_result(self):
        """测试创建失败结果"""
        result = ParseResult(success=False, error_message="网络错误")

        assert not result.success
        assert result.error_message == "网络错误"

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        image_info = ImageInfo(url="https://example.com/image.jpg")
        result = ParseResult(
            image_info=image_info,
            content="这是一张风景图片",
            input_tokens=150,
            output_tokens=30,
            success=True,
        )

        data = result.to_dict()
        restored = ParseResult.from_dict(data)

        assert restored.content == result.content
        assert restored.input_tokens == result.input_tokens
        assert restored.success == result.success
        assert restored.image_info.url == image_info.url


class TestQuotaStatus:
    """QuotaStatus 数据类测试"""

    def test_create_quota_status(self):
        """测试创建配额状态"""
        status = QuotaStatus(date="2026-03-29", used=50, total=200)

        assert status.date == "2026-03-29"
        assert status.used == 50
        assert status.total == 200

    def test_remaining(self):
        """测试剩余配额计算"""
        status = QuotaStatus(used=50, total=200)

        assert status.remaining == 150

        # 测试超额情况
        exhausted_status = QuotaStatus(used=250, total=200)
        assert exhausted_status.remaining == 0

    def test_is_exhausted(self):
        """测试配额耗尽检查"""
        not_exhausted = QuotaStatus(used=50, total=200)
        exhausted = QuotaStatus(used=200, total=200)
        over_exhausted = QuotaStatus(used=250, total=200)

        assert not not_exhausted.is_exhausted
        assert exhausted.is_exhausted
        assert over_exhausted.is_exhausted

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        status = QuotaStatus(date="2026-03-29", used=50, total=200)

        data = status.to_dict()
        restored = QuotaStatus.from_dict(data)

        assert restored.date == status.date
        assert restored.used == status.used
        assert restored.total == status.total


class TestMessageImages:
    """MessageImages 数据类测试"""

    def test_create_empty_message_images(self):
        """测试创建空消息图片集合"""
        images = MessageImages(message_id="msg_123")

        assert images.message_id == "msg_123"
        assert len(images.current_images) == 0
        assert len(images.reply_images) == 0
        assert not images.has_images

    def test_add_images(self):
        """测试添加图片"""
        images = MessageImages(message_id="msg_123")

        current_img = ImageInfo(url="https://example.com/1.jpg", source="user")
        reply_img = ImageInfo(url="https://example.com/2.jpg", source="forward")

        images.current_images.append(current_img)
        images.reply_images.append(reply_img)

        assert images.has_images
        assert images.total_count == 2
        assert len(images.all_images) == 2

    def test_all_images(self):
        """测试获取所有图片"""
        images = MessageImages(message_id="msg_123")

        img1 = ImageInfo(url="https://example.com/1.jpg")
        img2 = ImageInfo(url="https://example.com/2.jpg")
        img3 = ImageInfo(url="https://example.com/3.jpg")

        images.current_images.append(img1)
        images.reply_images.extend([img2, img3])

        all_imgs = images.all_images

        assert len(all_imgs) == 3
        assert all_imgs[0] == img1
        assert all_imgs[1] == img2
        assert all_imgs[2] == img3

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        images = MessageImages(message_id="msg_123", is_llm_trigger=True)

        img1 = ImageInfo(url="https://example.com/1.jpg")
        images.current_images.append(img1)

        data = images.to_dict()
        restored = MessageImages.from_dict(data)

        assert restored.message_id == images.message_id
        assert restored.is_llm_trigger == images.is_llm_trigger
        assert len(restored.current_images) == 1

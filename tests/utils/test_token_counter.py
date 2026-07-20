"""Token 计数工具测试"""

from iris_memory.utils.token_counter import (
    _TIKTOKEN_AVAILABLE,
    count_tokens,
    get_encoder,
    count_messages_tokens,
)


class TestTokenCounter:
    """Token 计数工具测试"""

    def test_count_tokens_english(self):
        """测试英文文本计数"""
        count = count_tokens("Hello, world!")
        assert count > 0
        # tiktoken 精确模式下为 4，字符估算模式下为 7
        if _TIKTOKEN_AVAILABLE and get_encoder() is not None:
            assert count == 4

    def test_count_tokens_chinese(self):
        """测试中文文本计数"""
        count = count_tokens("你好，世界！")
        assert count > 0

    def test_count_tokens_empty_string(self):
        """测试空字符串"""
        count = count_tokens("")
        assert count == 0

    def test_count_tokens_long_text(self):
        """测试长文本"""
        long_text = "This is a longer piece of text. " * 100
        count = count_tokens(long_text)
        assert count > 100

    def test_get_encoder_caching(self):
        """测试编码器缓存"""
        encoder1 = get_encoder("cl100k_base")
        encoder2 = get_encoder("cl100k_base")
        assert encoder1 is encoder2

    def test_count_messages_tokens_empty(self):
        """测试空消息列表"""
        count = count_messages_tokens([])
        assert count == 0

    def test_count_messages_tokens_single(self):
        """测试单条消息"""
        messages = [{"role": "user", "content": "Hello"}]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_count_messages_tokens_multiple(self):
        """测试多条消息"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_count_messages_tokens_with_format_overhead(self):
        """测试消息格式开销"""
        messages = [{"role": "user", "content": "test"}]
        count = count_messages_tokens(messages)
        assert count > 1

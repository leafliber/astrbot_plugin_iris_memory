"""画像分析器测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from iris_memory.profile.analyzer import (
    ProfileAnalyzer,
    _slim_profile_dict,
    _truncate_messages,
)
from iris_memory.profile.models import UpdateTier


class TestProfileAnalyzer:
    """画像分析器测试"""

    @pytest.fixture
    def mock_llm_manager(self):
        """创建模拟的 LLMManager"""
        manager = MagicMock()
        manager.generate_direct = AsyncMock()
        return manager

    @pytest.fixture
    def analyzer(self, mock_llm_manager):
        """创建 ProfileAnalyzer 实例"""
        return ProfileAnalyzer(mock_llm_manager)

    @pytest.mark.asyncio
    async def test_analyze_group_profile(self, analyzer, mock_llm_manager):
        """测试分析群聊画像"""
        llm_response = json.dumps(
            {"interests": ["技术", "AI"], "atmosphere_tags": ["轻松", "技术范"]},
            ensure_ascii=False,
        )
        mock_llm_manager.generate_direct.return_value = llm_response

        messages = ["今天讨论了AI技术", "yyds!", "这个方案绝了"]
        current_profile = {}

        result = await analyzer.analyze_group_profile(messages, current_profile)

        assert result["interests"] == ["技术", "AI"]
        assert result["atmosphere_tags"] == ["轻松", "技术范"]
        mock_llm_manager.generate_direct.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_user_profile(self, analyzer, mock_llm_manager):
        """测试分析用户画像"""
        llm_response = json.dumps(
            {
                "personality_tags": ["外向", "幽默"],
                "interests": ["编程", "游戏"],
                "language_style": "简洁",
            },
            ensure_ascii=False,
        )
        mock_llm_manager.generate_direct.return_value = llm_response

        messages = ["哈哈哈今天天气真好", "最近在学Python"]
        current_profile = {}

        result = await analyzer.analyze_user_profile(messages, current_profile)

        assert result["personality_tags"] == ["外向", "幽默"]
        assert result["interests"] == ["编程", "游戏"]
        assert result["language_style"] == "简洁"
        mock_llm_manager.generate_direct.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_with_llm_failure(self, analyzer, mock_llm_manager):
        """测试 LLM 调用失败"""
        mock_llm_manager.generate_direct.side_effect = Exception("LLM 调用失败")

        messages = ["测试消息"]
        current_profile = {}

        result = await analyzer.analyze_group_profile(messages, current_profile)

        # 失败时返回空字典
        assert result == {}

    @pytest.mark.asyncio
    async def test_analyze_with_invalid_json(self, analyzer, mock_llm_manager):
        """测试 LLM 返回无效 JSON"""
        mock_llm_manager.generate_direct.return_value = "这不是 JSON"

        messages = ["测试消息"]
        current_profile = {}

        result = await analyzer.analyze_group_profile(messages, current_profile)

        # 无效 JSON 时返回空字典
        assert result == {}

    def test_build_group_analysis_prompt(self, analyzer):
        """测试构建群聊分析 prompt"""
        messages = ["消息1", "消息2", "消息3"]
        current_profile = {"interests": ["技术"]}

        prompt = analyzer._build_group_analysis_prompt(messages, current_profile)

        assert "群聊画像特征" in prompt
        assert "消息1" in prompt
        assert "消息2" in prompt
        assert "技术" in prompt

    def test_build_user_analysis_prompt(self, analyzer):
        """测试构建用户分析 prompt"""
        messages = ["用户消息1", "用户消息2"]
        current_profile = {"personality_tags": ["外向"]}

        prompt = analyzer._build_user_analysis_prompt(messages, current_profile)

        assert "用户画像特征" in prompt
        assert "用户消息1" in prompt
        assert "用户消息2" in prompt
        assert "外向" in prompt

    @pytest.mark.asyncio
    async def test_group_analysis_truncates_to_newest_messages(
        self, analyzer, mock_llm_manager
    ):
        """回归：消息超过 max_messages 时应保留最新（末尾）的消息

        历史 bug：使用 ``messages[:max_messages]`` 保留最旧的 N 条，丢弃了
        最近对话。修复后使用 ``messages[-max_messages:]`` 保留最新的 N 条。
        """
        mock_llm_manager.generate_direct.return_value = "{}"
        messages = [f"message-{i}" for i in range(10)]
        current_profile = {}

        mock_config = MagicMock()
        mock_config.get = MagicMock(
            side_effect=lambda key, default=None: {
                "profile_max_messages_for_analysis": 5,
            }.get(key, default)
        )

        with patch("iris_memory.profile.analyzer.get_config", return_value=mock_config):
            await analyzer.analyze_group_profile(messages, current_profile)

        mock_llm_manager.generate_direct.assert_called_once()
        prompt = mock_llm_manager.generate_direct.call_args.kwargs["prompt"]

        # 最新的 5 条（message-5 .. message-9）应出现在 prompt 中
        for i in range(5, 10):
            assert f"message-{i}" in prompt
        # 最旧的 5 条（message-0 .. message-4）不应出现在 prompt 中
        for i in range(0, 5):
            assert f"message-{i}" not in prompt


class TestSlimProfileDict:
    """_slim_profile_dict 辅助函数测试"""

    def test_strips_metadata_keys(self):
        """剥离 field_meta/update_tracker/version"""
        profile = {
            "user_id": "u1",
            "user_name": "小明",
            "field_meta": {"k": "v"},
            "update_tracker": {"last_mid_update_time": "2024-01-01"},
            "version": 1,
        }
        slimmed = _slim_profile_dict(profile)
        assert "field_meta" not in slimmed
        assert "update_tracker" not in slimmed
        assert "version" not in slimmed
        assert slimmed["user_id"] == "u1"
        assert slimmed["user_name"] == "小明"

    def test_filters_empty_values(self):
        """过滤空值（None/空字符串/空列表/空字典）"""
        profile = {
            "user_id": "u1",
            "user_name": "",
            "interests": [],
            "occupation": None,
            "custom_fields": {},
            "language_style": "简洁",
        }
        slimmed = _slim_profile_dict(profile)
        assert "user_name" not in slimmed
        assert "interests" not in slimmed
        assert "occupation" not in slimmed
        assert "custom_fields" not in slimmed
        assert slimmed["language_style"] == "简洁"

    def test_preserves_numeric_zero(self):
        """数值 0 不被过滤（favorability=0 是有效值）"""
        profile = {
            "user_id": "u1",
            "favorability": 0,
            "emotional_baseline": "稳定",
        }
        slimmed = _slim_profile_dict(profile)
        # 0 是有效数值，保留
        assert slimmed.get("favorability") == 0
        assert slimmed["emotional_baseline"] == "稳定"


class TestTruncateMessages:
    """_truncate_messages 辅助函数测试"""

    def test_no_truncation_when_max_chars_zero(self):
        """max_chars=0 不截断"""
        messages = ["短消息", "一条很长的消息" * 100]
        result = _truncate_messages(messages, 0)
        assert result == messages

    def test_truncates_long_messages(self):
        """超长消息被截断并加省略号"""
        messages = ["短消息", "x" * 200]
        result = _truncate_messages(messages, 50)
        assert result[0] == "短消息"
        assert len(result[1]) == 51  # 50 字符 + 省略号
        assert result[1].endswith("…")

    def test_short_messages_unchanged(self):
        """短于 max_chars 的消息不变"""
        messages = ["abc", "def"]
        result = _truncate_messages(messages, 100)
        assert result == messages


class TestFavorabilityDeltaInPrompt:
    """好感度 delta 在 prompt 中的测试"""

    @pytest.fixture
    def mock_llm_manager(self):
        manager = MagicMock()
        manager.generate_direct = AsyncMock()
        return manager

    @pytest.fixture
    def analyzer(self, mock_llm_manager):
        return ProfileAnalyzer(mock_llm_manager)

    def test_mid_prompt_includes_favorability_delta_when_enabled(self, analyzer):
        """favorability_enable=True 时 MID prompt 包含 favorability_delta"""
        mock_config = MagicMock()
        mock_config.get = MagicMock(
            side_effect=lambda key, default=None: {
                "profile.favorability_enable": True,
                "profile_message_max_chars": 0,
                "profile_max_messages_for_user_analysis": 30,
            }.get(key, default)
        )
        with patch("iris_memory.profile.analyzer.get_config", return_value=mock_config):
            prompt = analyzer._build_user_analysis_prompt(["消息"], {}, UpdateTier.MID)
        assert "favorability_delta" in prompt

    def test_mid_prompt_excludes_favorability_delta_when_disabled(self, analyzer):
        """favorability_enable=False 时 MID prompt 不包含 favorability_delta"""
        mock_config = MagicMock()
        mock_config.get = MagicMock(
            side_effect=lambda key, default=None: {
                "profile.favorability_enable": False,
                "profile_message_max_chars": 0,
                "profile_max_messages_for_user_analysis": 30,
            }.get(key, default)
        )
        with patch("iris_memory.profile.analyzer.get_config", return_value=mock_config):
            prompt = analyzer._build_user_analysis_prompt(["消息"], {}, UpdateTier.MID)
        assert "favorability_delta" not in prompt

    def test_long_prompt_never_includes_favorability_delta(self, analyzer):
        """LONG prompt 不包含 favorability_delta（好感度是 MID 层职责）"""
        mock_config = MagicMock()
        mock_config.get = MagicMock(
            side_effect=lambda key, default=None: {
                "profile.favorability_enable": True,
                "profile_message_max_chars": 0,
                "profile_max_messages_for_user_analysis": 30,
            }.get(key, default)
        )
        with patch("iris_memory.profile.analyzer.get_config", return_value=mock_config):
            prompt = analyzer._build_user_analysis_prompt(["消息"], {}, UpdateTier.LONG)
        assert "favorability_delta" not in prompt

    def test_combined_prompt_includes_favorability_delta(self, analyzer):
        """combined prompt 包含 favorability_delta 和长期字段"""
        mock_config = MagicMock()
        mock_config.get = MagicMock(
            side_effect=lambda key, default=None: {
                "profile.favorability_enable": True,
                "profile_message_max_chars": 0,
                "profile_max_messages_for_user_analysis": 30,
            }.get(key, default)
        )
        with patch("iris_memory.profile.analyzer.get_config", return_value=mock_config):
            prompt = analyzer._build_user_analysis_prompt(
                ["消息"], {}, UpdateTier.MID, combined=True
            )
        assert "favorability_delta" in prompt
        assert "occupation" in prompt
        assert "bot_relationship" in prompt
        assert "important_events" in prompt

    @pytest.mark.asyncio
    async def test_analyze_user_profile_combined_calls_llm_once(
        self, analyzer, mock_llm_manager
    ):
        """combined=True 触发单次 LLM 调用返回 MID+LONG 字段"""
        llm_response = json.dumps(
            {
                "personality_tags": ["外向"],
                "interests": ["编程"],
                "language_style": "简洁",
                "communication_style": "随意",
                "emotional_baseline": "稳定",
                "favorability_delta": 5,
                "occupation": "工程师",
                "bot_relationship": "朋友",
                "important_events": ["入职"],
                "taboo_topics": [],
                "important_dates": [],
                "custom_fields": {},
            },
            ensure_ascii=False,
        )
        mock_llm_manager.generate_direct.return_value = llm_response

        result = await analyzer.analyze_user_profile(
            ["消息"], {}, tier=UpdateTier.MID, combined=True
        )

        assert result["favorability_delta"] == 5
        assert result["occupation"] == "工程师"
        mock_llm_manager.generate_direct.assert_called_once()

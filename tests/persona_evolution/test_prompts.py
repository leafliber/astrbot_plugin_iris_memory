"""三阶段提示词安全约束测试（文档 §2.5/§9）"""

from iris_memory.persona_evolution.prompts import (
    build_analysis_prompt,
    build_generation_prompt,
    build_review_prompt,
)


class TestAnalysisPrompt:
    def test_corpus_wrapped_and_marked_untrusted(self):
        prompt = build_analysis_prompt(["你好呀", "今晚吃什么"], "自然拟人")
        assert "<corpus>" in prompt and "</corpus>" in prompt
        assert "不可信数据" in prompt
        assert "你好呀" in prompt
        assert "JSON" in prompt

    def test_corpus_only_in_data_block(self):
        # 指令部分（<corpus> 之前）不得出现语料原文
        prompt = build_analysis_prompt(["独特的语料标记xyz"], "目标")
        instruction_part = prompt.split("<corpus>")[0]
        assert "独特的语料标记xyz" not in instruction_part


class TestGenerationPrompt:
    def test_no_raw_corpus(self):
        # 阶段 B 只接收结构化画像，提示词中不得出现原始语料
        prompt = build_generation_prompt(
            current_prompt="你是 Iris。",
            edit_mode="managed_block",
            goal_snapshot={"text": "自然拟人"},
            style_profile={"tone": ["轻松"]},
            protected_fragments=[],
            block_max_chars=1500,
            max_change_ratio=0.2,
            full_max_growth_ratio=1.25,
            full_max_length=20000,
        )
        assert "<corpus>" not in prompt
        assert "<style_profile>" in prompt
        assert "轻松" in prompt  # 画像内容在
        assert "JSON" in prompt

    def test_protected_fragments_listed(self):
        prompt = build_generation_prompt(
            current_prompt="你是 Iris。",
            edit_mode="full_prompt",
            goal_snapshot={"text": ""},
            style_profile={},
            protected_fragments=["核心身份片段"],
            block_max_chars=1500,
            max_change_ratio=0.2,
            full_max_growth_ratio=1.25,
            full_max_length=20000,
        )
        assert "核心身份片段" in prompt
        assert "完整人格" in prompt


class TestReviewPrompt:
    def test_self_contained_no_corpus(self):
        prompt = build_review_prompt(
            base_prompt="旧人格",
            candidate_prompt="新人格",
            goal_snapshot={"text": "目标"},
            style_profile={"tone": []},
        )
        assert "<corpus>" not in prompt
        assert "旧人格" in prompt and "新人格" in prompt
        assert "prompt_injection_suspected" in prompt

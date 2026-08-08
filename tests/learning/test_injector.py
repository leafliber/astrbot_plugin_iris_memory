"""学习上下文注入组装测试"""

from unittest.mock import MagicMock, patch

import pytest

from iris_memory.learning.injector import build_learning_context
from iris_memory.learning.jargon import JargonLearner


def _event(text=""):
    event = MagicMock()
    event.message_str = text
    return event


def _patch_adapter():
    """patch injector 的平台适配层，固定群 ID"""
    adapter = MagicMock()
    adapter.get_session_id.return_value = "sess1"
    adapter.get_group_id.return_value = "g1"
    return patch("iris_memory.learning.injector.get_adapter", return_value=adapter)


def _seed_approved(storage):
    """准备 approved 数据：1 表达模式 + 1 黑话 + 1 对话样例"""
    pat = storage.insert_pattern("g1", "chat", "早上好呀")
    storage.update_status("expression_pattern", [pat], "approved")
    storage.upsert_jargon_count("g1", "yyds", 5)
    term = storage.get_jargon_terms_for_inference([3])[0]
    storage.mark_jargon_inferred(term["id"], "永远的神", 0.9)
    pid = storage.insert_pair("g1", "u1", "早", "早呀")
    storage.update_status("few_shot", [pid], "approved")


class TestEmpty:
    """空内容与 pending 过滤"""

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty(self, config, storage):
        meta = {}
        with _patch_adapter():
            text = await build_learning_context(
                _event("你好"), storage, JargonLearner(storage), meta
            )
        assert text == ""
        assert meta["skipped"] == "empty"

    @pytest.mark.asyncio
    async def test_pending_only_returns_empty(self, config, storage):
        storage.insert_pair("g1", "u1", "早", "早呀")  # pending_review
        storage.insert_pattern("g1", "chat", "早上好呀")  # pending_review
        meta = {}
        with _patch_adapter():
            text = await build_learning_context(
                _event("你好"), storage, JargonLearner(storage), meta
            )
        assert text == ""
        assert meta["skipped"] == "empty"


class TestAssembly:
    """approved 数据组装"""

    @pytest.mark.asyncio
    async def test_sections_assembled(self, config, storage):
        _seed_approved(storage)
        meta = {}
        with _patch_adapter():
            text = await build_learning_context(
                _event("这波 yyds"), storage, JargonLearner(storage), meta
            )
        assert "## 群聊用语与表达风格" in text
        assert "### 群黑话" in text and "yyds = 永远的神" in text
        assert "### 常用表达" in text and "早上好呀" in text
        assert "### 对话风格样例" in text and "早呀" in text
        assert meta["pattern_count"] == 1
        assert meta["jargon_count"] == 1
        assert meta["few_shot_count"] == 1
        assert meta["budget_tokens"] == 600
        assert meta["used_tokens"] > 0

    @pytest.mark.asyncio
    async def test_pattern_hit_recorded(self, config, storage):
        _seed_approved(storage)
        with _patch_adapter():
            await build_learning_context(
                _event("你好"), storage, JargonLearner(storage), {}
            )
        row = storage.get_approved_patterns("g1", 5)[0]
        assert row["hit_count"] == 1
        assert row["last_hit_at"] is not None

    @pytest.mark.asyncio
    async def test_jargon_only_when_message_hits(self, config, storage):
        _seed_approved(storage)
        with _patch_adapter():
            text = await build_learning_context(
                _event("没有命中词"), storage, JargonLearner(storage), {}
            )
        assert "### 群黑话" not in text
        assert "### 常用表达" in text


class TestTruncation:
    """单条截断与整体预算裁剪"""

    @pytest.mark.asyncio
    async def test_item_truncation(self, config, storage):
        config.set_hidden("learning_inject_max_item_chars", 5)
        pat = storage.insert_pattern("g1", "chat", "这是一个很长很长的表达句式")
        storage.update_status("expression_pattern", [pat], "approved")
        with _patch_adapter():
            text = await build_learning_context(
                _event("你好"), storage, JargonLearner(storage), {}
            )
        assert "这是一个很…" in text
        assert "很长的表达句式" not in text

    @pytest.mark.asyncio
    async def test_budget_trim_order(self, config, storage, monkeypatch):
        """整体超预算时按 few_shot → pattern → jargon 顺序裁剪"""
        # 用字符数代替 token 估算，便于精确控制预算
        monkeypatch.setattr(
            "iris_memory.learning.injector.count_tokens", lambda t: len(t)
        )
        _seed_approved(storage)
        config.set_hidden("learning_inject_max_tokens", 60)
        meta = {}
        with _patch_adapter():
            text = await build_learning_context(
                _event("这波 yyds"), storage, JargonLearner(storage), meta
            )
        # few_shot 最先被裁掉，pattern 与 jargon 保留
        assert "### 对话风格样例" not in text
        assert "### 常用表达" in text
        assert "### 群黑话" in text
        assert meta["few_shot_count"] == 0
        assert meta["pattern_count"] == 1
        assert meta["jargon_count"] == 1
        assert meta["dropped_by_budget"] == 1

    @pytest.mark.asyncio
    async def test_budget_trim_to_jargon_only(self, config, storage, monkeypatch):
        monkeypatch.setattr(
            "iris_memory.learning.injector.count_tokens", lambda t: len(t)
        )
        _seed_approved(storage)
        config.set_hidden("learning_inject_max_tokens", 40)
        meta = {}
        with _patch_adapter():
            text = await build_learning_context(
                _event("这波 yyds"), storage, JargonLearner(storage), meta
            )
        assert "### 对话风格样例" not in text
        assert "### 常用表达" not in text
        assert "### 群黑话" in text
        assert meta["dropped_by_budget"] == 2

    @pytest.mark.asyncio
    async def test_budget_exceeded_returns_empty(self, config, storage, monkeypatch):
        monkeypatch.setattr(
            "iris_memory.learning.injector.count_tokens", lambda t: len(t)
        )
        _seed_approved(storage)
        config.set_hidden("learning_inject_max_tokens", 1)  # 连 header 都放不下
        meta = {}
        with _patch_adapter():
            text = await build_learning_context(
                _event("这波 yyds"), storage, JargonLearner(storage), meta
            )
        assert text == ""
        assert meta["skipped"] == "budget_exceeded"

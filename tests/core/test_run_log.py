"""运行日志模块测试"""

import pytest

from iris_memory.core.run_log import (
    RunLogManager,
    get_run_log_manager,
    reset_run_log_manager,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_run_log_manager()
    yield
    reset_run_log_manager()


class TestRunLogManager:
    def test_record_and_get(self):
        manager = RunLogManager()
        manager.record("llm_call", "测试调用成功", module="test", tokens=10)

        entries = manager.get_entries()
        assert len(entries) == 1
        entry = entries[0]
        assert entry["type"] == "llm_call"
        assert entry["type_label"] == "LLM 调用"
        assert entry["title"] == "测试调用成功"
        assert entry["success"] is True
        assert entry["detail"]["module"] == "test"
        assert entry["detail"]["tokens"] == 10
        assert entry["id"] == 1
        assert entry["timestamp"]

    def test_entries_sorted_newest_first(self):
        manager = RunLogManager()
        manager.record("llm_call", "第一条")
        manager.record("injection", "第二条")
        manager.record("proactive", "第三条")

        entries = manager.get_entries()
        assert [e["title"] for e in entries] == ["第三条", "第二条", "第一条"]

    def test_filter_by_type(self):
        manager = RunLogManager()
        manager.record("llm_call", "LLM 1")
        manager.record("llm_call", "LLM 2")
        manager.record("injection", "注入 1")

        entries = manager.get_entries(log_type="llm_call")
        assert len(entries) == 2
        assert all(e["type"] == "llm_call" for e in entries)

    def test_limit(self):
        manager = RunLogManager()
        for i in range(5):
            manager.record("llm_call", f"调用 {i}")

        entries = manager.get_entries(limit=2)
        assert len(entries) == 2
        assert entries[0]["title"] == "调用 4"

    def test_per_type_max_entries_default_10(self):
        manager = RunLogManager()
        for i in range(15):
            manager.record("llm_call", f"调用 {i}")
        for i in range(3):
            manager.record("injection", f"注入 {i}")

        assert manager.get_counts()["llm_call"] == 10
        assert manager.get_counts()["injection"] == 3

        entries = manager.get_entries(log_type="llm_call")
        assert entries[-1]["title"] == "调用 5"

    def test_long_string_truncated(self):
        manager = RunLogManager()
        long_text = "x" * 5000
        manager.record("llm_call", "长文本", prompt=long_text)

        entry = manager.get_entries()[0]
        prompt = entry["detail"]["prompt"]
        assert len(prompt) < 5000
        assert "截断，原始 5000 字符" in prompt

    def test_nested_truncation(self):
        manager = RunLogManager()
        manager.record(
            "injection",
            "嵌套",
            sections={"l1": {"content": "y" * 3000}},
            items=["z" * 3000],
        )

        entry = manager.get_entries()[0]
        assert "截断，原始 3000 字符" in entry["detail"]["sections"]["l1"]["content"]
        assert "截断，原始 3000 字符" in entry["detail"]["items"][0]

    def test_unknown_type_ignored(self):
        manager = RunLogManager()
        manager.record("unknown_type", "不应记录")
        assert manager.get_entries() == []

    def test_clear_all(self):
        manager = RunLogManager()
        manager.record("llm_call", "A")
        manager.record("injection", "B")

        cleared = manager.clear()
        assert cleared == 2
        assert manager.get_entries() == []

    def test_clear_by_type(self):
        manager = RunLogManager()
        manager.record("llm_call", "A")
        manager.record("injection", "B")

        cleared = manager.clear("llm_call")
        assert cleared == 1
        remaining = manager.get_entries()
        assert len(remaining) == 1
        assert remaining[0]["type"] == "injection"

    def test_counts(self):
        manager = RunLogManager()
        manager.record("llm_call", "A")
        manager.record("llm_call", "B")
        manager.record("proactive", "C")

        counts = manager.get_counts()
        assert counts == {"llm_call": 2, "injection": 0, "proactive": 1}

    def test_failure_status_recorded(self):
        manager = RunLogManager()
        manager.record("llm_call", "调用失败", success=False, error="超时")

        entry = manager.get_entries()[0]
        assert entry["success"] is False
        assert entry["detail"]["error"] == "超时"


class TestSingleton:
    def test_get_run_log_manager_returns_same_instance(self):
        m1 = get_run_log_manager()
        m2 = get_run_log_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_run_log_manager()
        reset_run_log_manager()
        m2 = get_run_log_manager()
        assert m1 is not m2

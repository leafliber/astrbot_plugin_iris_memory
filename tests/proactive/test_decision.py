"""proactive.decision.DecisionCore 测试

覆盖 build_prompt 组装：willingness 人格、thread 锚点块、motive 指令、
token 截断；以及 decide 的成功 / 异常 / 非法动机路径。
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from iris_memory.proactive.decision import (
    DecisionCore,
    DecisionRequest,
    build_anchor_block,
    classify_decision_error,
)
from iris_memory.proactive.perception import ContextPackager, SlidingWindow
from iris_memory.proactive.state import StateManager, ThreadAnchor

GID = "g1"


def _core(nm_config, overrides=None):
    cm = nm_config(overrides=overrides)
    st = StateManager(cm)
    win = SlidingWindow(cm)
    pk = ContextPackager(cm)
    return cm, st, win, DecisionCore(cm, st, win, pk)


def _req(motive="chime_in", wake="message", **kwargs):
    return DecisionRequest(group_id=GID, wake=wake, motive=motive, **kwargs)


class TestWillingnessPersona:
    def test_medium_default(self, nm_config):
        _, _, _, core = _core(nm_config)
        user_prompt, system_prompt = core.build_prompt(_req())
        assert "友善、适度的参与者" in user_prompt
        assert "适度参与的群成员" in system_prompt

    def test_low_persona(self, nm_config):
        _, st, _, core = _core(nm_config)
        st.set_willingness(GID, "low")
        user_prompt, system_prompt = core.build_prompt(_req())
        assert "安静、克制" in user_prompt
        assert "安静的观察者" in system_prompt

    def test_high_persona(self, nm_config):
        _, st, _, core = _core(nm_config)
        st.set_willingness(GID, "high")
        user_prompt, system_prompt = core.build_prompt(_req())
        assert "活跃、热情" in user_prompt
        assert "活跃的群成员" in system_prompt


class TestAnchorBlock:
    def test_empty_anchor_no_block(self, nm_config):
        assert build_anchor_block(ThreadAnchor(), "chime_in") == ""
        _, _, _, core = _core(nm_config)
        user_prompt, _ = core.build_prompt(_req())
        assert "<thread>" not in user_prompt

    def test_anchor_block_content(self):
        anchor = ThreadAnchor(
            kind="chime_in",
            bot_message="周末去哪玩",
            participants={"u2", "u1"},
            keywords={"露营"},
            reason="感兴趣",
        )
        text = build_anchor_block(anchor, "chime_in")
        assert text.startswith("\n\n<thread>")
        assert text.endswith("</thread>")
        assert '你之前在群里说："周末去哪玩"' in text
        assert "你关注这些用户：u1, u2" in text
        assert "你关注这些关键词：露营" in text
        assert "原因：感兴趣" in text
        # 非 follow_up 动机不带进展提示
        assert "新进展" not in text

    def test_follow_up_motive_appends_progress_hint(self):
        anchor = ThreadAnchor(kind="follow_up", bot_message="hi")
        text = build_anchor_block(anchor, "follow_up")
        assert "现在相关对话有了新进展" in text

    def test_anchor_in_prompt(self, nm_config):
        _, st, _, core = _core(nm_config)
        st.write_anchor(GID, kind="chime_in", bot_message="去哪玩", users=["u1"])
        user_prompt, _ = core.build_prompt(_req())
        assert "<thread>" in user_prompt
        assert "去哪玩" in user_prompt


class TestMotiveInstructions:
    def test_chime_in_instruction(self, nm_config):
        _, _, _, core = _core(nm_config)
        user_prompt, _ = core.build_prompt(_req("chime_in"))
        assert "<instruction>" in user_prompt
        assert "本次为常规采样评估" in user_prompt

    def test_follow_up_instruction(self, nm_config):
        _, _, _, core = _core(nm_config)
        user_prompt, _ = core.build_prompt(_req("follow_up"))
        assert "本次为跟进评估" in user_prompt

    def test_watch_instruction(self, nm_config):
        _, _, _, core = _core(nm_config)
        user_prompt, _ = core.build_prompt(_req("watch"))
        assert "本次为被动回复后的跟进评估" in user_prompt

    def test_initiate_instruction_with_quiet_minutes(self, nm_config):
        _, _, _, core = _core(nm_config)
        user_prompt, _ = core.build_prompt(_req("initiate", wake="timer", quiet_minutes=42))
        assert "群里已经安静了 42 分钟" in user_prompt
        # medium 意愿的发起风格
        assert "偶尔可以在群冷场时开启话题" in user_prompt

    def test_initiate_custom_instruction(self, nm_config):
        _, _, _, core = _core(nm_config, {"proactive_instruction": "多聊技术话题"})
        user_prompt, _ = core.build_prompt(_req("initiate", wake="timer", quiet_minutes=10))
        assert "话题倾向：多聊技术话题" in user_prompt


class TestObservationBlock:
    def test_observation_in_prompt(self, nm_config):
        _, st, _, core = _core(nm_config)
        st.set_observation(GID, "在聊游戏")
        user_prompt, _ = core.build_prompt(_req())
        assert "<recent_observation>之前的观察：在聊游戏</recent_observation>" in user_prompt

    def test_no_observation_no_block(self, nm_config):
        _, _, _, core = _core(nm_config)
        user_prompt, _ = core.build_prompt(_req())
        assert "<recent_observation>" not in user_prompt


class TestTokenTruncation:
    def test_oldest_messages_dropped(self, nm_config):
        # window_size=30 容纳全部消息，max_token=1000（属性钳制下限）触发截断
        _, st, win, core = _core(nm_config, {"window_size": 30, "max_token": 1000})
        for i in range(30):
            win.append(GID, _msg(f"User{i}", f"u{i}", "word " * 50))
        user_prompt, _ = core.build_prompt(_req())
        assert '<iris:reply-context motive="chime_in">' in user_prompt
        assert "</iris:reply-context>" in user_prompt
        # 最早的消息被丢弃，最新的保留
        assert "[User0(u0)]" not in user_prompt
        assert "[User29(u29)]" in user_prompt

    def test_no_truncation_within_budget(self, nm_config):
        _, _, win, core = _core(nm_config)
        win.append(GID, _msg("User1", "u1", "短消息"))
        user_prompt, _ = core.build_prompt(_req())
        assert "[User1(u1)] 短消息" in user_prompt


def _msg(name, uid, content):
    from iris_memory.proactive.perception import WindowMessage

    return WindowMessage(
        sender_id=uid,
        sender_name=name,
        content=content,
        timestamp=time.time(),
    )


class TestSelfMarking:
    def test_own_message_marked_as_self(self, nm_config):
        cm = nm_config()
        st = StateManager(cm)
        win = SlidingWindow(cm)
        pk = ContextPackager(cm, self_id_get=lambda: "bot1")
        core = DecisionCore(cm, st, win, pk)
        win.append(GID, _msg("我", "bot1", "我先说的"))
        win.append(GID, _msg("User1", "u1", "好的"))
        user_prompt, _ = core.build_prompt(_req())
        assert "[我(bot1)] 我先说的" in user_prompt
        assert "[User1(u1)] 好的" in user_prompt

    def test_legacy_hardcoded_name_overridden_by_self_id(self, nm_config):
        cm = nm_config()
        st = StateManager(cm)
        win = SlidingWindow(cm)
        pk = ContextPackager(cm, self_id_get=lambda: "bot1")
        core = DecisionCore(cm, st, win, pk)
        win.append(GID, _msg("Iris", "bot1", "旧记录"))
        user_prompt, _ = core.build_prompt(_req())
        assert "[我(bot1)] 旧记录" in user_prompt
        assert "[Iris(bot1)]" not in user_prompt

    def test_unknown_self_id_keeps_original_name(self, nm_config):
        _, _, win, core = _core(nm_config)
        win.append(GID, _msg("User1", "u1", "hi"))
        user_prompt, _ = core.build_prompt(_req())
        assert "[User1(u1)] hi" in user_prompt

    def test_decision_system_prompt_identifies_self(self, nm_config):
        _, st, _, core = _core(nm_config)
        for level in ("low", "medium", "high"):
            st.set_willingness(GID, level)
            _, system_prompt = core.build_prompt(_req())
            assert "名字为「我」的条目" in system_prompt
            assert "不存在任何人替你代答" in system_prompt


class TestTimeHintInjection:
    def test_time_hint_prepended_to_system_prompt(self, nm_config):
        core = _core_with_hint(nm_config, "Current datetime: 2026-07-30 10:00 (CST), Weekday: Thursday")
        _, system_prompt = core.build_prompt(_req())
        assert system_prompt.startswith(
            "<system_reminder>Current datetime: 2026-07-30 10:00 (CST), Weekday: Thursday</system_reminder>"
        )
        assert "适度参与的群成员" in system_prompt

    def test_empty_time_hint_leaves_system_prompt_untouched(self, nm_config):
        core = _core_with_hint(nm_config, "")
        _, system_prompt = core.build_prompt(_req())
        assert "<system_reminder>" not in system_prompt
        assert system_prompt.startswith("你正在观察一个群聊")

    def test_no_time_hint_get_default(self, nm_config):
        _, _, _, core = _core(nm_config)
        _, system_prompt = core.build_prompt(_req())
        assert "<system_reminder>" not in system_prompt


def _core_with_hint(nm_config, hint):
    cm = nm_config()
    st = StateManager(cm)
    win = SlidingWindow(cm)
    pk = ContextPackager(cm)
    return DecisionCore(cm, st, win, pk, time_hint_get=lambda gid: hint)


class TestDecide:
    @pytest.mark.asyncio
    async def test_success(self, nm_config):
        _, _, _, core = _core(nm_config)
        generate_direct = AsyncMock(
            return_value='{"action": "speak", "obs": "ok"}'
        )
        llm_manager = SimpleNamespace(generate_direct=generate_direct)
        outcome = await core.decide(_req(), llm_manager, "p1")
        assert outcome.error == ""
        assert outcome.decision is not None
        assert outcome.decision.should_speak is True
        assert outcome.decision.mode == "chime_in"
        assert outcome.raw_text == '{"action": "speak", "obs": "ok"}'
        assert outcome.duration_ms >= 0
        assert outcome.system_prompt and outcome.user_prompt
        generate_direct.assert_awaited_once()
        kwargs = generate_direct.await_args.kwargs
        assert kwargs["provider_id"] == "p1"
        assert kwargs["module"] == "proactive_decision_chime_in"

    @pytest.mark.asyncio
    async def test_llm_error_returned_not_raised(self, nm_config):
        _, _, _, core = _core(nm_config)
        llm_manager = SimpleNamespace(
            generate_direct=AsyncMock(side_effect=RuntimeError("boom"))
        )
        outcome = await core.decide(_req(), llm_manager, "p1")
        assert outcome.decision is None
        assert outcome.error == "boom"
        assert outcome.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_1026_is_non_retryable_and_records_dynamic_sources(self, nm_config):
        from iris_memory.core.run_log import (
            get_run_log_manager,
            reset_run_log_manager,
        )

        reset_run_log_manager()
        _, state, window, core = _core(nm_config)
        state.set_observation(GID, "正在讨论一条消息")
        state.write_anchor(GID, kind="chime_in", bot_message="此前回复")
        window.append(GID, _msg("User1", "u1", "触发安全检查的原始消息"))
        error = RuntimeError("input new_sensitive (1026)")

        outcome = await core.decide(
            _req(),
            SimpleNamespace(generate_direct=AsyncMock(side_effect=error)),
            "minimax-provider",
        )

        assert outcome.error_kind == "input_content_safety_1026"
        assert outcome.retryable is False
        assert {item["source"] for item in outcome.dynamic_context_sources} >= {
            "recent_observation",
            "thread.bot_message",
            "window_message",
        }
        window_source = next(
            item
            for item in outcome.dynamic_context_sources
            if item["source"] == "window_message"
        )
        assert window_source["content"] == "触发安全检查的原始消息"
        assert window_source["sender_id"] == "u1"

        log = get_run_log_manager().get_entries("proactive", limit=1)[0]
        detail = log["detail"]
        assert detail["user_prompt"] == "[Provider 输入安全过滤拒绝，动态 prompt 已脱敏]"
        assert "触发安全检查的原始消息" not in str(detail)
        logged_window = next(
            item
            for item in detail["dynamic_context_sources"]
            if item["source"] == "window_message"
        )
        assert logged_window["redacted"] is True
        assert "content" not in logged_window
        assert "sender_id" not in logged_window

    def test_error_classifier_keeps_other_provider_errors_retryable(self):
        assert classify_decision_error("Error code: 422 input new_sensitive (1026)") == (
            "input_content_safety_1026",
            False,
        )
        assert classify_decision_error("timeout") == ("provider_error", True)

    @pytest.mark.asyncio
    async def test_invalid_motive_asserts(self, nm_config):
        _, _, _, core = _core(nm_config)
        llm_generate = AsyncMock()
        with pytest.raises(AssertionError):
            await core.decide(_req("bogus"), llm_generate, "p1")

"""proactive.decision.DecisionCore 测试

覆盖 build_prompt 组装：willingness 人格、thread 锚点块、motive 指令、
token 截断；以及 decide 的成功 / 异常 / 非法动机路径。
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from iris_memory.proactive.decision import DecisionCore, DecisionRequest, build_anchor_block
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


class TestDecide:
    @pytest.mark.asyncio
    async def test_success(self, nm_config):
        _, _, _, core = _core(nm_config)
        llm_generate = AsyncMock(
            return_value=SimpleNamespace(
                completion_text='{"action": "speak", "obs": "ok"}'
            )
        )
        outcome = await core.decide(_req(), llm_generate, "p1")
        assert outcome.error == ""
        assert outcome.decision is not None
        assert outcome.decision.should_speak is True
        assert outcome.decision.mode == "chime_in"
        assert outcome.raw_text == '{"action": "speak", "obs": "ok"}'
        assert outcome.duration_ms >= 0
        assert outcome.system_prompt and outcome.user_prompt
        llm_generate.assert_awaited_once()
        kwargs = llm_generate.await_args.kwargs
        assert kwargs["chat_provider_id"] == "p1"

    @pytest.mark.asyncio
    async def test_llm_error_returned_not_raised(self, nm_config):
        _, _, _, core = _core(nm_config)
        llm_generate = AsyncMock(side_effect=RuntimeError("boom"))
        outcome = await core.decide(_req(), llm_generate, "p1")
        assert outcome.decision is None
        assert outcome.error == "boom"
        assert outcome.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_invalid_motive_asserts(self, nm_config):
        _, _, _, core = _core(nm_config)
        llm_generate = AsyncMock()
        with pytest.raises(AssertionError):
            await core.decide(_req("bogus"), llm_generate, "p1")

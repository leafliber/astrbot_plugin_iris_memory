"""main.py 连续性接线：回执才写 L1（T03）、手动发言、只读频道拦截、重放不恢复、合批。"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import astrbot_plugin_iris_memory.iris_memory.core.persona as persona_mod
import astrbot_plugin_iris_memory.iris_memory.platform as platform_mod
import pytest
import pytest_asyncio
from astrbot_plugin_iris_memory.iris_memory.continuity import models as m
from astrbot_plugin_iris_memory.iris_memory.continuity.service import ContinuityService
from astrbot_plugin_iris_memory.main import IrisMemoryPlugin

OWNER = "100000001"
SELF_ID = "200000002"
SCOPE = f"private:{OWNER}"
CHANNEL = "-1001000000001"
GROUP = "-1001000000002"


class FakeConfig:
    def __init__(self, **continuity) -> None:
        self.values = {"proactive.silent_sessions": [CHANNEL, "-100555"]}
        for k, v in continuity.items():
            self.values[f"continuity.{k}"] = v

    def get(self, key, default=None):
        return self.values.get(key, default)


class FakeResult:
    def __init__(self, text: str) -> None:
        self.chain = [SimpleNamespace(text=text)]
        self._text = text

    def get_plain_text(self) -> str:
        return self._text if self.chain else ""


class FakeEvent:
    def __init__(
        self,
        *,
        text: str = "",
        sender: str = OWNER,
        group_id: str = "",
        msg_id: int | None = 100,
        out: bool = False,
        session_id: str | None = None,
        platform: str = "telethon_userbot",
    ) -> None:
        self.message_str = text
        raw = SimpleNamespace(
            id=msg_id, date=datetime.now(timezone.utc), out=out, reply_to=None
        )
        self.message_obj = SimpleNamespace(
            message_id=str(msg_id) if msg_id is not None else "",
            raw_message=raw,
            session_id=session_id or (group_id or sender),
            message_str=text,
            timestamp=int(time.time()),
        )
        self._sender = sender
        self._group = group_id
        self._platform = platform
        self.extras: dict = {}
        self.stopped = False
        self.result: FakeResult | None = None
        self.unified_msg_origin = (
            f"userbot:GroupMessage:{group_id}" if group_id else f"userbot:FriendMessage:{sender}"
        )
        self.is_at_or_wake_command = True

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def stop_event(self):
        self.stopped = True

    def is_stopped(self):
        return self.stopped

    def get_result(self):
        return self.result

    def get_group_id(self):
        return self._group

    def is_private_chat(self):
        return not self._group

    def get_sender_id(self):
        return self._sender

    def get_self_id(self):
        return SELF_ID

    def get_sender_name(self):
        return "她"

    def get_platform_name(self):
        return self._platform

    def get_platform_id(self):
        return "userbot"

    def get_messages(self):
        return []


class FakeState:
    def __init__(self) -> None:
        self.resume_calls: list[str] = []

    def get_whitelist(self):
        return [SCOPE]

    def resume_after_owner_message(self, gid):
        self.resume_calls.append(gid)
        return True

    def is_whitelisted(self, gid):
        return False


@pytest_asyncio.fixture
async def plugin(tmp_path, monkeypatch):
    cfg = FakeConfig(merge_quiet_ms=200, merge_cap_ms=1000)
    receipt_platform = SimpleNamespace(send_with_receipt=AsyncMock(return_value=[1]))
    context = SimpleNamespace(get_platform_inst=lambda pid: receipt_platform)
    state = FakeState()
    svc = ContinuityService(
        context=context,
        config_source=cfg,
        state=state,
        owner_ids_get=lambda: [OWNER],
        silent_sessions_get=lambda: cfg.get("proactive.silent_sessions", []),
        data_dir=tmp_path,
        llm_manager=SimpleNamespace(generate=AsyncMock(return_value='{"ops":[]}')),
    )
    await svc.start()

    p = object.__new__(IrisMemoryPlugin)
    p.context = context
    p.config = cfg
    p._continuity = svc
    p._state = state
    p._llm_manager = None
    p._passive_active = {}
    p._reply_in_progress = {}
    p._triggering = {}
    p._reply_config = SimpleNamespace(enabled=False, private_whitelist=[OWNER])
    l1 = SimpleNamespace(add_message=AsyncMock(return_value=True))
    p.component_manager = SimpleNamespace(
        get_available_component=lambda name: l1 if name == "l1_buffer" else None
    )
    p.l1 = l1

    adapter = SimpleNamespace(get_session_id=lambda e: SCOPE if e.is_private_chat() else e.get_group_id())
    monkeypatch.setattr(platform_mod, "get_adapter", lambda e: adapter)
    monkeypatch.setattr(persona_mod, "resolve_persona", AsyncMock(return_value="default"))
    yield p
    await svc.stop()


def _response(text: str):
    return SimpleNamespace(completion_text=text, usage=None)


# ---------------------------------------------------------------- T03


@pytest.mark.asyncio
async def test_t03_assistant_written_to_l1_only_with_receipt(plugin):
    store = plugin._continuity.store
    ev = FakeEvent(text="周五考试，记得问我", msg_id=200)
    plugin._passive_active[SCOPE] = time.time()
    await plugin.on_llm_response(ev, _response("好，到时候我问你。"))

    # 未发送前不写 L1；私聊 passive 标记被释放（_passive_active 泄漏修复）
    plugin.l1.add_message.assert_not_awaited()
    assert SCOPE not in plugin._passive_active
    assert ev.extras["iris_pending_assistant"]["content"] == "好，到时候我问你。"

    ev.result = FakeResult("好，到时候我问你。")
    ev.extras["telethon_sent_message_ids"] = [9001]
    await plugin.on_message_sent(ev)

    plugin.l1.add_message.assert_awaited_once()
    kwargs = plugin.l1.add_message.await_args.kwargs
    assert kwargs["role"] == "assistant" and kwargs["group_id"] == SCOPE
    assert kwargs["metadata"] == {"message_ids": [9001], "turn": "reply:200"}
    row = store.outbound_by_tg_id(SCOPE, 9001)
    assert row["status"] == m.OUT_SENT and row["origin"] == "reply" and row["author"] == "bot"
    assert row["reply_to_msg_id"] == 200
    # 承诺检测（有回执）
    assert [i["kind"] for i in store.list_items(scope_key=SCOPE)] == [m.KIND_BOT_PROMISE]

    # 同一事件重复 after_message_sent：不重复写 L1
    await plugin.on_message_sent(ev)
    plugin.l1.add_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_t03_no_receipt_means_no_l1_and_failed_outbound(plugin):
    store = plugin._continuity.store
    ev = FakeEvent(text="在吗", msg_id=300)
    await plugin.on_llm_response(ev, _response("在的，回头告诉你。"))
    ev.result = FakeResult("在的，回头告诉你。")
    await plugin.on_message_sent(ev)  # 无 telethon_sent_message_ids

    plugin.l1.add_message.assert_not_awaited()
    assert ev.extras.get("iris_pending_assistant") is None
    rows = store.outbound_in_state([m.OUT_FAILED], SCOPE)
    assert len(rows) == 1 and rows[0]["text"] == "在的，回头告诉你。"
    assert store.list_items(scope_key=SCOPE) == []  # 没发出去 → 不建承诺


@pytest.mark.asyncio
async def test_non_receipt_platform_keeps_immediate_l1_write(plugin):
    ev = FakeEvent(text="hi", msg_id=301, platform="aiocqhttp")
    await plugin.on_llm_response(ev, _response("hello"))
    plugin.l1.add_message.assert_awaited_once()
    assert "iris_pending_assistant" not in ev.extras
    ev.result = FakeResult("hello")
    await plugin.on_message_sent(ev)
    plugin.l1.add_message.assert_awaited_once()
    rows = plugin._continuity.store.outbound_in_state([m.OUT_UNCERTAIN], SCOPE)
    assert len(rows) == 1 and rows[0]["author"] == "unknown"


# ---------------------------------------------------------------- manual outgoing


@pytest.mark.asyncio
async def test_manual_outgoing_recorded_and_stopped(plugin):
    store = plugin._continuity.store
    ev = FakeEvent(text="我手动回一句", sender=SELF_ID, msg_id=777, out=True, session_id=OWNER)
    await plugin.on_self_outgoing(ev)
    assert ev.stopped
    row = store.outbound_by_tg_id(SCOPE, 777)
    assert row["author"] == "manual" and row["origin"] == "manual" and row["status"] == m.OUT_SENT
    # 重放幂等
    await plugin.on_self_outgoing(FakeEvent(text="我手动回一句", sender=SELF_ID, msg_id=777,
                                            out=True, session_id=OWNER))
    assert len(store.recent_outbound(SCOPE, authors=("manual",))) == 1


@pytest.mark.asyncio
async def test_incoming_owner_message_not_treated_as_manual(plugin):
    ev = FakeEvent(text="在吗", msg_id=778)
    await plugin.on_self_outgoing(ev)
    assert not ev.stopped
    assert plugin._continuity.store.outbound_by_tg_id(SCOPE, 778) is None


@pytest.mark.asyncio
async def test_manual_outgoing_stopped_even_without_service(plugin):
    plugin._continuity = None
    ev = FakeEvent(text="x", sender=SELF_ID, msg_id=779, out=True, session_id=OWNER)
    await plugin.on_self_outgoing(ev)
    assert ev.stopped


# ---------------------------------------------------------------- send guard


@pytest.mark.asyncio
@pytest.mark.parametrize("gid", [CHANNEL, "1000000001", "-100555", "555"])
async def test_reply_to_read_only_session_is_cleared(plugin, gid):
    ev = FakeEvent(text="hi", group_id=gid)
    ev.result = FakeResult("我来回复一下")
    await plugin.on_decorating_result(ev)
    assert ev.stopped and ev.result.chain == []


@pytest.mark.asyncio
async def test_reply_to_normal_group_not_blocked(plugin):
    plugin._error_processor = None
    plugin._markdown_stripper = None
    ev = FakeEvent(text="hi", group_id=GROUP)
    ev.result = FakeResult("正常回复")
    await plugin.on_decorating_result(ev)
    assert not ev.stopped and ev.result.chain


# ---------------------------------------------------------------- owner watermark / resume


@pytest.mark.asyncio
async def test_owner_replay_does_not_resume_streak(plugin):
    state = plugin._state
    await plugin.on_message(FakeEvent(text="我回来啦", msg_id=500))
    assert state.resume_calls == [SCOPE]
    await plugin.on_message(FakeEvent(text="我回来啦", msg_id=500))  # 重放
    await plugin.on_message(FakeEvent(text="旧消息", msg_id=499))  # 更旧
    assert state.resume_calls == [SCOPE]
    await plugin.on_message(FakeEvent(text="新消息", msg_id=501))
    assert state.resume_calls == [SCOPE, SCOPE]


@pytest.mark.asyncio
async def test_non_owner_and_group_do_not_resume(plugin):
    state = plugin._state
    await plugin.on_message(FakeEvent(text="hi", sender="111", msg_id=600))
    await plugin.on_message(FakeEvent(text="hi", group_id=GROUP, msg_id=601))
    assert state.resume_calls == []


# ---------------------------------------------------------------- batching / injection


@pytest.mark.asyncio
async def test_batching_handler_merges_owner_burst(plugin):
    e1 = FakeEvent(text="今天", msg_id=700)
    e2 = FakeEvent(text="面试过了", msg_id=701)
    await asyncio.gather(plugin.on_continuity_batch(e1), plugin.on_continuity_batch(e2))
    assert e1.stopped and not e2.stopped
    assert e2.message_str == "今天\n面试过了"
    # 非主人 / 命令不参与
    e3 = FakeEvent(text="今天", sender="111", msg_id=702)
    await plugin.on_continuity_batch(e3)
    assert not e3.stopped


@pytest.mark.asyncio
async def test_llm_request_injects_continuity_after_memory(plugin):
    store = plugin._continuity.store
    store.create_item(
        subject_id=OWNER, subject_type="user", scope_key=SCOPE, topic="周五交论文",
        summary="周五要交论文", kind=m.KIND_PLAN, source_kind=m.SOURCE_SHARED_DIALOGUE,
        source_refs=[], allowed_audiences=[SCOPE],
    )
    plugin.component_manager = None  # 记忆注入由既有测试覆盖；这里只看连续性追加
    ev = FakeEvent(text="晚安", msg_id=800)
    await plugin.on_message(ev)  # closure → ended
    req = SimpleNamespace(prompt="晚安", extra_user_content_parts=[])
    await plugin.on_llm_request(ev, req)
    texts = [p.text for p in req.extra_user_content_parts]
    assert any("周五交论文" in t and t.startswith("<iris:continuity>") for t in texts)
    assert any("收尾" in t for t in texts)

    group_req = SimpleNamespace(prompt="hi", extra_user_content_parts=[])
    await plugin.on_llm_request(FakeEvent(text="hi", group_id=GROUP, msg_id=801), group_req)
    assert not any("周五交论文" in p.text for p in group_req.extra_user_content_parts)


# ---------------------------------------------------------------- profile injection (llm_request_hook)


def _profiles():
    user = SimpleNamespace(
        user_id=OWNER, user_name="小雨", historical_names=["雨"], personality_tags=["安静"],
        interests=["摄影"], occupation="学生", language_style="短句", communication_style="直接",
        emotional_baseline="最近焦虑", favorability=80, bot_relationship="主人",
        important_dates=["11-06 交论文", {"date": "12-01", "description": "生日"}, {"date": "12-24"}],
        important_events=["搬家"], taboo_topics=["前任"], custom_fields={"宠物": "猫"},
    )
    group = SimpleNamespace(interests=[], atmosphere_tags=["活跃"], long_term_tags=[],
                            blacklist_topics=[], custom_fields={})
    return group, user


def test_profile_injection_str_dates_no_typeerror(monkeypatch):
    import astrbot_plugin_iris_memory.iris_memory.config as config_mod
    from astrbot_plugin_iris_memory.iris_memory.core.llm_request_hook import (
        _format_profiles_for_injection,
    )

    monkeypatch.setattr(config_mod, "get_config", lambda: FakeConfig())
    group, user = _profiles()
    text = _format_profiles_for_injection(group, user)
    assert "重要日期: 11-06 交论文, 12-01(生日), 12-24" in text
    assert "重要事件: 搬家" in text and "禁忌: 前任" in text and "职业: 学生" in text


def test_profile_injection_group_only_name_level(monkeypatch):
    import astrbot_plugin_iris_memory.iris_memory.config as config_mod
    from astrbot_plugin_iris_memory.iris_memory.core.llm_request_hook import (
        _format_profiles_for_injection,
    )

    monkeypatch.setattr(config_mod, "get_config", lambda: FakeConfig())
    group, user = _profiles()
    text = _format_profiles_for_injection(group, user, is_group=True)
    assert "昵称: 小雨" in text and "曾用昵称: 雨" in text and "称呼: 主人" in text
    for leaked in ("交论文", "生日", "搬家", "前任", "学生", "焦虑", "好感度", "猫", "摄影", "安静"):
        assert leaked not in text, leaked
    assert "氛围: 活跃" in text  # 群自身画像照常


# ---------------------------------------------------------------- review fixes


def _with_learning(plugin):
    learning = SimpleNamespace(on_response=AsyncMock())
    comps = {"l1_buffer": plugin.l1, "learning": learning}
    plugin.component_manager = SimpleNamespace(get_available_component=lambda name: comps.get(name))
    return learning


@pytest.mark.asyncio
async def test_t03_failed_send_stops_event_and_skips_learning(plugin):
    """F3：回执平台发送失败 → stop_event（AstrBot 跳过 _save_to_history），学习不采集。"""
    learning = _with_learning(plugin)
    ev = FakeEvent(text="周五考试", msg_id=900)
    await plugin.on_llm_response(ev, _response("好，到时候我问你。"))
    learning.on_response.assert_not_awaited()  # 回执前不学
    ev.result = FakeResult("好，到时候我问你。")
    await plugin.on_message_sent(ev)  # 无回执
    assert ev.stopped
    learning.on_response.assert_not_awaited()
    assert ev.extras.get("iris_pending_learning") is None


@pytest.mark.asyncio
async def test_t03_sent_reply_not_stopped_and_learned_after_receipt(plugin):
    learning = _with_learning(plugin)
    ev = FakeEvent(text="周五考试", msg_id=901)
    await plugin.on_llm_response(ev, _response("加油"))
    learning.on_response.assert_not_awaited()
    ev.result = FakeResult("加油")
    ev.extras["telethon_sent_message_ids"] = [9200]
    await plugin.on_message_sent(ev)
    assert not ev.stopped
    learning.on_response.assert_awaited_once()
    await plugin.on_message_sent(ev)  # 重复回调不重复学
    learning.on_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_receipt_platform_learns_immediately_and_never_stopped(plugin):
    learning = _with_learning(plugin)
    ev = FakeEvent(text="hi", msg_id=902, platform="aiocqhttp")
    await plugin.on_llm_response(ev, _response("hello"))
    learning.on_response.assert_awaited_once()
    ev.result = FakeResult("hello")
    await plugin.on_message_sent(ev)
    assert not ev.stopped
    learning.on_response.assert_awaited_once()


def test_profile_injection_never_contains_numeric_favorability(monkeypatch):
    """P4：好感度数值永不注入，与 profile.favorability_enable 无关。"""
    import astrbot_plugin_iris_memory.iris_memory.config as config_mod
    from astrbot_plugin_iris_memory.iris_memory.core.llm_request_hook import (
        _format_profiles_for_injection,
    )

    for enabled in (True, False):
        cfg = FakeConfig()
        cfg.values["profile.favorability_enable"] = enabled
        monkeypatch.setattr(config_mod, "get_config", lambda c=cfg: c)
        group, user = _profiles()
        text = _format_profiles_for_injection(group, user)
        assert "好感" not in text and "80" not in text
        assert "称呼: 主人" in text


@pytest.mark.asyncio
async def test_on_all_message_silent_check_ignores_thread_suffix(plugin):
    """P5：话题会话 id 带 #thread 后缀，仍按 silent 群处理。"""
    plugin.component_manager = None
    ev = FakeEvent(text="hi", sender="123", group_id="-100555#7", msg_id=903)
    await plugin.on_all_message(ev)
    assert ev.stopped
    assert plugin._blocked_send_target(FakeEvent(text="hi", group_id=f"{CHANNEL}#2"))


@pytest.mark.asyncio
async def test_offline_catch_up_resume_cb_uses_state_rule(plugin):
    """F8 接线：main 提供的 resume_cb 走 state.resume_after_owner_message。"""
    assert plugin._continuity_resume(SCOPE) is True
    assert plugin._state.resume_calls == [SCOPE]


# ---------------------------------------------------------------- T16 L2 probe


class _FakeL2:
    def __init__(self):
        self.rows = {}

    async def add_memory_with_result(self, content, metadata=None, skip_dedup=False, persona_id="default"):
        mid = f"m{len(self.rows) + 1}"
        self.rows[mid] = (content, dict(metadata or {}))
        return mid, True

    async def retrieve(self, query, group_id=None, top_k=10, persona_id="default"):
        out = []
        for mid, (content, meta) in self.rows.items():
            if group_id and meta.get("group_id") != group_id:
                continue
            out.append(SimpleNamespace(entry=SimpleNamespace(id=mid, content=content), score=0.9))
        return out

    async def delete_entries(self, ids):
        for i in ids:
            self.rows.pop(i, None)
        return True


@pytest.mark.asyncio
async def test_l2_probe_write_search_delete(plugin):
    import json

    l2 = _FakeL2()
    l2.rows["real"] = ("她喜欢猫", {"group_id": ""})
    plugin.component_manager = SimpleNamespace(get_available_component=lambda n: l2 if n == "l2_memory" else None)
    base = plugin._continuity.store.db_path.parent

    async def run(req):
        (base / "l2_probe.json").write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")
        await plugin._run_l2_probe()
        assert not (base / "l2_probe.json").exists()
        return json.loads((base / "l2_probe.result.json").read_text(encoding="utf-8"))

    assert (await run({"op": "write", "text": "没有标记"}))["error"] == "bad_text"
    w = await run({"op": "write", "text": "【验收测试记忆】她说部署验收用的暗号是蓝色风铃"})
    assert w["ok"] and l2.rows[w["memory_id"]][1]["group_id"] == ""
    sr = await run({"op": "search", "query": "暗号"})
    assert any(h["is_probe"] for h in sr["private"]) and not sr["group"]
    assert "她喜欢猫" not in json.dumps(sr, ensure_ascii=False)  # 结果不含其它记忆正文
    assert (await run({"op": "delete", "memory_id": "real"}))["error"] == "not_a_probe_memory"
    d = await run({"op": "delete", "memory_id": w["memory_id"]})
    assert d["ok"] and w["memory_id"] not in l2.rows and "real" in l2.rows

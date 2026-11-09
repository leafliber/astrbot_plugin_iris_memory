"""ContinuityService：配置收敛、软回滚保留发送边界与记账、故障隔离。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from astrbot_plugin_iris_memory.iris_memory.continuity import models as m
from astrbot_plugin_iris_memory.iris_memory.continuity.batching import PASS, STOP
from astrbot_plugin_iris_memory.iris_memory.continuity.closure import ENDING_HINT
from astrbot_plugin_iris_memory.iris_memory.continuity.proactive_bridge import (
    REASON_FORBIDDEN,
    REASON_NOT_OWNER,
    REASON_SILENT,
)
from astrbot_plugin_iris_memory.iris_memory.continuity.service import (
    ContinuityConfig,
    ContinuityService,
    is_silent_or_forbidden,
)

OWNER = "100000001"
SCOPE = f"private:{OWNER}"
CHANNEL = "-1001000000001"
GROUP = "-1001000000002"
ROOT = Path(__file__).resolve().parents[2]
T0 = 1_793_000_000.0  # 2026-10 附近


class Clock:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeConfig:
    """Iris Config 风格：get("continuity.<key>", default)"""

    def __init__(self, **continuity) -> None:
        self.values = {"proactive.silent_sessions": [CHANNEL, "-100777"]}
        for k, v in continuity.items():
            self.values[f"continuity.{k}"] = v

    def set(self, key: str, value) -> None:
        self.values[f"continuity.{key}"] = value

    def get(self, key, default=None):
        return self.values.get(key, default)


class FakeState:
    def __init__(self) -> None:
        self.resumed: list[str] = []

    def get_whitelist(self):
        return [SCOPE, GROUP]

    def resume_after_owner_message(self, gid):
        self.resumed.append(gid)
        return True


def _service(tmp_path: Path, cfg: FakeConfig, clock: Clock | None = None, llm=None) -> ContinuityService:
    return ContinuityService(
        context=SimpleNamespace(get_platform_inst=lambda pid: None),
        config_source=cfg,
        state=FakeState(),
        owner_ids_get=lambda: [OWNER],
        silent_sessions_get=lambda: cfg.get("proactive.silent_sessions", []),
        data_dir=tmp_path / "data",
        llm_manager=llm or SimpleNamespace(generate=AsyncMock(return_value='{"ops":[]}')),
        clock=clock or Clock(),
    )


@pytest_asyncio.fixture
async def svc(tmp_path):
    cfg = FakeConfig(extract_debounce_s=0)
    s = _service(tmp_path, cfg)
    s.test_cfg = cfg
    await s.start()
    yield s
    await s.stop()


def _create_item(s: ContinuityService, topic: str = "周五交论文") -> dict:
    return s.store.create_item(
        subject_id=OWNER,
        subject_type="user",
        scope_key=SCOPE,
        topic=topic,
        summary=f"{topic}，说好到时候问问",
        kind=m.KIND_PLAN,
        source_kind=m.SOURCE_SHARED_DIALOGUE,
        source_refs=[],
        allowed_audiences=[SCOPE],
        not_before=T0 - 10,
        expires_at=T0 + 86400,
    )


# ---------------------------------------------------------------- config


def test_config_defaults_match_design():
    c = ContinuityConfig.from_config(None)
    assert c == ContinuityConfig()
    assert c.enable and c.items_enable and c.closure_enable and c.batching_enable
    assert not c.channel_ingest_enable and not c.digest_enable
    assert (c.merge_quiet_ms, c.merge_cap_ms, c.explicit_wait_timeout_s) == (1200, 2000, 180)
    assert (c.inject_max_items, c.inject_max_chars, c.closure_cooldown_minutes) == (3, 600, 120)
    assert (c.extract_debounce_s, c.extract_timeout_s, c.extract_queue_max) == (6, 30, 50)
    assert c.digest_interval_hours == 3


def test_config_coercion_and_clamps():
    c = ContinuityConfig.from_config(
        {
            "continuity": {
                "enable": "false",
                "items_enable": 0,
                "batching_enable": "yes",
                "closure_enable": "garbage",
                "merge_quiet_ms": "3000",
                "merge_cap_ms": 100,  # 小于 quiet → 抬到 quiet
                "inject_max_items": 99,
                "inject_max_chars": -5,
                "extract_timeout_s": True,  # bool 不当数字
                "digest_interval_hours": "abc",
                "observe_chat_ids": ["-1001", "x", -1001, "-1002", True],
                "owner_channel_ids": "-1003, -1004",
            }
        }
    )
    assert c.enable is False and c.items_enable is False
    assert c.batching_enable is True and c.closure_enable is True  # 非法值回退默认
    assert c.merge_quiet_ms == 3000 and c.merge_cap_ms == 3000
    assert c.inject_max_items == 10 and c.inject_max_chars == 100
    assert c.extract_timeout_s == 30 and c.digest_interval_hours == 3
    assert c.observe_chat_ids == (-1001, -1002)
    assert c.owner_channel_ids == (-1003, -1004)


def test_master_switch_gates_derived_flags():
    on = ContinuityConfig.from_config(FakeConfig(channel_ingest_enable=True, digest_enable=True,
                                                 observe_chat_ids=["-1001"]))
    assert on.items_active and on.topics_active and on.closure_active and on.batching_active
    assert on.channel_active and on.digest_active
    off = ContinuityConfig.from_config(FakeConfig(enable=False, channel_ingest_enable=True,
                                                  observe_chat_ids=["-1001"]))
    assert not any([off.items_active, off.topics_active, off.closure_active,
                    off.batching_active, off.channel_active, off.digest_active])


def test_config_reader_failure_falls_back_to_defaults():
    class Broken:
        def get(self, key, default=None):
            raise RuntimeError("boom")

    assert ContinuityConfig.from_config(Broken()) == ContinuityConfig()


def test_schema_and_defaults_dataclass_cover_all_keys():
    from astrbot_plugin_iris_memory.iris_memory.config.defaults import Defaults

    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    items = schema["continuity"]["items"]
    svc_keys = set(ContinuityConfig().as_dict())
    assert set(items) == svc_keys
    schema_defaults = {k: v["default"] for k, v in items.items()}
    # schema 默认值经 from_config 收敛后等于服务默认
    assert ContinuityConfig.from_config({"continuity": schema_defaults}) == ContinuityConfig()
    # Defaults dataclass 与 schema 一致（Config.get 缺键回退）
    d = Defaults().get_section_defaults("continuity")
    assert d == schema_defaults
    assert Defaults().get_by_flat_key("continuity.merge_quiet_ms") == 1200


def test_silent_or_forbidden_variants():
    assert is_silent_or_forbidden(CHANNEL)  # 内置禁发频道
    assert is_silent_or_forbidden("1000000001")
    assert is_silent_or_forbidden("-1000000001")
    assert is_silent_or_forbidden("-100777", ["-100777"])
    assert is_silent_or_forbidden("777", ["-100777"])
    assert not is_silent_or_forbidden(GROUP, ["-100777"])
    assert not is_silent_or_forbidden("", [CHANNEL])


# ---------------------------------------------------------------- lifecycle


@pytest.mark.asyncio
async def test_store_opened_at_data_dir_path(svc, tmp_path):
    assert svc.is_open
    assert (tmp_path / "data" / "continuity" / "continuity.db").exists()
    assert svc.worker.running


@pytest.mark.asyncio
async def test_store_open_failure_degrades(tmp_path):
    s = _service(tmp_path, FakeConfig())
    s.store.open = Mock(side_effect=OSError("disk"))
    await s.start()
    assert not s.is_open
    assert await s.note_owner_message(SCOPE, 1, T0, text="明天考试") is False
    assert await s.assemble(SCOPE, sender_id=OWNER) == ""
    assert await s.record_reply_sent(SCOPE, "好", [1]) is None
    assert await s.record_manual(SCOPE, 1, "x") is None
    assert await s.closure_hint(SCOPE) is None
    await s.stop()


# ---------------------------------------------------------------- enabled path


@pytest.mark.asyncio
async def test_owner_message_records_inbound_and_watermark(svc):
    svc.worker.notify = Mock()
    assert await svc.note_owner_message(SCOPE, 10, T0, text="周五要交论文", sender_id=OWNER) is True
    row = svc.store.get_inbound(SCOPE, 10)
    assert row is not None and row["text"] == "周五要交论文"
    svc.worker.notify.assert_called_once_with(SCOPE)
    # 重放：不推进、不重复入队
    assert await svc.note_owner_message(SCOPE, 10, T0, text="周五要交论文", sender_id=OWNER) is False
    assert svc.worker.notify.call_count == 1
    # 命令 / 纯媒体不入收件箱，但仍推进水位
    assert await svc.note_owner_message(SCOPE, 11, T0, text="/iris_mem l1") is True
    assert await svc.note_owner_message(SCOPE, 12, T0, text="[贴纸]", media_only=True) is True
    assert svc.store.get_inbound(SCOPE, 11) is None and svc.store.get_inbound(SCOPE, 12) is None


@pytest.mark.asyncio
async def test_non_owner_scope_ignored(svc):
    assert await svc.note_owner_message("private:111", 5, T0, text="周五考试") is False
    assert svc.store.get_watermark("private:111") is None
    assert svc.owner_scope("111") is None and svc.owner_scope(OWNER) == SCOPE


@pytest.mark.asyncio
async def test_reply_sent_records_promise_once(svc):
    row = await svc.record_reply_sent(SCOPE, "好呀，到时候我提醒你。", [501], reply_to_msg_id=10)
    assert row["status"] == m.OUT_SENT and row["turn_id"] == "reply:10"
    promises = [i for i in svc.store.list_items(scope_key=SCOPE) if i["kind"] == m.KIND_BOT_PROMISE]
    assert len(promises) == 1
    # 回执重放：不新建出站、不重复承诺
    again = await svc.record_reply_sent(SCOPE, "好呀，到时候我提醒你。", [501], reply_to_msg_id=10)
    assert again["id"] == row["id"]
    promises = [i for i in svc.store.list_items(scope_key=SCOPE) if i["kind"] == m.KIND_BOT_PROMISE]
    assert len(promises) == 1


@pytest.mark.asyncio
async def test_reply_without_receipt_is_failed_and_no_promise(svc):
    row = await svc.record_reply_sent(SCOPE, "到时候我提醒你", [], receipt_supported=True)
    assert row["status"] == m.OUT_FAILED
    assert svc.store.list_items(scope_key=SCOPE) == []
    row = await svc.record_reply_sent(SCOPE, "到时候我提醒你", [], receipt_supported=False)
    assert row["status"] == m.OUT_UNCERTAIN and row["author"] == "unknown"


@pytest.mark.asyncio
async def test_closure_hint_after_owner_goodnight(svc):
    await svc.note_owner_message(SCOPE, 20, T0, text="晚安")
    assert await svc.closure_hint(SCOPE) == ENDING_HINT
    await svc.note_owner_message(SCOPE, 21, T0, text="对了你明天干嘛？")
    assert await svc.closure_hint(SCOPE) is None


@pytest.mark.asyncio
async def test_assemble_includes_items_when_enabled(svc):
    _create_item(svc)
    svc.store.put_private_context("identity", "你是 Frisk。", "persona_split", SCOPE)
    text = await svc.assemble(SCOPE, sender_id=OWNER)
    assert "周五交论文" in text and "你是 Frisk" in text
    # 群聊：不注入私聊专属层与主人事项
    assert "周五交论文" not in await svc.assemble(GROUP, sender_id=OWNER)
    assert "Frisk" not in await svc.assemble(GROUP, sender_id=OWNER)


# ---------------------------------------------------------------- soft rollback


@pytest.mark.asyncio
async def test_soft_rollback_keeps_guards_and_bookkeeping(svc):
    _create_item(svc)
    svc.store.put_private_context("identity", "你是 Frisk。", "persona_split", SCOPE)
    svc.test_cfg.set("enable", False)
    cfg = svc.cfg
    assert not cfg.items_active and not cfg.closure_active and not cfg.batching_active

    # 发送目标硬拦截仍生效
    b = svc.bridge
    assert b.check_target(CHANNEL, f"userbot:GroupMessage:{CHANNEL}") in (REASON_SILENT, REASON_FORBIDDEN)
    assert b.check_target("-100777", "userbot:GroupMessage:-100777") == REASON_SILENT
    assert b.check_target("private:111", "userbot:FriendMessage:111") == REASON_NOT_OWNER
    assert b.check_target(SCOPE, f"userbot:FriendMessage:{OWNER}") == ""

    # 水位（暂停恢复依据）照常推进，但不入收件箱、不改收尾
    svc.worker.notify = Mock()
    assert await svc.note_owner_message(SCOPE, 30, T0, text="晚安，周五考试") is True
    assert svc.store.get_watermark(SCOPE)["last_msg_id"] == 30
    assert svc.store.get_inbound(SCOPE, 30) is None
    svc.worker.notify.assert_not_called()
    assert svc.store.get_closure(SCOPE) is None
    assert await svc.closure_hint(SCOPE) is None

    # 出站记账保留；承诺事项不再新建
    row = await svc.record_reply_sent(SCOPE, "到时候我提醒你", [601])
    assert row["status"] == m.OUT_SENT
    assert await svc.record_manual(SCOPE, 602, "手动")
    assert svc.store.outbound_by_tg_id(SCOPE, 602)["author"] == "manual"
    assert not [i for i in svc.store.list_items(scope_key=SCOPE) if i["kind"] == m.KIND_BOT_PROMISE]

    # 不选题、不注入事项；私聊专属人格层保留
    assert await b.candidates(SCOPE, T0) == []
    text = await svc.assemble(SCOPE, sender_id=OWNER)
    assert "Frisk" in text and "周五交论文" not in text
    assert await svc.assemble(GROUP, sender_id=OWNER) == ""

    # 不合批：事件直接放行、不 stop
    ev = SimpleNamespace(message_str="嗯", stop_event=Mock(), get_messages=lambda: [])
    assert await svc.gate_batch(SCOPE, ev) == PASS
    ev.stop_event.assert_not_called()


@pytest.mark.asyncio
async def test_batching_gate_merges_when_enabled(tmp_path):
    cfg = FakeConfig(merge_quiet_ms=200, merge_cap_ms=1000)
    s = _service(tmp_path, cfg)
    await s.start()
    try:
        def ev(text):
            e = SimpleNamespace(message_str=text, get_messages=lambda: [], stopped=False)
            e.message_obj = SimpleNamespace(message_str=text)
            e.stop_event = lambda: setattr(e, "stopped", True)
            return e

        e1, e2 = ev("我今天"), ev("去面试了")
        r = await asyncio.gather(s.gate_batch(SCOPE, e1), s.gate_batch(SCOPE, e2))
        assert r == [STOP, PASS]
        assert e1.stopped and not e2.stopped
        assert e2.message_str == "我今天\n去面试了" == e2.message_obj.message_str
        # 非主人会话不合批
        assert await s.gate_batch("private:111", ev("x")) == PASS
    finally:
        await s.stop()


# ---------------------------------------------------------------- failure isolation


@pytest.mark.asyncio
async def test_store_failures_degrade_without_raising(svc):
    boom = Mock(side_effect=RuntimeError("db locked"))
    for name in (
        "advance_watermark", "record_inbound", "get_private_context", "list_items",
        "recent_outbound", "get_closure", "put_closure", "outbound_by_tg_id",
        "add_outbound", "save_window", "load_windows", "expire_items",
    ):
        setattr(svc.store, name, boom)
    assert await svc.note_owner_message(SCOPE, 40, T0, text="周五考试") is False
    assert await svc.assemble(SCOPE, sender_id=OWNER) == ""
    assert await svc.closure_hint(SCOPE) is None
    assert await svc.record_reply_sent(SCOPE, "好", [700]) is None
    assert await svc.record_manual(SCOPE, 701, "x") is None
    window = SimpleNamespace(get_messages=lambda k: [SimpleNamespace(
        sender_id="1", sender_name="a", content="hi", timestamp=T0)])
    assert await svc.save_windows(window, [SCOPE]) == 0
    assert await svc.restore_windows(window) == 0


@pytest.mark.asyncio
async def test_on_llm_request_survives_continuity_store_failure(tmp_path):
    from astrbot_plugin_iris_memory.main import IrisMemoryPlugin

    s = _service(tmp_path, FakeConfig())
    await s.start()
    try:
        s.store.get_private_context = Mock(side_effect=RuntimeError("x"))
        s.store.list_items = Mock(side_effect=RuntimeError("x"))
        s.store.recent_outbound = Mock(side_effect=RuntimeError("x"))
        s.store.get_closure = Mock(side_effect=RuntimeError("x"))

        plugin = object.__new__(IrisMemoryPlugin)
        plugin.component_manager = None
        plugin._llm_manager = None
        plugin._triggering = {}
        plugin._continuity = s
        extras = {"iris_speak_hint": "SPEAK"}
        event = SimpleNamespace(
            get_group_id=lambda: "",
            is_private_chat=lambda: True,
            get_sender_id=lambda: OWNER,
            get_extra=lambda key, default=None: extras.get(key, default),
            set_extra=lambda key, value: extras.__setitem__(key, value),
        )
        req = SimpleNamespace(prompt="hi", extra_user_content_parts=[])
        await plugin.on_llm_request(event, req)
        assert [p.text for p in req.extra_user_content_parts] == ["SPEAK"]
    finally:
        await s.stop()


# ---------------------------------------------------------------- acceptance probe

import json as _json  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402
from unittest.mock import AsyncMock as _AM  # noqa: E402

from astrbot_plugin_iris_memory.iris_memory.continuity.service import (  # noqa: E402
    ContinuityService as _Svc,
    PROBE_FILENAME as _PF,
    PROBE_RESULT_FILENAME as _PRF,
)


async def _probe_service(tmp_path, platform):
    svc = _Svc(
        context=_NS(get_platform_inst=lambda pid: platform),
        config_source={"continuity": {}, "proactive.silent_sessions": ["-1001000000001"]},
        state=_NS(get_whitelist=lambda: []),
        owner_ids_get=lambda: ["100000001"],
        silent_sessions_get=lambda: ["-1001000000001"],
        data_dir=tmp_path,
        llm_manager=_NS(generate=_AM(return_value='{"ops":[]}')),
    )
    await svc.start()
    return svc


@pytest.mark.asyncio
async def test_acceptance_probe_sends_once_to_owner_with_receipt(tmp_path):
    platform = _NS(send_with_receipt=_AM(return_value=[4242]), meta=lambda: _NS(id="userbot"))
    svc = await _probe_service(tmp_path, platform)
    try:
        base = svc.store.db_path.parent
        assert await svc.run_acceptance_probe() is None  # 无文件不做任何事
        (base / _PF).write_text(_json.dumps({"text": "【验收】链路测试"}), encoding="utf-8")
        res = await svc.run_acceptance_probe()
        assert res["status"] == "sent" and res["tg_msg_ids"] == [4242]
        assert not (base / _PF).exists()  # 一次性
        assert _json.loads((base / _PRF).read_text(encoding="utf-8"))["status"] == "sent"
        session = platform.send_with_receipt.await_args.args[0]
        assert "100000001" in str(getattr(session, "session_id", session))
        row = svc.store.outbound_by_tg_id("private:100000001", 4242)
        assert row["origin"] == "probe" and row["author"] == "bot"
        # 第二次调用：文件已删，不会重发
        assert await svc.run_acceptance_probe() is None
        assert platform.send_with_receipt.await_count == 1
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_acceptance_probe_refuses_bad_text_and_daily_limit(tmp_path):
    platform = _NS(send_with_receipt=_AM(side_effect=[[11], [12], [13], [14]]), meta=lambda: _NS(id="userbot"))
    svc = await _probe_service(tmp_path, platform)
    try:
        base = svc.store.db_path.parent
        (base / _PF).write_text(_json.dumps({"text": "没有前缀"}), encoding="utf-8")
        assert (await svc.run_acceptance_probe())["reason"] == "bad_text"
        for i in range(3):
            (base / _PF).write_text(_json.dumps({"text": f"【验收】第{i}次"}), encoding="utf-8")
            assert (await svc.run_acceptance_probe())["status"] == "sent"
        (base / _PF).write_text(_json.dumps({"text": "【验收】超额"}), encoding="utf-8")
        assert (await svc.run_acceptance_probe())["reason"] == "daily_limit"
        assert platform.send_with_receipt.await_count == 3
    finally:
        await svc.stop()


# ---------------------------------------------------------------- review fixes


def test_silent_or_forbidden_normalizes_thread_suffix():
    # 话题群会话 id 带 #thread 后缀：按所属群判定（P5）
    assert is_silent_or_forbidden(f"{CHANNEL}#42")
    assert is_silent_or_forbidden("-100777#5", ["-100777"])
    assert is_silent_or_forbidden("-100777", ["-100777#5"])
    assert is_silent_or_forbidden("777#9", ["-100777"])
    assert not is_silent_or_forbidden(f"{GROUP}#3", ["-100777"])


class _OwnerMsg:
    def __init__(self, i, text="", sender=int(OWNER), out=False, action=None, media=None):
        from datetime import datetime, timezone

        self.id = i
        self.date = datetime.fromtimestamp(T0 - 1000 + i, tz=timezone.utc)
        self.sender_id = sender
        self.raw_text = text
        self.out = out
        self.action = action
        self.media = media
        self.reply_to = None


class _OwnerClient:
    def __init__(self, history):
        self.history = history
        self.iter_calls: list[dict] = []
        self.sent: list = []

    def is_connected(self):
        return True

    async def send_message(self, *a, **kw):  # 补采绝不发送
        self.sent.append((a, kw))

    async def iter_messages(self, peer, limit=None, min_id=0, reverse=False):
        self.iter_calls.append({"peer": peer, "limit": limit, "min_id": min_id, "reverse": reverse})
        for x in sorted((x for x in self.history if x.id > (min_id or 0)), key=lambda x: x.id):
            yield x


@pytest.mark.asyncio
async def test_owner_offline_catch_up_advances_resumes_once_and_records(tmp_path):
    """F8：绑定 client 后补采离线期间主人私聊：水位推进、只恢复一次、记入收件箱、不回复。
    channel_ingest 关闭（默认）也运行。"""
    cfg = FakeConfig(extract_debounce_s=0)
    client = _OwnerClient(
        [
            _OwnerMsg(10, "旧消息"),
            _OwnerMsg(11, "周五交报告，不用提醒"),
            _OwnerMsg(12, "我手机发的", sender=None, out=True),
            _OwnerMsg(13, "别人", sender=12345),
            _OwnerMsg(14, "我回来啦"),
        ]
    )
    inst = SimpleNamespace(get_client=lambda: client)
    resumed: list[str] = []
    s = ContinuityService(
        context=SimpleNamespace(get_platform_inst=lambda pid: inst),
        config_source=cfg,
        state=FakeState(),
        owner_ids_get=lambda: [OWNER],
        silent_sessions_get=lambda: cfg.get("proactive.silent_sessions", []),
        data_dir=tmp_path / "data",
        llm_manager=SimpleNamespace(generate=AsyncMock(return_value='{"ops":[]}')),
        clock=Clock(),
        resume_cb=resumed.append,
    )
    # 首次运行（无水位）前先建立水位 = 10（模拟上次在线时看到的最后一条）
    await asyncio.to_thread(s.store.open)
    s.store.advance_watermark(SCOPE, 10, T0 - 990)
    s.store.close()
    await s.start()
    try:
        assert not s.cfg.channel_active
        assert s._owner_catchup is not None
        await s._owner_catchup.check_binding()
        for _ in range(50):
            if resumed:
                break
            await asyncio.sleep(0.01)
        assert resumed == [SCOPE]
        assert s.store.get_watermark(SCOPE)["last_msg_id"] == 14
        assert client.iter_calls[0] == {"peer": int(OWNER), "limit": 50, "min_id": 10, "reverse": True}
        assert len(client.iter_calls) == 1  # 同一 client 不重复补采
        assert client.sent == []
        rows = s.store._conn().execute(
            "SELECT msg_id FROM inbound WHERE scope_key=? ORDER BY msg_id", (SCOPE,)
        ).fetchall()
        assert [r[0] for r in rows] == [11, 14]
        # 再补采一次（重连）：没有新消息 → 不再恢复
        assert await s.catch_up_owner_messages(SCOPE, [{"msg_id": 14, "msg_ts": T0, "text": "我回来啦"}]) is False
        assert resumed == [SCOPE]
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_owner_catch_up_skips_without_watermark(tmp_path):
    from astrbot_plugin_iris_memory.iris_memory.continuity.channel_ingest import OwnerCatchUp

    client = _OwnerClient([_OwnerMsg(1, "很久以前")])
    fed: list = []

    async def feed(scope, msgs):
        fed.append((scope, msgs))

    cu = OwnerCatchUp(
        lambda: SimpleNamespace(get_client=lambda: client),
        lambda: [OWNER],
        lambda scope: None,
        feed,
        Clock(),
    )
    assert await cu.check_binding() is True
    assert client.iter_calls == [] and fed == []


@pytest.mark.asyncio
async def test_explicit_wait_survives_restart_via_store(tmp_path):
    """F10：显式等待写在 closures 表里，重启后新的 batcher 仍保持等待"""
    import time as _t

    platform = _NS(send_with_receipt=_AM(return_value=[1]), meta=lambda: _NS(id="userbot"))
    svc = await _probe_service(tmp_path, platform)
    try:
        scope = "private:100000001"
        svc.store.put_closure(scope, state="waiting", explicit_wait=1, wait_since=_t.time() - 5)
        assert svc.batcher._closure_getter is not None
        ev = _NS(message_str="然后还有一件事", has_media=False, stopped=False)
        ev.stop = lambda: setattr(ev, "stopped", True)
        verdict = await svc.batcher.gate(scope, ev)
        assert verdict == "stop" and ev.stopped
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_background_llm_calls_use_summary_provider(tmp_path):
    """后台抽取/摘要没有事件：显式传入 l1_buffer.summary_provider（现网无 AstrBot 默认 provider）"""
    calls = []

    class LM:
        async def generate(self, prompt, **kw):
            calls.append(kw)
            return '{"ops":[]}'

    class Cfg:
        def get(self, key, default=None):
            return {"l1_buffer.summary_provider": "deepseek"}.get(key, default)

    svc = _Svc(
        context=_NS(get_platform_inst=lambda pid: None),
        config_source=Cfg(),
        state=_NS(get_whitelist=lambda: []),
        owner_ids_get=lambda: ["100000001"],
        silent_sessions_get=lambda: [],
        data_dir=tmp_path,
        llm_manager=LM(),
    )
    await svc._llm_extract("p", 5)
    await svc._llm_digest("p", 5)
    assert [c["provider_id"] for c in calls] == ["deepseek", "deepseek"]
    assert [c["module"] for c in calls] == ["continuity_extract", "continuity_digest"]

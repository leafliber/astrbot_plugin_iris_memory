"""私聊主动消息门控与会话键的单元测试。"""

from types import SimpleNamespace

from astrbot_plugin_iris_memory.iris_memory.proactive.perception import Gatekeeper
from astrbot_plugin_iris_memory.main import IrisMemoryPlugin

PRIVATE_UID = "100000001"


def _event(*, private: bool, sender_id: str = PRIVATE_UID, text: str = "在吗"):
    return SimpleNamespace(
        message_str=text,
        is_private_chat=lambda: private,
        get_group_id=lambda: None if private else "123456",
        get_sender_id=lambda: sender_id,
        set_result=lambda msg: None,
    )


def _gate(config) -> Gatekeeper:
    state = SimpleNamespace(is_muted=lambda: False, is_whitelisted=lambda key: False)
    return Gatekeeper(config, state)


def test_private_chat_rejected_when_switch_off():
    cfg = SimpleNamespace(private_enabled=False, private_whitelist=[PRIVATE_UID])
    assert _gate(cfg).should_process(_event(private=True)) is False


def test_private_chat_rejected_when_not_whitelisted():
    cfg = SimpleNamespace(private_enabled=True, private_whitelist=["111"])
    assert _gate(cfg).should_process(_event(private=True)) is False


def test_private_chat_rejected_when_whitelist_empty():
    cfg = SimpleNamespace(private_enabled=True, private_whitelist=[])
    assert _gate(cfg).should_process(_event(private=True)) is False


def test_private_chat_allowed_when_enabled_and_listed():
    cfg = SimpleNamespace(private_enabled=True, private_whitelist=[PRIVATE_UID])
    assert _gate(cfg).should_process(_event(private=True)) is True


def test_group_path_still_requires_whitelist():
    cfg = SimpleNamespace(private_enabled=True, private_whitelist=[PRIVATE_UID])
    assert _gate(cfg).should_process(_event(private=False)) is False


def test_session_key_for_private_chat():
    plugin = object.__new__(IrisMemoryPlugin)
    assert plugin._get_session_key(_event(private=True)) == f"private:{PRIVATE_UID}"


def test_session_key_for_group_chat():
    plugin = object.__new__(IrisMemoryPlugin)
    assert plugin._get_session_key(_event(private=False)) == "123456"

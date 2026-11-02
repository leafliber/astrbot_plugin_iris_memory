"""Pytest 配置文件"""

import sys
import types
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent

# 以固定包名注册插件根目录，模拟 AstrBot 运行时的
# data.plugins.<插件目录名> 包前缀：main.py / iris_memory 的相对导入
# 在测试中可用，且不依赖仓库目录名、不污染 sys.path。
if "astrbot_plugin_iris_memory" not in sys.modules:
    _pkg = types.ModuleType("astrbot_plugin_iris_memory")
    _pkg.__path__ = [str(project_root)]
    _pkg.__package__ = "astrbot_plugin_iris_memory"
    sys.modules["astrbot_plugin_iris_memory"] = _pkg


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


# 代码里的默认禁发频道为空（由部署方填写）；测试中注入一个示例频道，
# 覆盖“频道永不主动发送 / 贴纸永不进入频道”等守卫。
TEST_FORBIDDEN_CHAT = "-1001000000001"


@pytest.fixture(autouse=True)
def _default_forbidden_chat(monkeypatch):
    chats = frozenset({TEST_FORBIDDEN_CHAT})
    from astrbot_plugin_iris_memory.iris_memory.continuity import proactive_bridge, service
    from astrbot_plugin_iris_memory.iris_memory.proactive import proactive
    from astrbot_plugin_iris_memory.iris_memory.stickers import library

    monkeypatch.setattr(proactive_bridge, "DEFAULT_FORBIDDEN_CHATS", chats)
    monkeypatch.setattr(service, "DEFAULT_FORBIDDEN_CHATS", chats)
    monkeypatch.setattr(proactive, "_LEGACY_FORBIDDEN_CHATS", chats)
    monkeypatch.setattr(library, "DEFAULT_BLOCKED_SCOPES", chats)
    yield

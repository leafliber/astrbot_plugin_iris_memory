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

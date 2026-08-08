"""学习模块测试共享 fixture"""

from pathlib import Path

import pytest

from iris_memory.config import init_config
from iris_memory.config.config import reset_config
from iris_memory.learning import LearningStorage


@pytest.fixture
def config(tmp_path: Path):
    """初始化全局配置单例（learning.enable=true），测试后重置

    get_config 是全局单例，learning 模块在模块顶层绑定了 get_config
    符号，因此这里用真实 init_config 而不是 patch。
    """
    cfg = init_config({"learning": {"enable": True}}, tmp_path)
    yield cfg
    reset_config()


@pytest.fixture
def storage(tmp_path: Path):
    """tmp_path 隔离的 LearningStorage 实例"""
    s = LearningStorage(tmp_path / "learning.db")
    s.init_schema()
    yield s
    s.close()

"""Iris Chat Memory - 原子持久化工具

提供原子文件写入：先写入同目录临时文件并 fsync，再 ``os.replace`` 覆盖目标。
这样进程在写入过程中崩溃时，目标文件要么是旧的完整内容、要么是新的完整内容，
不会被截断损坏（此前多处直接 ``open(w)`` 覆盖写，崩溃会留下截断文件，
而部分加载逻辑会静默丢弃全部数据回到默认）。
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """原子写入文本文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件必须与目标在同一目录（同一文件系统），os.replace 才原子
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: Path | str,
    data: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
) -> None:
    """原子写入 JSON 文件。"""
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=ensure_ascii, indent=indent),
    )

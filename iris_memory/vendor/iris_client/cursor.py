"""Explicit cursor checkpoint. Commit only after durably processing a page."""

import json
import os
import tempfile
from pathlib import Path

from .errors import ProtocolError
from .types import NotificationPage


class NotificationCursor:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.value = json.loads(self.path.read_text()) if self.path.exists() else 0
        if type(self.value) is not int or self.value < 0:
            raise ProtocolError()

    def commit(self, page: NotificationPage) -> None:
        value = page["next_cursor"]
        if type(value) is not int or value < self.value:
            raise ProtocolError()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=".iris-cursor-")
        try:
            with os.fdopen(fd, "w") as output:
                output.write(json.dumps(value))
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.value = value

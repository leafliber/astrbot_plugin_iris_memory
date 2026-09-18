"""Errors contain fixed public codes, never transport exception text or secrets."""


class ControlError(Exception):
    def __init__(self, code: str, status: int = 400, *, revision: int | None = None):
        super().__init__(code)
        self.code, self.status, self.revision = code, status, revision

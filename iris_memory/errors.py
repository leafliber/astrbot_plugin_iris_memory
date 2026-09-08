"""Stable plugin errors; transport details remain out of user responses."""


class IrisError(Exception):
    def __init__(self, code: str, message: str, *, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def as_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}


class Conflict(IrisError):
    def __init__(self, message="配置或数据已被修改，请刷新后重试"):
        super().__init__("revision_conflict", message)

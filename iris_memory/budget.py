"""Persistent conservative reservations for plugin-initiated model work."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from .errors import IrisError
from .storage import encode


def tokens(value):
    # UTF-8 bytes are a conservative upper bound for byte-level tokenizers.
    # Exact provider tokenizers may replace this without weakening the bound.
    return len((value if isinstance(value, str) else encode(value)).encode("utf-8"))


class Budget:
    def __init__(self, control):
        self.control = control

    async def reserve(self, amount):
        settings = self.control.settings
        day = datetime.now(ZoneInfo(settings["timezone"])).date().isoformat()

        def reserve(db):
            row = db.execute(
                "SELECT revision,value FROM documents WHERE collection='budget' AND key=?",
                (day,),
            ).fetchone()
            value = json.loads(row[1]) if row else {"calls": 0, "tokens": 0}
            if (
                value["calls"] + 1 > settings["daily_model_calls"]
                or value["tokens"] + amount > settings["daily_model_tokens"]
            ):
                raise IrisError("budget_exhausted", "插件今日模型预算已耗尽")
            value = {"calls": value["calls"] + 1, "tokens": value["tokens"] + amount}
            db.execute(
                "INSERT INTO documents VALUES('budget',?,?,?) ON CONFLICT(collection,key) DO UPDATE SET revision=excluded.revision,value=excluded.value",
                (day, (row[0] if row else 0) + 1, encode(value)),
            )
            db.execute(
                "DELETE FROM documents WHERE collection='budget' AND key < date(?, '-31 day')",
                (day,),
            )
            return value

        return await self.control.store.run(reserve)

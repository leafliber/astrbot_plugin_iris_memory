"""
PersonaChangeRecord - 画像变更审计记录

从 user_persona.py 提取，遵循 SRP 原则。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PersonaChangeRecord:
    """画像单次变更的审计记录"""
    timestamp: str = ""
    field_name: str = ""
    old_value: Any = None
    new_value: Any = None
    source_memory_id: Optional[str] = None
    memory_type: Optional[str] = None
    rule_id: str = ""          # 触发规则标识
    confidence: float = 0.0
    evidence_type: str = "inferred"  # confirmed / inferred / contested

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.timestamp,
            "field": self.field_name,
            "old": self.old_value,
            "new": self.new_value,
            "mem_id": self.source_memory_id,
            "mem_type": self.memory_type,
            "rule": self.rule_id,
            "conf": self.confidence,
            "ev": self.evidence_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PersonaChangeRecord:
        return cls(
            timestamp=d.get("ts", ""),
            field_name=d.get("field", ""),
            old_value=d.get("old"),
            new_value=d.get("new"),
            source_memory_id=d.get("mem_id"),
            memory_type=d.get("mem_type"),
            rule_id=d.get("rule", ""),
            confidence=d.get("conf", 0.0),
            evidence_type=d.get("ev", "inferred"),
        )

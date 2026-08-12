"""暗语学习内部模型。"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CandidateCluster:
    cluster_id: str
    group_id: str
    candidate_ids: List[int]
    terms: List[str]
    canonical_hint: str
    message_count: int
    user_count: int
    span_hours: float
    local_score: float
    contexts: List[Dict[str, str]] = field(default_factory=list)
    review_token: str = ""


@dataclass
class ReviewVerdict:
    cluster_id: str
    decision: str
    category: str
    canonical_term: str
    aliases: List[str]
    meaning: str
    confidence: float
    reason: str

"""暗语候选的本地统计评分、准入与子串聚类。"""

import math
from collections import Counter
from typing import Any, Dict, List

from iris_memory.config import get_config
from iris_memory.learning.jargon_clustering import (
    cluster_candidate_items,
    select_candidate_representative,
)
from .models import CandidateCluster


def _entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    result = 0.0
    for count in counts.values():
        probability = count / total
        result -= probability * math.log(probability + 1e-12)
    return min(1.0, result / math.log(max(2, len(counts))))


class CandidateScorer:
    def score_and_cluster(self, snapshots: List[Dict[str, Any]], now: float) -> List[CandidateCluster]:
        config = get_config()
        min_messages = config.get_int("learning_jargon_min_messages", 6) or 6
        small_users = config.get_int("learning_jargon_min_users_small", 2) or 2
        large_users = config.get_int("learning_jargon_min_users_large", 3) or 3
        large_group = config.get_int("learning_jargon_large_group_users", 10) or 10
        max_user_ratio = config.get_float("learning_jargon_max_single_user_ratio", 0.6) or 0.6
        min_span = config.get_float("learning_jargon_min_span_hours", 6.0) or 6.0
        fast_messages = config.get_int("learning_jargon_fast_track_messages", 8) or 8
        fast_users = config.get_int("learning_jargon_fast_track_users", 5) or 5
        retry_days = config.get_int("learning_jargon_retry_days", 7) or 7
        retry_multiplier = (
            config.get_float("learning_jargon_retry_evidence_multiplier", 2.0) or 2.0
        )
        max_attempts = config.get_int("learning_jargon_max_llm_attempts", 2) or 2
        support_ratio = config.get_float("learning_jargon_substring_support_ratio", 0.85) or 0.85
        count_ratio = config.get_float("learning_jargon_substring_count_ratio", 0.8) or 0.8

        counts = {(s["group_id"], s["term"]): int(s["message_count"]) for s in snapshots}
        length_totals: Counter = Counter()
        for snapshot in snapshots:
            length_totals[(snapshot["group_id"], len(snapshot["term"]))] += int(
                snapshot["message_count"]
            )
        eligible: List[Dict[str, Any]] = []
        scores: Dict[int, float] = {}
        for item in snapshots:
            messages = int(item["message_count"])
            user_counts = item.get("user_counts") or {}
            users = len(user_counts)
            required_users = large_users if int(item.get("active_group_users", 0)) >= large_group else small_users
            user_ratio = max(user_counts.values(), default=0) / max(1, sum(user_counts.values()))
            span_hours = max(0.0, (float(item["last_seen_at"]) - float(item["first_seen_at"])) / 3600)
            fast_track = messages >= fast_messages and users >= fast_users
            attempts = int(item.get("llm_attempts") or 0)
            retry_ready = (
                item.get("state") != "deferred"
                or (
                    attempts < max_attempts
                    and now >= float(item.get("next_review_at") or (float(item.get("last_llm_at") or 0) + retry_days * 86400))
                    and messages >= max(min_messages, int(item.get("evidence_count_at_review") or 0) * retry_multiplier)
                )
            )

            frequency = min(1.0, math.log1p(messages) / math.log1p(max(min_messages * 4, 2)))
            diversity = min(1.0, users / max(required_users, 1))
            boundary = (_entropy(item.get("left_neighbors") or {}) + _entropy(item.get("right_neighbors") or {})) / 2
            span_score = min(1.0, span_hours / max(min_span, 1.0))
            cohesion_parts = []
            term = item["term"]
            for index in range(2, len(term) - 1):
                left_count = counts.get((item["group_id"], term[:index]))
                right_count = counts.get((item["group_id"], term[index:]))
                if left_count and right_count:
                    p_term = messages / max(1, length_totals[(item["group_id"], len(term))])
                    p_left = left_count / max(1, length_totals[(item["group_id"], index)])
                    p_right = right_count / max(
                        1, length_totals[(item["group_id"], len(term) - index)]
                    )
                    pmi = math.log((p_term + 1e-12) / (p_left * p_right + 1e-12))
                    cohesion_parts.append(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, pmi)))))
            cohesion = min(cohesion_parts, default=0.5)
            score = max(0.0, min(1.0,
                0.25 * frequency + 0.20 * diversity + 0.20 * cohesion
                + 0.15 * boundary + 0.10 * span_score + 0.10 * (1.0 - user_ratio)
            ))
            scores[int(item["id"])] = score
            item["local_score"] = score
            item["span_hours"] = span_hours
            if (
                messages >= min_messages and users >= required_users
                and user_ratio <= max_user_ratio and (span_hours >= min_span or fast_track)
                and retry_ready and attempts < max_attempts
            ):
                eligible.append(item)

        # 使用管理页共用的同源聚类，同时连接包含片段和只错开一字的滑窗。
        clusters: List[CandidateCluster] = []
        for component in cluster_candidate_items(eligible, support_ratio, count_ratio):
            canonical = select_candidate_representative(component)
            contexts: List[Dict[str, str]] = []
            for member in component:
                for context in member.get("contexts") or []:
                    if not any(c.get("user_id") == context.get("user_id") for c in contexts):
                        contexts.append(context)
                    if len(contexts) >= 4:
                        break
            ids = sorted(int(i["id"]) for i in component)
            group_id = str(canonical["group_id"])
            clusters.append(CandidateCluster(
                cluster_id=f"{group_id}:{ids[0]}", group_id=group_id,
                candidate_ids=ids, terms=[i["term"] for i in component],
                canonical_hint=canonical["term"],
                message_count=max(int(i["message_count"]) for i in component),
                user_count=max(len(i.get("user_counts") or {}) for i in component),
                span_hours=max(float(i["span_hours"]) for i in component),
                local_score=max(float(i["local_score"]) for i in component),
                contexts=contexts,
            ))
        clusters.sort(key=lambda c: (c.local_score, c.message_count, c.user_count), reverse=True)
        self.scores = scores
        return clusters

"""暗语学习服务：采集、批量审查、晋升、生命周期和注入匹配。"""

import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from iris_memory.config import get_config
from iris_memory.core import get_logger
from iris_memory.learning.storage import LearningStorage
from .extractor import CandidateExtractor
from .models import CandidateCluster, ReviewVerdict
from .reviewer import JargonReviewer
from .scorer import CandidateScorer

logger = get_logger("learning.jargon")
_APPROVED_CATEGORIES = {"slang", "group_code", "nickname", "meme", "abbreviation"}


class JargonLearner:
    def __init__(self, storage: LearningStorage):
        self._storage = storage
        config = get_config()
        self._extractor = CandidateExtractor(
            config.get_int("learning_jargon_ngram_max", 6) or 6,
            config.get_int("learning_jargon_candidates_per_message", 64) or 64,
        )
        self._scorer = CandidateScorer()
        self._reviewer = JargonReviewer()

    def record_message(
        self, group_id: str, user_id: str, raw_text: str, *, is_group: bool, is_bot: bool = False,
        now: Optional[float] = None,
    ) -> int:
        if self._extractor.should_skip(raw_text, is_group=is_group, is_bot=is_bot):
            return 0
        extracted = self._extractor.extract(raw_text)
        config = get_config()
        cooldown = config.get_float("learning_jargon_user_cooldown_minutes", 30) or 30
        cooldown *= 60
        timestamp = now or time.time()
        window_seconds = (config.get_int("learning_jargon_window_days", 14) or 14) * 86400
        self._storage.record_jargon_group_activity(group_id, user_id, timestamp)
        formal_terms = self._storage.observe_formal_jargon(
            group_id, extracted.normalized, user_id, timestamp, cooldown, window_seconds
        )
        if formal_terms:
            extracted.observations = [
                obs for obs in extracted.observations
                if not any(obs["term"] in formal for formal in formal_terms)
            ]
        if not extracted.observations:
            return 0
        return self._storage.record_jargon_observations(
            group_id, user_id, extracted.message_hash, extracted.normalized,
            extracted.observations, timestamp, cooldown,
        )

    def prepare_review(self, now: Optional[float] = None) -> List[CandidateCluster]:
        now = now or time.time()
        config = get_config()
        window = config.get_int("learning_jargon_window_days", 14) or 14
        snapshots = self._storage.get_jargon_candidate_snapshots(window, now)
        clusters = self._scorer.score_and_cluster(snapshots, now)
        self._storage.set_candidate_scores(getattr(self._scorer, "scores", {}))
        if not clusters:
            return []

        trigger = config.get_int("learning_jargon_llm_trigger_size", 8) or 8
        max_wait = (config.get_float("learning_jargon_llm_max_wait_hours", 6.0) or 6.0) * 3600
        snapshot_by_id = {int(s["id"]): s for s in snapshots}
        oldest = min(float(snapshot_by_id[c.candidate_ids[0]]["first_seen_at"]) for c in clusters)
        if len(clusters) < trigger and now - oldest < max_wait:
            return []

        batch_size = config.get_int("learning_jargon_llm_batch_size", 12) or 12
        per_group = config.get_int("learning_jargon_max_clusters_per_group", 4) or 4
        selected, group_counts = [], defaultdict(int)
        for cluster in clusters:
            if group_counts[cluster.group_id] >= per_group:
                continue
            selected.append(cluster)
            group_counts[cluster.group_id] += 1
            if len(selected) >= batch_size:
                break
        if not selected:
            return []

        day = datetime.fromtimestamp(now).date().isoformat()
        daily_limit = config.get_int("learning_jargon_llm_daily_limit", 4) or 4
        min_interval = (
            config.get_float("learning_jargon_llm_min_interval_hours", 6.0) or 6.0
        ) * 3600
        if not self._storage.reserve_jargon_llm_call(day, len(selected), daily_limit, min_interval, now):
            return []

        token = uuid.uuid4().hex
        ids = [cid for cluster in selected for cid in cluster.candidate_ids]
        claimed = {int(row["id"]) for row in self._storage.claim_jargon_candidates(ids, token, now)}
        result = []
        for cluster in selected:
            if all(cid in claimed for cid in cluster.candidate_ids):
                cluster.review_token = token
                result.append(cluster)
        if len(result) != len(selected):
            self._storage.release_jargon_claim(token)
            return []
        return result

    async def request_verdicts(
        self, clusters: List[CandidateCluster], llm_manager: Any
    ) -> Optional[List[ReviewVerdict]]:
        return await self._reviewer.review(clusters, llm_manager)

    def apply_verdicts(
        self, clusters: List[CandidateCluster], verdicts: Optional[List[ReviewVerdict]],
        now: Optional[float] = None,
    ) -> None:
        if not clusters:
            return
        token = clusters[0].review_token
        now = now or time.time()
        config = get_config()
        approve_conf = config.get_float("learning_jargon_approve_confidence", 0.85) or 0.85
        reject_conf = config.get_float("learning_jargon_reject_confidence", 0.75) or 0.75
        retry_days = config.get_int("learning_jargon_retry_days", 7) or 7
        if not verdicts:
            for cluster in clusters:
                self._storage.apply_jargon_verdict(
                    cluster.candidate_ids, token, "defer", "uncertain",
                    cluster.canonical_hint, "", 0.0, "LLM 调用或输出解析失败",
                    cluster.message_count, now + retry_days * 86400, [],
                )
            return
        by_cluster = {cluster.cluster_id: cluster for cluster in clusters}
        handled = set()
        for verdict in verdicts:
            cluster = by_cluster.get(verdict.cluster_id)
            if not cluster:
                continue
            decision = "defer"
            if (
                verdict.decision == "approve" and verdict.category in _APPROVED_CATEGORIES
                and verdict.confidence >= approve_conf and verdict.meaning
            ):
                decision = "approve"
            elif verdict.decision == "reject" and verdict.confidence >= reject_conf:
                decision = "reject"
            next_review = now + retry_days * 86400 if decision == "defer" else None
            self._storage.apply_jargon_verdict(
                cluster.candidate_ids, token, decision, verdict.category,
                verdict.canonical_term, verdict.meaning, verdict.confidence,
                verdict.reason, cluster.message_count, next_review, verdict.aliases,
            )
            handled.update(cluster.candidate_ids)
        unhandled = [c for c in clusters if not all(cid in handled for cid in c.candidate_ids)]
        for cluster in unhandled:
            self._storage.apply_jargon_verdict(
                cluster.candidate_ids, token, "defer", "uncertain",
                cluster.canonical_hint, "", 0.0, "LLM 未返回该候选簇",
                cluster.message_count, now + retry_days * 86400, [],
            )

    def maintain(self, now: Optional[float] = None) -> Dict[str, int]:
        now = now or time.time()
        config = get_config()
        return self._storage.maintain_jargon(
            now,
            config.get_int("learning_jargon_window_days", 14) or 14,
            config.get_int("learning_jargon_candidate_expire_days", 30) or 30,
            config.get_int("learning_jargon_rejected_retention_days", 60) or 60,
            config.get_int("learning_jargon_dormant_days", 60) or 60,
            config.get_int("learning_jargon_max_candidates_per_group", 3000) or 3000,
        )

    def match_terms(self, group_id: str, text: str, max_items: int = 5) -> List[Dict[str, Any]]:
        if not text:
            return []
        matches = []
        for item in self._storage.get_active_jargon(group_id):
            variants = [item["term"], *(item.get("aliases") or [])]
            for variant in variants:
                start = 0
                while variant and (position := text.find(variant, start)) >= 0:
                    matches.append((position, position + len(variant), len(variant), item))
                    start = position + len(variant)
        matches.sort(key=lambda m: (-m[2], -float(m[3].get("confidence") or 0), m[0]))
        selected, occupied, seen = [], [], set()
        for start, end, _, item in matches:
            if int(item["id"]) in seen or any(not (end <= a or start >= b) for a, b in occupied):
                continue
            selected.append(item)
            occupied.append((start, end))
            seen.add(int(item["id"]))
            if len(selected) >= max_items:
                break
        self._storage.record_jargon_hits([int(item["id"]) for item in selected])
        return selected

"""
Iris Chat Memory - 人格自迭代均衡抽样器

文档 §7.4 分层均衡抽样：
1. 先按群分桶，再按用户分桶；
2. 多群场景单群不超过 group_max_ratio（默认 35%）；
3. 多用户场景单用户不超过 user_max_ratio（默认 20%）；
4. 指定单群/单用户时取消相应比例限制；
5. 桶内按 Van der Corput 均匀散布顺序抽取，覆盖不同时间段；
6. 桶间轮转抽取，不按消息量加权，防止高活跃群/用户支配人格。
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from iris_memory.core import get_logger

logger = get_logger("persona_evolution.sampler")


def _spread_order(n: int) -> List[int]:
    """生成 0..n-1 的均匀散布顺序（Van der Corput 序列排序）

    前 k 个下标始终均匀覆盖整个区间，保证先抽到的样本
    分散在不同时间段，而非集中在最近一次突发话题。
    """

    def vdc(i: int) -> float:
        result, base = 0.0, 0.5
        while i:
            result += base * (i & 1)
            i >>= 1
            base /= 2
        return result

    return sorted(range(n), key=vdc)


def stratified_sample(
    samples: List[Dict[str, Any]],
    max_count: int = 60,
    group_max_ratio: float = 0.35,
    user_max_ratio: float = 0.20,
    single_group_scope: bool = False,
    single_user_scope: bool = False,
) -> List[Dict[str, Any]]:
    """分层均衡抽样

    Args:
        samples: 候选语料（字典列表，含 id/group_id/user_id/created_at）
        max_count: 最多抽取条数
        group_max_ratio: 多群场景单群占比上限
        user_max_ratio: 多用户场景单用户占比上限
        single_group_scope: Job 指定单群时为 True，取消群占比限制
        single_user_scope: Job 指定单用户时为 True，取消用户占比限制

    Returns:
        抽取的语料列表（按 id 升序）
    """
    if max_count <= 0 or not samples:
        return []
    if len(samples) <= max_count:
        return sorted(samples, key=lambda s: s.get("id", 0))

    distinct_groups = {s.get("group_id", "") for s in samples}
    distinct_users = {s.get("user_id", "") for s in samples}

    # 只有一个群/用户时比例限制无意义；指定单群/单用户时同样取消
    apply_group_cap = len(distinct_groups) > 1 and not single_group_scope
    apply_user_cap = len(distinct_users) > 1 and not single_user_scope

    group_cap = max(1, int(max_count * group_max_ratio)) if apply_group_cap else max_count
    user_cap = max(1, int(max_count * user_max_ratio)) if apply_user_cap else max_count

    # 按 (群, 用户) 分桶，桶内按时间均匀散布排序
    buckets: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.get("group_id", ""), sample.get("user_id", ""))].append(sample)
    bucket_queues: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for key, items in buckets.items():
        items.sort(key=lambda s: (s.get("created_at", 0.0), s.get("id", 0)))
        order = _spread_order(len(items))
        bucket_queues[key] = [items[i] for i in order]

    group_buckets: Dict[str, List[tuple[str, str]]] = defaultdict(list)
    for key in bucket_queues:
        group_buckets[key[0]].append(key)

    # 群→用户双层轮转抽取，不持锁不按消息量加权
    selected: List[Dict[str, Any]] = []
    group_taken: Dict[str, int] = defaultdict(int)
    user_taken: Dict[str, int] = defaultdict(int)

    while len(selected) < max_count:
        progressed = False
        for group_id in sorted(group_buckets):
            if len(selected) >= max_count:
                break
            if group_taken[group_id] >= group_cap:
                continue
            for key in sorted(group_buckets[group_id]):
                if len(selected) >= max_count:
                    break
                user_id = key[1]
                queue = bucket_queues[key]
                if not queue:
                    continue
                if group_taken[group_id] >= group_cap:
                    break
                if user_taken[user_id] >= user_cap:
                    continue
                selected.append(queue.pop(0))
                group_taken[group_id] += 1
                user_taken[user_id] += 1
                progressed = True
        if not progressed:
            break

    return sorted(selected, key=lambda s: s.get("id", 0))

"""暗语候选的确定性同源聚类。

提取层有意保留高召回的中文 n-gram；本模块把来自同一批消息的包含片段、
以及只错开一个字符的相邻滑窗折叠为稳定候选簇，供本地审查和管理页共用。
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _support_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_support = set(left.get("support_hashes") or [])
    right_support = set(right.get("support_hashes") or [])
    if not left_support or not right_support:
        return 0.0
    return len(left_support & right_support) / min(len(left_support), len(right_support))


def _frequency_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_count = max(0, int(left.get("message_count") or 0))
    right_count = max(0, int(right.get("message_count") or 0))
    maximum = max(left_count, right_count)
    return min(left_count, right_count) / maximum if maximum else 0.0


def _edge_overlap(left: str, right: str) -> int:
    """返回两个词首尾相接时的最大重叠长度。"""
    maximum = min(len(left), len(right)) - 1
    for size in range(maximum, 1, -1):
        if left[-size:] == right[:size] or right[-size:] == left[:size]:
            return size
    return 0


def _lexically_related(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    if left in right or right in left:
        return True
    # 直接连接等长或近等长、只错开一个字符的滑窗。即使中间的短词因单条
    # 消息候选上限被截掉，也仍能把“终于收到地球/于收到地球信”归为一簇。
    required = max(2, min(len(left), len(right)) - 1)
    return _edge_overlap(left, right) >= required


def candidate_representative_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    """代表词排序键：已判定 canonical 优先，其次闭合长词、统计分和证据量。"""
    term = str(item.get("term") or "")
    canonical = str(item.get("canonical_term") or "")
    return (
        int(bool(canonical and term == canonical)),
        len(term),
        float(item.get("local_score") or 0.0),
        int(item.get("message_count") or 0),
        int(item.get("user_count") or len(item.get("user_counts") or {})),
        -int(item.get("id") or 0),
    )


def select_candidate_representative(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("候选簇不能为空")
    return max(items, key=candidate_representative_key)


def _candidate_pairs(items: Sequence[Dict[str, Any]]) -> Iterable[Tuple[int, int]]:
    """按包含子串和首尾滑窗索引产生少量候选边，避免全量 O(n²) 比较。"""
    by_term: Dict[str, List[int]] = defaultdict(list)
    prefix_index: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for index, item in enumerate(items):
        term = str(item.get("term") or "")
        by_term[term].append(index)
        for size in range(2, len(term)):
            prefix_index[(size, term[:size])].append(index)

    pairs = set()
    for index, item in enumerate(items):
        term = str(item.get("term") or "")
        # 包含关系：枚举短词而不是两两扫描。
        for start in range(len(term)):
            for end in range(start + 2, len(term) + 1):
                fragment = term[start:end]
                if fragment == term:
                    continue
                for other in by_term.get(fragment, []):
                    pairs.add((min(index, other), max(index, other)))
        # 相邻滑窗：用当前后缀查其他词前缀，最终关系条件会限制为仅错一字。
        for size in range(2, len(term)):
            for other in prefix_index.get((size, term[-size:]), []):
                if other != index:
                    pairs.add((min(index, other), max(index, other)))
    return pairs


def cluster_candidate_items(
    items: Sequence[Dict[str, Any]],
    support_ratio: float = 0.85,
    count_ratio: float = 0.8,
    partition_fields: Sequence[str] = ("group_id",),
) -> List[List[Dict[str, Any]]]:
    """将候选折叠为确定性同源簇。

    只有文本相邻、证据集合高度重合且频次接近的候选才会连边。簇 ID 可由
    调用方使用最小候选 ID 构造，因此不会受输入顺序或分页影响。
    """
    partitions: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        partitions[tuple(item.get(field) for field in partition_fields)].append(item)

    result: List[List[Dict[str, Any]]] = []
    for partition in partitions.values():
        ordered = sorted(partition, key=lambda item: int(item.get("id") or 0))
        parents = list(range(len(ordered)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        for left_index, right_index in _candidate_pairs(ordered):
            left, right = ordered[left_index], ordered[right_index]
            if not _lexically_related(str(left.get("term") or ""), str(right.get("term") or "")):
                continue
            if _frequency_similarity(left, right) < count_ratio:
                continue
            if _support_similarity(left, right) < support_ratio:
                continue
            union(left_index, right_index)

        components: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for index, item in enumerate(ordered):
            components[find(index)].append(item)
        result.extend(components.values())

    for component in result:
        component.sort(key=candidate_representative_key, reverse=True)
    result.sort(
        key=lambda component: (
            float(select_candidate_representative(component).get("local_score") or 0.0),
            int(select_candidate_representative(component).get("message_count") or 0),
            -min(int(item.get("id") or 0) for item in component),
        ),
        reverse=True,
    )
    return result

"""
Iris Chat Memory - 遗忘权重算法

实现记忆遗忘评分算法，用于评估记忆的重要性和淘汰优先级。

算法公式：S = w1·R + w2·F + w3·C + w4·(1 - D)

其中：
- R (Recency): 近因性 - 最近访问时间的影响
- F (Frequency): 频率性 - 访问次数的影响
- C (Confidence): 置信度 - 记忆质量的影响
- D (Degree): 孤立度 - 缺乏关联的影响（图谱中使用）

得分越高，记忆越重要，越不容易被淘汰。
"""

from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING, cast
import math

from iris_memory.config import get_config

if TYPE_CHECKING:
    from iris_memory.l2_memory.models import MemoryEntry


# ============================================================================
# 遗忘权重计算
# ============================================================================


def calculate_recency(
    last_access_time: Optional[str], lambda_decay: float = 0.1
) -> float:
    """计算近因性得分

    使用指数衰减函数计算近因性得分，最近访问的记忆得分更高。

    Args:
        last_access_time: 最近访问时间（ISO 格式字符串）
        lambda_decay: 衰减系数，越大则衰减越快

    Returns:
        近因性得分 [0, 1]，越接近 1 表示越近期访问

    Examples:
        >>> now = datetime.now().isoformat()
        >>> calculate_recency(now)
        1.0
        >>> old_time = (datetime.now() - timedelta(days=30)).isoformat()
        >>> calculate_recency(old_time)
        0.05
    """
    if not last_access_time:
        # 无访问记录，使用创建时间的一半得分
        return 0.5

    try:
        access_dt = datetime.fromisoformat(last_access_time)
        now = datetime.now()
        days_elapsed = (now - access_dt).total_seconds() / 86400

        # 指数衰减：exp(-lambda * t)
        recency = math.exp(-lambda_decay * days_elapsed)
        return max(0.0, min(1.0, recency))

    except (ValueError, TypeError):
        return 0.5


def calculate_frequency(access_count: int, max_count: int = 100) -> float:
    """计算频率性得分

    使用对数函数计算频率得分，访问次数越多得分越高。

    Args:
        access_count: 访问次数
        max_count: 参考最大访问次数（用于归一化）

    Returns:
        频率性得分 [0, 1]，越接近 1 表示访问越频繁

    Examples:
        >>> calculate_frequency(0)
        0.0
        >>> calculate_frequency(10)
        0.52
        >>> calculate_frequency(100)
        1.0
    """
    if access_count <= 0:
        return 0.0

    # 对数归一化：log(count + 1) / log(max_count + 1)
    normalized = math.log(access_count + 1) / math.log(max_count + 1)
    return max(0.0, min(1.0, normalized))


def calculate_confidence(confidence: float) -> float:
    """计算置信度得分

    直接返回置信度值，假设置信度已在 [0, 1] 范围内。

    Args:
        confidence: 原始置信度值

    Returns:
        置信度得分 [0, 1]

    Examples:
        >>> calculate_confidence(0.85)
        0.85
    """
    return max(0.0, min(1.0, confidence))


def calculate_isolation_degree(metadata: Dict[str, Any]) -> float:
    """计算孤立度得分

    孤立度表示记忆缺乏关联的程度。在 L2 阶段返回固定值 0，
    在 L3 知识图谱阶段根据节点连接数计算。

    公式：D = 1.0 / (connected_count + 1)

    Args:
        metadata: 记忆元数据，包含 connected_count 字段

    Returns:
        孤立度得分 [0, 1]，越接近 1 表示越孤立

    Note:
        - L2 阶段：connected_count 为 0，返回 0（不参与评分）
        - L3 阶段：connected_count 为节点的连接边数，连接越多孤立度越低
    """
    # 从 metadata 中获取连接数
    connected_count = metadata.get("connected_count", 0)

    has_connected_count = "connected_count" in metadata

    if not has_connected_count:
        return 0.0

    if connected_count == 0:
        return 1.0

    isolation = 1.0 / (connected_count + 1)

    return isolation


def calculate_forgetting_score(
    entry: "MemoryEntry", weights: Optional[Dict[str, float]] = None
) -> float:
    """计算综合遗忘评分

    综合考虑近因性、频率性、置信度和孤立度，计算记忆的重要性得分。
    得分越高，记忆越重要，越不容易被淘汰。

    公式：S = w1·R + w2·F + w3·C + w4·(1 - D)

    Args:
        entry: 记忆条目
        weights: 权重字典，包含 w1, w2, w3, w4

    Returns:
        综合评分 [0, 1]，越接近 1 表示越重要

    Examples:
        >>> from iris_memory.l2_memory.models import MemoryEntry
        >>> entry = MemoryEntry(
        ...     id="mem_001",
        ...     content="测试记忆",
        ...     metadata={
        ...         "last_access_time": datetime.now().isoformat(),
        ...         "access_count": 5,
        ...         "confidence": 0.85
        ...     }
        ... )
        >>> score = calculate_forgetting_score(entry)
        >>> 0 < score < 1
        True
    """
    config = get_config()

    if weights is None:
        # L2 记忆无 connected_count，calculate_isolation_degree 恒返回 0，
        # isolation 权重项不产生实际贡献；权重从隐藏配置读取（fallback 与
        # dataclass 默认一致，供 config 系统未就绪时兜底）。
        weights = {
            "w1": cast(float, config.get("forgetting_l2_weight_recency", 0.4)),
            "w2": cast(float, config.get("forgetting_l2_weight_frequency", 0.35)),
            "w3": cast(float, config.get("forgetting_l2_weight_confidence", 0.25)),
            "w4": cast(float, config.get("forgetting_l2_weight_isolation", 0.0)),
        }

    lambda_decay = float(config.get("forgetting_lambda", 0.1))  # type: ignore[arg-type]

    R = calculate_recency(entry.last_access_time, lambda_decay=lambda_decay)
    F = calculate_frequency(entry.access_count)
    C = calculate_confidence(entry.confidence)
    D = calculate_isolation_degree(entry.metadata)

    score = (
        weights["w1"] * R
        + weights["w2"] * F
        + weights["w3"] * C
        + weights["w4"] * (1 - D)
    )

    weight_sum = sum(weights.values())
    if weight_sum > 0:
        score /= weight_sum

    return max(0.0, min(1.0, score))


def should_evict(
    entry: "MemoryEntry", threshold: Optional[float] = None, retention_days: int = 30
) -> bool:
    """判断记忆是否应该被淘汰

    综合考虑遗忘评分、保留期和低置信度标记，判断记忆是否应该被淘汰。

    淘汰条件（满足任一即淘汰）：
    1. 遗忘评分极低（低于 immediate_eviction_threshold），无需等待保留期直接淘汰
    2. 遗忘评分低于阈值 且 距上次访问超过保留期
    3. 被标记为低置信度的记忆，阈值提高 30% 以加速淘汰
    4. 被标记为无主体的记忆（subjectless），阈值提高 20% 以加速淘汰

    Args:
        entry: 记忆条目
        threshold: 遗忘阈值，None 时使用配置值 forgetting_threshold
        retention_days: 保留期天数

    Returns:
        是否应该被淘汰

    Examples:
        >>> from iris_memory.l2_memory.models import MemoryEntry
        >>> entry = MemoryEntry(
        ...     id="mem_001",
        ...     content="旧记忆",
        ...     metadata={
        ...         "last_access_time": "2024-01-01T00:00:00",
        ...         "access_count": 0,
        ...         "confidence": 0.1
        ...     }
        ... )
        >>> should_evict(entry)
        True
    """
    config = get_config()
    evict_threshold = (
        threshold if threshold is not None
        else cast(float, config.get("forgetting_threshold", 0.3))
    )
    immediate_threshold = cast(
        float, config.get("forgetting_immediate_eviction_threshold", 0.1)
    )

    if entry.metadata.get("low_confidence"):
        evict_threshold *= 1.3

    # 无主体记忆（总结时未能关联到具体用户）加速淘汰：
    # 这类记忆无法在下游 L3 图谱中建立 Person 关联，长期占据 L2 无实际价值。
    # 提高淘汰阈值 20%（比 low_confidence 的 30% 温和，因为提示词优化后
    # LLM 已尽量包含主体，仍无主体的可能是群聊通用话题）。
    if entry.metadata.get("subjectless"):
        evict_threshold *= 1.2

    score = calculate_forgetting_score(entry)

    if score < immediate_threshold:
        return True

    if score < evict_threshold:
        # 评分低于阈值，检查保留期
        last_access = entry.last_access_time
        if last_access:
            try:
                access_dt = datetime.fromisoformat(last_access)
                days_elapsed = (datetime.now() - access_dt).days

                if days_elapsed > retention_days:
                    return True
            except (ValueError, TypeError):
                pass
        else:
            # 无访问记录，根据评分决定
            return True

    return False


def calculate_kg_forgetting_score(
    last_access_time: Optional[str] = None,
    access_count: int = 0,
    confidence: float = 1.0,
    connected_count: int = 0,
    source_memory_count: int = 0,
    lambda_decay: float = 0.05,
) -> float:
    """L3 知识图谱专用遗忘评分

    L3 是高度抽象的结构化知识，遗忘逻辑应与 L2 不同：
    - 结构重要性（连接度）远比访问频率重要
    - 来源记忆数越多，节点越稳固（被多次验证）
    - 近因性衰减更慢（抽象知识时效性更长）

    公式：S = w1·R + w2·(1-D) + w3·C + w4·V

    其中：
    - R (Recency): 近因性，衰减系数更低（0.05 vs 0.1）
    - D (Degree): 孤立度 = 1/(connected_count+1)
    - C (Confidence): 置信度
    - V (Verification): 验证度 = min(1.0, log(source_memory_count + 1) / log(6))，
      source_memory_count 为 0 时取 0；对数曲线使少量来源即可获得较高验证度，
      与 ``source_memory_count >= 3 永不淘汰`` 的保护阈值相配合。

    权重：w1=0.15, w2=0.40, w3=0.15, w4=0.30
    结构重要性（1-D）占 40%，验证度占 30%，远高于 L2。

    Args:
        last_access_time: 最近访问时间（ISO 格式字符串）
        access_count: 访问次数
        confidence: 置信度
        connected_count: 连接边数
        source_memory_count: 来源记忆数量
        lambda_decay: 衰减系数（默认 0.05，比 L2 更慢）

    Returns:
        综合评分 [0, 1]，越接近 1 表示越重要
    """
    R = calculate_recency(last_access_time, lambda_decay=lambda_decay)

    if connected_count == 0:
        D = 1.0
    else:
        D = 1.0 / (connected_count + 1)

    C = calculate_confidence(confidence)

    V = (
        min(1.0, math.log(source_memory_count + 1) / math.log(6))
        if source_memory_count > 0
        else 0.0
    )

    config = get_config()
    w_recency = cast(float, config.get("forgetting_kg_weight_recency", 0.15))
    w_structure = cast(float, config.get("forgetting_kg_weight_structure", 0.40))
    w_confidence = cast(float, config.get("forgetting_kg_weight_confidence", 0.15))
    w_verification = cast(float, config.get("forgetting_kg_weight_verification", 0.30))

    score = (
        w_recency * R + w_structure * (1 - D) + w_confidence * C + w_verification * V
    )

    weight_sum = w_recency + w_structure + w_confidence + w_verification
    if weight_sum > 0:
        score /= weight_sum

    return max(0.0, min(1.0, score))


def should_evict_kg_node(
    last_access_time: Optional[str] = None,
    access_count: int = 0,
    confidence: float = 1.0,
    connected_count: int = 0,
    source_memory_count: int = 0,
    threshold: float = 0.3,
    retention_days: int = 30,
) -> bool:
    """判断 L3 知识图谱节点是否应该被淘汰

    与 L2 不同：
    - 连接度 >= 5 的枢纽节点永不淘汰
    - 来源记忆数 >= 3 的节点永不淘汰
    - 评分低于阈值且超过保留期才淘汰

    Args:
        last_access_time: 最近访问时间
        access_count: 访问次数
        confidence: 置信度
        connected_count: 连接边数
        source_memory_count: 来源记忆数量
        threshold: 遗忘阈值
        retention_days: 保留期天数

    Returns:
        是否应该被淘汰
    """
    if connected_count >= 5:
        return False

    if source_memory_count >= 3:
        return False

    lambda_decay = cast(float, get_config().get("forgetting_lambda_kg", 0.05))
    score = calculate_kg_forgetting_score(
        last_access_time=last_access_time,
        access_count=access_count,
        confidence=confidence,
        connected_count=connected_count,
        source_memory_count=source_memory_count,
        lambda_decay=lambda_decay,
    )

    if score < threshold:
        if last_access_time:
            try:
                access_dt = datetime.fromisoformat(last_access_time)
                days_elapsed = (datetime.now() - access_dt).days
                if days_elapsed > retention_days:
                    return True
            except (ValueError, TypeError):
                pass
        else:
            return True

    return False

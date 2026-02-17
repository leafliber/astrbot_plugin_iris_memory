"""
场景自适应配置模块

包含两个核心组件：
1. GroupActivityTracker - 群活跃度追踪器，基于滑动窗口计算群活跃度
2. ActivityAwareConfigProvider - 活跃度感知配置提供者，根据活跃度返回调整后的配置

设计要点：
- 滑动窗口（默认3小时）加权平均，避免瞬时波动
- 滞后(hysteresis)机制防止边界振荡
- LRU 缓存减少重复计算
- 通过 KV 存储持久化群活跃度状态
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core.defaults import (
    ACTIVITY_HYSTERESIS_RATIO,
    ACTIVITY_PRESETS,
    ACTIVITY_THRESHOLDS,
    ACTIVITY_WINDOW_HOURS,
    DEFAULTS,
    GroupActivityLevel,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("activity_config")


# ========== 群活跃度追踪器 ==========


@dataclass
class _HourlyBucket:
    """单小时消息桶"""
    hour_ts: int  # 该小时的开始时间戳（整点）
    count: int = 0


@dataclass
class GroupActivityState:
    """单个群的活跃度状态（可序列化）"""
    group_id: str
    level: GroupActivityLevel = GroupActivityLevel.MODERATE
    hourly_counts: List[Tuple[int, int]] = field(default_factory=list)
    # (hour_ts, count) 列表，保留最近 ACTIVITY_WINDOW_HOURS 小时
    last_calc_ts: float = 0.0
    messages_per_hour: float = 0.0  # 上一次计算的每小时消息数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "level": self.level.value,
            "hourly_counts": self.hourly_counts,
            "last_calc_ts": self.last_calc_ts,
            "messages_per_hour": self.messages_per_hour,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupActivityState":
        level_str = data.get("level", "moderate")
        try:
            level = GroupActivityLevel(level_str)
        except ValueError:
            level = GroupActivityLevel.MODERATE
        return cls(
            group_id=data["group_id"],
            level=level,
            hourly_counts=data.get("hourly_counts", []),
            last_calc_ts=data.get("last_calc_ts", 0.0),
            messages_per_hour=data.get("messages_per_hour", 0.0),
        )


class GroupActivityTracker:
    """群活跃度追踪器

    职责：
    - 记录每个群每小时的消息数量
    - 基于滑动窗口计算加权平均活跃度
    - 应用滞后机制确定活跃度等级
    """

    def __init__(
        self,
        window_hours: int = ACTIVITY_WINDOW_HOURS,
        hysteresis_ratio: float = ACTIVITY_HYSTERESIS_RATIO,
        calc_interval: int = DEFAULTS.activity_adaptive.activity_calc_interval,
    ):
        self._window_hours = window_hours
        self._hysteresis_ratio = hysteresis_ratio
        self._calc_interval = calc_interval

        # group_id -> GroupActivityState
        self._states: Dict[str, GroupActivityState] = {}

    # ---------- 消息计数 ----------

    def record_message(self, group_id: str) -> None:
        """记录一条群消息（由 SessionManager 或消息处理流程调用）"""
        if not group_id:
            return

        state = self._ensure_state(group_id)
        current_hour = self._current_hour_ts()

        # 在最新桶中计数
        if state.hourly_counts and state.hourly_counts[-1][0] == current_hour:
            state.hourly_counts[-1] = (current_hour, state.hourly_counts[-1][1] + 1)
        else:
            state.hourly_counts.append((current_hour, 1))

        # 清理过期桶
        self._prune_buckets(state)

    # ---------- 活跃度计算 ----------

    def get_activity_level(self, group_id: str) -> GroupActivityLevel:
        """获取群活跃度等级（带缓存，间隔内不重新计算）"""
        if not group_id:
            return GroupActivityLevel.MODERATE

        state = self._ensure_state(group_id)
        now = time.time()

        if now - state.last_calc_ts >= self._calc_interval:
            self._recalculate(state)

        return state.level

    def get_messages_per_hour(self, group_id: str) -> float:
        """获取群每小时消息数（上次计算值）"""
        state = self._states.get(group_id)
        if not state:
            return 0.0
        return state.messages_per_hour

    def force_recalculate(self, group_id: str) -> GroupActivityLevel:
        """强制重新计算活跃度"""
        state = self._ensure_state(group_id)
        self._recalculate(state)
        return state.level

    def get_all_states(self) -> Dict[str, GroupActivityState]:
        """获取所有群的活跃度状态"""
        return dict(self._states)

    # ---------- 核心计算 ----------

    def _recalculate(self, state: GroupActivityState) -> None:
        """重新计算活跃度等级（加权平均 + 滞后）"""
        self._prune_buckets(state)
        mph = self._weighted_average(state)
        state.messages_per_hour = mph
        state.last_calc_ts = time.time()

        new_level = self._classify_with_hysteresis(mph, state.level)
        if new_level != state.level:
            old_level = state.level
            state.level = new_level
            logger.info(
                f"Group {state.group_id} activity: "
                f"{old_level.value} -> {new_level.value} "
                f"({mph:.1f} msgs/h)"
            )

    def _weighted_average(self, state: GroupActivityState) -> float:
        """滑动窗口加权平均

        最新一小时权重 1.0，前一小时 0.7，更早 0.5
        """
        if not state.hourly_counts:
            return 0.0

        current_hour = self._current_hour_ts()
        weights = {0: 1.0, 1: 0.7, 2: 0.5}
        total_weighted = 0.0
        total_weight = 0.0

        for hour_ts, count in state.hourly_counts:
            hours_ago = (current_hour - hour_ts) // 3600
            w = weights.get(hours_ago, 0.3)
            total_weighted += count * w
            total_weight += w

        if total_weight == 0:
            return 0.0
        return total_weighted / total_weight

    def _classify_with_hysteresis(
        self, mph: float, current_level: GroupActivityLevel
    ) -> GroupActivityLevel:
        """带滞后的活跃度分类

        升级需超阈值 (1 + hysteresis_ratio)
        降级需低于阈值 (1 - hysteresis_ratio)
        """
        quiet_upper = ACTIVITY_THRESHOLDS["quiet_upper"]
        moderate_upper = ACTIVITY_THRESHOLDS["moderate_upper"]
        active_upper = ACTIVITY_THRESHOLDS["active_upper"]
        h = self._hysteresis_ratio

        # 确定正常分级（无滞后）
        if mph < quiet_upper:
            raw_level = GroupActivityLevel.QUIET
        elif mph < moderate_upper:
            raw_level = GroupActivityLevel.MODERATE
        elif mph < active_upper:
            raw_level = GroupActivityLevel.ACTIVE
        else:
            raw_level = GroupActivityLevel.INTENSIVE

        # 如果与当前等级相同，无变化
        if raw_level == current_level:
            return current_level

        # 等级排序
        level_order = [
            GroupActivityLevel.QUIET,
            GroupActivityLevel.MODERATE,
            GroupActivityLevel.ACTIVE,
            GroupActivityLevel.INTENSIVE,
        ]
        raw_idx = level_order.index(raw_level)
        cur_idx = level_order.index(current_level)

        if raw_idx > cur_idx:
            # 升级：需超阈值 * (1 + h)
            thresholds = [quiet_upper, moderate_upper, active_upper]
            boundary = thresholds[cur_idx] if cur_idx < len(thresholds) else active_upper
            if mph >= boundary * (1 + h):
                return raw_level
            return current_level
        else:
            # 降级：需低于阈值 * (1 - h)
            thresholds = [quiet_upper, moderate_upper, active_upper]
            boundary = thresholds[raw_idx] if raw_idx < len(thresholds) else quiet_upper
            if mph < boundary * (1 - h):
                return raw_level
            return current_level

    # ---------- 工具方法 ----------

    def _ensure_state(self, group_id: str) -> GroupActivityState:
        if group_id not in self._states:
            self._states[group_id] = GroupActivityState(group_id=group_id)
        return self._states[group_id]

    def _current_hour_ts(self) -> int:
        """获取当前整点时间戳"""
        now = int(time.time())
        return now - (now % 3600)

    def _prune_buckets(self, state: GroupActivityState) -> None:
        """清理超出滑动窗口的桶"""
        cutoff = self._current_hour_ts() - self._window_hours * 3600
        state.hourly_counts = [
            (ts, c) for ts, c in state.hourly_counts if ts >= cutoff
        ]

    # ---------- 序列化 ----------

    def serialize(self) -> Dict[str, Any]:
        """序列化所有群活跃度状态"""
        return {
            gid: state.to_dict()
            for gid, state in self._states.items()
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """从序列化数据恢复"""
        self._states.clear()
        for gid, state_data in data.items():
            try:
                self._states[gid] = GroupActivityState.from_dict(state_data)
            except Exception as e:
                logger.warning(f"Failed to deserialize activity state for {gid}: {e}")
        logger.info(f"Loaded activity states for {len(self._states)} groups")


# ========== 活跃度感知配置提供者 ==========


class ActivityAwareConfigProvider:
    """活跃度感知配置提供者

    根据群活跃度等级返回调整后的配置参数。

    优先级：
    1. 活跃度预设值（如果启用了自适应）
    2. 全局默认值（从 DEFAULTS 获取）
    """

    # 支持的配置键及其在 DEFAULTS 中的位置
    _DEFAULT_FALLBACKS: Dict[str, Tuple[str, str]] = {
        "cooldown_seconds": ("proactive_reply", "cooldown_seconds"),
        "max_daily_replies": ("proactive_reply", "max_daily_replies"),
        "batch_threshold_count": ("message_processing", "batch_threshold_count"),
        "batch_threshold_interval": ("message_processing", "batch_threshold_interval"),
        "daily_analysis_budget": ("image_analysis", "daily_analysis_budget"),
        "chat_context_count": ("llm_integration", "chat_context_count"),
        "reply_temperature": ("proactive_reply", "reply_temperature"),
    }

    def __init__(
        self,
        tracker: GroupActivityTracker,
        enabled: bool = True,
        cache_ttl: int = DEFAULTS.activity_adaptive.config_cache_ttl,
    ):
        self._tracker = tracker
        self._enabled = enabled
        self._cache_ttl = cache_ttl

        # 配置缓存：group_id -> {key: (value, expire_ts)}
        self._cache: Dict[str, Dict[str, Tuple[Any, float]]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            self._cache.clear()

    @property
    def tracker(self) -> GroupActivityTracker:
        return self._tracker

    def get_config(self, group_id: Optional[str], key: str) -> Any:
        """获取指定群的配置值

        Args:
            group_id: 群 ID（None 或空字符串返回默认值）
            key: 配置键，如 "cooldown_seconds"

        Returns:
            调整后的配置值
        """
        # 非群聊 or 未启用自适应 → 返回默认值
        if not group_id or not self._enabled:
            return self._get_default(key)

        # 检查缓存
        cached = self._get_from_cache(group_id, key)
        if cached is not None:
            return cached

        # 获取活跃度等级
        level = self._tracker.get_activity_level(group_id)

        # 从预设获取配置值
        preset_value = ACTIVITY_PRESETS.get(key, level)
        if preset_value is None:
            value = self._get_default(key)
        else:
            value = preset_value

        # 写入缓存
        self._set_cache(group_id, key, value)
        return value

    def get_group_config_snapshot(self, group_id: str) -> Dict[str, Any]:
        """获取指定群的所有自适应配置快照"""
        result: Dict[str, Any] = {}
        for key in self._DEFAULT_FALLBACKS:
            result[key] = self.get_config(group_id, key)
        return result

    def get_group_activity_summary(self, group_id: str) -> Dict[str, Any]:
        """获取群活跃度摘要（用于状态展示）"""
        level = self._tracker.get_activity_level(group_id)
        mph = self._tracker.get_messages_per_hour(group_id)
        config_snapshot = self.get_group_config_snapshot(group_id)
        return {
            "group_id": group_id,
            "activity_level": level.value,
            "messages_per_hour": round(mph, 1),
            "config": config_snapshot,
        }

    def get_all_activity_summaries(self) -> List[Dict[str, Any]]:
        """获取所有群的活跃度摘要"""
        summaries = []
        for gid, state in self._tracker.get_all_states().items():
            summaries.append({
                "group_id": gid,
                "activity_level": state.level.value,
                "messages_per_hour": round(state.messages_per_hour, 1),
            })
        # 按活跃度降序
        level_order = {"intensive": 0, "active": 1, "moderate": 2, "quiet": 3}
        summaries.sort(key=lambda s: level_order.get(s["activity_level"], 99))
        return summaries

    def invalidate_cache(self, group_id: Optional[str] = None) -> None:
        """清除配置缓存"""
        if group_id:
            self._cache.pop(group_id, None)
        else:
            self._cache.clear()

    # ---------- 内部方法 ----------

    def _get_default(self, key: str) -> Any:
        """获取默认配置值"""
        fb = self._DEFAULT_FALLBACKS.get(key)
        if fb:
            section_name, attr_name = fb
            section = getattr(DEFAULTS, section_name, None)
            if section:
                return getattr(section, attr_name, None)
        return None

    def _get_from_cache(self, group_id: str, key: str) -> Any:
        """从缓存获取（TTL 过期返回 None）"""
        group_cache = self._cache.get(group_id)
        if not group_cache:
            return None
        entry = group_cache.get(key)
        if not entry:
            return None
        value, expire_ts = entry
        if time.time() > expire_ts:
            del group_cache[key]
            return None
        return value

    def _set_cache(self, group_id: str, key: str, value: Any) -> None:
        """写入缓存"""
        if group_id not in self._cache:
            self._cache[group_id] = {}
        self._cache[group_id][key] = (value, time.time() + self._cache_ttl)

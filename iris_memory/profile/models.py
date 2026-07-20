"""
Iris Chat Memory - 画像数据模型

定义群聊画像和用户画像的数据结构。
支持三层更新频率和字段置信度管理。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime
from enum import Enum
from difflib import SequenceMatcher


class UpdateTier(Enum):
    """字段更新层级

    中期字段：按时间间隔或总结次数触发LLM分析
    长期字段：仅检测到显著新信息时更新，需高置信度
    """

    MID = "mid"
    LONG = "long"


@dataclass
class FieldMeta:
    """字段元数据

    跟踪单个字段的置信度和更新历史。

    Attributes:
        confidence: 置信度 0.0~1.0，越高越可靠
        last_updated: 最近更新时间
        update_count: 累计更新次数
        source: 最近一次更新来源（rule/llm/manual）
    """

    confidence: float = 0.0
    last_updated: Optional[datetime] = None
    update_count: int = 0
    source: str = ""

    def should_update(self, tier: UpdateTier, min_confidence: float = 0.0) -> bool:
        """判断字段是否需要更新

        Args:
            tier: 字段更新层级
            min_confidence: 最低置信度阈值，低于此值需要更新

        Returns:
            是否需要更新
        """
        if self.confidence < min_confidence:
            return True
        if self.update_count == 0:
            return True
        return False

    def record_update(self, confidence: float, source: str = "rule") -> None:
        """记录一次更新

        Args:
            confidence: 本次更新置信度
            source: 更新来源
        """
        self.confidence = confidence
        self.last_updated = datetime.now()
        self.update_count += 1
        self.source = source


@dataclass
class ProfileUpdateTracker:
    """画像更新追踪器

    跟踪各层级字段的更新状态，用于控制更新频率。

    Attributes:
        summary_count_since_mid_update: 自上次中期更新以来的总结次数
        last_mid_update_time: 上次中期更新时间
        last_long_update_time: 上次长期更新时间
    """

    summary_count_since_mid_update: int = 0
    last_mid_update_time: Optional[datetime] = None
    last_long_update_time: Optional[datetime] = None

    def should_update_mid(
        self, interval_summaries: int = 5, interval_hours: float = 24.0
    ) -> bool:
        """判断是否应该进行中期更新

        Args:
            interval_summaries: 每隔多少次总结触发一次中期更新
            interval_hours: 最长间隔小时数，超过此时间也触发更新

        Returns:
            是否应该更新
        """
        if self.summary_count_since_mid_update >= interval_summaries:
            return True
        if self.last_mid_update_time is None:
            return True
        if interval_hours > 0:
            elapsed = (
                datetime.now() - self.last_mid_update_time
            ).total_seconds() / 3600
            if elapsed >= interval_hours:
                return True
        return False

    def should_update_long(self, interval_hours: float = 168.0) -> bool:
        """判断是否应该进行长期更新

        Args:
            interval_hours: 最长间隔小时数（默认7天）

        Returns:
            是否应该更新
        """
        if self.last_long_update_time is None:
            return True
        if interval_hours > 0:
            elapsed = (
                datetime.now() - self.last_long_update_time
            ).total_seconds() / 3600
            if elapsed >= interval_hours:
                return True
        return False

    def record_mid_update(self) -> None:
        """记录中期更新"""
        self.summary_count_since_mid_update = 0
        self.last_mid_update_time = datetime.now()

    def record_long_update(self) -> None:
        """记录长期更新"""
        self.last_long_update_time = datetime.now()

    def increment_summary_count(self) -> None:
        """总结次数+1"""
        self.summary_count_since_mid_update += 1


# ============================================================================
# 画像元数据混入类
# ============================================================================


class ProfileMetadataMixin:
    """画像元数据混入类

    提供字段元数据和更新追踪器的通用操作方法。
    """

    field_meta: Dict[str, Dict]
    update_tracker: Dict

    def get_update_tracker(self) -> ProfileUpdateTracker:
        """获取更新追踪器（从dict恢复）"""
        if not self.update_tracker:
            return ProfileUpdateTracker()
        tracker = ProfileUpdateTracker()
        data = self.update_tracker
        tracker.summary_count_since_mid_update = data.get(
            "summary_count_since_mid_update", 0
        )
        if data.get("last_mid_update_time"):
            if isinstance(data["last_mid_update_time"], str):
                tracker.last_mid_update_time = datetime.fromisoformat(
                    data["last_mid_update_time"]
                )
            elif isinstance(data["last_mid_update_time"], datetime):
                tracker.last_mid_update_time = data["last_mid_update_time"]
        if data.get("last_long_update_time"):
            if isinstance(data["last_long_update_time"], str):
                tracker.last_long_update_time = datetime.fromisoformat(
                    data["last_long_update_time"]
                )
            elif isinstance(data["last_long_update_time"], datetime):
                tracker.last_long_update_time = data["last_long_update_time"]
        return tracker

    def set_update_tracker(self, tracker: ProfileUpdateTracker) -> None:
        """保存更新追踪器（转为dict存储）"""
        self.update_tracker = {
            "summary_count_since_mid_update": tracker.summary_count_since_mid_update,
            "last_mid_update_time": tracker.last_mid_update_time.isoformat()
            if tracker.last_mid_update_time
            else None,
            "last_long_update_time": tracker.last_long_update_time.isoformat()
            if tracker.last_long_update_time
            else None,
        }

    def get_field_meta(self, field_name: str) -> FieldMeta:
        """获取字段元数据"""
        if field_name not in self.field_meta:
            return FieldMeta()
        data = self.field_meta[field_name]
        meta = FieldMeta()
        meta.confidence = data.get("confidence", 0.0)
        meta.update_count = data.get("update_count", 0)
        meta.source = data.get("source", "")
        if data.get("last_updated"):
            if isinstance(data["last_updated"], str):
                meta.last_updated = datetime.fromisoformat(data["last_updated"])
            elif isinstance(data["last_updated"], datetime):
                meta.last_updated = data["last_updated"]
        return meta

    def set_field_meta(self, field_name: str, meta: FieldMeta) -> None:
        """设置字段元数据"""
        self.field_meta[field_name] = {
            "confidence": meta.confidence,
            "last_updated": meta.last_updated.isoformat()
            if meta.last_updated
            else None,
            "update_count": meta.update_count,
            "source": meta.source,
        }


# ============================================================================
# 群聊画像
# ============================================================================

GROUP_FIELD_TIERS: Dict[str, UpdateTier] = {
    "interests": UpdateTier.MID,
    "atmosphere_tags": UpdateTier.MID,
    "long_term_tags": UpdateTier.LONG,
    "blacklist_topics": UpdateTier.LONG,
}


@dataclass
class GroupProfile(ProfileMetadataMixin):
    """群聊画像

    记录群聊的整体特征和行为模式。
    仅保留中长期字段。

    Attributes:
        group_id: 群聊ID
        group_name: 群聊名称
        version: 版本号（用于版本控制）

        interests: 群聊兴趣点（中期，LLM分析更新）
        atmosphere_tags: 氛围标签（中期）

        long_term_tags: 核心特征标签（长期）
        blacklist_topics: 禁忌话题（长期）

        custom_fields: 扩展字段

        field_meta: 各字段元数据（置信度、更新时间等）
        update_tracker: 更新追踪器（控制更新频率）
    """

    group_id: str
    group_name: str = ""
    version: int = 1

    interests: List[str] = field(default_factory=list)
    atmosphere_tags: List[str] = field(default_factory=list)

    long_term_tags: List[str] = field(default_factory=list)
    blacklist_topics: List[str] = field(default_factory=list)

    custom_fields: Dict[str, str] = field(default_factory=dict)

    field_meta: Dict[str, Dict] = field(default_factory=dict)
    update_tracker: Dict = field(default_factory=dict)

    FIELD_TIERS = GROUP_FIELD_TIERS


# ============================================================================
# 用户画像
# ============================================================================

USER_FIELD_TIERS: Dict[str, UpdateTier] = {
    "personality_tags": UpdateTier.MID,
    "interests": UpdateTier.MID,
    "language_style": UpdateTier.MID,
    "communication_style": UpdateTier.MID,
    "emotional_baseline": UpdateTier.MID,
    "favorability": UpdateTier.MID,
    "occupation": UpdateTier.LONG,
    "bot_relationship": UpdateTier.LONG,
    "important_dates": UpdateTier.LONG,
    "taboo_topics": UpdateTier.LONG,
    "important_events": UpdateTier.LONG,
}


# 好感度等级分段（与前端 getFavorabilityColor 保持一致）
FAVORABILITY_LEVELS: List[Tuple[float, str]] = [
    (20.0, "陌生"),
    (40.0, "认识"),
    (60.0, "熟悉"),
    (80.0, "友好"),
    (101.0, "亲密"),
]


def favorability_level(score: float) -> str:
    """根据好感度数值返回等级标签

    分段：[0,20) 陌生 / [20,40) 认识 / [40,60) 熟悉 / [60,80) 友好 / [80,100] 亲密

    Args:
        score: 好感度数值 0-100

    Returns:
        等级标签字符串
    """
    clamped = max(0.0, min(100.0, score))
    for threshold, label in FAVORABILITY_LEVELS:
        if clamped < threshold:
            return label
    return "亲密"


@dataclass
class UserProfile(ProfileMetadataMixin):
    """用户画像

    记录用户的个人特征和行为模式。
    仅保留中长期字段。

    Attributes:
        user_id: 用户ID
        user_name: 用户昵称
        version: 版本号（用于版本控制）

        historical_names: 历史曾用ID（实时更新）

        personality_tags: 性格标签（中期，LLM分析更新）
        interests: 兴趣爱好（中期）
        occupation: 职业/身份（长期）
        language_style: 常用语言风格（中期）
        communication_style: 期望的沟通偏好（中期，如简洁/详细/随意/正式）
        emotional_baseline: 情感基线（中期，如稳定/敏感/乐观/低落）
        favorability: 好感度 0-100（中期，随近期互动演化）

        bot_relationship: 对bot的称呼/关系设定（长期）
        important_dates: 重要纪念日（长期）
        taboo_topics: 禁忌话题（长期）
        important_events: 历史重要事件（长期）

        custom_fields: 扩展字段

        field_meta: 各字段元数据（置信度、更新时间等）
        update_tracker: 更新追踪器（控制更新频率）
    """

    user_id: str
    user_name: str = ""
    version: int = 1

    historical_names: List[str] = field(default_factory=list)

    personality_tags: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    occupation: str = ""
    language_style: str = ""
    communication_style: str = ""
    emotional_baseline: str = ""
    favorability: float = 0.0

    bot_relationship: str = ""
    important_dates: List[Dict[str, str]] = field(default_factory=list)
    taboo_topics: List[str] = field(default_factory=list)
    important_events: List[str] = field(default_factory=list)

    custom_fields: Dict[str, str] = field(default_factory=dict)

    field_meta: Dict[str, Dict] = field(default_factory=dict)
    update_tracker: Dict = field(default_factory=dict)

    FIELD_TIERS = USER_FIELD_TIERS


# ============================================================================
# 辅助函数
# ============================================================================


def profile_to_dict(profile: Union[GroupProfile, UserProfile]) -> dict:
    """将画像对象转换为字典（处理datetime序列化）

    Args:
        profile: 画像对象（GroupProfile 或 UserProfile）

    Returns:
        可JSON序列化的字典
    """
    data = {}
    for key, value in profile.__dict__.items():
        if key == "FIELD_TIERS":
            continue
        if isinstance(value, datetime):
            data[key] = value.isoformat()
        elif isinstance(value, UpdateTier):
            data[key] = value.value
        else:
            data[key] = value
    return data


def dict_to_group_profile(data: dict) -> GroupProfile:
    """从字典创建群聊画像对象

    兼容旧版数据：缺少 field_meta 和 update_tracker 时使用默认值。
    自动忽略已移除的字段。

    Args:
        data: 字典数据

    Returns:
        GroupProfile 对象
    """
    data.pop("FIELD_TIERS", None)

    valid_fields = {f.name for f in GroupProfile.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return GroupProfile(**filtered)


def dict_to_user_profile(data: dict) -> UserProfile:
    """从字典创建用户画像对象

    兼容旧版数据：缺少 field_meta 和 update_tracker 时使用默认值。
    自动忽略已移除的字段。

    Args:
        data: 字典数据

    Returns:
        UserProfile 对象
    """
    data.pop("FIELD_TIERS", None)

    valid_fields = {f.name for f in UserProfile.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return UserProfile(**filtered)


def merge_list_field(
    existing: List[str],
    new_values: List[str],
    max_items: int = 10,
    replace_threshold: int = 1,
) -> List[str]:
    """智能合并列表字段

    当新值数量 >= replace_threshold 时，视为 LLM 给出了完整的替换列表，
    直接用新值替换旧值（而非追加），避免列表无限膨胀。
    否则将新值追加到旧值前面（新值优先），去重后截断。

    Args:
        existing: 现有列表
        new_values: 新值列表
        max_items: 最大保留项数
        replace_threshold: 替换阈值，新值数量达到此值时替换而非追加

    Returns:
        合并后的列表
    """
    if not new_values:
        return existing

    if len(new_values) >= replace_threshold:
        seen = set()
        deduped = []
        for item in new_values:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped[:max_items]

    merged = list(new_values)
    seen = set(new_values)

    for item in existing:
        if item not in seen:
            merged.append(item)
            seen.add(item)

    return merged[:max_items]


def should_overwrite_field(
    existing_value,
    new_value,
    existing_confidence: float,
    new_confidence: float,
    min_confidence_gap: float = 0.2,
) -> bool:
    """判断是否应该用新值覆盖旧值

    置信度显著更高时才覆盖，避免频繁小幅变动。

    Args:
        existing_value: 现有值
        new_value: 新值
        existing_confidence: 现有置信度
        new_confidence: 新置信度
        min_confidence_gap: 最小置信度差距

    Returns:
        是否应该覆盖
    """
    if not existing_value:
        return True
    if not new_value:
        return False
    if existing_value == new_value:
        return False
    if new_confidence > existing_confidence + min_confidence_gap:
        return True
    if existing_confidence < 0.3:
        return True
    return False


def _find_similar_key(
    existing: Dict[str, str],
    new_key: str,
    threshold: float = 0.55,
) -> Optional[str]:
    """在已有字段中查找与新 key 语义相似的 key

    使用 SequenceMatcher 计算字符级相似度。
    同时检查子串包含关系，应对中文短语的语义重叠。

    Args:
        existing: 已有字段字典
        new_key: 新字段名
        threshold: 相似度阈值，高于此值视为相似

    Returns:
        最相似的已有 key，不存在时返回 None
    """
    if not existing or not new_key:
        return None

    best_key: Optional[str] = None
    best_score: float = threshold

    for existing_key in existing:
        if existing_key == new_key:
            return existing_key

        if new_key in existing_key or existing_key in new_key:
            shorter = min(len(new_key), len(existing_key))
            longer = max(len(new_key), len(existing_key))
            score = shorter / longer
            if score > best_score:
                best_score = score
                best_key = existing_key
                continue

        ratio = SequenceMatcher(None, new_key, existing_key).ratio()
        if ratio > best_score:
            best_score = ratio
            best_key = existing_key

    return best_key


def merge_custom_fields(
    existing: Dict[str, str],
    new_fields: Dict[str, str],
    max_fields: int = 10,
    similarity_threshold: float = 0.55,
    confidence: float = 0.7,
) -> Tuple[Dict[str, str], bool]:
    """智能合并自定义字段

    解决两个核心问题：
    1. 字段无限增长：超过 max_fields 时截断最旧的字段
    2. 高相似度字段：新 key 与已有 key 相似时合并到已有 key

    合并策略：
    - 精确匹配已有 key → 按置信度决定是否覆盖值
    - 相似匹配已有 key → 合并到已有 key（值取更详细的）
    - 无匹配 → 新增字段
    - 超过上限 → 截断最旧字段

    Args:
        existing: 已有字段字典
        new_fields: 新字段字典
        max_fields: 最大字段数量
        similarity_threshold: key 相似度阈值
        confidence: 本次更新置信度

    Returns:
        (合并后的字典, 是否有变更)
    """
    if not new_fields:
        return existing, False

    merged = dict(existing)
    changed = False

    for new_key, new_value in new_fields.items():
        if not new_key or not new_value:
            continue

        if new_key in merged:
            # 精确匹配：LLM 显式提供了该 key 的更新值，按置信度决定覆盖。
            # 此前用 existing_confidence=0.5 导致中期更新（confidence=0.7）
            # 无法刷新（0.7 > 0.5+0.2=0.7 为 False，非严格大于）。
            # 降至 0.4 使 0.7 > 0.6=True 可覆盖，0.3 > 0.6=False 仍不覆盖。
            if should_overwrite_field(merged[new_key], new_value, 0.4, confidence):
                merged[new_key] = new_value
                changed = True
            continue

        similar_key = _find_similar_key(merged, new_key, similarity_threshold)
        if similar_key is not None:
            if should_overwrite_field(merged[similar_key], new_value, 0.4, confidence):
                merged[similar_key] = new_value
                changed = True
            continue

        merged[new_key] = new_value
        changed = True

    if len(merged) > max_fields:
        keys_to_keep = list(merged.keys())[-max_fields:]
        merged = {k: merged[k] for k in keys_to_keep}
        changed = True

    return merged, changed


class ProfileConfig:
    """画像配置辅助类

    提供画像更新间隔配置的统一访问接口。
    """

    DEFAULT_MID_INTERVAL_SUMMARIES = 5
    DEFAULT_MID_INTERVAL_HOURS = 24.0
    DEFAULT_LONG_INTERVAL_HOURS = 168.0

    @classmethod
    def get_mid_update_interval_summaries(cls, config) -> int:
        """获取中期更新的总结次数间隔"""
        return config.get(
            "profile_mid_update_interval_summaries", cls.DEFAULT_MID_INTERVAL_SUMMARIES
        )

    @classmethod
    def get_mid_update_interval_hours(cls, config) -> float:
        """获取中期更新的时间间隔（小时）"""
        return config.get(
            "profile_mid_update_interval_hours", cls.DEFAULT_MID_INTERVAL_HOURS
        )

    @classmethod
    def get_long_update_interval_hours(cls, config) -> float:
        """获取长期更新的时间间隔（小时）"""
        return config.get(
            "profile_long_update_interval_hours", cls.DEFAULT_LONG_INTERVAL_HOURS
        )

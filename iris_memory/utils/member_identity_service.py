"""
成员身份服务 - MemberIdentityService

统一的成员标识管理中心，解决以下问题：
1. 维护 user_id -> 稳定昵称 的映射
2. 追踪成员名称变更历史
3. 提供统一的成员标识生成接口
4. 追踪成员活跃度
5. 提供群成员列表

所有涉及成员标识的模块（ChatHistoryBuffer、MemoryStorage、
MemberIdentityContext、TokenManager）都通过本服务获取标识，
确保一致性。
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set, Any

from iris_memory.utils.logger import get_logger
from iris_memory.utils.member_utils import short_member_id

logger = get_logger("member_identity")


@dataclass
class MemberProfile:
    """成员档案 - 维护一个 user_id 的全部身份信息"""

    user_id: str
    # 当前首选名称（最后确认的稳定名称）
    preferred_name: str = ""
    # 名称变更历史：[(old_name, new_name, timestamp), ...]
    name_history: List[Dict[str, str]] = field(default_factory=list)
    # 该用户参与的群组集合
    groups: Set[str] = field(default_factory=set)
    # 最后活跃时间
    last_active: datetime = field(default_factory=datetime.now)
    # 消息计数（粗粒度活跃度指标）
    message_count: int = 0
    # 首次出现时间
    first_seen: datetime = field(default_factory=datetime.now)

    @property
    def short_id(self) -> str:
        return short_member_id(self.user_id)

    @property
    def display_tag(self) -> str:
        """生成带短ID后缀的标识标签"""
        name = self.preferred_name or "成员"
        sid = self.short_id
        if sid:
            return f"{name}#{sid}"
        return name

    @property
    def inactive_days(self) -> float:
        """距最后活跃的天数"""
        return (datetime.now() - self.last_active).total_seconds() / 86400

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferred_name": self.preferred_name,
            "name_history": self.name_history,
            "groups": list(self.groups),
            "last_active": self.last_active.isoformat(),
            "message_count": self.message_count,
            "first_seen": self.first_seen.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemberProfile":
        last_active = data.get("last_active")
        if isinstance(last_active, str):
            last_active = datetime.fromisoformat(last_active)
        elif last_active is None:
            last_active = datetime.now()

        first_seen = data.get("first_seen")
        if isinstance(first_seen, str):
            first_seen = datetime.fromisoformat(first_seen)
        elif first_seen is None:
            first_seen = datetime.now()

        return cls(
            user_id=data.get("user_id", ""),
            preferred_name=data.get("preferred_name", ""),
            name_history=data.get("name_history", []),
            groups=set(data.get("groups", [])),
            last_active=last_active,
            message_count=data.get("message_count", 0),
            first_seen=first_seen,
        )


class MemberIdentityService:
    """成员身份服务

    通过单例接口，所有子系统可以获取一致的成员标识。

    核心能力：
    - ``resolve_tag(user_id, sender_name, group_id)``：统一标识入口
    - ``get_group_members(group_id)``：获取群成员列表标签
    - ``get_activity_score(user_id)``：活跃度 0-1 归一化
    - ``is_same_member(tag_a, tag_b)``：基于 user_id 判断是否同一人
    - 名称变更自动追踪
    """

    # 名称历史保留上限
    _MAX_NAME_HISTORY = 10
    # 活跃度半衰期（天）——超过此天数活跃度衰减到 0.5
    _ACTIVITY_HALF_LIFE_DAYS = 30.0

    def __init__(self) -> None:
        # user_id → MemberProfile
        self._profiles: Dict[str, MemberProfile] = {}
        # group_id → set of user_id
        self._group_members: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    async def resolve_tag(
        self,
        user_id: str,
        sender_name: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> str:
        """获取或更新成员标识标签（统一入口）

        如果 sender_name 与档案中不同，自动记录名称变更。

        Args:
            user_id: 用户唯一ID
            sender_name: 当前消息携带的显示名称（可为None）
            group_id: 群组ID（可为None）

        Returns:
            str: 格式为 ``名称#短ID`` 的标签
        """
        if not user_id:
            return sender_name or ""

        async with self._lock:
            profile = self._profiles.get(user_id)
            if profile is None:
                profile = MemberProfile(
                    user_id=user_id,
                    preferred_name=(sender_name or "").strip(),
                )
                self._profiles[user_id] = profile

            # 更新名称
            new_name = (sender_name or "").strip()
            if new_name and new_name != profile.preferred_name:
                self._record_name_change(profile, new_name)

            # 更新群组归属
            if group_id:
                profile.groups.add(group_id)
                self._group_members[group_id].add(user_id)

            # 更新活跃度
            profile.last_active = datetime.now()
            profile.message_count += 1

            return profile.display_tag

    def resolve_tag_sync(
        self,
        user_id: str,
        sender_name: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> str:
        """同步版本——供不方便 await 的格式化路径使用"""
        if not user_id:
            return sender_name or ""

        profile = self._profiles.get(user_id)
        if profile is None:
            profile = MemberProfile(
                user_id=user_id,
                preferred_name=(sender_name or "").strip(),
            )
            self._profiles[user_id] = profile

        new_name = (sender_name or "").strip()
        if new_name and new_name != profile.preferred_name:
            self._record_name_change(profile, new_name)

        if group_id:
            profile.groups.add(group_id)
            self._group_members[group_id].add(user_id)

        profile.last_active = datetime.now()
        # 同步版不增加 message_count，避免重复计数
        return profile.display_tag

    # ------------------------------------------------------------------
    # 群成员列表
    # ------------------------------------------------------------------

    def get_group_members(self, group_id: str) -> List[str]:
        """获取群组中所有已知成员的标签列表

        Args:
            group_id: 群组ID

        Returns:
            按最后活跃时间倒序排列的成员标签列表
        """
        member_ids = self._group_members.get(group_id, set())
        profiles = [
            self._profiles[uid]
            for uid in member_ids
            if uid in self._profiles
        ]
        # 按最后活跃时间排序
        profiles.sort(key=lambda p: p.last_active, reverse=True)
        return [p.display_tag for p in profiles]

    def get_group_member_count(self, group_id: str) -> int:
        """获取群组已知成员数量"""
        return len(self._group_members.get(group_id, set()))

    # ------------------------------------------------------------------
    # 活跃度
    # ------------------------------------------------------------------

    def get_activity_score(self, user_id: str) -> float:
        """计算成员活跃度评分（0-1）

        使用时间衰减 + 消息频率的复合指标。

        Args:
            user_id: 用户ID

        Returns:
            活跃度评分，不存在的用户返回 0
        """
        profile = self._profiles.get(user_id)
        if not profile:
            return 0.0

        # 时间衰减分量：指数衰减
        days_inactive = profile.inactive_days
        time_score = 0.5 ** (days_inactive / self._ACTIVITY_HALF_LIFE_DAYS)

        # 消息频率分量：对数归一化
        import math
        freq_score = min(1.0, math.log1p(profile.message_count) / math.log1p(100))

        # 综合（时间 70%，频率 30%）
        return 0.7 * time_score + 0.3 * freq_score

    def is_active(self, user_id: str, threshold: float = 0.2) -> bool:
        """判断成员是否处于活跃状态"""
        return self.get_activity_score(user_id) >= threshold

    # ------------------------------------------------------------------
    # 名称追踪
    # ------------------------------------------------------------------

    def get_name_history(self, user_id: str) -> List[Dict[str, str]]:
        """获取成员名称变更历史"""
        profile = self._profiles.get(user_id)
        if not profile:
            return []
        return list(profile.name_history)

    def get_all_known_names(self, user_id: str) -> List[str]:
        """获取成员所有已知名称（含当前名称）"""
        profile = self._profiles.get(user_id)
        if not profile:
            return []

        names = set()
        if profile.preferred_name:
            names.add(profile.preferred_name)
        for entry in profile.name_history:
            old = entry.get("old_name", "")
            if old:
                names.add(old)
        return list(names)

    # ------------------------------------------------------------------
    # 身份比对
    # ------------------------------------------------------------------

    def is_same_member(self, tag_a: str, tag_b: str) -> bool:
        """根据标签判断是否为同一成员

        比较规则：如果两个标签的 #短ID 后缀相同则认为同一人。
        """
        id_a = self._extract_short_id(tag_a)
        id_b = self._extract_short_id(tag_b)
        if not id_a or not id_b:
            return False
        return id_a == id_b

    def get_user_id_by_tag(self, tag: str) -> Optional[str]:
        """根据标签反查 user_id"""
        sid = self._extract_short_id(tag)
        if not sid:
            return None
        for uid, profile in self._profiles.items():
            if profile.short_id == sid:
                return uid
        return None

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """序列化为可持久化字典"""
        return {
            "profiles": {
                uid: profile.to_dict()
                for uid, profile in self._profiles.items()
            },
            "group_members": {
                gid: list(members)
                for gid, members in self._group_members.items()
            },
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """从持久化字典恢复状态"""
        if not data:
            return

        profiles_data = data.get("profiles", {})
        for uid, profile_dict in profiles_data.items():
            self._profiles[uid] = MemberProfile.from_dict(profile_dict)

        group_data = data.get("group_members", {})
        for gid, members in group_data.items():
            self._group_members[gid] = set(members)

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def cleanup_inactive(self, inactive_days: float = 180) -> int:
        """清理长期不活跃成员档案

        Args:
            inactive_days: 超过此天数认为不活跃

        Returns:
            清理的成员数量
        """
        threshold = datetime.now() - timedelta(days=inactive_days)
        to_remove = [
            uid
            for uid, p in self._profiles.items()
            if p.last_active < threshold
        ]

        for uid in to_remove:
            profile = self._profiles.pop(uid, None)
            if profile:
                for gid in profile.groups:
                    self._group_members.get(gid, set()).discard(uid)

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            "total_profiles": len(self._profiles),
            "total_groups": len(self._group_members),
            "active_members": sum(
                1 for p in self._profiles.values() if self.is_active(p.user_id)
            ),
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _record_name_change(self, profile: MemberProfile, new_name: str) -> None:
        """记录名称变更"""
        old_name = profile.preferred_name
        if old_name == new_name:
            return

        if old_name:  # 首次设置不记录到历史
            profile.name_history.append({
                "old_name": old_name,
                "new_name": new_name,
                "timestamp": datetime.now().isoformat(),
            })

            # 保留上限
            if len(profile.name_history) > self._MAX_NAME_HISTORY:
                profile.name_history = profile.name_history[-self._MAX_NAME_HISTORY:]

            logger.info(
                f"Member name changed: user={profile.user_id}, "
                f"'{old_name}' -> '{new_name}'"
            )

        profile.preferred_name = new_name

    @staticmethod
    def _extract_short_id(tag: str) -> str:
        """从标签中提取 #短ID 后缀"""
        if "#" in tag:
            return tag.rsplit("#", 1)[-1]
        return ""

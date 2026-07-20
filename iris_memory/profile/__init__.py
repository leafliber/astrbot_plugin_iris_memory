"""
Iris Chat Memory - 画像系统模块

提供群聊画像和用户画像的存储、管理和分析功能。
支持三层更新频率策略和智能合并。
"""

from .models import (
    GroupProfile,
    UserProfile,
    FieldMeta,
    ProfileUpdateTracker,
    UpdateTier,
    profile_to_dict,
    dict_to_group_profile,
    dict_to_user_profile,
    merge_list_field,
    should_overwrite_field,
    ProfileConfig,
)
from .storage import ProfileStorage
from .group_profile import GroupProfileManager
from .user_profile import UserProfileManager
from .analyzer import ProfileAnalyzer

__all__ = [
    # 数据模型
    "GroupProfile",
    "UserProfile",
    "FieldMeta",
    "ProfileUpdateTracker",
    "UpdateTier",
    "profile_to_dict",
    "dict_to_group_profile",
    "dict_to_user_profile",
    "merge_list_field",
    "should_overwrite_field",
    "ProfileConfig",
    # 存储组件
    "ProfileStorage",
    # 管理器
    "GroupProfileManager",
    "UserProfileManager",
    # 分析器
    "ProfileAnalyzer",
]

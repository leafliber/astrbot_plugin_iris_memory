"""画像数据模型测试"""

from iris_memory.profile.models import (
    GroupProfile,
    UserProfile,
    profile_to_dict,
    dict_to_group_profile,
    dict_to_user_profile,
    merge_custom_fields,
    merge_list_field,
    _find_similar_key,
    favorability_level,
    FAVORABILITY_LEVELS,
)


class TestGroupProfile:
    """群聊画像数据模型测试"""

    def test_create_group_profile(self):
        """测试创建群聊画像"""
        profile = GroupProfile(group_id="group_123")

        assert profile.group_id == "group_123"
        assert profile.group_name == ""
        assert profile.version == 1
        assert profile.interests == []
        assert profile.atmosphere_tags == []

    def test_group_profile_with_data(self):
        """测试带数据的群聊画像"""
        profile = GroupProfile(
            group_id="group_123",
            group_name="技术交流群",
            interests=["技术", "AI"],
            atmosphere_tags=["轻松", "技术范"],
        )

        assert profile.group_name == "技术交流群"
        assert profile.interests == ["技术", "AI"]
        assert profile.atmosphere_tags == ["轻松", "技术范"]

    def test_profile_to_dict(self):
        """测试画像转字典"""
        profile = GroupProfile(group_id="group_123", group_name="测试群")

        data = profile_to_dict(profile)

        assert data["group_id"] == "group_123"
        assert data["group_name"] == "测试群"

    def test_dict_to_group_profile(self):
        """测试字典转群聊画像"""
        data = {
            "group_id": "group_123",
            "group_name": "测试群",
            "version": 2,
            "interests": ["技术"],
        }

        profile = dict_to_group_profile(data)

        assert profile.group_id == "group_123"
        assert profile.group_name == "测试群"
        assert profile.version == 2
        assert profile.interests == ["技术"]


class TestUserProfile:
    """用户画像数据模型测试"""

    def test_create_user_profile(self):
        """测试创建用户画像"""
        profile = UserProfile(user_id="user_456")

        assert profile.user_id == "user_456"
        assert profile.user_name == ""
        assert profile.version == 1
        assert profile.personality_tags == []
        assert profile.interests == []

    def test_user_profile_with_data(self):
        """测试带数据的用户画像"""
        profile = UserProfile(
            user_id="user_456",
            user_name="小明",
            personality_tags=["外向", "幽默"],
            interests=["编程", "游戏"],
        )

        assert profile.user_name == "小明"
        assert profile.personality_tags == ["外向", "幽默"]
        assert profile.interests == ["编程", "游戏"]

    def test_user_profile_to_dict(self):
        """测试用户画像转字典"""
        profile = UserProfile(user_id="user_456", user_name="小明")

        data = profile_to_dict(profile)

        assert data["user_id"] == "user_456"
        assert data["user_name"] == "小明"

    def test_dict_to_user_profile(self):
        """测试字典转用户画像"""
        data = {
            "user_id": "user_456",
            "user_name": "小明",
            "version": 3,
            "personality_tags": ["外向"],
        }

        profile = dict_to_user_profile(data)

        assert profile.user_id == "user_456"
        assert profile.user_name == "小明"
        assert profile.version == 3
        assert profile.personality_tags == ["外向"]


class TestFindSimilarKey:
    """_find_similar_key 测试"""

    def test_exact_match(self):
        result = _find_similar_key({"家乡": "北京"}, "家乡")
        assert result == "家乡"

    def test_similar_chinese_keys(self):
        result = _find_similar_key({"喜欢的食物": "火锅"}, "爱吃的食物")
        assert result == "喜欢的食物"

    def test_similar_english_keys(self):
        result = _find_similar_key({"favorite_food": "hotpot"}, "favorite_foods")
        assert result == "favorite_food"

    def test_substring_containment(self):
        result = _find_similar_key({"活跃时段": "晚上"}, "活跃时间")
        assert result == "活跃时段"

    def test_no_similar_key(self):
        result = _find_similar_key({"家乡": "北京"}, "宠物")
        assert result is None

    def test_empty_existing(self):
        result = _find_similar_key({}, "家乡")
        assert result is None

    def test_empty_new_key(self):
        result = _find_similar_key({"家乡": "北京"}, "")
        assert result is None

    def test_returns_best_match(self):
        existing = {"食物": "米饭", "喜欢的食物": "火锅"}
        result = _find_similar_key(existing, "爱吃的食物")
        assert result == "喜欢的食物"


class TestMergeCustomFields:
    """merge_custom_fields 测试"""

    def test_add_new_fields(self):
        existing = {"家乡": "北京"}
        new_fields = {"宠物": "猫"}
        merged, changed = merge_custom_fields(existing, new_fields)
        assert changed is True
        assert merged["家乡"] == "北京"
        assert merged["宠物"] == "猫"

    def test_no_change_when_empty_new(self):
        existing = {"家乡": "北京"}
        merged, changed = merge_custom_fields(existing, {})
        assert changed is False
        assert merged == existing

    def test_overwrite_existing_key_with_high_confidence(self):
        existing = {"家乡": "北京"}
        new_fields = {"家乡": "上海"}
        merged, changed = merge_custom_fields(existing, new_fields, confidence=0.9)
        assert changed is True
        assert merged["家乡"] == "上海"

    def test_no_overwrite_with_low_confidence(self):
        existing = {"家乡": "北京"}
        new_fields = {"家乡": "上海"}
        merged, changed = merge_custom_fields(existing, new_fields, confidence=0.3)
        assert changed is False
        assert merged["家乡"] == "北京"

    def test_overwrite_existing_key_with_mid_confidence(self):
        """回归：confidence=0.7 应能覆盖已有字段值

        此前 existing_confidence=0.5 时，should_overwrite_field 判定为
        0.7 > 0.5+0.2=0.7 为 False（非严格大于），导致中期更新无法刷新字段。
        修复后将 existing_confidence 降至 0.4，0.7 > 0.6=True 可覆盖。
        """
        existing = {"key": "old"}
        new_fields = {"key": "new"}
        merged, changed = merge_custom_fields(existing, new_fields, confidence=0.7)
        assert changed is True
        assert merged["key"] == "new"

    def test_merge_similar_key(self):
        existing = {"喜欢的食物": "火锅"}
        new_fields = {"爱吃的食物": "烤肉"}
        merged, changed = merge_custom_fields(existing, new_fields, confidence=0.9)
        assert changed is True
        assert "喜欢的食物" in merged
        assert "爱吃的食物" not in merged
        assert merged["喜欢的食物"] == "烤肉"

    def test_max_fields_limit(self):
        existing = {
            "家乡": "北京",
            "宠物": "猫",
            "职业": "工程师",
            "年龄": "25",
            "身高": "175",
        }
        new_fields = {"爱好": "游泳", "学历": "本科", "血型": "A", "星座": "天秤"}
        merged, changed = merge_custom_fields(existing, new_fields, max_fields=7)
        assert len(merged) == 7
        assert changed is True

    def test_skip_empty_key_or_value(self):
        existing = {"家乡": "北京"}
        new_fields = {"": "值", "宠物": ""}
        merged, changed = merge_custom_fields(existing, new_fields)
        assert changed is False
        assert len(merged) == 1

    def test_preserves_existing_when_no_new(self):
        existing = {"家乡": "北京", "宠物": "猫"}
        merged, changed = merge_custom_fields(existing, None)
        assert changed is False
        assert merged == existing

    def test_trimming_keeps_latest(self):
        existing = {
            "家乡": "北京",
            "宠物": "猫",
            "职业": "工程师",
            "年龄": "25",
            "身高": "175",
            "学历": "本科",
            "血型": "A",
            "星座": "天秤",
            "爱好": "游泳",
            "特长": "钢琴",
        }
        new_fields = {"方言": "粤语"}
        merged, changed = merge_custom_fields(existing, new_fields, max_fields=10)
        assert len(merged) == 10
        assert "方言" in merged
        assert "家乡" not in merged


class TestMergeListField:
    """merge_list_field 测试"""

    def test_replaces_when_new_values_above_default_threshold(self):
        """回归：默认 replace_threshold=1，任何非空新值列表都视为完整替换

        此前默认阈值为 5，导致少于 5 项的新列表会被追加而非替换，
        造成列表字段无限膨胀。修复后将默认阈值降至 1。
        """
        result = merge_list_field(["old1", "old2", "old3"], ["new1"])
        assert result == ["new1"]


class TestFavorabilityLevel:
    """favorability_level 等级分段测试"""

    def test_level_boundaries(self):
        """验证 5 级分段边界"""
        assert favorability_level(0) == "陌生"
        assert favorability_level(19.9) == "陌生"
        assert favorability_level(20) == "认识"
        assert favorability_level(39.9) == "认识"
        assert favorability_level(40) == "熟悉"
        assert favorability_level(59.9) == "熟悉"
        assert favorability_level(60) == "友好"
        assert favorability_level(79.9) == "友好"
        assert favorability_level(80) == "亲密"
        assert favorability_level(100) == "亲密"

    def test_clamps_out_of_range(self):
        """超出范围的值被夹紧到 [0, 100]"""
        assert favorability_level(-10) == "陌生"
        assert favorability_level(150) == "亲密"

    def test_user_profile_has_favorability_field(self):
        """UserProfile 默认 favorability=0.0"""
        profile = UserProfile(user_id="u1")
        assert profile.favorability == 0.0

    def test_favorability_in_field_tiers_is_mid(self):
        """favorability 字段层级为 MID"""
        from iris_memory.profile.models import USER_FIELD_TIERS, UpdateTier

        assert USER_FIELD_TIERS.get("favorability") == UpdateTier.MID

    def test_favorability_levels_constant(self):
        """FAVORABILITY_LEVELS 常量有 5 个分段"""
        assert len(FAVORABILITY_LEVELS) == 5
        labels = [label for _, label in FAVORABILITY_LEVELS]
        assert labels == ["陌生", "认识", "熟悉", "友好", "亲密"]

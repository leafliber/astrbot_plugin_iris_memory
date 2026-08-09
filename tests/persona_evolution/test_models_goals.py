"""人格自迭代数据模型与目标预设测试"""

from iris_memory.persona_evolution import (
    ApprovalMode,
    EditMode,
    ErrorCode,
    JobStatus,
    RevisionStatus,
    RunStatus,
    TriggerType,
)
from iris_memory.persona_evolution.goals import (
    GOAL_PRESET_VERSION,
    GOAL_PRESETS,
    build_goal_snapshot,
    get_goal_preset,
)


class TestEnums:
    """状态枚举完整覆盖后续阶段所需取值"""

    def test_job_status(self):
        assert {s.value for s in JobStatus} == {
            "active",
            "paused",
            "conflict",
            "paused_error",
        }

    def test_edit_mode(self):
        assert {s.value for s in EditMode} == {"managed_block", "full_prompt"}

    def test_approval_mode(self):
        assert {s.value for s in ApprovalMode} == {"auto", "manual"}

    def test_revision_status(self):
        assert {s.value for s in RevisionStatus} == {
            "candidate",
            "publishing",
            "applied",
            "rejected",
            "failed_validation",
            "publish_failed",
            "external_change",
            "rollback",
            "no_change",
        }

    def test_trigger_type(self):
        assert {t.value for s in TriggerType for t in [s]} == {
            "auto",
            "manual",
            "rollback",
        }

    def test_run_status(self):
        assert {s.value for s in RunStatus} == {"running", "success", "failed"}

    def test_error_code_stable_strings(self):
        """错误码为机器可读字符串，供 Web API 使用"""
        assert ErrorCode.BASE_HASH_MISMATCH.value == "base_hash_mismatch"
        assert ErrorCode.BLOCK_OUTSIDE_MODIFIED.value == "block_outside_modified"
        assert ErrorCode.INSUFFICIENT_SAMPLES.value == "insufficient_samples"
        assert ErrorCode.CIRCUIT_OPEN.value == "circuit_open"
        # 文档 §10 的 12 条闸门错误码齐备
        gate_codes = {
            "invalid_json",
            "empty_candidate",
            "persona_mismatch",
            "base_hash_mismatch",
            "block_outside_modified",
            "marker_invalid",
            "length_exceeded",
            "protected_fragment_missing",
            "privacy_leak",
            "corpus_reuse",
            "forbidden_field_modified",
            "no_change",
        }
        assert gate_codes <= {e.value for e in ErrorCode}


class TestGoalPresets:
    """文档 §5.2 的 8 个目标预设"""

    def test_eight_presets(self):
        assert set(GOAL_PRESETS) == {
            "natural",
            "warm",
            "concise",
            "humorous",
            "professional",
            "proactive",
            "group_style",
            "custom",
        }

    def test_presets_have_display_name_and_text(self):
        for preset_id, preset in GOAL_PRESETS.items():
            assert preset.display_name
            if preset_id != "custom":
                assert preset.text  # custom 文本由管理员提供

    def test_get_goal_preset(self):
        assert get_goal_preset("warm").display_name == "温暖共情"
        assert get_goal_preset("missing") is None

    def test_snapshot_contains_version_and_full_text(self):
        snapshot = build_goal_snapshot("concise")
        assert snapshot["preset_id"] == "concise"
        assert snapshot["preset_version"] == GOAL_PRESET_VERSION
        assert snapshot["text"] == GOAL_PRESETS["concise"].text
        assert snapshot["display_name"] == "简洁直接"

    def test_snapshot_custom_uses_custom_goal(self):
        snapshot = build_goal_snapshot("custom", custom_goal="  多聊技术话题  ")
        assert snapshot["preset_id"] == "custom"
        assert snapshot["text"] == "多聊技术话题"
        assert snapshot["custom_goal"] == "多聊技术话题"

    def test_snapshot_unknown_preset_degrades_to_custom(self):
        snapshot = build_goal_snapshot("no_such_preset", custom_goal="目标X")
        assert snapshot["preset_id"] == "custom"
        assert snapshot["text"] == "目标X"

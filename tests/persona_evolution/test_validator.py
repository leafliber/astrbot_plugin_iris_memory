"""确定性发布闸门测试（文档 §10 全部 12 条，逐条正反用例）"""

from iris_memory.persona_evolution.models import EditMode, ErrorCode, EvolutionJob
from iris_memory.persona_evolution.publisher import (
    MANAGED_BLOCK_BEGIN,
    MANAGED_BLOCK_END,
    append_managed_block,
    persona_hash,
    split_managed_block,
)
from iris_memory.persona_evolution.validator import validate_candidate

BASE = f"你是 Iris，一个群聊助手。\n\n{MANAGED_BLOCK_BEGIN}\n旧风格\n{MANAGED_BLOCK_END}\n"
CANDIDATE = f"你是 Iris，一个群聊助手。\n\n{MANAGED_BLOCK_BEGIN}\n新风格：短句为主\n{MANAGED_BLOCK_END}\n"


def _job(**overrides) -> EvolutionJob:
    fields = {"persona_id": "p1"}
    fields.update(overrides)
    return EvolutionJob(**fields)


def _validate(candidate=CANDIDATE, job=None, base=BASE, current=None, base_hash=None, **kw):
    current = BASE if current is None else current
    if base_hash is None:
        base_hash = persona_hash(current)
    return validate_candidate(
        candidate_prompt=candidate,
        job=job or _job(),
        persona_id="p1",
        base_prompt=base,
        base_hash=base_hash,
        current_prompt=current,
        **kw,
    )


def _codes(outcome):
    return {f.code for f in outcome.failures}


class TestHappyPath:
    def test_managed_block_passes(self):
        outcome = _validate()
        assert outcome.passed, outcome.to_snapshot()
        assert outcome.no_change is False
        assert "forbidden_fields_by_convention" in outcome.checks


class TestRule1JsonSchema:
    def test_non_string_candidate(self):
        outcome = _validate(candidate={"prompt": "x"})
        assert ErrorCode.INVALID_JSON in _codes(outcome)
        assert outcome.passed is False


class TestRule2CandidateSanity:
    def test_empty(self):
        assert ErrorCode.EMPTY_CANDIDATE in _codes(_validate(candidate=""))

    def test_whitespace_only(self):
        assert ErrorCode.EMPTY_CANDIDATE in _codes(_validate(candidate="  \n\t "))

    def test_nul_char(self):
        assert ErrorCode.EMPTY_CANDIDATE in _codes(
            _validate(candidate=CANDIDATE + "\x00")
        )

    def test_abnormal_control_char(self):
        assert ErrorCode.EMPTY_CANDIDATE in _codes(
            _validate(candidate=CANDIDATE + "\x07")
        )

    def test_newline_tab_allowed(self):
        outcome = _validate(candidate=CANDIDATE + "\n\t追加")
        assert ErrorCode.EMPTY_CANDIDATE not in _codes(outcome)


class TestRule3PersonaMismatch:
    def test_mismatch(self):
        outcome = _validate(job=_job(persona_id="p2"))
        assert ErrorCode.PERSONA_MISMATCH in _codes(outcome)

    def test_match(self):
        outcome = _validate()
        assert ErrorCode.PERSONA_MISMATCH not in _codes(outcome)


class TestRule4BaseHash:
    def test_hash_mismatch(self):
        # 当前 persona 已被改动，与生成基线不一致
        outcome = _validate(current=BASE + "外部改动", base_hash=persona_hash(BASE))
        assert ErrorCode.BASE_HASH_MISMATCH in _codes(outcome)

    def test_hash_match(self):
        outcome = _validate()
        assert ErrorCode.BASE_HASH_MISMATCH not in _codes(outcome)


class TestRule5BlockOutside:
    def test_outside_modified(self):
        candidate = CANDIDATE.replace("你是 Iris，一个群聊助手。", "你是 Moca。")
        assert ErrorCode.BLOCK_OUTSIDE_MODIFIED in _codes(_validate(candidate=candidate))

    def test_outside_whitespace_modified(self):
        # 区块外哪怕一个换行不同也算修改（逐字节相同）
        candidate = CANDIDATE.replace("助手。\n\n", "助手。\n\n\n")
        assert ErrorCode.BLOCK_OUTSIDE_MODIFIED in _codes(_validate(candidate=candidate))

    def test_outside_unchanged_passes(self):
        assert ErrorCode.BLOCK_OUTSIDE_MODIFIED not in _codes(_validate())


class TestRule6Markers:
    def test_duplicate_begin(self):
        candidate = CANDIDATE + f"\n{MANAGED_BLOCK_BEGIN}\nx\n{MANAGED_BLOCK_END}\n"
        assert ErrorCode.MARKER_INVALID in _codes(_validate(candidate=candidate))

    def test_missing_end(self):
        candidate = CANDIDATE.replace(MANAGED_BLOCK_END, "")
        assert ErrorCode.MARKER_INVALID in _codes(_validate(candidate=candidate))

    def test_wrong_order(self):
        candidate = (
            "你是 Iris，一个群聊助手。\n\n"
            f"{MANAGED_BLOCK_END}\n新风格\n{MANAGED_BLOCK_BEGIN}\n"
        )
        assert ErrorCode.MARKER_INVALID in _codes(_validate(candidate=candidate))

    def test_nested(self):
        inner = f"新{MANAGED_BLOCK_BEGIN}嵌套{MANAGED_BLOCK_END}风格"
        candidate = (
            f"你是 Iris，一个群聊助手。\n\n{MANAGED_BLOCK_BEGIN}\n"
            f"{inner}\n{MANAGED_BLOCK_END}\n"
        )
        assert ErrorCode.MARKER_INVALID in _codes(_validate(candidate=candidate))

    def test_full_mode_allows_absent_markers(self):
        base = "你是 Iris，一个群聊助手，没有标记。"
        candidate = "你是 Iris，一个群聊助手，没有标记。更简洁。"
        outcome = _validate(
            candidate=candidate,
            base=base,
            current=base,
            job=_job(edit_mode=EditMode.FULL_PROMPT.value),
        )
        assert ErrorCode.MARKER_INVALID not in _codes(outcome)

    def test_full_mode_invalid_markers_rejected(self):
        base = "你是 Iris，一个群聊助手。"
        candidate = base + f"\n{MANAGED_BLOCK_BEGIN}\nx\n"
        outcome = _validate(
            candidate=candidate,
            base=base,
            current=base,
            job=_job(edit_mode=EditMode.FULL_PROMPT.value),
        )
        assert ErrorCode.MARKER_INVALID in _codes(outcome)


class TestRule7LengthAndChangeRatio:
    def test_block_too_long(self):
        inner = "长" * 1600
        candidate = (
            f"你是 Iris，一个群聊助手。\n\n{MANAGED_BLOCK_BEGIN}\n"
            f"{inner}\n{MANAGED_BLOCK_END}\n"
        )
        assert ErrorCode.LENGTH_EXCEEDED in _codes(_validate(candidate=candidate))

    def test_block_within_limit(self):
        inner = "中" * 1490  # 含两侧换行后区块内容 1492 字符，低于 1500 上限
        candidate = (
            f"你是 Iris，一个群聊助手。\n\n{MANAGED_BLOCK_BEGIN}\n"
            f"{inner}\n{MANAGED_BLOCK_END}\n"
        )
        assert ErrorCode.LENGTH_EXCEEDED not in _codes(_validate(candidate=candidate))

    def test_full_growth_exceeded(self):
        base = "短" * 100
        candidate = base + "增" * 100  # 200 > 100*1.25
        outcome = _validate(
            candidate=candidate,
            base=base,
            current=base,
            job=_job(edit_mode=EditMode.FULL_PROMPT.value),
        )
        assert ErrorCode.LENGTH_EXCEEDED in _codes(outcome)

    def test_full_absolute_limit(self):
        base = "基" * 19000
        candidate = base + "增" * 1500  # 20500 > 20000，但 < 19000*1.25
        outcome = _validate(
            candidate=candidate,
            base=base,
            current=base,
            job=_job(edit_mode=EditMode.FULL_PROMPT.value),
            full_max_change_ratio=0.40,
        )
        assert ErrorCode.LENGTH_EXCEEDED in _codes(outcome)

    def test_full_change_ratio_exceeded(self):
        base = "AAAAAAAAAAAAAAAAAAAA"  # 20 字符
        candidate = "BBBBBBBBBBBBBBBBBBBB"  # 完全不同 → 改动率 ~1.0
        outcome = _validate(
            candidate=candidate,
            base=base,
            current=base,
            job=_job(edit_mode=EditMode.FULL_PROMPT.value),
        )
        assert ErrorCode.LENGTH_EXCEEDED in _codes(outcome)

    def test_change_ratio_config_clamped_to_040(self):
        # 配置 0.9 也应被钳制到 0.40 上限
        base = "A" * 100
        candidate = "A" * 40 + "B" * 60  # 改动率 ~0.6
        outcome = _validate(
            candidate=candidate,
            base=base,
            current=base,
            job=_job(edit_mode=EditMode.FULL_PROMPT.value),
            full_max_change_ratio=0.9,
        )
        assert ErrorCode.LENGTH_EXCEEDED in _codes(outcome)

    def test_full_small_change_passes(self):
        base = "你是 Iris，一个群聊助手，回答简洁直接。" * 5
        candidate = base.replace("简洁", "更简洁", 1)
        outcome = _validate(
            candidate=candidate,
            base=base,
            current=base,
            job=_job(edit_mode=EditMode.FULL_PROMPT.value),
        )
        assert ErrorCode.LENGTH_EXCEEDED not in _codes(outcome)


class TestRule8ProtectedFragments:
    def test_fragment_missing(self):
        job = _job(protected_fragments=["绝不能删除的设定"])
        assert ErrorCode.PROTECTED_FRAGMENT_MISSING in _codes(_validate(job=job))

    def test_fragment_present(self):
        base = BASE.replace("你是 Iris", "绝不能删除的设定\n你是 Iris")
        candidate = CANDIDATE.replace("你是 Iris", "绝不能删除的设定\n你是 Iris")
        job = _job(protected_fragments=["绝不能删除的设定"])
        outcome = _validate(candidate=candidate, job=job, base=base, current=base)
        assert ErrorCode.PROTECTED_FRAGMENT_MISSING not in _codes(outcome)


class TestRule9PrivacyLeak:
    def test_group_id_leak(self):
        job = _job(source_group_ids=["12345678"])
        candidate = CANDIDATE.replace("新风格", "来自12345678群的新风格")
        assert ErrorCode.PRIVACY_LEAK in _codes(_validate(candidate=candidate, job=job))

    def test_user_id_leak(self):
        job = _job(source_user_ids=["987654321"])
        candidate = CANDIDATE.replace("新风格", "学习987654321的新风格")
        assert ErrorCode.PRIVACY_LEAK in _codes(_validate(candidate=candidate, job=job))

    def test_user_name_leak(self):
        candidate = CANDIDATE.replace("新风格", "像张三那样说话")
        assert ErrorCode.PRIVACY_LEAK in _codes(
            _validate(candidate=candidate, known_user_names=["张三"])
        )

    def test_no_leak(self):
        outcome = _validate(
            job=_job(source_group_ids=["12345678"], source_user_ids=["987654321"]),
            known_user_names=["张三"],
        )
        assert ErrorCode.PRIVACY_LEAK not in _codes(outcome)


class TestRule10CorpusReuse:
    def test_reuse_over_limit(self):
        corpus = "今晚大家一起去吃火锅吧，我请客你们随意"  # 19 字符
        candidate = CANDIDATE.replace("新风格：短句为主", corpus)
        assert ErrorCode.CORPUS_REUSE in _codes(
            _validate(candidate=candidate, corpus_texts=[corpus])
        )

    def test_reuse_under_limit(self):
        corpus = "今晚大家一起去吃火锅吧，我请客你们随意"  # 19 字符语料
        candidate = CANDIDATE.replace("新风格：短句为主", "今晚大家一起")  # 仅 6 字符重合
        outcome = _validate(candidate=candidate, corpus_texts=[corpus])
        assert ErrorCode.CORPUS_REUSE not in _codes(outcome)

    def test_exactly_at_limit_rejected(self):
        corpus = "今" * 16
        candidate = CANDIDATE.replace("新风格：短句为主", corpus)
        assert ErrorCode.CORPUS_REUSE in _codes(
            _validate(candidate=candidate, corpus_texts=[corpus], max_reuse_chars=16)
        )


class TestRule11ForbiddenFields:
    def test_convention_check_recorded(self):
        # 由发布调用方式保证（只传 persona_id+system_prompt），此处记录约定
        outcome = _validate()
        assert "forbidden_fields_by_convention" in outcome.checks


class TestRule12NoChange:
    def test_same_hash_no_change(self):
        outcome = _validate(candidate=BASE)
        assert outcome.no_change is True
        assert outcome.passed is False
        assert not outcome.failures

    def test_different_hash_publishes(self):
        outcome = _validate()
        assert outcome.no_change is False


class TestFirstRunAppendedBlock:
    def test_appended_base_passes(self):
        raw = "你是 Iris，一个群聊助手。"
        effective_base = append_managed_block(raw)
        candidate = effective_base.replace(
            "[自动迭代生成的表达风格、语气、长度、节奏与互动习惯]", "新风格"
        )
        outcome = _validate(candidate=candidate, base=effective_base, current=raw)
        assert outcome.passed, outcome.to_snapshot()


class TestSplitManagedBlock:
    def test_ok(self):
        split = split_managed_block(BASE)
        assert split.status == "ok"
        assert "旧风格" in split.inner

    def test_absent(self):
        assert split_managed_block("没有任何标记").status == "absent"

    def test_duplicate(self):
        assert split_managed_block(BASE + BASE).status == "invalid"

    def test_single_side(self):
        assert split_managed_block(MANAGED_BLOCK_BEGIN + "只有头").status == "invalid"
        assert split_managed_block("只有尾" + MANAGED_BLOCK_END).status == "invalid"

    def test_nested(self):
        nested = (
            f"{MANAGED_BLOCK_BEGIN}\n外{MANAGED_BLOCK_BEGIN}内"
            f"{MANAGED_BLOCK_END}\n{MANAGED_BLOCK_END}"
        )
        assert split_managed_block(nested).status == "invalid"

    def test_wrong_order(self):
        wrong = f"{MANAGED_BLOCK_END}\nx\n{MANAGED_BLOCK_BEGIN}"
        assert split_managed_block(wrong).status == "invalid"

    def test_append_block_idempotent_shape(self):
        appended = append_managed_block("原文")
        split = split_managed_block(appended)
        assert split.status == "ok"
        assert appended.startswith("原文")

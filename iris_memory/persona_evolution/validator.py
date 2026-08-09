"""
Iris Chat Memory - 人格自迭代确定性发布闸门

文档 §10 全部 12 条，纯代码校验、不调用 LLM，逐条返回机器可读
ErrorCode（models.py）。无论自动或手动模式，发布前必须全部通过。
"""

import difflib
from dataclasses import dataclass, field
from typing import Any, List, Optional

from iris_memory.core import get_logger
from .models import EditMode, ErrorCode, EvolutionJob
from .publisher import (
    MARKERS_ABSENT,
    MARKERS_OK,
    persona_hash,
    split_managed_block,
)

logger = get_logger("persona_evolution.validator")

# 允许出现的控制字符（换行/回车/制表符之外的 C0/C1 控制字符一律拒绝）
_ALLOWED_CONTROL = {"\n", "\r", "\t"}

# full 模式改动率配置的硬上限（文档 §6.2：可配置但不得超过 40%）
MAX_CHANGE_RATIO_CAP = 0.40

# 隐私泄漏检查中用户名的最小长度（单字用户名误报率过高，不参与检查）
_MIN_NAME_LEAK_CHARS = 2


@dataclass
class GateFailure:
    """单条闸门失败"""

    code: ErrorCode
    message: str


@dataclass
class ValidationOutcome:
    """闸门校验结果

    Attributes:
        passed: 全部通过（no_change 不算通过，但也不算失败）
        no_change: 候选与当前哈希一致（规则 12），不发布记 no_change
        failures: 失败明细（机器可读错误码 + 人类可读原因）
        checks: 已执行的检查项清单（Revision validation 快照用）
    """

    passed: bool
    no_change: bool = False
    failures: List[GateFailure] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)

    def to_snapshot(self) -> dict:
        """序列化为 Revision.validation 快照"""
        return {
            "passed": self.passed,
            "no_change": self.no_change,
            "checks": self.checks,
            "failures": [
                {"code": f.code.value, "message": f.message} for f in self.failures
            ],
        }


def _has_abnormal_control_chars(text: str) -> bool:
    for ch in text:
        if ch in _ALLOWED_CONTROL:
            continue
        code = ord(ch)
        if code < 0x20 or code == 0x7F or 0x80 <= code <= 0x9F:
            return True
    return False


def _longest_common_substring_len(a: str, b: str) -> int:
    """两字符串最长公共连续子串长度（difflib 实现）"""
    if not a or not b:
        return 0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    ).size


def validate_candidate(
    *,
    candidate_prompt: Any,
    job: EvolutionJob,
    persona_id: str,
    base_prompt: str,
    base_hash: str,
    current_prompt: str,
    corpus_texts: Optional[List[str]] = None,
    known_user_names: Optional[List[str]] = None,
    block_max_chars: int = 1500,
    full_max_change_ratio: float = 0.20,
    full_max_growth_ratio: float = 1.25,
    full_max_length: int = 20000,
    max_reuse_chars: int = 16,
) -> ValidationOutcome:
    """执行文档 §10 的 12 条确定性发布闸门

    Args:
        candidate_prompt: 阶段 B 原始输出（规则 1 校验其类型）
        job: 迭代 Job（persona_id / edit_mode / protected_fragments /
            source_group_ids / source_user_ids）
        persona_id: 本次生成针对的 Persona ID（规则 3）
        base_prompt: 生成时使用的 system_prompt（受控模式为已补标记的有效基线）
        base_hash: 生成时记录的真实 Persona 哈希（规则 4）
        current_prompt: 校验时重读的当前 system_prompt
        corpus_texts: 本轮使用的语料文本（规则 10）
        known_user_names: 语料范围内已知用户名（规则 9）
        其余: 隐藏高级参数（文档 §15.2）

    Returns:
        ValidationOutcome（failures 含全部未通过项，不短路）
    """
    failures: List[GateFailure] = []
    checks: List[str] = []
    no_change = False

    # 1. JSON schema / 字段类型 / 枚举合法（生成输出类型校验）
    checks.append("json_schema")
    if not isinstance(candidate_prompt, str):
        failures.append(
            GateFailure(ErrorCode.INVALID_JSON, "candidate_prompt 不是字符串")
        )
        return ValidationOutcome(
            passed=False, failures=failures, checks=checks
        )

    # 2. 候选非空、无 NUL、无异常控制字符
    checks.append("candidate_sanity")
    if not candidate_prompt.strip():
        failures.append(GateFailure(ErrorCode.EMPTY_CANDIDATE, "候选为空"))
    elif _has_abnormal_control_chars(candidate_prompt):
        failures.append(
            GateFailure(ErrorCode.EMPTY_CANDIDATE, "候选含 NUL 或异常控制字符")
        )

    # 3. Persona ID 与 Job 一致
    checks.append("persona_match")
    if persona_id != job.persona_id:
        failures.append(
            GateFailure(
                ErrorCode.PERSONA_MISMATCH,
                f"目标 Persona {persona_id} 与 Job {job.persona_id} 不一致",
            )
        )

    # 4. 当前 Persona 哈希等于生成时的 base_hash
    checks.append("base_hash_match")
    if persona_hash(current_prompt) != base_hash:
        failures.append(
            GateFailure(
                ErrorCode.BASE_HASH_MISMATCH,
                "当前 Persona 哈希与生成基线不一致（疑似外部修改）",
            )
        )

    # 5+6. 标记合法性与区块外逐字节不变
    checks.append("markers_and_block")
    base_split = split_managed_block(base_prompt)
    cand_split = split_managed_block(candidate_prompt)
    if job.edit_mode == EditMode.MANAGED_BLOCK.value:
        if base_split.status == MARKERS_ABSENT:
            # 服务层负责在生成前补齐标记，此处视为基线非法
            failures.append(
                GateFailure(ErrorCode.MARKER_INVALID, "基线缺少受控区块标记")
            )
        elif cand_split.status != MARKERS_OK:
            failures.append(
                GateFailure(
                    ErrorCode.MARKER_INVALID,
                    "候选标记重复/缺失/嵌套或顺序错误",
                )
            )
        elif (
            cand_split.before != base_split.before
            or cand_split.after != base_split.after
        ):
            failures.append(
                GateFailure(
                    ErrorCode.BLOCK_OUTSIDE_MODIFIED,
                    "受控模式下区块外内容被修改",
                )
            )
    else:
        # full_prompt：标记若存在必须保持合法（便于切回受控模式）
        if cand_split.status not in (MARKERS_OK, MARKERS_ABSENT):
            failures.append(
                GateFailure(
                    ErrorCode.MARKER_INVALID,
                    "候选标记重复/缺失/嵌套或顺序错误",
                )
            )

    # 7. 长度 / 增长率 / 改动率
    checks.append("length_and_change_ratio")
    if job.edit_mode == EditMode.MANAGED_BLOCK.value:
        if cand_split.status == MARKERS_OK and len(cand_split.inner) > block_max_chars:
            failures.append(
                GateFailure(
                    ErrorCode.LENGTH_EXCEEDED,
                    f"受控区块内容 {len(cand_split.inner)} 字符"
                    f"超过上限 {block_max_chars}",
                )
            )
    else:
        length_failed = False
        if len(candidate_prompt) > int(len(base_prompt) * full_max_growth_ratio):
            length_failed = True
            failures.append(
                GateFailure(
                    ErrorCode.LENGTH_EXCEEDED,
                    f"候选长度超过原长度的 {full_max_growth_ratio:.0%}",
                )
            )
        if len(candidate_prompt) > full_max_length:
            length_failed = True
            failures.append(
                GateFailure(
                    ErrorCode.LENGTH_EXCEEDED,
                    f"候选长度 {len(candidate_prompt)} 超过绝对上限 {full_max_length}",
                )
            )
        # 长度已越界则跳过改动率计算（同一错误码；
        # 长文本 SequenceMatcher 在极端重复内容下耗时过高）
        ratio_cap = min(full_max_change_ratio, MAX_CHANGE_RATIO_CAP)
        if base_prompt and not length_failed:
            similarity = difflib.SequenceMatcher(
                None, base_prompt, candidate_prompt, autojunk=False
            ).ratio()
            change_ratio = 1.0 - similarity
            if change_ratio > ratio_cap:
                failures.append(
                    GateFailure(
                        ErrorCode.LENGTH_EXCEEDED,
                        f"字符改动率 {change_ratio:.1%} 超过上限 {ratio_cap:.0%}",
                    )
                )

    # 8. protected_fragments 逐字存在
    checks.append("protected_fragments")
    for frag in job.protected_fragments:
        if frag and frag not in candidate_prompt:
            failures.append(
                GateFailure(
                    ErrorCode.PROTECTED_FRAGMENT_MISSING,
                    f"保护片段丢失：{frag[:30]}...",
                )
            )

    # 9. 候选不含来源群 ID / 用户 ID / 已知用户名
    checks.append("privacy_leak")
    for gid in job.source_group_ids:
        if gid and gid in candidate_prompt:
            failures.append(
                GateFailure(ErrorCode.PRIVACY_LEAK, "候选包含来源群 ID")
            )
            break
    for uid in job.source_user_ids:
        if uid and uid in candidate_prompt:
            failures.append(
                GateFailure(ErrorCode.PRIVACY_LEAK, "候选包含来源用户 ID")
            )
            break
    for name in known_user_names or []:
        if name and len(name) >= _MIN_NAME_LEAK_CHARS and name in candidate_prompt:
            failures.append(
                GateFailure(ErrorCode.PRIVACY_LEAK, f"候选包含已知用户名：{name}")
            )
            break

    # 10. 与任一语料不得有 >= max_reuse_chars 连续复用
    checks.append("corpus_reuse")
    for text in corpus_texts or []:
        if len(text) < max_reuse_chars:
            continue
        reused = _longest_common_substring_len(candidate_prompt, text)
        if reused >= max_reuse_chars:
            failures.append(
                GateFailure(
                    ErrorCode.CORPUS_REUSE,
                    f"候选与语料存在 {reused} 字符连续复用"
                    f"（上限 {max_reuse_chars}）",
                )
            )
            break

    # 11. 非 system_prompt 字段不变：由发布调用方式保证
    # （update_persona 只传 persona_id + system_prompt），此处记录约定
    checks.append("forbidden_fields_by_convention")

    # 12. 候选哈希不同于当前哈希，否则 no_change 不发布
    checks.append("no_change")
    if persona_hash(candidate_prompt) == persona_hash(current_prompt):
        no_change = True

    passed = not failures and not no_change
    return ValidationOutcome(
        passed=passed,
        no_change=no_change,
        failures=failures,
        checks=checks,
    )

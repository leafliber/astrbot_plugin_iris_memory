"""
Iris Chat Memory - 人格自迭代数据模型

定义 Job / Run / Revision / StyleSample 数据类与全部状态枚举、
机器可读错误码。枚举值与数据库、Web API 共用，是后续阶段
（迭代引擎、发布闸门、Web 路由）的契约层。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(Enum):
    """Job 状态"""

    ACTIVE = "active"  # 正常运行，允许自动触发
    PAUSED = "paused"  # 管理员暂停
    CONFLICT = "conflict"  # 检测到外部修改或标记冲突，停止自动发布
    PAUSED_ERROR = "paused_error"  # 连续失败熔断，需管理员查看后恢复


class EditMode(Enum):
    """编辑模式"""

    MANAGED_BLOCK = "managed_block"  # 只维护 IRIS_EVOLUTION 受控区块
    FULL_PROMPT = "full_prompt"  # 允许改写完整 system_prompt（强制独立审查）


class ApprovalMode(Enum):
    """审批模式"""

    AUTO = "auto"  # 校验通过后自动发布
    MANUAL = "manual"  # 生成后停在 candidate，等待管理员批准


class RevisionStatus(Enum):
    """Revision 状态"""

    CANDIDATE = "candidate"  # 候选已生成，等待审批或发布
    PUBLISHING = "publishing"  # 发布意图已记录，正在调用 PersonaManager
    APPLIED = "applied"  # 已发布并回读验证
    REJECTED = "rejected"  # 管理员拒绝
    FAILED_VALIDATION = "failed_validation"  # 确定性闸门/审查未通过
    PUBLISH_FAILED = "publish_failed"  # 发布调用失败或中断后哈希仍为基线
    EXTERNAL_CHANGE = "external_change"  # 记录外部修改快照
    ROLLBACK = "rollback"  # 回滚产生的新版本
    NO_CHANGE = "no_change"  # 候选与当前一致，未发布


class TriggerType(Enum):
    """触发方式"""

    AUTO = "auto"  # 自动门槛触发
    MANUAL = "manual"  # 管理员手动执行
    ROLLBACK = "rollback"  # 回滚操作


class RunStatus(Enum):
    """Run 执行状态"""

    RUNNING = "running"  # 执行中
    SUCCESS = "success"  # 成功（含 no_change）
    FAILED = "failed"  # 失败（error_code/error_message 记录原因）


class ErrorCode(Enum):
    """机器可读错误码

    供 Web API 错误响应与 evolution_runs.error_code 使用，
    取值稳定，不随文案调整变化。
    """

    # 通用
    INTERNAL_ERROR = "internal_error"
    INVALID_PARAMS = "invalid_params"
    NOT_FOUND = "not_found"

    # Job / 触发
    JOB_NOT_ACTIVE = "job_not_active"
    JOB_ALREADY_EXISTS = "job_already_exists"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    TRIGGER_CONDITIONS_NOT_MET = "trigger_conditions_not_met"
    CIRCUIT_OPEN = "circuit_open"
    CONFLICT_UNRESOLVED = "conflict_unresolved"

    # LLM 调用
    PROVIDER_ERROR = "provider_error"
    PROVIDER_TIMEOUT = "provider_timeout"
    ANALYSIS_PARSE_FAILED = "analysis_parse_failed"
    ANALYSIS_LOW_CONFIDENCE = "analysis_low_confidence"
    GENERATION_PARSE_FAILED = "generation_parse_failed"

    # 确定性发布闸门（文档 §10）
    INVALID_JSON = "invalid_json"  # 1. JSON schema/类型/枚举非法
    EMPTY_CANDIDATE = "empty_candidate"  # 2. 候选为空或含异常控制字符
    PERSONA_MISMATCH = "persona_mismatch"  # 3. Persona ID 与 Job 不一致
    BASE_HASH_MISMATCH = "base_hash_mismatch"  # 4. 当前哈希与 base_hash 不一致
    BLOCK_OUTSIDE_MODIFIED = "block_outside_modified"  # 5. 受控模式下区块外被修改
    MARKER_INVALID = "marker_invalid"  # 6. 标记重复/缺失/嵌套/顺序错误
    LENGTH_EXCEEDED = "length_exceeded"  # 7. 长度/增长率越界
    PROTECTED_FRAGMENT_MISSING = "protected_fragment_missing"  # 8. 保护片段丢失
    PRIVACY_LEAK = "privacy_leak"  # 9. 候选包含来源群/用户 ID 或用户名
    CORPUS_REUSE = "corpus_reuse"  # 10. 与语料出现超长连续复用
    FORBIDDEN_FIELD_MODIFIED = "forbidden_field_modified"  # 11. 修改了非 system_prompt 字段
    NO_CHANGE = "no_change"  # 12. 候选与当前哈希一致

    # 阶段 C 审查
    REVIEW_FAILED = "review_failed"
    REVIEW_PARSE_FAILED = "review_parse_failed"

    # 发布
    PUBLISH_FAILED = "publish_failed"
    PERSONA_NOT_FOUND = "persona_not_found"
    EXTERNAL_CHANGE = "external_change"


@dataclass
class EvolutionJob:
    """迭代任务：一个 Job 对应一个目标 Persona（persona_id 唯一）

    Attributes:
        source_group_ids: 语料来源群过滤，空列表表示全部群聊
        source_user_ids: 语料来源用户过滤，空列表表示匹配群内全部真人用户
        last_sample_cursor: 自动触发语料计数基线（最近一次成功迭代时的最大 Sample ID）
    """

    persona_id: str
    id: int = 0
    name: str = ""
    goal_preset_id: str = "natural"
    custom_goal: str = ""
    source_group_ids: List[str] = field(default_factory=list)
    source_user_ids: List[str] = field(default_factory=list)
    edit_mode: str = EditMode.MANAGED_BLOCK.value
    approval_mode: str = ApprovalMode.AUTO.value
    status: str = JobStatus.ACTIVE.value
    trigger_sample_count: int = 100
    min_interval_hours: int = 24
    provider_id: str = ""
    reviewer_provider_id: str = ""
    protected_fragments: List[str] = field(default_factory=list)
    last_success_at: Optional[float] = None
    last_sample_cursor: int = 0
    last_applied_revision_id: Optional[int] = None
    consecutive_failures: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class StyleSample:
    """风格语料样本（入库前已脱敏、规范化、去重）"""

    normalized_text: str
    dedupe_hash: str
    id: int = 0
    platform: str = ""
    group_id: str = ""
    group_name: str = ""
    user_id: str = ""
    user_name: str = ""
    message_id: Optional[str] = None
    char_count: int = 0
    created_at: float = 0.0


@dataclass
class EvolutionRun:
    """一次迭代执行记录（持久审计日志）"""

    job_id: int
    id: int = 0
    trigger_type: str = TriggerType.AUTO.value
    status: str = RunStatus.RUNNING.value
    sample_cursor_from: int = 0
    sample_cursor_to: int = 0
    eligible_count: int = 0
    selected_count: int = 0
    started_at: float = 0.0
    finished_at: Optional[float] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    analysis_tokens: int = 0
    generation_tokens: int = 0
    review_tokens: int = 0


@dataclass
class PersonaRevision:
    """人格版本：保存修改前后完整快照，是回滚真相来源

    Attributes:
        version: 同一 Job 内单调递增版本号，(job_id, version) 唯一
        goal_snapshot: 当次使用的目标完整快照（含预设版本号）
        extra: 其余 JSON 快照字段（风格画像/校验/审查/Provider），
            后续阶段填充，本阶段仅保留存储位
    """

    job_id: int
    version: int
    id: int = 0
    parent_revision_id: Optional[int] = None
    status: str = RevisionStatus.CANDIDATE.value
    trigger_type: str = TriggerType.AUTO.value
    edit_mode: str = EditMode.MANAGED_BLOCK.value
    approval_mode: str = ApprovalMode.AUTO.value
    base_prompt: Optional[str] = None
    result_prompt: Optional[str] = None
    base_hash: Optional[str] = None
    result_hash: Optional[str] = None
    goal_snapshot: Dict[str, Any] = field(default_factory=dict)
    style_profile: Dict[str, Any] = field(default_factory=dict)
    change_summary: List[str] = field(default_factory=list)
    rationale: str = ""
    decision_reason: str = ""  # 拒绝/回滚/采纳基线等管理决策原因（文档 §13.1）
    confidence: Optional[float] = None
    validation: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, Any] = field(default_factory=dict)
    provider_snapshot: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    applied_at: Optional[float] = None

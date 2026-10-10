"""Host API shapes. Omission is distinct from JSON null, especially for PATCH.

Unknown response fields are retained on the wire. Flexible diagnostic records
are JSON objects; they are not suitable for direct inclusion in LLM prompts.
"""

from typing import Literal, Required, TypedDict

type JSON = None | bool | int | float | str | list[JSON] | dict[str, JSON]
type JSONObject = dict[str, JSON]
type DetailValue = str | int | float | bool | None
type MemoryKind = Literal["事件", "事实", "偏好", "关系", "观点", "计划", "自我", "其他"]
type Stance = Literal["亲历", "转述", "观点", "推断", "设定"]
type NotificationKind = Literal["goal_reminder", "goal_dedup_result"]


class ErrorField(TypedDict):
    field: str
    message: str


class OtherEntriesPending(TypedDict):
    code: Literal["other_entries_pending"]
    entry_count: int
    time_from: str
    time_to: str
    time_basis: Literal["received_at"]
    message: str


class MediaUpload(TypedDict, total=False):
    entry_id: Required[str]
    content_type: Required[str]
    data_base64: Required[str]
    understanding_text: str | None


class Media(TypedDict):
    id: str
    sha256: str
    content_type: str
    size_bytes: int
    kind: Literal["image", "audio", "video"]
    understanding_source: Literal["host", "system", "refused", "unprocessed"]
    understanding_text: str
    completed_at: str | None
    last_error: str | None


class CustomPace(TypedDict):
    count: int
    idle_seconds: int
    max_wait_seconds: int


class Message(TypedDict, total=False):
    sender: Required[str]
    content: Required[str]
    occurred_at: Required[str]
    dedupe_key: Required[str]
    platform: str
    entry_name: str | None
    entry_kind: str
    kind: Literal["message", "self_output", "action_result", "event"]
    account_id: str | None
    scene_identity: str | None
    quote_author: str | None
    quote_author_account_id: str | None
    quote_content: str | None
    media_ids: list[str]
    pace: Literal["realtime", "standard", "economy"] | CustomPace


class ReceiveResult(TypedDict):
    message_ids: list[int]
    pending_count: int


class LearnResult(TypedDict):
    accepted: bool
    pending_count: int
    paused: bool
    reason: str | None


class PrepareOptions(TypedDict, total=False):
    judge: bool
    judge_budget_seconds: float
    text: str | None
    participants: list[str] | None
    known_memory_ids: list[int]
    recent_limit: int
    memory_limit: int
    token_budget: int
    goal_limit: int


class SearchOptions(TypedDict, total=False):
    entry_id: str | None
    text: str
    people: list[str]
    kinds: list[MemoryKind]
    stances: list[Stance]
    time_from: str | None
    time_to: str | None
    include_forgotten: bool
    include_goals: bool
    include_state: bool
    limit: int


class Persona(TypedDict):
    version: int | None
    content: str
    generated_at: str | None
    needs_update: bool
    stale_basis_count: int


class Person(TypedDict):
    id: str
    name: str


class Memory(TypedDict, total=False):
    id: Required[int]
    content: Required[str]
    reason: Required[Literal["relevant", "person_highlight"]]
    kind: MemoryKind
    stance: Stance
    belief: int
    importance: int
    retention: int
    revision: int
    event_time: str | None
    speaker_subject_id: str
    speaker_name: str
    about: list[Person]
    sources: list[JSONObject]
    subject_annotations: JSONObject
    lifecycle: str
    world: str | None


class RecentMessage(TypedDict, total=False):
    id: Required[int]
    content: Required[str]
    sender_name: str
    sender_subject_id: str
    kind: str
    occurred_at: str
    unlearned: bool
    quote_author_name: str | None
    quote_content: str | None


class StateSource(TypedDict, total=False):
    host: str | None
    entry_id: str | None


class StatePatch(StateSource, total=False):
    details: dict[str, DetailValue]
    mood: str | None


class StatePut(StatePatch, total=False):
    activity: Required[str]
    started_at: str


class StateDetail(TypedDict):
    value: DetailValue
    updated_at: str


class CurrentState(TypedDict, total=False):
    activity: str
    activity_updated_at: str
    details: dict[str, StateDetail]
    mood: str | None
    mood_updated_at: str | None
    started_at: str
    start_time_basis: Literal["host", "first_report"]
    duration_seconds: float
    updated_at: str
    possibly_stale: bool
    stale_after_minutes: int
    host: str | None
    entry_id: str | None


class GoalCreate(StateSource, total=False):
    content: Required[str]
    kind: Literal["normal", "question"]
    deadline: str | None
    reminder_minutes: int | None
    people: list[str]
    host_key: str | None


class GoalPatch(TypedDict, total=False):
    expected_revision: int | None
    state: Literal["completed", "abandoned"]
    deadline: str | None
    reminder_minutes: int | None


class GoalQuery(TypedDict, total=False):
    state: Literal["open", "completed", "abandoned"]
    kind: Literal["normal", "question"]
    overdue: bool
    due_soon: bool
    entry_id: str
    limit: int
    offset: int


class Goal(TypedDict, total=False):
    id: Required[int]
    content: Required[str]
    kind: Literal["normal", "question"]
    origin: str
    state: Literal["open", "completed", "abandoned"]
    deadline: str | None
    deadline_at: str | None
    deadline_unresolved: bool
    reminder_minutes: int | None
    effective_reminder_minutes: int | None
    people: list[str]
    entry_id: str | None
    host_key: str | None
    revision: int
    created_at: str
    updated_at: str
    closed_at: str | None
    closed_by: str | None
    merged_into: int | None
    overdue: bool
    due_soon: bool
    possible_duplicate: bool
    possible_duplicate_ids: list[int]


class GoalList(TypedDict):
    items: list[Goal]
    total: int
    limit: int
    offset: int


class DedupResult(TypedDict):
    status: Literal["created", "merged", "possible_duplicate", "pending"]
    target_id: int | None


class GoalReceipt(TypedDict):
    goal: Goal
    submitted_id: int
    dedup: DedupResult


class SearchResult(TypedDict, total=False):
    recall_id: Required[str]
    memories: Required[list[Memory]]
    hints: Required[list[JSONObject]]
    state: CurrentState
    goals: list[Goal]


class Prepared(TypedDict):
    recall_id: str
    persona: Persona
    memories: list[Memory]
    recent_messages: list[RecentMessage]
    state: CurrentState
    goals: list[Goal]
    hints: list[JSONObject]
    judgment: JSONObject


class FeedbackResult(TypedDict):
    recall_id: str
    accepted: list[int]
    strengthened: list[int]


class Notification(TypedDict, total=False):
    id: Required[int]
    kind: Required[NotificationKind | str]
    goal_id: int | None
    content: str
    reminder_kind: Literal["soon", "due", "overdue", "immediate"] | None
    deadline_at: str | None
    scheduled_at: str
    published_at: str
    status: Literal["pending", "taken", "cancelled"]
    taken_at: str | None
    cancelled_at: str | None
    payload_json: str | None


class NotificationPage(TypedDict):
    items: list[Notification]
    next_cursor: int
    has_more: bool


class ServiceStatus(TypedDict, total=False):
    service: Required[str]
    models: list[JSONObject]
    entries: list[JSONObject]
    backlog: list[JSONObject]
    batches: list[JSONObject]
    model_health: JSONObject
    usage: JSONObject
    scheduler: JSONObject
    budget: JSONObject
    memory_gap_count: int
    missing_vectors: int
    learning_calls_24h: list[JSONObject]
    learning_latency_24h: JSONObject
    timeouts_seconds: JSONObject

"""
Iris Chat Memory - 默认配置定义

使用 dataclass 提供类型安全的配置定义，支持：
- IDE 自动补全
- 编译时类型检查
- 字段注释即文档
- 扁平化键名访问
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict


@dataclass
class ImageParsingConfig:
    """图片解析配置"""

    enable: bool = False
    provider: str = ""
    mode: Literal["all", "related"] = "related"
    daily_quota: int = 200


@dataclass
class L1BufferConfig:
    """L1 消息上下文缓冲配置（用户可见选项）"""

    enable: bool = True
    summary_provider: str = ""
    inject_queue_length: int = 50
    image_parsing: ImageParsingConfig = field(default_factory=ImageParsingConfig)


@dataclass
class L2MemoryConfig:
    """L2 记忆库配置"""

    enable: bool = True
    embedding_source: Literal["provider", "local"] = "provider"
    embedding_provider: str = ""
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    top_k: int = 10
    relevance_threshold: float = 0.3


@dataclass
class L3KGConfig:
    """L3 知识图谱配置"""

    enable: bool = True
    extraction_provider: str = ""


@dataclass
class ProfileConfig:
    """画像系统配置"""

    enable: bool = True
    analysis_provider: str = ""
    enable_auto_injection: bool = True
    favorability_enable: bool = True


@dataclass
class IsolationConfig:
    """隔离配置"""

    enable_group_memory_isolation: bool = False
    enable_group_isolation: bool = False
    enable_persona_isolation: bool = False


@dataclass
class ScheduledTasksConfig:
    """梦境任务配置"""

    provider: str = ""
    enable_dream: bool = True
    dream_enable_consolidation: bool = True
    dream_enable_temporal_anchor: bool = True
    dream_enable_contradiction: bool = True
    dream_enable_pattern_discovery: bool = True
    dream_enable_knowledge_extract: bool = True
    dream_enable_pruning: bool = True


@dataclass
class ContextControlConfig:
    """上下文控制配置"""

    enable_conversation_cleanup: bool = True


@dataclass
class HiddenConfig:
    """隐藏配置(内部参数)

    这些配置项不会在 WebUI 中展示，用于控制内部行为。
    支持运行时热修改，并自动持久化到 data/iris_memory/hidden_config.json

    每个字段通过 metadata 声明 description 和 group，
    供 Web 前端自动渲染，增删字段时无需手动同步路由代码。
    """

    # Token 预算控制
    token_budget_max_tokens: int = field(
        default=2000,
        metadata={"description": "Token 预算上限", "group": "Token 预算"},
    )

    # L1 缓冲内部参数
    l1_segment_1_length: int = field(
        default=10,
        metadata={
            "description": "L1-1 最新段消息数（始终注入上下文）",
            "group": "L1 缓冲",
        },
    )
    l1_segment_3_length: int = field(
        default=10,
        metadata={
            "description": "L1-3 缓冲段消息数（辅助总结理解）",
            "group": "L1 缓冲",
        },
    )
    l1_max_queue_tokens: int = field(
        default=4000,
        metadata={"description": "队列最大 Token 数，超限触发总结", "group": "L1 缓冲"},
    )
    l1_max_single_message_tokens: int = field(
        default=500,
        metadata={
            "description": "单条消息最大 Token 数，普通消息超限丢弃，合并转发消息超限截断",
            "group": "L1 缓冲",
        },
    )
    l1_inject_max_content_chars: int = field(
        default=300,
        metadata={
            "description": "注入时单条消息最大字符数，0 不截断",
            "group": "L1 缓冲",
        },
    )
    l1_max_memories_per_summary: int = field(
        default=10,
        metadata={"description": "每次总结写入 L2 的最大记忆条数", "group": "L1 缓冲"},
    )

    # 遗忘权重算法参数
    forgetting_lambda: float = field(
        default=0.1,
        metadata={"description": "近因性衰减系数", "group": "遗忘算法"},
    )
    forgetting_threshold: float = field(
        default=0.3,
        metadata={"description": "遗忘阈值", "group": "遗忘算法"},
    )
    forgetting_immediate_eviction_threshold: float = field(
        default=0.1,
        metadata={"description": "极端低分直接淘汰阈值", "group": "遗忘算法"},
    )

    # L2 记忆遗忘评分权重 S = w_recency·R + w_frequency·F + w_confidence·C + w_isolation·(1-D)
    forgetting_l2_weight_recency: float = field(
        default=0.4,
        metadata={"description": "L2 遗忘评分: 近因性权重", "group": "遗忘算法"},
    )
    forgetting_l2_weight_frequency: float = field(
        default=0.35,
        metadata={"description": "L2 遗忘评分: 频率性权重", "group": "遗忘算法"},
    )
    forgetting_l2_weight_confidence: float = field(
        default=0.25,
        metadata={"description": "L2 遗忘评分: 置信度权重", "group": "遗忘算法"},
    )
    forgetting_l2_weight_isolation: float = field(
        default=0.0,
        metadata={
            "description": "L2 遗忘评分: 孤立度权重（设为 0 则禁用孤立度因子）",
            "group": "遗忘算法",
        },
    )
    l2_retention_days: int = field(
        default=30,
        metadata={
            "description": "L2 记忆保留天数（超期且低分才淘汰）",
            "group": "遗忘算法",
        },
    )
    l2_checkpoint_writes: int = field(
        default=50,
        metadata={
            "description": "L2 FAISS 索引每 N 次写入异步落盘一次，收敛崩溃丢失窗口（0=禁用，仅关闭时保存）",
            "group": "持久化",
        },
    )

    # L3 知识图谱遗忘评分权重 S = w_recency·R + w_structure·(1-D) + w_confidence·C + w_verification·V
    forgetting_kg_weight_recency: float = field(
        default=0.15,
        metadata={"description": "KG 遗忘评分: 近因性权重", "group": "遗忘算法"},
    )
    forgetting_kg_weight_structure: float = field(
        default=0.40,
        metadata={"description": "KG 遗忘评分: 结构重要性权重", "group": "遗忘算法"},
    )
    forgetting_kg_weight_confidence: float = field(
        default=0.15,
        metadata={"description": "KG 遗忘评分: 置信度权重", "group": "遗忘算法"},
    )
    forgetting_kg_weight_verification: float = field(
        default=0.30,
        metadata={"description": "KG 遗忘评分: 验证度权重", "group": "遗忘算法"},
    )
    forgetting_lambda_kg: float = field(
        default=0.05,
        metadata={
            "description": "KG 近因性衰减系数（比 L2 更慢）",
            "group": "遗忘算法",
        },
    )

    enable_context_logging: bool = field(
        default=False,
        metadata={"description": "启用 LLM 上下文日志输出", "group": "LLM 调用管理"},
    )

    l2_similarity_threshold: float = field(
        default=0.87,
        metadata={"description": "L2 去重相似度阈值", "group": "L2 记忆"},
    )
    l2_timeout_ms: int = field(
        default=4000,
        metadata={"description": "L2 检索超时(ms)", "group": "L2 记忆"},
    )

    # L3 知识图谱参数
    l3_timeout_ms: int = field(
        default=1500,
        metadata={"description": "L3 检索超时(ms)", "group": "L3 知识图谱"},
    )
    l3_expansion_depth: int = field(
        default=2,
        metadata={"description": "图谱检索路径扩展深度", "group": "L3 知识图谱"},
    )
    l3_enable_type_whitelist: bool = field(
        default=True,
        metadata={"description": "启用 LLM 实体类型白名单约束", "group": "L3 知识图谱"},
    )
    l3_max_inject_tokens: int = field(
        default=600,
        metadata={
            "description": "知识图谱注入上下文最大 token 数",
            "group": "L3 知识图谱",
        },
    )
    node_confidence_threshold: float = field(
        default=0.3,
        metadata={"description": "节点最低置信度", "group": "L3 知识图谱"},
    )
    forgetting_threshold_kg: float = field(
        default=0.2,
        metadata={"description": "知识图谱遗忘阈值", "group": "L3 知识图谱"},
    )
    kg_retention_days: int = field(
        default=30,
        metadata={"description": "知识图谱保留天数", "group": "L3 知识图谱"},
    )

    # LLM 调用管理参数
    call_log_max_entries: int = field(
        default=100,
        metadata={"description": "调用日志最大保留条数", "group": "LLM 调用管理"},
    )
    llm_call_timeout_ms: int = field(
        default=60000,
        metadata={
            "description": "LLM 调用全局超时(ms)，0 表示不限制。兜底防止 provider 卡死阻塞会话锁",
            "group": "LLM 调用管理",
        },
    )

    # 梦境任务参数
    dream_task_interval_hours: int = field(
        default=24,
        metadata={"description": "梦境任务间隔(小时)", "group": "梦境任务"},
    )
    dream_consolidation_similarity_threshold: float = field(
        default=0.85,
        metadata={"description": "合并相似度阈值", "group": "梦境任务"},
    )
    dream_consolidation_batch_size: int = field(
        default=10,
        metadata={"description": "合并批处理大小", "group": "梦境任务"},
    )
    dream_consolidation_scan_budget: int = field(
        default=200,
        metadata={"description": "每轮扫描记忆条数上限", "group": "梦境任务"},
    )
    dream_consolidation_query_batch_size: int = field(
        default=50,
        metadata={"description": "向量检索批量查询大小", "group": "梦境任务"},
    )
    dream_consolidation_query_top_k: int = field(
        default=5,
        metadata={
            "description": "合并阶段向量检索每条返回的近邻数",
            "group": "梦境任务",
        },
    )
    dream_consolidation_max_group_size: int = field(
        default=5,
        metadata={"description": "单组合并最大条目数", "group": "梦境任务"},
    )
    dream_temporal_anchor_batch_size: int = field(
        default=50,
        metadata={"description": "时间锚定批处理大小", "group": "梦境任务"},
    )
    dream_contradiction_similarity_floor: float = field(
        default=0.55,
        metadata={"description": "矛盾检测相似度下限", "group": "梦境任务"},
    )
    dream_contradiction_similarity_ceiling: float = field(
        default=0.85,
        metadata={"description": "矛盾检测相似度上限", "group": "梦境任务"},
    )
    dream_contradiction_max_groups: int = field(
        default=20,
        metadata={"description": "矛盾检测最大分组数", "group": "梦境任务"},
    )
    dream_contradiction_scan_budget: int = field(
        default=200,
        metadata={"description": "矛盾检测每轮扫描记忆条数上限", "group": "梦境任务"},
    )
    dream_contradiction_query_batch_size: int = field(
        default=50,
        metadata={"description": "矛盾检测向量检索批量查询大小", "group": "梦境任务"},
    )
    dream_pattern_sample_size: int = field(
        default=30,
        metadata={"description": "模式发现采样数", "group": "梦境任务"},
    )
    dream_pattern_min_confidence: str = field(
        default="medium",
        metadata={"description": "模式发现最低置信度", "group": "梦境任务"},
    )
    dream_knowledge_extract_min_unprocessed: int = field(
        default=10,
        metadata={"description": "最小未处理记忆数量阈值", "group": "梦境任务"},
    )
    dream_knowledge_extract_batch_size: int = field(
        default=20,
        metadata={"description": "每批处理记忆数", "group": "梦境任务"},
    )
    eviction_batch_size: int = field(
        default=100,
        metadata={"description": "淘汰批处理大小", "group": "梦境任务"},
    )
    image_cache_cleanup_interval_hours: int = field(
        default=24,
        metadata={"description": "图片缓存清理任务间隔(小时)", "group": "梦境任务"},
    )

    # L3 知识图谱提取 - 相关记忆检索权重
    kg_extraction_semantic_weight: float = field(
        default=0.5,
        metadata={"description": "语义相似记忆权重", "group": "L3 知识图谱"},
    )
    kg_extraction_same_group_weight: float = field(
        default=0.3,
        metadata={"description": "同群聊记忆权重", "group": "L3 知识图谱"},
    )
    kg_extraction_same_user_weight: float = field(
        default=0.2,
        metadata={"description": "同用户记忆权重", "group": "L3 知识图谱"},
    )

    # 画像系统参数
    profile_max_messages_for_analysis: int = field(
        default=50,
        metadata={"description": "群聊画像分析时最大消息数", "group": "画像系统"},
    )
    profile_max_messages_for_user_analysis: int = field(
        default=30,
        metadata={"description": "用户画像分析时最大消息数", "group": "画像系统"},
    )
    profile_mid_update_interval_summaries: int = field(
        default=5,
        metadata={"description": "中期更新: 每隔N次总结触发", "group": "画像系统"},
    )
    profile_mid_update_interval_hours: float = field(
        default=24.0,
        metadata={"description": "中期更新: 最短间隔(小时)", "group": "画像系统"},
    )
    profile_long_update_interval_hours: float = field(
        default=168.0,
        metadata={"description": "长期更新: 最短间隔(小时)", "group": "画像系统"},
    )
    profile_favorability_max_delta: float = field(
        default=20.0,
        metadata={"description": "好感度单次更新最大变化量（夹紧用）", "group": "画像系统"},
    )
    profile_message_max_chars: int = field(
        default=150,
        metadata={"description": "画像分析时单条消息最大字符数（0 不截断）", "group": "画像系统"},
    )

    # 图片解析（从 _conf_schema.json 迁移）
    image_max_parse_per_request: int = field(
        default=3,
        metadata={
            "description": "单次请求最大图片解析数（与并发批次对齐）",
            "group": "图片处理",
        },
    )
    image_max_concurrent_parse: int = field(
        default=3,
        metadata={"description": "最大并发图片解析数", "group": "图片处理"},
    )
    image_cache_retention_days: int = field(
        default=7,
        metadata={"description": "图片解析结果缓存保留天数", "group": "图片处理"},
    )
    image_skip_on_passive_trigger: bool = field(
        default=True,
        metadata={"description": "被动触发时跳过图片解析", "group": "图片处理"},
    )
    image_parse_timeout_ms: int = field(
        default=30000,
        metadata={
            "description": "单次请求图片解析整体超时(ms)，0 表示不限制",
            "group": "图片处理",
        },
    )

    # 图片去重参数
    image_phash_enable: bool = field(
        default=True,
        metadata={"description": "启用 pHash 感知哈希去重", "group": "图片处理"},
    )
    image_phash_threshold: int = field(
        default=10,
        metadata={"description": "pHash 汉明距离阈值", "group": "图片处理"},
    )

    # 无效图过滤参数
    image_filter_enable: bool = field(
        default=True,
        metadata={"description": "启用无效图过滤(纯色/过小)", "group": "图片处理"},
    )
    image_filter_min_size: int = field(
        default=16,
        metadata={"description": "最小图片尺寸(像素)", "group": "图片处理"},
    )
    image_filter_std_threshold: float = field(
        default=5.0,
        metadata={"description": "纯色检测标准差阈值", "group": "图片处理"},
    )

    # 输入清理参数
    input_sanitizer_enable: bool = field(
        default=True,
        metadata={"description": "启用 Prompt 注入过滤", "group": "输入清理"},
    )
    input_sanitizer_max_length: int = field(
        default=10000,
        metadata={"description": "输入最大长度", "group": "输入清理"},
    )

    # 遗忘确认参数
    forgetting_llm_confirm_enable: bool = field(
        default=False,
        metadata={"description": "启用 LLM 最终兜底确认遗忘", "group": "遗忘确认"},
    )
    forgetting_llm_confirm_provider: str = field(
        default="",
        metadata={
            "description": "确认使用的 Provider(空则使用默认)",
            "group": "遗忘确认",
        },
    )
    forgetting_llm_confirm_threshold: float = field(
        default=0.15,
        metadata={"description": "评分低于此值才触发 LLM 确认", "group": "遗忘确认"},
    )

    # L2 查询改写参数
    l2_query_rewrite_enable: bool = field(
        default=True,
        metadata={"description": "启用 L2 检索查询改写", "group": "L2 查询改写"},
    )
    l2_query_rewrite_provider: str = field(
        default="",
        metadata={
            "description": "查询改写使用的 Provider(空则使用默认)",
            "group": "L2 查询改写",
        },
    )
    l2_query_rewrite_timeout_ms: int = field(
        default=3000,
        metadata={"description": "查询改写超时(ms)", "group": "L2 查询改写"},
    )

    # 上下文清理参数
    enable_legacy_cleanup: bool = field(
        default=False,
        metadata={
            "description": "启用旧版对话清理模式(完成后删除整个对话)，默认关闭。"
            "关闭时采用请求前清空 contexts 策略，保留对话 ID 以兼容主动回复",
            "group": "上下文控制",
        },
    )

    # 主动回复 - 基本参数
    reply_mute_start_hour: int = field(
        default=1,
        metadata={
            "description": "静音时段开始小时(0-23)，静音时段内 Iris 不会主动触发任何回复",
            "group": "主动回复·基本参数",
        },
    )
    reply_mute_start_minute: int = field(
        default=0,
        metadata={
            "description": "静音时段开始分钟(0-59)",
            "group": "主动回复·基本参数",
        },
    )
    reply_mute_end_hour: int = field(
        default=7,
        metadata={
            "description": "静音时段结束小时(0-23)",
            "group": "主动回复·基本参数",
        },
    )
    reply_mute_end_minute: int = field(
        default=0,
        metadata={
            "description": "静音时段结束分钟(0-59)",
            "group": "主动回复·基本参数",
        },
    )
    reply_window_size: int = field(
        default=15,
        metadata={
            "description": "滑动记忆窗口大小（条数），保留最近 N 条有效发言",
            "group": "主动回复·基本参数",
        },
    )
    reply_default_n: int = field(
        default=15,
        metadata={
            "description": "默认消息计数阈值 N，每收到 N 条有效消息触发一次采样",
            "group": "主动回复·基本参数",
        },
    )
    reply_default_t: int = field(
        default=30,
        metadata={
            "description": "默认时间间隔阈值 T（分钟），距上次采样超过 T 分钟且有新消息时触发",
            "group": "主动回复·基本参数",
        },
    )
    reply_max_token: int = field(
        default=3000,
        metadata={
            "description": "上下文 token 上限，提交给 LLM 的上下文最大 token 数",
            "group": "主动回复·基本参数",
        },
    )
    reply_follow_up_ttl: int = field(
        default=10,
        metadata={
            "description": "跟进锚点默认 TTL（分钟），对话锚点中关注用户/关键词的存活时长",
            "group": "主动回复·基本参数",
        },
    )
    reply_follow_up_aggregate_window: int = field(
        default=6,
        metadata={
            "description": "follow-up 消息聚合等待窗口（秒），锚点命中后等待此时间再触发 LLM 评估",
            "group": "主动回复·基本参数",
        },
    )
    reply_quality_threshold: float = field(
        default=0.2,
        metadata={
            "description": "消息质量评分阈值，低于此阈值的消息不进入滑动窗口",
            "group": "主动回复·基本参数",
        },
    )
    reply_trigger_min_interval: int = field(
        default=30,
        metadata={
            "description": "触发阶段最小间隔（秒），同一群两次触发 LLM 调用之间的最小时间间隔",
            "group": "主动回复·基本参数",
        },
    )
    reply_boost_factor: float = field(
        default=0.6,
        metadata={
            "description": "回复后频率提升系数，Iris 回复后有效阈值临时乘以此系数（<1.0 降低阈值）",
            "group": "主动回复·基本参数",
        },
    )
    reply_boost_duration: int = field(
        default=15,
        metadata={
            "description": "频率提升持续时间（分钟），回复后 boost 效果的持续时间，之后线性衰减回正常",
            "group": "主动回复·基本参数",
        },
    )
    reply_max_boosted_replies: int = field(
        default=5,
        metadata={
            "description": "最大连续回复 boost 次数，连续回复不超过此次数时享受完整 boost，超出后逐渐减弱",
            "group": "主动回复·基本参数",
        },
    )

    # 主动回复 - 主动发起
    proactive_enabled: bool = field(
        default=False,
        metadata={
            "description": "启用主动发起会话，开启后 Iris 会在群冷场或话题结束时主动开启话题",
            "group": "主动回复·主动发起",
        },
    )
    proactive_check_interval: int = field(
        default=5,
        metadata={
            "description": "发起检查周期（分钟），每隔此时间扫描一次白名单群，评估是否满足发起条件",
            "group": "主动回复·主动发起",
        },
    )
    proactive_quiet_minutes: int = field(
        default=120,
        metadata={
            "description": "冷场静默阈值（分钟），群内最后一条消息超过此时间无人说话，才考虑主动发起",
            "group": "主动回复·主动发起",
        },
    )
    proactive_max_per_day: int = field(
        default=2,
        metadata={
            "description": "每日最大发起次数，每个群每天最多主动发起的次数",
            "group": "主动回复·主动发起",
        },
    )
    proactive_min_interval: int = field(
        default=360,
        metadata={
            "description": "两次发起最小间隔（分钟），同一群两次主动发起之间的最小时间间隔",
            "group": "主动回复·主动发起",
        },
    )
    proactive_drift_delay: int = field(
        default=15,
        metadata={
            "description": "话题结束后发起延迟（分钟），检测到话题结束（drifted）后，若持续静默此时间可提前发起新话题",
            "group": "主动回复·主动发起",
        },
    )
    proactive_pending_timeout: int = field(
        default=30,
        metadata={
            "description": "发起接话等待（分钟），发起后等待群友接话的时间，超时视为无人接话",
            "group": "主动回复·主动发起",
        },
    )
    proactive_max_streak: int = field(
        default=2,
        metadata={
            "description": "当日无人接话上限（次），连续发起无人接话达到此次数后，当天不再发起",
            "group": "主动回复·主动发起",
        },
    )
    proactive_instruction: str = field(
        default="",
        metadata={
            "description": "发起话题倾向（可选），自定义发起话题的偏好说明，如「多聊技术话题」，留空由 Iris 自行发挥",
            "group": "主动回复·主动发起",
        },
    )
    proactive_max_message_len: int = field(
        default=300,
        metadata={
            "description": "发起消息最大长度（字符），主动发起消息超过此长度将被截断",
            "group": "主动回复·主动发起",
        },
    )


@dataclass
class Defaults:
    """所有默认配置的统一入口

    提供扁平化键名访问方法，支持 "l1_buffer.enable" 格式的键名。
    """

    l1_buffer: L1BufferConfig = field(default_factory=L1BufferConfig)
    l2_memory: L2MemoryConfig = field(default_factory=L2MemoryConfig)
    l3_kg: L3KGConfig = field(default_factory=L3KGConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    isolation_config: IsolationConfig = field(default_factory=IsolationConfig)
    scheduled_tasks: ScheduledTasksConfig = field(default_factory=ScheduledTasksConfig)
    context_control: ContextControlConfig = field(default_factory=ContextControlConfig)
    hidden: HiddenConfig = field(default_factory=HiddenConfig)

    def get_by_flat_key(self, flat_key: str) -> Optional[object]:
        """通过扁平化键名获取默认值

        Args:
            flat_key: 扁平化键名，支持两种格式：
                - "l1_buffer.enable" (用户配置)
                - "debug_mode" (隐藏配置)

        Returns:
            默认值，找不到返回 None

        Examples:
            >>> defaults = Defaults()
            >>> defaults.get_by_flat_key("l1_buffer.enable")
            True
            >>> defaults.get_by_flat_key("debug_mode")
            False
        """
        parts = flat_key.split(".")

        if len(parts) == 1:
            # 隐藏配置项(单层键名)
            return getattr(self.hidden, parts[0], None)
        elif len(parts) == 2:
            # 用户配置项(双层键名：section.key)
            section, key = parts
            section_config = getattr(self, section, None)
            if section_config is not None:
                return getattr(section_config, key, None)
        elif len(parts) == 3:
            # 嵌套配置项(三层键名：section.subsection.key)
            section, subsection, key = parts
            section_config = getattr(self, section, None)
            if section_config is not None:
                subsection_config = getattr(section_config, subsection, None)
                if subsection_config is not None:
                    return getattr(subsection_config, key, None)

        return None

    def get_section_defaults(self, section: str) -> Dict[str, object]:
        """获取指定配置分组的所有默认值

        Args:
            section: 配置分组名，如 "l1_buffer"

        Returns:
            配置字典

        Examples:
            >>> defaults = Defaults()
            >>> l1_defaults = defaults.get_section_defaults("l1_buffer")
            >>> print(l1_defaults["enable"])
            True
        """
        section_config = getattr(self, section, None)
        if section_config is None:
            return {}
        return asdict(section_config)

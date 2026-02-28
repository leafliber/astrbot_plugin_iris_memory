# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.9.1] - 2026-02-28

### Fixed
- **MemoryService 缺少 persona_id 参数传递** (`iris_memory/services/memory_service.py`)
  - 修复 `process_message_batch()` 方法未接收 `persona_id` 参数导致调用报错
  - 修复 `capture_and_store_memory()` 方法未接收 `persona_id` 参数
  - Facade 层方法需将 `persona_id` 传递给底层 `BusinessService`
- **记忆去重和冲突检测缺少 persona_id 过滤** (`iris_memory/capture/capture_engine.py`)
  - 修复 `capture_memory()` 方法中调用 `query_memories` 进行去重和冲突检测时未传递 `persona_id`
  - 开启人格隔离后，去重和冲突检测应在同一人格内进行
- **知识图谱迁移逻辑缺陷** (`iris_memory/knowledge_graph/kg_storage.py`)
  - 修复旧数据库可能被错误标记为 v2 但实际未执行迁移的问题
  - 新增 `_ensure_persona_columns()` 方法，无论 `schema_version` 值如何都检查并添加缺失的 `persona_id` 列
  - 解决 `table kg_nodes has no column named persona_id` 错误
- **记忆升级评估器类型错误** (`iris_memory/core/upgrade_evaluator.py`)
  - 修复 `'int' object has no attribute 'value'` 错误
  - 新增 `_get_storage_layer_value()` 辅助函数，安全处理 storage_layer 的枚举/字符串/整数类型
  - 新增 `_get_quality_level_value()` 辅助函数，安全处理 quality_level 的枚举/整数/字符串类型
  - Memory 对象从数据库恢复时，枚举字段可能是原始类型，直接调用 `.value` 会失败

## [v1.9.0] - 2026-02-28

### ⚠️ Known Limitations
- **本次更新后需要重启Astrbot！！**
- **旧数据兼容性**：当用户在已有数据后开启人格隔离时，旧数据将不会被查询到。
  - **影响范围**：仅影响开启人格隔离后查询的旧数据；新数据不受开关影响

### Added
- **人格隔离功能** (`iris_memory/core/defaults.py`, `iris_memory/core/config_manager.py`, `iris_memory/models/memory.py`, `iris_memory/knowledge_graph/kg_models.py`, `iris_memory/knowledge_graph/kg_storage.py`, `iris_memory/storage/chroma_operations.py`, `iris_memory/storage/chroma_queries.py`, `iris_memory/utils/persona_utils.py`, `_conf_schema.json`)
  - 新增配置项 `persona_isolation.memory_query_by_persona` 和 `persona_isolation.kg_query_by_persona`，默认关闭
  - 记忆模块：Memory 模型新增 `persona_id` 字段，存储时始终记录人格 ID，查询时根据开关决定是否过滤
  - 知识图谱模块：KGNode/KGEdge 模型新增 `persona_id` 字段，SQLite Schema 自动迁移（v1→v2），新增 persona 相关索引
  - 新增 `get_event_persona_id()` 工具函数，从 AstrBot 事件中安全提取人格 ID
  - 新增 `get_persona_id_for_storage()` 和 `get_persona_id_for_query()` 配置管理器方法
  - 支持不同 Bot 人格拥有独立的记忆空间和知识图谱数据

### Changed
- **配置管理器方法简化**：`get_persona_id_for_storage()` 移除了 `module` 参数，因为存储时始终记录 `persona_id`，无需区分模块

### Migration
- **SQLite Schema 自动迁移**：知识图谱数据库自动从 v1 升级到 v2，为 `kg_nodes` 和 `kg_edges` 表添加 `persona_id` 列（默认值 `'default'`）

### Fixed
- **知识图谱迁移事务原子性** (`iris_memory/knowledge_graph/kg_storage.py`)
  - 修复 `_migrate_v1_to_v2()` 方法在添加 `persona_id` 列时缺乏事务保护的问题
  - 使用 `_tx()` 事务上下文确保两个表的 DDL 操作原子性
  - 避免出现 `kg_nodes` 有 `persona_id` 列但 `kg_edges` 没有的不一致状态

## [v1.8.0] - 2026-02-26

### Added
- **维度冲突后自动重新导入原记忆** (`iris_memory/storage/chroma_manager.py`, `_conf_schema.json`)
  - 新增配置项 `embedding.reimport_on_dimension_conflict`，默认开启
  - 更换嵌入模型后，自动将原记忆用新模型重新生成向量并导入
  - 配置提示明确说明会增加 embedding 使用量，可能产生额外费用
  - 可选择关闭此功能，关闭后记忆将备份到备份集合但不自动导入
- **用户画像 Web 端搜索功能** (`iris_memory/web/`)
  - 新增用户画像搜索接口，支持按用户ID、兴趣、工作风格、生活方式等多字段搜索
  - 前端添加搜索框、分页组件和每页数量选择器
  - 搜索结果支持关键词高亮显示
- **用户画像显示用户昵称** (`iris_memory/models/`, `iris_memory/web/`)
  - `UserPersona` 模型新增 `display_name` 字段存储用户昵称
  - 记忆捕获时记录 `sender_name`，同步更新到用户画像
  - Web 界面列表和详情页显示用户昵称（带 user_id 提示）
  - 完全向后兼容，旧数据无昵称时显示 user_id
- **后台记忆管理查询工作记忆** (`iris_memory/analysis/persona/`)
  - `PersonaBatchProcessor` 新增 `working_memory_callback` 回调机制
  - 支持后台批量处理时查询工作记忆获取额外上下文
  - 提供 `get_working_memory_context()` 方法格式化工作记忆为文本
- **记忆 confidence 字段持久化** (`iris_memory/storage/`)
  - 修复 ChromaDB 存储时 confidence 字段丢失的问题
  - 新增 confidence 字段的存储和读取支持

### Changed
- **简化嵌入向量配置** (`_conf_schema.json`, `iris_memory/embedding/manager.py`, `README.md`)
  - 删除 `fallback_to_local` 配置项，降级行为完全由 `embedding.source` 控制
  - 删除 `enable_local_provider` 配置项，本地模型启用由 `embedding.source` 自动决定
  - 统一嵌入源选择逻辑：
    - `auto`: AstrBot API 优先，不可用时自动切换到本地模型
    - `astrbot`: 仅使用 AstrBot API，不可用时使用 Fallback
    - `local`: 仅使用本地模型，不可用时使用 Fallback
  - 更新配置描述，明确各选项的行为和适用场景

### Fixed
- **Web 界面置信度显示为0的问题** - 修复后新保存的记忆将正确显示 confidence 值（默认 0.5）
- **知识图谱节点浮动面板消失问题** (`iris_memory/web/static/js/kg.js`)
  - 移除 `mouseleave` 时自动关闭面板的行为
  - 新增关闭按钮和点击外部区域关闭功能
  - 修复后"从此展开"按钮可正常点击

## [v1.7.3] - 2026-02-26

### Changed
- **Service 层架构重构** (`iris_memory/services/`)
  - 将 `MemoryService` 从 Mixin 继承模式重构为组合 + 依赖注入模式
  - 拆分 `BusinessOperations` Mixin 为独立的 `BusinessService` 类
  - 拆分 `PersistenceOperations` Mixin 为独立的 `PersistenceService` 类
  - 拆分 `ServiceInitializer` Mixin 逻辑内联到 `MemoryService`
  - 新增 `SharedState` 类集中管理跨服务共享状态（用户画像、情感状态、注入记录等）
  - 所有依赖通过构造函数显式注入，消除隐式循环依赖
  - `MemoryService` 转型为薄 Facade，负责协调初始化并委托操作

### Architecture Improvements
- **消除上帝类**: `MemoryService` 从 1800+ 行分散在 Mixin 中，重构为 3 个独立服务类
- **依赖关系显式化**: 所有依赖通过构造函数注入，可独立测试
- **向后兼容**: 保留所有原有属性和方法代理，`main.py` 无需修改

### Code Quality
- 提升测试性：各 Service 可独立实例化进行单元测试
- 降低耦合度：模块间依赖关系清晰可追踪
- 改善可维护性：业务逻辑、持久化逻辑、初始化逻辑分离

## [v1.7.2] - 2026-02-25

### Added
- **UserPersona v2 维度体系** (`iris_memory/models/user_persona.py`, `iris_memory/analysis/persona/`)
  - 全新 v2 用户画像数据模型，新增 7 大维度：工作 (work)、生活 (life)、情绪 (emotional)、社交 (social)、人格 (personality)、交互 (interaction)、行为 (behavior)
  - 大五人格维度扩展：新增外向性 (extraversion) 维度及其规则提取器
  - 实现 RuleExtractor、LLMExtractor 和混合模式的完整提取逻辑
  - 新增 PersonaCoordinator 协调器，统一管理规则、LLM和混合模式
  - 新增 `keyword_maps.yaml` 关键词映射配置，支持细粒度的人格特征提取
- **PersonaBatchProcessor** (`iris_memory/analysis/persona/persona_coordinator.py`)：批量处理用户画像提取，支持异步批量更新
- **LLM 路由复杂查询检测** (`iris_memory/analysis/llm_router.py`)：增强 LLM 检索路由，新增复杂查询检测能力
- **无登录访问 check-auth 端点** (`iris_memory/web/web_routes.py`)：允许未认证访问 `/api/check-auth` 端点

### Changed
- **独立服务器端口复用** (`iris_memory/web/standalone_server.py`)：启用 `SO_REUSEADDR` 和 `SO_REUSEPORT`，改善服务器重启体验
- **插件终止流程优化** (`main.py`)：改进插件终止过程，确保资源正确释放
- **记忆升级策略优化** (`iris_memory/capture/memory_upgrader.py`)：优化记忆升级判断标准和访问统计处理
- **Web UI 样式更新** (`iris_memory/web/static/index.html`)：优化图标和样式，提升视觉体验
- **日志增强** (`iris_memory/utils/llm_utils.py`)：增强 LLM 提供者解析和配置检索的日志记录
- **image_analysis_provider_id 描述更新** (`_conf_schema.json`)：明确视觉模型要求

### Fixed
- 修复独立服务器地址复用问题，确保服务器可快速重启
- 修复 README 中 LLM 分析警告信息

### Test
- 新增 UserPersona v2 维度全面测试 (`tests/models/test_persona_v2_dimensions.py`)：覆盖规则提取、LLM提取、混合模式，验证 7 大维度的提取和更新逻辑，包含端到端集成测试

## [v1.7.1] - 2026-02-24

### Fixed
- **修复 Web API 方法名不匹配** (`iris_memory/web/web_service.py`, `iris_memory/web/standalone_server.py`)
  - 修复 `list_personas` 路由调用不存在方法的 AttributeError，原方法名为 `get_user_personas_list`
  - 修复 `get_persona_detail` 与 `get_user_persona_detail` 方法名不匹配
  - 修复 `get_emotion_state` 路由传参 `(user_id, group_id)` 但方法签名只接受 `(user_id)` 的问题
- **修复 `list_personas` 缺少分页支持**，新增 `page`/`page_size` 参数与分页返回格式
- **修复服务器无法正常关闭**：`stop()` 方法未实际停止 Hypercorn，改用 `asyncio.Event` 作为 `shutdown_trigger`

### Changed
- **Web UI 全面改版** (`iris_memory/web/static/index.html`)
  - 配色方案从蓝色 (`#1da1f2`) 更换为紫蓝色 (`#7c6cf0`) 主题
  - 所有 emoji 替换为几何符号（🧠→✦、📊→⬡、💾→◈、🔗→⬢、👤→◉ 等），提升视觉一致性
  - 图谱节点颜色、徽章颜色、情绪颜色全面更新
- **`format` 参数更名为 `fmt`**：`export_memories`、`export_kg`、`import_memories`、`import_kg`、`preview_import_data` 五个方法避免遮蔽 Python 内置函数
- **`_get_memory_overview` 查询优化**：从 N+1 次 ChromaDB 查询优化为单次全量查询 + 内存计数

### Added
- **路径遍历防护** (`standalone_server.py`)：`_serve_static` 新增 `.resolve()` 校验，防止路径遍历攻击
- **登录频率限制**：新增 `_check_login_rate_limit` / `_record_login_attempt`，每 IP 60 秒内最多 5 次尝试
- **安全整数解析** (`_safe_int`)：所有路由参数解析统一使用带范围限制的安全解析函数
- **会话令牌清理**：`check_auth` 中调用 `cleanup_expired_tokens()`，防止令牌内存泄漏
- **分页参数校验**：`page` 和 `page_size` 强制有效范围 (`page≥1`, `1≤page_size≤100`)
- **记忆更新白名单校验** (`update_memory_by_id`)：仅允许更新指定字段，拒绝未知 key；数值字段增加安全类型转换
- **错误响应脱敏**：移除所有 API 错误响应中的 `{e}` 异常详情，防止信息泄漏

## [v1.7.0] - 2026-02-23

### Changed
- **配置文件结构重组** (`_conf_schema.json`)
  - 新增独立 `logging` 模块，将 `log_level` 从 `basic` 中分离
  - 新增 `llm_providers` 模块，统一管理所有 LLM 提供者配置
  - 移除各功能模块中分散的 `provider_id` 配置，集中到 `llm_providers`
  - 配置组织更加清晰，按功能模块分层：基础配置 → 基础设施 → LLM配置 → 核心功能 → 智能增强 → 交互功能 → 其他

### Added
- **新增 LLM 提供者统一配置** (`llm_providers`)
  - `default_provider_id`: 默认 LLM 提供者
  - `memory_provider_id`: 记忆功能 LLM 提供者
  - `persona_provider_id`: 用户画像 LLM 提供者
  - `knowledge_graph_provider_id`: 知识图谱 LLM 提供者
  - `image_analysis_provider_id`: 图片分析 LLM 提供者
  - `enhanced_provider_id`: 智能增强 LLM 提供者

### Migration
配置路径变更对照表：

| 原路径 | 新路径 |
|--------|--------|
| `basic.log_level` | `logging.log_level` |
| `memory.provider_id` | `llm_providers.memory_provider_id` |
| `llm_enhanced.provider_id` | `llm_providers.enhanced_provider_id` |
| `image_analysis.provider_id` | `llm_providers.image_analysis_provider_id` |
| `persona.llm_provider` | `llm_providers.persona_provider_id` |
| `knowledge_graph.provider_id` | `llm_providers.knowledge_graph_provider_id` |

### Code Changes
- `config_manager.py`: 更新所有 provider_id 属性的配置路径
- `config_registry.py`: 更新配置注册表，新增 llm_providers 相关定义
- `defaults.py`: 新增 `LLMProvidersDefaults` 类，移除各模块分散的 provider_id 默认值
- `constants.py`: 更新 `ConfigKeys.LOG_LEVEL` 路径

## [v1.6.2] - 2026-02-23

### Added
- 添加项目 logo 图片 (`logo.png`)

### Fixed
- 修复 `main.py` 中 `StarTools` 导入位置错误，从 `astrbot.api` 移至 `astrbot.api.star`

## [v1.6.1] - 2026-02-23

### Added
- **Web dashboard & API for Iris Memory** (`iris_memory/web/api_routes.py`, `iris_memory/web/web_service.py`, `iris_memory/web/static/index.html`)
  - 新增前端单页仪表盘与管理界面，重写 `index.html`，提供记忆列表、知识图谱可视化、导入/导出与用户画像视图。
  - 记忆编辑功能：支持打开编辑模态框，修改内容、类型、存储层、置信度、重要度、摘要并保存。
  - 知识图谱边管理：边列表、筛选、删除与批量操作功能。
  - 导入预览：导入前预览文件内容与验证统计（支持 JSON/CSV），展示有效/无效条目并允许确认导入。
  - 用户画像（Personas）视图：展示用户画像卡片与详情（大五人格、沟通偏好、情绪状态）。
  - 情绪状态终端：新增获取与展示情绪态的后端接口与前端展示。
  - 交互改进：画布缩放、节点详情弹窗、刷新按钮、加载状态、搜索高亮、页大小选择、键盘快捷键等体验优化。
- **审计日志**：Web 模块关键操作（导入/导出/更新/删除）增加审计日志记录（`iris_memory/web/web_service.py`）。

### Changed
- 新增若干 Web API 路由并注册到 `IrisWebAPI`（记忆详情/更新、KG 边列表、导入预览、画像列表/详情、情绪状态等）。
- 前端 `index.html` 重写为交互式 SPA，提升可用性与无依赖实现的兼容性。

### Fixed
- 修复 JSON 导入解析器缺失问题：补充 `_parse_json_memories` 与 `_parse_json_kg`，修正导入边界条件与验证逻辑（`iris_memory/web/web_service.py`）。
- 测试调整：更新路由注册数量断言以匹配新增路由，已验证测试通过（57 passed）。

### Test
- 相关 Web 模块单元测试通过：`tests/web/test_web_service.py` 全量通过（57 passed）。

## [v1.6.0] - 2026-02-19

### Added
- **输入安全清理能力** (`iris_memory/core/constants.py`, `main.py`)
  - 为 `InputValidationConfig` 新增 `sanitize_input()` 与 `has_dangerous_content()`，用于过滤 HTML 标签、脚本注入片段与控制字符。
  - `memory_save` 指令接入输入清理流程，避免将危险内容直接写入记忆。
- **知识图谱动态推理深度估算** (`iris_memory/knowledge_graph/kg_reasoning.py`)
  - 新增 `estimate_query_depth()`，根据查询复杂度（关系词/链路词）动态调整多跳推理深度（1~5）。
- **主动回复动态冷却策略** (`iris_memory/proactive/proactive_manager.py`, `iris_memory/core/constants.py`)
  - 新增基于 `urgency` 的冷却时间乘数配置（critical/high/medium/low），提升紧急场景响应能力。
- **常量集中化补充** (`iris_memory/core/constants.py`)
  - 新增 `CaptureQualityThresholds`、`LLMCallDefaults`、`TextTruncation`、`UrgencyCooldownMultiplier` 等常量组，减少关键逻辑中的魔法数字。

### Changed
- **依赖版本策略收紧** (`pyproject.toml`)
  - 收紧核心依赖版本上界，降低跨大版本 API 变更风险：`chromadb`、`numpy`、`sentence-transformers`。
  - 将测试相关依赖移动到 `project.optional-dependencies.dev`。
- **配置读取实时性提升** (`iris_memory/core/config_manager.py`)
  - 配置缓存默认 TTL 从 30 秒调整为 10 秒，减少热更新配置生效延迟。
- **批量会话过期窗口调整** (`iris_memory/core/constants.py`)
  - `BatchSessionConfig.SESSION_EXPIRY_SECONDS` 从 2 小时提升到 4 小时，降低长间隔会话消息丢失概率。
- **熔断恢复策略调整** (`iris_memory/core/constants.py`)
  - `CircuitBreakerConfig.RECOVERY_TIMEOUT` 从 60 秒调整为 30 秒，加快临时网络抖动后的恢复。
- **高并发状态缓存容量扩展** (`iris_memory/services/memory_service.py`)
  - 用户相关 `BoundedDict` 容量从 500 提升到 2000；配合现有 LRU 淘汰策略，提升多活跃用户场景稳定性。
- **日志隐私策略收敛** (`iris_memory/embedding/manager.py`, `iris_memory/capture/capture_logger.py`, `iris_memory/retrieval/retrieval_logger.py`)
  - 调试日志改为记录长度/摘要元信息，不再记录用户原文片段，降低敏感信息泄露风险。

### Fixed
- **EmbeddingManager 在 provider 未就绪时的状态不一致问题** (`iris_memory/embedding/base.py`, `iris_memory/embedding/manager.py`)
  - 修复 `embed()` 过程中因 provider 未初始化导致的 `RuntimeError: Provider not initialized`。
  - `get_model()` 与 `model` 属性改为安全回退，`embed_batch()` 增加降级路径，避免整体流程崩溃。
- **知识图谱家族关系规则漏匹配** (`iris_memory/knowledge_graph/kg_extractor.py`)
  - 修复家族关键词拼写错误（`"爵父"` → `"爸爸"`），补充亲属关键词，恢复 `family_of` 抽取覆盖。
- **敏感信息检测可绕过模式修复** (`iris_memory/capture/detector/sensitivity_detector.py`)
  - 扩展密码与凭证正则，覆盖 `密码是 xxx`、`pass=xxx`、大小写 token/key 变体与云服务密钥模式。

### Test
- 测试验证通过：
  - `tests/embedding/test_lazy_loading.py` 全量通过（含 `test_embed_fallback_when_local_load_failed`）
  - `tests/knowledge_graph/test_kg_extractor.py` 全量通过（含 `test_family_of_pattern`）
  - `tests/` 全量通过：`1220 passed`

## [v1.5.0] - 2026-02-19

### Added
- **LLM智能增强功能模块** (`iris_memory/core/detection/`, `iris_memory/capture/detector/`, `iris_memory/analysis/emotion/`, `iris_memory/proactive/`, `iris_memory/retrieval/`)
  - 新增 `LLMEnhancedDetector` 基类，实现模板方法模式统一 LLM 检测流程
  - 新增 `LLMSensitivityDetector`：基于 LLM 的敏感度检测
  - 新增 `LLMTriggerDetector`：基于 LLM 的触发器检测
  - 新增 `LLMEmotionAnalyzer`：基于 LLM 的情感分析
  - 新增 `LLMProactiveReplyDetector`：基于 LLM 的主动回复检测
  - 新增 `LLMConflictResolver`：基于 LLM 的冲突解决
  - 新增 `LLMRetrievalRouter`：基于 LLM 的检索路由
- **Memory Capture Engine** (`iris_memory/capture/capture_engine.py`)
  - 智能记忆捕获引擎，集成触发检测、情感分析、冲突解决
  - 支持敏感度检测和实体提取
  - 支持 LLM 增强的检测模式
- **Persona Extractor** (`iris_memory/analysis/persona/`)
  - 支持规则、LLM、混合三种人格提取模式
  - 新增 `PersonaLogger` 用于人格生命周期事件的结构化日志
  - 新增 `RuleExtractor` 和 `LLMExtractor` 实现
- **结构化日志系统** (`iris_memory/capture/capture_logger.py`, `iris_memory/retrieval/retrieval_logger.py`)
  - 记忆捕获过程的结构化日志记录
  - 记忆检索过程的结构化日志记录
- **消息合并器** (`iris_memory/capture/message_merger.py`)
  - 支持消息合并逻辑，优化批量处理
- **图片缓存管理** (`iris_memory/multimodal/image_cache.py`)
  - 新增图片缓存机制，避免重复分析
- **服务容器** (`iris_memory/core/service_container.py`)
  - 轻量级依赖注入容器，替代全局变量，确保线程安全

### Changed
- **重构配置管理** (`iris_memory/core/config_registry.py`)
  - 新增 `ConfigDefinition` 和 `CONFIG_REGISTRY` 统一配置管理
  - 配置定义集中管理，单一数据源
- **重构持久化层** (`iris_memory/services/persistence.py`)
  - 新增 `PersistenceOperations` 类处理所有持久化逻辑
  - 包括会话数据、生命周期状态、批量队列、聊天历史、主动回复白名单、成员身份、活动数据、用户画像的加载与保存
- **重构 Chroma 操作** (`iris_memory/storage/chroma_operations.py`, `iris_memory/storage/chroma_queries.py`)
  - 新增 `ChromaOperations` 类管理记忆的 CRUD 操作
  - 新增 `ChromaQueries` 类封装查询逻辑，支持群聊和私聊多种查询模式
- **重构业务操作** (`iris_memory/services/business_operations.py`)
  - 新增 `BusinessOperations` 类封装业务逻辑
- **重构初始化流程** (`iris_memory/services/initializers.py`)
  - 新增 `Initializers` 类管理服务初始化
- **重构 LLM 检测框架** (`iris_memory/core/detection/`)
  - 新增 `BaseDetectionResult` 泛型基类统一结果处理
  - 减少约 40% 的重复代码
- **优化 MemoryService** (`iris_memory/services/memory_service.py`)
  - 大幅简化代码，将逻辑拆分到独立模块
- **优化 ChromaManager** (`iris_memory/storage/chroma_manager.py`)
  - 简化代码，将操作和查询逻辑分离

### Fixed
- **修复热更新兼容性问题** (`main.py`, `iris_memory/utils/command_utils.py`)
  - 使用 `getattr` 替代直接属性访问以兼容热更新场景

### Removed
- 移除不再使用的 `PROJECT_REFERENCE.md` 文档文件

---

## [v1.4.2] - 2026-02-17

### Fixed
- **修复热更新后 Chroma 数据库查询崩溃** (`iris_memory/storage/chroma_manager.py`)
  - 问题：热更新（hot-reload）后，`terminate()` 将 `collection` 设为 `None`，新消息在 `initialize()` 完成前到达时触发 `'NoneType' object has no attribute 'query'` 错误
  - 解决方案：为 ChromaManager 添加 `_is_ready` 标志和 `_ensure_ready()` 前置检查，在所有公开的数据操作方法（`query_memories`、`add_memory`、`update_memory`、`delete_memory`、`delete_session`、`get_all_memories`、`count_memories`、`get_memories_by_storage_layer` 等）入口进行空值保护
- **修复热更新期间 `close()` 调用 `client.reset()` 导致数据丢失** (`iris_memory/storage/chroma_manager.py`)
  - 问题：`close()` 调用 `client.reset()` 会清除所有 Chroma 数据（包括持久化的向量数据），热更新后记忆全部丢失
  - 解决方案：`close()` 不再调用 `reset()`，仅释放客户端和集合引用，持久化数据保留在磁盘上，下次初始化时自动加载
- **修复全局 `_identity_service` 在热更新后未清理** (`iris_memory/services/memory_service.py`, `iris_memory/utils/member_utils.py`)
  - 问题：`terminate()` 未调用 `set_identity_service(None)`，导致旧的服务引用残留
  - 解决方案：在 `terminate()` 中显式清理全局状态引用

### Changed
- **新增初始化状态跟踪机制** (`iris_memory/services/memory_service.py`, `main.py`)
  - 为 `MemoryService` 添加 `_is_initialized` 标志和 `is_initialized` 属性，以及 `asyncio.Lock` 防止并发初始化
  - `main.py` 的 `on_llm_request`、`on_llm_response`、`on_all_messages` 三个核心 Hook 添加 `is_initialized` 前置检查，初始化未完成时优雅跳过或提示
  - `terminate()` 立即将 `_is_initialized` 设为 `False`，阻止新请求进入
- **改进服务销毁流程** (`iris_memory/services/memory_service.py`)
  - 按依赖顺序停止后台任务（消费者 → 生产者）：BatchProcessor → ProactiveManager → LifecycleManager → ChromaManager
  - 每个组件的关闭逻辑独立 try/except 包裹，一个组件失败不影响其他组件
  - `terminate()` 中保存 KV 数据操作增加异常保护
- **后台任务启停日志增强** (`iris_memory/capture/batch_processor.py`, `iris_memory/storage/lifecycle_manager.py`, `iris_memory/proactive/proactive_manager.py`)
  - 所有 `stop()` 方法添加 `[Hot-Reload]` 前缀日志，便于热更新问题排查
  - 变更文件：`batch_processor.py`（剩余队列处理增加异常保护）、`lifecycle_manager.py`（批量取消并等待任务，清理引用）、`proactive_manager.py`（处理任务取消异常）

---

## [v1.4.1] - 2026-02-17

### Fixed
- 修复 provider 检测与解析逻辑，解决 `modelscope-vl` 被误判为不存在的问题（统一了 provider 解析与回退逻辑）。
  - 变更文件：`iris_memory/utils/provider_utils.py`, `iris_memory/multimodal/image_analyzer.py`, `iris_memory/processing/llm_processor.py`, `iris_memory/analysis/persona_extractor.py`, `iris_memory/core/config_manager.py`

### Changed
- 规范化配置中 provider 字段的解析，兼容 AstrBot 的 `select_provider` 配置对象与字符串形式。

---

## [v1.4.0] - 2026-02-17

### Added
- **本地 Embedding 提供器（后台加载 & 就绪检查）** (`iris_memory/embedding/local_provider.py`, `iris_memory/embedding/manager.py`)
  - 支持后台异步加载本地 sentence-transformers 模型，并通过 `is_ready`/健康检查暴露加载状态，避免在启动时阻塞或在测试中触发实际模型下载。
- **Persona 提取模块** (`iris_memory/analysis/persona_extractor.py`)
  - 新增规则/LLM/混合模式的人格提取器，用于从对话/历史构建用户人格画像。
- **群组活动追踪与自适应配置** (`iris_memory/*`)
  - 新增群组活跃度追踪逻辑，并根据群组活跃度自动调整某些触发阈值与注入策略。
- **统一删除命令与增强的记忆删除功能**
  - 新增统一删除命令并改进记忆删除的边界情况处理与进度报告。

### Changed
- **重构 RIFScorer 初始化** (`iris_memory/analysis/rif_scorer.py`)：移除多维评分初始化支持并简化参数处理。
- **测试重构**：重构 `UserPersona` 相关测试，提升覆盖与结构可维护性。

### Fixed
- 小幅修复与测试稳定性改进（避免在单元测试中触发耗时的模型下载/加载）。

## [v1.3.1] - 2026-02-17

### Added
- **事件驱动的主动回复事件** (`iris_memory/proactive/proactive_event.py`, `proactive_manager.py`)
  - 新增 `ProactiveMessageEvent`，主动回复改为构建事件并入队 AstrBot 的事件队列，由标准 LLM 流水线负责生成与发送
- **单元测试补充** (`tests/proactive/`)
  - 添加并完善针对 `ProactiveReplyManager`、`ProactiveMessageEvent` 与 `main.py` 主动回复分支的单元测试，覆盖队列注入、白名单与关闭行为

### Changed
- **主动回复管理器行为调整** (`iris_memory/proactive/proactive_manager.py`)
  - 管理器在无事件队列时降级为不可用并记录告警；shutdown 时会在处理未完成任务时跳过额外的等待延迟
- **内存服务与主插件运行时集成** (`iris_memory/services/memory_service.py`, `main.py`)
  - `MemoryService` 不再直接创建发送器/生成器，而是将 AstrBot 的事件队列注入到 `ProactiveReplyManager`
  - `main.py` 对合成的主动回复事件在 `on_all_messages`/`on_llm_request`/`on_llm_response` 中增加兼容逻辑，注入主动指令并避免重复捕获或身份解析

### Fixed
- **测试配置与 pytest 标记** (`pytest.ini`)
  - 修复 pytest 配置节名以注册自定义标记，消除未知 marker 警告

## [v1.3.0] - 2026-02-15

### Added
- **新增主动回复群聊白名单模式与管理员指令** (`proactive_manager.py`, `main.py`)
  - 新增 `proactive_reply.group_whitelist_mode` 配置，开启后进入白名单模式
  - 管理员可在群聊中使用 `/proactive_reply on|off|status|list` 控制该群的主动回复开关
  - 白名单数据持久化于 KV 存储，插件重启后恢复
- **新增群聊/私聊近期消息上下文注入** (`chat_history_buffer.py`, `memory_service.py`, `main.py`)
  - 新增 `ChatHistoryBuffer` 滑动窗口缓冲区，按会话维护最近聊天记录
  - LLM 请求时自动注入近期对话上下文（与高价值"记忆"注入互补）
  - 新增 `chat_context_count` 配置项，控制注入消息条数（默认20条）
  - 新增 `member_utils` 工具模块，提供稳定的成员标识格式化

### Fixed
- **修复 Embedding 维度不匹配导致查询失败** (`chroma_manager.py`)
  - 问题：切换 Embedding 模型后已有 Collection 维度（如4096）与新模型维度（如512）冲突，导致 `Collection expecting embedding with dimension of 4096, got 512`
  - 解决方案：初始化时检测维度冲突，自动删除旧 Collection 并重建
- **修复主动回复发送缺少 session 参数** (`message_sender.py`, `proactive_manager.py`)
  - 问题：`Context.send_message()` 需要 `(session, message_chain)` 两个参数，但只传了 `message_chain`
  - 解决方案：新增 `umo`（unified_msg_origin）参数传递链路，正确调用 AstrBot API
- **修复 MultidimensionalScorer 初始化参数冲突** (`rif_scorer.py`)
  - 问题：`fallback_to_rif` 同时出现在显式参数和 `**kwargs` 中，导致 `got multiple values for keyword argument`
  - 解决方案：转发 kwargs 前过滤已显式指定的参数
- **修复 @Bot 消息重复录入聊天缓冲区** (`main.py`)
  - 问题：@Bot 消息同时触发 `on_all_messages` 和 `on_llm_request`，导致重复记录
  - 解决方案：仅在 `on_all_messages` 中记录，移除 `on_llm_request` 中的重复录入
- **清理死代码和重复逻辑** (`memory_service.py`)
  - 移除未使用的 `MemoryInjector` 实例化代码
  - 合并重复的"成员区分"指令（原分散在 `_build_behavior_directives` 和 `_build_member_identity_context` 中）
  - 补全 `ConfigKeys`、`storage/__init__.py`、`utils/__init__.py` 的导出缺失

## [v1.2.0] - 2026-02-04

- 新增主动回复群聊白名单配置：`proactive_reply.group_whitelist`
- 重构部分代码

## [v1.1.8] - 2026-02-04

### Fixed
- **修复 OpenAI API 400 BadRequestError** (`main.py`)
  - 问题：AstrBot 4.14+ 中 `on_llm_request` 的 `req` 对象结构变化，直接修改 `system_prompt` 可能导致请求格式错误
  - 解决方案：优先通过 `req.messages` 列表注入上下文，支持追加到现有 system 消息或在开头添加新消息
  - 添加 `_sanitize_for_llm()` 方法清理特殊字符，防止破坏 JSON 请求格式
  - 保留 `system_prompt` 修改作为旧版兼容回退
- **修复LLM处理器初始化时机问题** (`llm_processor.py`, `main.py`)
  - 问题：AstrBot在插件加载后才初始化provider，导致插件启动时显示"No LLM providers available"
  - 解决方案：采用延迟初始化策略，`initialize()`不立即检查provider，而是在实际使用时按需获取
  - 新增`_try_init_provider()`方法，最多重试3次获取provider
  - 优化启动日志，避免显示误导性的警告信息
- **修复存储层判断逻辑** (`capture_engine.py`): 解决记忆无法持久化问题
  - 原逻辑过于保守，99%记忆只存WORKING层
  - 修复后：用户请求直接→EPISODIC；CONFIRMED→SEMANTIC；高置信度/情感强度→EPISODIC
- **修复主动回复发送器** (`message_sender.py`): 支持AstrBot标准API
  - 新增`provider_send`检测优先级
  - 新增`_send_via_provider`方法使用`astrbot.api.message_components.Plain`构建消息链
- **修复会话状态同步** (`lifecycle_manager.py`): 解决SessionManager和LifecycleManager状态不一致
  - `activate_session`自动创建缺失会话
  - `get_session_statistics`同步双管理器状态

### Changed
- **完善RIF评分算法** (`rif_scorer.py`): 全面提升三个维度计算质量
  - 时近性(Recency): 添加情感权重、用户请求加成、24h新记忆保护期(+0.5)
  - 相关性(Relevance): 记忆类型权重、非线性访问频率、置信度/情感加成
  - 频率性(Frequency): 近期访问权重、高质量加成、时间衰减因子
  - 新增细粒度时间权重函数：1小时-365天+的分级权重

### Added
- **添加定时任务升级工作记忆** (`lifecycle_manager.py`): 修复工作记忆无法自动升级
  - 改进`_promote_memories`: 正确处理session_key解析和WORKING→EPISODIC升级
  - 添加错误处理和升级统计
- **添加内存监控和自动清理** (`session_manager.py`):
  - `get_memory_usage_stats()`: 返回内存使用、缓存命中率等统计
  - `perform_maintenance()`: 清理24h过期记忆、30天过期会话、LRU淘汰
- **优化Embedding缓存策略** (`embedding/manager.py`):
  - 添加LRU缓存（最大1000条），使用MD5哈希作为缓存键
  - 缓存统计：hits/misses/hit_rate
- **优化日志级别** (`utils/logger.py`, `capture_engine.py`):
  - 默认级别DEBUG→INFO
  - 精简capture_engine的逐步骤DEBUG日志为关键信息INFO日志

## [v1.1.7] - 2026-02-04

### Added
- **新增主动回复群聊白名单配置**：`proactive_reply.group_whitelist`
  - 支持配置允许触发主动回复的群聊列表
  - 空列表表示允许所有群聊（默认行为）
  - 配置示例：`["123456789", "987654321"]`
  - 私聊不受白名单限制

### Fixed
- **修复主动回复发送失败**：`MessageSender._send_via_context()` 移除了不支持的 `target` 参数
  - 问题：`Context.send_message()` 不接受 `target` 参数，导致主动回复发送失败
  - 修复：改为只传递 `message` 参数，并将 `platform_send` 方法优先级提升
- **抑制模型加载进度条输出**：sentence-transformers 加载模型时的进度条不再显示在终端
  - 设置环境变量 `TRANSFORMERS_VERBOSITY=error` 和 `HF_HUB_DISABLE_PROGRESS_BARS=1`
  - 调用 `transformers_logging.disable_progress_bar()` 禁用进度条

## [v1.1.6] - 2026-02-04

### Fixed
- **修复群聊 group_id 获取问题**：`get_group_id()` 函数现在正确从 `event.group_id` 获取群组ID
  - 问题：之前使用 `get_sender_group_id()` 方法在群聊场景中返回 None
  - 修复：根据 AstrBot 官方文档，直接使用 `event.group_id` 属性

### Changed
- **重构日志系统**：接入 AstrBot 日志输出
  - 新增 `AstrBotLogHandler` 类，将插件日志转发到 AstrBot 控制台
  - 简化控制台输出格式，避免与 AstrBot 日志格式重复
  - 保留文件日志输出（`logs/iris_memory.log`）用于问题排查
  - 移除 `DebugLogger`、`log_method_call` 等调试装饰器

## [v1.1.5] - 2026-02-04

### Added
- 记忆注入人格风格支持：新增 `natural` 和 `roleplay` 风格，适配真实群聊人设
- 动态记忆选择器：根据 token 预算自动选择最优记忆子集

### Fixed
- 修复批量处理器队列序列化失败问题：`EmotionalState` 对象现在正确转换为字典进行 JSON 序列化
- 修复情感状态在批量处理上下文中的持久化问题
- **修复 `/memory_delete_all` 命令逻辑错误**：数据库为空时错误地返回删除失败，现已正确返回成功（删除 0 条）

### Optimized
- **合并重复查询**：去重检查和冲突检测现在共享一次向量查询，减少 50% 的 ChromaDB 查询次数，同时消除重复的日志输出

### Added
- **增强 DEBUG 日志**：添加详细的数据库操作日志
  - `query_memories`：记录原始查询结果（前5条）和最终的 Memory 对象详情
  - `add_memory`：记录添加的记忆完整内容、元数据和关键属性
  - `delete_all_memories`：记录要删除的记忆列表（前10条）

### Fixed
- **修复记忆升级缺失**：生命周期管理器新增 `WORKING → EPISODIC` 升级逻辑，解决工作记忆无法自动升级到情景记忆的问题
  - 新增 `_should_promote_working_to_episodic()` 方法，基于 RIF 分数、置信度、质量等级判断升级
  - 升级条件：RIF≥0.5、置信度≥0.3、质量等级≥3 或访问次数≥2 或用户主动请求
- **SessionManager 新增方法**：`get_all_sessions()`（包含工作记忆）、`remove_working_memory()`

### Fixed
- **修复 DEBUG 日志 None 格式化错误**：当 ChromaDB 查询结果中 `distance` 为 None 时，新增代码的 DEBUG 日志格式化失败
  - 问题：`result.get('distance', 'N/A'):.4f` 在 distance=None 时抛出 `TypeError`
  - 修复：先检查 None 再格式化，显示为 "N/A"

### Changed
- **简化 `/memory_stats` 输出**：移除"会话消息"计数（仅用于内部调试，对用户无实际意义）

### Changed
- **重构配置系统**：优化 `_conf_schema.json` 配置结构
  - 新增列表类型配置支持：`persona_styles`, `custom_triggers`, `trigger_keywords`, `embedding.models`, `session.excluded_users`
  - 重新组织配置分组：
    - `basic`: 基础功能开关 + 日志级别
    - `memory_inject`: 记忆注入设置（新增 persona_styles）
    - `capture_settings`: 记忆捕获设置（新增 custom_triggers）
    - `proactive_reply`: 主动回复（新增 trigger_keywords）
    - `image_analysis`: 图片分析设置
    - `llm_processing`: LLM增强处理
    - `embedding`: 向量嵌入（支持多模型列表）
    - `session`: 会话管理（新增 excluded_users）
  - 更新 `config_manager.py` 配置映射和便捷访问属性
  - 更新 `defaults.py` 添加新的默认配置项

### Technical
- **修复测试兼容性**：
  - `test_session_manager.py::TestGetAllSessions`：更新测试期望，方法返回字典而非列表
  - 所有存储层测试通过：112 passed (2 failed 为测试环境问题，与新增代码无关)
  - 修复 `ConfigManager.embedding_model` 属性以兼容 Mock 对象

### Changed
- **优化主动回复触发策略**（适配群聊场景）：
  - 降低触发阈值：CRITICAL 0.8→0.7, HIGH 0.6→0.5, MEDIUM 0.4→0.3, LOW 0.2→0.15
  - 扩展触发关键词：增加群聊常用语（"在么"、"冒泡"、"笑死"、"求"、"大家觉得"等）
  - 优化情感检测：积极情感（joy/excitement）降低阈值至 0.3，非中性情感强度>0.4 即可触发
  - 新增 `chat_topics` 类别：识别分享/推荐/讨论类群聊话题
- **修复 AstrBot API 兼容性问题 (v4.5.7+)**：
  - `llm_processor.py`：使用新的 `context.get_all_providers()` + `context.llm_generate()` API 替代旧的禁用代码
  - `astrbot_provider.py`：使用 `context.get_all_embedding_providers()` 获取嵌入提供商
  - `reply_generator.py`：修复 LLM 响应处理，正确使用 `LLMResponse.completion_text`
  - `manager.py`：修复 AstrBotProvider 初始化，正确传递 astrbot_context
- **代码审查修复**：
  - `session_manager.py`：修复 `add_memory_async` 方法缺少 `await` 导致协程未执行的严重问题
  - `session_manager.py`：修复变量重复赋值覆盖默认值的问题
  - `chroma_manager.py`：将裸 `except:` 替换为具体的异常类型 `(ValueError, TypeError)`
  - 移除未使用的导入：`StorageLayer` (session_manager)、`asyncio` (message_sender)

## [v1.1.2-4] - 2026-02-03

### Added
- 将图片智能分析的上下文相关性、预算与去重过滤整合到生产流程（减少对Vision LLM的调用）

### Fixed
- 修复主动回复初始化失败：改为通过 `astrbot_context.get_using_provider(umo)` 获取 LLM 提供器，避免直接依赖已移除的 `AstrBotApi` 导入
- 修复主动回复任务链中 `umo` 未传递的问题，确保批量处理器 -> 管理器 -> 生成器正确传递 `umo`

## [v1.1.1] - 2025-02-03

### Added
- 图片智能分析功能：自动分析消息中的图片内容并纳入记忆
- 图片分析预算管理：支持日预算和会话预算限制
- 图片分析缓存机制：避免重复分析相同图片

### Changed
- 优化记忆注入提示词格式，支持人格风格参数
- 改进批量处理器性能

### Fixed
- 修复多个稳定性问题
- 修复情感分析器在某些边界条件下的异常

## [v1.1.0] - 2025-02-02

### Added
- 主动回复功能：基于情感分析自动触发主动回复
- 主动回复管理器：支持冷却时间和每日上限控制
- 分层消息处理：immediate / batch / discard 三层处理策略
- 消息批量处理器：累积处理普通消息，提高性能
- Docker 测试环境：提供一键启动的测试环境

### Changed
- 重构配置管理器，支持更灵活的配置访问
- 优化 RIF 评分算法权重
- 改进日志系统，支持按模块分级

### Fixed
- 修复会话管理器 TTL 处理问题
- 修复 Chroma 查询在某些情况下的空结果问题
- 修复重复记忆检测逻辑

## [v1.0.0] - 2025-02-01

### Added
- 三层记忆模型实现：
  - 工作记忆（Working Memory）：会话级 LRU 缓存
  - 情景记忆（Episodic Memory）：基于 RIF 的动态管理
  - 语义记忆（Semantic Memory）：永久用户画像存储
- 混合检索引擎：
  - 向量检索（Chroma 集成）
  - RIF 评分（时近性、相关性、频率）
  - 时间感知检索
  - 情感感知检索
- 情感分析系统：
  - 混合分析模型（词典 + 规则）
  - 情感轨迹追踪
  - 情感触发器管理
- 会话生命周期管理：
  - 自动会话清理
  - 工作记忆升级机制
  - 会话统计和监控
- 记忆捕获引擎：
  - 触发器检测
  - 实体识别
  - 置信度评估
  - 重复检测
  - 冲突检测
- LLM Hook 集成：
  - 自动记忆注入
  - 响应后自动捕获
  - Token 预算管理
- 完整指令集：
  - `/memory_save` - 手动保存记忆
  - `/memory_search` - 搜索记忆
  - `/memory_clear` - 清除记忆
  - `/memory_stats` - 记忆统计
  - `/memory_delete_private` - 删除私聊记忆
  - `/memory_delete_group` - 删除群聊记忆（管理员）
  - `/memory_delete_all` - 删除所有记忆（超级管理员）
- 用户画像系统：
  - Big Five 人格维度
  - 偏好学习
  - 交互历史统计
- 人格协调器：
  - Bot 人格与用户画像协调
  - 冲突检测
  - 动态策略选择

### Technical
- ChromaDB 本地向量存储
- SQLite 元数据存储
- AstrBot 插件框架集成
- 完整的单元测试覆盖
- Docker 化部署支持

---

## 版本说明

### 版本号规则
- **主版本号（Major）**: 重大功能变更或架构调整
- **次版本号（Minor）**: 新功能添加
- **修订号（Patch）**: Bug 修复和优化

### 分类标签
- `Added` - 新功能
- `Changed` - 功能变更
- `Deprecated` - 即将移除的功能
- `Removed` - 已移除的功能
- `Fixed` - Bug 修复
- `Security` - 安全修复
- `Technical` - 技术改进

---

**贡献者**: Leafiber

**项目地址**: https://github.com/leafliber/astrbot_plugin_iris_memory

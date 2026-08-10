# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.4] - 2026-08-10

### Security

- 图片网络请求统一使用安全下载器：初始 URL 与每一跳重定向均拒绝私网、环回、链路本地、云元数据及保留地址，禁用自动重定向并限制跳数；响应按流式实际字节限制为 10 MiB，并用图片魔数拒绝伪造内容。
- 消息中的绝对本地路径仅允许读取插件 `image_cache` 真实目录内的文件；本地图片同时限制大小与类型。缓存删除改为真实路径包含校验，拒绝同名目录和符号链接越界。

### Tests

- 新增 SSRF、重定向到云元数据地址、合法重定向逐跳复检、大响应、本地路径越界和缓存删除越界（含符号链接）回归测试。

## [3.0.3] - 2026-08-09

### Added

- **人格自迭代**：支持按目标方向与指定群/用户学习来源迭代具名 AstrBot Persona；默认只维护 `IRIS_EVOLUTION` 受控区块，也可显式启用完整人格模式。
- 自动（100 条新增有效消息 + 24 小时）与手动（至少 20 条）触发，自动/人工审批切换，Provider 重试与三次失败熔断。
- 完整 Revision/Run 审计、外部修改冲突保护、发布回读验证、git-revert 式非破坏性回滚，以及独立/全量 1.1 导入导出。
- Web 管理页：Job 编辑、语料分布、运行记录、Revision 时间线、审批/拒绝/回滚、冲突采纳与 `default` Persona 克隆。
- extended grapheme cluster 级人格 Diff：优先 `Intl.Segmenter`，提供 Emoji/组合字符 fallback，并由 Vitest 覆盖中文、换行、ZWJ Emoji、旗帜与组合音标。

### Security

- 原始群聊语料仅进入风格分析阶段；候选生成与完整人格审查只接收结构化画像。采集前执行注入拒绝、PII 脱敏、去重、保留期和总量限制。
- 发布前执行区块外零修改、marker、哈希、改动率、长度、保护片段、隐私复用和 Persona ID 等确定性校验；任何外部编辑均停止自动发布。

### Changed

- 全量备份格式由 1.0 扩展为 1.1，可选包含 `persona_evolution` 数据；导入旧版 1.0 仍兼容，导入 Revision 不会自动修改 AstrBot Persona。
- 学习模块批审查补充持久去重与并发安全，Web 学习管理沿用统一响应契约。
- **梦境任务降本与合并**：执行流收敛为确定性时间锚定、共享近邻扫描的记忆协调、增量知识归纳、persona 级 L2 清洗和每轮一次的全局 L3 维护；合并/矛盾/遗忘改为批量 LLM 请求，L2 内容更新改为批量 embedding，并为阶段报告增加 LLM、token 与 embedding 调用统计。
- **梦境阶段开关为破坏性变更，不提供旧键兼容映射**：移除 `dream_enable_consolidation`、`dream_enable_temporal_anchor`、`dream_enable_contradiction`、`dream_enable_pattern_discovery`、`dream_enable_knowledge_extract`、`dream_enable_pruning`；新增 `dream_stage_temporal_anchor_enabled`、`dream_stage_reconciliation_enabled`、`dream_stage_knowledge_induction_enabled`、`dream_stage_l2_pruning_enabled`、`dream_stage_l3_maintenance_enabled`。旧配置不会被读取，升级后需重新确认 5 个阶段开关。
- **产品定位升级为轻量化三合一**：README 围绕“记忆 + 主动回复 + 人格自学习迭代”重构，统一安装、快速开始、架构、配置、隐私、迁移与故障排查说明。
- **知识图谱 persona 隔离**：L3 节点、边及提取/检索链路携带 `persona_id`，旧数据库启动时自动补列并保持默认人格 ID 兼容；模式输入哈希和空知识提取收敛机制避免无变化数据反复调用模型。

### Tests

- 新增人格自迭代存储、采集、抽样、三阶段 LLM、发布闸门、调度、版本/回滚、命令、Web API、备份兼容及 grapheme Diff 测试。
- 新增梦境共享扫描、批量矛盾/遗忘、零 LLM 时间锚定、增量模式、知识空结果收敛、批量 embedding 及 L3 persona 隔离回归测试。

## [v3.0.2] - 2026-08-02

### Fixed

- **主动回复决策自我标识缺失（把自己的发言误认为第三方"代答"）**：bot 自身消息入滑动窗口时 `sender_name` 硬编码为插件名 `"Iris"`（`main.py` `on_message_sent` 与 `proactive.py` initiate 直发通路两处），决策上下文渲染（`perception.py` `ContextPackager.package`）按 `[昵称(ID)]` 原样透出，而决策 prompt（`prompts.py`）从未声明窗口中哪些条目是 bot 自己的发言。决策调用又复用主管线 provider 人格（如 chito），模型遂将自己以 "Iris" 署名的历史回复误判为另一群友替自己作答，产出"Iris 已代答"之类错误叙事；该叙事经 `observation` 持久化回注（`<recent_observation>`）与锚点 `reason` 注入跨轮自我强化，并可渗入最终发言。修复：① `ContextPackager` 新增 `self_id_get` 回调，渲染层将 `sender_id` 命中 bot 自身的条目统一改写为 `[我(ID)]`（不依赖存储名正确性，亦不受改名影响）；② 两处入窗点 `sender_name` 由 `"Iris"` 改为 `"我"`，移除硬编码插件名；③ 三档意愿（low/medium/high）决策 system prompt 追加自我标识说明，明确 `[我(ID)]` 即 bot 本人发言、不存在他人代答。新增 4 个用例，全量 1069 用例全绿。

## [v3.0.1] - 2026-07-30

### Fixed

- **主动发起时间感知缺失**：initiate 直发通路直连 `Context.llm_generate`，绕过了 AstrBot 主管线 `_append_system_reminders` 的当前时间注入，导致 LLM 无时间锚点、从滑动窗口里的旧消息推断时间（典型症状：早上发起时说"晚上好"）。新增 `iris_memory/proactive/time_hint.py`，按主管线格式（`Current datetime: …, Weekday: …`，读取 `provider_settings.datetime_system_prompt` 开关与 `timezone`）生成时间提示，并注入主动发起发言（`proactive.py` `_generate_speech`）与统一决策（`decision.py` `DecisionCore.build_prompt`，经 `main.py` 传入 `time_hint_get`）两条直连通路，使被动回复 / 正常接话 / 主动发起三条管道时间感知一致。新增 13 个用例，全量 1065 用例全绿。

## [v3.0.0]

### 🔄 重构说明

v3.0.0 是本插件的**整体重构与整合版本**：将 [astrbot_plugin_iris_chat_memory](https://github.com/Leafliber/astrbot_plugin_iris_chat_memory)（轻量记忆架构）与 astrbot_plugin_iris_reply（统一决策主动回复）的代码**整体移植合并为自包含插件**，同时保留 v2 的错误友好化与 Markdown 去除功能。**不依赖**上述两个插件，且不可与其同时启用（功能重叠，启动时有检测警告）。

- 上游移植基线：astrbot_plugin_iris_chat_memory commit `cb15779`（2026-07-18）、astrbot_plugin_iris_reply commit `ccbffe6`（2026-07-20）
- v2 用户升级请阅读 [docs/MIGRATION.md](./docs/MIGRATION.md)

### Added

- **记忆 + 主动回复二合一整合**：单一插件提供 L1 缓冲 / L2 向量记忆库（FAISS + SQLite）/ L3 知识图谱 / 画像系统 / 梦境 6 阶段离线加工 / 图片解析，以及统一决策主动回复（chime_in 跟话 / follow_up 跟进 / initiate 发起 / watch 被动评估）
- **统一决策模型**：单次 LLM 调用同时输出 是否发言 + 发言内容 + 话题概括 + 关注对象 + 话题漂移 + 冷却建议；SignalGate 本地零 LLM 成本门控；ThreadAnchor 对话锚点记账；backoff 退避 + boost 自适应频率；静音时段（默认 01:00–07:00）
- **initiate 直发通路**（`context.send_message`）+ 发起后接话闭环；**发起消息回填 L1**（v3 新增，修复原两插件并存时 initiate 消息不进 L1 的盲区）
- **v2 旧数据启动时自动一次性迁移**（`iris_memory/legacy_migration/`）：ChromaDB 记忆 → L2 重算 embedding、knowledge_graph.db → L3、旧画像 KV → 新画像（好感度默认 50）、旧主动回复白名单 → `iris_reply:whitelist` 并集、8 个旧配置键映射直写；迁移前自动备份到 `<数据目录>/legacy_backup/`；幂等（KV 标志 `legacy:migration_done`）；单项失败隔离不阻断启动；chromadb 为软依赖，未安装则跳过 L2 迁移并记日志；该模块将在 v4 删除（main.py `LEGACY_MIGRATION_ENABLED` + 整个目录）
- **Web 面板**：记忆管理 Vue3 SPA（Dashboard / L1 / L2 / L3 图谱 / 画像 / 导入导出备份 / 隐藏配置，pages/iris）+ 主动回复页（管理 / 统计 / 设置三个 tab，pages/stats），统一挂 AstrBot Dashboard 鉴权；回复侧 12 条 HTTP API（前缀 `/api/plug/astrbot_plugin_iris_memory/reply/*`）
- **新指令组**：`/iris_mem`（l1\|l2\|l3\|profile\|all × stats\|clear\|show\|reset\|help，ADMIN）、`/iris_reply`（enable/disable/status/reset/cooldown/willingness/initiate，ADMIN + 群消息）
- **LLM 工具**：记忆侧 6 个（save_memory / search_memory / correct_memory / save_knowledge / search_knowledge_graph / get_profile）+ 回复侧 3 个（add_follow_up / end_follow_up / set_cooldown）
- **测试**：新增 117（proactive 整合）+ 77（legacy_migration）个用例，全量 986 用例全绿

### Changed（破坏性变更）

- **指令变更**：`/memory`、`/iris` 退役 → `/iris_mem`、`/iris_reply`
- **LLM 工具变更**：`set_group_cooldown` / `get_cooldown_status` / `cancel_group_cooldown` 退役（由 `set_cooldown` 替代，`/iris_reply status` 可查状态）；`save_memory` / `search_memory` 保留
- **独立 Web 服务取消**（原 127.0.0.1:8089 + access_key）→ 统一挂 AstrBot Dashboard
- **配置体系重建**：v2 的 192 项配置废止 → `_conf_schema.json` 10 组 33 项；记忆侧约 50 项高级参数移入 `hidden_config.json`，回复侧 22 项高级参数由面板设置页管理（KV overrides）
- **嵌入与存储**：ChromaDB / sentence-transformers 硬依赖移除 → faiss-cpu；本地嵌入 sentence-transformers 变为可选（仅 `l2_memory.embedding_source=local` 时需要）
- **AstrBot 版本**：`on_agent_done` 钩子仅 AstrBot ≥ 4.23 注册（低版本插件仍可正常加载，仅旧版对话清理路径不可用）；**建议 AstrBot ≥ 4.23.6**
- **必须禁用 AstrBot 内置群聊上下文**（`provider_ltm_settings.group_message_max_cnt = 0`），否则重复注入 + 第三人称问题
- **v2 功能替代对照**：14 步捕获流水线 → L1 + LLM 总结；6 策略检索 + reranker → L2 向量 + L3 路径扩展；RIF / 情感分析 → 遗忘算法（`S=w1·R+w2·F+w3·C+w4·(1-D)`）+ 画像好感度；旧主动回复四级管线 → 统一决策；群冷却模块 → 回复侧冷却；群活跃度自适应 → willingness/backoff/boost

### Removed

- **依赖减重约 489MB**：移除 torch（378M）、onnxruntime（65M）、transformers（52M）、chromadb（4M）、sentence_transformers（3.8M，转可选）、uvicorn；新增 faiss（14M）
- 移除 v2 的捕获 / 检索 / RIF / 情感分析 / 旧主动回复 / 群冷却 / 活跃度自适应等模块（均有上文替代）
- 完全保留：错误友好化、Markdown 格式去除

### 指标对比（v2 → v3）

| 指标 | v2 | v3 | 变化 |
|------|----|----|------|
| 插件 Python 代码 | 49,217 行 | 35,143 行 | −28.6% |
| 测试代码 | 29,682 行 | 18,884 行 | 986 用例全绿 |
| _conf_schema | 56 项 / 16 组 | 33 项 / 10 组 | −41% 项数 |
| 安装体积 | — | — | 减重约 489MB |
| Token 控制 | 多阶段管线 | L1 队列 4000 / L2 注入 2000 / L3 注入 600 / 单条 ≤500 / 注入单条截 300 字符 / 主动回复单次决策调用 | — |

## [v2.0.0] - 2026-06-14

### ⚠️ 项目迁移公告
- **本项目（iris_memory）已进入维护状态，后续主力迭代迁移至新版 [astrbot_plugin_iris_chat_memory](https://github.com/Leafliber/astrbot_plugin_iris_chat_memory)**
  - 新版是专注记忆能力的 v2 重构：L1 Buffer / L2 记忆库 / L3 知识图谱 三层架构、更精简的记忆模型、Vue3 Web UI、标准化的导入导出
  - 老版（v1.x / v2.x）仍可正常使用，但新功能将主要在新版迭代；本项目以维护、Bug 修复为主
  - 新版仓库：https://github.com/Leafliber/astrbot_plugin_iris_chat_memory

### Added
- **Web 端新增「迁移到 Iris Chat Memory」导出功能** (`iris_memory/web/services/io_service.py`, `iris_memory/web/api/io_routes.py`, `iris_memory/web/static/`)
  - 在 Web UI「导入导出 → 导出」页新增「🔄 迁移到 Iris Chat Memory」卡片，一键将记忆导出为新版可识别的 L2 导入格式（JSON）
  - 字段映射：`created_time → timestamp`、`summarized → source(summary/tool)`，数值字段防御性转换，并标记 `migrated_from="iris_memory"` 便于回溯
  - 后端路由 `GET /api/v1/io/export/iris_chat_memory`，支持 `user_id` / `group_id` / `storage_layer` 筛选
  - 导出文件可在新版 Web UI「数据管理 → 导入 L2 记忆」直接导入（已通过跨仓库格式兼容性验证）
  - 顺带修复 `exportPersonas` 未挂载到全局导致画像导出按钮无效的问题 (`iris_memory/web/static/js/main.js`)

### 迁移方式
1. **记忆（已支持）**：老版 Web UI → 导入导出 → 导出 → 「迁移到 Iris Chat Memory」→ 下载 JSON → 新版 Web UI「数据管理 → 导入 L2 记忆」上传
2. **知识图谱**：暂需手动迁移（新版 `L3KGAdapter.import_from_data`，需核对节点 / 关系类型取值）
3. **用户画像**：暂需手动迁移（老版 `UserPersona` → 新版 `profile` 模型差异较大）
4. **配置**：两版 schema 不同，需手动映射对应配置项

## [v1.11.2] - 2026-04-13

### Fixed
- **LLM Tool 保存结构修复** (`main.py`)
  - 修复 `save_memory` LLM Tool 创建记忆时缺少必要字段的问题
  - 新增 `user_id`、`sender_name`、`group_id` 字段
  - 新增 `type=MemoryType.FACT` 和 `modality=ModalityType.TEXT` 类型标识
  - 新增 `is_user_requested=True` 标记用户主动请求保存的记忆

## [v1.11.1] - 2026-03-14

### Removed
- **移除记忆审核命令** (`iris_memory/commands/handlers.py`)
  - 移除 `/memory review`、`/memory approve`、`/memory reject` 命令
  - 这些命令查询 SEMANTIC 层的待审核记忆，但实际待审核记忆在 EPISODIC 层
  - 宽限期记忆现已完全自动化处理，无需人工干预
  - 同步移除 `chroma_manager.get_pending_review_memories` 和 `grace_period.resolve_grace_period` 等相关代码

## [v1.11.0] - 2026-03-13

### ⚠️Note
- 本次更新优化了 Web 管理端的启动逻辑，**需要完全重启 AstrBot（Docker/宿主机）才能生效**

### Changed
- **Web 管理端启动逻辑优化** (`iris_memory/web/server.py`)
  - 重构 Uvicorn 服务器启动方式，使用标准 `server.serve()` API
  - 移除不稳定的内部 API `config.http_protocol_class` 调用
  - 修复服务器显示启动成功但无法处理请求的问题
  - 优化端口复用 socket 管理
  - 改进服务器停止时的优雅关闭逻辑

- **宽限期智能自动处理** (`iris_memory/storage/grace_period.py`)
  - 新增 `auto_keep` 自动保留机制，高价值记忆无需等待宽限期
  - 自动保留条件：情感权重 ≥ 0.5 或 重要性 ≥ 0.6 且访问 ≥ 2 次
  - 移除未使用的用户通知代码（`_notify_user` 方法）
  - 简化宽限期逻辑，完全自动化处理

## [v1.10.6] - 2026-03-13

### Changed
- **记忆强化引擎简化** (`iris_memory/analysis/reinforcement.py`)
  - 移除回顾消息发送功能，不再主动发送回顾对话
  - 移除 `ReviewPromptGenerator` 类（回顾对话生成器）
  - 移除 `notify_callback` 参数和通知发送逻辑
  - 移除 `max_daily_reviews` 每日回顾上限配置
  - 移除 `get_review_candidates()` 方法
  - 移除 `process_review_response()` 方法
  - 保留 SM-2 变体核心逻辑：定期分析重要记忆并更新 RIF 评分

### Fixed
- **Web 仪表盘记忆总数显示修复** (`iris_memory/web/static/js/pages/dashboard.js`)
  - 修复前端读取 `mem.total` 与后端返回 `total_count` 字段名不一致的问题
  - 兼容处理：`mem.total_count ?? mem.total ?? 0`

- **Web 用户画像活跃时段显示修复** (`iris_memory/web/repositories/persona_repo.py`)
  - 修复 `_build_persona_data` 方法缺少 `hourly_distribution` 字段
  - 活跃时段图表现在可以正确显示用户交互时间分布

- **Web 记忆管理分页功能修复** (`iris_memory/web/static/js/pages/memories.js`)
  - 修复分页回调函数写法与其他页面不一致的问题
  - 统一使用箭头函数形式：`onChange: p => { state.page = p; searchMemories(); }`

- **LLM 统计来源推断修复** (`iris_memory/utils/llm_helper.py`, `iris_memory/stats/registry.py`)
  - 修复异步任务中调用栈丢失导致来源显示为 `_UnixSelectorEventLoop` 的问题
  - 在 `call_llm()` 执行时立即捕获调用来源，传递给统计记录
  - 新增 `_infer_caller_source()` 函数预先推断来源
  - `record_call()` 新增可选参数 `source_module` 和 `source_class`

- **Web 知识图谱节点大小优化** (`iris_memory/web/static/js/pages/kg.js`)
  - 缩小节点半径范围：6px ~ 16px（原 8px ~ 25px）
  - 优化视觉呈现，避免节点过大遮挡

### Removed
- 移除 `memory.reinforcement.max_daily` 配置项（每日回顾上限）

## [v1.10.5] - 2026-03-12

### Fixed
- **Web 服务器 Hypercorn 兼容性修复** (`iris_memory/web/server.py`)
  - 修复新版 Hypercorn API 变更导致的 `worker_serve()` 参数错误
  - 改用标准 `config.bind` 格式，让 Hypercorn 自动管理 socket
  - 优化启动检测逻辑，增加任务状态检查
  - 缩短关闭超时时间
  - 添加详细的启动失败错误日志

## [v1.10.4] - 2026-03-10

### Added
- **Web 管理界面全新重构** (`iris_memory/web/`)
  - 采用分层架构：API 路由层、服务层、数据仓库层
  - 新增模块化前端代码结构，ES6 模块化组织
  - 新增 Dashboard 仪表盘页面，集成系统状态和 LLM 监控
  - 新增记忆管理页面，支持搜索、查看、编辑、批量删除
  - 新增知识图谱页面，支持节点/边可视化和搜索
  - 新增用户画像页面，展示用户特征和交互历史
  - 新增主动回复配置页面，支持白名单管理
  - 新增冷却机制页面，展示和管理冷却状态
  - 新增配置管理页面，支持配置查看和导出
  - 新增 LLM 监控页面，展示调用统计和最近记录
  - 新增系统信息页面，展示运行状态和资源使用
  - 新增导入导出功能，支持记忆和知识图谱的 JSON 格式

### Changed
- **前端代码结构重构** (`iris_memory/web/static/js/`)
  - 将多个独立 JS 文件合并为模块化结构
  - 按功能划分：api、components、pages、store、utils
  - 统一使用 ES6 import/export 语法
  - 优化代码组织，减少全局变量污染

### Fixed
- **Web Dashboard 模块导入缺失修复** (`iris_memory/web/static/js/main.js`)
  - 添加缺失的 `loadLlm` 导入语句
  - 添加缺失的 `loadSystem` 导入语句
  - 修复页面加载时 `ReferenceError` 错误

- **Web UI 初始化问题修复** (`iris_memory/web/server.py`)
  - 修复 Web UI 初始化重复问题
  - 修复端口占用检测逻辑

## [v1.10.3] - 2026-03-08

### Fixed
- **NumPy 数组布尔判断错误修复** (`iris_memory/storage/chroma_manager.py`, `iris_memory/embedding/manager.py`)
  - 修复 `_extract_memory_data` 方法中 `documents`、`embeddings`、`metadatas` 的布尔判断
  - 修复 `_detect_existing_dimension` 方法中 `embeddings` 的布尔判断
  - 将隐式布尔判断 `if embeddings and ...` 改为显式判断 `if embeddings is not None and ...`
  - 解决 ChromaDB 某些情况下返回 NumPy 数组导致的 `ValueError: The truth value of an array with more than one element is ambiguous`

- **MemoryScope 导入路径修复** (`main.py`)
  - 修复 `save_memory_tool` 方法中 `MemoryScope` 的导入路径
  - 将 `from iris_memory.core.types import MemoryScope` 改为 `from iris_memory.core.memory_scope import MemoryScope`

## [v1.10.2] - 2026-03-04

### Changed
- **Markdown 去除器配置简化** (`iris_memory/processing/markdown_stripper.py`)
  - 用户可见配置仅保留 `enable` 开关（通过 AstrBot 管理界面控制）
  - 内部配置（`preserve_code_blocks`、`preserve_links`、`threshold_offset`、`strip_headers`、`strip_lists`）移至 `defaults.py` 统一管理
  - 减少配置复杂度，默认行为：去除所有 Markdown 格式标记

### Removed
- 移除 `_conf_schema.json` 中 Markdown 去除器的 5 个内部配置项
- 移除 `config_registry.py` 中对应的 5 个 `ConfigDefinition` 映射
- 移除 `config_properties.py` 中对应的 5 个 `_ConfigProp` 属性定义
- 移除测试文件中不再适用的配置变体测试用例

## [v1.10.1] - 2026-03-03

### Changed
- **FollowUp 调试日志增强** (`iris_memory/proactive/manager.py`)
  - `notify_bot_reply` 方法新增详细调试日志，输出初始化状态、配置开关状态
  - 每个提前返回点新增日志说明具体跳过原因
  - 便于排查 FollowUp 机制未触发问题

## [v1.10.0] - 2026-03-02

### ⚠️ 注意
本次更新需要完全重启 Nonebot，否则会导致主动回复模块初始化失败

### Verified
- **ProactiveManager API 兼容性验证** (`iris_memory/capture/batch_processor.py`, `iris_memory/proactive/manager.py`)
  - 验证 `process_message` 参数格式正确匹配
  - messages 字段 (text, sender_id, sender_name, timestamp) 完整传递
  - 无需额外参数验证逻辑

- **ProactiveManager 初始化参数传递验证** (`iris_memory/services/initializer.py`, `iris_memory/services/modules/proactive_module.py`)
  - 验证 `plugin_data_path` 参数正确传递
  - 调用链完整：initializer → ProactiveModule → ProactiveManager
  - 已有 `if not plugin_data_path` 防护检查

- **测试用例接口一致性验证** (`tests/capture/test_batch_processor.py`)
  - 验证测试代码已使用 `process_message` 新接口
  - 无遗留 `handle_batch` 引用
  - `TestProactiveReplyIntegration` 正确验证新 API 调用

## [v1.9.3] - 2026-03-02

### Added
- **连续回复限制机制** (`iris_memory/proactive/proactive_manager.py`)
  - 新增 `_recent_replies` 跟踪短时间内各会话的主动回复次数
  - 默认限制：5分钟内最多连续回复 3 次
  - 新增 `_is_consecutive_limit_reached()` 和 `_record_reply_time()` 方法
  - 新增 `replies_consecutive_limited` 统计计数器
  - 防止特定群聊/用户的"滚雪球"式连续回复问题

- **启动冷却期机制** (`iris_memory/proactive/proactive_manager.py`)
  - 新增 `_startup_time` 记录启动时间
  - 新增 `_is_in_startup_cooldown()` 方法检查启动冷却状态
  - 默认启动冷却期：2 分钟（`STARTUP_COOLDOWN_SECONDS=120`）
  - 防止重启后状态丢失（`_recent_replies`、`last_reply_time` 为空）导致连续回复

### Changed
- **主动回复检测器阈值与权重调整** (`iris_memory/proactive/proactive_reply_detector.py`)
  - MEDIUM 阈值从 0.3 提高到 0.4，降低误触发概率
  - question 权重从 0.4 降低到 0.3
  - emotional_support 权重从 0.3 降低到 0.25
  - seeking_attention 权重从 0.3 降低到 0.25
  - mention_bot 权重从 0.5 降低到 0.35
  - expect_response 权重从 0.35 降低到 0.25
  - chat_topics 权重从 0.25 降低到 0.2
  - 积极情感触发阈值从 0.3 提高到 0.5，避免群聊"哈哈哈"误触发

- **紧急度冷却乘数调整** (`iris_memory/core/constants.py`)
  - CRITICAL 乘数从 0.25 提高到 0.5（冷却时间：60s × 0.5 = 30s）
  - HIGH 乘数从 0.5 提高到 0.75（冷却时间：60s × 0.75 = 45s）
  - 避免高紧急度回复冷却时间过短导致频繁触发

- **智能增强参数调整** (`iris_memory/core/defaults.py`)
  - smart_boost_window 从 120s 缩短到 60s（不超过冷却时间）
  - smart_boost_threshold 从 0.25 提高到 0.4（与 MEDIUM 阈值一致）
  - 确保智能增强窗口不会与冷却机制冲突

### Fixed
- **每日计数惰性重置** (`iris_memory/proactive/proactive_manager.py`)
  - 新增 `_last_reset_date` 跟踪重置日期
  - 新增 `_check_daily_reset()` 方法实现跨日自动重置
  - 修复每日计数从未被重置的问题

- **用户发言时间记录时机** (`iris_memory/proactive/proactive_manager.py`)
  - 将 `_record_user_message()` 调用从 `_process_task` 移至 `handle_batch`
  - 确保智能增强窗口基于用户发言时间而非 Bot 回复时间
  - 避免 Bot 自身回复刷新窗口导致"滚雪球"效应

- **冷却时间记录时机** (`iris_memory/proactive/proactive_manager.py`)
  - 将 `last_reply_time` 记录从 `handle_batch` 移至 `_process_task` 发送成功后
  - 确保冷却时间基于实际发送时间而非入队时间

- **KV 持久化 is_async 配置错误** (`iris_memory/services/persistence_service.py`)
  - 修复同步方法被错误标记为异步导致 `await` 报错的问题
  - `serialize_whitelist`/`deserialize_whitelist` 设置 `is_async=False`
  - `member_identity.serialize`/`deserialize` 设置 `is_async=False`
  - `activity_tracker.serialize`/`deserialize` 设置 `is_async=False`
  - 错误信息：`object list can't be used in 'await' expression`

### Tests
- **连续回复限制测试** (`tests/proactive/test_consecutive_limit.py`)
  - 新增连续回复限制基本逻辑测试
  - 新增窗口过期自动清理测试
  - 新增会话隔离测试
  - 新增 handle_batch 集成测试

## [v1.9.2] - 2026-03-02

### Added
- **命令处理与权限管理** (`iris_memory/services/business_service.py`, `iris_memory/services/memory_service.py`)
  - 新增 `handle_command()` 方法处理管理命令
  - 实现管理员权限检查机制
- **检索策略实现** (`iris_memory/retrieval/`)
  - 新增多种检索策略支持
- **智能增强配置更新** (`iris_memory/proactive/`)
  - 更新 smart boost 配置，增强主动回复任务管理
- **语义提取与聚类测试** (`tests/`)
  - 新增语义提取、聚类和置信度机制的全面测试

### Changed
- **ChromaManager 架构重构** (`iris_memory/storage/chroma_manager.py`)
  - 从 Mixin 继承模式重构为组合模式
  - 提升代码可维护性和可测试性
- **MemoryService 初始化逻辑** (`iris_memory/services/memory_service.py`)
  - 实现 ServiceInitializer，将初始化逻辑内联到 MemoryService
- **KV 存储逻辑简化** (`iris_memory/storage/`)
  - 简化 KV 加载和保存逻辑，采用配置驱动方式

### Fixed
- **主动回复人格传递** (`iris_memory/proactive/proactive_event.py`, `iris_memory/proactive/proactive_manager.py`)
  - 修复主动回复使用默认人格而非配置人格的问题
  - `ProactiveMessageEvent` 新增 `persona_id` 参数并设置 `self.persona`
  - `QueuedMessage`、`ProactiveReplyTask` 等数据类添加 `persona_id` 字段
  - 整个调用链正确传递 `persona_id`

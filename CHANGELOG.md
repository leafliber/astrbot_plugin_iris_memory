# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **平台适配器与 AstrBot 4.x 对齐（群角色/原始昵称/群名片读取失效）**：AstrBot 的 `MessageMember` 仅含 `user_id`/`nickname`，适配器此前读取其不存在的 `card`/`role` 字段——`get_user_role` 群聊恒返回 `member`（owner/admin 识别失效），`get_user_nickname` 实际返回的是 AstrBot 合并的"群名片或昵称"。现群名片/原始昵称/群角色改从 raw OneBot 载荷的 `sender` 字典读取，载荷缺失时回退 AstrBot 合并值。
- **@用户定向指令失效**：`executor` 以 `get_message_outline()` 解析指令，真实 @ 在其中渲染为 `[At:123456]` 而非 `@名字`，parser 无法识别；且 `get_mentioned_users` 读取 raw at 段的 `data.name`（多数协议端不提供该字段），名称匹配必然落空。现 parser 支持 `[At:123456]`（outline 形式，直接得 ID）、`@名字(123456)`（message_str 形式）与纯文本 `@名字` 三种形式，@提及名称改读消息链上 AstrBot 已调 API 解析好的 `At` 组件。
- **多账号部署下引用回填与合并转发静默失败**：`get_msg`/`get_forward_msg` 调用未携带 `self_id` 路由参数，aiocqhttp 多反向 WS 连接下无法路由到事件所属协议端（`ApiNotAvailable` 被兜底吞掉，单连接不受影响）。现对齐 AstrBot 核心的 `routing_params` 写法统一携带 `self_id`。
- **引用消息重复调用 get_msg**：AstrBot 转换消息时已为每个 reply 段调过 `get_msg` 并把完整结果放进消息链 `Reply` 组件，适配器现优先直接读取（发送者与纯文本一次拿全，消除双倍 API 调用），链上无 `Reply` 或仅剩裸 id 时才回退 raw 解析与 API 查询。
- **群名称读取**：优先读取 AstrBot 结构化字段 `message_obj.group.group_name`（aiocqhttp 的 `"N/A"` 缺省哨兵映射为空），`GenericAdapter`/`CronAdapter` 同步接入（此前恒返回空字符串）。
- **用户级 clear/delete 漏删**（`iris_mem l2/l3/all clear` 默认与 `@用户` 范围）：L2 `delete_by_user` 此前仅匹配 `active_users`，漏删 `save_memory` 工具写入的仅有 `user_id` 的记忆（清后仍被检索注入、Web 手删才可清除）；现命中条件为 `metadata.user_id == user` 或 `active_users` CSV 精确包含，`save_memory` 写入同步补齐 `active_users` 对齐 L1 总结形态。
- **L3 按用户/按群删除漏删**：`delete_by_user` 此前仅按 `name == user_id` 精确匹配，现增加 `properties.user_id` 精确命中；跨群共享节点（`group_id` 列被其他群覆盖）通过 `properties.group_ids` CSV 命中（彻底删除语义，用户已确认）；两方法均支持 `persona_id` 过滤，`json_valid` 守卫容忍存量损坏 JSON。
- **命令清除的 persona 归属**：`l2/l3/all clear` 群级与用户级路径现经 `resolve_persona` 解析当前人格并传入删除调用，非 default persona 命名空间的记忆不再清空落空（`--all` 保持跨 persona 全清）。
- **Web 导出按钮无反应**：`apiDownload` 增加降级链路——宿主桥接 `download` 缺失/报错/20s 无响应超时时，自动改经 `apiGet` 拉取 JSON 并在 iframe 内构造 Blob 下载（覆盖全部 6 个导出端点），双路失败才报错；兼容 AstrBot 桥接较旧版本与浏览器拦截场景。
- **知识图谱按用户 ID 搜索不命中（存量数据）**：`_build_user_aliases` 无昵称来源的用户现以空别名列表入映射，使 `name=user_id` 的 Person 节点能打上 `properties.user_id` 标记；`save_knowledge` 工具与 dream 模式发现写入的 Person 节点同样补打标记；`merge_person_nodes_by_user_id` 支持画像别名映射（`build_profile_alias_map`，昵称/曾用名驱动），Web「合并重复节点」与 `iris_mem l3 merge` 均接入，存量无标记昵称节点按映射吸收合并或就地补打标记。
- 注册 `/memory` 旧版指令指引别名（管理员）：返回迁移指引而非静默无效，帮助 v2 迁移用户改用 `/iris_mem`。

### Changed

- 平台注册表对齐 AstrBot 4.x 实际协议名：`qqofficial` 更名 `qq_official`（仍待实现、降级 GenericAdapter），删除 `qq`/`gewechat` 死条目；telegram、webchat、wecom、`qq_official_webhook` 等未注册平台行为不变，统一经 GenericAdapter 降级。
- 移除 string/CQ 码消息段解析死代码（AstrBot 上游直接丢弃非 array 格式消息，相关分支永不可达）。
- 知识图谱「节点」列表标签的本地过滤框升级为服务端搜索（enter/按钮触发，走既有 `keyword` 检索链路，支持用户 ID/名称/内容关键词）。
- Web「合并重复节点」端点与命令入口对齐：补调 `merge_person_nodes_by_user_id`（此前 Web 侧漏调）。

### Tests

- 新增 qq_official 适配器测试（四场景基础字段与 openid 稳定标签、raw_data 读取与 message_id 键归一化、链优先引用/图片/提及提取、机器人标记 At 排除、频道角色映射、`get_msg_by_id` 场景守卫与异常降级、工厂双平台键注册）；`fakes.py` 新增 `FakeBotpyRawMessage`（`__slots__` 形状探针，刻意无 `__dict__`，对齐 AstrBot Patched botpy 消息对象）与 `make_qq_official_event` 四场景事件工厂。
- 新增按 AstrBot 4.27.2 真实形状构造的事件夹具（`tests/platform/fakes.py`：MessageMember 仅 user_id/nickname、消息链使用真实 astrbot 组件）与 AstrBot 兼容性契约测试（`test_astrbot_compat.py`，直接断言真实 AstrBot 的字段/组件/会话形状），Mock 自动伪造属性导致的适配器形状漂移今后会立即暴露；新增指令解析器测试覆盖三种 @ 形式与 scope 校验（此前 `CommandParser.parse`/`execute_command` 零覆盖）。
- 新增 L2 用户级删除作用域（工具记忆/多用户场景/global 保护/CSV 精确匹配）、L3 按用户与按群删除（标记命中/跨群 CSV/损坏 JSON/persona 隔离/边级联）、命令层 persona 透传、提取器空别名用户打标、`save_knowledge` Person 归一化、画像别名映射合并与构建、前端 `apiDownload` 降级链路（vitest）回归测试；修正 `save_memory` schema 快照漂移。

### Added

- **qq_official（QQ 官方机器人）平台适配器**：新增 `QQOfficialAdapter` 并注册 `qq_official`/`qq_official_webhook` 两个平台键（此前降级 GenericAdapter，引用消息、图片提取、被@用户、message_id 元数据全部丢失）。四种消息场景（群/单聊/频道/频道私信）统一适配，通过 raw 载荷特征字段探测场景、不依赖 botpy 包；引用消息与附件图片从消息链提取（AstrBot 已解析好群引用内容与归一化 https URL，图片含被引用消息内图片）；频道场景支持 mentions 真实用户提取、`member.roles` 角色映射与 `get_message` 按 ID 兜底；raw 载荷读取 Patched botpy 对象的 `raw_data`（botpy 消息类全部 `__slots__` 无 `__dict__`，通用 `__dict__` 回退必然失败）并把 `id` 键归一化为 `message_id`。平台隐私限制下的降级：群/单聊无昵称与群名（说话人以 openid 前缀派生稳定标签 `成员_xxxx`/`用户_xxxx`，跨消息恒定，群聊记忆可区分说话人）；群聊被@用户不透出（mentions 只含机器人自身，并排除 AstrBot 塞入的机器人标记 At）；无合并转发拉取。同一用户群/单聊/频道为三个互不关联的 ID 空间，画像与 Person 节点天然按场景隔离。
- 新增默认启用的纯 `@` 回复接管：关闭 AstrBot 内置空提及等待后，纯 `@` 可进入标准 LLM 管道并仅使用 Iris L1 上下文。
- **L2 混合检索**：向量语义路与 SQLite FTS5（trigram 分词）关键词路双路召回，RRF 倒数排名融合排序，提升专名、术语、数字等关键词的召回率。`enable_hybrid_retrieval` 可见开关默认开启，RRF k、关键词候选池倍数等调优参数为隐藏配置；关键词索引随记忆增删改双写维护并在启动时全量重建，SQLite 无 FTS5/trigram 支持时自动降级为纯向量检索。
- **L2 召回调试页**：L2 管理页新增「召回调试」标签，可分别查看向量路/关键词路命中明细（ID、分数、内容）与 RRF 融合排序，并支持查看 FTS 索引状态与手动重建。
- **记忆重要度（importance）**：总结提示词同批输出（high/medium/low 三档映射 0.8/0.5/0.2）、`save_memory` 工具参数传入；接入遗忘评分新增 w5·I 项（`forgetting_l2_weight_importance` 默认 0.3，不参与检索排序）。
- **命中强化**：对话检索命中时渐近提升重要度（I' = I + step·(1−I)，`l2_hit_reinforcement_step` 默认 0.1，隐藏开关默认开），跨 0.65/0.35 阈值时同步 importance_level；梦境扫描不触发。
- **归档与恢复**：梦境遗忘淘汰改为移入归档表（保留原向量，恢复免重嵌入），30 天保留期（`l2_archive_retention_days`）由梦境自动清除超期项；支持 `iris_mem l2 restore <id>` 指令与 L2 管理页「归档」标签（浏览/恢复/彻底删除）。用户显式 clear/删除仍为硬删。
- **TTL 过期**：`save_memory` 工具新增 `ttl_hours` 可选参数，为临时事实写入 `expires_at`；向量路、关键词路检索均过滤过期记忆，梦境每轮硬删过期条目。
- **`/remember` 学习手令**：`iris_mem learning remember <上下文> => <表达示例>` 将对话对与表达模式双写进风格学习链（scene 取上下文前 20 字），人工手令直接 approved 生效，无需等待审查。
- **全局共享作用域**：`save_memory` 工具新增 `scope` 参数（group/global）、L2 编辑对话框可切换作用域；`scope=global` 的记忆不受群隔离限制（向量路、关键词路、计数守卫均豁免），RRF 融合分按 `l2_hybrid_global_scope_factor`（默认 0.8）降权；群级/用户级 clear 跳过全局记忆，`clear --all` 仍全删。

### Changed

- 将纯 `@` 回复、错误消息友好化和 Markdown 输出清理统一归入 `extras` 辅助功能配置组。
- LLM Tool 统一注册重构：9 个工具（记忆侧 6 个 + 主动回复侧 3 个）全部为显式 `FunctionTool.call()` 实现，由 `iris_memory/tools/registry.py` 的 `build_llm_tools()` 构建、`register_llm_tools()` 经 `Context.add_llm_tools()` 注册并在其后统一写入插件归属；移除 `main.py` 中 3 个 `@filter.llm_tool` 装饰器方法与 6 个临时归属修复薄子类。工具名称、描述、Schema 与返回文案保持不变。

### Tests

- 新增 LLM Tool registry 完整性、精确 Schema 回归、主动回复工具行为、插件归属与 Dashboard 序列化、重复注册（热重载模拟）测试。

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

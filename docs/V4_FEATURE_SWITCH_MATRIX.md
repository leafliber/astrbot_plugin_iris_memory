# dev.4 功能覆盖与 Pages 独立开关矩阵

日期：2026-09-08。状态：**设计清单，所有新增键尚未实施**。配套[总计划](V4_THIN_ADAPTER_PLAN.md)。

对照基线为 v3 `3.2.0/b752ffb` 与插件 `4a8e0d0`。核查来源包括 v3 `_conf_schema.json`、`iris_memory/config/defaults.py`、工具注册表、主动/学习/人格/平台处理和数据管理代码，以及[现状对照](V4_V3_FUNCTION_COMPARISON.md)。矩阵包含 v3 能力和已有 v4 增量，不表示每行当前可用。

## 1. 读表及默认值

- 每个反引号键都是**独立开关**；同一单元格多个键必须分别控制，不能在实现时缩成一项。键为拟定设计名，W0 冻结命名。
- 下表“Core”表示通过公开业务/策略接口承担功能；不表示该接口已经存在。总计划第 3 节列出本轮确认的接口与未确认项。
- 新安装各可选功能默认关闭，用户从 Pages 选择。接入总开关开启后的含义是范围内所有消息；不要求同时启用上下文、模型、学习或主动模块。
- 已有配置保留用户意图；不会将旧粗粒度开关自动解释为所有新功能均开启。v3 导入另有预览映射。`background_retention` 单独沿用 Core 目前确定的 30 天默认，关闭表示不自动到期清理。
- 手动功能（纠正、图谱编辑、导入等）的开关控制业务入口是否启用，具体操作仍有对象/权限/版本校验；它们不是自动运行任务。
- 原始事件结构、严格身份/授权、幂等、来源有效性、正文转义、删除传播、基础交付恢复属于正确性条件，不做可关闭选项。高级策略可配置，基础正确性不能依赖用户勾选。
- 另设 `plugin.enabled` 业务总闸与 `context.enabled` 采用总闸；关闭总闸保留所有子项设置。全局关闭不能被来源覆盖，Pages 仍可用于恢复配置。

## 2. 消息、上下文与长期记忆

| ID | v3/已有功能与目标 | 独立 Pages 开关 | 承接方式与关闭语义 | 工作包 |
| --- | --- | --- | --- | --- |
| F01 | 用户/助手 L1 消息采集，扩展为全量 Observation | `observation.enabled`，来源级 `ingress.enabled` | 插件规范化与交付→Core；关闭暂停相应新接收，未 ACK 队列保留，非采集功能不联动关闭 | W2 |
| F02 | L1 近期环境补充 | `context.recent` | Core 原文上下文→统一组装；关闭只停止原文注入 | W3 |
| F03 | 滚动摘要的采用 | `context.summaries` | Core Episode 及来源覆盖；关闭不删除摘要、不停止原文查询 | W3 |
| F04 | 自动滚动总结 | `processing.auto_summary` | Core 数量/等待触发与持久批次；关闭停止新自动摘要并按合同处理在途任务 | W3 |
| F05 | 手动总结/重新整理入口 | `processing.manual_summary` | Pages 调用 Core 显式摘要作业；与自动开关分离，保留作业查询 | W3 |
| F06 | 自动长期事实提炼 | `processing.claim_extraction` | Core 候选、主体/Scope/证据校验；关闭不影响原文和摘要读写 | W3 |
| F07 | 记忆查询（Pages/LLM 工具） | `memory.search` | Core Recall；关闭入口，不关闭后台提炼 | W3 |
| F08 | 长期记忆自动注入 | `context.long_term` | 统一组装与预算；关闭不禁止手动查询 | W3 |
| F09 | 显式保存事实 | `memory.remember` | Core Claim 与明确主体/证据；关闭显式新写入，不等于关闭 Observation | W4 |
| F10 | 显式纠正事实 | `memory.correct` | Core 修订与来源；关闭纠正入口，既有纠正持续有效 | W4 |
| F11 | 遗忘条目及按用户/会话/范围删除 | `data.forget` | Core 预览及受权删除，明确 Claim/Observation/派生/宿主历史范围；已提交删除继续传播 | W4 |
| F12 | 语义向量检索与构建 | `retrieval.vector` | Core 索引/检索策略；关闭该可选路由与其自动构建意图，不删旧索引；需真实策略接口 | W4 |
| F13 | v3 FTS 混合检索 | `retrieval.hybrid` | Core 公开路由策略，关闭回落到已启用基础路线；不能仅过滤返回值假装未执行 | W4 |
| F14 | 可选查询改写 | `retrieval.query_rewrite` | 优先 Core 公共能力；插件可做公开字段允许的时间/主体查询构造；额外模型改写关闭后零调用 | W4 |
| F15 | 检索使用反馈/命中强化 | `retrieval.usage_feedback` | 按 Core Usage 合同反馈，不把模型可见冒充真实使用；无 Usage 的原文查询只记本地诊断 | W3/W4 |
| F16 | 群聊记忆/画像共享与人格隔离配置 | `sharing.long_term`、`sharing.profiles` | Pages 显式共享政策→Core 授权；关闭按来源隔离，原文始终不因 Agent 相同自动共享；不同主体始终隔离 | W2/W4 |
| F17 | 可选宿主上下文接管 | `context.host_cleanup` | 默认保留宿主历史；兼容模式只经宿主公开入口、且 Core 上下文完整可用后启用；不恢复默认整段删除 | W6 |

F01 的原文采集不被 F02～F17 的开关影响。后台是否提炼、是否查询、是否自动注入是不同动作，不能复用一个 `memory` 开关。

## 3. 图谱、画像与梦境维护

| ID | v3/已有功能与目标 | 独立 Pages 开关 | 承接方式与关闭语义 | 工作包 |
| --- | --- | --- | --- | --- |
| F18 | L3 关系/知识提取 | `processing.relation_extraction` | Core Relation/Claim 及证据；关闭新增自动关系，不影响已有读取 | W4 |
| F19 | 图谱查询及上下文采用 | `graph.search`、`context.graph` | Core 图召回；查询与自动注入分别控制，禁止插件重建图检索引擎 | W4 |
| F20 | 图谱画布、节点/边管理及保存知识工具 | `graph.browser`、`graph.edit` | 受限范围分页画布→Core 公共读写；缺公开接口即阻塞，不直写内部库 | W4/W6 |
| F21 | 用户画像更新与读取/采用 | `profile.user_update`、`profile.user_read`、`context.user_profile` | Core 画像及有效来源；三个动作分别停止，关闭更新不隐藏已有有效画像 | W4 |
| F22 | 群画像、群氛围与读取/采用 | `profile.group_update`、`profile.group_read`、`context.group_profile` | Core 公共群/Space 表示，不能拿某一用户画像替代群画像；接口需冻结 | W4 |
| F23 | 好感度演化与采用 | `profile.affinity_update`、`context.affinity` | Core 明确非事实的关系/状态语义与策略；不由消息数量硬推亲密度，不另建分数事实库 | W4 |
| F24 | 梦境总开关与时间锚定 | `maintenance.dream`、`maintenance.temporal_anchor` | Core 同一作业框架；总开关关闭全部梦境业务，子项可单独关闭；不是停止基础 worker | W7 |
| F25 | 梦境记忆合并/矛盾协调 | `maintenance.reconciliation` | Core 协调策略、修订与证据；关闭不影响用户显式纠正 | W7 |
| F26 | 梦境知识归纳 | `maintenance.knowledge_induction` | Core 归纳候选；关闭零该阶段模型工作 | W7 |
| F27 | 梦境 L2 遗忘/清洗及可选模型复核 | `maintenance.memory_pruning`、`maintenance.pruning_review` | Core 候选清洗和独立复核；模型复核仅针对自动淘汰，不能否决用户主动遗忘 | W7 |
| F28 | 梦境 L3 全局维护 | `maintenance.graph` | Core 图索引及关系整理策略；关闭此阶段，不关闭一般删除失效传播 | W7 |
| F29 | 原始背景观察保留期 | `retention.background` | 默认开启、30 天；关闭普通到期清理，Core 按来源依赖与 Hold 判断保留；不清理未 ACK 插件队列 | W3/W7 |
| F30 | 手动索引诊断/重建 | `maintenance.index_tools` | Pages 提交 Core 作业并查询；关闭管理入口不停止已有必要索引一致性工作 | W4 |

梦境五阶段必须分别给出效果用例。现有 `run_pending()` 能工作、某种索引能重建，都不能直接算作对应 v3 阶段已覆盖。

## 4. 主动陪伴、任务与人格学习

| ID | v3/已有功能与目标 | 独立 Pages 开关 | 承接方式与关闭语义 | 工作包 |
| --- | --- | --- | --- | --- |
| F31 | 主动发送总闸 | `proactive.enabled`，来源级 `egress.proactive` | 插件统一发送路径；关闭阻止新主动效果，任务/Focus 可继续保存和管理 | W5 |
| F32 | 群聊自然插话 `chime_in` | `proactive.chime_in` | 本地门控＋必要时统一决策；关闭此动机，其他提醒/跟进继续 | W5 |
| F33 | 指定用户/话题事件跟进 `follow_up` | `proactive.follow_up` | Core Focus＋新 Observation 信号＋统一发送；关闭发言动机不删除关注点 | W5 |
| F34 | 群聊静默主动发起 `initiate` | `proactive.group_initiate` | 共用静音、冷却、滚动限额与上下文；关闭不影响响应新消息 | W5 |
| F35 | 私聊静默关心 | `proactive.private_initiate` | 保留 v4 行为，与群聊独立；关闭后不以间隔 0 间接代替开关 | W5 |
| F36 | 被动回复后跟进评估 `watch` | `proactive.watch` | 可选有预算评估，只更新合法 Focus/门控；关闭不停止背景消息采集 | W5 |
| F37 | Task 提醒发送 | `proactive.task_reminders` | Core Task 到期，插件确定性发送；关闭提醒不完成/取消任务，仍服从 F31 | W5 |
| F38 | Task / Focus 管理和冷却工具 | `tasks.manage`、`focus.manage`、`proactive.cooldown_tool` | Pages/LLM 统一受权入口；管理和自动发言分开，关闭工具不绕过既有冷却规则 | W5/W6 |
| F39 | 完整人格编辑与采用 | `persona.manage`、`persona.apply` | 插件主记录、草稿/发布/历史/回滚/绑定/导入导出；采用关闭不删除人格，也不改变观察 Agent | W1/W6 |
| F40 | 人格自动采样及草稿生成 | `learning.persona_sampling`、`learning.persona_draft` | Core 授权来源引用＋插件生成工作流；采样无第二份长期聊天语料库 | W6 |
| F41 | 人格独立模型审查 | `learning.persona_review` | 独立审查 Provider 和有界预算；关闭后按人工发布路径，不能冒充自动审查通过 | W6 |
| F42 | 人格自动发布 | `learning.persona_auto_publish` | 默认关闭，开启需证据/审查/策略满足；发布前 CAS 与源重验；关闭不影响人工发布 | W6 |
| F43 | 表达模式学习与采用 | `learning.expressions`、`context.expressions` | Core 来源和公共候选表示＋插件审查/采用；关闭学习不阻止已批准模式的可选采用 | W6 |
| F44 | 黑话抽取/聚类、审查及匹配采用 | `learning.jargon`、`learning.jargon_review`、`context.jargon` | Core 公共表示、候选状态/来源＋插件审查；关闭审查不自动批准未审查黑话 | W6 |
| F45 | 对话样例学习与注入 | `learning.examples`、`context.examples` | Core 来源引用和版本；统一预算注入，关闭生成与关闭采用各自独立 | W6 |
| F46 | 学习候选衰减/清理 | `learning.retention` | Core 受控候选/来源生命周期，插件只清理已失效流程引用；不维护独立遗忘算法 | W6/W7 |

人格演进需恢复来源范围、定时/阈值触发、Run/Revision/Diff、审查、审批、发布前证据重验和回滚体验。这些管理步骤不是每一步都开一个定时器；可选自动行为按 F40～F42 控制，人工管理由 F39 控制。

## 5. 媒体、平台、数据与运行

| ID | v3/已有功能与目标 | 独立 Pages 开关 | 承接方式与关闭语义 | 工作包 |
| --- | --- | --- | --- | --- |
| F47 | 图片理解 | `media.image_description` | 原图消息先 Observation，描述异步派生；关闭不遗漏纯图片消息 | W6 |
| F48 | 图片内容哈希复用、感知去重和无效图过滤 | `media.cache`、`media.perceptual_reuse`、`media.analysis_filter` | 插件有界计算缓存；仅筛选/复用解析工作，不去掉原始事件，不能把近似图错误当相同证据 | W6 |
| F49 | 历史图片解析回填 | `media.backfill` | 使用 Core 可见来源和宿主可用媒体，幂等补派生引用；关闭不影响当前图片原始接收 | W6 |
| F50 | 引用补充和合并转发展开 | `platform.reply_enrichment`、`platform.forward_expansion` | 原始引用/结构始终保留；开关控制可选内容获取/上下文展开，保留转发来源真实性边界 | W2/W6 |
| F51 | 纯 @ 回复 | `platform.pure_at_reply` | 宿主消息触发与 Core 近期补充；关闭仅取消额外唤醒，纯 @ 仍是 Observation | W6 |
| F52 | 错误消息友好化 | `output.friendly_errors` | 插件输出适配；关闭恢复宿主行为，不把错误文本当用户事实 | W6 |
| F53 | Markdown 清理 | `output.strip_markdown` | 仅出站格式适配；已确认观察反映实际发出内容 | W6 |
| F54 | v3 输入清理 | `context.input_filter` | 可选模型输入清理；原始 Observation 不改写。授权、数据/指令区分和输出转义始终执行 | W6 |
| F55 | 旧管理命令及工具兼容 | `compat.commands_v3`、`compat.tools_v3` | 别名路由到新服务和相同权限/开关；不保留第二套实现；语义不等价时明确报错或说明 | W6 |
| F56 | 手动备份、导出/恢复 | `data.export`、`data.restore` | 插件控制数据＋Core 受权一致快照；恢复有预览、版本与引用检查；不自动替换正在使用的库 | W4 |
| F57 | 自动备份 | `data.scheduled_backup` | 使用同一备份接口和现有调度，独立时间/保留参数；关闭只停止新计划 | W7 |
| F58 | v3→v4 数据/配置迁移 | `compat.import_v3` | 只读预检、范围预览、幂等导入、映射报告与回退；开关开启本身不执行导入 | W7 |
| F59 | 详细运行日志及上下文正文 | `diagnostics.logs`、`diagnostics.content` | 脱敏、轮转、查询/导出；正文独立显式开启，关闭不停止基本错误和交付状态 | W1/W7 |
| F60 | 主动决策统计及检索路线诊断 | `diagnostics.proactive_stats`、`diagnostics.retrieval_details` | 有界持久统计；关闭不停止静音/冷却所需状态，也不关闭错误报告 | W5/W7 |
| F61 | 自动遗忘/学习/媒体/主动等模型成本治理 | 各行为按自身开关；可选明细为 `diagnostics.model_usage` | 共用预算/限并发属于运行约束，不能关闭某一业务后仍由另一独立循环调用同一模型工作；账本不随明细开关停用 | W1～W7 |

## 6. v3 内部开关的整合取舍

| 旧开关/机制 | 新计划 |
| --- | --- |
| 三段 FIFO、L1/L2/L3 独立注入字数与窗口 | 合并为 Core 原文/摘要/Recall＋一个动态总预算；保留类别独立采用开关和上限，旧数值不机械复制 |
| `image_parsing.mode=all/related`、`image_skip_on_passive_trigger` | 仅保留为 F47 的解析触发策略；绝不控制原始图片观察是否入库 |
| `l3_enable_type_whitelist` | Core 合法候选类型及证据校验不能禁用；若提供可选领域类型偏好，作为 F18 的高级策略公开，不绕过 schema |
| `enable_legacy_cleanup` | 不恢复自动删除整个宿主会话的默认行为；F17 提供有前提的请求上下文接管，完整历史删除另走明确数据管理范围 |
| 群隔离/画像隔离/人格隔离 | 显式主体与 Space 映射、F16 受权长期共享取代模糊隔离组合；关闭隔离不意味着关闭授权 |
| 多个旧队列、主管线预算、梦境自己的 worker | 合并传输与公共工作调度，各业务保留单独开关和配额；关闭业务不切断基础交付/失效处理 |
| 旧图谱、画像、黑话独立存储/导出格式 | 消费 Core 公共表示和一致数据管理，F58 负责旧格式导入；接口不全时标明外部阻塞 |

这是实现优化的范围，不是删掉对应用户能力的理由。删除历史、候选校验等正确性约束不能为了复刻旧开关而变成任意可关闭项。

## 7. 旧工具对应关系与完成判定

| v3 工具 | 新入口/归属 |
| --- | --- |
| `save_memory` | `iris_remember` / F09 |
| `search_memory` | `iris_recall` / F07 |
| `correct_memory` | `iris_correct` / F10 |
| `save_knowledge` | 新知识/关系写入入口 / F20，须冻结实际可写 Core 合同 |
| `search_knowledge_graph` | 图谱查询入口 / F19 |
| `get_profile` | 显式目标的画像查询入口 / F21/F22 |
| `add_follow_up`、`end_follow_up` | Core Focus 管理 / F38，跟进发言另由 F33 开启 |
| `set_cooldown` | 统一主动冷却工具 / F38 |

兼容别名开启与底层功能开启是两个条件；不能借旧名绕过新开关。Pages 的记住、画像和关注操作均显式选择参与者，不使用群里“最后说话的人”猜测主体。

每一行关闭前至少需要：旧场景或明确替代行为、真实后端处理、可见独立开关、关闭后的负向用例、依赖/权限错误说明、对应验证证据。只写页面卡片、配置字段、空方法或 Core capability 名称不能关闭缺项。

后续验收给每行标记“已验收 / 部分 / 外部阻塞 / 平台限制 / 明确替代”，并列剩余行为。不以开关数量、工具数量或代码量推算覆盖百分比；没有单独决定的功能仍在计划范围内。

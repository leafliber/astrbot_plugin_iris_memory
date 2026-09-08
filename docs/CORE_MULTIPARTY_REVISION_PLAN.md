# Core 修改方案：多人 Observation、跨空间 Agent 与可归属提炼

日期：2026-09-08。状态：**供 Core 项目实施评审的方案，尚未实施**。

适用场景：群聊、直播间、多人频道、私聊，以及同一个持续 Agent 在多个来源空间接收事件。消费方方案见 [插件 v4dev 修改方案](V4_OBSERVATION_REVISION_PLAN.md)。

## 1. 目标与边界

环境消息直接复用现有 Observation。保留 Observation → RecentContext / Episode / Claim / Relation / Profile / Index 的现有体系，不新建 EnvironmentMemory 类型或另一套环境记忆数据库。

本轮 Core 需要补齐三个主要接缝：

1. **多人事件输入：** 安全保留发言者、回复对象、提及和可确认的互动信号。
2. **主体/空间注册：** 将持续 Agent 与来源 Space 的公开注册分离，支持一个 Agent 的多个环境。
3. **认知输入与归属：** 为总结和提炼提供安全、明确的参与者信息；Scope 和实体映射由 Core 固定和校验。

为使上述场景实际可用，还需要两个配套接缝：授权观察接收与 Active Surface 发言权区分、Recent/Recall 返回足够的来源和显示信息。

Core 保持宿主无关：不 import AstrBot、直播平台 SDK 或插件对象，不负责平台连接、平台发言决策和具体消息发送。Python 库与 HTTP 服务/独立 SDK 必须共享同一业务实现；继续沿用 Core 的包分发方式。

文中的新增方法、字段版本、能力名称均为**建议契约**，不是现有可直接调用的 API。正式实现应遵循当前 operation/schema 生成流程，不同时手写三套定义。

## 2. 已有基础与真实缺口

基线为 Core 当前 `dev/chores` 工作区：HEAD `7629152` 加未提交的 Embedded/插件接缝；不是单独该提交或任意同版本 wheel。此次为源码核查，没有执行新功能测试。

| 位置 | 已有能力 | 需要补齐 |
| --- | --- | --- |
| `domain/observation.py`、`application/observation.py` | actor identity、可选 Entity 快照、scope、source/event/cursor、结构化 payload、原子 Batch | 交互元数据语义；服务端自动冻结 actor Entity 快照；来源级注册/授权检查 |
| `application/provisioning.py` | `create_agent/create_space/create_session`；底层允许 Agent 下多个 Space | 独立、持久、可恢复的公开 Space/Session 注册方法及 HTTP/SDK 消费入口 |
| `embedded.py` | `provision_agent()`、`register_actor()` | Agent/Space 耦合注册及 128 组合限制；本地/远程批量身份接缝和避免无限内存授权集合 |
| `application/reflection.py::_provider_observation()` | 传递 ID/revision/role/kind/time/content/payload/privacy | 没有安全发言者/参与者映射；默认 DTO 不能可靠解释多人“我/你/他” |
| `embedded_providers.py::_Cognitive` | 借用宿主异步模型，限制调用与生命周期 | 没有 HTTP Provider 等价的候选 Scope 绑定；新输入版本适配 |
| `application/reflection.py` 的 Claim 提交 | 候选可形成 Claim | 当前缺少 `subject_entity_id` 时会按 self 处理；多人新输入必须拒绝缺失主体，而非默认归 Iris |
| `application/recall.py`、`business_views.py` | Recent Observation 候选和资源引用 | 公共候选未完整提供发言者、时间、回复和 source event 等渲染/去重元数据 |
| `application/observation.py` | Required Surface 对 Batch 作调用前和提交时检查 | 授权的其他宿主不能仅因没有发言租约就丢失环境观察；需要新增受控接收策略 |

特别说明：当前只提交 `actor_external_identity_id` 时，校验路径主要检查租户，存储仍使用 Draft 中的 Entity 字段；不会自动补齐缺失的 `actor_entity_id_at_ingest`。不能把 schema 有字段当成服务端已经冻结了归属快照。

## 3. 核心模型约定

| 模型 | 本方案含义 |
| --- | --- |
| Agent | 持续 AI 身份及其记忆拥有者；不按群、观众、请求或人格修订创建 |
| Space | 该 Agent 所在的一个来源环境，如群、房间、私聊 |
| Session | 可选的有界交互/直播场次；不是每条消息或 LLM cycle |
| Entity / ExternalIdentity | 内容涉及的稳定主体与平台身份；不与 Agent 所有权混为一谈 |
| Observation | 已确认接收/发生的事件；人类旁观消息仍是正常 Observation |
| RecentContext | 有界、可重建的近期视图，恢复当前环境上下文 |
| Episode / Claim / Relation | 有来源的经历、事实和关系；归属与可见性由 Core 校验 |

原始观察不因总结而变成全 Agent 公开内容，不通过删除 `space_id/session_id` 放大可见性。一个 Agent 在不同 Space 的数据仍遵守现有 Scope、Privacy 和用途规则。

第一版只解决**一个 Agent 多个 Space**。同一真实房间被不同 Agent 观察时，可按现有 Agent 归属分别记录；不在本次引入跨 Agent 的全局内容共享、自动记忆合并或新的 ACL 系统。现有非 Agent 独占 Space 语义也不需要为了首版被扩大。

## 4. C1：公开、持久的主体/空间与身份注册

### 4.1 Agent 与 Space 分离

建议提供以下公共语义，名字以正式契约冻结为准：

| 建议操作 | 输入重点 | 返回/要求 |
| --- | --- | --- |
| `ensure_agent` | 宿主稳定主体键、显示名称 | 稳定 Agent ID；只允许可信本地管理能力或远程管理平面，普通观察凭据不能任意创建 Agent |
| `ensure_space` | 已授权 Agent、连接器来源命名空间、外部空间键、kind | 稳定 Space ID、revision、授权注册结果；新请求不携带任意授权 tenant |
| `ensure_session` | 已授权 Space、稳定场次/会话键 | 稳定 Session ID；未知真实会话时允许不调用 |
| 注册查询 | 按宿主注册键或分页查找 | 用于重启恢复和诊断，不必加载全部成员/空间到内存 |
| `resolve_actors` | 有上限的一批平台身份、可选显示名、来源注册引用 | 每项的 identity、绑定快照/版本或明确 unresolved/conflict；不返回无关人物资料 |

Source Namespace 区分平台真实身份命名空间与宿主连接实例。同一平台用户若 ID 在全平台稳定，可在授权域内复用；平台按应用/群/频道派生的 ID 必须使用不同 realm。跨平台同昵称不能合并。

### 4.2 持久化与授权

- 复用 ProvisioningService 和 IdentityService 的现有规则；公共 façade 不直写 SQL 绕过它们。
- 建立或复用 SQLite 中的来源注册记录，唯一键建议包含 tenant、授权的连接器命名空间、Agent 和外部空间键。显示名称不进入稳定唯一键；更新名称使用独立修订操作。
- 注册、创建和授予该来源使用权必须是原子事务或可恢复的持久操作；数据库成功后进程崩溃不能只因 JSON manifest 未写入就丢失注册。
- Embedded 启动从持久注册恢复可信权限。Remote 凭据只在预先授予的 Agent/来源范围内注册 Space/身份；注册成功后的可见范围由服务端产生，不能要求客户端自行修改 allowed sets。
- 普通 Remote 路径不得获取全局 admin。新增能力只表示受限来源注册；不授权任意绑定他人身份、跨空间读取或人格发布。
- `resolve_actors` 对可信平台原生身份可以按策略建立新 Entity/已确认绑定，真实性由获授权宿主连接器保证；平台缺失身份时返回 unresolved。已冲突、撤销或 tombstone 的身份不自动复活/重绑。
- 身份解析不等于同意所有用途或解除 Privacy。不能通过“注册观众”向调用方授予其所有历史私有信息，也不能把不断增长的观众 ID 塞入常驻授权集合。
- 按 tenant/Agent/连接器设置空间、身份新建速率和总量限制，采用 SQLite 查询及有界缓存。达到上限明确拒绝，不复用不相关空间。

原 `provision_agent()` 返回 Agent+默认 Space 的行为保留兼容；新方法解除每增加群聊就增加 Agent 的耦合。解除/停用注册与删除历史不同，必须保留引用完整性。

## 5. C2：多人 Observation 的身份与互动语义

### 5.1 复用既有顶层字段

发言者继续使用 `actor_external_identity_id`。服务端首次接收时在写事务中解析有效绑定，冻结 `actor_entity_id_at_ingest` 及必要绑定版本；未知绑定保持明确的 unresolved，不猜测。

兼容调用方提交的 Entity 快照，但必须连同 identity 校验；单独 Entity ID 仍然拒绝。服务端派生字段与调用方请求指纹分开处理：同一原始请求重试不因之后身份变化而产生伪幂等冲突；已有结果仍需经过当前授权检查。历史 at-ingest 快照与当前绑定视图不得互相覆盖。

首次派生快照原则上代表 Core 接收时的绑定。延迟上报的真实事件时间、外部身份及源证据单独保留；不能未经证明把当前绑定伪装成平台发生时已经验证的绑定。

`source_event_id` 在现有去重域内必须足够唯一，宿主按平台命名空间＋来源空间＋真实消息 ID 构造。多个房间的相同原生 ID 不应冲突；不同宿主监听同一来源时，只有经注册证明相同的来源才共享该命名空间。

### 5.2 标准化现有 structured_payload 的一个扩展

优先在 `structured_payload.interaction` 内定义有版本的通用交互结构，不新增 Observation 主体类型。下面是**拟新增结构示意**，现有 `observe.batch.v1` 不代表已经校验/解释它：

```json
{
  "interaction": {
    "version": 1,
    "speaker_display_name": "张三",
    "reply_to": {"source_event_id": "已带来源命名空间的消息键"},
    "mentions": [{"identity_id": "Core 外部身份引用", "basis": "platform_mention"}],
    "agent_signals": ["mentioned", "reply_target"]
  }
}
```

规定：

1. 消息主要发言者只在已有顶层 actor 字段表达，避免两处身份互相冲突。显示名来自发生时快照，仅用于显示。
2. mentions 是平台可验证的 @；正文提及和同名匹配不升级为已解析身份。未解析提及可以用独立的有限文本字段表达，但不授予实体权限。
3. 回复引用按来源命名空间解析，同一 Agent/Space 内校验目标存在、可见且未被撤回。默认不跨 Space 取正文；目标未到达/不可见时保持 unresolved，不编造对话。
4. `agent_signals` 是宿主接收时的明确信号。是否发言由宿主决定；字段不能授予发送权，也不证明 Agent 实际参与过。
5. 之后的参与由新助手 Observation 及其回复关系体现。解析迟到引用可以更新可重建投影，不偷偷改原始消息或既有幂等指纹。
6. Core 对已知版本严格校验类型、长度、数量及身份/来源授权；其他命名空间的原有 payload 继续兼容。未知交互版本只允许明确协商的原样保存/忽略策略，不承诺参与者提炼。
7. 该结构只能承载事件关系与元数据，不能带任意数据库定位器、可执行指令或远程抓取地址；Core 不根据回复 ID 自动发起平台网络请求。

建议初版每条引用/提及数设置较小上限，例如合计 32，交互元数据上限例如 8 KiB；这些是待压力测试后冻结的限额，不是现有约束。超限应明确拒绝或报告适配器有声明的省略，不能静默伪造完整参与者列表。

### 5.3 直播与多人事件的共性

| 场景 | 记录原则 |
| --- | --- |
| 群成员/观众弹幕/主播人类发言 | 正常 `user` 消息，使用稳定 actor；不因发生在直播间就合并为一个“大用户” |
| Iris 文本/语音输出 | 已确认实际生效的部分记录为 assistant，部分效果保留 proof；输入/输出源分别去重 |
| 其他机器人消息 | 使用其独立身份和约定角色，不能因它叫“Iris”就当成本 Agent |
| 礼物、入场、关注等 | 平台确认事件使用约定的 external kind 和结构化字段；礼物不自动推导亲密关系、承诺或用户意图 |
| 直播画面/工具描述 | 保留来源类型、生成者与原始引用；派生模型描述不能冒充观众陈述 |
| 高并发弹幕 | 一批多个 Observation，保留每条作者、时间与 ID；批处理节省事务/网络，不丢掉主体边界 |

消息去重不是正文去重。真实重复刷屏可在投影/模型输入中压缩呈现并列出覆盖来源，原始事件仍按配置持久保存。模型候选不能引用压缩展示虚构的字符 span；只能引用实际选入输入的原观察及其原文范围。

## 6. C3：带参与者上下文的认知输入与服务端绑定

### 6.1 新的显式输入版本

建议引入可协商的 `CognitiveInput v2`，由 Core 在固定窗口内构造，供本地异步 Provider 和远程 HTTP Provider 共用。当前 v1 方法默认行为保持兼容；v2 通过明确字段/方法版本启用，不靠探测任意 Python 参数或把任意字段塞给旧模型。

建议输入包含：

| 部分 | 内容 |
| --- | --- |
| 窗口描述 | 输入版本、窗口引用、起止时间、来源水位摘要、是否有源缺口、目标消息与仅供理解的邻接消息标识 |
| 参与者表 | 只包含本次可见观察涉及的有界参与者；窗口内稳定 handle、安全显示名、person/bot 等已验证类别、是否本 Agent |
| 观察列表 | observation ID/revision、原始 role/kind/time/content、actor handle、已验证 mentions/reply refs、被允许传出的结构化字段 |
| 任务约束 | 总结或提炼的目标、候选 schema、数量/时间预算；属于受信配置，不来源于用户正文 |

不提供整租户身份目录、所有群成员列表、完整人物画像或宿主平台凭据。Tenant/Agent 授权 ID 与有效 Scope 保留在服务端固定窗口中，不要求模型生成。

参与者 handle 如 `p1/p2/self` 只在本次窗口有效，Core 保存该次映射与版本用于校验/重放；名字“self”、昵称“管理员”或正文中的 `p2` 不会改变映射。相同源快照重放应重建相同映射，不能依赖全局当前说话人变量。

### 6.2 候选的主体与证据

v2 候选显式返回 `subject_ref`；关系使用 source/target handle；Core 将合法 handle 转成已有 Claim/Relation 的 Entity 字段后，再执行现有 schema/领域校验。v2 名称在契约冻结前仍属建议。

- “张三：我喜欢茶”可以形成关于 p1 的 Claim；不是关于当前提问者，也不是 self。
- “张三说李四喜欢茶”只能标为转述来源，不能升级为李四亲口确认；证据权威性由来源类型和规则决定。
- 发言者/指代无法确认时可以保留摘要或未归属 Note，但不能以省略主体触发 `subject_is_self=True`。
- self 必须显式选择且有合适证据；接收旁观消息不能证明 Agent 发言、参加活动或作出承诺。
- 模型输出未知 handle、窗口外 Entity、伪造源引用必须拒绝；不据此新建/合并身份。
- 不能从观众的计划自动创建 Active Task，不能直接修改 Binding、Persona Current 或执行外部动作，继续沿用既有候选限制。

证据仍指向原 Observation 的 ID/revision/span。字符坐标沿用现有实现的原文坐标，不把格式化后的“昵称：正文”作为新原文；摘要本身不成为一份新的独立事实证据。

### 6.3 Scope、身份及来源冻结

统一本地/HTTP 的绑定流程：

```text
固定授权与来源窗口
    → 解析/裁剪参与者与引用
    → Provider 生成无授权字段的候选
    → Core 解析 handle，绑定固定 Scope 和必要 Privacy
    → schema / 主体 / 证据 / 当前权限 / 来源修订校验
    → 幂等提交 Canonical 资源和后续索引作业
```

HTTP 现有 `bind_candidate_scope` 规则应抽到共用应用逻辑，并明确 Async 适配器契约。兼容 v1 中显式提供的 Scope：缺失字段按声明策略绑定；与固定窗口不符或扩权的字段拒绝，不静默覆盖后放行。

调用期间相关 Identity/Binding 变更、来源撤回/修订或权限撤销，都要在提交前重验。固定映射至少保存身份/绑定版本或摘要；不直接把旧候选重映射到另一个人。处理方式为按既定历史绑定规则验证，或报告 stale 并重建窗口。

窗口摘要、候选指纹和重放材料应包含输入 schema、参与者映射、来源 revision、Prompt/模型/策略版本。不能把模型自己的上轮输出作为下一轮新的独立证据。

### 6.4 有界总结/提炼，兼顾直播负载

复用现有 Outbox、固定水位、Provider 治理和持久进度，不引入分布式调度系统。

- 按 Agent＋Space＋明确 Session/时间边界分窗，不混合不同群原文。
- 按目标消息数、字符数和最大等待时间触发；近期消息先可用，后台模型慢不会阻塞 Observe。
- 不对每条弹幕调用模型/Embedding；批量处理并保持逐条证据。先复用已有总结与提炼流水线，一次模型同时输出两者可另行测评，不作为此次接缝改造前提。
- 固定窗口大于容量时拆分并保存继续位置；不能仅 `LIMIT 500` 后将整个更大范围标为完成。邻接上下文与本批目标分开，重叠部分不能重复计作新的提炼来源。
- 同一 Agent 的多个 Space 公平调度，直播高流量不能饿死群聊和纠正/遗忘通道；只需有界轮转、配额与低优先级认知作业。
- 当前原子 Observe + Outbox 行为保持；可合并投影/调度唤醒，但不能丢弃已接受的事件和处理水位。
- 身份表仅覆盖本次有界窗口，所有不透明历史资料都不传给模型。连接器、参与者缓存和模型在途任务均有上限。

## 7. C4：区分观察接收与发言权

这是同 Agent 多宿主并行观察的配套要求。当前 Required Active Surface 对整批 Observation 检查租约；不能在插件里靠反复抢租约或关闭服务端校验绕过。

建议增加**明确启用、受来源授权约束的 observation ingress 策略/能力**：

- 已授权采集来源的 user/external 入站事件可以在没有主动发言租约时提交；tenant、Agent、Space、来源、身份、Privacy 与幂等检查继续执行。
- 仅凭 `role=user` 或 `agent_signals` 不得获得豁免，须由服务端凭据/注册和声明的事件类型共同判断。宿主仍负责真实平台效果证明，Core 不声称能独立证明任意 JSON 来自平台。
- assistant/tool 的实际效果、任务/人格写入等继续按各自严格规则处理，不能借入站权限伪造助手承诺。
- 发言决定、租约获取/续租/释放仍在现有 Active Surface 机制中；多个来源能观察不代表多个宿主都能主动发言。
- 新策略默认不改变旧 Required 客户端行为。发布时明确 capability、服务端配置、失败分类和双入口测试。
- 延迟到达的已确认助手效果若发生在旧 lease epoch，继续遵循现有效果证明规则；本次不因“延迟”直接免除检查。如现有接口不支持该证明，单列待办。

## 8. C5：Recent/Recall 的多人来源视图

复用现有 RecentContext 和 Recall 路由，给公共候选增加版本化、可选的 `provenance/interaction` 元数据，至少包括：

- 原观察 ID/revision、来源事件键、发生时间、Space/Session。
- 可见发言者的安全标签和受控引用；隐藏身份时返回明确的匿名标签，不泄露真实账号。
- 经过授权解析的回复/提及关系；无法解析与没有回复要区分。
- 摘要的覆盖来源/范围及完整性标识；大量来源可用有界分页引用或 coverage token，不把上千个 ID 塞到每个候选。
- 原始观察、模型派生描述、摘要和当前有效 Claim 的类别区别。

Recent 请求默认限定同一 Space。Session 级查询与同 Space 背景补充需有显式模式/授权，而不是修改 Scope null 匹配规则。原始观察不会因 Agent 相同自动出现在其他房间。

插件去重时可用 source IDs/coverage；缺少来源对应关系不能假装已精确去重。Core 仍负责 Tombstone、版本回读与 Privacy；不把一次返回的引用当作永久可读权限，也不因只是显示昵称而绕开参与者隐私。

## 9. 本地、HTTP、SDK 契约与兼容

1. 以 `contracts/source/contracts.json` 为契约来源，生成公共 API/OpenAPI/JSON Schema 并同步 Python/TypeScript SDK；服务端调用共享应用服务。
2. C1 公共操作通过 Embedded façade 和 HTTP/SDK 提供等价语义；远程不要求调用方借用 Console admin 密钥或私有 `_request_json`。
3. 新 interaction 与 cognitive input 版本分别协商；旧 Observe 不带新字段仍可保存，但不能宣称具有新版多人提炼保证。
4. 仅把 JSON 塞入 `structured_payload` 并得到 200，不等于 Core 支持交互校验、主体解析和模型输入。Capability 必须对应真实装配与测试。
5. 候选能力名可参考 `host.spaces.v1`、`identity.resolve-batch.v1`、`observation.interaction.v1`、`cognition.input.v2`、`observe.ingress.v1`、`recent.provenance.v1`；均待正式命名/版本冻结，不能提前当作已发布值。
6. 包版本、HTTP 契约版本、认知输入版本和数据库 schema 分别管理。建立新的明确发行快照，不再次依赖“同版本号但接口不同”的隐式判断。
7. 不改变 client-only 依赖隔离；SDK 不依赖 Core、SQLite、向量引擎。新参与者/注册功能不引入常驻模型、重型队列或新数据库。

## 10. 持久化迁移与错误处理

Observation 原表和 JSON payload 优先复用。新增注册或参与者关系索引有必要时使用 SQLite migration；索引存引用，不复制正文。Core 自己管理 migration、备份恢复和安装资源打包。

旧记录缺失 actor/interaction 时按 unknown 展示，不从昵称猜填历史。旧 JSON bootstrap 注册可做一次幂等导入并校验现有 Agent/Space 归属；不能自动合并不同 Agent 的历史。

分别返回身份未解析/绑定冲突、来源未授权、空间未注册、引用不可见、能力不支持、版本不支持、容量满、候选 stale 和 Provider 失败。Observe 持久成功与认知 pending/failed 分开，客户端不得用空结果吞掉失败。

批量业务需保留原有原子提交规则。新身份解析的每项结果与实际写入事务边界要明确；“批次部分成功”的新操作不能伪装成原 Observe all-or-nothing。结果未知写入的重试仍使用原请求及幂等键。

## 11. 实施工作包

| 顺序 | 工作包 | 最小可验收交付 |
| --- | --- | --- |
| C0 | ADR 与契约冻结 | 不新增环境记忆类型；身份命名空间、Scope、输入版本、权限边界及兼容规则明确 |
| C1 | 注册接缝 | 同一 Agent 新建/恢复多个 Space，批量身份解析，本地/HTTP/SDK 公共链路 |
| C2 | Observation 互动信息 | 服务端 actor 快照、interaction 校验与来源去重；可保留未解析引用 |
| C3 | 认知输入及候选处理 | 有界参与者表、显式主体、共用 Scope 绑定、来源/身份版本提交校验 |
| C4 | 授权入站观察 | Required 发言权与观察接收的可选区分；旧策略保持兼容 |
| C5 | 近期来源视图 | 多人上下文渲染/去重所需的公共数据，覆盖范围与匿名策略 |
| C6 | 安装与验收 | 新 wheel/SDK、双入口共享用例、源码以外安装 smoke、文档与迁移说明 |

C1/C2 与 C3/C5 可以在接口冻结后分工。优先完成明确的小步闭环，避免先换检索引擎或重做 IAM。

## 12. 验收清单

| 场景 | 必须证明 |
| --- | --- |
| 同 Agent 两群＋直播间 | 一个 Agent、不同 Space；重启/同键并发不重复注册，私有原文不互串 |
| 两人“我喜欢……” | 两条 Claim 分属正确 Entity；不落到提问者或 self |
| 同名、改名、跨应用 openid | 显示变化不改身份；无证据不合并，不同 realm 隔离 |
| 转述、@、回复、缺失引用 | 分清发言者与事实主体；未解析目标不猜测；无跨空间越权补文 |
| 主体缺失、伪造 handle | v2 候选被明确拒绝/降级，不利用旧 self 默认值 |
| Binding/来源在模型期间变化 | 旧候选不误归属；revision、Tombstone、Privacy 重验 |
| 异步 Provider 无 Scope | Core 绑定后真实写入且能 Recall；模型显式扩权被拒绝 |
| 并发多窗口 | handle/Scope/参与者映射无全局串用，取消不关闭宿主共享 Provider |
| 直播爆发与安静群并存 | 队列/缓存/模型任务有界；群聊和纠正不被饿死；所有接受的原始事件保留可追踪进度 |
| 超过单窗口容量 | 多个有界窗口完整推进，不将未处理余量标为完成；重复/乱序不重复提炼 |
| 一主体两个宿主 | 新策略下授权观察可同时接收，非持有者不能据此发言；旧 Required 行为仍有回归测试 |
| Recent 与宿主历史重叠 | 返回正确来源和覆盖，摘要部分重叠可解释；更改/删除后不返回旧有效事实 |
| 无稳定 ID /游标重置 | 明确退化边界；有真实 stream epoch 时不冲突，不伪造连续游标 |
| Fake 模型端到端 | 两入口均完成观察→Recent→总结/Claim→Recall，来源与主体可验证 |
| 真实模型与安装产物 | 小规模中文多人/直播场景质量另行记录；wheel/SDK 干净环境通过，不把 Fake 测试当模型质量证明 |

初步性能验收可使用不同负载档位的合成事件，例如 1/20/100 条每秒，分别报告接受率、持久队列增长、Observe P95、模型调用数、峰值 RSS 和恢复耗时；这些是测试输入建议，不是吞吐承诺。以目标部署机器和配置冻结标准，不借远程模式隐藏服务端成本。

## 13. 不在本次自动完成的内容

平台连接/发言算法属于宿主；全量 v3 数据迁移、跨 Agent 历史合并、未知人物自动身份链接、完整直播平台适配、检索引擎替换和群聊概率插话不属于这三个主要接缝的实现前提。

完成标准是可安装的同一 Core 引擎与独立 SDK，经过多人双入口用例验证；不能仅新增字段/Protocol/空方法或把原文拼入 Prompt 就宣布完成。

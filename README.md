# Iris Memory 4.0.0-dev

面向 AstrBot 的 Iris 宿主插件。AstrBot 负责聊天平台、回复模型和实际发送；同一台电脑上的 **Iris 服务**负责近期消息、长期记忆、人物要点、人格、目标、提醒与后台整理。插件不再内置 v3 的记忆数据库、知识图谱或梦境引擎。

v4 在 Iris 服务基础上提供自然插话、目标跟进、静默发起、观察和独立开关的私聊关心。**每个会话默认关闭主动回复**，由管理员显式开启。

## 安装与首次连接

要求 **AstrBot >=4.28.2,<5，Python >=3.12**。AstrBot 4.28 的 `requires-python` 是 `>=3.12`；已核对 4.28.2 wheel 的消息、工具和 Pages 接口。插件运行依赖仅 `httpx`，Iris Python 客户端已内置，无需另装 `iris-client`。

1. 在运行 AstrBot 的同一台电脑安装 [iris-core](https://github.com/AirObject/iris-core)，按其 README 完成首次设置。先启动 Iris：

   ```sh
   cd /path/to/iris-core
   uv sync
   uv run iris --data-dir /path/to/iris-data serve --no-open
   ```

   浏览器打开 `http://127.0.0.1:8080`。在 Iris 管理页配置角色、学习／对话和向量模型；没有模型也能接收消息及进行已有记忆的全文检索，但不能据此认为新消息已经学会。AstrBot 的回复模型仍在 AstrBot 中配置。

2. 为插件创建宿主令牌，建议仅授权 `astrbot:` 前缀。Iris 的宿主令牌与管理员登录凭据不同。若使用离线命令，先停止 Iris，沿用上一步的数据目录：

   ```sh
   uv run iris --data-dir /path/to/iris-data tokens create --host astrbot --prefix 'astrbot:'
   ```

   将一次性返回值直接填入 AstrBot 插件配置，不要保存到代码、聊天、截图或日志。随后重新启动 Iris。也可使用 Iris 管理端的宿主令牌接口在线创建。**本插件只读取配置中的令牌，不读取环境变量或凭据文件。**

3. 将本仓库的 v4 功能分支完整安装到 AstrBot 的 `data/plugins/astrbot_plugin_iris_memory/`，由 AstrBot 安装 `requirements.txt`。测试版安装 `dev.4`；主动回复 PR 合入前使用 `feat/v4-proactive`。默认 `master` 仍是 v3。停用重复的旧记忆／主动回复插件后重载。

4. 在插件配置中填写 Iris 地址（默认 `http://127.0.0.1:8080`）和令牌。地址只能是 `127.0.0.1`、`localhost` 或 `[::1]` 的 HTTP(S) origin。Iris 默认只监听回环地址；分开容器的 `127.0.0.1` 不是同一个网络空间，本版不提供跨机部署方案。

5. 将 AstrBot 的 **`provider_ltm_settings.group_message_max_cnt=0`**。插件在启动及接收事件时检查，冲突会出现在日志、`status` 和 Pages。`prepare` 成功后默认清空本次请求的旧 `contexts`，使用 Iris 的近期消息；Iris 不可用或超时则保留 AstrBot 原请求继续回复。

6. 执行 `/iris_mem status`，或打开 AstrBot 插件详情 → Pages → 状态。确认 Iris 已连接，并核对实际入口映射。

一个 Iris 服务代表一个角色；连接多个 AstrBot persona 不会自动得到独立的 Iris 人格。需要角色隔离时应使用独立 Iris 数据目录／实例及插件配置实例。

## v3 功能如何承接

| v3 大功能 | v4 实际承接 | 状态 |
| --- | --- | --- |
| L1 近期上下文、接管 AstrBot 上下文 | `/prepare` 的 `recent_messages`；成功时接管本次 contexts，检查框架配置 | 已实现 |
| L2 长期记忆、遗忘、梦境加工 | Iris 接收消息并后台学习、维护、整理；插件注入准备材料 | 已接入，质量取决于 Iris 与模型配置 |
| L3 知识图谱、用户／群画像 | 人物要点与按人物定向检索；群画像等通过可选入口画像注入 | 人物查询已接入；入口画像待 Iris 提供，不再维护独立图谱、画像表或好感度 |
| `search_memory` | `/memories/search`，始终带当前入口 | 已实现 |
| `save_memory`、`correct_memory` | 等待当前消息接收后，对本入口调用 `/learn` | 已实现；按真实对话学习，不直接编辑、不伪造工具参数为用户原话 |
| `get_profile`、`search_knowledge_graph` | 按人物名字／主体 ID 查询，保留立场、相信程度、来源 | 已实现；同名需澄清 |
| `save_knowledge` | 使用 `save_memory` 的学习流程 | 不再注册旧工具 |
| `add_follow_up`、`end_follow_up` | 本入口 Iris 目标、稳定 host_key、修订冲突保护 | 已实现；完成／放弃须明确，不由提醒自动触发 |
| 人格自学习 | Iris persona 经 prepare 注入，管理页确认更新 | 已接入；`persona` 指令显示版本与说明 |
| 图片理解 | 下载／读取图片 → `/media` → 消息 `media_ids`；理解由 Iris 负责或复用宿主说明 | 已实现 |
| 纯 @ 回复、错误友好化、Markdown 清理 | 轻量小工具，分别开关 | 已重写 |
| `/iris_mem` 指令 | status、search、persona、goals、learn、help | 已实现；管理操作返回 Iris 管理页地址 |
| Dashboard | AstrBot Pages：连接、健康、用量、入口映射、主动回复设置与统计、管理页地址 | 已实现；不显示令牌 |
| 自然跟话、跟进、主动发起、观察、冷却 | 本地门控、单次 AstrBot Provider 决策、Iris 准备材料、逐会话设置与统计 | 已实现；包含 `set_cooldown` 和独立私聊关心开关 |
| 表达、黑话与迁移 | 表达模式／黑话由 Iris 入口画像承接；对话样例学习未接入；v3 数据迁移不做 | 详见下方完整对照 |

## 指令和工具

```text
/iris_mem status
/iris_mem search 桂花茶
/iris_mem persona
/iris_mem goals
/iris_mem learn
/iris_mem help
```

删除、编辑、清空、导出、导入请在 Iris 管理页操作。v3 的管理子命令会引导到该地址。`goals` 最多展示本入口 100 条开放目标；更多目标在管理页查看。`learn` 返回接受／暂停情况，不返回“保存成功”。

八个工具各有开关（含管理员 `set_cooldown`），另有总开关 `tools_enabled`。搜索和人物工具只检索当前入口可见材料。跟进创建始终携带本入口 `entry_id`；结束前分页查询并校验归属，再提交 `expected_revision`。宿主 v1 的 PATCH 不接受 `entry_id` 字段，因此归属检查在提交前完成。全局及其他入口目标不通过插件结束；冲突和合并跳转均要求重新查询，不自动跟随。创建／修改无入口的全局目标需要 Iris 的 `all` 范围令牌，本插件不使用该权限。

## 主动回复

管理员在需要的群或私聊执行：

```text
/iris_reply enable
/iris_reply disable
/iris_reply status
/iris_reply reset
/iris_reply cooldown [分钟]
/iris_reply willingness [低|中|高]
/iris_reply initiate
```

指令与 `set_cooldown` 工具只接受 **AstrBot 管理员**；群管理员身份本身不等于 AstrBot 管理员。Pages 使用 AstrBot 登录权限，可按会话修改开关、意愿、冷却和静音时段，查看调用／发言的动机分布。`reset` 明确重置统计、每日调用额度、滚动发起额度、冷却与关注锚点，保留会话开关、意愿、静音设置。`cooldown` 不填时为 5 分钟，管理员指令／Pages 可用 0 清除；工具只允许 1—120 分钟。

| 动机 | 行为 |
| --- | --- |
| `chime_in` | 本地点名／别名、问句信号触发自然插话候选。低意愿只接受点名或关注对象；中意愿加问句；高意愿加轻量求助词。已被 AstrBot 唤醒的直接回复、指令与收到的私聊消息仍走正常管道，避免双重回复。 |
| `follow_up` | 关注对象就是本入口开放的 Iris 目标。`add_follow_up` 正文写明人物名字／账号或话题；`end_follow_up` 明确完成／放弃。用人物出现、话题词重合和短期关注锚点做本地门控，决策结合 prepare 中的目标判断是否发言。 |
| `initiate` | 开启群自动发起后，达到静默底线才进行概率试验；通过后随机预约。收到新消息取消预约，发送前再次检查新消息、普通回复、管理员设置、静音、冷却和限额。静音结束或电脑休眠跨过静音窗口后重新随机延后。 |
| `watch` | 决策选择不发言时只更新已有 Iris 目标的关注锚点，不创建目标、不完成目标；可另开低频观察，对至少 10 字的非触发消息评估。即使模型在 watch 模式要求发言，也不会发送。 |
| 私聊关心 | `proactive_private_care_enabled` 独立开关，另需在该私聊 enable；按 initiate 相同的预约、限额和复查规则执行。收到私聊时仍由 AstrBot 正常回复。 |

本地门控未通过时不调用决策模型。通过后先等待当前输入被 Iris 接收，再取一次 prepare，使用 persona、记忆、近期消息、状态、目标、可选入口画像和安全运行提示；随后**一次调用 AstrBot Provider**，同时输出是否发言、最终内容、话题、`why_now`、`topic_source`、冷却建议和关注目标 ID。不再进行第二轮生成，不执行工具。Iris 自身的召回判断仍受原 `judge` 配置控制。

`proactive_provider_id` 为空时使用当前会话选中的 Provider。JSON 不合法、发言缺乏话题／时机／来源说明、内容过长、超时、期间有新活动时不发送，失败调用仍计入每日额度。成功后直接使用 PL-1 的 `self_output` 回流队列；发送 False、异常或取消不回流、不增加发言统计。使用反馈仍受 `assume_injected_used` 的原有近似规则控制。

| 默认参数 | 值 |
| --- | --- |
| 总开关 / 每会话开关 | 允许 / **关闭** |
| 群自动发起 / 私聊关心 / 定期观察 | **均关闭**，分别配置 |
| 意愿 / 别名 | 中 / Iris、鸢尾 |
| 发言后冷却 / 两次决策间隔 | 5 分钟 / 30 秒 |
| 静音时段 | 01:00–07:00，AstrBot 所在电脑时区；相同起止时间表示不静音 |
| 每会话每日决策上限 | 20 次；按本机日期换日，包含观察、失败及超时 |
| 最小静默 / 概率检查 | 120 分钟 / 每 5 分钟 20%，再按意愿调整 |
| 随机预约延后 | 5—30 分钟 |
| 成功发起滚动 24 小时上限 / 最短间隔 | 2 次 / 360 分钟；连续两次无人接话也暂停，直到有人发言 |
| 决策／发送超时 / 发言长度 | 每次 60 秒 / 300 字符 |
| 定期观察间隔 / 锚点有效期 | 10 分钟 / 10 分钟 |

`/iris_reply initiate` 是管理员手动尝试，跳过自动发起开关、概率和静默底线；仍要求会话开启，并遵守总开关、私聊关心开关、静音、冷却和限额。重启后保留设置和统计，旧预约取消，等待该会话首次新消息后恢复自动调度。

本地目标缓存至多读取本入口 1000 个开放目标，5 分钟刷新；工具创建／结束会使缓存立即失效，空列表或失败 30 秒后重试。最终材料仍受 Iris prepare 的目标预算约束；匹配目标未出现在准备材料中时不发言。目标内容的词匹配只用于减少模型调用，不代表语义理解、实际记住或目标完成。系统没有 v3 独立的关注对象数据库。

## 入口画像与暂不提供的 v3 功能

`$entry_profile` 为可选分区：Iris 返回 `{text, version, generated_at, possibly_stale}` 时注入这些字段；字段缺失或为 `null` 时替换为空字符串。旧 Iris 无需立即升级也可继续运行，主动决策同样兼容。画像是可能过时的描述，不是指令。生成群画像、群氛围、表达模式与黑话的工作由 **Iris 入口画像**承接，插件不再独立学习或存储这些内容。

以下逐项对应 v3，避免把“转移到 Iris”误认为完整保留了原交互：

| v3 项目 | v4 本版状态／后续归属 |
| --- | --- |
| 表达学习、采纳流水线 | 不保留原插件流水线；表达模式由 Iris 入口画像承接，待服务提供。 |
| 黑话抽取、聚类、审核、匹配注入 | 不保留独立词库；由 Iris 入口画像承接，待服务提供。 |
| 对话样例学习与注入 | 暂不提供。 |
| 独立知识图谱画布、节点／边编辑、路径扩展、`save_knowledge` | 不提供；人物要点、人物查询与 `save_memory` 学习流程已接入。 |
| 群画像、群氛围 | 可选入口画像接线已完成；由 Iris 生成并提供。 |
| 好感度演变与注入 | 暂不提供。 |
| 独立结构化用户画像表 | 不提供；已由人物定向记忆查询承接。 |
| 聊天工具直接写入／按 ID 纠错 | 不提供直接编辑；聊天中立即学习真实对话，精确编辑使用 Iris 管理页。 |
| 按用户／群／全部清空 | 不在插件删除；已改由 Iris 管理页批量遗忘／删除承接，具体范围能力以服务版本为准。 |
| 检索策略选择、独立 L1/L2/L3 注入预算 | 不保留插件策略切换；使用 Iris 检索与统一预算。 |
| 五个梦境阶段独立开关 | 不保留；由 Iris 维护、整理统一调度。 |
| AstrBot Embedding／本地 sentence-transformers | 不内置；在 Iris 配置向量模型。 |
| 直接进化 AstrBot 指定 Persona 及采样、差异、冲突、自动发布、回滚流水线 | 不保留 AstrBot Persona 修改；Iris 人格版本经 prepare 注入，在 Iris 管理页确认更新。 |
| 图片感知哈希复用、去重、无效图过滤、旧版配额策略 | 暂不提供完整 v3 策略；已有类型、单张大小、张数及超时限制。 |
| 历史图片回填 | 暂不提供。 |
| 合并转发展开 | 暂不提供；只接 AstrBot 已解析的标准消息链。 |
| QQ 官方平台专用细节 | 已支持稳定显示标签、群／私聊／频道身份隔离；完整平台适配仍需真实环境验收，不猜测平台未提供的信息。 |
| 模型输入净化 | 不搬运 v3 完整净化流水线；已有材料字段投影、预算和数据／指令区分。 |
| 完整 L1/L2/L3／画像／学习／人格数据 Dashboard | 不保留；使用 AstrBot 状态／主动回复面板和 Iris 管理页。 |
| 隐藏参数、界面偏好、日志／检索诊断页面 | 完整旧页面暂不提供；主动决策调用、发言与动机统计已实现，参数在 AstrBot 配置和 Pages。 |
| 插件导入导出、自动备份 | 导入导出使用 Iris 管理页；自动备份已改由 Iris 定时备份承接，服务能力未发布时插件不代做。 |
| v3 数据／配置迁移（记忆、画像、白名单等） | **不做，用户已决定**；不读取或转换 v3 数据目录。 |
| 全部 v3 指令／工具别名 | 只保留本 README 列出的接口；旧管理操作引导到 Iris 管理页，部分名称语义已改变。 |

## 入口、可见范围与平台

默认入口为 `astrbot:<平台实例ID>:<group|private>:<群号或账号>`；平台实例 ID 不同即入口不同，群内按个人划分的 AstrBot 会话仍进入同一个群入口。组成部分先百分号转义，超长映射使用稳定摘要。模板支持 `$prefix/$platform/$platform_id/$kind/$session_id`；`entry_overrides` 是按 `unified_msg_origin` 精确覆盖的 JSON。入口不可含 `/`、`\`、`?`、`#`、控制字符，不能为 `.`、`..` 或空白。

**令牌范围与记忆可见范围是两层规则。** 本插件始终发送入口参数；仍需在 Iris 设置入口的 `entry_only` 或指定入口列表，才能限制跨群记忆。Iris 默认 `shared`，不要把前缀令牌误当成私密记忆隔离。故意将多个会话覆盖到同一入口会共享原始近期消息。

- **aiocqhttp / OneBot**：通过 AstrBot 标准消息链处理群／私聊、账号、引用、提及和图片。同平台实例的同账号跨群／私聊保持同一主体。
- **qq_official / qq_official_webhook**：群、单聊、频道账号命名空间分别隔离；无昵称时使用稳定标签。引用与图片使用 AstrBot 已解析的组件。平台没提供的昵称、引用作者、其他用户提及不猜测；不拉取合并转发。主动发言使用 AstrBot `Context.send_message`；4.28.2 的 QQ 官方适配器不保证支持此通路，平台频控和主动消息权限仍需真实环境验收。失败不回流。
- **其他平台**：通过 AstrBot 通用事件与消息组件接入；能力取决于适配器实际提供的信息。尚未连接真实 QQ 平台验收。

## 配置与可靠性

全部配置在 `_conf_schema.json`，AstrBot 配置页有中文说明。默认 prepare 总预算 8 秒、召回判断 5 秒且开启；记忆最多 8 条／1500 token，近期最多 20 条，本地注入最多 12000 字符。图片最多 4 张，每张 10 MiB，下载与上传共用 2 秒；失败保留描述／占位继续文字流程。默认内存队列 256 条、后台操作预算 1 秒。

图片下载使用不带 Iris 鉴权的独立客户端，支持 HTTP(S)、本地文件、file URI、base64 组件。不会把媒体 URL、文件路径或 Base64 当作聊天正文。音频／视频／附件本版仅保留占位，不做额外理解。

默认 `force_non_streaming=true`，使用 AstrBot 已支持的事件覆盖。插件只在事件 `send` 正常返回后回流 `self_output`；发送异常、False、取消，以及仅触发 `after_message_sent` 都不是确认。原生流式发送、其他 Context 主动发送、插件监听前的发送不保证捕获；平台内部吞错也无法证明用户已读。

默认 `assume_injected_used=false`。可选近似反馈只有在成功 LLM 回复后，对完整注入的 `relevant` 记忆提交；人物要点、裁掉材料、普通指令结果和失败发送不强化。该近似可能过度强化，请有明确需要再开启。主动回复沿用这套规则；其他可信宿主可通过客户端提交有实际证据的反馈。

提醒游标和收件箱在 SQLite 同一事务保存，按服务／令牌摘要分开，令牌本身不落库。默认 `context` 模式，只注入本次 prepare 目标分区中对应的提醒；成功 LLM 回复后才本地消费。该消费不保证模型说出提醒或用户已读。

`active` 提醒需明确配置目标 `unified_msg_origin` 和 Iris 入口，含义是把令牌范围内提醒集中发送到该会话，应使用相应窄范围令牌。**目标会话必须已 enable，且本次启动已观察到消息**；也遵守主动回复总开关、静音、普通回复互斥与冷却，关闭时提醒留在收件箱。False／异常保留重试；发送与本地确认之间崩溃可能重发。目标去重通知仅作控制通知，未知种类保留本地；积压 1000 条后暂停拉取。通知取走、发出、被读与目标完成是不同状态。

会话设置、冷却、统计和滚动发起记录保存在插件数据目录的独立 SQLite 文件，按服务／令牌摘要隔离；不保存令牌、完整模型提示或原始决策。换令牌会重新默认关闭会话。默认意愿和静音时段在首次观察会话时写入；已有会话保留设置，通过指令／Pages 调整。

消息与输出队列仅在内存中，溢出／后台失败记固定故障码，需要平台重放；不承诺持久补发、恰好一次送达或 24 小时稳定性。缺少平台消息 ID 时仅保证同事件内去重。日志不记录令牌、HTTP 头、异常原文或聊天内容。

## 常见问题

**主动回复没发言？** 先在目标会话由管理员执行 `/iris_reply enable` 和 `status`。检查总开关、静音、冷却、每日调用额度；自动发起另需打开 `proactive_initiate_enabled`，私聊需打开独立关心开关。无信号、决策选择观察或接收到新消息也不会发送。

**主动回复太频繁？** 降低意愿、延长冷却，或在 Pages 关闭该会话；`/iris_reply disable` 同时停止发往它的主动提醒。

**无法连接或 401/403？** 先启动同机 Iris，检查端口、令牌是否撤销及最终入口是否在范围内。管理页 Cookie 不能代替宿主令牌；令牌更换后重载插件。Iris 接口不接受自定义 Host 域名。

**收到消息却没记住？** 接收不等于学习，成功学习也不必然形成记忆。检查 `status` 的模型健康、积压和用量，在 Iris 管理页检查批次、记忆缺口与运行记录。`learn` 只是请求尽快处理；空模型配置不会产生新记忆。

**重复上下文或第三人称复述？** 设 `group_message_max_cnt=0`，停用重复插件，确认没有把不同群映射到同一个入口。Iris persona 为空时继续沿用 AstrBot 人设。

**纯 @ 没回复？** 开启插件 `pure_at_reply`，关闭 AstrBot 的 `empty_mention_waiting`，确认是只提及机器人且没有正文、其他提及或附件。插件通过标准 LLM 管道使用 Iris 材料；不会伪造用户聊天内容。

**Markdown 没有清理？** 准备转为图片的长文保留 Markdown；显式 `use_t2i_=true` 也保留。普通代码正文和下划线标识符不会被当作格式删除。

**图片没有理解？** 检查图片大小、预算和 Iris 图片模型配置。上传与消息接收不等待模型，后台理解失败会降级。默认预算适合本机网络，可按需要增大。

**Dashboard 上的 localhost 打不开？** 地址是 Iris 所在电脑的回环地址。远程浏览 AstrBot 的设备不能靠自己的 localhost 访问它；请在 Iris 所在电脑打开管理页。状态页通过 AstrBot bridge 请求，浏览器不接触宿主令牌。

**v3 数据和配置会迁移吗？** 不会。本次清空 dev.4 实现并保留 Git 历史，v3 数据目录不读取、不转换。部署前自行保留旧环境；不能直接复用 v3 的存储文件。

## 开发与验证

推荐把环境、缓存、完整日志和临时数据库放在仓库外。以下路径由使用者选择：

```sh
export UV_CACHE_DIR=/path/to/iris-eval-artifacts/pl2/uv-cache
export UV_PROJECT_ENVIRONMENT=/path/to/iris-eval-artifacts/pl2/venv312
uv sync --locked --python 3.12
IRIS_CORE_PATH=/path/to/iris-core uv run --locked pytest -q \
  -o cache_dir=/path/to/iris-eval-artifacts/pl2/pytest-cache \
  --basetemp=/path/to/iris-eval-artifacts/pl2/pytest-tmp
uv run --locked ruff check --no-cache .
uv run --locked ruff format --check .
```

真实集成测试使用 `IRIS_CORE_PATH/.venv/bin/python`，可通过 `IRIS_CORE_PYTHON` 改为其他已准备的 Python；只读导入 core 源码，设置不写字节码。它在临时目录离线创建令牌（只在内存捕获），空模型配置启动真实 HTTP 服务，覆盖接收→prepare→search→goals→notifications，也验证去重、空反馈和明确结束。不会读取真实模型配置、密码文件或既有凭据。未设置 `IRIS_CORE_PATH` 时该项明确跳过；提 PR 必须启用运行。

其余测试涵盖同步／异步客户端、AstrBot 替身、假 HTTP 服务、接收／注入／回流／反馈、提醒重启、图片限额、工具、指令、平台映射和小工具；假 Provider 还覆盖全部主动动机、门控、限额、静音／休眠恢复、并发复查、回流、权限、Pages 与设置持久化。测试通过不等于真实平台或模型学习质量已经验收。

来源和许可状态见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。连接器原始约束快照见 [CONNECTOR_DESIGN.md](docs/CONNECTOR_DESIGN.md)；该文件保留为上游设计快照，本仓库新增主动策略与配置差异以本 README 和实现为准。

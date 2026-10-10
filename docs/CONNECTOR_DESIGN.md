# AstrBot ↔ Iris：宿主接口 v1

## 参考与验证范围

第一步只读参考 iris-core `main@e727a000d5c1778aeb508d7c5112321242af23b8`：README 的宿主／令牌／状态／目标／媒体说明，DECISIONS 2026-10-10「M4 计划」，设计 1.2、11.2、11.3、15、18.2、19，`api.py`、`trial.py` 和 `trial_reply_v1.md`。

第二步对齐 `main@43f8221` 冻结契约：`docs/host-api.md` 和 OpenAPI 示例；接口差异集中在客户端 compat/types、插件 media/prompt 中。

实际安装并核对 **AstrBot 4.28.2**（Python 3.12.14，独立临时环境安装 PyPI wheel，`--no-deps`，用于检查真实 API 与实现；没有启动平台或安装平台运行依赖）。在线参考：[Star 最小插件](https://docs.astrbot.app/dev/star/guides/simple.html)、[消息与 LLM 钩子](https://docs.astrbot.app/dev/star/guides/listen-message-event.html)、[插件配置](https://docs.astrbot.app/dev/star/guides/plugin-config.html)。网页可能先于／晚于 wheel 更新，行为以下列已安装源码为准。

| 接口／实现位置（AstrBot 4.28.2） | 本插件的使用方式 |
| --- | --- |
| `astrbot.api.star.Star`，`initialize/terminate` | 创建并关闭后台工作与 HTTP client |
| `filter.event_message_type(ALL, priority=10000)` | 收集进入 AstrBot 管道的群聊／私聊事件，不要求唤醒 LLM |
| `filter.on_llm_request`，`ProviderRequest.system_prompt` | 回复生成之前注入本轮材料 |
| `filter.after_message_sent`，`pipeline/respond/stage.py` | 观察发送后事件；不单独作为发送成功证据 |
| `AstrMessageEvent.send` | 实例级包装，保存参数快照，在调用正常返回后回流 |
| `Context.send_message` | 仅用于管理员配置的主动提醒，False 不确认 |
| `StarTools.get_data_dir` | 插件自己的持久化提醒收件箱 |

`after_message_sent` 在响应阶段捕获发送异常后仍会调用，空结果、原生流式路径还可能不调用。插件不会把 LLM 生成完成、装饰结果或 `get_result()` 当作已发证据。包装仅限当前事件实例，保留原方法的返回、异常和取消，不修改平台类。单次发送可包含文本和媒体占位；分段成功的部分分别回流，失败部分不回流。同一 hook 再调用不会重复回流。平台适配器正常返回只代表其发送调用成功，不承诺用户已读；若适配器内部吞错，插件无法获得更强证据。

为使核心路径可确认输出，`force_non_streaming=true` 默认在接收事件时设置 AstrBot 已支持的 `enable_streaming=false`。此覆盖在 `process_stage/method/agent_sub_stages/internal.py` 中先于请求使用。关闭该开关后，流式 fallback 若调用该事件的 `send`，仍能记录成功片段；原生流式、工具通过其他 Context 实例主动发送，以及在本插件监听前发送的其他插件消息不保证捕获。本步不拦截整个宿主；这几类需要平台级确认适配后再扩大支持。

## 入口、账号、引用与媒体

默认入口：`astrbot:<platform_id>:<group|private>:<session_id>`。这里 `platform_id` 是适配器**实例** ID，不是可能重复的类型名；群聊使用群号，私聊使用发送者账号，避免 AstrBot 按群内个人切分会话时把一个群拆成多个入口。组成部分先百分号转义，超过 200 字符用带前缀的 SHA-256 固定 ID，保证稳定。

`entry_prefix` 默认 `astrbot:`；`entry_template` 支持 `$prefix/$platform/$platform_id/$kind/$session_id`（这些是可用变量，不是斜杠分隔格式）；`entry_overrides` 是以原始 `unified_msg_origin` 为键的精确覆盖 JSON。覆盖优先于模板。规则若故意把不同来源映到同一个入口，会共享原始近期消息，管理员应按其期望配置。令牌范围应覆盖最终入口 ID。

账号以 Iris `platform=astrbot:<平台类型>:<实例ID>` 与原始 `account_id` 联合隔离；显示名来自 `get_sender_name()`，不会用显示名合并账号。私聊和群聊中同平台实例、同账号是同一个主体。引用独立提交 `quote_author/quote_author_account_id/quote_content`，不混进当前说话人的正文。输入去重键为入口与平台消息 ID 的摘要；缺少平台 ID 时为当前事件生成 UUID，仅保证事件内重入去重，重启重放需要平台补齐稳定 ID。

消息时间使用平台 timestamp 转 UTC，有问题时用接收时间。已成功发送的输出使用 `kind=self_output, sender=我`，独立发送序号键与实际发送时间。主动提醒的输出使用通知 ID 稳定键、配置的入口 ID 和目标会话的群／私聊类型。

图片按 AstrBot 4.28.2 `Image.url` 优先、`file` 次之、`path` 兜底取源；支持 HTTP(S)、本地路径/file URI，以及组件的 base64://。实现集中于 `media.py`，不调用会自行落盘或打印 URL 的通用转换流程。URL 下载使用独立无鉴权 HTTP client，最多三次重定向；绝不携带 Iris Bearer 头。根据文件头识别 PNG/JPEG/GIF/WebP，限制 10 MiB，Base64 上传后再提交消息的 media_ids。可信宿主 extra `iris_image_descriptions: list[str]` 按图片顺序附 understanding_text；文字描述也保留在消息中。

默认启用图片，最多四张，总下载＋上传预算 2 秒，均可配置。图片过多、读取／上传失败或超时使用描述／占位，已成功上传的仍引用，随后照常提交文本。音频／视频／附件保留占位。本地文件读取在线程执行，网络与上传可取消；不会无限等待。上传重试可能产生孤立对象，由 Iris 宽限清理。接收成功不代表图片已理解或消息已学习。

## 回复准备与提示模板

收到事件立即放入 256 项有界队列，图片也在后台处理。`on_llm_request` 在同一个总预算内等待接收成功，再请求 prepare；默认 **8 秒**，召回判断默认 **开启**，判断预算 **5 秒**，三个值可配置。依据规划者给出的 M2 无答案误返结果，默认开启判断以减少无关记忆注入。HTTP 单次上限至少覆盖 prepare 总预算；后台非图片操作仍受单独默认 1 秒总预算约束。图片消息的后台预算为图片预算＋提交预算。总等待超时、异常或输入未成功提交时保留宿主原提示继续回复，并只记录固定降级码。

默认从 Iris 本入口近期消息推导参与者与查询；同名账号不在插件内猜主体。可信宿主可逐事件提供 `iris_participants`、`iris_text`（含空字符串）和 `iris_known_memory_ids`。插件不跨轮缓存“已知记忆”，因为无法确定 AstrBot 上下文何时被裁剪；只把宿主明确声明仍在上下文的 ID 传给 Iris。

`prompt.py` 投影 persona、记忆、近期消息、状态、目标／询问、运行提示和待处理提醒。模板使用 Python `string.Template`，支持 `$persona/$memories/$recent_messages/$state/$goals/$hints/$reminders`，没有 eval 或任意字段访问。运行提示只按已识别 code 翻译固定中文语句；不注入 `message/last_error/calls/judgment` 等诊断。R02 other_entries_pending 只接受正整数 entry_count、带时区且有序的 time_from/time_to，以及 received_at 时间基准，转换成 UTC 数量／时间说明；忽略服务端 message、入口标识和所有额外字段。未知 code 不注入。

默认注入上限为 12000 Unicode 字符（不是 token；不含宿主原系统提示与连接换行）。各分区按模板出现次数分配额度；记忆／提醒按完整条目放入，近期消息优先保留最新；过大 persona／状态整体省略为 null。不会截断成半段 JSON。仅实际完整出现在模板里的 relevant 记忆可进入可选使用反馈；person_highlight 不算已用。文案明确记忆、引用与消息是数据，保留立场和相信程度，不允许它们取得控制权限；persona 为空时沿用宿主身份，不虚构设定。

按本任务要求，材料追加到系统提示。新版 AstrBot 文档建议动态材料放额外用户内容块以保留提示缓存；这里采用用户指定的系统提示方案，可能降低缓存复用。以后改变消息角色需作为单独行为变更评审。

## 使用反馈

默认 `assume_injected_used=false`，仅注入不足以判断真实使用。开启后，在有成功发送且结果属于 AstrBot LLM_RESULT 时，对完整注入的 relevant ID 提交一次 feedback。该选项把“注入”近似为“使用”，可能过度强化；不会对人物要点、被裁掉条目、失败发送或普通命令结果报使用。宿主若有更准确的证据，可直接用客户端 `feedback`。插件不解析 LLM 自报的任意 ID，不自动改变目标状态。

## 提醒与本地可靠性

每 30 秒拉一页，最多 30 条；`after` 来自 SQLite 收件箱。页面内容和 `next_cursor` 在一个事务中保存，再进行任何主动发送。重启读取保存的游标，通知 ID 去重，重复拉取不会复活已处理条目。地址与令牌的单向摘要隔离检查点，数据库不保存凭据；换服务／令牌会重新拉取，已取通知可能重放，不能据此声称恰好一次送达。

默认 context 模式：只选择同时出现在本次 prepare 目标分区中的 goal_reminder，可用 `reminder_entry_id` 再限定入口。提醒注入后仍保留，直到发生成功 LLM 回复；这里的本地消费含义是“材料已提供并发生回复”，并不证明模型说出了提醒，更不证明用户已读。模板省略／预算裁掉、LLM 失败或发送失败均保留。

active 模式必须显式配置 `reminder_target`（AstrBot unified_msg_origin）和 `reminder_entry_id`。通知本身目前不含入口，不能猜接收者。配置表示把此令牌范围内拉到的目标提醒集中发送到这个会话；Iris 的窄范围令牌应与此用途匹配。`Context.send_message` 成功才移出本地待处理列表并提交 self_output；False／异常下次轮询再试，不完成目标。崩溃若发生在平台已发送、本地确认尚未提交之间，可能重复发送，这是 HTTP 拉取和平台发送之间的已知窗口。

goal_dedup_result 为控制通知，记录固定计数码后移出本地待处理队列，不发送或注入。goal_reminder 的 soon/due/overdue/immediate 由 Iris 判定。未识别种类保留本地，不直接发送或注入。收件箱最多积压 1000 项后暂停拉取并记录 `reminder_inbox_full`，避免无限增长。消息／输出提交队列为内存队列，后台操作默认 1 秒总预算、最多一次内部重试；失败或队列满记录固定故障码，丢失项需要宿主重放，**本步不宣称持久消息补发或 24 小时门槛通过**。

## 故障与配置

`main.py` 只做 AstrBot 接线；缺令牌或无效配置会禁用桥接并留日志，不阻断机器人。运行异常只写固定操作码，无异常字符串、聊天正文、HTTP 头、URL 或令牌。令牌只从 AstrBot 插件配置的 `token` 或环境 `IRIS_HOST_TOKEN` 读取；插件配置优先，UI 标记 `secret=true`，不另读凭据文件。

完整配置在 `_conf_schema.json`：地址、令牌、入口规则、模板、提示预算、队列／超时、注入／判断／反馈／非流式开关，以及提醒周期、方式和目标。插件卸载取消任务、解开等待并关闭客户端。后台错误计数在 `Bridge.counts`，日志可据固定码排障。

## 安装与测试

这是 monorepo，AstrBot 插件市场不能直接把仓库根目录当插件安装。本步使用本地部署：

1. 在 **AstrBot 使用的 Python 环境**安装本仓库 `python/`：`uv pip install --python <AstrBot环境的python> /绝对路径/iris-connectors/python`。
2. 将 `astrbot/astrbot_plugin_iris/` 整个目录部署到 AstrBot 的 `data/plugins/astrbot_plugin_iris/`，保留子模块、metadata 和配置 schema。目录中的 requirements 依赖已安装的 `iris-client==0.1.0`；本步没有发布 PyPI/npm 包。
3. 在插件设置提供地址和令牌，重载插件；按需开启提醒主动模式。保留默认非流式覆盖可确认核心回复路径。

本次执行仅构建仓库，没有写入任何真实 AstrBot 数据目录。测试 `uv run --locked pytest` 使用事件、上下文、Star、装饰器替身，不连接真实平台。HTTP 单元测试为 respx；TypeScript 为 mock fetch。另用临时数据目录、空模型配置和内存内离线令牌跑真实 Iris HTTP 冒烟，材料留在指定仓库外目录。

客户端 v1 对齐和模拟运行边界见 [根 README](../README.md)。图片、超时、提示白名单、通知种类均有替身测试；没有连接真实平台。

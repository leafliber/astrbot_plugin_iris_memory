# Iris Memory 4.0.0-dev

面向 AstrBot 的 Iris 宿主插件。AstrBot 负责聊天平台、回复模型和实际发送；同一台电脑上的 **Iris 服务**负责近期消息、长期记忆、人物要点、人格、目标、提醒与后台整理。插件不再内置 v3 的记忆数据库、知识图谱或梦境引擎。

本次是 **PL-1 基础重构**。自然跟话、主动发起和主动回复面板由 PL-2 实现；当前提供 Iris 目标与连接器原有的提醒。

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

3. 将本仓库的 v4 功能分支完整安装到 AstrBot 的 `data/plugins/astrbot_plugin_iris_memory/`，由 AstrBot 安装 `requirements.txt`。测试版 PR 合入前，请确认安装的是 `feat/v4-core`，默认 `master` 仍是 v3。停用重复的旧记忆／主动回复插件后重载。

4. 在插件配置中填写 Iris 地址（默认 `http://127.0.0.1:8080`）和令牌。地址只能是 `127.0.0.1`、`localhost` 或 `[::1]` 的 HTTP(S) origin。Iris 默认只监听回环地址；分开容器的 `127.0.0.1` 不是同一个网络空间，本版不提供跨机部署方案。

5. 将 AstrBot 的 **`provider_ltm_settings.group_message_max_cnt=0`**。插件在启动及接收事件时检查，冲突会出现在日志、`status` 和 Pages。`prepare` 成功后默认清空本次请求的旧 `contexts`，使用 Iris 的近期消息；Iris 不可用或超时则保留 AstrBot 原请求继续回复。

6. 执行 `/iris_mem status`，或打开 AstrBot 插件详情 → Pages → 状态。确认 Iris 已连接，并核对实际入口映射。

一个 Iris 服务代表一个角色；连接多个 AstrBot persona 不会自动得到独立的 Iris 人格。需要角色隔离时应使用独立 Iris 数据目录／实例及插件配置实例。

## v3 功能如何承接

| v3 大功能 | v4 实际承接 | PL-1 状态 |
| --- | --- | --- |
| L1 近期上下文、接管 AstrBot 上下文 | `/prepare` 的 `recent_messages`；成功时接管本次 contexts，检查框架配置 | 已实现 |
| L2 长期记忆、遗忘、梦境加工 | Iris 接收消息并后台学习、维护、整理；插件注入准备材料 | 已接入，质量取决于 Iris 与模型配置 |
| L3 知识图谱、用户／群画像 | 人物要点与按人物定向检索；群信息作为入口可见记忆查询 | 已实现承接；不再有独立图谱、画像表或好感度 |
| `search_memory` | `/memories/search`，始终带当前入口 | 已实现 |
| `save_memory`、`correct_memory` | 等待当前消息接收后，对本入口调用 `/learn` | 已实现；按真实对话学习，不直接编辑、不伪造工具参数为用户原话 |
| `get_profile`、`search_knowledge_graph` | 按人物名字／主体 ID 查询，保留立场、相信程度、来源 | 已实现；同名需澄清 |
| `save_knowledge` | 使用 `save_memory` 的学习流程 | 不再注册旧工具 |
| `add_follow_up`、`end_follow_up` | 本入口 Iris 目标、稳定 host_key、修订冲突保护 | 已实现；完成／放弃须明确，不由提醒自动触发 |
| 人格自学习 | Iris persona 经 prepare 注入，管理页确认更新 | 已接入；`persona` 指令显示版本与说明 |
| 图片理解 | 下载／读取图片 → `/media` → 消息 `media_ids`；理解由 Iris 负责或复用宿主说明 | 已实现 |
| 纯 @ 回复、错误友好化、Markdown 清理 | 轻量小工具，分别开关 | 已重写 |
| `/iris_mem` 指令 | status、search、persona、goals、learn、help | 已实现；管理操作返回 Iris 管理页地址 |
| Dashboard | AstrBot Pages：连接、健康、用量、入口映射、令牌是否配置、管理页地址 | 已实现；不显示令牌 |
| 自然跟话、主动发起、冷却、主动回复面板 | `iris_memory/proactive/` 挂接点 | PL-2；本版没有这些策略和 `set_cooldown` 工具 |
| 表达学习、黑话、对话样例、v3 数据迁移 | 不搬运旧数据库与旧学习管线 | 本版不做 |

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

七个工具各有开关，另有总开关 `tools_enabled`。搜索和人物工具只检索当前入口可见材料。跟进创建始终携带本入口 `entry_id`；结束前分页查询并校验归属，再提交 `expected_revision`。宿主 v1 的 PATCH 不接受 `entry_id` 字段，因此归属检查在提交前完成。全局及其他入口目标不通过插件结束；冲突和合并跳转均要求重新查询，不自动跟随。创建／修改无入口的全局目标需要 Iris 的 `all` 范围令牌，PL-1 不使用该权限。

## 入口、可见范围与平台

默认入口为 `astrbot:<平台实例ID>:<group|private>:<群号或账号>`；平台实例 ID 不同即入口不同，群内按个人划分的 AstrBot 会话仍进入同一个群入口。组成部分先百分号转义，超长映射使用稳定摘要。模板支持 `$prefix/$platform/$platform_id/$kind/$session_id`；`entry_overrides` 是按 `unified_msg_origin` 精确覆盖的 JSON。入口不可含 `/`、`\`、`?`、`#`、控制字符，不能为 `.`、`..` 或空白。

**令牌范围与记忆可见范围是两层规则。** 本插件始终发送入口参数；仍需在 Iris 设置入口的 `entry_only` 或指定入口列表，才能限制跨群记忆。Iris 默认 `shared`，不要把前缀令牌误当成私密记忆隔离。故意将多个会话覆盖到同一入口会共享原始近期消息。

- **aiocqhttp / OneBot**：通过 AstrBot 标准消息链处理群／私聊、账号、引用、提及和图片。同平台实例的同账号跨群／私聊保持同一主体。
- **qq_official / qq_official_webhook**：群、单聊、频道账号命名空间分别隔离；无昵称时使用稳定标签。引用与图片使用 AstrBot 已解析的组件。平台没提供的昵称、引用作者、其他用户提及不猜测；不拉取合并转发。官方平台频控及主动消息权限仍由平台决定。
- **其他平台**：通过 AstrBot 通用事件与消息组件接入；能力取决于适配器实际提供的信息。尚未连接真实 QQ 平台验收。

## 配置与可靠性

全部配置在 `_conf_schema.json`，AstrBot 配置页有中文说明。默认 prepare 总预算 8 秒、召回判断 5 秒且开启；记忆最多 8 条／1500 token，近期最多 20 条，本地注入最多 12000 字符。图片最多 4 张，每张 10 MiB，下载与上传共用 2 秒；失败保留描述／占位继续文字流程。默认内存队列 256 条、后台操作预算 1 秒。

图片下载使用不带 Iris 鉴权的独立客户端，支持 HTTP(S)、本地文件、file URI、base64 组件。不会把媒体 URL、文件路径或 Base64 当作聊天正文。音频／视频／附件本版仅保留占位，不做额外理解。

默认 `force_non_streaming=true`，使用 AstrBot 已支持的事件覆盖。插件只在事件 `send` 正常返回后回流 `self_output`；发送异常、False、取消，以及仅触发 `after_message_sent` 都不是确认。原生流式发送、其他 Context 主动发送、插件监听前的发送不保证捕获；平台内部吞错也无法证明用户已读。

默认 `assume_injected_used=false`。可选近似反馈只有在成功 LLM 回复后，对完整注入的 `relevant` 记忆提交；人物要点、裁掉材料、普通指令结果和失败发送不强化。该近似可能过度强化，请有明确需要再开启。PL-2 或其他可信宿主可以通过客户端提交有实际证据的反馈。

提醒游标和收件箱在 SQLite 同一事务保存，按服务／令牌摘要分开，令牌本身不落库。默认 `context` 模式，只注入本次 prepare 目标分区中对应的提醒；成功 LLM 回复后才本地消费。该消费不保证模型说出提醒或用户已读。

`active` 提醒需明确配置目标 `unified_msg_origin` 和 Iris 入口，含义是把令牌范围内提醒集中发送到该会话，应使用相应窄范围令牌。False／异常保留重试；发送与本地确认之间崩溃可能重发。目标去重通知仅作控制通知，未知种类保留本地；积压 1000 条后暂停拉取。通知取走、发出、被读与目标完成是不同状态。

消息与输出队列仅在内存中，溢出／后台失败记固定故障码，需要平台重放；不承诺持久补发、恰好一次送达或 24 小时稳定性。缺少平台消息 ID 时仅保证同事件内去重。日志不记录令牌、HTTP 头、异常原文或聊天内容。

## 常见问题

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
export UV_CACHE_DIR=/path/to/iris-eval-artifacts/pl1/uv-cache
export UV_PROJECT_ENVIRONMENT=/path/to/iris-eval-artifacts/pl1/venv312
uv sync --locked --python 3.12
IRIS_CORE_PATH=/path/to/iris-core uv run --locked pytest -q \
  -o cache_dir=/path/to/iris-eval-artifacts/pl1/pytest-cache \
  --basetemp=/path/to/iris-eval-artifacts/pl1/pytest-tmp
uv run --locked ruff check --no-cache .
uv run --locked ruff format --check .
```

真实集成测试使用 `IRIS_CORE_PATH/.venv/bin/python`，可通过 `IRIS_CORE_PYTHON` 改为其他已准备的 Python；只读导入 core 源码，设置不写字节码。它在临时目录离线创建令牌（只在内存捕获），空模型配置启动真实 HTTP 服务，覆盖接收→prepare→search→goals→notifications，也验证去重、空反馈和明确结束。不会读取真实模型配置、密码文件或既有凭据。未设置 `IRIS_CORE_PATH` 时该项明确跳过；提 PR 必须启用运行。

其余测试涵盖同步／异步客户端、AstrBot 替身、假 HTTP 服务、接收／注入／回流／反馈、提醒重启、图片限额、工具、指令、平台映射和小工具。测试通过不等于真实平台或模型学习质量已经验收。

来源和许可状态见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。连接器原始约束快照见 [CONNECTOR_DESIGN.md](docs/CONNECTOR_DESIGN.md)，本仓库差异与 PL-2 挂接见 [PL2_HANDOFF.md](docs/PL2_HANDOFF.md)。

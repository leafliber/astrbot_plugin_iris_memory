# Iris Memory v4

从零构建的轻量 AstrBot 插件：插件管理配置、完整人格和实际交互；记忆存储、检索、画像、任务与关注点交给 Iris Memory Core。

当前为 **`4.0.0-dev.1` 开发候选**，所在分支 `dev.4`。双模式、Pages、完整人格、显式记忆和基础提醒已有实现；真实模型质量对照、24 小时长稳和全部平台验收尚未完成。

当前尚不能完整替代 v3：群聊插话/事件跟进、表达和黑话学习、完整画像/图谱管理、范围删除、配套备份恢复及 v3 数据迁移仍有缺项；助手消息与自动记忆沉淀也未形成完整闭环。

下一轮方向是范围内所有消息进入 Observation，插件保持薄层，逐项恢复或优化 v3 功能，并在 Pages 独立控制采集、加工、采用和发送。当前**仅计划，未开始实施**，既有代码检查点为 `4a8e0d0`。详细工作文档放在本地 `docs/`，已排除 Git 跟踪；本 README 独立提供安装和运行所需信息。

## 使用方式

| 模式 | 插件调用 | 数据与模型 |
| --- | --- | --- |
| 本地 `local` | import `iris_memory_core.embedded.EmbeddedMemory` | Core 在 AstrBot 进程中运行，数据保存在插件持久目录；可借用 AstrBot Embedding Provider |
| 远程 `remote` | `iris_memory_sdk.AsyncIrisMemoryClient` | 数据、索引和模型由远程 Core 服务维护 |

本插件不构建或发行 Python 包，也不复制 Core/SDK 源码。插件加载、配置和人格编辑不依赖 Core/SDK 已安装；开启记忆模块时才加载所选后端。生产插件不启动 Web 服务器、sidecar 或监听端口。

## 安装开发候选

使用 Python **3.12+**；AstrBot 接口参考版本为 **4.28.0**。先将本分支插件文件置于 AstrBot 的插件目录，通过 AstrBot 加载。复制工作目录时排除 `.venv`、`.local-backups`、测试缓存和本地数据。

Core/SDK 的接缝目前属于未发布开发快照。原消费验证的 wheel 版本虽然显示 Core `0.15.0`、SDK `0.11.1`，同版本号的其他产物不一定含有这些接口。以下为 **2026-09-08 原消费快照**，不代表 Core 正在进行的 Observation 改造已经通过插件验收：

| wheel | SHA-256 |
| --- | --- |
| `iris_memory_core-0.15.0-py3-none-any.whl` | `c31e39b024b057035d4323c1d721329518cae2de9ceb5a165b9b7355a7a89647` |
| `iris_memory_sdk-0.11.1-py3-none-any.whl` | `0ab86f1ecfc707e5a9702b8474f66e4a3d675d4a6a52b48b82666ceceb2fe982` |

请取得对应 Core 项目的构建产物，核对摘要，在 **AstrBot 实际使用的 Python 环境** 安装：

```sh
# 本地模式：把路径替换为取得的实际 wheel 路径
python -m pip install /path/to/iris_memory_core-0.15.0-py3-none-any.whl
python tools/check_dependencies.py --mode local

# 远程模式：只需要独立 SDK
python -m pip install /path/to/iris_memory_sdk-0.11.1-py3-none-any.whl
python tools/check_dependencies.py --mode remote
```

同时使用两种模式可以安装两个 wheel。仅本地基础模式不需要 SDK、FastAPI、FAISS、NumPy；启用向量时按 Core 文档安装同一 Core wheel 的 `[vector]` 依赖，并在 Pages 手动提交索引重建。只安装 SDK 时不需要 Core。

`requirements.txt` 没有指向同版本的未经验证注册表包，避免自动装错。正式依赖发布前需要手动安装开发 wheel；这不是插件自身的 pip 分发入口。

本地 Core 当前采用 Unix `flock`，Windows 暂不支持。本次 macOS 测试环境 SQLite 为 `3.50.4`，使用 `allow_local_sqlite` 开发例外完成接口验证；正式使用须满足 Core 的 SQLite allowlist。Pages 中“开发 SQLite 例外”默认关闭，不作为生产兼容承诺。不要让独立服务和 Embedded 共用同一个数据目录。

## 首次配置

1. AstrBot 的插件 JSON Schema **只保留** `core_mode`。选择本地或远程后重载插件。
2. 打开插件 Pages 的 **Iris** 页面 → **配置管理**，其他设置全部在这里修改。
3. 在目标群聊或私聊发送 `/iris status`，将返回的会话键添加到“启用会话键”。空列表不采集消息。
4. 在“功能模块”同时开启需要的功能及依赖。例如基础记忆勾选“记忆”和“上下文”。所有可选模块初始关闭。
5. 若需要插件人格，在“人格”保存完整正文、发布，然后设置默认人格或绑定会话；开启“人格采用”。

配置使用 SQLite 修订号控制并发，应用新配置时停止旧模块并启动新模块，启动失败回退旧配置。关闭超时则保留资源所有权、停止接纳工作，不通过另建 Runtime 绕过排空。人格草稿不会自动采用，发布在下一轮对话生效。

### 远程绑定

每个“会话键 / 人格 ID”需要独立的、已授权的 Core Agent/Space。未启用人格时人格部分填写 `default`。插件拒绝不同绑定共用同一个 Agent，避免会话串记忆和人格镜像互相覆盖。

```json
{
  "chat:实际会话键/default": {
    "agent_id": "服务端预配置的 Agent ID",
    "space_id": "服务端预配置的 Space ID",
    "realm": "与服务端身份一致的 realm",
    "entity_id": "私聊用户的实体 ID"
  }
}
```

`entity_id` 是私聊简化配置。群聊显式记忆需要在“远程用户实体映射”配置每位用户：`{"realm/平台用户ID":"entity_id"}`，不会将群内所有人当成同一实体。服务端身份绑定必须已存在；JSON Scope 不能替代服务端凭据授权。

服务端启用 Required Active Surface 时，填写“远程应用实例 ID”，应用凭据需有 `active-surface.v1`；插件按需取得/续租，关闭释放，不抢占其他宿主。人格镜像需 `persona.mirror.v1`，远程不会调用 AstrBot 模型或复制宿主密钥。更换模式或 URL 不包含数据迁移；远程故障不自动本地写入。

### 本地会话容量

当前 Core 没有公开的独立会话 Space 创建方法。插件使用公开 `provision_agent()` 给每个“会话 / 人格”分配一个 Agent/Space，所有组合共用一个 Runtime。Core 当前最多保存 **128 个注册组合**；达到上限会报错，不复用不相关会话的 Scope。人格修订不占用新组合。解除绑定、停用人格不会自动删除 Core 注册或迁移历史记忆。

## 功能

| 模块 | 已实现的行为 |
| --- | --- |
| 记忆 | 用户观察交付队列、显式事实写入/纠正/遗忘、统一 Recall、画像读取；失败保留分类和请求 ID |
| 上下文 | 保留宿主历史、系统规则和工具；按剩余窗口选择完整候选，去重，临时注入，报告 Usage |
| 人格采用 | 插件主记录、草稿/发布/历史/回滚、绑定、导入导出；AstrBot 执行镜像继承工具/技能权限；Core 镜像 CAS 与回读验证 |
| 主动陪伴 | Core Task 提醒、静音/冷却/补发期限；可选私聊静默关心，复用人格和上下文；发送前核对任务、会话及人格 |
| 人格学习 | 依据有效 Claim 生成完整正文草稿，生成后与发布前校验证据；必须手动发布 |
| 图片描述 | 每条最多两图，共用插件模型预算；标记模型描述来源，不冒充用户陈述 |
| 后台维护 | 一个低频 worker 调用 Core 有界维护批次；按权限手动重建索引；远程 worker 由服务端管理 |
| 完整运行日志 | 可选结构化日志、异常栈、配置/模块/Core/模型/注入/发送关联、轮转、脱敏、查询与导出 |

Task 和 Focus 均保存在 Core；插件仅保存路由与发送回执。提醒发送不自动完成用户任务。主动发送的“accepted”只表示平台调用返回，不能保证用户送达；unknown 不盲重发，也不写成 committed 助手记忆。

`iris_recall`、`iris_remember`、`iris_correct`、`iris_forget`、`iris_task` 是 LLM 工具；每次执行仍检查当前模块与会话范围。人格发布、配置和模块管理只在已登录的 Pages 管理界面进行。

“遗忘此条事实”删除指定 Claim。Core 的原始观察记录不随 Claim 一并删除；页面会明确说明。上下文不把原始观察再次当成长期事实注入，避免纠正/遗忘后旧对话被当成有效事实。历史消息仍由 AstrBot 管理。

## 成本与持久化

- 插件只开一个 SQLite 连接和一个数据库执行线程；所有队列有条数/字节上限。
- 模型默认不用于额外的逐轮决策；确定性提醒不调用模型。自动私聊发起的间隔默认 `0`，即关闭。
- 插件主动/学习/媒体/Embedding 共享持久日调用与 Token 预算，先预留，超时不退款。预留以 UTF-8 字节估算并加输出上限，偏保守，不宣称等于账单 Token。AstrBot 自身被动回复不计入此预算。
- 详细日志默认关闭；正文默认只记录摘要。显式开启正文后仍脱敏已知凭据；日志可包含对话，请按需启用。默认轮转总量 100 MiB、保留 7 天。
- `StarTools.get_data_dir()` 下使用 `plugin.sqlite3`、`core/`（仅本地记忆启用）、`logs/`（仅详细日志启用）；不写源码或 site-packages。

**自动记忆提炼暂不可用**：本地 `AsyncCognitiveAdapter` 无法从现有观察 DTO 得到候选必需的 tenant/agent Scope，当前桥接也未补绑定。填写本地“认知 Provider”会得到 `core_cognitive_scope_missing`，不静默运行无效候选；须在 Core 完成固定窗口候选绑定并通过新安装物验证后接入。远程服务自行配置的认知 Provider 不受插件这条本地检查控制。

## 开发与验证

```sh
python -m pip install -r requirements-dev.txt
# 先安装上文两个已验证 wheel；远程集成测试还需要 Core 的 server 可选依赖
python -m pytest -q
ruff check iris_memory main.py tests tools
ruff format --check iris_memory main.py tests tools
node --check pages/iris/app.js
python tools/check_dependencies.py --mode local
python tools/check_dependencies.py --mode remote
python tools/measure_baseline.py --mode control
python tools/measure_baseline.py --mode local
```

`tools/smoke_install.py` 要分别在只装 Core、只装 SDK 的两个干净 Python 环境运行：

```sh
/path/to/core-only/python -I tools/smoke_install.py --mode local
/path/to/sdk-only/python -I tools/smoke_install.py --mode remote
```

可选的 Pages 开发预览使用临时数据库、Fake 宿主和真实 Pages API：

```sh
python tools/preview_pages.py
# 浏览器打开 http://127.0.0.1:8769/
```

预览脚本需要测试环境 FastAPI/Uvicorn；它不被插件导入，不在 AstrBot 中启动，也不会读用户真实数据。Core API 集成测试使用已安装 wheel、临时 SQLite 和实际 ASGI HTTP 路由；不依赖模型 Key。

旧 v3 配置、SQLite、索引及历史不会自动迁入 v4。本次开工时残留的未跟踪 v3 文件已移至本地 `.local-backups/v3-before-v4-20260908/`，未删除或提交。这不是迁移器。

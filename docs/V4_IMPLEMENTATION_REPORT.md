# v4 开发候选实施与验收报告

日期：2026-09-08。分支：`dev.4`；初始提交 `801ae9d` 已清空源码，仅保留技术指南。本次未提交、推送或发布。

## 实际实现

| 层 | 文件 | 行为 |
| --- | --- | --- |
| AstrBot 边界 | `main.py`、`iris_memory/host.py`、`identity.py` | 公开事件/工具/Web API、持久数据路径、稳定平台身份、借用 Provider、执行人格与权限保留 |
| 控制层 | `control.py`、`config.py`、`storage.py` | SQLite CAS 配置、按需导入、依赖检查、热启停、失败回退、generation、数据库线程 |
| 双后端 | `backends.py` | Local Embedded 公共 API；Remote 独立 SDK、连接池、能力协商、按需租约；无故障自动本地写入 |
| 人格主数据 | `personas.py`、`modules/persona.py` | 全文草稿/发布/历史/回滚/绑定/停用，Core 镜像 CAS，AstrBot 承载恢复 |
| 记忆与上下文 | `modules/memory.py`、`modules/context.py`、`outbox.py` | 观察、显式事实、纠正遗忘、画像、Recall、Task/Focus、临时候选注入、Usage |
| 陪伴与可选能力 | `modules/proactive.py`、`learning.py`、`media.py`、`maintenance.py`、`budget.py` | 提醒、私聊主动发起、证据草稿、图片描述、Core 维护、持久预算 |
| Pages 与诊断 | `api.py`、`diagnostics.py`、`pages/iris/*` | 7 个内部子页面、鉴权入口、配置/模块/人格/记忆/任务/日志真实操作，零前端框架依赖 |
| 验证工具 | `tests/`、`tools/` | 双入口消费、生命周期/配置/人格/作用域测试，依赖探针、隔离安装 smoke、临时 UI 预览、RSS smoke |

未在插件中创建 Python 打包配置或发行包。只有 `_conf_schema.json` 的 `core_mode` 留在 AstrBot 配置；其余设置由 Pages 的字段描述和 SQLite 维护。

## v3 行为对应与明确变化

v3 参考本仓库 `master`，不复制旧实现。

逐项功能与当前缺项见 [v4dev 与 v3 功能对照](V4_V3_FUNCTION_COMPARISON.md)，以下仅为概要。

| v3 行为 | v4 路线与当前状态 |
| --- | --- |
| `/iris_mem`、`/memory`、`/iris_reply` 管理命令 | 统一迁到 Pages；`/iris status` 只登记/显示当前会话；旧命令名未保留 |
| 多层记忆、画像、图谱、FTS/向量 | 消费 Core 统一 Recall 与 Profile；向量按需配置并显式重建；插件无第二份长期事实库 |
| v3 已有临时注入，另有可选宿主上下文清理及各层截断 | 保留宿主内容，统一剩余预算；预算不足跳过完整候选，不截断人格/工具/历史。`mark_as_temp` 本身不是 v4 新增能力 |
| 自动总结与提炼 | 观察与明确事实可用；本地自动认知受新发现 Core 接缝缺口阻塞，不伪实现 |
| 主动滑动窗口与独立决策 | Core Task/Focus 参与统一召回；确定性提醒、私聊静默关心；未复刻 v3 群聊概率抢答策略 |
| 人格片段演进 | 完整人格主记录、修订、手动发布、证据草稿；无自动人格发布 |
| 图片/表达学习 | 可选图片描述、基于事实的人格草稿；没有批量历史媒体回填或另一套表达事实库 |
| Bot 生成回复进入 L1；另有发送后窗口/锚点回填 | 被动 Hook 当前只报告 Recall Usage，主动保存发送效果状态；尚无助手已送达观察回填和通用平台送达回执适配，助手记忆闭环仍欠缺 |
| v3 数据迁移 | 不自动导入；旧未跟踪文件已备份，新数据目录独立 |

## 已执行验证

依赖先从 **只读 Core 源码的临时副本** 构建 wheel，安装后消费；无 editable Core、源码 PYTHONPATH 或用户真实数据库。Core/SDK 原工作区未被本次修改；结束时对构建快照中的 260 个源文件重新计算 SHA-256，与开工快照完全一致。

- wheel：Core `0.15.0`、SDK `0.11.1`，Contract `1.12.0`、Schema `24`；确切摘要见 `DEPENDENCY_SNAPSHOT.json`。
- Python `3.12.13`，macOS arm64，SQLite `3.50.4`，测试显式启用开发 SQLite 例外。
- `python -m pytest -q`：**31 passed in 2.78s**。覆盖实际 Embedded 和 SDK → Core ASGI 路由，包含 Required Active Surface；没有以 mock HTTP 返回值替代 Core 业务链路。
- 核心用例：写入幂等、正文冲突、重启持久化、跨会话拒绝、纠正/遗忘、Recall Usage、完整人格分块镜像、外部镜像修改冲突、Task/Focus、取消任务阻止发送。
- 生命周期：50 次配置驱动人格/日志热启停；另 50 次真实 Embedded 启停；结束后无 Iris 后台 Task/数据库线程。模型与宿主共享资源由宿主管理。
- 并发与故障：配置 CAS、启动失败回退、在途任务热停、会话并发登记、持久预算、有限交付队列、脱敏日志、宿主人格外部编辑冲突。
- `ruff check`、`ruff format --check`；`node --check pages/iris/app.js`。
- 两个全新虚拟环境：只装 SDK 的环境没有 Core；只装 Core 基础 wheel 的环境没有 SDK/FAISS/FastAPI。分别执行 `python -I tools/smoke_install.py` 成功。本地环境实际持久写入和召回成功；SDK 环境验证客户端构造/关闭和控制层，完整远程业务由上述双入口测试覆盖。
- 浏览器：临时 SQLite + 真实 Pages API + Fake AstrBot bridge/宿主，实测人格新建/发布、配置保存、模块开关、日志查询；这是 UI 消费测试，不是完整 AstrBot 进程验收。

工具限制记录：第一次离线安装 Ruff 因缓存没有 wheel 失败，随后使用 Core 现有虚拟环境内 **只读调用** Ruff `0.14.14` 完成检查。首次临时 UI 服务监听受到沙箱限制，获自动审批后仅绑定 `127.0.0.1:8769`，未改变生产插件结构。

## 轻量基线

独立子进程、空临时数据库、无模型/向量，`tools/measure_baseline.py` 实测：

| 模式 | 基线进程峰值 RSS | 启动后峰值 RSS | 峰值增量 | 启动耗时 |
| --- | ---: | ---: | ---: | ---: |
| 仅控制层 | 28.44 MiB | 29.50 MiB | 1.06 MiB | 0.001 s |
| 本地基础记忆 | 27.20 MiB | 53.83 MiB | 26.62 MiB | 0.230 s |

两者均未加载 FAISS/NumPy/FastAPI/Uvicorn，关闭后 Iris 线程数为 0。这是进程高水位 smoke，不是 AstrBot 整体增量测量，也不是千/万条记忆、向量索引或重建峰值。不能据此宣称大规模生产内存达标。

## 待验收与外部限制

1. Core 本地异步认知候选 Scope 缺口，见 `CORE_V4_FOLLOWUP.md`；本次未修改 Core，未启用无效提炼。
2. 本地注册上限 128 个“会话 / 人格”组合；远程需预配；Windows 本地不支持；SQLite 正式 allowlist 仍需宿主认证。
3. 参考 AstrBot `4.28.0` / `a412146401426c0cdff8bbefb8627a03da519da8` 的源码与公开文档实现，并用公共接口 fixture 测试人格采用、权限保留、恢复。尚未启动用户真实 AstrBot，未证明所有 Agent Runner、平台、WebChat 强制人格组合均兼容。
4. 真实模型 Embedding/图片/人格草稿质量、预算账单误差、120 场景 v3/v4 检索对照、群聊/私聊各 100 次完整宿主周期及 24 小时长稳未执行。
5. 遗忘 Claim 不会删原始观察；完整历史对话和平台送达回执不由当前实现接管。自动群聊概率发起、历史数据迁移、日志 SSE/全量日志归档尚未实现，当前使用按需查询和查询结果导出。
6. Core 模型质量与生产发布门禁不因插件测试通过而关闭。本轮交付为可继续联调的开发候选，不标记 M4 正式验收完成。

## 复现入口

安装与配置见根目录 `README.md`；`requirements-dev.txt` 仅用于插件测试。先安装摘要对应的 Core/SDK wheel，再运行测试、依赖探针与临时 UI 脚本。不得用测试脚本读写开发者真实记忆目录。

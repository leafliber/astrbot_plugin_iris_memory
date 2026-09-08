# v4 实施中发现的 Core 后续事项

日期：2026-09-08。仅为消费方反馈；本次没有修改 Core / SDK。

后续完整实施建议见 [Core 多人场景接入与提炼修改方案](CORE_MULTIPARTY_REVISION_PLAN.md)，覆盖公开 Space/身份注册、Observation 互动信息、参与者认知输入、Scope 绑定及必要的观察接收/Recent 配套。该方案还明确了当前仅传外部身份 ID 不会自动冻结 Entity 快照的问题。均为待实施事项，不因文档新增而关闭缺口。

## 1. 本地异步认知候选缺少作用域绑定（阻塞自动提炼）

**核查对象：** Core `7629152` 加本轮接缝工作区内容，安装物见 `DEPENDENCY_SNAPSHOT.json`。

源码事实：

- `application/reflection.py::_provider_observation()` 传给 Provider 的 DTO 包含 `id/revision/role/kind/occurred_us/content/structured_payload/privacy_labels`，不包含 tenant/agent/space。
- 同文件 `_validate_candidate_envelope()` 要求候选 `scope.tenant_id` 与 `scope.agent_id` 等于固定窗口的作用域；缺失也拒绝。
- `providers/cognitive_http.py::HttpCognitiveProvider` 有 `bind_candidate_scope = True`，提交路径据此为没有 Scope 的候选绑定固定窗口作用域。
- `embedded_providers.py::_Cognitive` 没有这个标记，公开 `AsyncCognitiveAdapter` 也没有等价绑定策略。插件不能根据这些观察 DTO 安全猜测当前 Agent；多人格、多会话不能用全局变量补齐。

**需要 Core 补齐：** 为可信宿主异步适配器提供等价的服务端固定窗口绑定，或提供经过裁剪的明确调用 Scope。由 Core 完成证据隐私标签继承与作用域校验，不能要求模型生成授权字段。

**验收：** Fake 异步 Provider 实际产出至少一条 Claim，经 `run_pending()` 持久提交并可 Recall；两组会话并发不串 Scope；显式扩权 Scope 被拒绝；来源删除/修订使旧候选失效；确认宿主没有读取 Core 私有数据库或运行时字段。

**插件当前处理：** `AstrBotHost.providers()` 对非空本地认知 Provider 返回 `core_cognitive_scope_missing`。显式记忆、观察、Recall、人格等不依赖它。以上是源码调用链核查；没有声称完成自动提炼模型质量测试。

## 2. 独立会话 Space 的公共创建能力（容量与模型改进）

当前 `provision_agent()` 最多保存 128 个 Agent/Space 注册，尚无公开的单独 Space/Session provisioning 接缝。插件为严格隔离采用每个“会话 / 人格”独立 Agent/Space，共用 Runtime。

建议公开宿主受权、幂等、持久的 Space 创建/查询能力，令 Persona → Agent、Conversation → Space 更贴近领域语义。需要定义创建上限、停用/恢复、Session FK、授权恢复和远程可信初始化，不通过客户端随意声明 tenant_id 授权。

在该能力存在前，插件明确限制 128 个注册组合；远程需运维预配，配置不允许跨组合共享 Agent。此项不阻塞有限容量下的首版使用。

## 3. 正式分发与运行平台

当前同名版本的开发快照必须按 wheel 摘要区分。需要 Core 项目冻结新的正式版本号及实际发行流程，给出经过验证的 Python/SQLite/操作系统组合；Windows 的目录所有权实现也需由 Core 决定。插件没有把开发 SQLite 例外当作正式支持。

## 4. 保留的服务端边界

- 单个 Claim 遗忘不等于来源 Observation 删除。若希望提供“忘掉这件事及其独占来源”语义，需要明确共享证据处理和可审计的受权接口。插件目前准确展示 Claim 删除范围。
- SDK 没有通用远程 Agent 创建、SSE；插件使用预配置绑定和按需查询，不为这些功能读取私有路由。
- 服务端真实模型质量、生产内存峰值、长稳、备份恢复和多进程部署认证仍由 Core 发布验收处理，插件的消费测试不能代替它们。

旧的 [G-01～G-11 清单](CORE_INTEGRATION_GAP_CHECKLIST.md) 保留为历史评审。G-01/G-02/G-06/G-07 的主要接缝已可消费；G-03 的 Embedding 接缝已接入，G-08 人格镜像已验证。本页的认知候选作用域问题属于新的消费期发现，不应被“接口已定义”掩盖。

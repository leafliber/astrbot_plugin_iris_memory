# PL-2 挂接与边界

PL-1 已删除 dev.4 原实现（首个提交只删除），随后基于 iris-connectors `a532e2b` 重建。插件包名 `astrbot_plugin_iris_memory`，版本 `4.0.0-dev`。主动回复策略、冷却、信号判断、主动发起和主动面板尚未实现。

## 接口

`iris_memory/proactive/__init__.py` 定义 `ProactiveHooks` 与 `create_proactive(plugin)`；当前 factory 返回 None。PL-2 在此返回自己的实例，`main.py` 已有以下生命周期：

1. `start()`：在 bridge 启动后调用。
2. `on_message(event)`：bridge 接收入队和纯 @ 准备之后调用。
3. `before_llm(event, request)`：Iris prepare 与注入之后调用。
4. `after_sent(event)`：AstrBot 的 after_message_sent 钩子之后调用；**该钩子自身不是发送成功证据**。
5. `close()`：先停止 PL-2 工作，再关闭 bridge／HTTP client／SQLite；卸载也移除本实例 Pages 路由。

`plugin.context` 是 AstrBot Context；`plugin.settings` 是只读配置；`plugin.bridge` 与 `plugin.service` 可能为 None（禁用／配置错误／缺令牌）。PL-2 应保留降级行为并自行持有、取消全部任务。

## 可复用能力

- `MemoryService.entry(event)`：按当前配置映射，禁止让 LLM 指定入口。
- `MemoryService.ensure_received(event)`：等待当前消息入 Iris 成功。外层需总预算，避免无限等待队列。
- `MemoryService.search/learn/goals/add_goal/end_goal`：本入口业务。`tool()` 已包含工具开关、总超时、固定错误文案。创建目标用事件去重键和内容生成 host_key；结束按所属入口与 expected_revision 提交。
- `Bridge.prepare(event, request)`：同轮总预算内等接收、拉材料、注入。成功后 `iris_prepared=(recall_id, Prompt)`，`Prompt` 含实际完整注入的 relevant ID 与提醒 ID。不会跨轮缓存 known_memory_ids。
- `Bridge.enqueue(call, timeout=...)`：有界内存工作队列，返回接收结果 Future；不是持久 outbox。
- `Bridge.record(code)`：只能传固定开发时定义的码，不传异常、URL、配置或聊天文本。
- `mapping.output_message(entry, text, stable_key)`：构造 `self_output`，PL-2 必须先获得发送调用成功，再入队。稳定键按一次实际主动发送设计，不能每次重试重新生成。
- `Bridge.inbox`：提醒的持久收件箱。其游标与页面保存原子提交；独立调度不得再启动同令牌的另一套通知消费。

事件 `send` 已被 bridge 实例包装。`iris_confirmed_send` 只证明本事件至少有一次发送正常返回，不足以绑定 PL-2 自己的任意消息。通过 `Context.send_message` 的主动发言不在这个观察范围内，PL-2 必须检查返回值并单独回流，不要从 `get_result()` 推测发送成功。原生 streaming 不保证捕获；默认关闭流式的事件覆盖应保留。

## 不可越过的边界

- 目标完成由明确完成／放弃动作触发，不由提醒拉取、发送或 LLM 回答自动完成。
- 默认关闭近似使用反馈；不要对全部召回、人物要点或被裁材料反馈。真实使用证据可调用 `bridge.client.feedback(recall_id, ids)`。
- Iris 通知本身不能提供可靠会话路由。连接器 active 提醒只支持管理员显式固定的会话＋入口；PL-2 的目标路由如需扩展应单独设计。
- 保留有限队列／预算、图片独立无鉴权下载、提示白名单、失败继续 AstrBot 正常流程及秘密值不进入日志的约束。
- 无入口全局目标的写操作需要 all 令牌；PL-1 工具不创建／修改它们。宿主 PATCH 没有 entry_id 字段，提交前查询核对。
- 本版本单 Iris 角色；不具备按 AstrBot persona 隔离或 v3 数据迁移。

状态页在 `pages/status/`，只读端点 `/<插件名>/status`。主动回复面板可新增页面及独立端点，沿用 AstrBot bridge 与登录校验，不能将宿主令牌传给页面。

## 核对材料

测试入口为 `tests/`；使用 README 的完整命令并设置 IRIS_CORE_PATH。真实 core 测试输出 `integration-result.json`、`core-service.log` 在 pytest 外部 basetemp 下。其他完整报告统一放仓库外 `iris-eval-artifacts`，不把真实对话或长日志提交。

接口核对基于 AstrBot 4.28.2 wheel；最低 Python 3.12。core 真实冒烟基线 `c6bf57d48df0cd53a1dfdffe4a93309caa5ae104`，测试无模型，不替代真实平台与学习质量验收。上游源快照缺明确许可证，见 THIRD_PARTY_NOTICES.md，需权利人补充授权说明。

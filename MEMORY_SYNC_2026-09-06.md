# Iris Chat Memory 0.3.0 修复核对记录

核对日期：2026-09-06。

来源：`astrbot_plugin_iris_chat_memory` 的 `0.3.0`，提交 `0557a4c1de1969bdccdb5a5de4b3724dc94aa6f6`。
目标：`astrbot_plugin_iris_memory`，基于 `b752ffb` 及工作区已有的 L2 周期审计改动。

## 发现并修复

| 问题 | 目标仓库原状 | 本次处理 |
| --- | --- | --- |
| 实际 Embedding 维度 | 优先使用 Provider 声明，探测失败回退 384；加载旧索引会覆盖维度 | 用真实向量探测，检查磁盘索引维度，必要时迁移；瞬时失败允许自动重试 |
| FAISS 运行期维度漂移 | 没有统一维度校验 | 单条/批量写入、检索、编辑、索引补回及归档恢复前校验 |
| 恢复期间删除/编辑 | 只复查索引槽位，没有复查 SQLite 内容 | 删除的条目不补回，编辑过的旧向量不补回，下一轮使用新内容 |
| 摘除脏向量竞争 | 使用旧快照直接删除槽位 | 持锁复查 SQLite，保留已重新使用的槽位 |
| free-list 过期 | 索引 ID 集一致时跳过清理；持久化使用旧快照 | 即使 ID 一致也清理已占用槽位，落盘时读取最新状态并去重 |
| 恢复中途取消 | 直到整轮完成才保存 | 每批补回立即保存，后续批次取消仍保留前面进度 |
| 迁移部分失败 | 非零导出/导入即可继续，最终落盘前清理备份 | 完整导出后才删旧库；完整导入、归档恢复及落盘成功后才清理两类备份 |
| 关闭竞争 | 未等待 checkpoint 或外部进行中的对账 | 等待任务/对账锁后关闭，DB 辅助方法持锁复查连接 |
| LLM 历史格式 | 字典直接传入要求 `Message` 的框架 API | 转换历史消息，保留输入内容；原生 Provider 调用保留其支持的格式 |
| Provider 类型 | 任意非空 Provider 均被接受 | 仅接受聊天 `Provider`，拒绝 Embedding 等非聊天实例 |
| 图谱辅助检索 | 将 `(文本, 节点 ID 集合)` 当文本返回 | 拆包，仅将文本拼入工具结果 |
| Web 空参数 | L1 缺会话参数、群画像 JSON null 可能触发异常 | 返回 400，继续支持显式空会话键 |

## 已具备的修复

- 指令解析已兼容 AstrBot 的 `[At:ID]` 与 `@名字(ID)` 形式，L2 / 全量清理已经传递当前人格。
- 用户级 L2 清理已覆盖只有 `user_id` 的工具记忆，并保留全局共享记忆的现有清理语义。
- L3 用户及群清理已覆盖稳定用户标记和跨群 `group_ids`，含损坏 JSON 容错与 LIKE 通配符转义；目标仓库额外具有人格过滤。
- 工具与梦境提取已有 Person 身份归一化、别名边引用及稳定 `properties.user_id` 标记。
- 隐藏配置已有串行写盘、失败保留脏标志、加载类型校验及默认值恢复通知。
- 索引已有原子保存、损坏文件保留、自愈及 checkpoint 任务强引用；工作区原有周期审计、退避重试逻辑继续保留。
- 额外核对 0.2.2 的图片安全修复：目标仓库的统一 `image/security.py` 下载器已经被预下载与解析路径使用，并包含公网 IP 固定连接、逐跳重定向校验和大小限制。

## 保留的功能差异

- FTS5 + 向量混合检索、RRF 排序、全局共享记忆、TTL、重要度和归档恢复。
- L3 人格隔离、五阶段梦境与跨轮持久化游标、L1 Outbox、LLM Governor、主动回复和学习模块。
- AstrBot 插件包内相对导入及运行时懒加载结构。
- 本次按行为核对修复，未将来源仓库整包类型标注覆盖到功能不同的目标模块；未执行全仓库 Pyright 清理。

## 验证

- 修改前基线：1825 passed。
- 引入适用的来源回归测试后：17 failed、88 passed，确认对应问题可复现。
- 修复后重点模块：213 passed。
- 最终全量：1856 passed、19 warnings；新增 31 项回归覆盖，包括目标仓库独有的归档迁移保护和关闭等待行为。
- 本次修改的 Python 文件：Ruff 通过；`git diff --check` 通过。
- 全仓库 Ruff 仍有 7 项既有问题，均在未修改文件：`iris_memory/persona_evolution/sampler.py`、`iris_memory/platform/qq_official.py`、`iris_memory/proactive/stats.py`、`iris_memory/utils/forgetting.py`、`tests/persona_evolution/test_revision_ops.py`、`tests/persona_evolution/test_service.py`、`tests/proactive/test_safety_cleanup.py`。未将其纳入本次行为修复。
- 本次未变更版本号，也未提交或发布目标仓库；原有未提交的周期审计改动保持在工作区。

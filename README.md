# Iris Memory —— 记忆 + 主动回复 二合一插件（v3）

![访问量](https://count.getloli.com/get/@astrbot_plugin_iris_memory?theme=moebooru)

面向 AstrBot 的**轻量分层记忆 + 统一决策主动回复**整合插件：一个插件同时让机器人"记住你"，并知道"什么时候该说话"。

v3.0.0 是 [Iris Chat Memory](https://github.com/Leafliber/astrbot_plugin_iris_chat_memory)（轻量记忆架构）与 Iris Reply（统一决策主动回复）的**自包含整合体**——代码已整体移植合并，**不依赖**另外两个插件。

---

> ## ⚠️ 安装前必读
>
> 1. **不可与 `astrbot_plugin_iris_chat_memory`、`astrbot_plugin_iris_reply` 同时启用**——功能完全重叠（启动时会有检测警告）。
> 2. **必须禁用 AstrBot 内置群聊上下文**：管理面板 → 服务提供商 → `provider_ltm_settings` → 将 `group_message_max_cnt` 设为 `0`。否则会出现记忆重复注入与"第三人称"叙述问题。
> 3. **建议 AstrBot ≥ 4.23.6**。低版本插件仍可正常加载，仅旧版对话清理路径（`on_agent_done` 钩子，≥4.23 才注册）不可用。
> 4. v2 老用户请直接阅读 [docs/MIGRATION.md](./docs/MIGRATION.md)——旧数据会在首次启动时**自动迁移**。

---

## 功能特性

| 模块 | 能力 |
|------|------|
| **L1 消息缓冲** | 三段式 FIFO 会话缓冲 + LLM 滚动总结，队列 token 上限 4000 |
| **L2 记忆库** | FAISS + SQLite 向量记忆，语义检索，遗忘算法自动衰减（`S=w1·R+w2·F+w3·C+w4·(1-D)`） |
| **L3 知识图谱** | SQLite nodes/edges 图谱，实体关系提取 + 路径扩展注入 |
| **画像系统** | 用户/群画像，字段级置信度，好感度 0–100，短/中/长三级更新频率 |
| **梦境加工** | 离线 6 阶段流水线（合并→时间锚定→矛盾消解→模式挖掘→知识提取→遗忘清洗），各阶段独立开关 |
| **主动回复** | 统一决策模型：一次 LLM 调用同时输出 是否发言 + 内容 + 话题概括 + 关注对象 + 漂移 + 冷却建议 |
| **图片解析** | 视觉模型解析对话图片：pHash 去重、日配额 200、缓存 7 天、被动触发自动跳过 |
| **Web 管理** | 挂载 AstrBot Dashboard（鉴权复用）：记忆管理 SPA + 主动回复统计/设置 |
| **辅助功能** | 错误友好化、Markdown 格式去除（自 v2 保留） |

### 主动回复的四种动机（motive）

- **chime_in 跟话**：话题相关时自然插话
- **follow_up 跟进**：对指定用户的持续跟进（LLM 可用工具添加/结束）
- **initiate 发起**：群里长时间无人说话时主动开话题（直发通路 `context.send_message`，发起后接话闭环，消息回填 L1）
- **watch 被动评估**：只观察不发言，更新话题锚点

本地 **SignalGate** 零 LLM 成本门控先行过滤，通过后才进入统一决策；配合 ThreadAnchor 对话锚点记账、backoff 退避 + boost 自适应频率、静音时段（默认 01:00–07:00），避免刷屏。

---

## 架构与数据流

```
群消息事件
   │
   ├─ on_message ────────────► SignalGate（本地门控，零 LLM 成本）
   │                              │ 命中 → 设置 iris_mode extra
   │                              ▼
   │                        统一决策（单次 LLM 调用，直调不触发钩子）
   │                              │ 决策发言 → 劫持主管线注入 SPEAK_HINTS
   │                              ▼
   ├─ on_all_message ──────► L1 缓冲（三段式 FIFO）＋ 图片入队
   │
   ▼
on_llm_request：先主动回复决策（可 stop_event）
                后记忆注入：清空 contexts，注入 L1 / L2 / L3 / 画像
                （全部 mark_as_temp，不污染持久会话）
   │
   ▼
LLM ──► on_llm_response：回复入 L1，按 iris_mode 记账（ThreadAnchor）
   │
   ▼
after_message_sent：写锚点 ──► 定时任务：梦境加工 / L1 总结 / 画像更新

离线：L1 总结 ─► L2 向量库（FAISS+SQLite）─► L3 知识图谱 ─► 画像系统
                 ▲                                        │
                 └────────── 梦境 6 阶段离线加工 ◄──────────┘
```

存储：L2/L3/画像用 SQLite + FAISS 落盘；主动回复侧为 KV-only（键前缀 `iris_reply:*`，30 秒脏保存）。

---

## 安装要求

- **AstrBot ≥ 4.23.6**（建议；≥4.23 起 `on_agent_done` 钩子才注册）
- Python 依赖（随插件自动安装）：`faiss-cpu`、`numpy`、`tiktoken`、`quart`、`Pillow`、`httpx`
- **Embedding Provider**：L2 记忆库默认使用 AstrBot 的 Embedding Provider（`l2_memory.embedding_source=provider`），装好后请先在插件配置里填写 `embedding_provider`
  - 可选：改为 `local` 使用本地模型，需自行 `pip install sentence-transformers`
- 相比 v2 已**移除** ChromaDB / torch / transformers 等硬依赖，安装体积减重约 489MB

---

## 快速开始

### 1) 安装与基础配置

1. 在 AstrBot 中安装本插件并重启。
2. **禁用内置群聊上下文**：`provider_ltm_settings.group_message_max_cnt = 0`。
3. 插件配置中设置 `l2_memory.embedding_provider`（你的 Embedding Provider ID）。
4. （可选）为主动回复设置 `proactive.provider_id`（决策模型）。

### 2) 验证记忆链路

让管理员在群里发送：

```
/iris_mem l1 stats
/iris_mem l2 stats
```

能看到 L1/L2 统计即正常。也可以直接和 Bot 对话，LLM 会自动调用 `save_memory` / `search_memory` 等工具存取记忆。

### 3) 开启主动回复

在目标群里（管理员）：

```
/iris_reply enable
/iris_reply status
```

### 4) 打开 Web 面板

AstrBot 管理面板 → 插件页签内进入本插件页面：

- **记忆管理页**（pages/iris）：Dashboard / L1 / L2 / L3 图谱 / 画像 / 导入导出备份 / 隐藏配置
- **主动回复页**（pages/stats）：管理 / 统计 / 设置 三个 tab

---

## 指令一览

### `/iris_mem` — 记忆管理（管理员）

格式：`/iris_mem <模块> <操作>`。模块 ∈ `l1 | l2 | l3 | profile | all`，各模块支持的操作：

| 模块 | 操作 |
|------|------|
| `l1` / `l2` / `l3` | `stats`、`clear` |
| `profile` | `show`、`reset`、`group` |
| `all` | `clear`（清空所有层级记忆，含画像） |

| 示例 | 说明 |
|------|------|
| `/iris_mem help` | 显示帮助 |
| `/iris_mem l2 stats` | 查看 L2 记忆库统计 |
| `/iris_mem l2 clear @张三` | 清除指定用户的 L2 记忆 |
| `/iris_mem l3 clear --group` | 清除当前群的知识图谱 |
| `/iris_mem profile show` | 查看用户画像 |
| `/iris_mem profile reset --all` | 重置全部画像 |
| `/iris_mem all clear` | 清空所有层级记忆（含画像） |

### `/iris_reply` — 主动回复管理（管理员 + 仅群消息）

| 指令 | 说明 |
|------|------|
| `/iris_reply enable` / `disable` | 开启 / 关闭当前群的主动回复 |
| `/iris_reply status` | 查看当前群状态（含冷却） |
| `/iris_reply reset` | 重置当前群状态 |
| `/iris_reply cooldown [分钟]` | 设置冷却（默认 5 分钟） |
| `/iris_reply willingness [低/中/高]` | 查看 / 设置回复意愿 |
| `/iris_reply initiate` | 强制触发一次主动发起 |

### LLM Function Calling 工具

记忆侧 6 个：`save_memory`、`search_memory`、`correct_memory`、`save_knowledge`、`search_knowledge_graph`、`get_profile`。
主动回复侧 3 个：`add_follow_up`、`end_follow_up`、`set_cooldown`。

---

## 配置概览

`_conf_schema.json` 共 **10 组 33 项**，全部可在 AstrBot 插件配置页修改：

| 配置组 | 说明 | 关键项 |
|--------|------|--------|
| `l1_buffer` | L1 消息上下文缓冲 | `enable`、`summary_provider`、`inject_queue_length`、`image_parsing` |
| `l2_memory` | L2 记忆库 | `enable`、`embedding_source`、`embedding_provider`、`embedding_model`、`top_k`、`relevance_threshold` |
| `l3_kg` | L3 知识图谱 | `enable`、`extraction_provider` |
| `profile` | 画像系统 | `enable`、`analysis_provider`、`enable_auto_injection`、`favorability_enable` |
| `isolation_config` | 隔离配置 | `enable_group_memory_isolation`、`enable_group_isolation`、`enable_persona_isolation` |
| `scheduled_tasks` | 梦境任务 | `enable_dream`、`provider`、6 个阶段独立开关 |
| `context_control` | 上下文接管 | `enable_conversation_cleanup` |
| `proactive` | 主动回复 | `enabled`、`stats_enabled`、`provider_id` |
| `error_friendly` | 错误消息友好化 | `enable` |
| `markdown_stripper` | Markdown 格式去除 | `enable` |

**高级参数**（不在配置页显示，按需手改）：

- 记忆侧约 50 项：插件数据目录下的 `hidden_config.json`（也可在记忆管理页"隐藏配置"中编辑）
- 主动回复侧 22 项：主动回复页"设置" tab 管理（KV overrides）

**Token 控制内置上限**：L1 队列 4000 token、L2 注入预算 2000、L3 注入预算 600、单条消息 ≤500 token、注入单条截断 300 字符；主动回复单次决策仅一次 LLM 调用。

---

## v2 用户迁移

v3 首次启动时**自动一次性迁移**旧数据（ChromaDB 记忆 → L2 重算 embedding、旧知识图谱 → L3、旧画像 → 新画像、旧主动回复白名单、8 个旧配置键映射），迁移前自动备份到 `<数据目录>/legacy_backup/`。

注意：迁移旧向量库需要先 `pip install chromadb`（软依赖，未安装则跳过 L2 迁移并记日志）。

完整迁移指南、验证清单与回滚方法见 **[docs/MIGRATION.md](./docs/MIGRATION.md)**。

---

## 常见问题（FAQ）

### 1. 记忆没有注入 / 出现"第三人称"复述？

几乎一定是 AstrBot 内置群聊上下文未关闭。把 `provider_ltm_settings.group_message_max_cnt` 设为 `0` 后重启。

### 2. 启动日志提示"功能重叠"警告？

你同时安装了 `astrbot_plugin_iris_chat_memory` 或 `astrbot_plugin_iris_reply`。v3 已整合二者，停用旧插件即可。

### 3. L2 检索不到内容？

- 确认 `l2_memory.enable=true` 且 `embedding_provider` 已正确配置。
- 用 `/iris_mem l2 stats` 确认记忆确实写入。
- 尝试降低 `l2_memory.relevance_threshold` 或提高 `top_k`。

### 4. 主动回复不说话 / 话太多？

- 先 `/iris_reply status` 看是否在冷却或静音时段（默认 01:00–07:00）。
- 用 `/iris_reply willingness 低/中/高` 调整意愿；高级参数（backoff/boost 等 22 项）在面板"设置" tab 调整。
- `proactive.stats_enabled=true` 后可在面板"统计" tab 查看每次决策记录。

### 5. 数据存哪？会上传云端吗？

全部本地存储（插件数据目录下的 SQLite / FAISS / KV）。仅在调用你配置的 LLM / Embedding Provider 时发送必要文本。

---

## 文档与链接

- [更新日志](./CHANGELOG.md)
- [v2 → v3 迁移指南](./docs/MIGRATION.md)
- [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)

---

## License 与其他信息

AGPL-3.0 license

欢迎提交 Issue 和 Pull Request。

感谢 AstrBot 提供的平台、以及其他开源项目提供的代码规范。

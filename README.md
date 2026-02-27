# Iris Memory Plugin

![访问量](https://count.getloli.com/get/@astrbot_plugin_iris_memory?theme=moebooru)

面向 AstrBot 的三层记忆插件：让机器人"记住你"、按上下文检索、并在对话中自动注入相关记忆。

⚠️注意：
- 插件默认只用规则（rule）。想要更智能的升级，请在 AstrBot 插件配置里开启 LLM 分析，并将模式改为 hybrid 或 llm
- 在 hybrid/llm 模式下，消息越多会产生越多调用，建议在配置中：关闭高耗选项、优先选用按token计费、最新且轻量的模型，以降低成本
- 为避免功能冲突，建议关闭 AstrBot 自带的“主动回复”和“上下文感知”
- 项目在快速迭代，如遇问题请提交 Issue

---

## 功能特性

### 三层记忆模型
- **工作记忆（Working Memory）**：会话内临时存储，LRU 缓存策略
- **情景记忆（Episodic Memory）**：基于 RIF 评分动态管理，选择性遗忘
- **语义记忆（Semantic Memory）**：永久保存用户画像和核心特征

### 核心能力
- ✅ **混合检索**：向量检索 + RIF 评分 + 时间感知 + 情感感知
- ✅ **RIF 评分系统**：基于时近性、相关性、频率的科学遗忘机制
- ✅ **知识图谱**：实体关系提取、多跳推理、图谱可视化
- ✅ **用户画像**：大五人格分析、沟通偏好、情绪状态追踪
- ✅ **情感分析**：混合模型（词典 + 规则 + LLM），支持 10 种情绪类型
- ✅ **会话隔离**：私聊和群聊完全隔离，基于 user_id + group_id
- ✅ **Chroma 集成**：本地向量数据库，支持高效检索
- ✅ **置信度控制**：5 级质量分级，动态升级机制
- ✅ **主动回复**：检测用户需要时主动发送消息（实验性）
- ✅ **图片分析**：Vision LLM 分析对话图片内容
- ✅ **错误友好化**：将技术错误消息转为友好提示
- ✅ **场景自适应**：根据群活跃度自动调整回复频率和处理参数
- ✅ **Web 管理界面**：独立端口访问的记忆管理仪表盘

### LLM 集成
- 自动在 LLM 请求前注入相关记忆
- 自动在 LLM 响应后捕获新记忆
- 注入近期聊天上下文，让 AI 了解当前话题
- 情感感知记忆过滤
- 知识图谱多跳推理增强

## 快速开始

### 1) 安装与启用

1. 在 AstrBot 中安装本插件。
2. 确认依赖可用（`sentence-transformers` 默认已安装）。

### 2) 推荐先做的两件事

- 为避免行为冲突，建议关闭 AstrBot 自带的“主动回复”和“上下文感知”。
- 如果希望记忆质量更自然，建议开启 `memory.use_llm`（会增加 token 消耗）。

### 3) 验证是否生效

发送以下命令检查：

- `/memory_save 我最喜欢喝冰美式`
- `/memory_search 喜欢喝什么`
- `/memory_stats`

能检索到刚刚保存的内容，即表示核心链路正常。
注：机器人会自己

---

## 常用指令

### 基础指令

| 指令 | 说明 |
|------|------|
| `/memory_save <内容>` | 手动保存一条记忆（通常置信度更高） |
| `/memory_search <关键词>` | 搜索相关记忆（支持语义检索） |
| `/memory_clear` | 清除当前会话的所有记忆 |
| `/memory_stats` | 查看当前会话记忆统计 |
| `/activity_status` | 查看各群活跃度状态（场景自适应） |

### 管理类指令

```bash
/memory_delete
/memory_delete current
/memory_delete private
/memory_delete group [shared|private|all]
/memory_delete all confirm

/proactive_reply on
/proactive_reply off
/proactive_reply status
/proactive_reply list

/iris_reset confirm          # 超管：重置所有插件数据（谨慎使用）
```

---

## 推荐配置（快速开始）

下面这套配置兼顾效果与成本，适合大多数用户：

| 配置项 | 建议值 | 说明 |
|--------|--------|------|
| `basic.enable_memory` | `true` | 开启记忆功能 |
| `basic.enable_inject` | `true` | 自动注入记忆 |
| `memory.use_llm` | `true` | 提升升级与摘要质量（消耗 token） |
| `memory.upgrade_mode` | `hybrid` | 规则 + LLM 平衡方案 |
| `memory.max_context_memories` | `3～8` | 注入太多会影响上下文长度 |
| `embedding.source` | `auto` | 自动选择 embedding 来源 |
| `proactive_reply.enable` | `false`（默认） | 建议先稳定记忆再开启 |
| `web_ui.enable` | 按需 | 仅在需要可视化管理时开启 |

---

## 配置说明（v1.7+）

> v1.7 起配置结构有调整：日志与 provider 配置已集中，旧路径请按本节更新。

### 基础与日志

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `basic.enable_memory` | 启用记忆功能 | `true` |
| `basic.enable_inject` | 自动注入记忆 | `true` |
| `logging.log_level` | 日志级别（DEBUG/INFO/WARNING/ERROR） | `INFO` |

### LLM 提供者（集中配置）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm_providers.default_provider_id` | 默认 LLM 提供者 | `""` |
| `llm_providers.memory_provider_id` | 记忆相关任务提供者 | `""` |
| `llm_providers.persona_provider_id` | 用户画像任务提供者 | `""` |
| `llm_providers.knowledge_graph_provider_id` | 图谱任务提供者 | `""` |
| `llm_providers.image_analysis_provider_id` | 图片分析提供者 | `""` |
| `llm_providers.enhanced_provider_id` | 智能增强任务提供者 | `""` |

### 记忆核心

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `memory.max_context_memories` | 每次注入记忆数量（1-10） | `5` |
| `memory.max_working_memory` | 工作记忆上限 | `20` |
| `memory.upgrade_mode` | 升级模式（rule/llm/hybrid） | `rule` |
| `memory.use_llm` | 启用 LLM 增强处理 | `false` |

### 智能增强与扩展功能

| 配置组 | 关键项 | 默认值 |
|--------|--------|--------|
| `llm_enhanced` | `sensitivity_mode` / `trigger_mode` / `emotion_mode` / `proactive_mode` / `conflict_mode` / `retrieval_mode` | `rule` |
| `knowledge_graph` | `enabled` / `extraction_mode` / `max_depth` / `max_facts` | `true` / `rule` / `3` / `8` |
| `persona` | `extraction_mode` | `rule` |
| `embedding` | `source` / `local_model` | `auto` / `BAAI/bge-small-zh-v1.5` |
| `image_analysis` | `enable` / `mode` / `daily_budget` | `true` / `auto` / `100` |
| `proactive_reply` | `enable` / `group_whitelist_mode` | `false` / `false` |
| `activity_adaptive` | `enable` | `true` |
| `error_friendly` | `enable` | `true` |
| `web_ui` | `enable` / `host` / `port` / `access_key` | `false` / `127.0.0.1` / `8089` / `""` |

Web UI 启用后默认访问：`http://127.0.0.1:8089`

---

## 常见问题（FAQ）

### 1. 为什么“记不住”或记忆效果很硬？

- 默认 `memory.use_llm=false`，只用规则时效果会更“机械”。
- 建议开启 `memory.use_llm=true`，并将 `memory.upgrade_mode` 设为 `hybrid`。
注：注意LLM在活跃的群聊会消耗相对大量的token

### 2. 为什么会出现回复冲突或重复发言？

- 常见原因是与 AstrBot 自带主动回复/上下文功能重叠。
- 建议关闭 AstrBot 同类能力，只保留本插件对应功能。

### 3. 搜索不到刚保存的记忆？

- 先用 `/memory_stats` 看当前会话是否有记录。
- 再确认你查询的是同一会话（私聊与群聊是隔离的）。
- 可尝试更语义化关键词，而不是只搜原句片段。

### 4. 开启 LLM 后 token 消耗太高怎么办？

- 降低 `memory.max_context_memories`（如 3）。
- 将部分 `llm_enhanced.*_mode` 切回 `rule`。
- 仅保留关键能力的 LLM 模式（如 `retrieval_mode=hybrid`）。

### 5. 切换 embedding 模型后检索变差或报维度问题？

- 不同模型维度可能不同（512/768/1024）。
- 请同步检查 `embedding.local_dimension`，必要时重建向量集合。

### 6. Web 管理页面打不开？

- 确认 `web_ui.enable=true`。
- 确认端口未被占用（默认 `8089`）。
- 若在远程访问，检查 `web_ui.host` 是否为 `0.0.0.0`。

### 7. 主动回复没反应？

- 需先开启 `proactive_reply.enable=true`。
- 若开了白名单模式，还需要在群里执行 `/proactive_reply on`。

### 8. 数据会上传到云端吗？

- 默认存储在本地（Chroma/SQLite）。
- 仅在你配置并调用外部 LLM 时，会向所选 provider 发送必要文本。

### 9. 如何彻底清空所有插件数据？

使用超管指令：
```
/iris_reset confirm
```

这会删除：
- 所有用户画像 (`user_personas`)
- 会话数据 (`sessions`)
- 聊天记录 (`chat_history`)
- 群成员信息 (`member_identity`)
- 批量处理队列 (`batch_queues`)
- 主动回复白名单 (`proactive_reply_whitelist`)
- 其他所有插件产生的 KV 存储数据

**注意**：执行后建议重启 AstrBot 以确保所有缓存已清空。

---

## 开发者参考（后置）

### 项目结构

```text
iris_memory/
├── core/                    # 类型定义、常量、配置管理
├── models/                  # Memory / UserPersona / EmotionalState
├── storage/                 # ChromaDB、缓存、会话管理
├── capture/                 # 触发器检测、分类、冲突解决
├── retrieval/               # 路由、重排序、LLM 检索
├── analysis/                # 情感分析、RIF、实体与画像
├── embedding/               # AstrBot/本地/fallback 向量策略
├── knowledge_graph/         # 三元组提取、存储、推理
├── multimodal/              # 图片分析与缓存
├── proactive/               # 主动回复管理与检测
├── processing/              # LLM 处理流程
├── services/                # 业务服务层
├── web/                     # Web UI 与 API
└── utils/                   # 通用工具
```

### 数据流（摘要）

1. 捕获：消息 → 触发检测 → 质量评估 → 存储
2. 检索：查询 → 路由 → 混合召回 → 重排 → 注入上下文
3. 图谱：消息 → 三元组提取 → 图谱存储 → 多跳推理

### 技术栈

- Python 3.12+
- AstrBot Plugin API
- Chroma（向量检索）
- SQLite + FTS5（知识图谱）
- sentence-transformers（本地 embedding）
- aiohttp（Web 管理服务）

---

## 注意事项

1. 私聊和群聊记忆严格隔离。
2. 敏感信息会进行过滤与风险控制。
3. 记忆写入前会进行输入清理（HTML/危险片段）。
4. 切换 embedding 时请关注维度兼容性。

---

## 文档与链接

- [功能详解](./FEATURES.md)
- [更新日志](./CHANGELOG.md)
- [框架文档](./framework.md)
- [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)
- [Chroma 文档](https://docs.trychroma.com/)

---

## License 与贡献

本插件基于 [companion-memory](./framework.md) 框架开发，欢迎提交 Issue 和 Pull Request。

仓库地址：https://github.com/leafliber/astrbot_plugin_iris_memory

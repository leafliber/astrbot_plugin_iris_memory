# Iris Memory Plugin

![访问量](https://count.getloli.com/get/@astrbot_plugin_iris_memory?theme=moebooru)

面向 AstrBot 的三层记忆插件：让机器人"记住你"，同时提供多种辅助功能。

---

> ## 📢 项目迁移公告（重要）
>
> 本项目（**iris_memory**）已进入**维护状态**，后续主力迭代已迁移至新版
> **[astrbot_plugin_iris_chat_memory](https://github.com/Leafliber/astrbot_plugin_iris_chat_memory)**
> （专注记忆能力的 v2 重构：L1 / L2 / L3 三层架构、精简记忆模型、Vue3 Web UI）。
>
> - **现状**：老版仍可正常使用；新功能将主要在新版迭代，本项目以维护与 Bug 修复为主。
> - **新版地址**：https://github.com/Leafliber/astrbot_plugin_iris_chat_memory
>
> **迁移到新版的方式：**
> 1. 在老版 Web UI「**导入导出 → 导出**」点击「**迁移到 Iris Chat Memory**」，将记忆导出为新版可识别的 JSON
> 2. 安装新版插件，在其 Web UI「**数据管理 → 导入 L2 记忆**」上传该 JSON 即可完成记忆迁移
> 3. 知识图谱 / 用户画像 / 配置暂需手动处理，详见 [更新日志](./CHANGELOG.md)

---

⚠️注意：
- 为避免功能冲突，建议关闭 AstrBot 自带的"主动回复"和"上下文感知"
- 插件默认只用规则（rule）。想要更智能的升级，请在 AstrBot 插件配置里开启 LLM 分析，并将模式改为 hybrid 或 llm
- 在 hybrid/llm 模式下，消息数量和LLM的调用数量是正相关的，建议在聊天数量多的场景中，配置：关闭高耗选项、低耗选项优先选用按token计费、最新且轻量的模型，以降低成本
- webui提供了丰富的可视化和管理功能，请在配置中开启
- 项目在快速迭代，如遇问题请提交 Issue

---

## 功能特性

### 三层记忆模型
- **工作记忆（Working Memory）**：会话内临时存储，LRU 缓存策略
- **情景记忆（Episodic Memory）**：基于 RIF 评分动态管理，选择性遗忘
- **语义记忆（Semantic Memory）**：永久保存用户画像和核心特征

### 核心能力
- ✅ **混合检索**：向量检索 + RIF 评分 + 时间感知 + 情感感知
- ✅ **知识图谱**：实体关系提取、多跳推理、图谱可视化
- ✅ **用户画像**：大五人格分析、沟通偏好、情绪状态追踪
- ✅ **主动回复**：检测用户需要时主动发送消息，自动跟进话题（实验性）
- ✅ **情感分析**：混合模型（词典 + 规则 + LLM），支持 10 种情绪类型
- ✅ **会话隔离**：私聊和群聊完全隔离，基于 user_id + group_id
- ✅ **人格隔离**：支持多 Bot 人格独立记忆存储与查询
- ✅ **Chroma 集成**：本地向量数据库，支持高效检索
- ✅ **置信度控制**：5 级质量分级，动态升级机制
- ✅ **Web 管理界面**：独立端口访问的全功能记忆管理仪表盘

### 辅助功能
- ✅ **图片分析**：Vision LLM 分析对话图片内容
- ✅ **错误友好化**：将技术错误消息转为友好提示
- ✅ **场景自适应**：根据群活跃度自动调整回复频率和处理参数
- ✅ **图增强检索**：向量检索结果结合知识图谱关系进行扩展
- ✅ **Markdown 去除**：去除 Markdown 格式标记，提升可读性，转图片时不处理确保渲染

---

## 快速开始

### 1) 安装与启用

1. 在 AstrBot 中安装本插件。
2. 确认依赖可用（`chromadb`, `sentence-transformers`已安装, 默认自动安装）。

### 2) 推荐先做的两件事

- 为避免行为冲突，建议关闭 AstrBot 自带的"主动回复"和"上下文感知"。
- 如果希望记忆质量更自然，建议在配置开启LLM相关的多个选项（可选，会增加 token 消耗）。

### 3) 验证是否生效

发送以下命令检查：

- `/memory save 我最喜欢喝冰美式`
- `/memory search 喜欢喝什么`
- `/memory stats`

能检索到刚刚保存的内容，即表示核心链路正常。

### 4) 启用 Web 管理界面（可选）

在插件配置中最下面开启 `web_ui.enable`，然后访问 `http://127.0.0.1:8089`（默认端口，且Astrbot在本地运行）。

⚠️ **Web 模块注意事项**：
- 远程访问时，将 `web_ui.host` 设为 `0.0.0.0`，并**强烈建议**设置 `access_key` 密钥

---

## 常用指令

### `/memory` - 记忆管理

| 命令 | 说明 |
|------|------|
| `/memory` | 显示帮助信息 |
| `/memory save <内容>` | 手动保存一条记忆（通常置信度更高） |
| `/memory search <关键词>` | 搜索相关记忆（支持语义检索） |
| `/memory clear` | 清除当前会话的所有记忆 |
| `/memory stats` | 查看当前会话记忆统计 |

### `/iris` - 系统管理

| 命令 | 说明 |
|------|------|
| `/iris` | 显示帮助信息 |
| `/iris activity` | 查看当前群活跃度状态 |
| `/iris activity all` | 查看所有群活跃度概览（管理员） |
| `/iris proactive on` | 开启当前群的主动回复（管理员） |
| `/iris proactive off` | 关闭当前群的主动回复（管理员） |
| `/iris proactive status` | 查看当前群的主动回复状态 |
| `/iris proactive list` | 查看所有已开启主动回复的群聊 |
| `/iris cooldown` | 开启群冷却（默认20分钟） |
| `/iris cooldown 30` | 冷却30分钟 |
| `/iris cooldown 1h` | 冷却1小时 |
| `/iris cooldown status` | 查看冷却状态 |
| `/iris cooldown off` | 取消冷却 |
| `/iris persona` | 查看当前用户画像状态 |
| `/iris persona reset` | 重置当前用户的画像 |
| `/iris persona reset <用户ID>` | 重置指定用户的画像（管理员） |
| `/iris persona clear all` | 清空所有用户画像（超管） |
| `/iris reset confirm` | 重置所有插件数据（超管，谨慎使用） |

### 删除记忆

```bash
/memory delete              # 删除当前会话记忆
/memory delete current      # 删除当前会话记忆
/memory delete private      # 删除我的私聊记忆
/memory delete group [shared|private|all]  # 删除群聊记忆（管理员）
/memory delete all confirm  # 删除所有记忆（超管）
```

---

## 推荐配置

下面这套配置兼顾效果与成本，适合大多数用户：

| 配置项 | 建议值 | 说明 |
|--------|--------|------|
| `basic.enable_memory` | `true` | 开启记忆功能 |
| `basic.enable_inject` | `true` | 自动注入记忆 |
| `memory.use_llm` | `true` | 提升升级与摘要质量（消耗 token） |
| `memory.upgrade_mode` | `hybrid` | 规则 + LLM 平衡方案 |
| `memory.max_context_memories` | `3～10` | 注入太多会影响上下文长度，消耗更多token |
| `embedding.source` | `auto` | 自动选择 embedding 来源 |
| `proactive_reply.enable` | `false`（默认） | 建议先稳定记忆再开启 |
| `web_ui.enable` | 按需 | 仅在需要可视化管理时开启，建议开启 |

---

## 配置说明

### 基础与日志

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `basic.enable_memory` | 启用记忆功能 | `true` |
| `basic.enable_inject` | 自动注入记忆 | `true` |
| `logging.log_level` | 日志级别（DEBUG/INFO/WARNING/ERROR） | `INFO` |

### LLM 提供者

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

### 嵌入向量

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `embedding.source` | 嵌入源（auto/astrbot/local） | `auto` |
| `embedding.astrbot_provider_id` | AstrBot Embedding 提供者 | `""` |
| `embedding.local_model` | 本地嵌入模型 | `BAAI/bge-small-zh-v1.5` |
| `embedding.local_dimension` | 本地模型嵌入维度 | `512` |
| `embedding.reimport_on_dimension_conflict` | 维度冲突后自动导入原记忆 | `true` |

### 智能增强

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm_enhanced.enable` | 启用 LLM 智能增强 | `false` |
| `llm_enhanced.sensitivity_mode` | 敏感度检测模式 | `rule` |
| `llm_enhanced.trigger_mode` | 触发器检测模式 | `rule` |
| `llm_enhanced.emotion_mode` | 情感分析模式 | `rule` |
| `llm_enhanced.conflict_mode` | 冲突解决模式 | `rule` |
| `llm_enhanced.retrieval_mode` | 检索路由模式 | `rule` |

### 知识图谱

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `knowledge_graph.enabled` | 启用知识图谱 | `true` |
| `knowledge_graph.extraction_mode` | 三元组提取模式 | `rule` |
| `knowledge_graph.max_depth` | 最大推理跳数 | `3` |
| `knowledge_graph.max_nodes_per_hop` | 每跳最大节点数 | `10` |
| `knowledge_graph.max_facts` | 注入事实数量上限 | `8` |
| `knowledge_graph.min_confidence` | 最小置信度 | `0.3` |

### 用户画像

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `persona.enabled` | 启用用户画像 | `true` |
| `persona.extraction_mode` | 画像提取模式 | `rule` |

### 图片分析

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `image_analysis.enable` | 启用图片分析 | `true` |
| `image_analysis.mode` | 分析模式（auto/brief/detailed/skip） | `auto` |
| `image_analysis.daily_budget` | 每日分析次数上限 | `100` |

### 主动回复

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `proactive_reply.enable` | 启用主动回复 | `false` |
| `proactive_reply.followup_after_all_replies` | Bot回复后继续跟进 | `false` |
| `proactive_reply.group_whitelist_mode` | 群聊白名单模式 | `false` |
| `proactive_reply.proactive_mode` | 主动回复模式 | `rule` |

### 其他功能

| 配置组 | 关键项 | 默认值 |
|--------|--------|--------|
| `activity_adaptive` | `enable` | `true` |
| `persona_isolation` | `memory_query_by_persona` / `kg_query_by_persona` | `false` / `false` |
| `error_friendly` | `enable` | `true` |
| `markdown_stripper` | `enable` | `true` |

### Web 管理界面

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `web_ui.enable` | 启用 Web 管理界面 | `false` |
| `web_ui.host` | 监听地址 | `127.0.0.1` |
| `web_ui.port` | Web 服务端口 | `8089` |
| `web_ui.access_key` | 访问密钥 | `""` |

---

## 常见问题（FAQ）

### 1. 为什么"记不住"或记忆效果很硬？

- 默认 `memory.use_llm=false`，只用规则时效果会更"机械"。
- 建议开启 `memory.use_llm=true`，并将 `memory.upgrade_mode` 设为 `hybrid`。
注：注意LLM在活跃的群聊会消耗相对大量的token

### 2. 为什么会出现回复冲突或重复发言？

- 常见原因是与 AstrBot 自带主动回复/上下文功能重叠。
- 建议关闭 AstrBot 同类能力，只保留本插件对应功能。

### 3. 搜索不到刚保存的记忆？

- 先用 `/memory stats` 看当前会话是否有记录。
- 再确认你查询的是同一会话（私聊与群聊是隔离的）。
- 可尝试更语义化关键词，而不是只搜原句片段。

### 4. 开启 LLM 后 token 消耗太高怎么办？

- 降低 `memory.max_context_memories`（如 3）。
- 将部分 `llm_enhanced.*_mode` 切回 `rule`。
- 仅保留关键能力的 LLM 模式（如 `retrieval_mode=hybrid`）。

### 5. 切换 embedding 模型后检索变差或报维度问题？

- 不同模型维度可能不同（512/768/1024）。
- 默认开启 `embedding.reimport_on_dimension_conflict=true`，会自动用新模型重新生成向量并导入原记忆。
- 如关闭此功能，需手动检查 `embedding.local_dimension` 并重建向量集合。

### 6. Web 管理页面打不开？

- 确认 `web_ui.enable=true`。
- 确认端口未被占用（默认 `8089`）。
- 若在远程访问，检查 `web_ui.host` 是否为 `0.0.0.0`。
- **重要**：修改 Web 配置后需完全重启 AstrBot（Docker/宿主机），仅重启插件可能无效。

### 7. 主动回复没反应？

- 需先开启 `proactive_reply.enable=true`。
- 若开了白名单模式，还需要在群里执行 `/iris proactive on`。

### 8. 数据会上传到云端吗？

- 默认存储在本地（Chroma/SQLite）。
- 仅在你配置并调用外部 LLM 时，会向所选 provider 发送必要文本。

### 9. 如何彻底清空所有插件数据？

使用超管指令：
```
/iris reset confirm
```

这会删除所有用户画像、会话数据、聊天记录、知识图谱、主动回复白名单等所有插件数据。

**注意**：执行后建议重启 AstrBot 以确保所有缓存已清空。

### 10. Web 服务器启动失败: [Error 98] Address already in use

需要彻底重启Astrbot，在面板重启可能不会有效，需要重启docker或者机器


---

## 其他注意事项

1. 私聊和群聊记忆严格隔离。
2. 敏感信息会进行过滤与风险控制。
3. 记忆写入前会进行输入清理（HTML/危险片段）。
4. 切换 embedding 时请关注维度兼容性。

---

## 三层记忆系统详解

本插件采用科学的三层记忆架构，模拟人类记忆系统，实现从短期到长期的记忆流转。

### 工作记忆（Working Memory）

**定位**：会话内临时存储，相当于人类的"短期记忆"

**特性**：
- 基于 LRU（最近最少使用）缓存策略
- 默认容量上限 10-20 条（可配置）
- 会话结束后自动清理或升级

**写入条件**：
- 置信度 < 0.5 的低质量记忆
- 情感强度较低的信息
- 临时性、上下文相关的对话内容

**升级到情景记忆的条件**（满足任一）：
- 访问 ≥ 3 次 且 重要性 > 0.5
- 情感强度 > 0.6
- 置信度 ≥ 0.7
- 用户主动请求保存
- RIF 评分 > 0.5 且 访问 ≥ 2 次
- 质量等级达到 HIGH_CONFIDENCE

---

### 情景记忆（Episodic Memory）

**定位**：基于 RIF 评分动态管理的中期记忆，相当于人类的"情景记忆"

**特性**：
- 存储在 Chroma 向量数据库，支持语义检索
- 基于 RIF 评分实现选择性遗忘
- 支持情感差异化衰减（正面记忆保留更久）

**RIF 评分公式**：
```
RIF = 0.4 × 时近性 + 0.3 × 相关性 + 0.3 × 频率
```

**时近性权重**：
| 时间范围 | 权重 |
|---------|------|
| 7 天内 | 1.2 |
| 7-30 天 | 1.0 |
| 30-90 天 | 0.8 |
| > 90 天 | 0.6 |

**升级到语义记忆的条件**（满足任一）：
- 访问 ≥ 5 次 且 置信度 > 0.65
- 质量等级为 CONFIRMED
- 重要性 ≥ 0.8 且 访问 ≥ 3 次
- 存在超过 7 天 且 访问 ≥ 3 次 且 置信度 > 0.6

**降级条件**：
- RIF 评分过低时可能被遗忘或降级回工作记忆

---

### 语义记忆（Semantic Memory）

**定位**：永久保存的核心记忆，相当于人类的"长期记忆"

**特性**：
- 最高级别的记忆存储
- 永久保存用户画像和核心特征
- 不会被自动遗忘

**保护机制**：
- 用户主动请求保存的记忆永不降级
- 质量等级为 CONFIRMED 的记忆永不降级
- 有保护标记的记忆永不降级

**降级条件**（需同时满足）：
- 置信度 < 0.5
- 超过 90 天未被访问
- RIF 评分 < 0.3
- 非用户主动请求
- 无保护标记
- 质量等级非 CONFIRMED

---

### 记忆流转图

```
                    ┌─────────────────┐
                    │   新记忆写入     │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      ┌───────────────┐            ┌───────────────┐
      │   工作记忆     │            │   情景记忆     │
      │  (WORKING)    │◄───────────│  (EPISODIC)   │
      └───────┬───────┘   降级     └───────┬───────┘
              │                             │
              │ 升级                        │ 升级
              │ (多次访问/高置信度)          │ (高价值/长期稳定)
              ▼                             ▼
      ┌───────────────┐            ┌───────────────┐
      │   情景记忆     │            │   语义记忆     │
      │  (EPISODIC)   │───────────►│  (SEMANTIC)   │
      └───────────────┘   升级     └───────────────┘
                                        │
                                        │ 降级
                                        │ (长期冷落/置信度下降)
                                        ▼
                                 ┌───────────────┐
                                 │   情景记忆     │
                                 │  (EPISODIC)   │
                                 └───────────────┘
```

---

### 辅助机制

**温和遗忘宽限期**：
- 新记忆享有 7 天宽限期，避免过早被遗忘
- 极低价值记忆（置信度 < 0.3、零访问、低情感）直接清除
- 高价值记忆自动保留（情感权重 ≥ 0.5 或 重要性 ≥ 0.6 且访问 ≥ 2 次）
- 中等价值记忆进入宽限期，7 天后自动清除

**情感差异化衰减**：
- 正面情感记忆：慢衰减（约 60 天半衰期），保留美好回忆
- 负面情感记忆：快衰减，避免消极情绪长期影响
- 中性记忆：标准衰减曲线

**记忆强化**：
- 定期分析重要记忆并更新 RIF 评分
- 防止高价值记忆因时间推移而衰减
- 基于访问模式动态调整记忆重要性

**核心信息快速通道**：
- 高置信度核心信息跳过工作记忆
- 直接存储到情景或语义记忆层
- 减少重要信息的升级延迟

---

## 文档与链接

- [更新日志](./CHANGELOG.md)
- [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)

---

## License 与其他信息

AGPL-3.0 license

欢迎提交 Issue 和 Pull Request。

感谢Astrbot提供的平台、以及其他开源项目提供的代码规范



# Iris Memory Plugin

**⚠️注意**：
- 当前插件在默认情况下，没有开启LLM分析，记忆升级会比较死板，建议在astrbot插件配置页面，启用llm分析，但同时会消耗比较多的token，请注意费用
- 建议关闭astrbot自带的主动回复和上下文感知，以避免冲突

**当前版本**: v1.7.1 | **Python**: 3.12+

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

---

## 安装与配置

### 安装依赖

#### 可选：本地嵌入模型支持（推荐安装）

**注意**：附加库可通过 AstrBot 控制台安装：
- 打开 AstrBot 控制台 → **更多功能** → **平台日志**
- 点击右上角 **安装 pip 库**，输入下面的包名确认安装
- 安装后可能需要重启 AstrBot 或插件

```
sentence-transformers（已默认安装）
```

---

## 配置说明

### 基础功能
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `basic.enable_memory` | 启用记忆功能 | true |
| `basic.enable_inject` | 自动注入记忆到对话 | true |
| `basic.log_level` | 日志级别 (DEBUG/INFO/WARNING/ERROR) | INFO |

### 记忆与 LLM 设置
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `memory.max_context_memories` | 注入记忆数量（1-10） | 3 |
| `memory.max_working_memory` | 工作记忆数量上限 | 10 |
| `memory.upgrade_mode` | 记忆升级模式 (rule/llm/hybrid) | rule |
| `memory.use_llm` | 使用 LLM 增强处理 | false |
| `memory.provider_id` | LLM 提供者（留空使用默认） | "" |

### LLM 智能增强（v1.6+）
| 配置项 | 说明 | 默认值 | 选项 |
|--------|------|--------|------|
| `llm_enhanced.provider_id` | LLM 增强提供者 | "" | 选择提供者 |
| `llm_enhanced.sensitivity_mode` | 敏感度检测模式 | rule | rule/llm/hybrid |
| `llm_enhanced.trigger_mode` | 触发器检测模式 | rule | rule/llm/hybrid |
| `llm_enhanced.emotion_mode` | 情感分析模式 | rule | rule/llm/hybrid |
| `llm_enhanced.proactive_mode` | 主动回复检测模式 | rule | rule/llm/hybrid |
| `llm_enhanced.conflict_mode` | 冲突解决模式 | rule | rule/llm/hybrid |
| `llm_enhanced.retrieval_mode` | 检索路由模式 | rule | rule/llm/hybrid |

### 知识图谱
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `knowledge_graph.enabled` | 启用知识图谱 | true |
| `knowledge_graph.extraction_mode` | 三元组提取模式 (rule/llm/hybrid) | rule |
| `knowledge_graph.provider_id` | 知识图谱 LLM 提供者 | "" |
| `knowledge_graph.max_depth` | 最大推理跳数（1-5） | 3 |
| `knowledge_graph.max_nodes_per_hop` | 每跳最大节点数（5-50） | 10 |
| `knowledge_graph.max_facts` | 注入事实数量上限（3-20） | 8 |
| `knowledge_graph.min_confidence` | 最小置信度（0.1-0.9） | 0.3 |

### 嵌入向量设置
| 配置项 | 说明 | 默认值 | 选项 |
|--------|------|--------|------|
| `embedding.source` | 嵌入源选择 | auto | auto/astrbot/local |
| `embedding.astrbot_provider_id` | AstrBot Embedding 提供者 | "" | 选择提供者 |
| `embedding.fallback_to_local` | AstrBot 不可用时降级到本地模型 | true | true/false |
| `embedding.local_model` | 本地嵌入模型 | BAAI/bge-small-zh-v1.5 | sentence-transformers 模型 |
| `embedding.local_dimension` | 本地模型嵌入维度 | 512 | 512/768/1024 等 |
| `embedding.enable_local_provider` | 启用本地嵌入模型 | true | true/false |

### 主动回复
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `proactive_reply.enable` | 启用主动回复 | false |
| `proactive_reply.group_whitelist_mode` | 群聊白名单模式 | false |

### 场景自适应（v1.6+）
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `activity_adaptive.enable` | 启用场景自适应 | true |

**功能说明**：根据群活跃度自动调整主动回复频率、批量处理参数等配置，实现温和型陪伴风格。

### 图片分析
| 配置项 | 说明 | 默认值 | 选项 |
|--------|------|--------|------|
| `image_analysis.enable` | 启用图片分析 | true | true/false |
| `image_analysis.mode` | 分析模式 | auto | auto/brief/detailed/skip |
| `image_analysis.daily_budget` | 每日分析次数上限 | 100 | 整数，0 表示无限制 |
| `image_analysis.provider_id` | 图片分析 LLM 提供者 | "" | 选择提供者 |

### 错误友好化
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `error_friendly.enable` | 启用错误消息友好化 | true |

### Web 管理界面（v1.6+）
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `web_ui.enable` | 启用 Web 管理界面 | false |
| `web_ui.port` | Web 服务端口（1024-65535） | 8089 |
| `web_ui.access_key` | 访问密钥（留空无需认证） | "" |
| `web_ui.host` | 监听地址 | 127.0.0.1 |

**访问方式**：启用后访问 `http://127.0.0.1:8089`

**功能**：
- 记忆列表、搜索、编辑、删除
- 知识图谱可视化与边管理
- 用户画像查看
- 情绪状态追踪
- 记忆导入/导出（JSON/CSV）

---

## 使用方法

### 指令列表

#### 基础指令

| 指令 | 说明 |
|------|------|
| `/memory_save <内容>` | 手动保存一条记忆（置信度更高） |
| `/memory_search <关键词>` | 搜索相关记忆，支持语义搜索 |
| `/memory_clear` | 清除当前会话（私聊或群聊）的所有记忆 |
| `/memory_stats` | 查看当前会话的记忆统计信息 |
| `/activity_status` | 查看各群活跃度状态（场景自适应功能） |

#### 管理员指令

##### 删除记忆
```
/memory_delete              # 删除当前会话记忆
/memory_delete current      # 删除当前会话记忆
/memory_delete private      # 删除我的私聊记忆
/memory_delete group [shared|private|all]  # 删除群聊记忆（管理员，群聊场景）
/memory_delete all confirm  # 删除所有记忆（超管，需确认）
```

##### 主动回复控制
```
/proactive_reply on      # 开启当前群的主动回复
/proactive_reply off     # 关闭当前群的主动回复
/proactive_reply status  # 查看当前群的状态
/proactive_reply list    # 查看所有已开启主动回复的群聊
```

### 自动捕获

当开启记忆功能时，插件会自动检测并捕获以下类型的记忆：

- **事实类**："我是"、"我有"、"我的工作是..."
- **偏好类**："我喜欢"、"我讨厌"、"我想要..."
- **情感类**："我觉得"、"感到"、"心情..."
- **关系类**："我们是朋友"、"你对我来说是..."

### 主动回复

开启主动回复功能后，当检测到用户可能需要回应时（如长时间沉默后说话、表达情绪等），机器人会主动发送消息。

需先在配置中开启 `proactive_reply.enable`，并可选择启用 `proactive_reply.group_whitelist_mode` 白名单模式。

---

## 架构设计

### 目录结构
```
iris_memory/
├── core/                    # 核心模块：类型定义、常量、配置管理、场景自适应
├── models/                  # 数据模型：Memory、UserPersona、EmotionalState
├── storage/                 # 存储模块：ChromaDB、缓存、会话管理、生命周期管理
├── capture/                 # 捕获模块：触发器检测、分类、冲突解决、批量处理
├── retrieval/               # 检索模块：路由、重排序、LLM 检索
├── analysis/                # 分析模块：情感分析、RIF 评分、实体提取、用户画像
├── embedding/               # 嵌入向量：策略模式 + 降级（AstrBot/本地/fallback）
├── knowledge_graph/         # 知识图谱：三元组提取、SQLite 存储、多跳推理
├── multimodal/              # 多模态：图片分析器与缓存
├── proactive/               # 主动回复：管理器、检测器、事件
├── processing/              # LLM 处理：分类/摘要（含熔断器）
├── services/                # 服务层：业务逻辑封装（Feature Module 模式）
├── web/                     # Web 管理：API 路由、独立服务、仪表盘（v1.6+）
└── utils/                   # 工具函数：指令解析、事件工具、LLM 帮助、Token 管理等
```

### 数据流

1. **捕获流程**
   ```
   用户消息 → 触发器检测 → 情感分析 → 敏感度检测
   → 质量评估 → RIF 评分 → 存储到 Chroma
   ```

2. **检索流程**
   ```
   用户查询 → 检索路由 → 混合检索 → 情感过滤
   → 结果重排序 → 注入 LLM 上下文（记忆 + 知识图谱 + 聊天历史）
   ```

3. **知识图谱流程**
   ```
   用户消息 → 三元组提取 (规则/LLM) → SQLite 存储
   → 多跳推理 → 构建上下文 → 注入 LLM
   ```

---

## 技术栈

- **开发语言**：Python 3.12+
- **插件框架**：AstrBot Plugin API (Star 类)
- **向量数据库**：Chroma（本地存储）
- **嵌入模型**：sentence-transformers（可选，BGE 系列）
- **知识图谱**：SQLite + FTS5 全文检索
- **情感分析**：混合模型（词典 + 规则 + LLM）
- **Web 框架**：aiohttp（独立服务）
- **数据处理**：numpy, python-dateutil, aiofiles

---

## 记忆质量等级

| 等级 | 分值 | 置信度 | 说明 |
|------|------|--------|------|
| CONFIRMED | 5 | 0.9-1.0 | 用户明确确认的信息 |
| HIGH_CONFIDENCE | 4 | 0.75-0.9 | 多次提及且一致 |
| MODERATE | 3 | 0.5-0.75 | 提及过但未验证 |
| LOW_CONFIDENCE | 2 | 0.3-0.5 | 推测或间接获取 |
| UNCERTAIN | 1 | 0.0-0.3 | 高度不确定 |

---

## 存储层自动切换

### 提升触发
- **工作→情景**：访问≥3 次 且 重要性>0.6，或 情感强度>0.7
- **情景→语义**：访问≥10 次 且 置信度>0.8，或 质量=CONFIRMED

### 降级触发
- **情景归档**：RIF 评分<0.4 且 30 天未访问
- **工作清除**：会话结束 24 小时后

---

## 注意事项

1. **会话隔离**：私聊和群聊的记忆完全隔离，不会互相影响
2. **敏感信息**：CRITICAL 级别信息（身份证号、密码等）默认不存储
3. **隐私保护**：所有数据存储在本地，不上传到云端
4. **性能优化**：工作记忆使用 LRU 缓存，自动清理过期记忆
5. **维度适配**：切换嵌入模型时自动检测维度冲突并重建 Collection
6. **输入安全**：保存记忆前自动过滤 HTML 标签和危险内容

---

## 开发参考

- [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)
- [companion-memory 框架文档](./framework.md)
- [Chroma 文档](https://docs.trychroma.com/)
- [功能特性详解](./FEATURES.md)
- [更新日志](./CHANGELOG.md)

---

## 未来计划

- [ ] 支持多模态记忆（语音）

---

## License

本插件基于 [companion-memory](./framework.md) 框架开发。

---

## 贡献

欢迎提交 Issue 和 Pull Request！

**仓库地址**: https://github.com/leafliber/astrbot_plugin_iris_memory

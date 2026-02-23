# Iris Memory Plugin

基于 companion-memory 框架实现的三层记忆插件，为 AstrBot 提供智能记忆管理能力。

## 功能特性

### 三层记忆模型
- **工作记忆（Working Memory）**：会话内临时存储，LRU缓存策略
- **情景记忆（Episodic Memory）**：基于RIF评分动态管理，选择性遗忘
- **语义记忆（Semantic Memory）**：永久保存用户画像和核心特征

### 核心能力
- ✅ 混合检索：向量检索 + RIF评分 + 时间感知 + 情感感知
- ✅ RIF评分系统：基于时近性、相关性、频率的科学遗忘机制
- ✅ 多维度评分：五维度评分系统（可选），提升记忆质量
- ✅ 情感分析：混合模型（词典 + 规则 + 轻量模型）
- ✅ 会话隔离：私聊和群聊完全隔离，基于user_id + group_id
- ✅ Chroma集成：本地向量数据库，支持高效检索
- ✅ 置信度控制：5级质量分级，动态升级机制
- ✅ 主动回复：检测用户需要时主动发送消息
- ✅ 图片分析：Vision LLM 分析对话图片内容
- ✅ 错误友好化：将技术错误消息转为友好提示

### LLM集成
- 自动在LLM请求前注入相关记忆
- 自动在LLM响应后捕获新记忆
- 注入近期聊天上下文，让AI了解当前话题
- 情感感知记忆过滤


## 使用方法

### 可选功能：通过 AstrBot 控制台 安装 pip 库

某些附加能力（如本地模型或额外分析库）是可选的，可直接在 AstrBot 控制台通过图形界面安装 pip 包：

- 打开 AstrBot 控制台 → 选择 **更多功能** → **平台日志**。
- 在**平台日志**界面右上角，点击 **安装 pip 库**（或类似按钮），在弹窗中输入要安装的包名或要求文件内容（例如 `sentence-transformers`），然后确认安装。
- 安装完成后，可能需要重启 AstrBot 或重启插件以使包生效。

以下是可选的pip包：

```
# 本地嵌入模型支持（可选）
sentence-transformers>=2.2.0
```

注意：仅安装受信任的包与版本，并在生产环境中谨慎操作。


### 指令列表

#### 基础指令

##### 1. 手动保存记忆
```
/memory_save 我喜欢吃披萨
```
显式保存一条记忆，置信度会更高。

##### 2. 搜索记忆
```
/memory_search 我喜欢什么
```
检索相关记忆，支持语义搜索。

##### 3. 清除当前会话记忆
```
/memory_clear
```
清除当前会话（私聊或当前群聊）的所有记忆。

##### 4. 记忆统计
```
/memory_stats
```
查看当前会话的记忆统计信息。

#### 管理员指令

##### 5. 统一删除记忆
```
/memory_delete              # 删除当前会话记忆
/memory_delete current      # 删除当前会话记忆
/memory_delete private      # 删除我的私聊记忆
/memory_delete group [shared|private|all]  # 删除群聊记忆（管理员，群聊场景）
/memory_delete all confirm  # 删除所有记忆（超管，需确认）
```
按不同范围删除记忆，支持精细控制。

##### 6. 主动回复控制（管理员）
```
/proactive_reply on      # 开启当前群的主动回复
/proactive_reply off     # 关闭当前群的主动回复
/proactive_reply status  # 查看当前群的状态
/proactive_reply list    # 查看所有已开启主动回复的群聊
```
控制群聊的主动回复功能（需先在配置中开启 `proactive_reply.group_whitelist_mode`）。

### 自动捕获

当开启记忆功能时，插件会自动检测并捕获以下类型的记忆：

- **事实类**："我是"、"我有"、"我的工作是..."
- **偏好类**："我喜欢"、"我讨厌"、"我想要..."
- **情感类**："我觉得"、"感到"、"心情..."
- **关系类**："我们是朋友"、"你对我来说是..."

### 主动回复

开启主动回复功能后，当检测到用户可能需要回应时（如长时间沉默后说话、表达情绪等），机器人会主动发送消息。

## 架构设计

### 目录结构
```
iris_memory/
├── core/                    # 核心模块
│   ├── types.py            # 数据类型定义
│   ├── constants.py        # 常量定义
│   ├── config_manager.py   # 配置管理器
│   ├── config_registry.py  # 配置注册表
│   ├── defaults.py         # 默认配置
│   ├── activity_config.py  # 场景自适应配置
│   ├── memory_scope.py     # 记忆可见性范围
│   ├── provider_utils.py   # LLM 提供者工具
│   └── service_container.py # 服务容器
├── models/                  # 数据模型
│   ├── memory.py           # Memory数据模型
│   ├── user_persona.py     # 用户画像模型
│   └── emotion_state.py    # 情感状态模型
├── storage/                 # 存储模块
│   ├── chroma_manager.py   # Chroma向量数据库管理
│   ├── chroma_operations.py # Chroma CRUD 操作
│   ├── chroma_queries.py   # Chroma 查询操作
│   ├── session_manager.py  # 会话隔离管理
│   ├── cache.py            # 工作记忆缓存
│   ├── chat_history_buffer.py # 聊天记录缓冲区
│   └── lifecycle_manager.py # 记忆生命周期管理
├── capture/                 # 捕获模块
│   ├── capture_engine.py   # 记忆捕获引擎
│   ├── batch_processor.py  # 批量消息处理器
│   ├── message_classifier.py # 消息分类器
│   ├── message_merger.py   # 消息合并器
│   ├── capture_logger.py   # 捕获日志工具
│   ├── detector/           # 检测器
│   │   ├── trigger_detector.py       # 触发器检测
│   │   ├── sensitivity_detector.py   # 敏感度检测
│   │   ├── llm_trigger_detector.py   # LLM触发器检测
│   │   └── llm_sensitivity_detector.py # LLM敏感度检测
│   └── conflict/           # 冲突处理
├── retrieval/               # 检索模块
│   ├── retrieval_engine.py # 记忆检索引擎
│   ├── retrieval_router.py # 检索路由器
│   ├── llm_retrieval_router.py # LLM检索路由
│   ├── reranker.py         # 结果重排序器
│   └── retrieval_logger.py # 检索日志工具
├── analysis/                # 分析模块
│   ├── rif_scorer.py       # RIF评分器
│   ├── emotion/            # 情感分析
│   │   ├── emotion_analyzer.py     # 情感分析器
│   │   └── llm_emotion_analyzer.py # LLM情感分析器
│   ├── entity/             # 实体提取
│   │   └── entity_extractor.py     # 实体提取器
│   └── persona/            # 用户画像分析
├── embedding/               # 嵌入向量模块
│   ├── manager.py          # 嵌入管理器（策略模式+降级）
│   ├── base.py             # 嵌入提供者基类
│   ├── astrbot_provider.py # AstrBot 嵌入提供者
│   ├── local_provider.py   # 本地模型提供者
│   └── fallback_provider.py # 降级提供者
├── knowledge_graph/         # 知识图谱模块
│   ├── kg_extractor.py     # 三元组提取（规则+LLM）
│   ├── kg_storage.py       # SQLite+FTS5 存储
│   ├── kg_models.py        # 图谱数据模型
│   ├── kg_reasoning.py     # 图谱推理
│   └── kg_context.py       # 图谱上下文构建
├── multimodal/              # 多模态模块
│   ├── image_analyzer.py   # 图片分析器
│   └── image_cache.py      # 图片缓存
├── proactive/               # 主动回复模块
│   ├── proactive_manager.py # 主动回复管理器
│   ├── proactive_reply_detector.py # 回复检测器
│   ├── llm_proactive_reply_detector.py # LLM回复检测
│   └── proactive_event.py  # 主动回复事件
├── processing/              # LLM 处理模块
│   └── llm_processor.py    # LLM 分类/摘要（含熔断器）
├── services/                # 服务层
│   ├── memory_service.py   # 记忆服务封装
│   ├── business_operations.py # 业务操作
│   ├── initializers.py     # 初始化编排
│   ├── persistence.py      # 持久化操作
│   └── modules/            # 功能模块
└── utils/                   # 工具函数
    ├── command_utils.py    # 指令解析
    ├── event_utils.py      # 事件工具
    ├── llm_helper.py       # 统一 LLM 调用
    ├── token_manager.py    # Token 预算管理
    ├── rate_limiter.py     # 速率限制器
    ├── bounded_dict.py     # 有界字典（防内存泄露）
    ├── logger.py           # 日志工具
    └── member_identity_service.py # 成员身份管理
```

### 数据流

1. **捕获流程**
   ```
   用户消息 → 触发器检测 → 情感分析 → 敏感度检测
   → 质量评估 → RIF评分 → 存储到Chroma
   ```

2. **检索流程**
   ```
   用户查询 → 检索路由 → 混合检索 → 情感过滤
   → 结果重排序 → 注入LLM上下文
   ```

## 技术栈

- **开发语言**：Python 3.12+
- **插件框架**：AstrBot Plugin API (Star类)
- **向量数据库**：Chroma（本地存储）
- **嵌入模型**：sentence-transformers（可选，BGE-M3）
- **情感分析**：内置混合模型（词典 + 规则）
- **数据处理**：numpy

## 配置说明

### 基础功能
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `basic.enable_memory` | 启用记忆功能 | true |
| `basic.enable_inject` | 自动注入记忆到对话 | true |
| `basic.log_level` | 日志级别 | INFO |

### 记忆与LLM设置
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `memory.max_context_memories` | 注入记忆数量（1-10） | 3 |
| `memory.max_working_memory` | 工作记忆数量上限 | 10 |
| `memory.upgrade_mode` | 记忆升级模式（rule/llm/hybrid） | rule |
| `memory.use_llm` | 使用LLM增强处理 | false |
| `memory.provider_id` | LLM提供者（留空使用默认） | "" |

### 批量处理
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `batch_processing.batch_threshold_count` | 批量处理消息数量阈值 | 20 |
| `batch_processing.short_message_threshold` | 短消息长度阈值 | 15 |
| `batch_processing.merge_time_window` | 消息合并时间窗口（秒） | 60 |
| `batch_processing.max_merge_count` | 最大合并消息数 | 5 |

### 主动回复
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `proactive_reply.enable` | 启用主动回复 | false |
| `proactive_reply.group_whitelist_mode` | 群聊白名单模式 | false |

### 图片分析
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `image_analysis.enable` | 启用图片分析 | true |
| `image_analysis.mode` | 分析模式（auto/brief/detailed/skip） | auto |
| `image_analysis.daily_budget` | 每日分析次数上限 | 100 |

### 错误友好化
| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `error_friendly.enable` | 启用错误消息友好化 | true |

### 记忆质量等级

- **CONFIRMED (5)**：用户明确确认的信息，置信度0.9-1.0
- **HIGH_CONFIDENCE (4)**：多次提及且一致，置信度0.75-0.9
- **MODERATE (3)**：提及过但未验证，置信度0.5-0.75
- **LOW_CONFIDENCE (2)**：推测或间接获取，置信度0.3-0.5
- **UNCERTAIN (1)**：高度不确定，置信度0.0-0.3

### 存储层自动切换

**提升触发：**
- 工作→情景：访问≥3次 且 重要性>0.6，或 情感强度>0.7
- 情景→语义：访问≥10次 且 置信度>0.8，或 质量=CONFIRMED

**降级触发：**
- 情景归档：RIF评分<0.4 且 30天未访问
- 工作清除：会话结束24小时后

## 开发参考

- [AstrBot插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)
- [companion-memory框架文档](./framework.md)
- [Chroma文档](https://docs.trychroma.com/)

## 注意事项

1. **会话隔离**：私聊和群聊的记忆完全隔离，不会互相影响
2. **敏感信息**：CRITICAL级别信息（身份证号、密码等）默认不存储
3. **隐私保护**：所有数据存储在本地，不上传到云端
4. **性能优化**：工作记忆使用LRU缓存，自动清理过期记忆
5. **维度适配**：切换嵌入模型时自动检测维度冲突并重建Collection

## 未来计划

- [ ] 支持多模态记忆（语音）

## 贡献

欢迎提交Issue和Pull Request！

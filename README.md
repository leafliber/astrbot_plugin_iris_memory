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
- ✅ 情感分析：混合模型（词典 + 规则 + 轻量模型）
- ✅ 会话隔离：私聊和群聊完全隔离，基于user_id + group_id
- ✅ Chroma集成：本地向量数据库，支持高效检索
- ✅ 置信度控制：5级质量分级，动态升级机制

### LLM集成
- 自动在LLM请求前注入相关记忆
- 自动在LLM响应后捕获新记忆
- 情感感知记忆过滤

## 安装

### Docker快速测试（推荐）

使用Docker进行快速测试，适合日常开发和验证：

```bash
cd docker
./manager.sh start
```

或运行测试示例：

```bash
cd docker
./test-example.sh
```

访问 http://localhost:6185 测试插件功能。

**管理命令**：
```bash
./manager.sh start    # 启动环境
./manager.sh stop     # 停止环境
./manager.sh restart  # 重启容器
./manager.sh logs     # 查看日志
./manager.sh test     # 运行测试
./manager.sh status   # 查看状态
./manager.sh help     # 查看所有命令
```

详细说明请参考：[Docker测试环境](./docker/)

### 本地安装

#### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启用插件
将插件文件夹放入AstrBot的`plugins`目录，然后在AstrBot配置中启用。

### 3. 配置插件
在AstrBot Web界面中配置插件参数：

**记忆配置**
- `auto_capture`: 是否自动捕获记忆（默认：true）
- `max_working_memory`: 工作记忆最大条数（默认：10）
- `rif_threshold`: RIF评分删除阈值（默认：0.4）

**Chroma配置**
- `embedding_model`: 嵌入模型名称（默认：BAAI/bge-m3）
- `embedding_dimension`: 嵌入向量维度（默认：1024）

**LLM集成**
- `enable_inject`: 是否在LLM请求中注入记忆（默认：true）
- `max_context_memories`: 注入到LLM的最大记忆数（默认：3）

## 使用方法

### 指令列表

#### 1. 手动保存记忆
```
/memory_save 我喜欢吃披萨
```
这将显式保存一条记忆，置信度会更高。

#### 2. 搜索记忆
```
/memory_search 我喜欢什么
```
检索相关记忆，支持语义搜索。

#### 3. 清除记忆
```
/memory_clear
```
清除当前会话的所有记忆（会话隔离）。

#### 4. 记忆统计
```
/memory_stats
```
查看当前会话的记忆统计信息。

### 自动捕获

当开启`auto_capture`时，插件会自动检测并捕获以下类型的记忆：

- **事实类**："我是"、"我有"、"我的工作是..."
- **偏好类**："我喜欢"、"我讨厌"、"我想要..."
- **情感类**："我觉得"、"感到"、"心情..."
- **关系类**："我们是朋友"、"你对我来说是..."

## 架构设计

### 目录结构
```
iris_memory/
├── core/                    # 核心模块
│   ├── types.py            # 数据类型定义
│   └── config.py           # 配置管理
├── storage/                 # 存储模块
│   ├── chroma_manager.py   # Chroma向量数据库管理
│   ├── session_manager.py  # 会话隔离管理
│   └── cache.py           # 工作记忆缓存
├── capture/                 # 捕获模块
│   ├── capture_engine.py   # 记忆捕获引擎
│   ├── trigger_detector.py # 触发器检测器
│   └── sensitivity_detector.py # 敏感度检测器
├── retrieval/               # 检索模块
│   ├── retrieval_engine.py # 记忆检索引擎
│   ├── retrieval_router.py # 检索路由器
│   └── reranker.py       # 结果重排序器
├── analysis/                # 分析模块
│   ├── emotion_analyzer.py # 情感分析器
│   ├── rif_scorer.py     # RIF评分器
│   └── entity_extractor.py # 实体提取器
└── models/                  # 数据模型
    ├── memory.py          # Memory数据模型
    ├── user_persona.py    # 用户画像模型
    └── emotion_state.py   # 情感状态模型
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

- **开发语言**：Python 3.8+
- **插件框架**：AstrBot Plugin API (Star类)
- **向量数据库**：Chroma（本地存储）
- **嵌入模型**：sentence-transformers（可选，BGE-M3）
- **情感分析**：内置混合模型（词典 + 规则）
- **数据处理**：numpy

## 配置说明

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
- [companion-memory框架文档](./companion-memory-framework.md)
- [Chroma文档](https://docs.trychroma.com/)

## 注意事项

1. **会话隔离**：私聊和群聊的记忆完全隔离，不会互相影响
2. **敏感信息**：CRITICAL级别信息（身份证号、密码等）默认不存储
3. **隐私保护**：所有数据存储在本地，不上传到云端
4. **性能优化**：工作记忆使用LRU缓存，自动清理过期记忆

## 未来计划

- [ ] 支持多模态记忆（语音、图像）
- [ ] 实现知识图谱支持多跳推理
- [ ] 集成更多嵌入模型（OpenAI、HuggingFace）
- [ ] 增加记忆可视化界面
- [ ] 支持记忆导出和导入

## License

本插件基于 [companion-memory](./companion-memory-framework.md) 框架开发。

## 贡献

欢迎提交Issue和Pull Request！

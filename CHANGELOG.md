# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.11.0] - 2026-03-13

### ⚠️Note
- 本次更新优化了 Web 管理端的启动逻辑，**需要完全重启 AstrBot（Docker/宿主机）才能生效**

### Changed
- **Web 管理端启动逻辑优化** (`iris_memory/web/server.py`)
  - 重构 Uvicorn 服务器启动方式，使用标准 `server.serve()` API
  - 移除不稳定的内部 API `config.http_protocol_class` 调用
  - 修复服务器显示启动成功但无法处理请求的问题
  - 优化端口复用 socket 管理
  - 改进服务器停止时的优雅关闭逻辑

## [v1.10.6] - 2026-03-13

### Changed
- **记忆强化引擎简化** (`iris_memory/analysis/reinforcement.py`)
  - 移除回顾消息发送功能，不再主动发送回顾对话
  - 移除 `ReviewPromptGenerator` 类（回顾对话生成器）
  - 移除 `notify_callback` 参数和通知发送逻辑
  - 移除 `max_daily_reviews` 每日回顾上限配置
  - 移除 `get_review_candidates()` 方法
  - 移除 `process_review_response()` 方法
  - 保留 SM-2 变体核心逻辑：定期分析重要记忆并更新 RIF 评分

### Fixed
- **Web 仪表盘记忆总数显示修复** (`iris_memory/web/static/js/pages/dashboard.js`)
  - 修复前端读取 `mem.total` 与后端返回 `total_count` 字段名不一致的问题
  - 兼容处理：`mem.total_count ?? mem.total ?? 0`

- **Web 用户画像活跃时段显示修复** (`iris_memory/web/repositories/persona_repo.py`)
  - 修复 `_build_persona_data` 方法缺少 `hourly_distribution` 字段
  - 活跃时段图表现在可以正确显示用户交互时间分布

- **Web 记忆管理分页功能修复** (`iris_memory/web/static/js/pages/memories.js`)
  - 修复分页回调函数写法与其他页面不一致的问题
  - 统一使用箭头函数形式：`onChange: p => { state.page = p; searchMemories(); }`

- **LLM 统计来源推断修复** (`iris_memory/utils/llm_helper.py`, `iris_memory/stats/registry.py`)
  - 修复异步任务中调用栈丢失导致来源显示为 `_UnixSelectorEventLoop` 的问题
  - 在 `call_llm()` 执行时立即捕获调用来源，传递给统计记录
  - 新增 `_infer_caller_source()` 函数预先推断来源
  - `record_call()` 新增可选参数 `source_module` 和 `source_class`

- **Web 知识图谱节点大小优化** (`iris_memory/web/static/js/pages/kg.js`)
  - 缩小节点半径范围：6px ~ 16px（原 8px ~ 25px）
  - 优化视觉呈现，避免节点过大遮挡

### Removed
- 移除 `memory.reinforcement.max_daily` 配置项（每日回顾上限）

## [v1.10.5] - 2026-03-12

### Fixed
- **Web 服务器 Hypercorn 兼容性修复** (`iris_memory/web/server.py`)
  - 修复新版 Hypercorn API 变更导致的 `worker_serve()` 参数错误
  - 改用标准 `config.bind` 格式，让 Hypercorn 自动管理 socket
  - 优化启动检测逻辑，增加任务状态检查
  - 缩短关闭超时时间
  - 添加详细的启动失败错误日志

## [v1.10.4] - 2026-03-10

### Added
- **Web 管理界面全新重构** (`iris_memory/web/`)
  - 采用分层架构：API 路由层、服务层、数据仓库层
  - 新增模块化前端代码结构，ES6 模块化组织
  - 新增 Dashboard 仪表盘页面，集成系统状态和 LLM 监控
  - 新增记忆管理页面，支持搜索、查看、编辑、批量删除
  - 新增知识图谱页面，支持节点/边可视化和搜索
  - 新增用户画像页面，展示用户特征和交互历史
  - 新增主动回复配置页面，支持白名单管理
  - 新增冷却机制页面，展示和管理冷却状态
  - 新增配置管理页面，支持配置查看和导出
  - 新增 LLM 监控页面，展示调用统计和最近记录
  - 新增系统信息页面，展示运行状态和资源使用
  - 新增导入导出功能，支持记忆和知识图谱的 JSON 格式

### Changed
- **前端代码结构重构** (`iris_memory/web/static/js/`)
  - 将多个独立 JS 文件合并为模块化结构
  - 按功能划分：api、components、pages、store、utils
  - 统一使用 ES6 import/export 语法
  - 优化代码组织，减少全局变量污染

### Fixed
- **Web Dashboard 模块导入缺失修复** (`iris_memory/web/static/js/main.js`)
  - 添加缺失的 `loadLlm` 导入语句
  - 添加缺失的 `loadSystem` 导入语句
  - 修复页面加载时 `ReferenceError` 错误

- **Web UI 初始化问题修复** (`iris_memory/web/server.py`)
  - 修复 Web UI 初始化重复问题
  - 修复端口占用检测逻辑

## [v1.10.3] - 2026-03-08

### Fixed
- **NumPy 数组布尔判断错误修复** (`iris_memory/storage/chroma_manager.py`, `iris_memory/embedding/manager.py`)
  - 修复 `_extract_memory_data` 方法中 `documents`、`embeddings`、`metadatas` 的布尔判断
  - 修复 `_detect_existing_dimension` 方法中 `embeddings` 的布尔判断
  - 将隐式布尔判断 `if embeddings and ...` 改为显式判断 `if embeddings is not None and ...`
  - 解决 ChromaDB 某些情况下返回 NumPy 数组导致的 `ValueError: The truth value of an array with more than one element is ambiguous`

- **MemoryScope 导入路径修复** (`main.py`)
  - 修复 `save_memory_tool` 方法中 `MemoryScope` 的导入路径
  - 将 `from iris_memory.core.types import MemoryScope` 改为 `from iris_memory.core.memory_scope import MemoryScope`

## [v1.10.2] - 2026-03-04

### Changed
- **Markdown 去除器配置简化** (`iris_memory/processing/markdown_stripper.py`)
  - 用户可见配置仅保留 `enable` 开关（通过 AstrBot 管理界面控制）
  - 内部配置（`preserve_code_blocks`、`preserve_links`、`threshold_offset`、`strip_headers`、`strip_lists`）移至 `defaults.py` 统一管理
  - 减少配置复杂度，默认行为：去除所有 Markdown 格式标记

### Removed
- 移除 `_conf_schema.json` 中 Markdown 去除器的 5 个内部配置项
- 移除 `config_registry.py` 中对应的 5 个 `ConfigDefinition` 映射
- 移除 `config_properties.py` 中对应的 5 个 `_ConfigProp` 属性定义
- 移除测试文件中不再适用的配置变体测试用例

## [v1.10.1] - 2026-03-03

### Changed
- **FollowUp 调试日志增强** (`iris_memory/proactive/manager.py`)
  - `notify_bot_reply` 方法新增详细调试日志，输出初始化状态、配置开关状态
  - 每个提前返回点新增日志说明具体跳过原因
  - 便于排查 FollowUp 机制未触发问题

## [v1.10.0] - 2026-03-02

### ⚠️ 注意
本次更新需要完全重启 Nonebot，否则会导致主动回复模块初始化失败

### Verified
- **ProactiveManager API 兼容性验证** (`iris_memory/capture/batch_processor.py`, `iris_memory/proactive/manager.py`)
  - 验证 `process_message` 参数格式正确匹配
  - messages 字段 (text, sender_id, sender_name, timestamp) 完整传递
  - 无需额外参数验证逻辑

- **ProactiveManager 初始化参数传递验证** (`iris_memory/services/initializer.py`, `iris_memory/services/modules/proactive_module.py`)
  - 验证 `plugin_data_path` 参数正确传递
  - 调用链完整：initializer → ProactiveModule → ProactiveManager
  - 已有 `if not plugin_data_path` 防护检查

- **测试用例接口一致性验证** (`tests/capture/test_batch_processor.py`)
  - 验证测试代码已使用 `process_message` 新接口
  - 无遗留 `handle_batch` 引用
  - `TestProactiveReplyIntegration` 正确验证新 API 调用

## [v1.9.3] - 2026-03-02

### Added
- **连续回复限制机制** (`iris_memory/proactive/proactive_manager.py`)
  - 新增 `_recent_replies` 跟踪短时间内各会话的主动回复次数
  - 默认限制：5分钟内最多连续回复 3 次
  - 新增 `_is_consecutive_limit_reached()` 和 `_record_reply_time()` 方法
  - 新增 `replies_consecutive_limited` 统计计数器
  - 防止特定群聊/用户的"滚雪球"式连续回复问题

- **启动冷却期机制** (`iris_memory/proactive/proactive_manager.py`)
  - 新增 `_startup_time` 记录启动时间
  - 新增 `_is_in_startup_cooldown()` 方法检查启动冷却状态
  - 默认启动冷却期：2 分钟（`STARTUP_COOLDOWN_SECONDS=120`）
  - 防止重启后状态丢失（`_recent_replies`、`last_reply_time` 为空）导致连续回复

### Changed
- **主动回复检测器阈值与权重调整** (`iris_memory/proactive/proactive_reply_detector.py`)
  - MEDIUM 阈值从 0.3 提高到 0.4，降低误触发概率
  - question 权重从 0.4 降低到 0.3
  - emotional_support 权重从 0.3 降低到 0.25
  - seeking_attention 权重从 0.3 降低到 0.25
  - mention_bot 权重从 0.5 降低到 0.35
  - expect_response 权重从 0.35 降低到 0.25
  - chat_topics 权重从 0.25 降低到 0.2
  - 积极情感触发阈值从 0.3 提高到 0.5，避免群聊"哈哈哈"误触发

- **紧急度冷却乘数调整** (`iris_memory/core/constants.py`)
  - CRITICAL 乘数从 0.25 提高到 0.5（冷却时间：60s × 0.5 = 30s）
  - HIGH 乘数从 0.5 提高到 0.75（冷却时间：60s × 0.75 = 45s）
  - 避免高紧急度回复冷却时间过短导致频繁触发

- **智能增强参数调整** (`iris_memory/core/defaults.py`)
  - smart_boost_window 从 120s 缩短到 60s（不超过冷却时间）
  - smart_boost_threshold 从 0.25 提高到 0.4（与 MEDIUM 阈值一致）
  - 确保智能增强窗口不会与冷却机制冲突

### Fixed
- **每日计数惰性重置** (`iris_memory/proactive/proactive_manager.py`)
  - 新增 `_last_reset_date` 跟踪重置日期
  - 新增 `_check_daily_reset()` 方法实现跨日自动重置
  - 修复每日计数从未被重置的问题

- **用户发言时间记录时机** (`iris_memory/proactive/proactive_manager.py`)
  - 将 `_record_user_message()` 调用从 `_process_task` 移至 `handle_batch`
  - 确保智能增强窗口基于用户发言时间而非 Bot 回复时间
  - 避免 Bot 自身回复刷新窗口导致"滚雪球"效应

- **冷却时间记录时机** (`iris_memory/proactive/proactive_manager.py`)
  - 将 `last_reply_time` 记录从 `handle_batch` 移至 `_process_task` 发送成功后
  - 确保冷却时间基于实际发送时间而非入队时间

- **KV 持久化 is_async 配置错误** (`iris_memory/services/persistence_service.py`)
  - 修复同步方法被错误标记为异步导致 `await` 报错的问题
  - `serialize_whitelist`/`deserialize_whitelist` 设置 `is_async=False`
  - `member_identity.serialize`/`deserialize` 设置 `is_async=False`
  - `activity_tracker.serialize`/`deserialize` 设置 `is_async=False`
  - 错误信息：`object list can't be used in 'await' expression`

### Tests
- **连续回复限制测试** (`tests/proactive/test_consecutive_limit.py`)
  - 新增连续回复限制基本逻辑测试
  - 新增窗口过期自动清理测试
  - 新增会话隔离测试
  - 新增 handle_batch 集成测试

## [v1.9.2] - 2026-03-02

### Added
- **命令处理与权限管理** (`iris_memory/services/business_service.py`, `iris_memory/services/memory_service.py`)
  - 新增 `handle_command()` 方法处理管理命令
  - 实现管理员权限检查机制
- **检索策略实现** (`iris_memory/retrieval/`)
  - 新增多种检索策略支持
- **智能增强配置更新** (`iris_memory/proactive/`)
  - 更新 smart boost 配置，增强主动回复任务管理
- **语义提取与聚类测试** (`tests/`)
  - 新增语义提取、聚类和置信度机制的全面测试

### Changed
- **ChromaManager 架构重构** (`iris_memory/storage/chroma_manager.py`)
  - 从 Mixin 继承模式重构为组合模式
  - 提升代码可维护性和可测试性
- **MemoryService 初始化逻辑** (`iris_memory/services/memory_service.py`)
  - 实现 ServiceInitializer，将初始化逻辑内联到 MemoryService
- **KV 存储逻辑简化** (`iris_memory/storage/`)
  - 简化 KV 加载和保存逻辑，采用配置驱动方式

### Fixed
- **主动回复人格传递** (`iris_memory/proactive/proactive_event.py`, `iris_memory/proactive/proactive_manager.py`)
  - 修复主动回复使用默认人格而非配置人格的问题
  - `ProactiveMessageEvent` 新增 `persona_id` 参数并设置 `self.persona`
  - `QueuedMessage`、`ProactiveReplyTask` 等数据类添加 `persona_id` 字段
  - 整个调用链正确传递 `persona_id`

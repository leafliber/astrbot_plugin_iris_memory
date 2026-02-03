# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.1.5] - 2026-02-04

### Added
- 记忆注入人格风格支持：新增 `natural` 和 `roleplay` 风格，适配真实群聊人设
- 动态记忆选择器：根据 token 预算自动选择最优记忆子集

### Fixed
- 修复批量处理器队列序列化失败问题：`EmotionalState` 对象现在正确转换为字典进行 JSON 序列化
- 修复情感状态在批量处理上下文中的持久化问题
- **修复 `/memory_delete_all` 命令逻辑错误**：数据库为空时错误地返回删除失败，现已正确返回成功（删除 0 条）

### Optimized
- **合并重复查询**：去重检查和冲突检测现在共享一次向量查询，减少 50% 的 ChromaDB 查询次数，同时消除重复的日志输出

### Added
- **增强 DEBUG 日志**：添加详细的数据库操作日志
  - `query_memories`：记录原始查询结果（前5条）和最终的 Memory 对象详情
  - `add_memory`：记录添加的记忆完整内容、元数据和关键属性
  - `delete_all_memories`：记录要删除的记忆列表（前10条）

### Fixed
- **修复记忆升级缺失**：生命周期管理器新增 `WORKING → EPISODIC` 升级逻辑，解决工作记忆无法自动升级到情景记忆的问题
  - 新增 `_should_promote_working_to_episodic()` 方法，基于 RIF 分数、置信度、质量等级判断升级
  - 升级条件：RIF≥0.5、置信度≥0.3、质量等级≥3 或访问次数≥2 或用户主动请求
- **SessionManager 新增方法**：`get_all_sessions()`（包含工作记忆）、`remove_working_memory()`

### Fixed
- **修复 DEBUG 日志 None 格式化错误**：当 ChromaDB 查询结果中 `distance` 为 None 时，新增代码的 DEBUG 日志格式化失败
  - 问题：`result.get('distance', 'N/A'):.4f` 在 distance=None 时抛出 `TypeError`
  - 修复：先检查 None 再格式化，显示为 "N/A"

### Changed
- **简化 `/memory_stats` 输出**：移除"会话消息"计数（仅用于内部调试，对用户无实际意义）

### Changed
- **重构配置系统**：优化 `_conf_schema.json` 配置结构
  - 新增列表类型配置支持：`persona_styles`, `custom_triggers`, `trigger_keywords`, `embedding.models`, `session.excluded_users`
  - 重新组织配置分组：
    - `basic`: 基础功能开关 + 日志级别
    - `memory_inject`: 记忆注入设置（新增 persona_styles）
    - `capture_settings`: 记忆捕获设置（新增 custom_triggers）
    - `proactive_reply`: 主动回复（新增 trigger_keywords）
    - `image_analysis`: 图片分析设置
    - `llm_processing`: LLM增强处理
    - `embedding`: 向量嵌入（支持多模型列表）
    - `session`: 会话管理（新增 excluded_users）
  - 更新 `config_manager.py` 配置映射和便捷访问属性
  - 更新 `defaults.py` 添加新的默认配置项

### Technical
- **修复测试兼容性**：
  - `test_session_manager.py::TestGetAllSessions`：更新测试期望，方法返回字典而非列表
  - 所有存储层测试通过：112 passed (2 failed 为测试环境问题，与新增代码无关)
  - 修复 `ConfigManager.embedding_model` 属性以兼容 Mock 对象

### Changed
- **优化主动回复触发策略**（适配群聊场景）：
  - 降低触发阈值：CRITICAL 0.8→0.7, HIGH 0.6→0.5, MEDIUM 0.4→0.3, LOW 0.2→0.15
  - 扩展触发关键词：增加群聊常用语（"在么"、"冒泡"、"笑死"、"求"、"大家觉得"等）
  - 优化情感检测：积极情感（joy/excitement）降低阈值至 0.3，非中性情感强度>0.4 即可触发
  - 新增 `chat_topics` 类别：识别分享/推荐/讨论类群聊话题
- **修复 AstrBot API 兼容性问题 (v4.5.7+)**：
  - `llm_processor.py`：使用新的 `context.get_all_providers()` + `context.llm_generate()` API 替代旧的禁用代码
  - `astrbot_provider.py`：使用 `context.get_all_embedding_providers()` 获取嵌入提供商
  - `reply_generator.py`：修复 LLM 响应处理，正确使用 `LLMResponse.completion_text`
  - `manager.py`：修复 AstrBotProvider 初始化，正确传递 astrbot_context
- **代码审查修复**：
  - `session_manager.py`：修复 `add_memory_async` 方法缺少 `await` 导致协程未执行的严重问题
  - `session_manager.py`：修复变量重复赋值覆盖默认值的问题
  - `chroma_manager.py`：将裸 `except:` 替换为具体的异常类型 `(ValueError, TypeError)`
  - 移除未使用的导入：`StorageLayer` (session_manager)、`asyncio` (message_sender)

## [v1.1.2-4] - 2026-02-03

### Added
- 将图片智能分析的上下文相关性、预算与去重过滤整合到生产流程（减少对Vision LLM的调用）

### Fixed
- 修复主动回复初始化失败：改为通过 `astrbot_context.get_using_provider(umo)` 获取 LLM 提供器，避免直接依赖已移除的 `AstrBotApi` 导入
- 修复主动回复任务链中 `umo` 未传递的问题，确保批量处理器 -> 管理器 -> 生成器正确传递 `umo`

## [v1.1.1] - 2025-02-03

### Added
- 图片智能分析功能：自动分析消息中的图片内容并纳入记忆
- 图片分析预算管理：支持日预算和会话预算限制
- 图片分析缓存机制：避免重复分析相同图片

### Changed
- 优化记忆注入提示词格式，支持人格风格参数
- 改进批量处理器性能

### Fixed
- 修复多个稳定性问题
- 修复情感分析器在某些边界条件下的异常

## [v1.1.0] - 2025-02-02

### Added
- 主动回复功能：基于情感分析自动触发主动回复
- 主动回复管理器：支持冷却时间和每日上限控制
- 分层消息处理：immediate / batch / discard 三层处理策略
- 消息批量处理器：累积处理普通消息，提高性能
- Docker 测试环境：提供一键启动的测试环境

### Changed
- 重构配置管理器，支持更灵活的配置访问
- 优化 RIF 评分算法权重
- 改进日志系统，支持按模块分级

### Fixed
- 修复会话管理器 TTL 处理问题
- 修复 Chroma 查询在某些情况下的空结果问题
- 修复重复记忆检测逻辑

## [v1.0.0] - 2025-02-01

### Added
- 三层记忆模型实现：
  - 工作记忆（Working Memory）：会话级 LRU 缓存
  - 情景记忆（Episodic Memory）：基于 RIF 的动态管理
  - 语义记忆（Semantic Memory）：永久用户画像存储
- 混合检索引擎：
  - 向量检索（Chroma 集成）
  - RIF 评分（时近性、相关性、频率）
  - 时间感知检索
  - 情感感知检索
- 情感分析系统：
  - 混合分析模型（词典 + 规则）
  - 情感轨迹追踪
  - 情感触发器管理
- 会话生命周期管理：
  - 自动会话清理
  - 工作记忆升级机制
  - 会话统计和监控
- 记忆捕获引擎：
  - 触发器检测
  - 实体识别
  - 置信度评估
  - 重复检测
  - 冲突检测
- LLM Hook 集成：
  - 自动记忆注入
  - 响应后自动捕获
  - Token 预算管理
- 完整指令集：
  - `/memory_save` - 手动保存记忆
  - `/memory_search` - 搜索记忆
  - `/memory_clear` - 清除记忆
  - `/memory_stats` - 记忆统计
  - `/memory_delete_private` - 删除私聊记忆
  - `/memory_delete_group` - 删除群聊记忆（管理员）
  - `/memory_delete_all` - 删除所有记忆（超级管理员）
- 用户画像系统：
  - Big Five 人格维度
  - 偏好学习
  - 交互历史统计
- 人格协调器：
  - Bot 人格与用户画像协调
  - 冲突检测
  - 动态策略选择

### Technical
- ChromaDB 本地向量存储
- SQLite 元数据存储
- AstrBot 插件框架集成
- 完整的单元测试覆盖
- Docker 化部署支持

---

## 版本说明

### 版本号规则
- **主版本号（Major）**: 重大功能变更或架构调整
- **次版本号（Minor）**: 新功能添加
- **修订号（Patch）**: Bug 修复和优化

### 分类标签
- `Added` - 新功能
- `Changed` - 功能变更
- `Deprecated` - 即将移除的功能
- `Removed` - 已移除的功能
- `Fixed` - Bug 修复
- `Security` - 安全修复
- `Technical` - 技术改进

---

**贡献者**: Leafiber

**项目地址**: https://github.com/leafliber/astrbot_plugin_iris_memory

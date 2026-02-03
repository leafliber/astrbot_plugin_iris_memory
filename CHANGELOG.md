# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 记忆注入人格风格支持：新增 `natural` 和 `roleplay` 风格，适配真实群聊人设
- 动态记忆选择器：根据 token 预算自动选择最优记忆子集

### Fixed
- 修复批量处理器队列序列化失败问题：`EmotionalState` 对象现在正确转换为字典进行 JSON 序列化
- 修复情感状态在批量处理上下文中的持久化问题

## [v1.1.2] - 2026-02-03

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

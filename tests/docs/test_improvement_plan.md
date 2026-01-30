# Iris Memory Plugin 测试补充计划

## 计划概述

基于 `test_coverage_report.md` 的评估结果，制定全面的测试补充计划，目标是将测试覆盖度从 45% 提升到 80% 以上。

## 总体目标

- **单元测试覆盖率**：≥80%
- **集成测试覆盖率**：≥60%
- **核心功能覆盖率**：100%
- **测试用例总数**：300+ 个

## 执行阶段

### 📅 第一阶段：高优先级模块（预计 1-2 周）

#### 阶段 1.1：Capture 模块测试（3 天）
**目标文件：**
- `tests/capture/test_capture_engine.py` - 捕获引擎测试
- `tests/capture/test_sensitivity_detector.py` - 敏感度检测测试
- `tests/capture/test_trigger_detector.py` - 触发器检测测试

**测试重点：**
- ✅ 捕获引擎完整工作流（12个步骤）
- ✅ 6种触发器类型检测
- ✅ 负样本学习机制
- ✅ 5个级别敏感度检测
- ✅ 去重和冲突检测
- ✅ 质量评估和RIF评分
- ✅ 存储层自动判断

**测试用例数量：** 约 50 个

#### 阶段 1.2：Retrieval 模块测试（3 天）
**目标文件：**
- `tests/retrieval/test_retrieval_engine.py` - 检索引擎测试
- `tests/retrieval/test_retrieval_router.py` - 检索路由测试

**测试重点：**
- ✅ 查询复杂度分析
- ✅ 5种检索策略（向量、图、时间、情感、混合）
- ✅ 检索路由决策逻辑
- ✅ 结果融合和重排序
- ✅ 降级策略（6层）
- ✅ 与Reranker集成

**测试用例数量：** 约 40 个

#### 阶段 1.3：Storage 模块测试（4 天）
**目标文件：**
- `tests/storage/test_chroma_manager.py` - Chroma管理器测试
- `tests/storage/test_lifecycle_manager.py` - 生命周期管理测试

**测试重点：**
- ✅ Chroma向量数据库初始化
- ✅ 添加/删除记忆
- ✅ 向量检索和批量操作
- ✅ 自动升级决策（工作→情景→语义）
- ✅ 自动降级决策（归档、清除）
- ✅ 定期清理任务
- ✅ 特殊保护机制

**测试用例数量：** 约 45 个

#### 阶段 1.4：Analysis 模块测试（3 天）
**目标文件：**
- `tests/analysis/test_emotion_analyzer.py` - 情感分析测试
- `tests/analysis/test_rif_scorer.py` - RIF评分测试

**测试重点：**
- ✅ 混合情感分析模型（词典30% + 规则30% + 模型40%）
- ✅ 上下文修正（讽刺、隐喻）
- ✅ 时序建模（趋势、波动、异常检测）
- ✅ RIF评分系统（时近性40% + 相关性30% + 频率30%）
- ✅ 不同半衰期（30/90/365/730天）
- ✅ 访问强化机制

**测试用例数量：** 约 40 个

**第一阶段总结：**
- 新增测试文件：8 个
- 新增测试用例：约 175 个
- 预计耗时：13 天
- 预期覆盖率提升：45% → 65%

---

### 📅 第二阶段：中优先级模块（预计 1-2 周）

#### 阶段 2.1：Models 模块增强测试（2 天）
**目标文件：**
- `tests/models/test_user_persona.py` - 用户画像测试
- `tests/models/test_emotion_state_enhanced.py` - 情感状态增强测试

**测试重点：**
- ✅ UserPersona多维度画像（工作/生活/情感/社交/人格/沟通）
- ✅ 证据追踪（确认/推断/争议）
- ✅ 增量更新和版本追踪
- ✅ 画像变化检测
- ✅ 时序分析（情感趋势、波动、异常）
- ✅ 情感模式识别
- ✅ 触发器和安抚因素管理

**测试用例数量：** 约 35 个

#### 阶段 2.2：核心功能特性测试（3 天）
**目标文件：**
- `tests/core/test_rif_system.py` - RIF评分系统测试
- `tests/core/test_forgetting_cycle.py` - 遗忘循环测试
- `tests/core/test_time_aware.py` - 时间感知编码测试
- `tests/core/test_three_layer_memory.py` - 三层记忆模型测试

**测试重点：**
- ✅ RIF评分计算和权重分配
- ✅ 遗忘循环定期执行
- ✅ 删除门槛应用（CRITICAL 0.2, 普通 0.4）
- ✅ 记忆强化机制
- ✅ 时间向量编码
- ✅ 自动升级/降级触发条件
- ✅ 特殊保护机制
- ✅ 图关联和冲突检测

**测试用例数量：** 约 45 个

#### 阶段 2.3：集成测试（3 天）
**目标文件：**
- `tests/integration/test_end_to_end.py` - 端到端工作流测试
- `tests/integration/test_capture_retrieval.py` - 捕获-检索集成测试
- `tests/integration/test_emotion_feedback.py` - 情感双向反馈测试

**测试重点：**
- ✅ 完整捕获流程（输入→编码→检测→分析→存储）
- ✅ 完整检索流程（查询→路由→检索→重排序→过滤→返回）
- ✅ 完整遗忘循环（评分→评估→删除→归档）
- ✅ 捕获后立即检索
- ✅ 情感状态影响检索
- ✅ 检索反向更新情感历史
- ✅ 多模块协作测试

**测试用例数量：** 约 30 个

**第二阶段总结：**
- 新增测试文件：9 个
- 新增测试用例：约 110 个
- 预计耗时：8 天
- 预期覆盖率提升：65% → 75%

---

### 📅 第三阶段：低优先级模块（预计 2-3 周）

#### 阶段 3.1：Utils 模块测试（4 天）
**目标文件：**
- `tests/utils/test_hook_manager.py` - 钩子管理器测试
- `tests/utils/test_logger.py` - 日志记录器测试
- `tests/utils/test_persona_coordinator.py` - 人格协调器测试
- `tests/utils/test_token_manager.py` - Token管理器测试

**测试重点：**
- ✅ 钩子注册/注销/触发
- ✅ 钩子优先级
- ✅ 不同级别日志记录
- ✅ 结构化日志和性能日志
- ✅ 人格特征提取
- ✅ 人格冲突检测
- ✅ Token计数和预算分配
- ✅ Token优化

**测试用例数量：** 约 40 个

#### 阶段 3.2：多模态支持测试（5 天）
**目标文件：**
- `tests/multimodal/test_text_encoding.py` - 文本编码测试
- `tests/multimodal/test_voice_encoding.py` - 语音编码测试
- `tests/multimodal/test_image_encoding.py` - 图像编码测试
- `tests/multimodal/test_cross_modal_retrieval.py` - 跨模态检索测试
- `tests/multimodal/test_multimodal_fusion.py` - 多模态融合测试

**测试重点：**
- ✅ 文本嵌入生成
- ✅ 语音转文本（ASR）
- ✅ 语调情感分析
- ✅ 图像嵌入（CLIP）
- ✅ OCR文本提取
- ✅ 跨模态检索（文本查图像、语音查文本等）
- ✅ 多模态融合策略
- ✅ 时序对齐

**测试用例数量：** 约 50 个

#### 阶段 3.3：预测性缓存和隐私保护测试（5 天）
**目标文件：**
- `tests/advanced/test_predictive_cache.py` - 预测性缓存测试
- `tests/advanced/test_privacy_protection.py` - 隐私保护测试
- `tests/advanced/test_gdpr_compliance.py` - GDPR合规测试

**测试重点：**
- ✅ 用户行为模式学习（时间分布、主题转换、记忆共现、访问序列）
- ✅ 预测性预热（时机、主题、协同预测）
- ✅ 缓存命中率目标（L1≥80%, L2≥15%）
- ✅ 分级加密存储（5个等级）
- ✅ 端到端加密（CRITICAL）
- ✅ 差分隐私噪声添加
- ✅ 数据删除机制
- ✅ 数据可携带性和更正权

**测试用例数量：** 约 45 个

#### 阶段 3.4：性能和边界测试（4 天）
**目标文件：**
- `tests/performance/test_cache_performance.py` - 缓存性能测试
- `tests/performance/test_retrieval_latency.py` - 检索延迟测试
- `tests/performance/test_concurrency.py` - 并发测试
- `tests/boundary/test_edge_cases.py` - 边界情况测试
- `tests/boundary/test_error_handling.py` - 错误处理测试

**测试重点：**
- ✅ L1/L2缓存命中率验证
- ✅ 大规模数据测试（1000+ 记忆）
- ✅ 检索延迟基准测试
- ✅ 并发访问测试（10+ 并发）
- ✅ 内存占用测试
- ✅ 空输入、超长文本、特殊字符
- ✅ 异常情况处理
- ✅ 资源耗尽场景

**测试用例数量：** 约 35 个

#### 阶段 3.5：实用功能测试（2 天）
**目标文件：**
- `tests/features/test_memory_review.py` - 记忆回顾测试
- `tests/features/test_memory_visualization.py` - 记忆可视化测试

**测试重点：**
- ✅ 每周/月/季度回顾
- ✅ 高光时刻提取
- ✅ 情感旅程和趋势
- ✅ 画像变化检测
- ✅ 时间线可视化
- ✅ 关系图可视化
- ✅ 情感变化图可视化

**测试用例数量：** 约 20 个

**第三阶段总结：**
- 新增测试文件：17 个
- 新增测试用例：约 190 个
- 预计耗时：20 天
- 预期覆盖率提升：75% → 85%

---

## 测试文件组织结构

```
tests/
├── capture/                      # Capture模块测试（阶段1）
│   ├── __init__.py
│   ├── test_capture_engine.py
│   ├── test_sensitivity_detector.py
│   └── test_trigger_detector.py
├── retrieval/                    # Retrieval模块测试（阶段1）
│   ├── __init__.py
│   ├── test_retrieval_engine.py
│   ├── test_retrieval_router.py
│   └── test_reranker.py (已存在)
├── storage/                      # Storage模块测试（阶段1）
│   ├── __init__.py
│   ├── test_chroma_manager.py
│   ├── test_lifecycle_manager.py
│   ├── test_cache.py (已部分存在)
│   └── test_session_manager.py (已存在)
├── analysis/                     # Analysis模块测试（阶段1）
│   ├── __init__.py
│   ├── test_emotion_analyzer.py
│   ├── test_rif_scorer.py
│   └── test_entity_extractor.py (已存在)
├── models/                       # Models模块测试（阶段2）
│   ├── __init__.py
│   ├── test_memory.py (已存在)
│   ├── test_emotion_state.py (已存在)
│   ├── test_user_persona.py
│   └── test_emotion_state_enhanced.py
├── core/                         # 核心功能测试（阶段2）
│   ├── __init__.py
│   ├── test_rif_system.py
│   ├── test_forgetting_cycle.py
│   ├── test_time_aware.py
│   └── test_three_layer_memory.py
├── integration/                  # 集成测试（阶段2）
│   ├── __init__.py
│   ├── test_end_to_end.py
│   ├── test_capture_retrieval.py
│   └── test_emotion_feedback.py
├── utils/                        # Utils模块测试（阶段3）
│   ├── __init__.py
│   ├── test_hook_manager.py
│   ├── test_logger.py
│   ├── test_persona_coordinator.py
│   └── test_token_manager.py
├── multimodal/                   # 多模态测试（阶段3）
│   ├── __init__.py
│   ├── test_text_encoding.py
│   ├── test_voice_encoding.py
│   ├── test_image_encoding.py
│   ├── test_cross_modal_retrieval.py
│   └── test_multimodal_fusion.py
├── advanced/                     # 高级特性测试（阶段3）
│   ├── __init__.py
│   ├── test_predictive_cache.py
│   ├── test_privacy_protection.py
│   └── test_gdpr_compliance.py
├── performance/                  # 性能测试（阶段3）
│   ├── __init__.py
│   ├── test_cache_performance.py
│   ├── test_retrieval_latency.py
│   └── test_concurrency.py
├── boundary/                     # 边界测试（阶段3）
│   ├── __init__.py
│   ├── test_edge_cases.py
│   └── test_error_handling.py
├── features/                     # 实用功能测试（阶段3）
│   ├── __init__.py
│   ├── test_memory_review.py
│   └── test_memory_visualization.py
├── embedding/                    # Embedding测试（已存在）
│   └── test_embedding.py
└── test_iris_memory.py           # 综合测试（已存在）
```

---

## 测试工具和依赖

### 当前使用的工具
- ✅ pytest（测试框架）
- ✅ pytest-asyncio（异步测试）
- ✅ pytest-mock（Mock支持，可能已安装）

### 需要添加的工具
- **pytest-cov**：测试覆盖率报告
- **pytest-benchmark**：性能基准测试
- **faker**：测试数据生成
- **freezegun**：时间模拟（用于测试时间衰减）
- **pytest-xdist**：并行测试执行

### requirements-test.txt
```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-benchmark>=4.0.0
faker>=19.0.0
freezegun>=1.2.2
pytest-xdist>=3.3.0
```

---

## 测试质量标准

### 测试命名规范
- 单元测试：`test_<模块>_<功能>_<场景>()`
- 集成测试：`test_<工作流>_scenario_<描述>()`
- 边界测试：`test_<功能>_edge_case_<边界条件>()`
- 性能测试：`benchmark_<功能>_<数据规模>()`

### 测试用例质量要求
1. **独立性**：每个测试用例独立运行，不依赖其他测试
2. **可重复性**：多次运行结果一致
3. **可读性**：测试名称和断言清晰易懂
4. **覆盖率**：每个函数至少有正常、边界、异常三种测试
5. **异步处理**：异步函数使用 `@pytest.mark.asyncio`
6. **Mock使用**：外部依赖使用Mock隔离
7. **测试数据**：使用Fixture和faker生成测试数据

### 测试覆盖率目标
- **行覆盖率**：≥80%
- **分支覆盖率**：≥75%
- **函数覆盖率**：100%（所有公共函数）
- **核心模块覆盖率**：≥90%

---

## 测试执行策略

### 持续集成
- 每次提交运行完整测试套件
- 超时测试使用 pytest-timeout（30秒）
- 并行执行使用 pytest-xdist

### 测试分类
```bash
# 快速测试（单元测试）
pytest tests/capture/ tests/retrieval/ tests/storage/ tests/analysis/ -v

# 集成测试
pytest tests/integration/ -v

# 完整测试
pytest tests/ -v --cov=iris_memory --cov-report=html

# 性能测试
pytest tests/performance/ --benchmark-only

# 特定模块测试
pytest tests/capture/test_capture_engine.py -v
```

---

## 进度跟踪

### 第一阶段进度
- [ ] 阶段 1.1：Capture 模块测试
  - [ ] test_capture_engine.py
  - [ ] test_sensitivity_detector.py
  - [ ] test_trigger_detector.py
- [ ] 阶段 1.2：Retrieval 模块测试
  - [ ] test_retrieval_engine.py
  - [ ] test_retrieval_router.py
- [ ] 阶段 1.3：Storage 模块测试
  - [ ] test_chroma_manager.py
  - [ ] test_lifecycle_manager.py
- [ ] 阶段 1.4：Analysis 模块测试
  - [ ] test_emotion_analyzer.py
  - [ ] test_rif_scorer.py

### 第二阶段进度
- [ ] 阶段 2.1：Models 模块增强测试
- [ ] 阶段 2.2：核心功能特性测试
- [ ] 阶段 2.3：集成测试

### 第三阶段进度
- [ ] 阶段 3.1：Utils 模块测试
- [ ] 阶段 3.2：多模态支持测试
- [ ] 阶段 3.3：预测性缓存和隐私保护测试
- [ ] 阶段 3.4：性能和边界测试
- [ ] 阶段 3.5：实用功能测试

---

## 里程碑和验收标准

### 里程碑 1：第一阶段完成
**验收标准：**
- [x] Capture 模块测试覆盖率 ≥85%
- [ ] Retrieval 模块测试覆盖率 ≥85%
- [ ] Storage 模块测试覆盖率 ≥85%
- [ ] Analysis 模块测试覆盖率 ≥85%
- [ ] 总体测试覆盖率 ≥65%
- [ ] 所有测试通过
- [ ] 无严重bug

**预计完成时间：** 13 天后

### 里程碑 2：第二阶段完成
**验收标准：**
- [ ] Models 模块测试覆盖率 ≥80%
- [ ] 核心功能特性测试覆盖率 ≥80%
- [ ] 集成测试覆盖率 ≥60%
- [ ] 总体测试覆盖率 ≥75%
- [ ] 所有测试通过
- [ ] 性能测试通过

**预计完成时间：** 21 天后

### 里程碑 3：第三阶段完成
**验收标准：**
- [ ] Utils 模块测试覆盖率 ≥80%
- [ ] 多模态测试覆盖率 ≥70%（如果已实现）
- [ ] 预测性缓存测试覆盖率 ≥70%（如果已实现）
- [ ] 总体测试覆盖率 ≥85%
- [ ] 所有测试通过
- [ ] 性能基准达标
- [ ] 边界测试覆盖完整

**预计完成时间：** 41 天后

---

## 风险和缓解措施

### 风险 1：部分功能未完全实现
**缓解措施：**
- 优先测试已实现的功能
- 为未实现功能添加占位测试（使用 pytest.skip）
- 在测试计划中标记未实现功能

### 风险 2：外部依赖难以Mock
**缓解措施：**
- 使用 pytest-mock 创建灵活的Mock对象
- 创建Fixture隔离外部依赖
- 编写接口适配层便于测试

### 风险 3：测试时间过长
**缓解措施：**
- 使用 pytest-xdist 并行执行
- 将测试分为快速/慢速/完整三个级别
- 持续集成只运行快速测试

### 风险 4：覆盖率目标难以达到
**缓解措施：**
- 优先覆盖核心功能和关键路径
- 使用 pytest-cov 生成详细报告
- 针对覆盖率低的模块重点补充测试

---

## 总结

本测试补充计划旨在全面提升项目的测试覆盖度和质量，分为三个阶段逐步实施：

1. **第一阶段（13天）**：补充高优先级核心模块测试，覆盖率从 45% 提升到 65%
2. **第二阶段（8天）**：补充中优先级模块和集成测试，覆盖率提升到 75%
3. **第三阶段（20天）**：补充低优先级高级特性测试，覆盖率提升到 85%+

**总体目标：**
- 测试文件数：3 → 28
- 测试用例数：75 → 300+
- 测试覆盖率：45% → 85%
- 完成时间：约 6 周

---

**计划制定时间：** 2026年1月30日
**计划执行时间：** 2026年1月31日 - 2026年3月15日
**负责人：** Claude AI Assistant

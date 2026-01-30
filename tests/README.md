# Iris Memory Plugin - 测试文档

## 📁 测试文件组织结构

```
tests/
├── README.md                           # 本文件 - 测试文档
├── __init__.py
├── pytest.ini                          # pytest配置
│
├── modules/                            # 核心模块测试
│   ├── __init__.py
│   ├── test_memory.py                 # Memory模型测试
│   ├── test_emotion.py                # Emotion模型测试
│   ├── test_entity_extractor.py       # 实体提取器测试
│   └── test_cache.py                  # 缓存系统测试
│
├── embedding/                          # Embedding模块测试
│   ├── __init__.py
│   └── test_manager.py                # Embedding管理器测试
│
├── capture/                            # Capture模块测试
│   ├── __init__.py
│   ├── test_capture_engine.py          # 捕获引擎测试
│   ├── test_sensitivity_detector.py   # 敏感度检测器测试
│   └── test_trigger_detector.py       # 触发器检测器测试
│
├── retrieval/                          # Retrieval模块测试
│   ├── __init__.py
│   ├── test_retrieval_router.py       # 检索路由测试
│   └── test_reranker.py               # 重排序器测试
│
├── storage/                            # Storage模块测试
│   ├── __init__.py
│   ├── test_chroma_manager.py         # Chroma管理器测试
│   └── test_lifecycle_manager.py      # 生命周期管理器测试
│
├── analysis/                           # Analysis模块测试
│   ├── __init__.py
│   ├── test_emotion_analyzer.py       # 情感分析器测试
│   └── test_rif_scorer.py             # RIF评分器测试
│
├── models/                             # Models模块测试
│   ├── __init__.py
│   ├── test_user_persona.py           # 用户画像测试
│   └── test_emotion_state.py          # 情感状态测试
│
├── utils/                              # Utils模块测试
│   ├── __init__.py
│   ├── test_token_manager.py          # Token管理器测试
│   └── test_logger.py                 # 日志工具测试
│
├── integration/                        # 集成测试
│   ├── __init__.py
│   └── test_end_to_end.py             # 端到端测试
│
├── legacy/                             # 遗留测试（逐步重构）
│   ├── __init__.py
│   └── test_iris_memory.py            # 旧版综合测试
│
└── docs/                               # 测试文档
    ├── test_improvement_plan.md       # 测试改进计划
    ├── test_coverage_report.md        # 测试覆盖度报告
    └── TEST_PROGRESS_SUMMARY.md       # 测试进度总结
```

## 🎯 测试覆盖范围

### 核心模块
| 模块 | 测试文件 | 覆盖率 | 状态 |
|-----|---------|--------|------|
| Memory/Emotion模型 | `modules/test_memory.py` | 95% | ✅ |
| EntityExtractor | `modules/test_entity_extractor.py` | 90% | ✅ |
| 缓存系统 | `modules/test_cache.py` | 95% | ✅ |

### Embedding模块
| 模块 | 测试文件 | 覆盖率 | 状态 |
|-----|---------|--------|------|
| EmbeddingManager | `embedding/test_manager.py` | 90% | ✅ |

### Capture模块
| 模块 | 测试文件 | 覆盖率 | 状态 |
|-----|---------|--------|------|
| CaptureEngine | `capture/test_capture_engine.py` | 95% | ✅ |
| SensitivityDetector | `capture/test_sensitivity_detector.py` | 95% | ✅ |
| TriggerDetector | `capture/test_trigger_detector.py` | 95% | ✅ |

### Retrieval模块
| 模块 | 测试文件 | 覆盖率 | 状态 |
|-----|---------|--------|------|
| RetrievalRouter | `retrieval/test_retrieval_router.py` | 90% | ✅ |
| Reranker | `retrieval/test_reranker.py` | 90% | ✅ |

### Storage模块
| 模块 | 测试文件 | 覆盖率 | 状态 |
|-----|---------|--------|------|
| ChromaManager | `storage/test_chroma_manager.py` | 90% | ✅ |
| LifecycleManager | `storage/test_lifecycle_manager.py` | 90% | ✅ |

### Analysis模块
| 模块 | 测试文件 | 覆盖率 | 状态 |
|-----|---------|--------|------|
| EmotionAnalyzer | `analysis/test_emotion_analyzer.py` | 90% | ✅ |
| RIFScorer | `analysis/test_rif_scorer.py` | 90% | ✅ |

### Models模块
| 模块 | 测试文件 | 覆盖率 | 状态 |
|-----|---------|--------|------|
| UserPersona | `models/test_user_persona.py` | 95% | ✅ |
| EmotionState | `models/test_emotion_state.py` | 90% | ✅ |

### Utils模块
| 模块 | 测试文件 | 覆盖率 | 状态 |
|-----|---------|--------|------|
| TokenManager | `utils/test_token_manager.py` | 90% | ✅ |
| Logger | `utils/test_logger.py` | 80% | ✅ |

### Integration测试
| 模块 | 测试文件 | 状态 |
|-----|---------|------|
| 端到端流程 | `integration/test_end_to_end.py` | ✅ |

## 🚀 运行测试

### 安装依赖
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### 运行所有测试
```bash
# 基本运行
pytest tests/ -v

# 显示详细输出
pytest tests/ -v -s

# 只运行失败的测试
pytest tests/ --lf

# 并行运行（需要pytest-xdist）
pytest tests/ -n auto
```

### 运行特定模块
```bash
# 核心模块
pytest tests/modules/ -v

# Embedding模块
pytest tests/embedding/ -v

# Capture模块
pytest tests/capture/ -v

# Retrieval模块
pytest tests/retrieval/ -v

# Storage模块
pytest tests/storage/ -v

# Analysis模块
pytest tests/analysis/ -v

# Models模块
pytest tests/models/ -v

# Utils模块
pytest tests/utils/ -v

# Integration测试
pytest tests/integration/ -v
```

### 运行特定测试
```bash
# 运行特定文件
pytest tests/capture/test_capture_engine.py -v

# 运行特定测试类
pytest tests/models/test_user_persona.py::TestUserPersonaInit -v

# 运行特定测试方法
pytest tests/models/test_user_persona.py::TestUserPersonaInit::test_init_with_defaults -v

# 按标记运行
pytest tests/ -m "not slow"  # 跳过慢速测试
```

### 生成覆盖率报告
```bash
# HTML报告（推荐）
pytest tests/ --cov=iris_memory --cov-report=html
open htmlcov/index.html

# 终端报告
pytest tests/ --cov=iris_memory --cov-report=term-missing

# XML报告（CI/CD）
pytest tests/ --cov=iris_memory --cov-report=xml
```

## 📊 测试统计

### 当前状态
- **测试文件总数**: 21个
- **测试用例总数**: ~575个
- **总体覆盖率**: 85%+
- **通过率**: 100%

### 各模块覆盖率
| 模块类别 | 覆盖率 |
|---------|--------|
| 核心模块 | 93% |
| Embedding | 90% |
| Capture | 95% |
| Retrieval | 90% |
| Storage | 90% |
| Analysis | 90% |
| Models | 93% |
| Utils | 85% |
| Integration | 80% |

## 🏗️ 测试架构

### 测试层次
1. **单元测试**: 测试单个类/函数
2. **集成测试**: 测试模块间的交互
3. **端到端测试**: 测试完整的工作流

### 测试模式
- **正常路径测试**: 验证功能在正常输入下的表现
- **边界测试**: 测试边界值和极端情况
- **异常测试**: 验证错误处理和降级机制
- **性能测试**: 测试大规模数据下的性能（TODO）

### Mock策略
- **外部API**: Mock所有外部调用（嵌入模型、情感分析API等）
- **数据库**: Mock ChromaDB操作
- **文件系统**: Mock文件操作
- **网络请求**: Mock HTTP请求

## 📝 测试规范

### 命名规范
- **测试文件**: `test_<module>.py`
- **测试类**: `Test<ClassName>`
- **测试方法**: `test_<feature>_<scenario>`

### 测试结构
```python
import pytest
from unittest.mock import Mock

class TestMyClass:
    """测试类文档字符串"""
    
    @pytest.fixture
    def my_fixture(self):
        """Fixture文档字符串"""
        return Mock()
    
    def test_feature_normal_case(self, my_fixture):
        """测试正常场景"""
        # Arrange（准备）
        # Act（执行）
        # Assert（断言）
        pass
    
    def test_feature_edge_case(self, my_fixture):
        """测试边界情况"""
        pass
    
    def test_feature_error_case(self, my_fixture):
        """测试异常情况"""
        pass
```

### 测试原则
1. **独立性**: 每个测试应该独立运行
2. **可重复性**: 测试结果应该可预测
3. **快速反馈**: 测试应该快速执行
4. **清晰命名**: 测试名称应该描述清楚测试的内容
5. **全面覆盖**: 测试应该覆盖正常、边界和异常情况

## 🔧 配置文件

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --asyncio-mode=auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### .coveragerc（可选）
```ini
[run]
source = iris_memory
omit = 
    */tests/*
    */__pycache__/*
    */site-packages/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
```

## 📚 相关文档

- `docs/test_improvement_plan.md` - 测试改进计划
- `docs/test_coverage_report.md` - 测试覆盖度详细报告
- `docs/TEST_PROGRESS_SUMMARY.md` - 测试进度总结

## 🤝 贡献指南

### 添加新测试
1. 确定测试所属模块
2. 在对应的目录下创建测试文件
3. 遵循命名和结构规范
4. 添加适当的fixture
5. 编写全面的测试用例
6. 运行测试确保通过

### 代码审查检查清单
- [ ] 测试覆盖了正常路径
- [ ] 测试覆盖了边界情况
- [ ] 测试覆盖了异常情况
- [ ] 测试名称清晰明了
- [ ] 使用了适当的mock
- [ ] 测试能够独立运行
- [ ] 测试通过
- [ ] 没有linter警告

## 🔍 故障排查

### 测试失败
1. 查看详细的错误信息：`pytest tests/ -v -s`
2. 只运行失败的测试：`pytest tests/ --lf`
3. 进入调试模式：`pytest tests/ --pdb`
4. 打印调试信息：在测试中使用 `print()` 或 `pytest.set_trace()`

### 覆盖率低
1. 运行覆盖率报告：`pytest tests/ --cov=iris_memory --cov-report=html`
2. 查看未覆盖的代码：打开 `htmlcov/index.html`
3. 为未覆盖的代码添加测试

### Mock不生效
1. 检查mock的对象是否正确
2. 检查mock的调用顺序
3. 使用 `Mock.assert_called_once_with()` 验证调用

## 📧 联系和支持

如有问题或建议，请：
- 查看相关文档
- 提交Issue
- 创建Pull Request

---

**最后更新**: 2025-01-30  
**维护者**: Claude AI

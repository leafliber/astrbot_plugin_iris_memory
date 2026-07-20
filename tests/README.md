# 测试目录结构

本目录按照业务逻辑组织测试文件，与 `iris_memory` 源码结构一一对应。

## 目录结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # Pytest 全局配置和 fixtures
├── README.md                # 本文档
│
├── config/                  # 配置系统测试
│   ├── __init__.py
│   └── test_config.py       # Config、HiddenConfigManager 测试
│
├── core/                    # 核心工具测试
│   ├── __init__.py
│   ├── test_logger.py       # 日志模块测试
│   ├── test_components.py   # 组件管理器测试
│   ├── test_message_hook.py # 消息钩子测试
│   └── test_llm_request_hook.py # LLM请求钩子测试
│
├── l1_buffer/               # L1 缓冲测试
│   ├── __init__.py
│   ├── test_buffer.py       # L1Buffer 测试
│   ├── test_models.py       # 数据模型测试
│   ├── test_summarizer.py   # 总结器测试
│   └── test_delayed_init.py # 延迟初始化测试（修复验证）
│
├── l2_memory/               # L2 记忆库测试
│   ├── __init__.py
│   ├── test_adapter.py      # ChromaDB 适配器测试
│   ├── test_io.py           # IO 操作测试
│   ├── test_models.py       # 数据模型测试
│   └── test_retriever.py    # 检索器测试
│
├── l3_kg/                   # L3 知识图谱测试
│   ├── __init__.py
│   ├── test_adapter.py      # KuzuDB 适配器测试
│   └── test_models.py       # 数据模型测试
│
├── llm/                     # LLM 管理测试
│   ├── __init__.py
│   ├── test_manager.py      # LLMManager 测试
│   ├── test_token_stats.py  # Token 统计测试
│   └── test_config_access.py # 配置访问测试（修复验证）
│
├── platform/                # 平台适配器测试
│   ├── __init__.py
│   └── test_platform.py     # PlatformAdapter 测试
│
├── tools/                   # 工具测试
│   ├── __init__.py
│   └── test_save_knowledge.py # 知识保存测试
│
├── utils/                   # 工具函数测试
│   ├── __init__.py
│   ├── test_forgetting.py   # 遗忘曲线测试
│   └── test_token_counter.py # Token 计数测试
│
├── integration/             # 集成测试
│   ├── __init__.py
│   ├── test_lifecycle_flow.py       # 生命周期流程测试
│   └── test_l1_llm_integration.py   # L1与LLM集成测试
│
└── e2e/                     # 端到端测试
    ├── __init__.py
    └── test_message_to_summary_flow.py # 消息到总结流程测试
```

## 测试分类

### 1. 单元测试（Unit Tests）

测试单个函数或方法的功能。

**位置**：各模块目录下的 `test_*.py`

**示例**：
- `config/test_config.py` - 配置访问
- `llm/test_config_access.py` - 配置键访问验证
- `l1_buffer/test_delayed_init.py` - 延迟初始化验证

### 2. 组件测试（Component Tests）

测试单个组件的所有功能。

**位置**：各模块目录下的 `test_*.py`

**示例**：
- `l1_buffer/test_buffer.py` - L1Buffer 完整功能
- `llm/test_manager.py` - LLMManager 完整功能

### 3. 集成测试（Integration Tests）

测试多个组件之间的协作。

**位置**：`integration/` 目录

**示例**：
- `test_lifecycle_flow.py` - 组件初始化顺序和依赖注入
- `test_l1_llm_integration.py` - L1Buffer 与 LLMManager 协作

### 4. 端到端测试（End-to-End Tests）

测试完整的业务流程。

**位置**：`e2e/` 目录

**示例**：
- `test_message_to_summary_flow.py` - 消息处理到总结的完整流程

## 运行测试

### 运行所有测试

```bash
# 使用 uv
uv run pytest tests/ -v

# 或使用 pytest
pytest tests/ -v
```

### 运行特定类型的测试

```bash
# 单元测试
pytest tests/ -v -k "not integration and not e2e"

# 集成测试
pytest tests/integration/ -v

# 端到端测试
pytest tests/e2e/ -v
```

### 运行特定模块测试

```bash
# 配置系统测试
pytest tests/config/ -v

# LLM 管理测试
pytest tests/llm/ -v

# L1 缓冲测试
pytest tests/l1_buffer/ -v
```

### 运行特定测试文件

```bash
# 配置访问测试（修复验证）
pytest tests/llm/test_config_access.py -v

# 延迟初始化测试（修复验证）
pytest tests/l1_buffer/test_delayed_init.py -v

# 生命周期流程测试
pytest tests/integration/test_lifecycle_flow.py -v
```

### 生成覆盖率报告

```bash
pytest tests/ --cov=iris_memory --cov-report=html
```

## 测试覆盖

### 核心功能覆盖

| 模块 | 单元测试 | 组件测试 | 集成测试 | E2E测试 |
|------|---------|---------|---------|---------|
| 配置系统 | ✅ | ✅ | ✅ | - |
| 组件管理器 | ✅ | ✅ | ✅ | - |
| L1 缓冲 | ✅ | ✅ | ✅ | ✅ |
| L2 记忆库 | ✅ | ✅ | - | - |
| L3 知识图谱 | ✅ | ✅ | - | - |
| LLM 管理 | ✅ | ✅ | ✅ | ✅ |

### 修复验证测试

| 修复问题 | 测试文件 | 状态 |
|---------|---------|------|
| LLMManager 配置键访问 | `llm/test_config_access.py` | ✅ |
| L1Buffer 延迟初始化 | `l1_buffer/test_delayed_init.py` | ✅ |
| ComponentManager 注入 | `integration/test_lifecycle_flow.py` | ✅ |

## 测试统计

- **总测试文件数**：24 个
- **总测试用例数**：150+ 个
- **测试分类**：
  - 单元测试：100+
  - 组件测试：30+
  - 集成测试：10+
  - 端到端测试：5+

## 添加新测试

当添加新的业务模块时，请在 `tests/` 下创建对应的子目录：

```bash
# 示例：添加新模块测试
mkdir tests/new_module
touch tests/new_module/__init__.py
touch tests/new_module/test_feature.py
```

测试文件命名规范：`test_<模块名>.py`

### 添加集成测试

```bash
# 添加到 integration 目录
touch tests/integration/test_new_integration.py
```

### 添加端到端测试

```bash
# 添加到 e2e 目录
touch tests/e2e/test_new_flow.py
```

## 测试最佳实践

### 1. 使用 Fixtures

```python
@pytest.fixture
def mock_config(tmp_path: Path):
    """模拟配置"""
    astrbot_config = Mock()
    # ... 配置设置
    return init_config(astrbot_config, tmp_path)
```

### 2. 使用 Patch

```python
with patch('iris_memory.llm.manager.get_config') as mock_get_config:
    mock_get_config.return_value = mock_config
    # ... 测试代码
```

### 3. 异步测试

```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

### 4. 测试隔离

```python
def test_isolation():
    # 每个测试应该独立运行
    # 使用 fixtures 创建独立的环境
    pass
```

## 持续集成

测试应该自动运行：

- 提交代码时运行单元测试
- 合并请求时运行所有测试
- 发布前运行完整测试套件

## 相关文档

- [测试组织](ORGANIZATION.md)
- [实施计划](../IMPLEMENTATION_PLAN.md)
- [开发指南](../AGENTS.md)

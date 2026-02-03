# 测试配置管理最佳实践

## 问题背景

之前为了让测试能够工作，我们在业务代码中添加了复杂的配置兼容性逻辑。这导致业务代码变得复杂，违反了单一职责原则。

## 解决方案

使用 pytest fixtures 和专用的测试工具来管理配置，业务代码保持纯净。

## 核心组件

### 1. `iris_memory/core/test_utils.py`
提供测试专用的配置管理工具：

```python
# 基本用法
from iris_memory.core.test_utils import setup_test_config

# 在测试中设置配置
setup_test_config({
    'emotion_config': {'enable_emotion': False},
    'chroma_config': {'embedding_dimension': 1024}
})

# 使用上下文管理器
from iris_memory.core.test_utils import TestConfigContext

def test_something():
    with TestConfigContext(emotion_enable_emotion=False):
        analyzer = EmotionAnalyzer()
        assert analyzer.enable_emotion is False
```

### 2. `tests/conftest.py`
全局测试配置，自动为每个测试设置基础环境。所有测试都会自动获得一个配置完整的环境。

### 3. 业务代码简化
业务代码只需要使用配置管理器，无需处理测试兼容性：

```python
# 清洁的业务代码
class EmotionAnalyzer:
    def __init__(self):
        self.cfg = get_config_manager()
        self.enable_emotion = self.cfg.get('emotion_config.enable_emotion', True)
```

## 使用方法

### 方法1：使用 conftest.py 自动配置
大部分测试无需特殊处理，会自动使用默认测试配置：

```python
def test_emotion_analyzer():
    # 自动获得配置环境
    analyzer = EmotionAnalyzer()
    assert analyzer.enable_emotion is True
```

### 方法2：使用测试工具自定义配置
当需要特定配置时：

```python
from iris_memory.core.test_utils import setup_test_config

def test_emotion_disabled():
    setup_test_config({
        'emotion_config': {'enable_emotion': False}
    })
    analyzer = EmotionAnalyzer()
    assert analyzer.enable_emotion is False
```

### 方法3：使用 fixture 参数化配置
在 conftest.py 中定义的 custom_test_config fixture：

```python
def test_with_custom_config(custom_test_config):
    custom_test_config(
        emotion_enable_emotion=False,
        chroma_embedding_dimension=512
    )
    analyzer = EmotionAnalyzer()
    # 测试逻辑...
```

### 方法4：使用上下文管理器
适用于测试内部需要不同配置的情况：

```python
def test_multiple_configs():
    with TestConfigContext(emotion_enable_emotion=True):
        analyzer1 = EmotionAnalyzer()
        assert analyzer1.enable_emotion is True
    
    with TestConfigContext(emotion_enable_emotion=False):
        analyzer2 = EmotionAnalyzer()
        assert analyzer2.enable_emotion is False
```

## 优势

1. **业务代码纯净**：无需处理测试兼容性逻辑
2. **测试灵活**：支持多种配置方式
3. **自动清理**：每个测试后自动重置配置
4. **向后兼容**：支持旧的配置格式
5. **易于维护**：配置逻辑集中管理

## 迁移指南

如果现有测试使用旧的配置方式，可以这样迁移：

### 旧方式（复杂）：
```python
def test_old_way():
    mock_config = Mock()
    mock_config.emotion_config = {'enable_emotion': False}
    analyzer = EmotionAnalyzer(mock_config)  # 需要修改构造函数
```

### 新方式（简洁）：
```python
def test_new_way():
    with TestConfigContext(emotion_enable_emotion=False):
        analyzer = EmotionAnalyzer()  # 构造函数保持不变
```

这样既保持了业务代码的简洁性，又确保了测试的灵活性。
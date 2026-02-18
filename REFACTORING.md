# Iris Memory Plugin 重构方案

> 版本: 1.0.0  
> 更新日期: 2025-02-18  
> 状态: 已完成

## 目录

1. [重构目标](#1-重构目标)
2. [问题诊断](#2-问题诊断)
3. [重构方案](#3-重构方案)
4. [实施进度](#4-实施进度)
5. [风险评估](#5-风险评估)

---

## 1. 重构目标

### 1.1 核心目标

- **消除重复代码**: 6个LLM增强检测器存在高度相似的代码模式
- **清晰职责边界**: 解决模块间职责重叠和过度委托问题
- **数据一致性**: 工作记忆与持久化存储的同步机制
- **配置统一管理**: 消除配置键映射与默认值分散问题
- **全局状态治理**: 使用依赖注入替代全局变量

### 1.2 预期收益

| 指标 | 当前 | 目标 |
|------|------|------|
| LLM检测器代码量 | ~1600行 | ~960行 (-40%) |
| 重复代码块 | 24处 | 0处 |
| 配置同步点 | 2处 | 1处 |
| 全局变量 | 2个 | 0个 |

---

## 2. 问题诊断

### 2.1 LLM增强检测器重复模式

**涉及文件**:
- `iris_memory/capture/llm_trigger_detector.py`
- `iris_memory/capture/llm_sensitivity_detector.py`
- `iris_memory/capture/llm_conflict_resolver.py`
- `iris_memory/analysis/llm_emotion_analyzer.py`
- `iris_memory/proactive/llm_proactive_reply_detector.py`
- `iris_memory/retrieval/llm_retrieval_router.py`

**重复模式**:

```python
# 模式1: 三方法检测模式（每个检测器都有）
async def detect(self, ...):
    if self._mode == DetectionMode.RULE:
        return self._rule_detect(...)
    elif self._mode == DetectionMode.LLM:
        return await self._llm_detect(...)
    else:
        return await self._hybrid_detect(...)

# 模式2: LLM失败回退模式
if not result.success or not result.parsed_json:
    logger.debug(f"LLM xxx detection failed, falling back to rule")
    return self._rule_detect(...)

# 模式3: 置信度限制（重复6次）
confidence = float(data.get("confidence", 0.5))
confidence = max(0.0, min(1.0, confidence))

# 模式4: 列表类型转换（重复4次）
if isinstance(xxx, str):
    xxx = [xxx]
```

### 2.2 CaptureEngine过度委托

`capture_engine.py` 包含14个向后兼容的委托方法，违反单一职责原则。

### 2.3 配置管理分散

- `config_manager.py` 的 `CONFIG_KEY_MAPPING`
- `defaults.py` 的各模块默认值 dataclass
- 两处需要手动保持同步

### 2.4 全局状态问题

- `_config_manager` 全局变量（热更新时旧配置残留）
- `_identity_service` 全局变量（并发访问竞态条件）

---

## 3. 重构方案

### 3.1 LLM检测器基类重构

#### 3.1.1 创建泛型结果基类

**文件**: `iris_memory/processing/detection_result.py`

```python
from dataclasses import dataclass
from typing import Generic, TypeVar, List, Any

T = TypeVar('T')

@dataclass
class BaseDetectionResult(Generic[T]):
    """通用检测结果基类"""
    confidence: float
    source: str  # "rule" | "llm" | "hybrid"
    reason: str
    
    def clamp_confidence(self) -> 'BaseDetectionResult':
        """限制置信度在有效范围内"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        return self
    
    @staticmethod
    def ensure_list(value: Any) -> List[Any]:
        """统一列表类型转换"""
        if isinstance(value, str):
            return [value]
        return value if isinstance(value, list) else []
    
    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """限制数值在有效范围内"""
        return max(min_val, min(max_val, value))
```

#### 3.1.2 增强LLMEnhancedBase基类

**文件**: `iris_memory/processing/llm_enhanced_base.py`

新增模板方法模式的检测流程：

```python
class LLMEnhancedDetector(LLMEnhancedBase, Generic[T]):
    """增强的LLM检测器基类 - 支持泛型结果类型"""
    
    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        mode: DetectionMode = DetectionMode.HYBRID,
        daily_limit: int = 0,
        max_tokens: int = 300,
    ):
        super().__init__(...)
    
    async def detect(self, *args, **kwargs) -> T:
        """模板方法 - 统一检测流程"""
        if self._mode == DetectionMode.RULE:
            return self._rule_detect(*args, **kwargs)
        elif self._mode == DetectionMode.LLM:
            return await self._llm_detect(*args, **kwargs)
        else:
            return await self._hybrid_detect(*args, **kwargs)
    
    async def _llm_detect(self, *args, **kwargs) -> T:
        """统一LLM检测流程"""
        prompt = self._build_prompt(*args, **kwargs)
        result = await self._call_llm(prompt)
        
        if not result.success or not result.parsed_json:
            logger.debug(f"LLM detection failed, falling back to rule")
            return self._rule_detect(*args, **kwargs)
        
        return self._parse_llm_result(result.parsed_json)
    
    @abstractmethod
    def _build_prompt(self, *args, **kwargs) -> str:
        """构建LLM提示词 - 子类实现"""
        pass
    
    @abstractmethod
    def _parse_llm_result(self, data: Dict[str, Any]) -> T:
        """解析LLM结果 - 子类实现"""
        pass
```

### 3.2 CaptureEngine委托方法废弃

将14个委托方法标记为废弃，添加 `DeprecationWarning`：

```python
def _check_duplicate_by_vector(self, ...):
    """已废弃: 请直接使用 self.conflict_resolver.check_duplicate_by_vector"""
    import warnings
    warnings.warn(
        "CaptureEngine._check_duplicate_by_vector is deprecated",
        DeprecationWarning,
        stacklevel=2
    )
    return self.conflict_resolver.check_duplicate_by_vector(...)
```

### 3.3 统一配置注册表

**文件**: `iris_memory/core/config_registry.py`

```python
@dataclass
class ConfigDefinition:
    """配置项定义"""
    key: str
    section: str
    attr: str
    default: Any
    description: str = ""
    value_type: type = Any

CONFIG_REGISTRY: Dict[str, ConfigDefinition] = {
    "basic.enable_memory": ConfigDefinition(
        key="basic.enable_memory",
        section="memory",
        attr="auto_capture",
        default=True,
        description="是否启用记忆功能",
    ),
    # ... 所有配置项统一定义
}
```

### 3.4 依赖注入容器

**文件**: `iris_memory/core/service_container.py`

```python
class ServiceContainer:
    """轻量级依赖注入容器"""
    
    _instance: Optional['ServiceContainer'] = None
    _lock = threading.Lock()
    
    def register(self, name: str, instance: Any) -> None:
        """注册服务实例"""
        self._services[name] = instance
    
    def get(self, name: str) -> Any:
        """获取服务"""
        return self._services.get(name)
    
    def clear(self) -> None:
        """清除所有服务（热更新时调用）"""
        self._services.clear()
```

---

## 4. 实施进度

### 阶段一：高优先级（已完成）

| 任务 | 状态 | 文件 |
|------|------|------|
| 创建泛型结果基类 | ✅ 完成 | `processing/detection_result.py` |
| 增强LLMEnhancedBase | ✅ 完成 | `processing/llm_enhanced_base.py` |
| 重构LLMTriggerDetector | ✅ 完成 | `capture/llm_trigger_detector.py` |
| 重构LLMSensitivityDetector | ✅ 完成 | `capture/llm_sensitivity_detector.py` |
| 重构LLMEmotionAnalyzer | ✅ 完成 | `analysis/llm_emotion_analyzer.py` |
| 重构LLMProactiveReplyDetector | ✅ 完成 | `proactive/llm_proactive_reply_detector.py` |
| 重构LLMConflictResolver | ✅ 完成 | `capture/llm_conflict_resolver.py` |
| 重构LLMRetrievalRouter | ✅ 完成 | `retrieval/llm_retrieval_router.py` |
| 废弃CaptureEngine委托方法 | ✅ 完成 | `capture/capture_engine.py` |

### 阶段二：中优先级（已完成）

| 任务 | 状态 | 文件 |
|------|------|------|
| 创建统一配置注册表 | ✅ 完成 | `core/config_registry.py` |
| 创建依赖注入容器 | ✅ 完成 | `core/service_container.py` |
| 更新__init__.py导出 | ✅ 完成 | `core/__init__.py`, `processing/__init__.py` |
| ConfigManager使用注册表 | ✅ 完成 | `core/config_manager.py` |

### 阶段三：验证（已完成）

| 任务 | 状态 |
|------|------|
| 运行单元测试 | ✅ 完成 (759 passed) |
| 模块导入验证 | ✅ 完成 |

---

## 5. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 重构引入新Bug | 中 | 高 | 增加单元测试覆盖率，渐进式重构 |
| 接口变更导致兼容性问题 | 高 | 中 | 保留废弃方法，添加迁移指南 |
| 性能回归 | 低 | 中 | 性能基准测试，关键路径优化 |
| 热更新场景数据丢失 | 低 | 高 | 实现检查点机制，增加恢复逻辑 |

---

## 附录

### A. 迁移指南

#### A.1 LLM检测器迁移

**旧代码**:
```python
class LLMTriggerDetector(LLMEnhancedBase):
    async def detect(self, text, context):
        if self._mode == DetectionMode.RULE:
            return self._rule_detect(text, context)
        # ... 重复的检测逻辑
```

**新代码**:
```python
class LLMTriggerDetector(LLMEnhancedDetector[TriggerDetectionResult]):
    def _build_prompt(self, text, **kwargs) -> str:
        return TRIGGER_DETECTION_PROMPT.format(text=text[:500])
    
    def _parse_llm_result(self, data: Dict) -> TriggerDetectionResult:
        return TriggerDetectionResult(
            should_remember=data.get("should_remember", False),
            confidence=self.clamp(data.get("confidence", 0.5)),
            ...
        )
```

#### A.2 CaptureEngine迁移

**旧代码**:
```python
duplicate = self._find_duplicate_from_results(memory, similar_memories)
```

**新代码**:
```python
duplicate = self.conflict_resolver.find_duplicate_from_results(memory, similar_memories)
```

### B. 测试清单

- [ ] LLMTriggerDetector 单元测试
- [ ] LLMSensitivityDetector 单元测试
- [ ] LLMEmotionAnalyzer 单元测试
- [ ] LLMProactiveReplyDetector 单元测试
- [ ] LLMConflictResolver 单元测试
- [ ] LLMRetrievalRouter 单元测试
- [ ] 集成测试：记忆捕获流程
- [ ] 集成测试：记忆检索流程
- [ ] 热更新场景测试

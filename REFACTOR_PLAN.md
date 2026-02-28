# Iris Memory Plugin 重构规划

> 诊断日期: 2026-02-28
> 当前版本: 1.9.1

## 一、诊断发现

### 1. 架构现状评估

**✅ 已有的良好实践：**

| 实践 | 状态 | 说明 |
|------|------|------|
| Facade + 组合模式 | ✅ 已实现 | `MemoryService` 作为门面，持有 `BusinessService` + `PersistenceService` |
| Feature Module 模式 | ✅ 已实现 | `StorageModule`, `AnalysisModule` 等 7 个模块 |
| SharedState 集中状态 | ✅ 已实现 | 跨服务共享状态集中管理 |
| Dict Dispatch 模式 | ✅ 已使用 | `handler_map`, `_STRATEGY_METHOD_MAP` 替代 if-else |
| 文件大小控制 | ⚠️ 部分超标 | 大部分 <300 行，少数超标 |

**❌ 未发现的"幻觉问题"：**

| 预期问题 | 实际情况 |
|----------|----------|
| "极其复杂的安全函数" | **不存在**。`sanitize_input()` 仅 10 行简单 HTML 移除，是合理的输入清理 |
| "长 if-else 链" | **不存在**。搜索 `elif.*elif.*elif` 无匹配，已使用 dict dispatch |
| "幻觉安全层" | **不存在**。`SensitivityDetector` 是合法业务需求（检测身份证、银行卡等） |

---

### 2. 实际存在的臃肿代码

#### 2.1 MemoryService（约 800 行）— ✅ 已重构

**重构结果：**

| 文件 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| `memory_service.py` | ~1000 行 | **651 行** | -349 行 (-35%) |
| `initializer.py` | (新建) | **411 行** | 新增 |

**架构改进：**

```
Before:
MemoryService (1000 行)
├── 30+ 向后兼容属性
├── 15+ _init_xxx() 初始化方法
├── 业务委托方法
└── 持久化委托方法

After:
MemoryService (651 行) — 薄 Facade
├── 向后兼容属性
├── initialize() → 委托给 ServiceInitializer
├── _create_services()
└── 业务/持久化委托方法

ServiceInitializer (411 行) — 初始化编排器
├── InitializerDeps (dataclass)
├── InitializerResult (dataclass)
├── initialize_all()
├── _init_phase_core()
├── _init_phase_enhanced()
└── _init_xxx() 方法群
```

**关键改进：**
1. **职责分离**：初始化逻辑完全抽取到 `ServiceInitializer`
2. **依赖注入**：通过 `InitializerDeps` dataclass 显式声明依赖
3. **可测试性**：`ServiceInitializer` 可独立单元测试
4. **向后兼容**：所有公开 API 保持不变，1590 个测试全部通过

#### 2.2 ChromaManager（约 500 行）— ✅ 已重构

**重构结果：**

| 文件 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| `chroma_manager.py` | ~662 行 (Mixin) | **702 行** | +40 行（增加委托方法） |
| `chroma_queries.py` | ~505 行 (Mixin) | **511 行** | 组合模式独立类 |
| `chroma_operations.py` | ~467 行 (Mixin) | **463 行** | 组合模式独立类 |

**架构改进：**

```
Before (Mixin 模式):
ChromaManager(ChromaQueries, ChromaOperations)
├── ChromaQueries 通过 self 隐式访问 manager 方法
├── ChromaOperations 通过 self 隐式访问 manager 方法
└── Mixin 继承导致隐式依赖

After (组合模式):
ChromaManager
├── _queries: ChromaQueries  # 显式组合
├── _operations: ChromaOperations  # 显式组合
├── _ensure_ready()  # 懒加载创建组件
└── 委托方法 (query_memories, add_memory, ...)
    └── 转发到对应组件

ChromaQueries(manager: ChromaManager)
├── 通过 self._manager 显式访问 manager
└── query_memories(), get_all_memories(), count_memories(), ...

ChromaOperations(manager: ChromaManager)
├── 通过 self._manager 显式访问 manager
└── add_memory(), update_memory(), delete_memory(), ...
```

**关键改进：**
1. **依赖透明**：`ChromaQueries` 和 `ChromaOperations` 通过构造函数显式接收 `manager` 引用
2. **模式一致**：与项目其他部分（如 `MemoryService` 持有 `BusinessService` + `PersistenceService`）保持一致
3. **可测试性**：组件可独立 mock 和测试
4. **向后兼容**：所有公开 API 保持不变，1680 个测试全部通过
5. **懒加载支持**：`_ensure_ready()` 支持测试场景下直接设置 `_is_ready` 和 `collection`

#### 2.3 PersistenceService（约 400 行）— KV 加载/保存逻辑冗长

```python
# 18 个 _load_xxx / _save_xxx 方法，每个 10-20 行
async def _load_session_data(...)
async def _load_lifecycle_state(...)
async def _load_batch_queues(...)
# ... 重复模式
```

**问题：** 可以通过配置驱动 + 反射简化。

---

### 3. 代码行数统计（非 web 模块）

| 文件 | 行数 | 状态 |
|------|------|------|
| `services/memory_service.py` | 651 | ✅ 已重构 |
| `services/initializer.py` | 411 | ✅ 新增 |
| `storage/chroma_manager.py` | 702 | ✅ 已重构 |
| `storage/chroma_queries.py` | 511 | ✅ 已重构 |
| `storage/chroma_operations.py` | 463 | ✅ 已重构 |
| `capture/capture_engine.py` | 669 | ⚠️ 超标 |
| `models/user_persona.py` | 1413 | ⚠️ 超标 |
| `retrieval/retrieval_engine.py` | 935 | ⚠️ 超标 |
| `services/business_service.py` | 900 | ⚠️ 超标 |
| `services/persistence_service.py` | 404 | ⚠️ 接近上限 |
| 其他文件 | <300 | ✅ 合规 |

---

## 二、重构规划

### 1. 目录结构调整建议

```
iris_memory/
├── plugin_main.py          # 新增：插件入口（从 main.py 抽取 Star 类）
│   职责：@register 元数据 + 事件装饰器 + 薄委托
│
├── services/
│   ├── memory_service.py   # 精简为 ~150 行 Facade
│   ├── business_service.py # 保持现状
│   ├── persistence_service.py # 精简（KV 操作配置驱动化）
│   ├── shared_state.py
│   ├── initializer.py      # 新增：初始化编排器（从 MemoryService 抽取）
│   └── modules/
│       ├── storage_module.py
│       ├── analysis_module.py
│       ├── ...
│
├── storage/
│   ├── chroma/
│   │   ├── __init__.py
│   │   ├── manager.py      # 重构为组合模式
│   │   ├── operations.py   # 保持
│   │   └── queries.py      # 保持
│   ├── session_manager.py
│   ├── lifecycle_manager.py
│   └── ...
│
├── capture/
│   ├── capture_engine.py   # 拆分为 2 个文件
│   ├── capture_core.py     # 新增：核心捕获逻辑
│   └── capture_quality.py  # 新增：质量评估逻辑
│
├── models/
│   ├── memory.py
│   ├── user_persona.py     # 拆分为 2 个文件
│   ├── persona_change.py   # 新增：PersonaChangeRecord + 审计逻辑
│   └── emotion_state.py
│
├── core/
│   ├── config_manager.py
│   ├── constants.py
│   ├── types.py
│   └── initialization/     # 新增目录
│       ├── __init__.py
│       ├── component_initializer.py  # 组件初始化器
│       └── init_sequence.py          # 初始化序列编排
│
├── commands/
│   ├── handlers.py         # 保持（已使用 dict dispatch）
│   ├── permissions.py
│   └── registry.py
│
└── utils/
    ├── logger.py
    ├── bounded_dict.py
    └── ...
```

### 2. 新文件职责说明

| 新文件 | 职责 | 来源 |
|--------|------|------|
| `plugin_main.py` | AstrBot 插件入口，`@register` + 事件装饰器 | 从 `main.py` 抽取 Star 类 |
| `services/initializer.py` | 初始化编排器，管理所有 `_init_xxx()` 方法 | 从 `MemoryService` 抽取 |
| `core/initialization/component_initializer.py` | 单个组件的初始化逻辑 | 从 `initializer.py` 细分 |
| `capture/capture_core.py` | 核心捕获流程（触发检测 → 敏感度 → 情感） | 从 `capture_engine.py` 拆分 |
| `capture/capture_quality.py` | 质量评估、RIF 评分、存储层确定 | 从 `capture_engine.py` 拆分 |
| `models/persona_change.py` | `PersonaChangeRecord` + 变更审计逻辑 | 从 `user_persona.py` 拆分 |
| `storage/chroma/manager.py` | ChromaDB 管理器（组合模式重构） | 重构 `chroma_manager.py` |

### 3. 核心重构任务

#### 任务 1：MemoryService 瘦身（P0）— ✅ 已完成

**Before:**
```python
class MemoryService:
    def __init__(self, ...):
        # 30+ 属性声明
        # 7 个 Module
        # SharedState
        # ...
    
    @property
    def chroma_manager(self): return self.storage.chroma_manager
    @property
    def capture_engine(self): return self.capture.capture_engine
    # ... 30 个向后兼容属性
    
    async def initialize(self):
        # 15+ 个 _init_xxx() 调用
    
    async def _init_core_components(self): ...
    async def _init_knowledge_graph(self): ...
    async def _init_llm_enhanced(self): ...
    # ... 15 个初始化方法
```

**After:**
```python
class MemoryService:
    """Facade - 仅持有 Module 引用 + 委托"""
    
    def __init__(self, context, config, plugin_data_path):
        self._deps = InitializerDeps(...)
        self._initializer = ServiceInitializer(self._deps)
        self._business: Optional[BusinessService] = None
        self._persistence: Optional[PersistenceService] = None
    
    async def initialize(self) -> None:
        result = await self._initializer.initialize_all()
        self._module_init_status = result.module_init_status
        self._create_services()
    
    # 业务方法直接委托
    async def capture_and_store_memory(self, ...):
        return await self._business.capture_and_store_memory(...)
```

**新增文件 `services/initializer.py`：**
```python
@dataclass
class InitializerDeps:
    """初始化器依赖项"""
    context: Context
    config: AstrBotConfig
    plugin_data_path: Path
    cfg: ConfigManager
    storage: StorageModule = field(default_factory=StorageModule)
    # ... 其他模块

@dataclass
class InitializerResult:
    """初始化结果"""
    modules: InitializerDeps
    module_init_status: Dict[str, bool]
    # ...

class ServiceInitializer:
    """服务初始化编排器"""
    
    async def initialize_all(self) -> InitializerResult:
        await self._init_phase_core()
        await self._init_phase_enhanced()
        await self._init_phase_finalize()
        return InitializerResult(...)
```

#### 任务 2：ChromaManager 组合模式重构（P1）— ✅ 已完成

**Before (Mixin):**
```python
class ChromaManager(ChromaQueries, ChromaOperations):
    def __init__(self, config, data_path, plugin_context):
        # Mixin 类通过 self 隐式共享状态
```

**After (组合):**
```python
class ChromaManager:
    def __init__(self, config, data_path, plugin_context):
        self._queries: Optional[ChromaQueries] = None
        self._operations: Optional[ChromaOperations] = None
    
    def _ensure_ready(self) -> None:
        """懒加载创建组件，支持测试场景"""
        if not self._is_ready or self.collection is None:
            raise RuntimeError("ChromaManager is not initialized...")
        if self._queries is None:
            self._queries = ChromaQueries(self)
        if self._operations is None:
            self._operations = ChromaOperations(self)
    
    async def query_memories(self, *args, **kwargs):
        self._ensure_ready()
        return await self._queries.query_memories(*args, **kwargs)
    
    async def add_memory(self, *args, **kwargs):
        self._ensure_ready()
        return await self._operations.add_memory(*args, **kwargs)
```

**验证结果：**
- 所有 1680 个测试通过
- 模块状态一致性验证通过
- 向后兼容验证成功

#### 任务 3：PersistenceService 配置驱动化（P2）

**Before:**
```python
async def load_from_kv(self, get_kv_data):
    await self._load_session_data(get_kv_data)
    await self._load_lifecycle_state(get_kv_data)
    await self._load_batch_queues(get_kv_data)
    # ... 9 个显式调用

async def _load_session_data(self, get_kv_data):
    if not self._storage.session_manager: return
    data = await get_kv_data(KVStoreKeys.SESSIONS, {})
    # ...
```

**After:**
```python
_KV_LOADERS = [
    ("session_manager", KVStoreKeys.SESSIONS, "deserialize_from_kv_storage"),
    ("lifecycle_manager", KVStoreKeys.LIFECYCLE_STATE, "deserialize_state"),
    # ...
]

async def load_from_kv(self, get_kv_data):
    for module_attr, key, method in _KV_LOADERS:
        component = getattr(self._storage, module_attr)
        if component:
            data = await get_kv_data(key, {})
            if data:
                getattr(component, method)(data)
```

---

## 三、需要确认的 AstrBot API

由于官方文档网站无法直接访问具体 API 文档页面，以下 API 需要确认：

| API | 当前使用方式 | 需确认事项 |
|-----|-------------|-----------|
| `@register` 装饰器 | `@register("iris_memory", "Author", "desc", "1.9.1")` | 参数签名是否正确？版本号位置？ |
| `filter.command` | `@filter.command("memory_save")` | 是否支持别名？返回值类型？ |
| `filter.on_llm_request()` | 修改 `req.system_prompt` | `req` 对象的完整属性？是否可修改？ |
| `filter.on_llm_response()` | 读取 `resp.completion_text` | `resp` 对象的完整属性？ |
| `StarTools.get_data_dir()` | 获取插件数据目录 | 返回 `str` 还是 `Path`？ |
| `event.plain_result()` | 返回文本结果 | 是否支持其他类型？ |
| `event.request_llm()` | 触发 LLM 请求 | `prompt` 参数格式？ |
| `self.get_kv_data` / `self.put_kv_data` | KV 存储读写 | 异步还是同步？参数格式？ |
| `AstrMessageEvent.get_sender_id()` | 获取发送者 ID | 是否有 `get_sender_name()`？ |
| `AstrMessageEvent.get_extra()` | 获取额外数据 | 完整 API？ |

**确认方式：**
1. 查看 AstrBot 源码：`astrbot/api/star.py`, `astrbot/api/event.py`
2. 参考社区插件示例
3. 查看 `.venv/lib/python3.12/site-packages/astrbot/` 下的源码

---

## 四、重构优先级

| 优先级 | 任务 | 风险 | 工作量 | 状态 |
|--------|------|------|--------|------|
| P0 | MemoryService 瘦身（抽取 Initializer） | 低 | 中 | ✅ 已完成 |
| P1 | ChromaManager 组合模式重构 | 中 | 中 | ✅ 已完成 |
| P2 | PersistenceService 配置驱动化 | 低 | 小 | ⏳ 待开始 |
| P3 | capture_engine.py 拆分 | 低 | 小 | ⏳ 待开始 |
| P4 | user_persona.py 拆分 | 低 | 小 | ⏳ 待开始 |

---

## 五、向后兼容策略

重构过程中需要保持以下向后兼容：

### 1. MemoryService 属性访问

旧代码可能直接访问 `service.chroma_manager` 等属性，需要保留 `@property` 代理：

```python
@property
def chroma_manager(self):
    """向后兼容：代理到 StorageModule"""
    return self.storage.chroma_manager
```

### 2. 方法签名

所有公开方法签名保持不变，仅重构内部实现。

### 3. 导入路径

保持 `from iris_memory.services.memory_service import MemoryService` 不变。

---

## 六、测试策略

每次重构后运行：

```bash
pytest tests/ -v
```

关键测试覆盖：
- 记忆捕获与存储
- 记忆检索
- 会话管理
- KV 持久化
- 主动回复

---

## 七、变更日志

| 日期 | 变更内容 |
|------|----------|
| 2026-02-28 | 创建重构规划文档 |
| 2026-02-28 | **P0 完成**：MemoryService 瘦身，抽取 ServiceInitializer（651 行 + 411 行） |
| 2026-02-28 | 所有 1590 个测试通过，向后兼容验证成功 |
| 2026-03-01 | **P1 完成**：ChromaManager 从 Mixin 模式重构为组合模式（702 + 511 + 463 行） |
| 2026-03-01 | 所有 1680 个测试通过，模块状态一致性验证通过 |
| 2026-03-01 | 创建 `1.10-dev` 分支，提交重构变更 |

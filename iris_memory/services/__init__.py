"""
服务层模块 - 封装核心业务逻辑

架构：
- MemoryService: Facade（门面），协调初始化并委托操作
- BusinessService: 业务操作（捕获、检索、LLM 上下文等）
- PersistenceService: 持久化操作（KV 加载/保存、服务销毁）
- SharedState: 跨服务共享状态（用户画像、情感状态等）
"""
from iris_memory.services.memory_service import MemoryService
from iris_memory.services.business_service import BusinessService
from iris_memory.services.persistence_service import PersistenceService
from iris_memory.services.shared_state import SharedState

__all__ = ["MemoryService", "BusinessService", "PersistenceService", "SharedState"]

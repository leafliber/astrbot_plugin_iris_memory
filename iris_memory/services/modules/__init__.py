"""
Feature Module 聚合层

将 MemoryService 的 20+ 个组件引用拆分为 6 个 Feature Module，
每个 Module 封装一组高内聚的组件，对外暴露简洁的接口。
"""
from iris_memory.services.modules.storage_module import StorageModule
from iris_memory.services.modules.analysis_module import AnalysisModule
from iris_memory.services.modules.llm_enhanced_module import LLMEnhancedModule
from iris_memory.services.modules.capture_module import CaptureModule
from iris_memory.services.modules.retrieval_module import RetrievalModule
from iris_memory.services.modules.proactive_module import ProactiveModule

__all__ = [
    "StorageModule",
    "AnalysisModule",
    "LLMEnhancedModule",
    "CaptureModule",
    "RetrievalModule",
    "ProactiveModule",
]

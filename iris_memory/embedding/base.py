"""
嵌入提供者基类 - 定义统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EmbeddingRequest:
    """嵌入请求"""
    text: str
    model: Optional[str] = None
    dimension: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    """嵌入响应"""
    embedding: np.ndarray
    model: str
    dimension: int
    token_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_list(self) -> List[float]:
        """转换为列表格式"""
        return self.embedding.tolist()


class EmbeddingProvider(ABC):
    """嵌入提供者抽象基类
    
    所有嵌入提供者必须实现此接口，确保策略模式的统一性。
    """

    def __init__(self, config: Any):
        """初始化提供者
        
        Args:
            config: 插件配置对象
        """
        self.config = config
        self._dimension = None
        self._model = None

    @property
    def is_ready(self) -> bool:
        """提供者是否已就绪（模型已加载完成）
        
        默认实现：初始化完成（_dimension 已设置）即为就绪。
        延迟加载的提供者应覆盖此属性。
        """
        return self._dimension is not None

    @property
    def dimension(self) -> int:
        """获取嵌入维度"""
        if self._dimension is None:
            # 延迟加载期间返回配置的默认维度
            from iris_memory.core.config_manager import get_config_manager
            return get_config_manager().embedding_dimension
        return self._dimension

    @property
    def model(self) -> str:
        """获取模型名称"""
        if self._model is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        return self._model

    @abstractmethod
    async def initialize(self) -> bool:
        """异步初始化提供者
        
        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量
        
        Args:
            request: 嵌入请求对象
            
        Returns:
            EmbeddingResponse: 嵌入响应对象
        """
        pass

    @abstractmethod
    async def embed_batch(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """批量生成嵌入向量
        
        Args:
            requests: 嵌入请求列表
            
        Returns:
            List[EmbeddingResponse]: 嵌入响应列表
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            "status": "ok",
            "model": self._model,
            "dimension": self._dimension
        }

    async def ensure_dimension(self, target_dimension: int) -> bool:
        """确保嵌入维度匹配目标维度
        
        Args:
            target_dimension: 目标维度
            
        Returns:
            bool: 是否匹配
        """
        return self.dimension == target_dimension

    async def adapt_dimension(self, embedding: np.ndarray, target_dimension: int) -> np.ndarray:
        """适配嵌入维度（扩展或截断）
        
        Args:
            embedding: 原始嵌入向量
            target_dimension: 目标维度
            
        Returns:
            np.ndarray: 适配后的嵌入向量
        """
        current_dim = len(embedding)
        
        if current_dim == target_dimension:
            return embedding
        
        if current_dim < target_dimension:
            # 扩展：用零填充
            padding = np.zeros(target_dimension - current_dim, dtype=embedding.dtype)
            return np.concatenate([embedding, padding])
        else:
            # 截断：保留前N维
            return embedding[:target_dimension]

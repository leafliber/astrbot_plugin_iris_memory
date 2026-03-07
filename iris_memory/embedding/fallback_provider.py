"""
降级嵌入提供者 - 伪随机向量
最后的保底选项，确保系统始终可用
"""

from typing import List, Dict, Any, Optional
import numpy as np
import hashlib

from .base import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from iris_memory.utils.logger import get_logger

# 模块logger
logger = get_logger("fallback_provider")


class FallbackProvider(EmbeddingProvider):
    """降级嵌入提供者
    
    使用伪随机向量作为嵌入，确保系统始终可用。
    注意：这是降级选项，不建议长期使用。
    """

    def __init__(self, config: Any):
        """初始化降级提供者
        
        Args:
            config: 插件配置对象
        """
        super().__init__(config)
        self._model = "fallback/pseudo-random"

    async def initialize(self) -> bool:
        """初始化降级提供者

        Returns:
            bool: 是否初始化成功
        """
        try:
            # 获取配置的维度
            from iris_memory.config import get_store
            self._dimension = get_store().get("embedding.local_dimension", 512)
            logger.debug("Initialized fallback embedding provider (backup only). Use pseudo-random vectors as a last resort.")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize fallback provider: {e}")
            return False

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量（伪随机）
        
        使用文本哈希作为种子的确定性伪随机向量，相同文本始终生成相同向量。
        所有维度都携带信息，而非仅前几维有值。
        
        Args:
            request: 嵌入请求对象
            
        Returns:
            EmbeddingResponse: 嵌入响应对象
        """
        dimension = request.dimension or self._dimension
        
        # 使用 SHA-256 哈希作为 PRNG 种子，确保确定性
        hash_bytes = hashlib.sha256(request.text.encode('utf-8')).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')
        rng = np.random.RandomState(seed)
        
        # 生成所有维度的伪随机向量
        embedding_array = rng.randn(dimension).astype(np.float32)
        
        # L2 归一化
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        
        return EmbeddingResponse(
            embedding=embedding_array,
            model=self._model,
            dimension=dimension,
            metadata={
                "warning": "Using fallback provider (backup only). This should not be the primary embedding source.",
                "request_metadata": request.metadata
            }
        )

    async def embed_batch(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """批量生成嵌入向量
        
        Args:
            requests: 嵌入请求列表
            
        Returns:
            List[EmbeddingResponse]: 嵌入响应列表
        """
        responses = []
        for request in requests:
            response = await self.embed(request)
            responses.append(response)
        return responses

    async def health_check(self) -> Dict[str, Any]:
        """健康检查

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            "provider": "fallback",
            "status": "ok",
            "model": self._model,
            "dimension": self._dimension,
            "available": True,
            "warning": "This is a fallback provider (backup only). All primary embedding providers are unavailable."
        }

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
            from iris_memory.core.config_manager import get_config_manager
            self._dimension = get_config_manager().embedding_local_dimension
            logger.debug("Initialized fallback embedding provider (backup only). Use pseudo-random vectors as a last resort.")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize fallback provider: {e}")
            return False

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量（伪随机）
        
        Args:
            request: 嵌入请求对象
            
        Returns:
            EmbeddingResponse: 嵌入响应对象
        """
        dimension = request.dimension or self._dimension
        
        # 使用哈希生成确定性的伪随机向量
        # 相同的文本总是生成相同的向量
        hash_obj = hashlib.md5(request.text.encode())
        hash_bytes = hash_obj.digest()
        
        # 将哈希转换为浮点数向量
        embedding = []
        for i in range(0, min(len(hash_bytes), dimension // 4)):
            byte_val = hash_bytes[i]
            embedding.append(byte_val / 255.0)
        
        # 填充到指定维度
        while len(embedding) < dimension:
            embedding.append(0.0)
        
        # 归一化向量（L2标准化）
        embedding_array = np.array(embedding, dtype=np.float32)
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

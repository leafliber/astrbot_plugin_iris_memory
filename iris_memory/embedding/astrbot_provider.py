"""
AstrBot 嵌入提供者 - 使用 AstrBot Embedding API
优先级最高的嵌入源
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from iris_memory.utils.logger import get_logger

# 模块logger
logger = get_logger("astrbot_provider")


class AstrBotProvider(EmbeddingProvider):
    """AstrBot Embedding API 嵌入提供者
    
    使用 AstrBot 内置的 Embedding 接口生成嵌入向量。
    这是最高优先级的嵌入源，优先使用。
    """

    def __init__(self, config: Any, astrbot_context: Any = None):
        """初始化 AstrBot 提供者
        
        Args:
            config: 插件配置对象
            astrbot_context: AstrBot上下文对象（可选，也可以后续通过set_context设置）
        """
        super().__init__(config)
        self.astrbot_context = astrbot_context
        self.embedding_provider = None
        self._dimension = 512  # 默认维度，会在初始化时更新
        self._model = "astrbot-embedding"

    def set_context(self, context: Any):
        """设置AstrBot上下文
        
        Args:
            context: AstrBot上下文对象
        """
        self.astrbot_context = context

    async def initialize(self) -> bool:
        """初始化 AstrBot 提供者

        Returns:
            bool: 是否初始化成功
        """
        if not self.astrbot_context:
            logger.debug("AstrBot context not available")
            return False
            
        try:
            # 获取嵌入提供商
            if hasattr(self.astrbot_context, 'get_all_embedding_providers'):
                providers = self.astrbot_context.get_all_embedding_providers()
                if providers:
                    self.embedding_provider = providers[0]
                    # 尝试获取维度信息
                    if hasattr(self.embedding_provider, 'dimension'):
                        self._dimension = self.embedding_provider.dimension
                    if hasattr(self.embedding_provider, 'model_name'):
                        self._model = self.embedding_provider.model_name
                    elif hasattr(self.embedding_provider, 'model'):
                        self._model = self.embedding_provider.model
                    logger.info(f"AstrBot embedding provider initialized: {self._model}, dimension={self._dimension}")
                    return True
                else:
                    logger.debug("No embedding providers available from AstrBot")
                    return False
            else:
                logger.debug("AstrBot context does not have get_all_embedding_providers method")
                return False
        except Exception as e:
            logger.warning(f"Failed to initialize AstrBot embedding provider: {e}")
            return False

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量

        Args:
            request: 嵌入请求对象

        Returns:
            EmbeddingResponse: 嵌入响应对象
        """
        if not self.embedding_provider:
            raise RuntimeError("AstrBot embedding provider not initialized. Call initialize() first.")

        try:
            # 调用 AstrBot 的嵌入接口
            # 根据 AstrBot EmbeddingProvider 的接口调用
            if hasattr(self.embedding_provider, 'embed'):
                # 新接口
                result = await self.embedding_provider.embed(request.text)
            elif hasattr(self.embedding_provider, 'get_embedding'):
                # 备选接口
                result = await self.embedding_provider.get_embedding(request.text)
            elif hasattr(self.embedding_provider, 'encode'):
                # 另一种可能的接口
                result = await self.embedding_provider.encode(request.text)
            else:
                raise RuntimeError("No suitable embedding method found in provider")
            
            # 处理结果
            if isinstance(result, np.ndarray):
                embedding = result
            elif isinstance(result, list):
                embedding = np.array(result, dtype=np.float32)
            elif hasattr(result, 'embedding'):
                embedding = np.array(result.embedding, dtype=np.float32)
            else:
                embedding = np.array(result, dtype=np.float32)
            
            return EmbeddingResponse(
                embedding=embedding,
                model=self._model,
                dimension=len(embedding),
                token_count=len(request.text) // 4,
                metadata={"provider": "astrbot"}
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

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
        status = {
            "provider": "astrbot",
            "status": "ok" if self.embedding_provider else "not_initialized",
            "model": self._model,
            "dimension": self._dimension,
            "available": self.embedding_provider is not None
        }
        
        # 测试调用
        if self.embedding_provider:
            try:
                test_result = await self.embed(EmbeddingRequest(text="test"))
                status["status"] = "ok"
                status["actual_dimension"] = test_result.dimension
            except Exception as e:
                status["status"] = "error"
                status["error"] = str(e)
        
        return status

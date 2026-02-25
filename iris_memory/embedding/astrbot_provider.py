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
    支持通过 provider_id 选择特定的 embedding provider。
    """

    def __init__(self, config: Any, astrbot_context: Any = None, provider_id: str = ""):
        """初始化 AstrBot 提供者
        
        Args:
            config: 插件配置对象
            astrbot_context: AstrBot上下文对象（可选）
            provider_id: 指定的 embedding provider ID，空字符串表示使用第一个可用的
        """
        super().__init__(config)
        self.astrbot_context = astrbot_context
        self.embedding_provider = None
        self.provider_id = provider_id
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

        支持通过 provider_id 选择特定的 embedding provider。
        如果 provider_id 为空，使用第一个可用的 provider。

        Returns:
            bool: 是否初始化成功
        """
        if not self.astrbot_context:
            logger.debug("AstrBot context not available")
            return False
            
        try:
            # 获取嵌入提供商
            if not hasattr(self.astrbot_context, 'get_all_embedding_providers'):
                logger.debug("AstrBot context does not have get_all_embedding_providers method")
                return False
            
            providers = self.astrbot_context.get_all_embedding_providers()
            if not providers:
                logger.debug("No embedding providers available from AstrBot")
                return False
            
            # 根据 provider_id 选择
            selected_provider = self._select_provider(providers)
            if selected_provider is None:
                return False
            
            self.embedding_provider = selected_provider
            
            # 获取维度和模型信息
            if hasattr(self.embedding_provider, 'dimension'):
                self._dimension = self.embedding_provider.dimension
            if hasattr(self.embedding_provider, 'model_name'):
                self._model = self.embedding_provider.model_name
            elif hasattr(self.embedding_provider, 'model'):
                self._model = self.embedding_provider.model
            
            # 通过实际嵌入调用检测真实维度
            actual_dimension = await self._detect_actual_dimension()
            if actual_dimension and actual_dimension != self._dimension:
                logger.debug(
                    f"Detected actual embedding dimension: {actual_dimension} "
                    f"(was {self._dimension})"
                )
                self._dimension = actual_dimension
            
            logger.debug(
                f"AstrBot embedding provider initialized: {self._model}, "
                f"dimension={self._dimension}"
                f"{f', provider_id={self.provider_id}' if self.provider_id else ''}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize AstrBot embedding provider: {e}")
            return False
    
    def _select_provider(self, providers: list) -> Any:
        """根据 provider_id 选择提供者
        
        Args:
            providers: 可用的 embedding providers 列表
            
        Returns:
            选中的 provider，或 None
        """
        if not self.provider_id:
            # 未指定 provider_id，使用第一个
            logger.debug(f"No provider_id specified, using first available ({len(providers)} total)")
            return providers[0]
        
        # 按 provider_id 匹配
        from iris_memory.core.provider_utils import extract_provider_id
        
        for provider in providers:
            pid = extract_provider_id(provider)
            if pid and pid == self.provider_id:
                logger.debug(f"Found matching embedding provider: {self.provider_id}")
                return provider
            if pid and pid.lower() == self.provider_id.lower():
                logger.debug(f"Found matching embedding provider (case-insensitive): {self.provider_id}")
                return provider
        
        # 未找到匹配的 provider
        available_ids = []
        for p in providers:
            pid = extract_provider_id(p)
            if pid:
                available_ids.append(pid)
        
        logger.warning(
            f"Embedding provider '{self.provider_id}' not found. "
            f"Available providers: {available_ids or ['(unable to extract IDs)']}. "
            f"Falling back to first available."
        )
        return providers[0]
    
    async def _detect_actual_dimension(self) -> Optional[int]:
        """通过实际嵌入调用检测真实的嵌入维度
        
        Returns:
            Optional[int]: 检测到的维度，失败返回 None
        """
        try:
            test_embedding = await self.embed(EmbeddingRequest(text="__dimension_test__"))
            return test_embedding.dimension
        except Exception as e:
            logger.debug(f"Failed to detect actual dimension via test embed: {e}")
            return None

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

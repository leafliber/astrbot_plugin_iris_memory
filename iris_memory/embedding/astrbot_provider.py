"""
AstrBot 嵌入提供者 - 使用 AstrBot LLM API
优先级最高的嵌入源
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse


class AstrBotProvider(EmbeddingProvider):
    """AstrBot LLM API 嵌入提供者
    
    使用 AstrBot 内置的 LLM 接口生成嵌入向量。
    这是最高优先级的嵌入源，优先使用。
    """

    def __init__(self, config: Any):
        """初始化 AstrBot 提供者
        
        Args:
            config: 插件配置对象
        """
        super().__init__(config)
        self.astrbot_context = None
        self.llm_api = None

    async def initialize(self) -> bool:
        """初始化 AstrBot 提供者
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            # 尝试获取 AstrBot 上下文
            # 从插件初始化时传入的 context 中获取 LLM API
            from astrbot.api import AstrBotApi
            
            # 延迟注入：在 ChromaManager 中会设置 context
            if hasattr(self.config, '_plugin_context'):
                self.astrbot_context = self.config._plugin_context
            else:
                # 尝试从配置中获取
                self.astrbot_context = self._get_config('_plugin_context', None)
            
            if not self.astrbot_context:
                return False
            
            # 获取 LLM API
            self.llm_api = AstrBotApi(self.astrbot_context)
            
            # 获取配置的模型名称
            self._model = self._get_config(
                "chroma_config.astrbot_model",
                self._get_config("chroma_config.embedding_model", "openai/text-embedding-ada-002")
            )
            
            # 默认维度（根据模型确定）
            model_dimension_map = {
                "openai/text-embedding-ada-002": 1536,
                "openai/text-embedding-3-small": 1536,
                "openai/text-embedding-3-large": 3072,
                "BAAI/bge-m3": 1024,
                "BAAI/bge-small": 512,
                "BAAI/bge-large": 1024,
            }
            self._dimension = model_dimension_map.get(
                self._model,
                self._get_config("chroma_config.embedding_dimension", 1536)
            )
            
            return True
            
        except Exception as e:
            from iris_memory.utils.logger import logger
            logger.warning(f"Failed to initialize AstrBot provider: {e}")
            return False

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量
        
        Args:
            request: 嵌入请求对象
            
        Returns:
            EmbeddingResponse: 嵌入响应对象
        """
        if not self.llm_api:
            raise RuntimeError("AstrBot provider not initialized. Call initialize() first.")
        
        try:
            # 调用 AstrBot LLM API 生成嵌入
            # 注意：需要根据实际 AstrBot API 调整
            # 这里假设有一个 get_embedding 方法
            
            # 尝试通过 LLM API 调用嵌入服务
            result = await self.llm_api.get_embedding(
                text=request.text,
                model=request.model or self._model
            )
            
            embedding = np.array(result["embedding"], dtype=np.float32)
            
            # 维度适配（如果需要）
            if request.dimension and len(embedding) != request.dimension:
                embedding = await self.adapt_dimension(embedding, request.dimension)
                self._dimension = request.dimension
            
            return EmbeddingResponse(
                embedding=embedding,
                model=self._model,
                dimension=len(embedding),
                token_count=result.get("token_count"),
                metadata=request.metadata
            )
            
        except Exception as e:
            from iris_memory.utils.logger import logger
            logger.error(f"Failed to generate embedding with AstrBot: {e}")
            raise

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
            "status": "ok" if self.llm_api else "not_initialized",
            "model": self._model,
            "dimension": self._dimension,
            "available": self.llm_api is not None
        }
        
        # 测试调用
        if self.llm_api:
            try:
                test_result = await self.embed(EmbeddingRequest(text="test"))
                status["status"] = "ok"
            except Exception as e:
                status["status"] = "error"
                status["error"] = str(e)
        
        return status

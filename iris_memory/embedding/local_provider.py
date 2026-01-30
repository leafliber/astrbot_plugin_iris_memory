"""
本地嵌入提供者 - 使用 sentence-transformers
作为降级选项（优先级第二）
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse


class LocalProvider(EmbeddingProvider):
    """本地嵌入提供者
    
    使用 sentence-transformers 在本地生成嵌入向量。
    作为降级选项，在 AstrBot API 不可用时使用。
    """

    def __init__(self, config: Any):
        """初始化本地提供者

        Args:
            config: 插件配置对象
        """
        super().__init__(config)
        self._model = None
        self.model_path = None
        self.device = "cpu"

    async def initialize(self) -> bool:
        """初始化本地提供者
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            # 检查依赖
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                from iris_memory.utils.logger import logger
                logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
                return False
            
            # 获取配置
            model_name = self._get_config(
                "chroma_config.embedding_model",
                "BAAI/bge-m3"
            )
            
            # 设备选择
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 加载模型
            from iris_memory.utils.logger import logger
            logger.info(f"Loading local embedding model: {model_name} on {self.device}")
            
            self.model = SentenceTransformer(model_name, device=self.device)
            self._model = model_name
            self._dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Local embedding provider initialized: {self._model} (dim={self._dimension})")
            return True
            
        except Exception as e:
            from iris_memory.utils.logger import logger
            logger.warning(f"Failed to initialize local provider: {e}")
            return False

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量
        
        Args:
            request: 嵌入请求对象
            
        Returns:
            EmbeddingResponse: 嵌入响应对象
        """
        if not self.model:
            raise RuntimeError("Local provider not initialized. Call initialize() first.")
        
        try:
            # 生成嵌入
            embedding = self.model.encode(
                request.text,
                convert_to_numpy=True,
                normalize_embeddings=True  # 标准化向量
            )
            
            # 维度适配（如果需要）
            if request.dimension and len(embedding) != request.dimension:
                embedding = await self.adapt_dimension(embedding, request.dimension)
                self._dimension = request.dimension
            
            return EmbeddingResponse(
                embedding=embedding,
                model=self._model,
                dimension=len(embedding),
                metadata=request.metadata
            )
            
        except Exception as e:
            from iris_memory.utils.logger import logger
            logger.error(f"Failed to generate embedding with local model: {e}")
            raise

    async def embed_batch(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """批量生成嵌入向量
        
        Args:
            requests: 嵌入请求列表
            
        Returns:
            List[EmbeddingResponse]: 嵌入响应列表
        """
        if not self.model:
            raise RuntimeError("Local provider not initialized. Call initialize() first.")
        
        try:
            texts = [req.text for req in requests]
            
            # 批量生成嵌入（更高效）
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            responses = []
            for i, request in enumerate(requests):
                embedding = embeddings[i]
                
                # 维度适配（如果需要）
                if request.dimension and len(embedding) != request.dimension:
                    embedding = await self.adapt_dimension(embedding, request.dimension)
                
                responses.append(EmbeddingResponse(
                    embedding=embedding,
                    model=self._model,
                    dimension=len(embedding),
                    metadata=request.metadata
                ))
            
            return responses
            
        except Exception as e:
            from iris_memory.utils.logger import logger
            logger.error(f"Failed to batch generate embeddings: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        status = {
            "provider": "local",
            "status": "ok" if self.model else "not_initialized",
            "model": self._model,
            "dimension": self._dimension,
            "device": self.device,
            "available": self.model is not None
        }
        
        # 测试调用
        if self.model:
            try:
                test_result = await self.embed(EmbeddingRequest(text="test"))
                status["status"] = "ok"
            except Exception as e:
                status["status"] = "error"
                status["error"] = str(e)
        
        return status

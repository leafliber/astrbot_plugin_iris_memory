"""
本地嵌入提供者 - 使用 sentence-transformers
作为降级选项（优先级第二）

支持后台异步加载模型，避免阻塞插件启动。
"""

import asyncio
import os
import threading
from typing import List, Dict, Any, Optional

import numpy as np

from .base import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from iris_memory.utils.logger import get_logger

# 模块logger
logger = get_logger("local_provider")


class LocalProvider(EmbeddingProvider):
    """本地嵌入提供者
    
    使用 sentence-transformers 在本地生成嵌入向量。
    作为降级选项，在 AstrBot API 不可用时使用。
    
    模型在后台线程中加载，不阻塞插件启动。通过 is_ready 属性
    查询模型是否加载完成。
    """

    def __init__(self, config: Any):
        """初始化本地提供者

        Args:
            config: 插件配置对象
        """
        super().__init__(config)
        self._model = None
        self._model_instance = None
        self.model_path = None
        self.device = "cpu"
        
        # 后台加载相关
        self._load_thread: Optional[threading.Thread] = None
        self._load_complete = threading.Event()
        self._load_error: Optional[Exception] = None

    @property
    def is_ready(self) -> bool:
        """模型是否已加载完成且可用"""
        return self._load_complete.is_set() and self._load_error is None

    async def initialize(self) -> bool:
        """初始化本地提供者

        仅检查依赖可用性并启动后台模型加载，不阻塞启动。

        Returns:
            bool: 依赖检查是否通过（True 表示后台加载已启动）
        """
        try:
            from iris_memory.core.config_manager import get_config_manager
            cfg = get_config_manager()

            # 检查依赖
            try:
                # 在导入前设置环境变量以抑制 transformers 输出
                os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                # 禁用 transformers 进度条
                from transformers.utils import logging as transformers_logging
                transformers_logging.disable_progress_bar()
                from sentence_transformers import SentenceTransformer  # noqa: F401
            except ImportError:
                logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
                return False
            
            # 检查 torch
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                logger.warning("torch not installed. Run: pip install torch")
                return False
            
            # 获取配置（修复：处理可能返回列表的情况）
            model_name = cfg.embedding_local_model
            # 如果是列表，取第一个元素
            if isinstance(model_name, list):
                model_name = model_name[0] if model_name else "BAAI/bge-small-zh-v1.5"
            
            self._model = model_name
            # 设置配置中的默认维度，模型加载完成后会更新为实际维度
            self._dimension = cfg.embedding_local_dimension
            
            # 启动后台加载任务
            self._start_background_load(model_name)
            logger.debug(f"LocalProvider initialized, background model loading started: {model_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize local provider: {e}")
            return False

    def _start_background_load(self, model_name: str):
        """在后台线程中加载模型
        
        Args:
            model_name: 模型名称或路径
        """
        def _load_model():
            try:
                from sentence_transformers import SentenceTransformer
                logger.debug(f"[Background] Loading local embedding model: {model_name} on {self.device}")
                self._model_instance = SentenceTransformer(model_name, device=self.device)
                actual_dim = self._model_instance.get_sentence_embedding_dimension()
                self._dimension = actual_dim
                logger.debug(
                    f"[Background] Local embedding model loaded successfully: "
                    f"{model_name} (dim={actual_dim}, device={self.device})"
                )
            except Exception as e:
                self._load_error = e
                logger.error(f"[Background] Failed to load local embedding model: {e}")
            finally:
                self._load_complete.set()
        
        self._load_thread = threading.Thread(
            target=_load_model, daemon=True, name="iris-local-embed-loader"
        )
        self._load_thread.start()

    def _wait_for_model(self, timeout: float = 120) -> None:
        """等待模型加载完成
        
        Args:
            timeout: 最大等待时间（秒）
            
        Raises:
            RuntimeError: 加载超时或加载失败
        """
        if self._load_complete.is_set():
            if self._load_error:
                raise RuntimeError(f"Local embedding model failed to load: {self._load_error}")
            return
        
        logger.debug("Waiting for local embedding model to finish loading...")
        if not self._load_complete.wait(timeout=timeout):
            raise RuntimeError(
                f"Local embedding model loading timed out after {timeout}s"
            )
        
        if self._load_error:
            raise RuntimeError(f"Local embedding model failed to load: {self._load_error}")

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量
        
        如果模型尚未加载完成，会在线程池中等待（不阻塞事件循环）。
        
        Args:
            request: 嵌入请求对象
            
        Returns:
            EmbeddingResponse: 嵌入响应对象
        """
        # 在线程池中等待模型加载完成，避免阻塞事件循环
        await asyncio.to_thread(self._wait_for_model)
        
        try:
            # CPU 密集型编码放到线程池执行
            embedding = await asyncio.to_thread(
                self._model_instance.encode,
                request.text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # 标准化向量
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
            logger.error(f"Failed to generate embedding with local model: {e}")
            raise

    async def embed_batch(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """批量生成嵌入向量
        
        如果模型尚未加载完成，会阻塞等待（带超时）。
        
        Args:
            requests: 嵌入请求列表
            
        Returns:
            List[EmbeddingResponse]: 嵌入响应列表
        """
        # 在线程池中等待模型加载完成
        await asyncio.to_thread(self._wait_for_model)
        
        try:
            texts = [req.text for req in requests]
            
            # CPU 密集型批量编码放到线程池执行
            embeddings = await asyncio.to_thread(
                self._model_instance.encode,
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
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
            logger.error(f"Failed to batch generate embeddings: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        loading = not self._load_complete.is_set()
        status = {
            "provider": "local",
            "status": "loading" if loading else ("ok" if self._model_instance else "error"),
            "model": self._model,
            "dimension": self._dimension,
            "device": self.device,
            "available": self._model_instance is not None,
            "is_ready": self.is_ready,
            "loading": loading,
            "load_error": str(self._load_error) if self._load_error else None,
        }
        
        # 测试调用（仅在模型已就绪时）
        if self._model_instance and self.is_ready:
            try:
                test_result = await self.embed(EmbeddingRequest(text="test"))
                status["status"] = "ok"
            except Exception as e:
                status["status"] = "error"
                status["error"] = str(e)
        
        return status

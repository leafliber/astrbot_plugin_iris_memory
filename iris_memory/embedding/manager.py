"""
嵌入管理器 - 策略模式和降级管理
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass

from .base import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from .astrbot_provider import AstrBotProvider
from .local_provider import LocalProvider
from .fallback_provider import FallbackProvider
from iris_memory.utils.logger import logger


class EmbeddingStrategy(str, Enum):
    """嵌入策略"""
    AUTO = "auto"  # 自动选择（按优先级降级）
    ASTRBOT = "astrbot"  # 仅使用 AstrBot
    LOCAL = "local"  # 仅使用本地模型
    FALLBACK = "fallback"  # 仅使用降级


@dataclass
class ProviderPriority:
    """提供者优先级配置"""
    provider_class: Type[EmbeddingProvider]
    priority: int
    enabled: bool = True


class EmbeddingManager:
    """嵌入管理器
    
    管理多种嵌入提供者，实现策略模式和自动降级。
    优先级：AstrBot → Local → Fallback
    """

    def __init__(self, config: Any, data_path: Optional[Any] = None):
        """初始化嵌入管理器
        
        Args:
            config: 插件配置对象
            data_path: 数据目录路径
        """
        self.config = config
        self.data_path = data_path
        
        # 提供者实例
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.current_provider: Optional[EmbeddingProvider] = None
        self.current_strategy: EmbeddingStrategy = EmbeddingStrategy.AUTO
        
        # 优先级配置
        self.priorities = [
            ProviderPriority(AstrBotProvider, 1),
            ProviderPriority(LocalProvider, 2),
            ProviderPriority(FallbackProvider, 3),
        ]
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "provider_usage": {}
        }
        
        # 插件上下文（用于 AstrBot API）
        self.plugin_context = None

    def set_plugin_context(self, context: Any):
        """设置插件上下文
        
        Args:
            context: AstrBot 插件上下文
        """
        self.plugin_context = context
        # 传递给配置对象
        if not hasattr(self.config, '_plugin_context'):
            self.config._plugin_context = context

    async def initialize(self) -> bool:
        """初始化嵌入管理器
        
        Returns:
            bool: 是否至少有一个提供者可用
        """
        logger.info("Initializing embedding manager...")
        
        # 获取用户配置的策略
        strategy_str = self._get_config(
            "chroma_config.embedding_strategy",
            "auto"
        ).lower()
        
        try:
            self.current_strategy = EmbeddingStrategy(strategy_str)
        except ValueError:
            logger.warning(f"Invalid strategy '{strategy_str}', using AUTO")
            self.current_strategy = EmbeddingStrategy.AUTO
        
        logger.info(f"Embedding strategy: {self.current_strategy.value}")
        
        # 根据策略初始化提供者
        if self.current_strategy == EmbeddingStrategy.AUTO:
            # 自动模式：按优先级初始化所有提供者
            for priority in self.priorities:
                provider = priority.provider_class(self.config)
                success = await provider.initialize()
                if success:
                    provider_name = provider.__class__.__name__.replace("Provider", "").lower()
                    self.providers[provider_name] = provider
                    self.stats["provider_usage"][provider_name] = 0
                    logger.info(f"Initialized embedding provider: {provider_name}")
                else:
                    logger.debug(f"Failed to initialize {priority.provider_class.__name__}")
            
            # 选择当前提供者（最高优先级）
            if self.providers:
                self.current_provider = self.providers[self._get_best_provider()]
                logger.info(f"Selected embedding provider: {self.current_provider.__class__.__name__}")
                return True
            
        elif self.current_strategy == EmbeddingStrategy.ASTRBOT:
            provider = AstrBotProvider(self.config)
            if await provider.initialize():
                self.providers["astrbot"] = provider
                self.current_provider = provider
                self.stats["provider_usage"]["astrbot"] = 0
                return True
        
        elif self.current_strategy == EmbeddingStrategy.LOCAL:
            provider = LocalProvider(self.config)
            if await provider.initialize():
                self.providers["local"] = provider
                self.current_provider = provider
                self.stats["provider_usage"]["local"] = 0
                return True
        
        elif self.current_strategy == EmbeddingStrategy.FALLBACK:
            provider = FallbackProvider(self.config)
            if await provider.initialize():
                self.providers["fallback"] = provider
                self.current_provider = provider
                self.stats["provider_usage"]["fallback"] = 0
                return True
        
        # 如果没有提供者可用，至少初始化降级提供者
        logger.warning("No embedding provider available, initializing fallback")
        provider = FallbackProvider(self.config)
        if await provider.initialize():
            self.providers["fallback"] = provider
            self.current_provider = provider
            self.stats["provider_usage"]["fallback"] = 0
            return True
        
        logger.error("Failed to initialize any embedding provider")
        return False

    def _get_best_provider(self) -> str:
        """获取最佳提供者（最高优先级）
        
        Returns:
            str: 提供者名称
        """
        for priority in self.priorities:
            provider_name = priority.provider_class.__name__.replace("Provider", "").lower()
            if provider_name in self.providers:
                return provider_name
        
        # 如果没有，返回降级
        return "fallback"

    async def embed(self, text: str, dimension: Optional[int] = None) -> List[float]:
        """生成嵌入向量（自动降级）
        
        Args:
            text: 文本内容
            dimension: 目标维度（可选）
            
        Returns:
            List[float]: 嵌入向量
        """
        self.stats["total_requests"] += 1
        
        # 如果有当前提供者，尝试使用
        if self.current_provider:
            try:
                request = EmbeddingRequest(
                    text=text,
                    dimension=dimension
                )
                response = await self.current_provider.embed(request)
                
                # 更新统计
                self.stats["successful_requests"] += 1
                provider_name = self.current_provider.__class__.__name__.replace("Provider", "").lower()
                self.stats["provider_usage"][provider_name] = self.stats["provider_usage"].get(provider_name, 0) + 1
                
                return response.to_list()
                
            except Exception as e:
                logger.warning(f"Current provider failed: {e}, trying fallback")
                self.stats["failed_requests"] += 1
        
        # 如果当前提供者失败，尝试降级
        return await self._embed_with_fallback(text, dimension)

    async def _embed_with_fallback(self, text: str, dimension: Optional[int] = None) -> List[float]:
        """使用降级策略生成嵌入
        
        Args:
            text: 文本内容
            dimension: 目标维度
            
        Returns:
            List[float]: 嵌入向量
        """
        # 按优先级尝试所有提供者
        for priority in self.priorities:
            provider_name = priority.provider_class.__name__.replace("Provider", "").lower()
            
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            
            try:
                request = EmbeddingRequest(text=text, dimension=dimension)
                response = await provider.embed(request)
                
                # 更新当前提供者
                self.current_provider = provider
                logger.info(f"Switched to provider: {provider_name}")
                
                # 更新统计
                self.stats["successful_requests"] += 1
                self.stats["provider_usage"][provider_name] = self.stats["provider_usage"].get(provider_name, 0) + 1
                
                return response.to_list()
                
            except Exception as e:
                logger.debug(f"Provider {provider_name} failed: {e}")
                continue
        
        # 如果所有都失败，使用降级提供者（必须存在）
        if "fallback" in self.providers:
            try:
                provider = self.providers["fallback"]
                request = EmbeddingRequest(text=text, dimension=dimension)
                response = await provider.embed(request)
                
                self.current_provider = provider
                self.stats["successful_requests"] += 1
                self.stats["provider_usage"]["fallback"] = self.stats["provider_usage"].get("fallback", 0) + 1
                
                logger.warning("Using fallback provider (pseudo-random vectors)")
                return response.to_list()
                
            except Exception as e:
                logger.error(f"Fallback provider failed: {e}")
        
        raise RuntimeError("All embedding providers failed")

    async def embed_batch(self, texts: List[str], dimension: Optional[int] = None) -> List[List[float]]:
        """批量生成嵌入向量
        
        Args:
            texts: 文本列表
            dimension: 目标维度
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        requests = [EmbeddingRequest(text=text, dimension=dimension) for text in texts]
        responses = await self.current_provider.embed_batch(requests)
        return [response.to_list() for response in responses]

    def get_dimension(self) -> int:
        """获取当前提供者的维度
        
        Returns:
            int: 嵌入维度
        """
        if self.current_provider:
            return self.current_provider.dimension
        return self._get_config("chroma_config.embedding_dimension", 1024)

    def get_model(self) -> str:
        """获取当前提供者的模型名称
        
        Returns:
            str: 模型名称
        """
        if self.current_provider:
            return self.current_provider.model
        return "unknown"

    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        results = {
            "strategy": self.current_strategy.value,
            "current_provider": self.current_provider.__class__.__name__ if self.current_provider else "none",
            "providers": {},
            "stats": self.stats
        }
        
        for name, provider in self.providers.items():
            try:
                results["providers"][name] = await provider.health_check()
            except Exception as e:
                results["providers"][name] = {"status": "error", "error": str(e)}
        
        return results

    async def detect_existing_dimension(self, collection) -> Optional[int]:
        """检测现有集合的嵌入维度
        
        Args:
            collection: Chroma 集合对象
            
        Returns:
            Optional[int]: 嵌入维度，如果无法检测则返回 None
        """
        try:
            # 获取一个样本记录
            results = collection.get(limit=1)
            
            if results['embeddings'] and results['embeddings'][0]:
                dimension = len(results['embeddings'][0])
                logger.info(f"Detected existing collection dimension: {dimension}")
                return dimension
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to detect existing dimension: {e}")
            return None

    def _get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键（支持点分隔）
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = getattr(value, k, value.get(k) if isinstance(value, dict) else default)
            return value if value is not None else default
        except (AttributeError, KeyError):
            return default

    async def switch_strategy(self, strategy: EmbeddingStrategy) -> bool:
        """切换嵌入策略
        
        Args:
            strategy: 目标策略
            
        Returns:
            bool: 是否切换成功
        """
        logger.info(f"Switching embedding strategy to: {strategy.value}")
        self.current_strategy = strategy
        
        # 重新初始化
        return await self.initialize()

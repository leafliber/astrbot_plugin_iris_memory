"""
嵌入管理器 - 策略模式和降级管理
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Type
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import time

from .base import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from .astrbot_provider import AstrBotProvider
from .local_provider import LocalProvider
from .fallback_provider import FallbackProvider
from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import CacheDefaults

# 模块logger
logger = get_logger("embedding_manager")


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
    
    新增：Embedding 缓存机制，减少重复计算
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
            "cache_hits": 0,
            "cache_misses": 0,
            "provider_usage": {}
        }
        
        # Embedding 缓存（文本哈希 -> (向量, 过期时间))，使用 OrderedDict 实现真正的 LRU
        self._embedding_cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._cache_max_size = CacheDefaults.EMBEDDING_CACHE_MAX_SIZE
        self._cache_ttl: float = CacheDefaults.EMBEDDING_CACHE_TTL
        self._cache_enabled = True   # 是否启用缓存
        self._cache_model_key: str = ""  # 缓存对应的模型标识，模型切换时自动清空
        
        # 插件上下文（用于 AstrBot API）
        self.plugin_context = None
    
    @property
    def is_ready(self) -> bool:
        """检查当前嵌入提供者是否已就绪（模型已加载完成）
        
        Returns:
            bool: 提供者是否可用
        """
        if self.current_provider is None:
            return False
        return self.current_provider.is_ready
    
    def _get_cache_key(self, text: str, dimension: Optional[int] = None) -> str:
        """生成缓存键
        
        Args:
            text: 文本内容
            dimension: 目标维度
            
        Returns:
            str: 缓存键（SHA256哈希）
        """
        key_str = f"{text}:{dimension or self.get_dimension()}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """从缓存获取 embedding（命中时移至末尾以实现 LRU，并检查 TTL）
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[List[float]]: 缓存的向量或 None
        """
        if not self._cache_enabled:
            return None
        entry = self._embedding_cache.get(cache_key)
        if entry is not None:
            embedding, expire_at = entry
            if time.monotonic() > expire_at:
                # TTL 过期，移除
                del self._embedding_cache[cache_key]
                return None
            # 移至末尾表示最近使用
            self._embedding_cache.move_to_end(cache_key)
            return embedding
        return None
    
    def _add_to_cache(self, cache_key: str, embedding: List[float]):
        """添加 embedding 到缓存
        
        Args:
            cache_key: 缓存键
            embedding: 嵌入向量
        """
        if not self._cache_enabled:
            return
        
        expire_at = time.monotonic() + self._cache_ttl
        
        # 如果 key 已存在，更新并移至末尾
        if cache_key in self._embedding_cache:
            self._embedding_cache.move_to_end(cache_key)
            self._embedding_cache[cache_key] = (embedding, expire_at)
            return
        
        # LRU 淘汰：超出容量时移除最旧（最前面）的条目
        while len(self._embedding_cache) >= self._cache_max_size:
            evicted_key, _ = self._embedding_cache.popitem(last=False)
            logger.debug(f"Cache eviction: removed oldest entry {evicted_key[:16]}...")
        
        self._embedding_cache[cache_key] = (embedding, expire_at)
    
    def clear_cache(self):
        """清空缓存"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计
        
        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        return {
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._embedding_cache),
            "cache_max_size": self._cache_max_size,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": (
                self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
            )
        }

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
        
        # 使用配置管理器获取策略
        from iris_memory.core.config_manager import get_config_manager
        cfg = get_config_manager()
        strategy_str = cfg.embedding_strategy.lower()
        
        try:
            self.current_strategy = EmbeddingStrategy(strategy_str)
        except ValueError:
            logger.warning(f"Invalid strategy '{strategy_str}', using AUTO")
            self.current_strategy = EmbeddingStrategy.AUTO
        
        logger.info(f"Embedding strategy: {self.current_strategy.value}")
        
        # 根据策略初始化提供者
        if self.current_strategy == EmbeddingStrategy.AUTO:
            # 自动模式：按优先级初始化所有提供者
            logger.debug("AUTO mode: initializing providers by priority...")
            for priority in self.priorities:
                provider_name = priority.provider_class.__name__
                logger.debug(f"Trying to initialize {provider_name} (priority={priority.priority})...")
                
                # 特殊处理 AstrBotProvider，需要传入 context
                if priority.provider_class == AstrBotProvider:
                    provider = AstrBotProvider(self.config, self.plugin_context)
                else:
                    provider = priority.provider_class(self.config)
                    
                success = await provider.initialize()
                if success:
                    short_name = provider_name.replace("Provider", "").lower()
                    self.providers[short_name] = provider
                    self.stats["provider_usage"][short_name] = 0
                    logger.info(f"Initialized embedding provider: {short_name}")
                else:
                    logger.debug(f"Failed to initialize {provider_name}")
            
            # 选择当前提供者（最高优先级）
            if self.providers:
                best_provider_name = self._get_best_provider()
                self.current_provider = self.providers[best_provider_name]
                if self.current_provider.is_ready:
                    logger.info(f"Selected embedding provider: {best_provider_name} (dimension={self.get_dimension()})")
                else:
                    logger.info(f"Selected embedding provider: {best_provider_name} (dimension=loading...)")
                logger.debug(f"Available providers: {list(self.providers.keys())}")
                return True
            
        elif self.current_strategy == EmbeddingStrategy.ASTRBOT:
            logger.debug("ASTRBOT mode: initializing AstrBot provider...")
            provider = AstrBotProvider(self.config, self.plugin_context)
            if await provider.initialize():
                self.providers["astrbot"] = provider
                self.current_provider = provider
                self.stats["provider_usage"]["astrbot"] = 0
                logger.info(f"Initialized AstrBot provider (dimension={self.get_dimension()})")
                return True
            logger.warning("Failed to initialize AstrBot provider")
        
        elif self.current_strategy == EmbeddingStrategy.LOCAL:
            logger.debug("LOCAL mode: initializing Local provider...")
            provider = LocalProvider(self.config)
            if await provider.initialize():
                self.providers["local"] = provider
                self.current_provider = provider
                self.stats["provider_usage"]["local"] = 0
                logger.info(f"Initialized Local provider (dimension={self.get_dimension()})")
                return True
            logger.warning("Failed to initialize Local provider")
        
        elif self.current_strategy == EmbeddingStrategy.FALLBACK:
            logger.debug("FALLBACK mode: initializing Fallback provider...")
            provider = FallbackProvider(self.config)
            if await provider.initialize():
                self.providers["fallback"] = provider
                self.current_provider = provider
                self.stats["provider_usage"]["fallback"] = 0
                logger.info(f"Initialized Fallback provider (dimension={self.get_dimension()})")
                return True
            logger.warning("Failed to initialize Fallback provider")
        
        # 如果没有提供者可用，至少初始化降级提供者
        logger.warning("No embedding provider available, initializing fallback as last resort")
        provider = FallbackProvider(self.config)
        if await provider.initialize():
            self.providers["fallback"] = provider
            self.current_provider = provider
            self.stats["provider_usage"]["fallback"] = 0
            logger.info(f"Initialized Fallback provider as fallback (dimension={self.get_dimension()})")
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
        """生成嵌入向量（自动降级 + 缓存）
        
        Args:
            text: 文本内容
            dimension: 目标维度（可选）
            
        Returns:
            List[float]: 嵌入向量
        """
        self.stats["total_requests"] += 1
        
        # 检查模型是否变更，变更时清空缓存
        current_model = self.get_model()
        if self._cache_model_key and self._cache_model_key != current_model:
            logger.info(
                f"Embedding model changed from '{self._cache_model_key}' to '{current_model}', "
                f"clearing cache ({len(self._embedding_cache)} entries)"
            )
            self.clear_cache()
        self._cache_model_key = current_model
        
        # 1. 检查缓存
        cache_key = self._get_cache_key(text, dimension)
        cached_embedding = self._get_from_cache(cache_key)
        if cached_embedding is not None:
            self.stats["cache_hits"] += 1
            return cached_embedding
        
        self.stats["cache_misses"] += 1
        
        # 2. 如果有当前提供者，尝试使用
        embedding_result = None
        if self.current_provider:
            provider_name = self.current_provider.__class__.__name__.replace("Provider", "").lower()
            try:
                request = EmbeddingRequest(
                    text=text,
                    dimension=dimension
                )
                response = await self.current_provider.embed(request)
                embedding_result = response.to_list()
                
                # 更新统计
                self.stats["successful_requests"] += 1
                self.stats["provider_usage"][provider_name] = self.stats["provider_usage"].get(provider_name, 0) + 1
                
            except Exception as e:
                logger.warning(f"Current provider {provider_name} failed: {e}, trying fallback")
                self.stats["failed_requests"] += 1
        
        # 3. 如果当前提供者失败，尝试降级
        if embedding_result is None:
            embedding_result = await self._embed_with_fallback(text, dimension)
        
        # 4. 添加到缓存
        self._add_to_cache(cache_key, embedding_result)
        
        return embedding_result

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
                logger.warning(f"Provider {provider_name} failed: {e}")
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
        if not self.current_provider:
            # 尝试降级而非直接报错
            logger.warning("No current provider for embed_batch, attempting fallback")
            results = []
            for text in texts:
                results.append(await self.embed(text, dimension))
            return results
        try:
            requests = [EmbeddingRequest(text=text, dimension=dimension) for text in texts]
            responses = await self.current_provider.embed_batch(requests)
            return [response.to_list() for response in responses]
        except Exception as e:
            logger.warning(f"embed_batch failed with current provider: {e}, falling back to individual embed")
            results = []
            for text in texts:
                results.append(await self.embed(text, dimension))
            return results

    def get_dimension(self) -> int:
        """获取当前提供者的维度
        
        Returns:
            int: 嵌入维度
        """
        if self.current_provider:
            return self.current_provider.dimension
        from iris_memory.core.config_manager import get_config_manager
        return get_config_manager().embedding_dimension

    def get_model(self) -> str:
        """获取当前提供者的模型名称
        
        Returns:
            str: 模型名称
        """
        if self.current_provider:
            try:
                return self.current_provider.model
            except Exception:
                return "unknown"
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
            # 获取一个样本记录（必须显式请求 embeddings）
            results = collection.get(limit=1, include=["embeddings"])
            
            if results.get('embeddings') and results['embeddings'][0]:
                dimension = len(results['embeddings'][0])
                logger.info(f"Detected existing collection dimension: {dimension}")
                return dimension
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to detect existing dimension: {e}")
            return None

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

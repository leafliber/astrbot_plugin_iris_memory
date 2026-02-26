"""
嵌入管理器 - 源选择

用户通过 embedding.source 选择嵌入源（auto / astrbot / local）：
- auto: 优先使用 AstrBot API，不可用时自动切换到本地模型，最后使用 Fallback
- astrbot: 仅使用 AstrBot API，不可用时使用 Fallback
- local: 仅使用本地模型，不可用时使用 Fallback

所有模式均由 embedding.source 控制，不再支持额外的降级配置选项。
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from collections import OrderedDict
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


class EmbeddingSource(str, Enum):
    """嵌入源选择"""
    AUTO = "auto"        # 自动选择（AstrBot 优先 → Local → Fallback）
    ASTRBOT = "astrbot"  # 仅使用 AstrBot embedding 服务
    LOCAL = "local"      # 仅使用本地模型


# 向后兼容别名
EmbeddingStrategy = EmbeddingSource


class EmbeddingManager:
    """嵌入管理器

    根据用户配置的 embedding.source 选择嵌入源，
    简化的初始化逻辑：以用户配置为主，降级为辅。

    支持 Embedding 缓存，减少重复计算。
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
        self.current_source: EmbeddingSource = EmbeddingSource.AUTO

        # 统计信息
        self.stats: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "provider_usage": {}
        }

        # Embedding 缓存（文本哈希 -> (向量, 过期时间)），使用 OrderedDict 实现 LRU
        self._embedding_cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._cache_max_size = CacheDefaults.EMBEDDING_CACHE_MAX_SIZE
        self._cache_ttl: float = CacheDefaults.EMBEDDING_CACHE_TTL
        self._cache_enabled = True
        self._cache_model_key: str = ""

        # 插件上下文（用于 AstrBot API）
        self.plugin_context = None

    @property
    def is_ready(self) -> bool:
        """检查当前嵌入提供者是否已就绪

        Returns:
            bool: 提供者是否可用
        """
        if self.current_provider is None:
            return False
        return self.current_provider.is_ready

    # ========== 缓存管理 ==========

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
                del self._embedding_cache[cache_key]
                return None
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

        if cache_key in self._embedding_cache:
            self._embedding_cache.move_to_end(cache_key)
            self._embedding_cache[cache_key] = (embedding, expire_at)
            return

        while len(self._embedding_cache) >= self._cache_max_size:
            evicted_key, _ = self._embedding_cache.popitem(last=False)
            logger.debug(f"Cache eviction: removed oldest entry {evicted_key[:16]}...")

        self._embedding_cache[cache_key] = (embedding, expire_at)

    def clear_cache(self):
        self._embedding_cache.clear()
        logger.debug("Embedding cache cleared")

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

    # ========== 上下文管理 ==========

    def set_plugin_context(self, context: Any):
        """设置插件上下文

        Args:
            context: AstrBot 插件上下文
        """
        self.plugin_context = context
        if not hasattr(self.config, '_plugin_context'):
            self.config._plugin_context = context

    # ========== 初始化逻辑 ==========

    async def initialize(self) -> bool:
        logger.debug("Initializing embedding manager...")

        from iris_memory.core.config_manager import get_config_manager
        cfg = get_config_manager()
        source_str = cfg.embedding_source.lower()

        try:
            self.current_source = EmbeddingSource(source_str)
        except ValueError:
            logger.warning(f"Invalid embedding source '{source_str}', using AUTO")
            self.current_source = EmbeddingSource.AUTO

        logger.debug(f"Embedding source: {self.current_source.value}")

        # 根据源选择初始化提供者
        if self.current_source == EmbeddingSource.AUTO:
            return await self._init_auto(cfg)
        elif self.current_source == EmbeddingSource.ASTRBOT:
            return await self._init_astrbot(cfg)
        elif self.current_source == EmbeddingSource.LOCAL:
            return await self._init_local(cfg)

        # 不应到达此处
        return await self._init_fallback_as_last_resort()

    async def _init_auto(self, cfg: Any) -> bool:
        """AUTO 模式：AstrBot API 优先 → 本地模型 → Fallback

        当 AstrBot API 不可用时，自动降级到本地模型。

        Args:
            cfg: 配置管理器

        Returns:
            bool: 是否初始化成功
        """
        # 1. 尝试 AstrBot API
        astrbot_ok = await self._try_init_astrbot(cfg)
        if astrbot_ok:
            self.current_provider = self.providers["astrbot"]
            logger.debug(
                f"Selected embedding provider: astrbot "
                f"(model={self.current_provider.model}, dimension={self.get_dimension()})"
            )
            return True

        # 2. AstrBot 失败，尝试 Local
        logger.debug("AstrBot unavailable, trying local provider")
        local_ok = await self._try_init_local(cfg)
        if local_ok:
            self.current_provider = self.providers["local"]
            if self.current_provider.is_ready:
                logger.debug(
                    f"AstrBot unavailable, using local provider "
                    f"(model={self.current_provider.model}, dimension={self.get_dimension()})"
                )
            else:
                logger.debug(
                    f"AstrBot unavailable, using local provider "
                    f"(model={self.current_provider.model}, dimension=loading...)"
                )
            return True
        logger.warning("Local provider initialization also failed")

        # 3. 都失败，使用 Fallback
        return await self._init_fallback_as_last_resort()

    async def _init_astrbot(self, cfg: Any) -> bool:
        """ASTRBOT 模式：仅使用 AstrBot API

        严格使用 AstrBot API，不可用时直接使用 Fallback，不尝试本地模型。

        Args:
            cfg: 配置管理器

        Returns:
            bool: 是否初始化成功
        """
        astrbot_ok = await self._try_init_astrbot(cfg)
        if astrbot_ok:
            self.current_provider = self.providers["astrbot"]
            logger.debug(
                f"Selected embedding provider: astrbot "
                f"(model={self.current_provider.model}, dimension={self.get_dimension()})"
            )
            return True

        logger.warning("AstrBot embedding provider unavailable, using fallback")
        return await self._init_fallback_as_last_resort()

    async def _init_local(self, cfg: Any) -> bool:
        """LOCAL 模式：仅使用本地模型

        严格使用本地模型，不尝试 AstrBot API。

        Args:
            cfg: 配置管理器

        Returns:
            bool: 是否初始化成功
        """
        local_ok = await self._try_init_local(cfg)
        if local_ok:
            self.current_provider = self.providers["local"]
            if self.current_provider.is_ready:
                logger.debug(
                    f"Selected embedding provider: local "
                    f"(model={self.current_provider.model}, dimension={self.get_dimension()})"
                )
            else:
                logger.debug(
                    f"Selected embedding provider: local "
                    f"(model={self.current_provider.model}, dimension=loading...)"
                )
            return True

        logger.warning("Local provider initialization failed, using fallback")
        return await self._init_fallback_as_last_resort()

    async def _try_init_astrbot(self, cfg: Any) -> bool:
        """尝试初始化 AstrBot 提供者

        Args:
            cfg: 配置管理器

        Returns:
            bool: 是否初始化成功
        """
        provider = AstrBotProvider(
            self.config,
            astrbot_context=self.plugin_context,
            provider_id=cfg.embedding_astrbot_provider_id,
        )
        success = await provider.initialize()
        if success:
            self.providers["astrbot"] = provider
            self.stats["provider_usage"]["astrbot"] = 0
            return True
        return False

    async def _try_init_local(self, cfg: Any) -> bool:
        """尝试初始化本地提供者

        Args:
            cfg: 配置管理器

        Returns:
            bool: 是否初始化成功
        """
        provider = LocalProvider(self.config)
        success = await provider.initialize()
        if success:
            self.providers["local"] = provider
            self.stats["provider_usage"]["local"] = 0
            return True
        return False

    async def _init_fallback_as_last_resort(self) -> bool:
        """初始化 Fallback 提供者作为最后手段

        Returns:
            bool: 是否初始化成功
        """
        logger.warning("Initializing fallback provider (pseudo-random vectors) as last resort")
        provider = FallbackProvider(self.config)
        if await provider.initialize():
            self.providers["fallback"] = provider
            self.current_provider = provider
            self.stats["provider_usage"]["fallback"] = 0
            logger.debug(f"Fallback provider initialized (dimension={self.get_dimension()})")
            return True

        logger.error("Failed to initialize any embedding provider")
        return False

    # ========== 嵌入生成 ==========

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
            logger.debug(
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

        # 2. 使用当前提供者生成嵌入
        embedding_result = None
        if self.current_provider:
            provider_name = self._get_provider_name(self.current_provider)
            try:
                request = EmbeddingRequest(text=text, dimension=dimension)
                response = await self.current_provider.embed(request)
                embedding_result = response.to_list()

                self.stats["successful_requests"] += 1
                self.stats["provider_usage"][provider_name] = \
                    self.stats["provider_usage"].get(provider_name, 0) + 1

            except Exception as e:
                logger.warning(f"Current provider {provider_name} failed: {e}, trying fallback")
                self.stats["failed_requests"] += 1

        # 3. 当前提供者失败，尝试降级
        if embedding_result is None:
            embedding_result = await self._embed_with_fallback(text, dimension)

        # 4. 添加到缓存
        self._add_to_cache(cache_key, embedding_result)

        return embedding_result

    async def _embed_with_fallback(self, text: str, dimension: Optional[int] = None) -> List[float]:
        """使用降级策略生成嵌入

        按 astrbot → local → fallback 的顺序尝试所有可用提供者。

        Args:
            text: 文本内容
            dimension: 目标维度

        Returns:
            List[float]: 嵌入向量
        """
        provider_order = ["astrbot", "local", "fallback"]

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]

            try:
                request = EmbeddingRequest(text=text, dimension=dimension)
                response = await provider.embed(request)

                # 切换当前提供者
                self.current_provider = provider
                logger.debug(f"Switched to provider: {provider_name}")

                self.stats["successful_requests"] += 1
                self.stats["provider_usage"][provider_name] = \
                    self.stats["provider_usage"].get(provider_name, 0) + 1

                return response.to_list()

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

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

    # ========== 信息查询 ==========

    def get_dimension(self) -> int:
        """获取当前提供者的维度

        Returns:
            int: 嵌入维度
        """
        if self.current_provider:
            return self.current_provider.dimension
        from iris_memory.core.config_manager import get_config_manager
        return get_config_manager().embedding_local_dimension

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
            "source": self.current_source.value,
            "current_provider": self._get_provider_name(self.current_provider) if self.current_provider else "none",
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
            results = collection.get(limit=1, include=["embeddings"])

            if results.get('embeddings') and results['embeddings'][0]:
                dimension = len(results['embeddings'][0])
                logger.debug(f"Detected existing collection dimension: {dimension}")
                return dimension

            return None

        except Exception as e:
            logger.debug(f"Failed to detect existing dimension: {e}")
            return None

    # ========== 内部工具 ==========

    @staticmethod
    def _get_provider_name(provider: Optional[EmbeddingProvider]) -> str:
        """获取提供者的简短名称

        Args:
            provider: 嵌入提供者实例

        Returns:
            str: 简短名称
        """
        if provider is None:
            return "none"
        return provider.__class__.__name__.replace("Provider", "").lower()

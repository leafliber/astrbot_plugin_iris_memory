"""
Iris Chat Memory - Token 统计管理

使用 AstrBot KV 存储持久化 Token 使用统计。
"""

from dataclasses import dataclass, asdict
from typing import Dict, TYPE_CHECKING
from collections import defaultdict

from iris_memory.core import get_logger
from iris_memory.core.storage import KVStorage

if TYPE_CHECKING:
    pass

logger = get_logger("token_stats")


@dataclass
class TokenUsage:
    """Token 使用统计

    记录某个模块或全局的 Token 使用情况。

    Attributes:
        total_input_tokens: 总输入 Token 数
        total_output_tokens: 总输出 Token 数
        total_calls: 总调用次数
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0

    @property
    def total_tokens(self) -> int:
        """总 Token 数"""
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> Dict[str, int]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "TokenUsage":
        """从字典创建实例"""
        return cls(
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_calls=data.get("total_calls", 0),
        )


class TokenStatsManager:
    """Token 统计管理器

    使用 AstrBot KV 存储持久化 Token 使用统计。
    支持全局统计和模块级统计。

    数据结构：
        - "token_stats:global": 全局统计
        - "token_stats:module:l1_summarizer": 模块统计
        - "token_stats:module:l3_kg_extraction": 模块统计

    Attributes:
        _storage: KV 存储适配器
        _cache: 内存缓存
    """

    KEY_PREFIX = "token_stats"

    def __init__(self, storage: KVStorage):
        """初始化统计管理器

        Args:
            storage: KV 存储适配器（实现 KVStorage 协议的对象）
        """
        self._storage = storage
        self._cache: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        # 已知模块集合：record_usage 时登记，get_all_stats 时遍历加载。
        # 解决 KV 存储无 list_keys 接口的问题。
        self._known_modules: set[str] = {"global"}

    def _get_kv_key(self, module: str) -> str:
        """获取 KV 存储键名

        Args:
            module: 模块名（"global" 或具体模块名）

        Returns:
            KV 存储键名
        """
        if module == "global":
            return f"{self.KEY_PREFIX}:global"
        else:
            return f"{self.KEY_PREFIX}:module:{module}"

    async def _load_from_kv(self, module: str) -> TokenUsage:
        """从 KV 存储加载统计数据

        Args:
            module: 模块名

        Returns:
            TokenUsage 实例
        """
        key = self._get_kv_key(module)
        try:
            data = await self._storage.get_kv_data(key, {})
            if data:
                usage = TokenUsage.from_dict(data)
                self._cache[module] = usage
                return usage
        except Exception as e:
            logger.warning(f"从 KV 存储加载 Token 统计失败：{module}, error={e}")

        return self._cache[module]

    async def _save_to_kv(self, module: str) -> None:
        """保存统计数据到 KV 存储

        Args:
            module: 模块名
        """
        key = self._get_kv_key(module)
        try:
            data = self._cache[module].to_dict()
            await self._storage.put_kv_data(key, data)
        except Exception as e:
            logger.warning(f"保存 Token 统计到 KV 存储失败：{module}, error={e}")

    async def record_usage(
        self, module: str, input_tokens: int, output_tokens: int
    ) -> None:
        """记录 Token 使用

        更新模块统计和全局统计。

        Args:
            module: 调用模块标识（如 "l1_summarizer"）
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
        """
        # 重启后 _cache 从 0 起，若不先回读历史累计，本次会话的小值会
        # 用 _save_to_kv 覆盖历史总量。get_stats 是唯一触发 _load_from_kv
        # 的路径——若重启后先调 record_usage 再调 get_stats，历史已丢失。
        # 修复：对未加载的模块（含 global）先 _load_from_kv 回读。
        if module not in self._cache:
            await self._load_from_kv(module)
        if module != "global" and "global" not in self._cache:
            await self._load_from_kv("global")

        # 登记模块名，供 get_all_stats 遍历加载
        self._known_modules.add(module)
        if module != "global":
            self._known_modules.add("global")

        self._cache[module].total_input_tokens += input_tokens
        self._cache[module].total_output_tokens += output_tokens
        self._cache[module].total_calls += 1

        if module != "global":
            self._cache["global"].total_input_tokens += input_tokens
            self._cache["global"].total_output_tokens += output_tokens
            self._cache["global"].total_calls += 1

        await self._save_to_kv(module)
        if module != "global":
            await self._save_to_kv("global")

        logger.debug(
            f"记录 Token 使用：module={module}, "
            f"input={input_tokens}, output={output_tokens}"
        )

    async def get_stats(self, module: str = "global") -> TokenUsage:
        """获取统计信息

        优先从内存缓存读取，缓存未命中时从 KV 存储加载。

        Args:
            module: 模块名（默认 "global"）

        Returns:
            TokenUsage 实例
        """
        if module not in self._cache:
            await self._load_from_kv(module)
        return self._cache[module]

    async def reset_stats(self, module: str = "global") -> None:
        """重置统计

        清空指定模块的统计数据。

        Args:
            module: 模块名（默认 "global"）
        """
        self._cache[module] = TokenUsage()
        await self._save_to_kv(module)
        logger.info(f"已重置 Token 统计：{module}")

    async def get_all_stats(self) -> Dict[str, TokenUsage]:
        """获取所有模块的统计

        遍历已知模块列表并从 KV 加载，确保重启后未触达的模块也出现在
        返回值中，而非仅返回内存缓存。已知模块通过 record_usage 时登记。

        Returns:
            模块名到 TokenUsage 的映射
        """
        # 从 KV 加载所有已知模块（含 global）
        for module in list(self._known_modules):
            if module not in self._cache:
                await self._load_from_kv(module)

        return dict(self._cache)

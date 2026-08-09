"""
Iris Chat Memory - Token 统计管理

使用 AstrBot KV 存储持久化 Token 使用统计。
"""

from dataclasses import dataclass, asdict
from typing import Dict, TYPE_CHECKING
from collections import defaultdict
import asyncio

from iris_memory.core import get_logger
from iris_memory.core.storage import KVStorage
from iris_memory.llm_modules import ALL_LLM_MODULES

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
    successful_calls: int = 0
    failed_calls: int = 0

    @property
    def total_tokens(self) -> int:
        """总 Token 数"""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def pending_calls(self) -> int:
        """已交给框架但尚未收到结果回调的调用数。"""
        return max(0, self.total_calls - self.successful_calls - self.failed_calls)

    def to_dict(self) -> Dict[str, int]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "TokenUsage":
        """从字典创建实例"""
        total_calls = data.get("total_calls", 0)
        successful_calls = data.get("successful_calls", total_calls)
        failed_calls = data.get("failed_calls", 0)
        # 主管线调用无法跨进程继续；上次退出前未结算的请求在重启时归为失败。
        failed_calls += max(0, total_calls - successful_calls - failed_calls)
        return cls(
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_calls=total_calls,
            # 旧数据的 total_calls 全部来自成功响应，迁移时视为成功调用。
            successful_calls=successful_calls,
            failed_calls=failed_calls,
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
    MODULES_KEY = f"{KEY_PREFIX}:modules"

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
        self._modules_loaded = False
        self._lock = asyncio.Lock()

    async def _load_module_index(self) -> None:
        """加载持久化模块索引，并兼容探测旧版本的固定模块键。"""
        if self._modules_loaded:
            return
        try:
            data = await self._storage.get_kv_data(self.MODULES_KEY, [])
            if isinstance(data, list):
                self._known_modules.update(
                    item for item in data if isinstance(item, str) and item
                )
        except Exception as e:
            logger.warning(f"加载 Token 统计模块索引失败：{e}")

        # KVStorage 没有 list_keys；标准模块清单也承担旧数据迁移职责。
        self._known_modules.update(ALL_LLM_MODULES)
        self._known_modules.add("global")
        self._modules_loaded = True

    async def _save_module_index(self) -> None:
        try:
            modules = sorted(module for module in self._known_modules if module != "global")
            await self._storage.put_kv_data(self.MODULES_KEY, modules)
        except Exception as e:
            logger.warning(f"保存 Token 统计模块索引失败：{e}")

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
        await self.record_result(
            module,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True,
        )

    async def record_failure(self, module: str) -> None:
        """记录一次已实际发起但失败的 Provider 调用。"""
        await self.record_result(module, input_tokens=0, output_tokens=0, success=False)

    async def record_attempt(self, module: str) -> None:
        """登记一次由 AstrBot 主管线接管的调用请求。"""
        async with self._lock:
            await self._load_module_index()
            if module not in self._cache:
                await self._load_from_kv(module)
            if module != "global" and "global" not in self._cache:
                await self._load_from_kv("global")
            self._known_modules.update({module, "global"})
            self._cache[module].total_calls += 1
            if module != "global":
                self._cache["global"].total_calls += 1
            await self._save_to_kv(module)
            if module != "global":
                await self._save_to_kv("global")
            await self._save_module_index()

    async def record_completion(
        self,
        module: str,
        *,
        input_tokens: int,
        output_tokens: int,
        success: bool,
    ) -> None:
        """结算已由 :meth:`record_attempt` 登记的主管线调用。"""
        async with self._lock:
            await self._load_module_index()
            if module not in self._cache:
                await self._load_from_kv(module)
            if module != "global" and "global" not in self._cache:
                await self._load_from_kv("global")
            targets = [self._cache[module]]
            if module != "global":
                targets.append(self._cache["global"])
            for usage in targets:
                usage.total_input_tokens += max(0, int(input_tokens))
                usage.total_output_tokens += max(0, int(output_tokens))
                if success:
                    usage.successful_calls += 1
                else:
                    usage.failed_calls += 1
            await self._save_to_kv(module)
            if module != "global":
                await self._save_to_kv("global")

    async def record_result(
        self,
        module: str,
        *,
        input_tokens: int,
        output_tokens: int,
        success: bool,
    ) -> None:
        """原子地累计一次调用结果，并持久化模块索引。"""
        async with self._lock:
            await self._load_module_index()
            # 重启后先回读历史累计，避免本次会话覆盖旧值。
            if module not in self._cache:
                await self._load_from_kv(module)
            if module != "global" and "global" not in self._cache:
                await self._load_from_kv("global")

            self._known_modules.add(module)
            self._known_modules.add("global")

            targets = [self._cache[module]]
            if module != "global":
                targets.append(self._cache["global"])
            for usage in targets:
                usage.total_input_tokens += max(0, int(input_tokens))
                usage.total_output_tokens += max(0, int(output_tokens))
                usage.total_calls += 1
                if success:
                    usage.successful_calls += 1
                else:
                    usage.failed_calls += 1

            await self._save_to_kv(module)
            if module != "global":
                await self._save_to_kv("global")
            await self._save_module_index()

        logger.debug(
            f"记录 LLM 调用：module={module}, success={success}, "
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
        async with self._lock:
            await self._load_module_index()
            if module not in self._cache:
                await self._load_from_kv(module)
            return self._cache[module]

    async def reset_stats(self, module: str = "global") -> None:
        """重置统计

        清空指定模块的统计数据。

        Args:
            module: 模块名（默认 "global"）
        """
        async with self._lock:
            await self._load_module_index()
            self._known_modules.add(module)
            self._cache[module] = TokenUsage()
            await self._save_to_kv(module)
            await self._save_module_index()
        logger.info(f"已重置 Token 统计：{module}")

    async def get_all_stats(self) -> Dict[str, TokenUsage]:
        """获取所有模块的统计

        遍历已知模块列表并从 KV 加载，确保重启后未触达的模块也出现在
        返回值中，而非仅返回内存缓存。已知模块通过 record_usage 时登记。

        Returns:
            模块名到 TokenUsage 的映射
        """
        async with self._lock:
            await self._load_module_index()
            # 从 KV 加载所有已知模块（含 global）。空的标准模块不展示，
            # 但一旦有历史/当前调用便会存在于缓存中并返回。
            for module in list(self._known_modules):
                if module not in self._cache:
                    await self._load_from_kv(module)

            return {
                module: usage
                for module, usage in self._cache.items()
                if module == "global" or usage.total_calls > 0
            }

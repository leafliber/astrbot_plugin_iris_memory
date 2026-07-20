"""
Iris Chat Memory - LLM 调用管理器

提供统一的 LLM 调用入口，支持 Token 统计与调用追踪。

调用方式：
- generate(): 通过 context.llm_generate() 调用，会触发 AstrBot 的 on_llm_request 钩子链。
  适用于需要走完整钩子流程的场景。
- generate_direct(): 直接调用 Provider.text_chat()，绕过所有 on_llm_request 钩子。
  适用于插件内部调用（图片解析、总结器、画像分析等），避免递归触发钩子和
  不必要的上下文注入（如 sampling 触发时的图片解析）。
"""

from typing import Optional, Dict, List, Any, TYPE_CHECKING
from datetime import datetime
from collections import deque
import asyncio
import uuid
import time

from iris_memory.core import Component, get_logger
from iris_memory.core.storage import KVStorage
from iris_memory.config import get_config
from .token_stats import TokenStatsManager
from .call_log import CallLog

if TYPE_CHECKING:
    from astrbot.api.star import Context
    from astrbot.api.provider import LLMResponse
    from astrbot.core.provider.provider import Provider

logger = get_logger("llm_manager")


class LLMManager(Component):
    """LLM 调用管理器

    提供统一的 LLM 调用入口，支持：
    - Token 统计与持久化（AstrBot KV 存储）
    - 调用日志记录（内存存储）
    - 模块级 Provider 配置
    - 调用追踪

    Attributes:
        _context: AstrBot Context 对象
        _storage: KV 存储适配器
        _token_stats: Token 统计管理器
        _call_logs: 调用日志队列
    """

    def __init__(self, context: "Context", storage: KVStorage):
        """初始化管理器

        Args:
            context: AstrBot Context 对象
            storage: KV 存储适配器（实现 KVStorage 协议的对象）
        """
        super().__init__()
        self._context = context
        self._storage = storage
        self._token_stats: Optional[TokenStatsManager] = None
        self._call_logs: deque[CallLog] = deque(maxlen=100)

    @property
    def name(self) -> str:
        """组件名称"""
        return "llm_manager"

    async def initialize(self) -> None:
        """初始化管理器

        创建 Token 统计管理器，加载配置。
        """
        try:
            config = get_config()

            self._token_stats = TokenStatsManager(self._storage)

            max_logs = config.get("call_log_max_entries", 100)
            self._call_logs = deque(maxlen=max_logs)

            self._is_available = True
            logger.info("LLMManager 初始化成功")

        except Exception as e:
            self._init_error = str(e)
            logger.error(f"LLMManager 初始化失败：{e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """关闭管理器"""
        self._reset_state()
        logger.info("LLMManager 已关闭")

    async def generate(
        self,
        prompt: str,
        module: str = "default",
        provider_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        contexts: Optional[List[Dict[str, str]]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        """生成文本响应

        Args:
            prompt: 输入提示词
            module: 调用模块标识（用于统计），如 "l1_summarizer"
            provider_id: Provider ID（留空使用模块配置或默认）
            temperature: 温度参数（暂不支持，AstrBot 内部处理）
            max_tokens: 最大输出 Token 数（暂不支持，AstrBot 内部处理）
            contexts: 上下文消息列表
            timeout: 调用超时（秒），None 使用配置 llm_call_timeout_ms，<=0 不超时
            **kwargs: 其他参数

        Returns:
            生成的文本响应

        Raises:
            RuntimeError: LLMManager 未初始化
            asyncio.TimeoutError: LLM 调用超时
            Exception: LLM 调用失败
        """
        if not self._is_available:
            raise RuntimeError("LLMManager 未初始化")

        actual_provider_id = await self._resolve_provider(module, provider_id)

        if not actual_provider_id:
            logger.warning(
                f"LLM 调用跳过：module={module}, 未配置 Provider 且无法获取默认 Provider。"
                f"请在插件配置中设置 {module} 的 Provider，或在 AstrBot 中配置默认 Provider。"
            )
            raise RuntimeError(
                f"未配置 Provider：module={module}。"
                f"请在插件配置中设置相应的 Provider（如 profile.analysis_provider），"
                f"或在 AstrBot 中配置默认 Provider。"
            )

        timeout_sec = self._resolve_call_timeout(timeout)

        start_time = time.time()
        call_id = str(uuid.uuid4())

        try:
            logger.debug(
                f"LLM 调用开始：module={module}, provider={actual_provider_id}"
            )

            llm_resp: "LLMResponse" = await self._call_with_timeout(
                self._context.llm_generate(
                    chat_provider_id=actual_provider_id,
                    prompt=prompt,
                    contexts=contexts or [],
                ),
                timeout_sec,
            )

            response_text = llm_resp.completion_text or ""

            duration_ms = int((time.time() - start_time) * 1000)

            if llm_resp.usage:
                input_tokens = llm_resp.usage.input_other + llm_resp.usage.input_cached
                output_tokens = llm_resp.usage.output
            else:
                input_tokens = 0
                output_tokens = 0

            if self._token_stats:
                await self._token_stats.record_usage(
                    module=module,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            log = CallLog(
                call_id=call_id,
                timestamp=datetime.now(),
                module=module,
                provider_id=actual_provider_id or "default",
                prompt=self._truncate_text(prompt, 500),
                response=self._truncate_text(response_text, 500),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                success=True,
            )
            self._call_logs.append(log)

            logger.info(
                f"LLM 调用成功：module={module}, "
                f"tokens={input_tokens}+{output_tokens}, "
                f"duration={duration_ms}ms"
            )

            return response_text

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            log = CallLog(
                call_id=call_id,
                timestamp=datetime.now(),
                module=module,
                provider_id=actual_provider_id or "default",
                prompt=self._truncate_text(prompt, 500),
                response="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=f"LLM 调用超时({timeout_sec:.1f}s)",
            )
            self._call_logs.append(log)
            logger.warning(
                f"LLM 调用超时：module={module}, "
                f"timeout={timeout_sec:.1f}s, duration={duration_ms}ms"
            )
            raise

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            log = CallLog(
                call_id=call_id,
                timestamp=datetime.now(),
                module=module,
                provider_id=actual_provider_id or "default",
                prompt=self._truncate_text(prompt, 500),
                response="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
            )
            self._call_logs.append(log)

            logger.error(f"LLM 调用失败：module={module}, error={e}")
            raise

    async def generate_direct(
        self,
        prompt: str,
        module: str = "default",
        provider_id: Optional[str] = None,
        contexts: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        image_urls: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """直接调用 Provider 生成文本响应（绕过 on_llm_request 钩子）

        通过 context.get_provider_by_id() 获取 Provider 实例，
        直接调用 provider.text_chat()，不经过 AstrBot 的 llm_generate 流程，
        因此不会触发 on_llm_request / on_llm_response 钩子。

        适用于插件内部调用（图片解析、总结器、画像分析、查询改写等），
        避免递归触发钩子和不必要的上下文注入。

        Args:
            prompt: 输入提示词
            module: 调用模块标识（用于统计），如 "image_parsing"
            provider_id: Provider ID（留空使用模块配置或默认）
            contexts: 上下文消息列表
            system_prompt: 系统提示词（可选）
            timeout: 调用超时（秒），None 使用配置 llm_call_timeout_ms，<=0 不超时
            **kwargs: 其他参数

        Returns:
            生成的文本响应

        Raises:
            RuntimeError: LLMManager 未初始化或 Provider 不可用
            asyncio.TimeoutError: LLM 调用超时
            Exception: LLM 调用失败
        """
        if not self._is_available:
            raise RuntimeError("LLMManager 未初始化")

        actual_provider_id = await self._resolve_provider(module, provider_id)

        if not actual_provider_id:
            raise RuntimeError(
                f"未配置 Provider：module={module}。"
                f"请在插件配置中设置相应的 Provider，"
                f"或在 AstrBot 中配置默认 Provider。"
            )

        provider = self._get_provider_instance(actual_provider_id)
        if not provider:
            raise RuntimeError(
                f"Provider 实例不可用：{actual_provider_id}。"
                f"请检查 AstrBot 中该 Provider 是否已启用。"
            )

        timeout_sec = self._resolve_call_timeout(timeout)

        start_time = time.time()
        call_id = str(uuid.uuid4())

        try:
            logger.debug(
                f"LLM 直接调用开始：module={module}, provider={actual_provider_id}"
            )

            llm_resp: "LLMResponse" = await self._call_with_timeout(
                provider.text_chat(
                    prompt=prompt,
                    contexts=contexts or [],
                    system_prompt=system_prompt,
                    image_urls=image_urls,
                ),
                timeout_sec,
            )

            response_text = llm_resp.completion_text or ""

            duration_ms = int((time.time() - start_time) * 1000)

            if llm_resp.usage:
                input_tokens = llm_resp.usage.input_other + llm_resp.usage.input_cached
                output_tokens = llm_resp.usage.output
            else:
                input_tokens = 0
                output_tokens = 0

            if self._token_stats:
                await self._token_stats.record_usage(
                    module=module,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            log = CallLog(
                call_id=call_id,
                timestamp=datetime.now(),
                module=module,
                provider_id=actual_provider_id,
                prompt=self._truncate_text(prompt, 500),
                response=self._truncate_text(response_text, 500),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                success=True,
            )
            self._call_logs.append(log)

            logger.info(
                f"LLM 直接调用成功：module={module}, "
                f"tokens={input_tokens}+{output_tokens}, "
                f"duration={duration_ms}ms"
            )

            return response_text

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            log = CallLog(
                call_id=call_id,
                timestamp=datetime.now(),
                module=module,
                provider_id=actual_provider_id,
                prompt=self._truncate_text(prompt, 500),
                response="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=f"LLM 直接调用超时({timeout_sec:.1f}s)",
            )
            self._call_logs.append(log)
            logger.warning(
                f"LLM 直接调用超时：module={module}, "
                f"timeout={timeout_sec:.1f}s, duration={duration_ms}ms"
            )
            raise

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            log = CallLog(
                call_id=call_id,
                timestamp=datetime.now(),
                module=module,
                provider_id=actual_provider_id,
                prompt=self._truncate_text(prompt, 500),
                response="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
            )
            self._call_logs.append(log)

            logger.error(f"LLM 直接调用失败：module={module}, error={e}")
            raise

    async def _call_with_timeout(self, coro, timeout_sec: Optional[float]):
        """执行协程，可选超时

        Args:
            coro: 待执行的协程
            timeout_sec: 超时秒数，None 表示不超时

        Returns:
            协程结果

        Raises:
            asyncio.TimeoutError: 超时
        """
        if timeout_sec is not None:
            return await asyncio.wait_for(coro, timeout=timeout_sec)
        return await coro

    def _resolve_call_timeout(self, timeout: Optional[float]) -> Optional[float]:
        """解析 LLM 调用超时秒数

        优先级：调用方传入 timeout > 配置 llm_call_timeout_ms > 不超时

        Args:
            timeout: 调用方传入的超时（秒），None 表示用配置默认

        Returns:
            超时秒数，None 表示不超时
        """
        if timeout is not None:
            return timeout if timeout > 0 else None
        config = get_config()
        timeout_ms = config.get("llm_call_timeout_ms", 60000)
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool):
            return None
        if timeout_ms <= 0:
            return None
        return timeout_ms / 1000.0

    def _get_provider_instance(self, provider_id: str) -> Optional["Provider"]:
        """获取 Provider 实例

        通过 context.get_provider_by_id() 获取 Provider 实例。

        Args:
            provider_id: Provider ID

        Returns:
            Provider 实例，不可用时返回 None
        """
        try:
            if hasattr(self._context, "get_provider_by_id"):
                provider = self._context.get_provider_by_id(provider_id)
                if provider:
                    return provider

            if hasattr(self._context, "provider_manager"):
                provider_manager = self._context.provider_manager
                if hasattr(provider_manager, "inst_map"):
                    return provider_manager.inst_map.get(provider_id)
        except Exception as e:
            logger.debug(f"获取 Provider 实例失败: {e}")

        return None

    async def resolve_provider(
        self, module: str = "default", provider_id: Optional[str] = None
    ) -> Optional[str]:
        """解析要使用的 Provider ID（公开接口）

        优先级：参数 > 模块配置 > AstrBot 默认 Provider

        Args:
            module: 模块名
            provider_id: 参数传入的 provider_id

        Returns:
            实际使用的 Provider ID，None 表示无法获取
        """
        return await self._resolve_provider(module, provider_id)

    async def _resolve_provider(
        self, module: str, provider_id: Optional[str]
    ) -> Optional[str]:
        """解析要使用的 Provider ID

        优先级：参数 > 模块配置 > AstrBot 默认 Provider

        Args:
            module: 模块名
            provider_id: 参数传入的 provider_id

        Returns:
            实际使用的 Provider ID，None 表示无法获取
        """
        if provider_id:
            return provider_id

        config = get_config()

        module_config_map = {
            "l1_summarizer": "l1_buffer.summary_provider",
            "l3_kg_extraction": "l3_kg.extraction_provider",
            "scheduled_tasks": "scheduled_tasks.provider",
            "dream_consolidation": "scheduled_tasks.provider",
            "dream_temporal_anchor": "scheduled_tasks.provider",
            "dream_contradiction": "scheduled_tasks.provider",
            "dream_pattern_discovery": "scheduled_tasks.provider",
            "dream_pruning_confirm": "scheduled_tasks.provider",
            "image_parsing": "l1_buffer.image_parsing.provider",
            "profile_analysis": "profile.analysis_provider",
            "l2_query_rewrite": "l2_query_rewrite_provider",
        }

        config_key = module_config_map.get(module)
        if config_key:
            configured_provider = config.get(config_key)
            if configured_provider:
                return configured_provider

        default_provider = self._get_default_provider()
        if default_provider:
            return default_provider

        return None

    def _get_default_provider(self) -> Optional[str]:
        """获取 AstrBot 默认 Provider ID

        Returns:
            默认 Provider ID，无法获取时返回 None
        """
        try:
            if hasattr(self._context, "get_config"):
                config = self._context.get_config()
                if config:
                    provider_settings = config.get("provider_settings", {})
                    default_provider_id = provider_settings.get("default_provider_id")
                    if default_provider_id:
                        return default_provider_id

            if hasattr(self._context, "provider_manager"):
                provider_manager = self._context.provider_manager
                if hasattr(provider_manager, "get_default_provider"):
                    default_provider = provider_manager.get_default_provider()
                    if default_provider and hasattr(default_provider, "id"):
                        return default_provider.id
                if hasattr(provider_manager, "providers"):
                    providers = provider_manager.providers
                    if providers:
                        first_provider = providers[0]
                        if hasattr(first_provider, "id"):
                            return first_provider.id

            if hasattr(self._context, "providers"):
                providers = self._context.providers
                if providers:
                    first_provider = providers[0]
                    if hasattr(first_provider, "id"):
                        return first_provider.id
        except Exception as e:
            logger.debug(f"获取默认 Provider 失败: {e}")

        return None

    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断文本

        Args:
            text: 原始文本
            max_length: 最大长度

        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    async def call(self, prompt: str, provider: str = "") -> str:
        """调用 LLM 生成响应（LLMCaller 协议接口）

        为兼容现有代码提供简化接口。

        Args:
            prompt: 输入提示词
            provider: Provider ID

        Returns:
            生成的文本响应
        """
        return await self.generate(
            prompt=prompt, module="default", provider_id=provider if provider else None
        )

    async def generate_with_images(
        self,
        prompt: str,
        image_urls: List[str],
        module: str = "default",
        provider_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """生成文本响应（支持图片输入）

        通过 provider.text_chat 的 image_urls 参数传递图片，
        由 AstrBot Provider 负责构建正确的多模态消息格式。

        Args:
            prompt: 输入提示词
            image_urls: 图片 URL 列表（支持 HTTP URL 或 base64 data URL）
            module: 调用模块标识（用于统计），如 "image_parsing"
            provider_id: Provider ID（留空使用模块配置或默认）
            **kwargs: 其他参数

        Returns:
            生成的文本响应

        Raises:
            RuntimeError: LLMManager 未初始化
            Exception: LLM 调用失败
        """
        if not self._is_available:
            raise RuntimeError("LLMManager 未初始化")

        return await self.generate_direct(
            prompt=prompt,
            module=module,
            provider_id=provider_id,
            image_urls=image_urls,
            **kwargs,
        )

    async def get_token_stats(self, module: str = "global") -> Dict[str, Any]:
        """获取 Token 统计

        Args:
            module: 模块名（默认 "global"）

        Returns:
            统计信息字典
        """
        if not self._token_stats:
            return {
                "module": module,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
            }

        usage = await self._token_stats.get_stats(module)
        return {
            "module": module,
            "total_input_tokens": usage.total_input_tokens,
            "total_output_tokens": usage.total_output_tokens,
            "total_calls": usage.total_calls,
        }

    async def get_all_token_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模块的 Token 统计

        Returns:
            模块名到统计信息的映射
        """
        if not self._token_stats:
            return {}

        all_stats = await self._token_stats.get_all_stats()
        return {
            module: {
                "total_input_tokens": usage.total_input_tokens,
                "total_output_tokens": usage.total_output_tokens,
                "total_calls": usage.total_calls,
            }
            for module, usage in all_stats.items()
        }

    async def reset_token_stats(self, module: str = "global") -> None:
        """重置 Token 统计

        Args:
            module: 模块名（默认 "global"）
        """
        if self._token_stats:
            await self._token_stats.reset_stats(module)
            logger.info(f"已重置 Token 统计：{module}")

    def get_recent_call_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的调用日志

        Args:
            limit: 返回数量（默认 20）

        Returns:
            调用日志列表
        """
        logs = list(self._call_logs)[-limit:]
        return [
            {
                "call_id": log.call_id,
                "timestamp": log.timestamp.isoformat(),
                "module": log.module,
                "provider_id": log.provider_id,
                "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens,
                "duration_ms": log.duration_ms,
                "success": log.success,
                "error_message": log.error_message,
            }
            for log in logs
        ]

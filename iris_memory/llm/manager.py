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

from ..core import Component, get_logger, get_run_log_manager
from ..core.storage import KVStorage
from ..config import get_config
from .token_stats import TokenStatsManager
from .call_log import CallLog
from .governor import (
    GovernorSettings,
    LLMCallGovernor,
    LLMLease,
    LLMQueueTimeoutError,
)
from .policy import CallPriority, coerce_priority
from .budget import LLMCallBudgetExceeded, current_llm_call_budget

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
        self._call_logs: deque[CallLog] = deque(maxlen=1000)
        self._governor: Optional[LLMCallGovernor] = None

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

            max_logs = self._config_int(config, "call_log_max_entries", 1000, minimum=1)
            self._call_logs = deque(maxlen=max_logs)

            self._governor = LLMCallGovernor(
                GovernorSettings(
                    global_concurrency=self._config_int(
                        config, "llm_global_concurrency", 4, minimum=1
                    ),
                    provider_concurrency=self._config_int(
                        config, "llm_provider_concurrency", 2, minimum=1
                    ),
                    provider_background_concurrency=self._config_int(
                        config,
                        "llm_provider_background_concurrency",
                        1,
                        minimum=1,
                    ),
                    provider_rpm=self._config_int(
                        config, "llm_provider_rpm", 30, minimum=0
                    ),
                    provider_min_interval_ms=self._config_int(
                        config, "llm_provider_min_interval_ms", 0, minimum=0
                    ),
                    queue_limit=self._config_int(
                        config, "llm_queue_limit", 500, minimum=1
                    ),
                    interactive_reserved_slots=self._config_int(
                        config, "llm_interactive_reserved_slots", 1, minimum=0
                    ),
                    circuit_failure_threshold=self._config_int(
                        config, "llm_circuit_failure_threshold", 5, minimum=0
                    ),
                    circuit_open_seconds=self._config_int(
                        config, "llm_circuit_open_seconds", 60, minimum=0
                    ),
                    aging_seconds=self._config_int(
                        config, "llm_priority_aging_seconds", 30, minimum=1
                    ),
                )
            )

            self._is_available = True
            logger.info("LLMManager 初始化成功")

        except Exception as e:
            self._init_error = str(e)
            logger.error(f"LLMManager 初始化失败：{e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """关闭管理器"""
        if self._governor:
            await self._governor.shutdown()
        self._governor = None
        self._reset_state()
        logger.info("LLMManager 已关闭")

    @staticmethod
    def _config_int(config, key: str, default: int, *, minimum: int) -> int:
        value = config.get(key, default)
        if isinstance(value, bool):
            return default
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError, OverflowError):
            return default

    @property
    def governor(self) -> LLMCallGovernor:
        if self._governor is None:
            raise RuntimeError("LLM Governor 未初始化")
        return self._governor

    @staticmethod
    def _extract_usage(llm_resp: "LLMResponse") -> tuple[int, int]:
        """兼容不同 Provider 的 usage 实现，缺失时按 0 处理。"""
        usage = getattr(llm_resp, "usage", None)
        if not usage:
            return 0, 0
        input_tokens = int(getattr(usage, "input_other", 0) or 0) + int(
            getattr(usage, "input_cached", 0) or 0
        )
        output_tokens = int(getattr(usage, "output", 0) or 0)
        return input_tokens, output_tokens

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
            from astrbot.core.agent.message import Message

            host_contexts = [
                item if isinstance(item, Message) else Message.model_validate(item)
                for item in (contexts or [])
            ]
            logger.debug(
                f"LLM 调用开始：module={module}, provider={actual_provider_id}"
            )

            llm_resp: "LLMResponse" = await self._call_with_timeout(
                self._context.llm_generate(
                    chat_provider_id=actual_provider_id,
                    prompt=prompt,
                    contexts=host_contexts,
                ),
                timeout_sec,
            )

            response_text = llm_resp.completion_text or ""

            duration_ms = int((time.time() - start_time) * 1000)

            input_tokens, output_tokens = self._extract_usage(llm_resp)

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
            self._record_run_log(
                path="hook",
                module=module,
                provider_id=actual_provider_id or "default",
                prompt=prompt,
                response_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                success=True,
                contexts_count=len(contexts or []),
            )

            logger.info(
                f"LLM 调用成功：module={module}, "
                f"tokens={input_tokens}+{output_tokens}, "
                f"duration={duration_ms}ms"
            )

            return response_text

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            if self._token_stats:
                await self._token_stats.record_failure(module)
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
            self._record_run_log(
                path="hook",
                module=module,
                provider_id=actual_provider_id or "default",
                prompt=prompt,
                response_text="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=f"LLM 调用超时({timeout_sec:.1f}s)",
                contexts_count=len(contexts or []),
            )
            logger.warning(
                f"LLM 调用超时：module={module}, "
                f"timeout={timeout_sec:.1f}s, duration={duration_ms}ms"
            )
            raise

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            if self._token_stats:
                await self._token_stats.record_failure(module)
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
            self._record_run_log(
                path="hook",
                module=module,
                provider_id=actual_provider_id or "default",
                prompt=prompt,
                response_text="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                contexts_count=len(contexts or []),
            )

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
        priority: CallPriority | int | str | None = None,
        queue_timeout: Optional[float] = None,
        total_timeout: Optional[float] = None,
        attempt: int = 1,
        job_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
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
            timeout: Provider 调用超时（秒），None 使用全局配置，<=0 不超时
            queue_timeout: 等待 Governor lease 的超时（秒）
            total_timeout: 排队加 Provider 调用的整体超时（秒）
            priority: 调用优先级，None 时由 module 映射
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

        provider_timeout_sec = self._resolve_call_timeout(timeout)
        queue_timeout_sec = self._resolve_queue_timeout(queue_timeout)
        total_timeout_sec = self._resolve_total_timeout(total_timeout)
        resolved_priority = coerce_priority(priority, module)
        start_time = time.time()
        monotonic_start = time.monotonic()
        call_id = str(uuid.uuid4())
        lease: Optional[LLMLease] = None
        provider_started_at: Optional[float] = None
        call_metadata = dict(metadata or {})

        try:
            logger.debug(
                f"LLM 直接调用开始：module={module}, provider={actual_provider_id}"
            )

            # Dream、Persona 复审等长任务可通过 task-local budget 设置硬
            # 上限。这里位于唯一 Provider 入口，子阶段即使遗漏预算检查也
            # 不可能突破真实调用次数和最小调用间隔。
            call_budget = current_llm_call_budget()
            if call_budget is not None:
                await call_budget.before_call(module)

            if total_timeout_sec is not None:
                queue_timeout_sec = self._bounded_timeout(
                    queue_timeout_sec, total_timeout_sec
                )
            lease = await self.governor.acquire(
                provider_id=actual_provider_id,
                module=module,
                priority=resolved_priority,
                queue_timeout=queue_timeout_sec,
            )

            elapsed = time.monotonic() - monotonic_start
            remaining_total = None
            if total_timeout_sec is not None:
                remaining_total = total_timeout_sec - elapsed
                if remaining_total <= 0:
                    raise asyncio.TimeoutError("LLM 整体超时（排队阶段已耗尽）")
            effective_provider_timeout = self._bounded_timeout(
                provider_timeout_sec, remaining_total
            )
            provider_started_at = time.monotonic()
            llm_resp: "LLMResponse" = await self._call_with_timeout(
                provider.text_chat(
                    prompt=prompt,
                    contexts=contexts or [],
                    system_prompt=system_prompt,
                    image_urls=image_urls,
                ),
                effective_provider_timeout,
            )
            provider_duration_ms = int(
                (time.monotonic() - provider_started_at) * 1000
            )
            await self.governor.record_success(actual_provider_id)

            response_text = llm_resp.completion_text or ""

            duration_ms = int((time.time() - start_time) * 1000)

            input_tokens, output_tokens = self._extract_usage(llm_resp)

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
                priority=resolved_priority.name,
                queue_wait_ms=lease.queue_wait_ms,
                provider_duration_ms=provider_duration_ms,
                attempt=max(1, attempt),
                in_flight_at_start=lease.in_flight_at_start,
                queue_depth_at_start=lease.queue_depth_at_start,
                job_id=job_id,
                metadata={
                    **call_metadata,
                    "provider_call_started": True,
                },
            )
            self._call_logs.append(log)
            self._record_run_log(
                path="direct",
                module=module,
                provider_id=actual_provider_id,
                prompt=prompt,
                response_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                success=True,
                contexts_count=len(contexts or []),
                system_prompt=system_prompt or "",
                image_count=len(image_urls or []),
                priority=resolved_priority.name,
                queue_wait_ms=lease.queue_wait_ms,
                provider_duration_ms=provider_duration_ms,
                in_flight_at_start=lease.in_flight_at_start,
                queue_depth_at_start=lease.queue_depth_at_start,
                attempt=max(1, attempt),
                job_id=job_id,
            )

            logger.info(
                f"LLM 直接调用成功：module={module}, "
                f"tokens={input_tokens}+{output_tokens}, "
                f"duration={duration_ms}ms"
            )

            return response_text

        except LLMCallBudgetExceeded:
            # 本地预算拒绝发生在 lease / Provider 调用之前，不计入 Provider
            # 失败与熔断；只保留独立聚合指标供管理页告警。
            self.governor.record_background_budget_exhausted()
            raise

        except LLMQueueTimeoutError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            if self._token_stats:
                await self._token_stats.record_failure(module)
            error_message = str(e)
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
                error_message=error_message,
                priority=resolved_priority.name,
                attempt=max(1, attempt),
                job_id=job_id,
                metadata={
                    **call_metadata,
                    "provider_call_started": False,
                },
            )
            self._call_logs.append(log)
            self._record_run_log(
                path="direct",
                module=module,
                provider_id=actual_provider_id,
                prompt=prompt,
                response_text="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=error_message,
                contexts_count=len(contexts or []),
                system_prompt=system_prompt or "",
                image_count=len(image_urls or []),
                priority=resolved_priority.name,
                attempt=max(1, attempt),
                job_id=job_id,
            )
            logger.warning(f"LLM 直接调用排队超时：module={module}, error={e}")
            raise

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            provider_duration_ms = (
                int((time.monotonic() - provider_started_at) * 1000)
                if provider_started_at is not None
                else 0
            )
            if lease is not None:
                await self.governor.record_failure(actual_provider_id)
            if self._token_stats:
                await self._token_stats.record_failure(module)
            error_message = "LLM Provider/整体调用超时"
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
                error_message=error_message,
                priority=resolved_priority.name,
                queue_wait_ms=lease.queue_wait_ms if lease else 0,
                provider_duration_ms=provider_duration_ms,
                attempt=max(1, attempt),
                in_flight_at_start=lease.in_flight_at_start if lease else 0,
                queue_depth_at_start=lease.queue_depth_at_start if lease else 0,
                job_id=job_id,
                metadata={
                    **call_metadata,
                    "provider_call_started": provider_started_at is not None,
                },
            )
            self._call_logs.append(log)
            self._record_run_log(
                path="direct",
                module=module,
                provider_id=actual_provider_id,
                prompt=prompt,
                response_text="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=error_message,
                contexts_count=len(contexts or []),
                system_prompt=system_prompt or "",
                image_count=len(image_urls or []),
                priority=resolved_priority.name,
                queue_wait_ms=lease.queue_wait_ms if lease else 0,
                provider_duration_ms=provider_duration_ms,
                in_flight_at_start=lease.in_flight_at_start if lease else 0,
                queue_depth_at_start=lease.queue_depth_at_start if lease else 0,
                attempt=max(1, attempt),
                job_id=job_id,
            )
            logger.warning(
                f"LLM 直接调用超时：module={module}, duration={duration_ms}ms"
            )
            raise

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            provider_duration_ms = (
                int((time.monotonic() - provider_started_at) * 1000)
                if provider_started_at is not None
                else 0
            )
            if lease is not None:
                await self.governor.record_failure(actual_provider_id)
            if self._token_stats:
                await self._token_stats.record_failure(module)
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
                priority=resolved_priority.name,
                queue_wait_ms=lease.queue_wait_ms if lease else 0,
                provider_duration_ms=provider_duration_ms,
                attempt=max(1, attempt),
                in_flight_at_start=lease.in_flight_at_start if lease else 0,
                queue_depth_at_start=lease.queue_depth_at_start if lease else 0,
                job_id=job_id,
                metadata={
                    **call_metadata,
                    "provider_call_started": provider_started_at is not None,
                },
            )
            self._call_logs.append(log)
            self._record_run_log(
                path="direct",
                module=module,
                provider_id=actual_provider_id,
                prompt=prompt,
                response_text="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                contexts_count=len(contexts or []),
                system_prompt=system_prompt or "",
                image_count=len(image_urls or []),
                priority=resolved_priority.name,
                queue_wait_ms=lease.queue_wait_ms if lease else 0,
                provider_duration_ms=provider_duration_ms,
                in_flight_at_start=lease.in_flight_at_start if lease else 0,
                queue_depth_at_start=lease.queue_depth_at_start if lease else 0,
                attempt=max(1, attempt),
                job_id=job_id,
            )

            logger.error(f"LLM 直接调用失败：module={module}, error={e}")
            raise
        finally:
            if lease is not None:
                await lease.release()

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
        """解析 Provider 实际调用超时秒数。

        优先级：调用方传入 > 显式 llm_provider_timeout_ms > 旧版
        llm_call_timeout_ms > 默认值。

        Args:
            timeout: 调用方传入的超时（秒），None 表示用配置默认

        Returns:
            超时秒数，None 表示不超时
        """
        if timeout is not None:
            return timeout if timeout > 0 else None
        config = get_config()
        timeout_ms = self._configured_timeout_ms(
            config,
            "llm_provider_timeout_ms",
            legacy_key="llm_call_timeout_ms",
            default=60000,
        )
        return self._milliseconds_to_seconds(timeout_ms)

    def _resolve_queue_timeout(
        self, timeout: Optional[float]
    ) -> Optional[float]:
        if timeout is not None:
            return timeout if timeout > 0 else None
        timeout_ms = self._configured_timeout_ms(
            get_config(), "llm_queue_timeout_ms", default=300000
        )
        return self._milliseconds_to_seconds(timeout_ms)

    def _resolve_total_timeout(
        self, timeout: Optional[float]
    ) -> Optional[float]:
        if timeout is not None:
            return timeout if timeout > 0 else None
        timeout_ms = self._configured_timeout_ms(
            get_config(), "llm_total_timeout_ms", default=360000
        )
        return self._milliseconds_to_seconds(timeout_ms)

    @staticmethod
    def _configured_timeout_ms(
        config,
        key: str,
        *,
        default: int,
        legacy_key: Optional[str] = None,
    ) -> int:
        """读取新超时键，并兼容旧 hidden_config 中的显式覆盖。"""

        hidden = {}
        try:
            hidden = config.get_all_hidden()
            if not isinstance(hidden, dict):
                hidden = {}
        except Exception:
            hidden = {}
        if key in hidden:
            value = hidden[key]
        elif legacy_key and legacy_key in hidden:
            value = hidden[legacy_key]
        else:
            value = config.get(key, None)
            if value is None and legacy_key:
                value = config.get(legacy_key, default)
            if value is None:
                value = default
        if isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _milliseconds_to_seconds(timeout_ms: int) -> Optional[float]:
        if timeout_ms <= 0:
            return None
        return timeout_ms / 1000.0

    @staticmethod
    def _bounded_timeout(
        primary: Optional[float], bound: Optional[float]
    ) -> Optional[float]:
        if primary is None:
            return bound
        if bound is None:
            return primary
        return max(0.001, min(primary, bound))

    def _get_provider_instance(self, provider_id: str) -> Optional["Provider"]:
        """获取 Provider 实例

        通过 context.get_provider_by_id() 获取 Provider 实例。

        Args:
            provider_id: Provider ID

        Returns:
            Provider 实例，不可用时返回 None
        """
        from astrbot.core.provider.provider import Provider

        try:
            if hasattr(self._context, "get_provider_by_id"):
                provider = self._context.get_provider_by_id(provider_id)
                if isinstance(provider, Provider):
                    return provider

            if hasattr(self._context, "provider_manager"):
                provider_manager = self._context.provider_manager
                if hasattr(provider_manager, "inst_map"):
                    provider = provider_manager.inst_map.get(provider_id)
                    return provider if isinstance(provider, Provider) else None
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
            "dream_knowledge_induction": "scheduled_tasks.provider",
            "dream_pruning_confirm": "scheduled_tasks.provider",
            "image_parsing": "l1_buffer.image_parsing.provider",
            "profile_analysis": "profile.analysis_provider",
            "l2_query_rewrite": "l2_query_rewrite_provider",
            "learning_review": "learning.review_provider",
            "learning_dialogue_review": "learning.review_provider",
            "learning_persona_review": "learning.review_provider",
            "learning_jargon_review": "learning.review_provider",
            "persona_evolution_analysis": "persona_evolution.provider",
            "persona_evolution_generate": "persona_evolution.provider",
            "persona_evolution_review": "persona_evolution.review_provider",
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

    def _record_run_log(
        self,
        *,
        path: str,
        module: str,
        provider_id: str,
        prompt: str,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int,
        success: bool,
        error_message: str = "",
        contexts_count: int = 0,
        system_prompt: str = "",
        image_count: int = 0,
        priority: str = "",
        queue_wait_ms: int = 0,
        provider_duration_ms: int = 0,
        in_flight_at_start: int = 0,
        queue_depth_at_start: int = 0,
        attempt: int = 1,
        job_id: str = "",
    ) -> None:
        """写入统一运行日志（llm_call 类型），失败不影响主流程"""
        try:
            path_label = {
                "direct": "直接调用",
                "framework": "主管线调用",
            }.get(path, "钩子调用")
            status = "成功" if success else "失败"
            get_run_log_manager().record(
                "llm_call",
                f"{module} {path_label}{status}",
                success=success,
                module=module,
                path=path,
                provider_id=provider_id,
                prompt=prompt,
                prompt_chars=len(prompt),
                system_prompt=system_prompt,
                contexts_count=contexts_count,
                image_count=image_count,
                response=response_text,
                response_chars=len(response_text),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                priority=priority,
                queue_wait_ms=queue_wait_ms,
                provider_duration_ms=provider_duration_ms,
                in_flight_at_start=in_flight_at_start,
                queue_depth_at_start=queue_depth_at_start,
                attempt=attempt,
                job_id=job_id,
                error=error_message,
            )
        except Exception:
            pass

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

    async def acquire_framework_lease(
        self,
        *,
        module: str,
        provider_id: str,
        queue_timeout: Optional[float] = None,
    ) -> LLMLease:
        """在主管线真正发给 Provider 前取得交互 lease。

        必须由 ``on_llm_request`` 在所有插件预处理完成后调用，避免查询改写等
        direct 调用在持有 framework permit 时发生嵌套死锁。
        """

        if not self._is_available:
            raise RuntimeError("LLMManager 未初始化")
        queue_timeout_sec = self._resolve_queue_timeout(queue_timeout)
        provider_timeout = self._resolve_call_timeout(None)
        total_timeout = self._resolve_total_timeout(None)
        watchdog_base = provider_timeout or total_timeout or 300.0
        return await self.governor.acquire(
            provider_id=provider_id,
            module=module,
            priority=CallPriority.INTERACTIVE,
            queue_timeout=queue_timeout_sec,
            lease_timeout=watchdog_base + 10.0,
        )

    async def release_framework_lease(
        self,
        lease_id: str,
        *,
        provider_id: str,
        success: bool,
    ) -> bool:
        """结算并释放主管线 lease；重复响应或 watchdog 后调用是安全的。"""

        if not lease_id or self._governor is None:
            return False
        if success:
            await self._governor.record_success(provider_id)
        else:
            await self._governor.record_failure(provider_id)
        return await self._governor.release(lease_id)

    def get_governor_metrics(self) -> Dict[str, Any]:
        if self._governor is None:
            return {
                "in_flight": 0,
                "queue_depth": 0,
                "queue_depth_by_priority": {},
            }
        return self._governor.get_metrics()

    def record_singleflight_join(self) -> None:
        if self._governor is not None:
            self._governor.record_singleflight_join()

    def record_retry(self) -> None:
        if self._governor is not None:
            self._governor.record_retry()

    async def record_framework_response(
        self,
        *,
        module: str,
        provider_id: str,
        response: "LLMResponse",
        started_at: Optional[float] = None,
        prompt: str = "",
        priority: str = CallPriority.INTERACTIVE.name,
        queue_wait_ms: int = 0,
        in_flight_at_start: int = 0,
        queue_depth_at_start: int = 0,
    ) -> None:
        """结算由 AstrBot 主管线执行、但由本插件触发的 LLM 响应。

        插话、跟进等回复必须保留 AstrBot 的人格、工具与钩子链，无法通过
        ``generate_direct`` 发起；它们在 ``on_llm_response`` 由本方法统一
        纳入 Token 统计和调用日志。
        """
        if not self._is_available:
            return

        response_text = getattr(response, "completion_text", "") or ""
        input_tokens, output_tokens = self._extract_usage(response)
        duration_ms = int(
            max(0.0, time.time() - (started_at or time.time())) * 1000
        )
        if self._token_stats:
            await self._token_stats.record_completion(
                module=module,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )

        log = CallLog(
            call_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            module=module,
            provider_id=provider_id or "default",
            prompt=self._truncate_text(prompt, 500),
            response=self._truncate_text(response_text, 500),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            success=True,
            priority=priority,
            queue_wait_ms=queue_wait_ms,
            provider_duration_ms=duration_ms,
            in_flight_at_start=in_flight_at_start,
            queue_depth_at_start=queue_depth_at_start,
            metadata={"path": "framework"},
        )
        self._call_logs.append(log)
        self._record_run_log(
            path="framework",
            module=module,
            provider_id=provider_id or "default",
            prompt=prompt,
            response_text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            success=True,
            priority=priority,
            queue_wait_ms=queue_wait_ms,
            provider_duration_ms=duration_ms,
            in_flight_at_start=in_flight_at_start,
            queue_depth_at_start=queue_depth_at_start,
        )

    async def record_framework_failure(
        self,
        *,
        module: str,
        provider_id: str,
        error_message: str,
        started_at: Optional[float] = None,
    ) -> None:
        """供框架异常钩子可用时结算主管线失败调用。"""
        if not self._is_available:
            return
        duration_ms = int(
            max(0.0, time.time() - (started_at or time.time())) * 1000
        )
        if self._token_stats:
            await self._token_stats.record_completion(
                module,
                input_tokens=0,
                output_tokens=0,
                success=False,
            )
        self._call_logs.append(
            CallLog(
                call_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                module=module,
                provider_id=provider_id or "default",
                prompt="",
                response="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error_message=error_message,
                metadata={"path": "framework"},
            )
        )

    async def record_framework_attempt(self, module: str) -> None:
        """在 AstrBot 主管线开始执行前登记插件触发的请求。"""
        if self._is_available and self._token_stats:
            await self._token_stats.record_attempt(module)

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
                "successful_calls": 0,
                "failed_calls": 0,
                "pending_calls": 0,
            }

        usage = await self._token_stats.get_stats(module)
        return {
            "module": module,
            "total_input_tokens": usage.total_input_tokens,
            "total_output_tokens": usage.total_output_tokens,
            "total_calls": usage.total_calls,
            "successful_calls": usage.successful_calls,
            "failed_calls": usage.failed_calls,
            "pending_calls": usage.pending_calls,
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
                "successful_calls": usage.successful_calls,
                "failed_calls": usage.failed_calls,
                "pending_calls": usage.pending_calls,
            }
            for module, usage in all_stats.items()
        }

    async def get_token_stats_for_days(
        self, days: int = 7
    ) -> Dict[str, Dict[str, Any]]:
        """获取包含今天在内的最近若干自然日 Token 统计。"""
        if not self._token_stats:
            return {}

        all_stats = await self._token_stats.get_stats_for_days(days)
        return {
            module: {
                "total_input_tokens": usage.total_input_tokens,
                "total_output_tokens": usage.total_output_tokens,
                "total_calls": usage.total_calls,
                "successful_calls": usage.successful_calls,
                "failed_calls": usage.failed_calls,
                "pending_calls": usage.pending_calls,
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
                "priority": log.priority,
                "queue_wait_ms": log.queue_wait_ms,
                "provider_duration_ms": log.provider_duration_ms,
                "attempt": log.attempt,
                "in_flight_at_start": log.in_flight_at_start,
                "queue_depth_at_start": log.queue_depth_at_start,
                "dedupe_hit": log.dedupe_hit,
                "retry_reason": log.retry_reason,
                "job_id": log.job_id,
                "success": log.success,
                "error_message": log.error_message,
                "metadata": log.metadata,
            }
            for log in logs
        ]

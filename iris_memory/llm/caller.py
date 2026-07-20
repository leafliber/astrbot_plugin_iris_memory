"""
Iris Chat Memory - LLM 调用接口

定义 LLM 调用协议接口，供各模块使用。
阶段 5 已实现 LLMManager，总结功能正式可用。

调用方式：
- call(): LLMCaller 协议接口，通过 context.llm_generate() 调用（会触发钩子）
- generate_direct(): 直接调用 Provider（绕过钩子），推荐插件内部使用
"""

from typing import Optional, List, Dict, Any, Protocol, runtime_checkable


@runtime_checkable
class LLMCaller(Protocol):
    """LLM 调用协议接口

    定义统一的 LLM 调用接口，供 Summarizer 等模块使用。

    阶段 5：LLMManager 实现此接口，总结功能正式可用

    Methods:
        call: 调用 LLM 生成响应（通过 context.llm_generate，会触发钩子）
        generate_direct: 直接调用 Provider 生成响应（绕过钩子）

    Examples:
        >>> class SimpleLLMCaller:
        ...     async def call(self, prompt: str, provider: str = "") -> str:
        ...         return "Response"
        ...
        >>> caller = SimpleLLMCaller()
        >>> isinstance(caller, LLMCaller)
        True
    """

    async def call(self, prompt: str, provider: str = "") -> str:
        """调用 LLM 生成响应

        Args:
            prompt: 输入提示词
            provider: 模型提供商（可选，留空使用默认）

        Returns:
            LLM 生成的响应文本

        Raises:
            Exception: LLM 调用失败时抛出
        """
        ...

    async def generate_direct(
        self,
        prompt: str,
        module: str = "default",
        provider_id: Optional[str] = None,
        contexts: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """直接调用 Provider 生成响应（绕过 on_llm_request 钩子）

        Args:
            prompt: 输入提示词
            module: 调用模块标识
            provider_id: Provider ID
            contexts: 上下文消息列表
            system_prompt: 系统提示词
            **kwargs: 其他参数

        Returns:
            LLM 生成的响应文本

        Raises:
            Exception: LLM 调用失败时抛出
        """
        ...

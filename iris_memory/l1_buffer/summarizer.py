"""
Iris Chat Memory - 总结器

负责触发总结逻辑，调用 LLM 生成总结。
阶段 5：LLMManager 实现后，总结功能正式可用
"""

from typing import Optional, TYPE_CHECKING, cast
import json
import re

from iris_memory.core import get_logger
from iris_memory.config import get_config
from .models import ContextMessage, SegmentedMessageQueue

if TYPE_CHECKING:
    from iris_memory.llm import LLMManager

logger = get_logger("summarizer")


class Summarizer:
    """消息总结器

    负责检查触发条件、调用 LLM 生成总结。

    Attributes:
        llm_manager: LLM 调用管理器实例
        provider: 总结使用的模型提供商

    Examples:
        >>> summarizer = Summarizer(llm_manager=llm_manager)
        >>> queue = MessageQueue(group_id="group_123")
        >>> await summarizer.summarize(queue)
    """

    def __init__(self, llm_manager: "LLMManager", provider: str = ""):
        """初始化总结器

        Args:
            llm_manager: LLM 调用管理器实例
            provider: 总结使用的模型提供商（留空使用默认）
        """
        self.llm_manager = llm_manager
        self.provider = provider
        logger.info("总结器已初始化")

    def should_summarize(self, queue: SegmentedMessageQueue) -> bool:
        """检查是否应该触发总结

        三段式队列的触发条件：
        1. L1-3 段已满（队列溢出到缓冲段）
        2. 总 Token 数超过限制

        Args:
            queue: 三段式消息队列

        Returns:
            是否应该触发总结
        """
        if queue.is_full():
            logger.debug(
                f"L1-3 段已满 ({len(queue.segment_3)}/{queue.segment_3_length})，触发总结"
            )
            return True

        config = get_config()
        max_tokens = cast(int, config.get("l1_max_queue_tokens", 4000))
        if queue.total_tokens >= max_tokens:
            logger.debug(
                f"队列 Token 数 {queue.total_tokens} >= {max_tokens}，触发总结"
            )
            return True

        return False

    async def summarize(
        self,
        context_messages: list[ContextMessage],
        target_messages: list[ContextMessage],
    ) -> Optional[str]:
        """总结消息列表

        调用 LLM 生成总结。context_messages 提供完整上下文（L1-1+L1-2+L1-3），
        但只总结 target_messages（L1-2）的内容。

        Args:
            context_messages: 全量上下文消息（L1-1+L1-2+L1-3，辅助理解）
            target_messages: 待总结的目标消息（L1-2，实际总结内容）

        Returns:
            总结文本

        Raises:
            Exception: LLM 调用失败时抛出
        """
        if not target_messages:
            logger.debug("目标消息列表为空，跳过总结")
            return None

        try:
            total_tokens = sum(msg.token_count for msg in target_messages)
            prompt = self._build_summary_prompt(context_messages, target_messages)

            logger.info(
                f"开始总结，上下文 {len(context_messages)} 条，"
                f"目标 L1-2 {len(target_messages)} 条，{total_tokens} tokens"
            )

            summary = await self.llm_manager.generate_direct(
                prompt=prompt,
                module="l1_summarizer",
                provider_id=self.provider if self.provider else None,
            )

            logger.info(f"总结完成，长度：{len(summary)} 字符")
            return summary

        except Exception as e:
            logger.error(f"总结失败：{e}", exc_info=True)
            raise

    def _build_summary_prompt(
        self,
        context_messages: list[ContextMessage],
        target_messages: list[ContextMessage],
    ) -> str:
        """构建总结提示词

        将消息列表转换为总结提示词。context_messages 提供完整对话上下文，
        target_messages 是实际需要总结的 L1-2 段消息。

        采用保守提取策略：宁缺毋滥，不确定的信息降低置信度，
        防止 L2 记忆爆炸增长。

        Args:
            context_messages: 全量上下文消息（L1-1+L1-2+L1-3）
            target_messages: 待总结的目标消息（L1-2）

        Returns:
            总结提示词
        """
        context_formatted = self._format_messages(context_messages)
        target_formatted = self._format_messages(target_messages)

        prompt = f"""分析以下对话，保守地提取记忆信息。

## 完整对话上下文（供理解参考）
{context_formatted}

## 需要总结的对话片段（仅提取此部分的记忆）
{target_formatted}

## 核心原则：宁缺毋滥
- 只提取确信有长期价值的信息，拿不准的降低置信度或不提取
- 一次对话通常只有 1-5 条值得记住的信息，没有则返回空数组

## 提取标准
1. **信息价值**：偏好、重要事实、明确计划、鲜明观点、技能经验等
2. **主体明确**：每条记忆必须用第三人称表述——"张三喜欢孙权"而非"有角色偏好"或"我喜欢孙权"。无法确定主体的不记录
3. **独立完整**：脱离上下文也能理解，不拼接不相关的内容
4. **非即时性**：未来对话仍可能有用。"想吃煎蛋""想去玩"等临时欲望不提取
5. **确定性**：来源明确、表述清晰，非猜测

### 必须排除
寒暄、简短回复、纯问题、一次性指令、纯情绪、确认性回复、模糊提及、推测性内容、闲聊零散细节、缺少主体的属性描述、即时性欲望（想吃/想喝/想去）

### 错误示例（绝对不要这样提取）
- ❌ "我想吃煎蛋8岁" → 第一人称 + 拼接了不相关信息，应分别判断是否值得记录且用第三人称
- ❌ "有角色偏好" → 缺少主体，不知道是谁的偏好
- ❌ "喜欢吃苹果" → 缺少主体，应表述为"张三喜欢吃苹果"
- ❌ "想睡觉" → 即时性欲望，无长期价值

## 置信度
- **high**：用户明确陈述（如"张三是程序员"）
- **medium**：合理推断但未直接确认（如张三讨论多个编程问题→可能对编程感兴趣）
- **low**：模糊或可能变化（如"张三最近在忙"）

## 输出格式
直接输出 JSON（禁止 ```json 包裹、禁止解释文字），第一个字符必须是 {{：

{{
  "memories": [
    {{"content": "张三是Python程序员，正在学习装饰器", "confidence": "high"}},
    {{"content": "李四下周三要去北京出差", "confidence": "high"}}
  ]
}}

无值得记录的信息时输出：{{"memories": []}}

confidence 只能是 high/medium/low。仅提取"需要总结的对话片段"中的记忆。"""

        return prompt

    @staticmethod
    def _format_messages(messages: list[ContextMessage]) -> str:
        """格式化消息列表为文本

        Args:
            messages: 消息列表

        Returns:
            格式化的消息文本
        """
        formatted = []
        for msg in messages:
            if msg.role == "user":
                user_name = msg.metadata.get("user_name") if msg.metadata else None
                if user_name:
                    formatted.append(f"[{user_name}]: {msg.content}")
                else:
                    formatted.append(f"[用户]: {msg.content}")
            else:
                formatted.append(f"[助手]: {msg.content}")
        return "\n".join(formatted)


def _strip_code_fences(text: str) -> str:
    """去除 Markdown 代码块包裹（```json ... ``` 或 ``` ... ```）。

    仅当整段文本就是一个代码块时才剥离，避免误伤包含代码块的混合文本。
    小模型经常把 JSON 输出包在 ```json 围栏里，剥离后可直接 json.loads。
    """
    match = re.match(
        r"^\s*```(?:json|JSON)?\s*\n([\s\S]*?)\n?\s*```\s*$",
        text,
    )
    if match:
        return match.group(1).strip()
    return text


def parse_summary_response(response: str) -> dict:
    """解析总结响应

    从 LLM 响应中提取 JSON 内容。支持两种格式：
    - 新格式：memories 为对象数组，每项包含 content 和 confidence
    - 旧格式：memories 为字符串数组（兼容）

    解析顺序：
    1. 去除 Markdown 代码块围栏（```json ... ```）后直接 json.loads
    2. 上述失败则用正则提取最外层 ``{...}`` 再 json.loads
    3. 仍失败则走文本回退（仅识别 ``- `` 开头的行）

    Args:
        response: LLM 响应文本

    Returns:
        解析后的字典，包含：
        - memories: 记忆列表（统一为对象列表，每项含 content 和 confidence）
        - group_profile: 群聊画像
        - user_profiles: 用户画像字典
        - json_parsed: 是否成功通过 JSON 解析（False 表示走了文本回退）
    """
    result: dict = {
        "memories": [],
        "group_profile": {},
        "user_profiles": {},
        "json_parsed": False,
    }

    if not response:
        return result

    cleaned = _strip_code_fences(response.strip())

    try:
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if not json_match:
                raise
            parsed = json.loads(json_match.group())

        result["json_parsed"] = True

        if "memories" in parsed:
            raw_memories = parsed["memories"]
            normalized: list[dict] = []

            for item in raw_memories:
                if isinstance(item, dict):
                    content = item.get("content", "")
                    confidence = item.get("confidence", "medium")
                    if confidence not in ("high", "medium", "low"):
                        confidence = "medium"
                    if content:
                        normalized.append(
                            {"content": content, "confidence": confidence}
                        )
                elif isinstance(item, str):
                    content = item.lstrip("- ").strip()
                    if content:
                        normalized.append({"content": content, "confidence": "medium"})

            result["memories"] = normalized

        if "group_profile" in parsed:
            result["group_profile"] = parsed["group_profile"]

        if "user_profiles" in parsed:
            result["user_profiles"] = parsed["user_profiles"]

        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"L1 总结 JSON 解析失败: {e}。"
            f"模型可能未按 JSON 格式输出，将尝试文本回退解析。"
            f"如大量出现此警告，建议更换支持 JSON 输出的模型。"
            f"\n--- LLM 原始返回 ---\n{response}\n--- 结束 ---"
        )

    lines = cleaned.split("\n")
    memories: list[dict] = []
    for line in lines:
        line = line.strip()
        if line.startswith("- "):
            content = line[2:].strip()
            if content:
                memories.append({"content": content, "confidence": "medium"})

    if memories:
        result["memories"] = memories

    return result


def confidence_to_float(confidence: str) -> float:
    """将置信度字符串转换为浮点数

    Args:
        confidence: 置信度级别（high/medium/low）

    Returns:
        置信度浮点数
    """
    mapping = {"high": 0.85, "medium": 0.6, "low": 0.35}
    return mapping.get(confidence, 0.5)


def format_memories_for_l2(memories: list[str]) -> str:
    """将记忆列表格式化为 L2 写入格式

    Args:
        memories: 记忆列表

    Returns:
        格式化的记忆文本
    """
    if not memories:
        return ""
    return "\n".join(memories)

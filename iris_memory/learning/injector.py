"""
Iris Chat Memory - 学习上下文注入组装

从学习库中取本群已通过的表达模式、消息命中的黑话解释、
few-shot 对话样例，组装 `## 群聊用语与表达风格` 分小节文本：
- 单条条目截断 learning_inject_max_item_chars（默认 200）；
- 整体超 learning_inject_max_tokens（默认 600）按
  few_shot → pattern → jargon 顺序裁剪；
- 无内容返回 ""。
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from iris_memory.config import get_config
from iris_memory.core import get_logger
from iris_memory.platform import get_adapter
from iris_memory.utils.token_counter import count_tokens
from .storage import LearningStorage
from .jargon import JargonLearner

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("learning.injector")

_SECTION_HEADER = "## 群聊用语与表达风格"


def _truncate(text: str, limit: int) -> str:
    """单条条目按字符上限截断"""
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "…"


async def build_learning_context(
    event: "AstrMessageEvent",
    storage: LearningStorage,
    jargon: JargonLearner,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """组装学习注入文本

    Args:
        event: AstrBot 消息事件
        storage: 学习存储实例
        jargon: 黑话学习器实例
        meta: 运行日志元信息字典（填充 counts/budget，照 llm_request_hook 风格）

    Returns:
        组装好的注入文本；无内容返回 ""
    """
    config = get_config()
    adapter = get_adapter(event)
    session_id = adapter.get_session_id(event)
    group_id = adapter.get_group_id(event) or session_id

    top_n = int(config.get("learning_pattern_top_n", 5) or 5)
    few_shot_max = int(config.get("learning_few_shot_max", 3) or 3)
    max_item_chars = int(config.get("learning_inject_max_item_chars", 200) or 200)
    max_tokens = int(config.get("learning_inject_max_tokens", 600) or 600)

    # 1. 已通过的表达模式 top-N（按命中数），并记录命中
    patterns = storage.get_approved_patterns(group_id, top_n)
    if patterns:
        storage.record_pattern_hit([int(p["id"]) for p in patterns])

    # 2. 用户消息命中的已推断黑话（最多 5 条）
    user_text = getattr(event, "message_str", "") or ""
    jargon_hits = jargon.match_terms(group_id, user_text, max_items=5)

    # 3. 已通过的 few-shot 对话样例（最近 N 条）
    few_shots = storage.get_approved_few_shots(group_id, few_shot_max)

    if not patterns and not jargon_hits and not few_shots:
        if meta is not None:
            meta["skipped"] = "empty"
        return ""

    few_shot_lines: List[str] = []
    for fs in few_shots:
        few_shot_lines.append(
            f"- 用户：{_truncate(fs['user_text'], max_item_chars)}\n"
            f"  回复：{_truncate(fs['bot_text'], max_item_chars)}"
        )
    pattern_lines = [
        f"- [{p['scene']}] {_truncate(p['expression'], max_item_chars)}"
        for p in patterns
    ]
    jargon_lines = [
        f"- {_truncate(j['term'], 20)} = {_truncate(j['meaning'], max_item_chars)}"
        for j in jargon_hits
    ]

    def assemble(
        fs: List[str], pt: List[str], jg: List[str]
    ) -> str:
        """按分小节组装完整文本"""
        parts = [_SECTION_HEADER]
        if jg:
            parts.append("### 群黑话\n" + "\n".join(jg))
        if pt:
            parts.append("### 常用表达\n" + "\n".join(pt))
        if fs:
            parts.append("### 对话风格样例\n" + "\n".join(fs))
        return "\n".join(parts)

    # 整体 token 预算裁剪：按 few_shot → pattern → jargon 顺序丢弃
    dropped = 0
    while True:
        text = assemble(few_shot_lines, pattern_lines, jargon_lines)
        if count_tokens(text) <= max_tokens:
            break
        if few_shot_lines:
            few_shot_lines.pop()
        elif pattern_lines:
            pattern_lines.pop()
        elif jargon_lines:
            jargon_lines.pop()
        else:
            break
        dropped += 1

    if not few_shot_lines and not pattern_lines and not jargon_lines:
        if meta is not None:
            meta["skipped"] = "budget_exceeded"
        return ""

    if meta is not None:
        meta["pattern_count"] = len(pattern_lines)
        meta["jargon_count"] = len(jargon_lines)
        meta["few_shot_count"] = len(few_shot_lines)
        meta["dropped_by_budget"] = dropped
        meta["budget_tokens"] = max_tokens
        meta["used_tokens"] = count_tokens(text)

    return text

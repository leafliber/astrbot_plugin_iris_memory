"""
Iris Chat Memory - 表达模式提取（规则，零 LLM）

从 user→bot 对话对中按规则提取"场景→表达"映射：
- 场景：按触发消息的标点与关键词规则分类（问句/指令/感叹/闲聊）；
- 表达：bot 回复中的口语化句式（开头称谓、语气词结尾、短句）。

同时负责表达模式的衰减淘汰（委托 LearningStorage）。
"""

import re
from typing import List

from iris_memory.core import get_logger
from .storage import LearningStorage

logger = get_logger("learning.expression")

# 语气词结尾（判定口语化句式）
_TONE_SUFFIXES = ("呀", "啊", "呢", "吧", "嘛", "哦", "哈", "啦", "哩")

# 问句特征词
_QUESTION_WORDS = ("吗", "呢", "什么", "怎么", "哪", "谁", "为啥", "为什么", "多少", "几")

# 指令特征词（祈使语气）
_COMMAND_WORDS = ("帮", "给", "发", "查", "搜", "打", "叫", "算", "翻译", "总结")

# 句子切分
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?\n])|(?<=[；;])")

# 常见动词开头（粗判指令句）
_VERB_START = re.compile(r"^[帮给发查搜打叫算请去把将拿看写说讲唱]")


def classify_scene(user_text: str) -> str:
    """按规则分类触发消息的场景类型

    Args:
        user_text: 用户消息文本

    Returns:
        场景标签：question / command / exclaim / chat
    """
    text = user_text.strip()
    if not text:
        return "chat"

    # 问句：问号结尾或含疑问词
    if "?" in text or "？" in text:
        return "question"
    if any(w in text for w in _QUESTION_WORDS) and (
        text.endswith(("吗", "呢", "么")) or "什么" in text or "怎么" in text
    ):
        return "question"

    # 指令：动词开头或含祈使特征词
    if _VERB_START.match(text) or any(w in text[:4] for w in _COMMAND_WORDS):
        return "command"

    # 感叹：叹号结尾
    if "!" in text or "！" in text:
        return "exclaim"

    return "chat"


def extract_expressions(bot_text: str, max_items: int = 2) -> List[str]:
    """从 bot 回复中提取口语化句式候选

    规则：
    - 开头称谓短句（第一个分句长度 ≤12 且以逗号结尾）；
    - 语气词结尾句（呀/啊/呢/吧/嘛/哦/哈）；
    - 长度 4-40 字符的完整短句；
    每条回复最多取 max_items 条。

    Args:
        bot_text: bot 回复文本
        max_items: 最多提取条数

    Returns:
        表达句式候选列表
    """
    if not bot_text:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(bot_text) if s.strip()]
    candidates: List[str] = []

    for sent in sentences:
        if len(candidates) >= max_items:
            break

        # 开头称谓短句："xxx，" 形式
        m = re.match(r"^([^，。！？!?,]{1,12})[，,]", sent)
        if m:
            opening = m.group(1).strip()
            if 2 <= len(opening) <= 12 and opening not in candidates:
                candidates.append(opening)
                continue

        # 语气词结尾句
        core = sent.rstrip("。！？!?~…")
        if core and core[-1] in _TONE_SUFFIXES and 4 <= len(core) <= 40:
            if core not in candidates:
                candidates.append(core)
                continue

        # 普通完整短句（剥离句末标点）
        plain = sent.rstrip("。！？!?~…")
        if 4 <= len(plain) <= 40 and plain not in candidates:
            candidates.append(plain)

    return candidates[:max_items]


def decay(storage: LearningStorage, decay_days: int, max_count: int) -> int:
    """衰减淘汰表达模式（委托存储层）

    Args:
        storage: 学习存储实例
        decay_days: 衰减天数阈值
        max_count: 模式总量上限

    Returns:
        删除的条数
    """
    removed = storage.decay_patterns(decay_days, max_count)
    if removed:
        logger.info(f"表达模式衰减淘汰 {removed} 条")
    return removed

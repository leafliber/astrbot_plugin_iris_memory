"""
Iris Chat Memory - 梦境阶段2：时间锚定

扫描记忆中的相对时间表达，利用记忆的 timestamp 元数据
将其转换为绝对日期。

例如：「昨天我们决定用Redis」→「2026年5月24日我们决定用Redis」

Features:
    - 正则扫描中文相对时间词
    - 以记忆 timestamp 为基准计算绝对日期
    - LLM 润色确保语句自然
    - 批量处理优化
"""

import re
from datetime import datetime, timedelta
from typing import Optional, cast

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager

logger = get_logger("dream.temporal_anchor")

_RELATIVE_TIME_PATTERN = re.compile(
    r"(大前天|前天|昨天|今天|刚才|刚刚|本周|上周|本月|上个月|去年|前年|今年"
    r"|(\d+)\s*天前|(\d+)\s*周前|(\d+)\s*月前|(\d+)\s*年前|(\d+)\s*小时前|(\d+)\s*分钟前)",
    re.IGNORECASE,
)


def _resolve_relative_time(match: re.Match, base_date: datetime) -> Optional[str]:
    groups = match.groups()
    keyword = groups[0]

    delta_map = {
        "今天": timedelta(days=0),
        "昨天": timedelta(days=1),
        "前天": timedelta(days=2),
        "大前天": timedelta(days=3),
        "刚才": timedelta(minutes=5),
        "刚刚": timedelta(minutes=5),
        "本周": timedelta(days=0),
        "上周": timedelta(weeks=1),
        "本月": timedelta(days=0),
        "上个月": timedelta(days=30),
        "去年": timedelta(days=365),
        "前年": timedelta(days=730),
        "今年": timedelta(days=0),
    }

    if keyword in delta_map:
        target_date = base_date - delta_map[keyword]
        return _format_date(target_date)

    for i, unit in enumerate(["天", "周", "月", "年", "小时", "分钟"], start=1):
        val_str = groups[i]
        if val_str:
            val = int(val_str)
            if unit == "天":
                target_date = base_date - timedelta(days=val)
            elif unit == "周":
                target_date = base_date - timedelta(weeks=val)
            elif unit == "月":
                target_date = base_date - timedelta(days=val * 30)
            elif unit == "年":
                target_date = base_date - timedelta(days=val * 365)
            elif unit == "小时":
                target_date = base_date - timedelta(hours=val)
            elif unit == "分钟":
                target_date = base_date - timedelta(minutes=val)
            else:
                continue
            return _format_date(target_date)

    return None


def _format_date(date: datetime) -> str:
    try:
        return date.strftime("%Y年%-m月%-d日")
    except ValueError:
        return date.strftime("%Y年%m月%d日").replace("年0", "年").replace("月0", "月")


class TemporalAnchorPhase:
    """时间锚定阶段

    扫描记忆中的相对时间表达，转换为绝对日期。
    """

    def __init__(self):
        self._batch_size = 50

    async def execute(
        self,
        l2: "L2MemoryAdapter",
        l3: Optional["L3KGAdapter"],
        llm: Optional["LLMManager"],
        entries: Optional[list] = None,
        persona_id: str = "default",
    ) -> dict:
        config = get_config()
        self._batch_size = cast(int, config.get("dream_temporal_anchor_batch_size"))

        if not llm:
            logger.warning("LLMManager 不可用，跳过时间锚定")
            return {"scanned": 0, "anchored": 0, "updated": 0}

        try:
            if entries is None:
                entries = await l2.get_all_entries(persona_id=persona_id)

            if not entries:
                logger.debug("L2 记忆库为空，跳过时间锚定")
                return {"scanned": 0, "anchored": 0, "updated": 0}

            logger.info(f"开始扫描 {len(entries)} 条记忆的相对时间表达...")

            scanned = 0
            anchored = 0
            updated = 0
            pending_updates = []

            for entry in entries:
                scanned += 1

                timestamp_str = entry.metadata.get("timestamp")
                if not timestamp_str:
                    continue

                try:
                    base_date = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    continue

                matches = list(_RELATIVE_TIME_PATTERN.finditer(entry.content))
                if not matches:
                    continue

                new_content = entry.content
                has_replacement = False

                for match in reversed(matches):
                    absolute_date = _resolve_relative_time(match, base_date)
                    if absolute_date:
                        new_content = (
                            new_content[: match.start()]
                            + absolute_date
                            + new_content[match.end() :]
                        )
                        has_replacement = True

                if not has_replacement:
                    continue

                anchored += 1
                pending_updates.append((entry, new_content))

                if len(pending_updates) >= self._batch_size:
                    updated += await self._flush_updates(pending_updates, l2, llm)
                    pending_updates = []

            if pending_updates:
                updated += await self._flush_updates(pending_updates, l2, llm)

            logger.info(
                f"时间锚定完成：扫描 {scanned}，发现相对时间 {anchored}，更新 {updated}"
            )
            return {"scanned": scanned, "anchored": anchored, "updated": updated}

        except Exception as e:
            logger.error(f"时间锚定失败：{e}", exc_info=True)
            return {"scanned": 0, "anchored": 0, "updated": 0, "error": str(e)}

    async def _flush_updates(
        self, pending: list, l2: "L2MemoryAdapter", llm: "LLMManager"
    ) -> int:
        updated = 0
        for entry, new_content in pending:
            polished = await self._polish_content(entry.content, new_content, llm)
            if polished:
                new_content = polished

            success = await l2.update_content(entry.id, new_content)
            if success:
                updated += 1
                logger.debug(
                    f"时间锚定：{entry.content[:50]}... -> {new_content[:50]}..."
                )
        return updated

    async def _polish_content(
        self, original: str, replaced: str, llm: "LLMManager"
    ) -> Optional[str]:
        try:
            prompt = f"""以下记忆中的相对时间已被替换为绝对日期，请润色使其语句自然流畅。

原文：{original}

替换后：{replaced}

要求：
1. 确保语句通顺自然
2. 不要改变任何事实信息
3. 仅输出润色后的内容，不要添加额外说明

润色后："""

            result = await llm.generate_direct(
                prompt=prompt, module="dream_temporal_anchor"
            )

            if not result or not result.strip():
                return None

            return result.strip()

        except Exception as e:
            logger.debug(f"LLM 润色失败，使用原始替换结果：{e}")
            return replaced

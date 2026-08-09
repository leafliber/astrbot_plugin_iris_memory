"""
Iris Chat Memory - 攒批 LLM 自动审查

对待审的 few_shot 对话对与 expression_pattern 表达模式做批量 LLM 审查：
- 内存待审队列，满 review_batch_size 由组件触发 run_review()；
- 一次调用输出 JSON 数组逐条 verdict；
  [{"id": .., "type": "pair|pattern", "pass": true|false, "reason": ..}]
- 审查维度：低质 / 敏感 / 复读 / 不像真实对话；
- pass → approved，fail → disabled；
- LLM 调用异常重试一次后放弃本轮，解析失败整体保留 pending 下轮再审。
"""

import json
import re
from typing import Any, Dict, List

from iris_memory.config import get_config
from iris_memory.core import get_logger
from iris_memory.llm_modules import LEARNING_DIALOGUE_REVIEW
from .storage import (
    LearningStorage,
    STATUS_APPROVED,
    STATUS_DISABLED,
    STATUS_PENDING,
)

logger = get_logger("learning.reviewer")

_SYSTEM_PROMPT = (
    "你是群聊内容审查助手。审查给定的对话样例与表达句式，"
    "过滤低质量、敏感、复读刷屏、不像真实群聊对话的内容。"
    "只输出 JSON 数组，不要输出其他内容。"
)


class LearningReviewer:
    """学习产物攒批审查器

    内存维护待审队列（pair id + pattern id），
    满批时由组件触发 run_review() 一次性送审。
    """

    def __init__(self, storage: LearningStorage):
        self._storage = storage
        self._pending_pair_ids: List[int] = []
        self._pending_pattern_ids: List[int] = []

    def enqueue(self, pair_id: int) -> None:
        """把一条对话对放入待审队列"""
        if pair_id not in self._pending_pair_ids:
            self._pending_pair_ids.append(pair_id)

    def enqueue_pattern(self, pattern_id: int) -> None:
        """把一条表达模式放入待审队列"""
        if pattern_id not in self._pending_pattern_ids:
            self._pending_pattern_ids.append(pattern_id)

    @property
    def queue_size(self) -> int:
        """当前待审队列长度"""
        return len(self._pending_pair_ids) + len(self._pending_pattern_ids)

    def is_batch_full(self) -> bool:
        """队列是否已达到攒批触发量"""
        batch_size = int(get_config().get("learning.review_batch_size", 10) or 10)
        return self.queue_size >= batch_size

    async def run_review(self, llm_manager: Any) -> bool:
        """执行一轮批量审查（fetch → LLM → 回写 三步的组合）

        组件层为控制锁粒度会直接调用下面三个分步方法
        （LLM await 不持有 db 锁），本方法供无锁场景与测试使用。

        Args:
            llm_manager: LLMManager 实例

        Returns:
            本轮是否有条目被裁决（LLM 调用失败/解析失败返回 False）
        """
        pairs, patterns = self.fetch_pending()
        if not pairs and not patterns:
            return False

        verdicts = await self.request_verdicts(llm_manager, pairs, patterns)
        if verdicts is None:
            return False

        self.apply_verdicts(verdicts, pairs, patterns)
        return True

    def fetch_pending(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """从库中取 pending 的 pairs + patterns（上限 batch_size*2）

        库中无待审条目时同步清空内存队列（队列可能滞后于库，
        如重启后内存队列丢失，以库为准）。

        Returns:
            (pending pairs, pending patterns)
        """
        config = get_config()
        batch_size = config.get_int("learning.review_batch_size", 10) or 10
        fetch_limit = batch_size * 2

        pairs = self._storage.get_pending_pairs(fetch_limit)
        patterns = self._storage.get_pending_patterns(fetch_limit)
        if not pairs and not patterns:
            self._pending_pair_ids.clear()
            self._pending_pattern_ids.clear()
        return pairs, patterns

    async def request_verdicts(
        self,
        llm_manager: Any,
        pairs: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]] | None:
        """拼一次 prompt 让 LLM 逐条裁决（纯 LLM 调用，不读写库）

        LLM 调用异常重试一次后放弃本轮；结果解析失败返回 None，
        两种失败都保留 pending 下轮再审。

        Returns:
            verdict 列表；失败返回 None
        """
        prompt = self._build_prompt(pairs, patterns)

        raw = None
        for attempt in (1, 2):
            try:
                raw = await llm_manager.generate_direct(
                    prompt=prompt,
                    module=LEARNING_DIALOGUE_REVIEW,
                    system_prompt=_SYSTEM_PROMPT,
                    timeout=60,
                )
                break
            except Exception as e:
                logger.warning(f"学习审查 LLM 调用失败（第 {attempt} 次）：{e}")
        if raw is None:
            return None

        verdicts = self._parse_verdicts(raw)
        if verdicts is None:
            logger.warning("学习审查结果解析失败，保留 pending 下轮再审")
        return verdicts

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, limit: int = 200) -> str:
        """审查用文本截断，防止超长样例撑爆 prompt"""
        text = (text or "").replace("\n", " ")
        return text if len(text) <= limit else text[:limit] + "…"

    def _build_prompt(
        self, pairs: List[Dict[str, Any]], patterns: List[Dict[str, Any]]
    ) -> str:
        """拼接批量审查 prompt"""
        lines = [
            "请逐条审查以下群聊学习内容，判断是否适合作为机器人的回复风格参考。",
            "拒绝标准：低质量（无意义/乱码）、敏感（政治/色情/辱骂/隐私）、",
            "复读刷屏、明显不像真实群聊对话。",
            "",
        ]
        for p in pairs:
            lines.append(
                f"[pair id={p['id']}] 用户：{self._truncate(p['user_text'])}"
                f" / 机器人：{self._truncate(p['bot_text'])}"
            )
        for p in patterns:
            lines.append(
                f"[pattern id={p['id']}] 场景={p['scene']}"
                f" 表达：{self._truncate(p['expression'], 100)}"
            )
        lines += [
            "",
            "对每一条输出裁决，格式为 JSON 数组：",
            '[{"id": 数字, "type": "pair或pattern", "pass": true或false, "reason": "一句话理由"}]',
            "只输出 JSON 数组，覆盖上面列出的所有条目。",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_verdicts(raw: str) -> List[Dict[str, Any]] | None:
        """容错解析 LLM 输出的 JSON 数组（提取第一个 [...] 块）

        Returns:
            verdict 列表；解析失败返回 None
        """
        if not raw:
            return None
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, list):
            return None
        return [v for v in data if isinstance(v, dict)]

    def apply_verdicts(
        self,
        verdicts: List[Dict[str, Any]],
        pairs: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
    ) -> None:
        """按裁决结果批量回写状态

        未被 LLM 覆盖到的条目保持 pending 下轮再审。
        回写带 expected_status=pending_review 比较条件：
        LLM 审查期间被管理员改过的条目不再被迟到裁决覆盖。
        """
        pair_ids = {p["id"] for p in pairs}
        pattern_ids = {p["id"] for p in patterns}

        approved_pairs: List[int] = []
        disabled_pairs: List[int] = []
        approved_patterns: List[int] = []
        disabled_patterns: List[int] = []

        for v in verdicts:
            try:
                vid = int(v.get("id"))
            except (TypeError, ValueError):
                continue
            vtype = str(v.get("type") or "")
            passed = bool(v.get("pass"))
            if vtype == "pair" and vid in pair_ids:
                (approved_pairs if passed else disabled_pairs).append(vid)
            elif vtype == "pattern" and vid in pattern_ids:
                (approved_patterns if passed else disabled_patterns).append(vid)

        expected = len(
            approved_pairs + disabled_pairs + approved_patterns + disabled_patterns
        )
        written = 0
        if approved_pairs:
            written += self._storage.update_status(
                "few_shot", approved_pairs, STATUS_APPROVED, STATUS_PENDING
            )
        if disabled_pairs:
            written += self._storage.update_status(
                "few_shot", disabled_pairs, STATUS_DISABLED, STATUS_PENDING
            )
        if approved_patterns:
            written += self._storage.update_status(
                "expression_pattern", approved_patterns, STATUS_APPROVED, STATUS_PENDING
            )
        if disabled_patterns:
            written += self._storage.update_status(
                "expression_pattern", disabled_patterns, STATUS_DISABLED, STATUS_PENDING
            )

        # 内存队列同步移除已裁决条目
        judged = set(approved_pairs + disabled_pairs)
        self._pending_pair_ids = [
            i for i in self._pending_pair_ids if i not in judged
        ]
        judged_p = set(approved_patterns + disabled_patterns)
        self._pending_pattern_ids = [
            i for i in self._pending_pattern_ids if i not in judged_p
        ]

        skipped = expected - written
        logger.info(
            f"学习审查完成：pair 通过 {len(approved_pairs)} 拒绝 {len(disabled_pairs)}，"
            f"pattern 通过 {len(approved_patterns)} 拒绝 {len(disabled_patterns)}"
            + (f"，{skipped} 条因审查期间被修改而跳过回写" if skipped else "")
        )

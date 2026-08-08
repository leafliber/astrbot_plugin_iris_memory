"""
Iris Chat Memory - 黑话学习

词频统计预筛（零 LLM）+ 低频 LLM 含义推断：
- 对清洗后文本做 2-4 字 n-gram 滑窗（仅 CJK 表意/字母数字字符段内），
  停用词过滤，内存计数，攒批刷盘进 jargon 表；
- count 跨过阈值档位（3/6/10/20/40/60/100）时触发一次含义推断：
  可选用 L1 缓冲中最近的相关消息原文做上下文，
  调 generate_direct(module="learning_review") 输出含义+置信度，
  confidence >= 0.5 才落库。
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.config import get_config
from iris_memory.core import get_logger
from .storage import LearningStorage

logger = get_logger("learning.jargon")

# 词频阈值档位：count 跨入新档时触发一次含义（重新）推断
INFER_THRESHOLDS = [3, 6, 10, 20, 40, 60, 100]

# 每累计多少次词条更新刷一次盘
_FLUSH_EVERY = 50

# 推断时从 L1 取的含词消息数上限
_L1_CONTEXT_LIMIT = 30

# CJK 表意文字 / 字母数字字符段（滑窗只在这些段内进行）
_TOKEN_RUN = re.compile(r"[一-鿿A-Za-z0-9]+")

# 内置停用词（常见虚词/代词/语气词，2 字为主；n-gram 滑窗会切出大量此类）
_STOP_WORDS = frozenset(
    "我们 你们 他们 她们 它们 自己 这个 那个 这些 那些 就是 不是 没有 什么 怎么 这样 那样 可以 因为 所以 但是 而且 还是 或者 如果 已经 正在 知道 觉得 感觉 应该 可能 现在 今天 明天 昨天 时候 地方 东西 事情 问题 一下 一点 一些 一样 一直 一个 每个 哪个 这么 那么 真的 确实 好的 好吧 是的 对呀 不是 不要 不用 不能 不会 哈哈 嘿嘿 嘻嘻 啊啊 嗯嗯 哦哦".split()
)

# 纯语气/虚词单字（用于过滤 n-gram 首尾的碎词）
_NOISE_CHARS = frozenset("的吗呢吧啊呀嘛哦哈啦了着过和跟与或又在就都也很还把被让向从往对于啊")


def _is_valid_term(term: str) -> bool:
    """过滤明显无意义的 n-gram 候选

    规则：停用词表命中、纯虚词组合、首尾均为语气字、纯数字。
    """
    if term in _STOP_WORDS:
        return False
    if term.isdigit():
        return False
    # 首尾都是语气虚字的组合（如"了吗呢"）噪声极高
    if term[0] in _NOISE_CHARS and term[-1] in _NOISE_CHARS:
        return False
    # 全部是语气虚字
    if all(c in _NOISE_CHARS for c in term):
        return False
    return True


class JargonLearner:
    """黑话词频统计与含义推断

    内存维护 {(group_id, term): count}，累计 _FLUSH_EVERY 次更新
    或组件 shutdown 时批量 upsert_jargon_count 刷盘。
    """

    def __init__(self, storage: LearningStorage):
        self._storage = storage
        # 内存计数 {(group_id, term): count}
        self._counts: Dict[Tuple[str, str], int] = {}
        # 自上次刷盘以来有变更的词条
        self._dirty: Dict[Tuple[str, str], int] = {}
        # 每词上次推断时的档位（跨档才重推）
        self._inferred_tier: Dict[Tuple[str, str], int] = {}
        self._pending_updates = 0

    # ------------------------------------------------------------------
    # 词频统计
    # ------------------------------------------------------------------

    def load_counts(self) -> None:
        """从库中加载词频计数与推断状态到内存（组件启动时调用）"""
        self._counts = self._storage.load_all_jargon_counts()
        # 已推断过的词条，档位初始化为当前 count 所处档位，
        # 避免重启后立即重复推断
        for row in self._storage.get_jargon_terms_for_inference(INFER_THRESHOLDS):
            key = (row["group_id"], row["term"])
            if row.get("last_inferred_at"):
                self._inferred_tier[key] = self._tier_of(int(row["count"]))
        logger.info(f"已加载 {len(self._counts)} 条黑话词频计数")

    def update_counts(self, group_id: str, text: str) -> None:
        """对一条消息文本做 n-gram 滑窗计数

        Args:
            group_id: 群 ID
            text: 清洗后的消息文本
        """
        if not text:
            return
        for run in _TOKEN_RUN.findall(text):
            run_len = len(run)
            for n in (2, 3, 4):
                if run_len < n:
                    break
                for i in range(run_len - n + 1):
                    term = run[i : i + n]
                    if not _is_valid_term(term):
                        continue
                    key = (group_id, term)
                    self._counts[key] = self._counts.get(key, 0) + 1
                    # _dirty 记录本轮待刷盘的增量
                    self._dirty[key] = self._dirty.get(key, 0) + 1
        self._pending_updates += 1
        if self._pending_updates >= _FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        """把脏词条计数增量批量刷盘，并回写库内最新计数到内存"""
        if not self._dirty:
            self._pending_updates = 0
            return
        for (group_id, term), delta in list(self._dirty.items()):
            try:
                new_total = self._storage.upsert_jargon_count(group_id, term, delta)
                self._counts[(group_id, term)] = new_total
            except Exception as e:
                logger.warning(f"黑话计数刷盘失败 [{group_id}:{term}]：{e}")
        self._dirty.clear()
        self._pending_updates = 0

    # ------------------------------------------------------------------
    # 含义推断
    # ------------------------------------------------------------------

    @staticmethod
    def _tier_of(count: int) -> int:
        """计算 count 所处阈值档位（不超过 count 的最大阈值，未达档为 0）"""
        tier = 0
        for t in INFER_THRESHOLDS:
            if count >= t:
                tier = t
        return tier

    def get_terms_for_inference(self) -> List[Dict[str, Any]]:
        """取需要（重新）推断含义的词条

        规则：count 达到最低档，且从未推断过，
        或当前档位高于上次推断档位（跨档）。
        """
        terms = self._storage.get_jargon_terms_for_inference(INFER_THRESHOLDS)
        result = []
        for item in terms:
            key = (item["group_id"], item["term"])
            current_tier = self._tier_of(int(item["count"]))
            if current_tier <= 0:
                continue
            if item.get("last_inferred_at") is None:
                result.append(item)
            elif current_tier > self._inferred_tier.get(key, 0):
                result.append(item)
        return result

    async def infer(
        self,
        term_info: Dict[str, Any],
        llm_manager: Any,
        l1_buffer: Any = None,
        session_id: str = "",
    ) -> Tuple[str, float] | None:
        """对单个词条调用 LLM 推断含义（纯推断，不读写库）

        learning.jargon_infer_use_l1 开启且 l1_buffer 可用时，
        取最近 _L1_CONTEXT_LIMIT 条含该词的消息原文拼上下文。
        本方法只更新内存推断档位，不写 storage——LLM await 期间
        不应持有组件 db 锁，落库由调用方在锁内完成
        （storage.mark_jargon_inferred）。

        Args:
            term_info: 词条字典（id/group_id/term/count）
            llm_manager: LLMManager 实例
            l1_buffer: L1 缓冲组件（可为 None）
            session_id: 会话 ID（取 L1 上下文用）

        Returns:
            (meaning, confidence)；LLM 失败/解析失败/置信度不足返回 None
        """
        term = term_info["term"]
        group_id = term_info["group_id"]
        config = get_config()

        context_text = ""
        if config.get("learning.jargon_infer_use_l1") and l1_buffer and session_id:
            try:
                messages = l1_buffer.get_context(session_id, max_length=None) or []
                hits = [m.content for m in messages if term in (m.content or "")]
                if hits:
                    context_text = "\n".join(hits[-_L1_CONTEXT_LIMIT:])
            except Exception as e:
                logger.debug(f"取 L1 上下文失败，回退纯词条推断：{e}")

        if context_text:
            prompt = (
                f"以下是群聊中包含词语「{term}」的最近消息：\n"
                f"{context_text}\n\n"
                f"请根据上下文推断「{term}」在这个群里的含义。"
                '只输出 JSON：{"meaning": "含义简述", "confidence": 0到1的置信度}'
            )
        else:
            prompt = (
                f"群聊中高频出现词语「{term}」（已出现 {term_info.get('count', 0)} 次），"
                f"请推断它作为网络用语/群内黑话的可能含义。"
                '只输出 JSON：{"meaning": "含义简述", "confidence": 0到1的置信度}'
                "；如果无法确定含义，confidence 给 0。"
            )

        system_prompt = (
            "你是群聊用语分析助手，擅长根据上下文推断网络用语和群内黑话的含义。"
            "只输出 JSON，不要输出其他内容。"
        )

        try:
            raw = await llm_manager.generate_direct(
                prompt=prompt,
                module="learning_review",
                system_prompt=system_prompt,
                timeout=60,
            )
        except Exception as e:
            logger.warning(f"黑话含义推断 LLM 调用失败 [{term}]：{e}")
            return None

        meaning, confidence = self._parse_infer_result(raw)
        key = (group_id, term)
        self._inferred_tier[key] = self._tier_of(int(term_info.get("count", 0)))

        if confidence < 0.5 or not meaning:
            logger.debug(f"黑话推断置信度不足 [{term}]：confidence={confidence}")
            return None

        return meaning, confidence

    @staticmethod
    def _parse_infer_result(raw: str) -> Tuple[str, float]:
        """容错解析 LLM 输出的 JSON（提取第一个 {...} 块）

        Returns:
            (meaning, confidence)，解析失败 confidence=0
        """
        if not raw:
            return "", 0.0
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return "", 0.0
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return "", 0.0
        meaning = str(data.get("meaning") or "").strip()
        try:
            confidence = float(data.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        return meaning, max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    # 注入侧
    # ------------------------------------------------------------------

    def match_terms(self, group_id: str, text: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """匹配消息文本中命中的已推断黑话

        Args:
            group_id: 群 ID
            text: 用户消息文本
            max_items: 最多返回条数

        Returns:
            命中的词条列表（含 term/meaning），按词长降序（长词优先）
        """
        if not text:
            return []
        active = self._storage.get_active_jargon(group_id)
        hits = [j for j in active if j["term"] in text]
        hits.sort(key=lambda j: len(j["term"]), reverse=True)
        return hits[:max_items]

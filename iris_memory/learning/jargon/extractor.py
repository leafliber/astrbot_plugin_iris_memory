"""暗语候选提取与消息入口的零成本过滤。"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List


_IMAGE_PLACEHOLDER = re.compile(r"\[(?:图:[^\]]*|IMG:[^\]]*)\]", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CQ = re.compile(r"\[CQ:[^\]]+\]")
_MENTION = re.compile(r"@\S+")
_COMMAND = re.compile(r"^\s*[/!！。.#＃]\S*")
_TOKEN = re.compile(r"[一-鿿]+|[A-Za-z][A-Za-z0-9_]{1,31}")

_STOP_WORDS = frozenset(
    "我们 你们 他们 她们 它们 自己 这个 那个 这些 那些 就是 不是 没有 什么 怎么 "
    "这样 那样 可以 因为 所以 但是 而且 还是 或者 如果 已经 正在 知道 觉得 感觉 "
    "应该 可能 现在 今天 明天 昨天 时候 地方 东西 事情 问题 一下 一点 一些 一样 "
    "一直 一个 每个 哪个 这么 那么 真的 确实 好的 好吧 是的 对呀 不要 不用 不能 "
    "不会 哈哈 嘿嘿 嘻嘻 啊啊 嗯嗯 哦哦".split()
)
_NOISE_CHARS = frozenset("的吗呢吧啊呀嘛哦哈啦了着过和跟与或又在就都也很还把被让向从往对于")


@dataclass
class ExtractedMessage:
    normalized: str
    message_hash: str
    observations: List[Dict[str, Any]]


def _valid_term(term: str) -> bool:
    if not term or term in _STOP_WORDS or term.isdigit():
        return False
    if len(set(term)) == 1:
        return False
    if all(char in _NOISE_CHARS for char in term):
        return False
    if term[0] in _NOISE_CHARS and term[-1] in _NOISE_CHARS:
        return False
    return True


class CandidateExtractor:
    def __init__(self, ngram_max: int = 6, max_candidates: int = 64):
        self.ngram_max = max(2, min(12, int(ngram_max)))
        self.max_candidates = max(1, int(max_candidates))

    @staticmethod
    def should_skip(raw_text: str, is_group: bool, is_bot: bool = False) -> bool:
        text = raw_text or ""
        if not is_group or is_bot or _COMMAND.match(text):
            return True
        stripped = _IMAGE_PLACEHOLDER.sub("", text).strip()
        return not stripped

    @staticmethod
    def normalize(raw_text: str) -> str:
        text = unicodedata.normalize("NFKC", raw_text or "").lower()
        text = _IMAGE_PLACEHOLDER.sub(" ", text)
        text = _CQ.sub(" ", text)
        text = _URL.sub(" ", text)
        text = _MENTION.sub(" ", text)
        return " ".join(text.split())

    def extract(self, raw_text: str) -> ExtractedMessage:
        normalized = self.normalize(raw_text)
        by_term: Dict[str, Dict[str, Any]] = {}
        for match in _TOKEN.finditer(normalized):
            token = match.group(0)
            if token[0].isascii():
                if _valid_term(token):
                    by_term.setdefault(token, {"term": token, "left": [], "right": []})
                continue
            length = len(token)
            for n in range(2, min(self.ngram_max, length) + 1):
                for index in range(length - n + 1):
                    term = token[index:index + n]
                    if not _valid_term(term):
                        continue
                    item = by_term.setdefault(term, {"term": term, "left": [], "right": []})
                    left = token[index - 1] if index > 0 else "<B>"
                    right_index = index + n
                    right = token[right_index] if right_index < length else "<E>"
                    if left not in item["left"]:
                        item["left"].append(left)
                    if right not in item["right"]:
                        item["right"].append(right)

        # 长词和字符多样性优先，限制超长消息造成的候选爆炸。
        observations = sorted(
            by_term.values(),
            key=lambda item: (len(set(item["term"])), len(item["term"]), item["term"]),
            reverse=True,
        )[: self.max_candidates]
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
        return ExtractedMessage(normalized, digest, observations)

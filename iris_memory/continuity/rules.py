"""规则预筛与语义分类（纯函数，无 I/O）。

Portions adapted from SurfCore (commit 59ec3066, MIT, © 2026 Derpyu520 and
qq-bridge / QSH contributors): the ending / question / request / ack regex
vocabularies and the classify-message structure from
``src/conversation-closure.js`` (classifyMessage), and the "not finished yet"
cue idea from ``dsh/agent-presets/qq-chat-v2/agent.cordis.yml``.
See THIRD_PARTY_NOTICES.md in this package for the full MIT license text.

与 SurfCore 的差异：
- 判定顺序为 wait_start → wait_done → question → request → ending → ack →
  substantive（question 先于 ending，“你睡了吗”是问题而不是道别）。
- wait_start 只认显式说法（“等我说完/我还没说完/先别回”…），
  不把“然后/但是/还有”等连接词当作等待信号。
- request 去掉过宽的“说说/给我/来个/看一下”；ending 的“下了/睡了”只在
  明确主语或整句时生效（“外面下了好大雨”不是道别）。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "WAIT_START", "WAIT_DONE", "QUESTION", "REQUEST", "ENDING", "ACK",
    "SUBSTANTIVE", "SILENCE", "CATEGORIES",
    "classify", "is_ack_only", "is_ending", "is_question", "is_request",
    "wants_no_reminder", "wants_reminder", "declines_topic",
    "signals_done", "signals_cancel", "signals_reschedule", "signals_result",
    "has_item_cue", "detect_bot_promise",
]

WAIT_START = "wait_start"
WAIT_DONE = "wait_done"
QUESTION = "question"
REQUEST = "request"
ENDING = "ending"
ACK = "ack"
SUBSTANTIVE = "substantive"
SILENCE = "silence"
CATEGORIES = (WAIT_START, WAIT_DONE, QUESTION, REQUEST, ENDING, ACK, SUBSTANTIVE, SILENCE)

# 结尾可忽略的语气/标点（NFKC 规范化后）
_TAIL = r"[呀啦哦喔呢吧啊嘛的滴嘞~。!?…,、.\s]*"
_NEG = r"(?<![没未不别])"

# ---------------------------------------------------------------------------
# 显式等待
# ---------------------------------------------------------------------------
_WAIT_START_RE = re.compile(
    r"等我(?:把话)?说完|我还没说完|我话还没说完|先别回|先别急着回|"
    r"等(?:下|一下|会儿?|等)我(?:再)?说完|让我说完|听我说完|我先说完"
)
_WAIT_DONE_RE = re.compile(
    r"(?<![没你])说完了(?![吗没么嘛?])|我讲完了|好了你说|你说吧|就这些|就这样了你|"
    r"你看看|你怎么看|你觉得呢|你觉得怎么样"
)

# ---------------------------------------------------------------------------
# 问题 / 请求（改编自 SurfCore QUESTION_WORD_RE / REQUEST_RE）
# ---------------------------------------------------------------------------
_QUESTION_MARK_RE = re.compile(r"\?")
_QUESTION_WORD_RE = re.compile(
    r"(?:吗|嘛|么|咋|怎么|为什么|为啥|什么|啥|哪个|哪里|哪儿|谁|多少|几点|几号|"
    r"是不是|有没有|能不能|可不可以|要不要|好不好|对不对|行不行|在不在|忙不忙|"
    r"睡了吗|吃了吗|了没有?|(?:你|那你|那|他|她)呢)"
    r"[。!…~\s]*$"
)
_REQUEST_RE = re.compile(
    r"帮我|帮忙|麻烦你|请你|能不能帮|可以帮我|你帮|教我|告诉我|讲讲|发我|"
    r"推荐|查一下|搜一下|找一下|写一个|写个|弄一个|整一个"
)

# ---------------------------------------------------------------------------
# 收尾（改编自 SurfCore END_ANY_RE / END_STRONG_RE）
# ---------------------------------------------------------------------------
_END_WORDS = (
    r"晚安|拜拜|拜|再见|886|88|先这样|先这么着|先撤|溜了|去忙了?|下了|睡了|我睡了|先睡了|"
    r"去睡了|睡觉了|不聊了|回头聊|改天聊|明天聊|下次聊|明天见|到此为止|好梦|做个好梦|"
    r"早点睡|早点休息|去睡吧|睡吧|bye|good ?night|gn"
)
_END_STRONG_RE = re.compile(rf"^(?:(?:{_END_WORDS}){_TAIL})+$")
# 句内出现即可（≤14 字时视为收尾）；不含易误伤的“下了/睡了/拜/88/gn”
_END_ANY_RE = re.compile(
    r"晚安|拜拜|再见|886|先这样|先这么着|先撤|溜了|去忙了|我下了|先下了|我睡了|先睡了|"
    r"去睡了|睡觉了|不聊了|回头聊|改天聊|明天聊|下次聊|明天见|到此为止|好梦|早点休息|早点睡"
)
_END_ANY_MAX_LEN = 14

# ---------------------------------------------------------------------------
# 应答 / 笑声（改编自 SurfCore ACK_RE）
# ---------------------------------------------------------------------------
_ACK_TOKEN = (
    r"嗯+|恩+|哦+|噢+|喔+|嗷+|好+|好的|好吧|好滴|好哒|行+|行吧|可以|收到|了解|明白|懂了|"
    r"知道了|ok|okay|okk+|没问题|谢谢|多谢|感谢|辛苦了|哈+|呵+|嘿+|嘻+|h{2,}|lol|xswl|"
    r"草|艹|笑死|笑死我了|绝了|666+|在|嗯哼|确实|对的|对|是啊|是的|是|成|噗+|呜+|啊+|诶|欸"
)
_ACK_RE = re.compile(rf"^(?:(?:{_ACK_TOKEN}){_TAIL})+$")

# ---------------------------------------------------------------------------
# 事项相关线索
# ---------------------------------------------------------------------------
_NO_REMIND_RE = re.compile(
    r"不用提醒|别提醒|不要提醒|不需要提醒|不必提醒|用不着提醒|甭提醒|无需提醒|"
    r"只是记着|只记着|记着就行|记着就好|记下就行|记一下就行|知道就行|先记着就行|"
    r"不用叫我|不用问我"
)
_REMIND_RE = re.compile(
    r"提醒我|提醒一下我|记得提醒|到时候叫我|到时候问我|到时候喊我|到时候提醒|"
    r"记得叫我|记得问我"
)
_DECLINE_RE = re.compile(
    r"不想(?:聊|提|说)(?![谎话])|别提了|别再提|不要再提|不要再问|别再问|别问了|"
    r"换个话题|不聊这个"
)
_DONE_RE = re.compile(
    rf"{_NEG}(?:交了|交上去了|搞定|做完了|完成了|结束了|考完了|弄好了|写完了|办完了|"
    rf"忙完了|弄完了|改完了|出结果了|"
    rf"(?:交|弄好|做好|写好|搞好|办好|做完|写完|弄完|忙完|改完|考完|办完|交上去)[啦咯喽])(?![吗没么嘛?])"
)
_CANCEL_RE = re.compile(r"取消了|不去了|不弄了|算了不|不做了|不考了|不办了|不交了|黄了|泡汤了")
_RESCHEDULE_RE = re.compile(r"改到|推迟|延期|改成|挪到|提前到|改期|往后推|推到")
_TIME_CUE_RE = re.compile(
    r"今天|今晚|今早|明天|明早|明晚|后天|昨天|前天|周[一二三四五六日天末]|星期|礼拜|"
    r"下周|这周|本周|下个?月|月底|月初|[0-9一二三四五六七八九十]+[号日]|"
    r"[0-9一二两三四五六七八九十]+点|上午|中午|下午|傍晚|晚上|凌晨|早上|"
    r"过几天|过两天|这几天|最近|到时候"
)
_PLAN_CUE_RE = re.compile(
    r"打算|准备|计划|约了|安排|要去|得去|要交|交报告|截止|ddl|deadline|due|考试|面试|"
    r"开会|答辩|预约|报名|出差|旅行|旅游|复诊|体检|看病|搬家|比赛|演出|航班|火车|"
    r"记着|记一下|别忘"
)
_RESULT_CUE_RE = re.compile(r"结果|成绩|出分|通过|没过|挂了|过了|录取|offer|中了|被拒")
# 结果已出（只用于放行 pending_result 的 complete）
_RESULT_OUT_RE = re.compile(
    rf"{_NEG}(?:通过了|过了|没过|挂了|录取了|被录取|拿到offer|中了|被拒|出分了|出成绩了|"
    rf"出结果了|结果出来了|成绩出来了|有结果了)(?![吗没么嘛?])"
)

# ---------------------------------------------------------------------------
# Frisk 自己的承诺
# ---------------------------------------------------------------------------
_BOT_PROMISE_RE = re.compile(
    r"我(?:帮你|给你|替你)记着|我(?:先)?记(?:下|住)了|我记着了|我会记(?:得|住|着)|"
    r"我会提醒你|"
    r"到时候我?(?:再)?(?:问问你|问你|提醒你|叫你|找你)|"
    r"回头(?:我)?(?:再)?(?:跟你说|告诉你|问你|找你)|"
    r"等你忙完(?:了)?(?:我)?再(?:问|找你|聊|说)|"
    r"我(?:晚点|待会儿?|等会儿?)(?:再)?(?:问你|找你|跟你说|告诉你)|"
    r"下次(?:我)?(?:再)?问你"
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _norm(text: str | None) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text)).strip().lower()


def _is_symbolic_only(t: str) -> bool:
    """非空且只含标点/符号/表情/空白/组合字符。"""
    if not t:
        return False
    for ch in t:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S", "Z", "M") or cat == "Cf" or ch.isspace():
            continue
        return False
    return True


def _question(t: str) -> bool:
    return bool(_QUESTION_MARK_RE.search(t) or _QUESTION_WORD_RE.search(t))


def _ending_raw(t: str) -> bool:
    if _END_STRONG_RE.match(t):
        return True
    return len(t) <= _END_ANY_MAX_LEN and bool(_END_ANY_RE.search(t))


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def classify(text: str | None, *, has_media: bool = False) -> str:
    """把一条消息归入 CATEGORIES 之一。

    顺序：wait_start → wait_done → question → request → ending → ack → substantive。
    空文本：有媒体为 substantive，否则 silence。
    """
    t = _norm(text)
    if not t:
        return SUBSTANTIVE if has_media else SILENCE
    if _WAIT_START_RE.search(t):
        return WAIT_START
    if _WAIT_DONE_RE.search(t):
        return WAIT_DONE
    if _question(t):
        return QUESTION
    if _REQUEST_RE.search(t):
        return REQUEST
    if _ending_raw(t):
        return ENDING
    if _ACK_RE.match(t) or _is_symbolic_only(t):
        return ACK
    return SUBSTANTIVE


def is_question(text: str | None) -> bool:
    t = _norm(text)
    return bool(t) and _question(t)


def is_request(text: str | None) -> bool:
    t = _norm(text)
    return bool(t) and bool(_REQUEST_RE.search(t))


def is_ending(text: str | None) -> bool:
    """道别/收尾语（问题优先：“你睡了吗”不是收尾）。可用于主人消息与 Frisk 回复。"""
    t = _norm(text)
    return bool(t) and not _question(t) and _ending_raw(t)


def is_ack_only(text: str | None) -> bool:
    """纯应答/笑声/表情/标点（“哈哈”“嗯”“好”“ok”“👍”“。。”）。"""
    t = _norm(text)
    return bool(t) and bool(_ACK_RE.match(t) or _is_symbolic_only(t))


def wants_no_reminder(text: str | None) -> bool:
    return bool(_NO_REMIND_RE.search(_norm(text)))


def wants_reminder(text: str | None) -> bool:
    """明确要求提醒；与“不用提醒”同时出现时以“不用提醒”为准。"""
    t = _norm(text)
    return bool(_REMIND_RE.search(t)) and not _NO_REMIND_RE.search(t)


def declines_topic(text: str | None) -> bool:
    return bool(_DECLINE_RE.search(_norm(text)))


def signals_done(text: str | None) -> bool:
    return bool(_DONE_RE.search(_norm(text)))


def signals_cancel(text: str | None) -> bool:
    return bool(_CANCEL_RE.search(_norm(text)))


def signals_reschedule(text: str | None) -> bool:
    return bool(_RESCHEDULE_RE.search(_norm(text)))


def signals_result(text: str | None) -> bool:
    """结果已出（通过/没过/录取/出成绩了…）；疑问形式不算。"""
    return bool(_RESULT_OUT_RE.search(_norm(text)))


def has_item_cue(text: str | None) -> bool:
    """是否含事项线索（时间/计划/结果/记着/提醒/拒绝/完成/取消/改期）。"""
    t = _norm(text)
    if not t or is_ack_only(t):
        return False
    return bool(
        _TIME_CUE_RE.search(t)
        or _PLAN_CUE_RE.search(t)
        or _RESULT_CUE_RE.search(t)
        or _NO_REMIND_RE.search(t)
        or _REMIND_RE.search(t)
        or _DECLINE_RE.search(t)
        or _DONE_RE.search(t)
        or _CANCEL_RE.search(t)
        or _RESCHEDULE_RE.search(t)
    )


def detect_bot_promise(text: str | None) -> str | None:
    """检测 Frisk 已发送文本中的承诺，返回原文中的证据子串；无承诺返回 None。

    证据取自原文（未规范化）以便作为子串校验；仅泛泛的“以后聊”不算承诺。
    """
    if not text:
        return None
    raw = str(text)
    m = _BOT_PROMISE_RE.search(raw)
    if m:
        return m.group(0)
    # 原文含全角/大小写差异时退回规范化文本匹配，再映射回原文
    t = unicodedata.normalize("NFKC", raw)
    m = _BOT_PROMISE_RE.search(t)
    if m and m.group(0) in raw:
        return m.group(0)
    return None

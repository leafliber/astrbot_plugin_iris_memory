"""rules：消息分类与事项线索。"""

import pytest
from astrbot_plugin_iris_memory.iris_memory.continuity import rules
from astrbot_plugin_iris_memory.iris_memory.continuity.rules import classify


@pytest.mark.parametrize(
    "text,expected",
    [
        ("等我说完", "wait_start"),
        ("我还没说完，先别回", "wait_start"),
        ("等下我说完再说", "wait_start"),
        ("说完了", "wait_done"),
        ("好了你说", "wait_done"),
        ("就这些，你怎么看", "wait_done"),
        ("你睡了吗", "question"),
        ("你睡了吗？晚安", "question"),
        ("下了没", "question"),
        ("明天聊啥？", "question"),
        ("为什么", "question"),
        ("帮我查一下天气", "request"),
        ("晚安", "ending"),
        ("晚安～", "ending"),
        ("好的晚安", "ending"),
        ("我先睡了", "ending"),
        ("拜拜", "ending"),
        ("哈哈", "ack"),
        ("哈哈哈哈", "ack"),
        ("嗯嗯", "ack"),
        ("好", "ack"),
        ("OK", "ack"),
        ("👍", "ack"),
        ("。。。", "ack"),
        ("今天去体检了，抽了好多血", "substantive"),
        ("外面下了好大雨", "substantive"),
        ("然后我就走了", "substantive"),
        ("但是", "substantive"),
        ("", "silence"),
        ("   ", "silence"),
    ],
)
def test_classify(text, expected):
    assert classify(text) == expected


def test_classify_media_only_is_substantive():
    assert classify("", has_media=True) == "substantive"


def test_wait_done_not_triggered_by_question_about_finishing():
    assert classify("你说完了吗") == "question"
    assert classify("我还没说完") == "wait_start"


def test_is_ending_question_first():
    assert rules.is_ending("晚安")
    assert rules.is_ending("晚安，做个好梦")
    assert not rules.is_ending("你睡了吗")
    assert not rules.is_ending("今天下了一整天雨，好烦")


@pytest.mark.parametrize("text", ["哈哈", "嗯", "好", "ok", "😂😂", "！！", "嗯嗯好的", "哈哈哈哈哈"])
def test_is_ack_only_true(text):
    assert rules.is_ack_only(text)


@pytest.mark.parametrize("text", ["", "哈哈你也是", "周五交报告", "好累"])
def test_is_ack_only_false(text):
    assert not rules.is_ack_only(text)


def test_no_reminder_with_item_cue():
    t = "周五交报告，先记着，不用提醒"
    assert rules.wants_no_reminder(t) is True
    assert rules.has_item_cue(t) is True
    assert rules.wants_reminder(t) is False


def test_wants_reminder():
    assert rules.wants_reminder("明天下午提醒我交材料")
    assert rules.wants_reminder("到时候问我考得怎么样")
    assert not rules.wants_reminder("不用提醒我")
    assert not rules.wants_reminder("明天交材料")


def test_declines_topic():
    assert rules.declines_topic("不想聊这个")
    assert rules.declines_topic("别提了")
    assert rules.declines_topic("换个话题吧")
    assert not rules.declines_topic("哈哈哈")
    assert not rules.declines_topic("我不想说谎")


def test_done_cancel_reschedule():
    assert rules.signals_done("报告交了")
    assert rules.signals_done("终于考完了")
    assert not rules.signals_done("还没搞定")
    assert not rules.signals_done("你交了吗")
    assert rules.signals_cancel("周末不去了")
    assert rules.signals_cancel("算了不弄了")
    assert rules.signals_reschedule("面试改到下周三")
    assert rules.signals_reschedule("答辩推迟了")
    assert not rules.signals_reschedule("哈哈")


@pytest.mark.parametrize(
    "text",
    ["明天面试", "下周三体检", "成绩出来了", "这事记着", "考完了", "不想聊这个", "改到周五"],
)
def test_has_item_cue_true(text):
    assert rules.has_item_cue(text)


@pytest.mark.parametrize("text", ["哈哈", "嗯嗯", "好的", "今天天气真好看啊"[2:], "我好困"])
def test_has_item_cue_false(text):
    assert not rules.has_item_cue(text)


@pytest.mark.parametrize(
    "text,evidence",
    [
        ("好，我帮你记着，周五别忘了", "我帮你记着"),
        ("我会记得的", "我会记得"),
        ("到时候我问问你结果", "到时候我问问你"),
        ("回头跟你说哈", "回头跟你说"),
        ("等你忙完我再问你", "等你忙完我再问"),
        ("那我会提醒你的", "我会提醒你"),
    ],
)
def test_detect_bot_promise(text, evidence):
    got = rules.detect_bot_promise(text)
    assert got == evidence
    assert got in text


@pytest.mark.parametrize("text", ["以后聊", "那以后聊吧", "晚安", "", None, "我提醒你，带伞"])
def test_detect_bot_promise_none(text):
    assert rules.detect_bot_promise(text) is None


@pytest.mark.parametrize(
    "text,expected",
    [("面试过了！", True), ("没过", True), ("出成绩了", True), ("过了吗", False), ("今天好累", False)],
)
def test_signals_result(text, expected):
    assert rules.signals_result(text) is expected


@pytest.mark.parametrize("text", ["弄好啦", "交啦", "写完啦"])
def test_signals_done_colloquial_tail(text):
    assert rules.signals_done(text)


def test_signals_done_question_not_done():
    assert not rules.signals_done("弄好啦吗")

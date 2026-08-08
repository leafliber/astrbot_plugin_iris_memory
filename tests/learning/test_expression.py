"""表达模式规则提取测试"""

from iris_memory.learning import expression


class TestClassifyScene:
    """场景分类"""

    def test_question_by_mark(self):
        assert expression.classify_scene("今天吃什么？") == "question"
        assert expression.classify_scene("what is this?") == "question"

    def test_question_by_word(self):
        assert expression.classify_scene("你觉得怎么样") == "question"
        assert expression.classify_scene("这是什么") == "question"

    def test_command(self):
        assert expression.classify_scene("帮我查一下天气") == "command"
        assert expression.classify_scene("发一下昨天的记录") == "command"

    def test_exclaim(self):
        assert expression.classify_scene("太好啦！") == "exclaim"
        assert expression.classify_scene("amazing!") == "exclaim"

    def test_chat_default(self):
        assert expression.classify_scene("晚上好各位") == "chat"

    def test_empty(self):
        assert expression.classify_scene("") == "chat"
        assert expression.classify_scene("   ") == "chat"


class TestExtractExpressions:
    """口语化句式提取"""

    def test_opening_address(self):
        exprs = expression.extract_expressions("宝子，这个问题问得好。")
        assert "宝子" in exprs

    def test_tone_suffix_sentence(self):
        exprs = expression.extract_expressions("今天天气真好呀。")
        assert "今天天气真好呀" in exprs

    def test_short_plain_sentence(self):
        exprs = expression.extract_expressions("我倒是觉得可以。")
        assert "我倒是觉得可以" in exprs

    def test_too_long_filtered(self):
        long_text = "这是一个非常长的句子" * 5 + "。"  # 远超 40 字
        assert expression.extract_expressions(long_text) == []

    def test_max_two_per_reply(self):
        text = "好呀。可以吧。行呢。没问题嘛。"
        exprs = expression.extract_expressions(text)
        assert len(exprs) <= 2

    def test_empty_input(self):
        assert expression.extract_expressions("") == []


class TestDecay:
    """衰减委托存储层"""

    def test_decay_delegates(self, storage):
        import time

        p1 = expression_storage_insert(storage)
        old = time.time() - 30 * 86400
        storage._db.execute(
            "UPDATE expression_pattern SET created_at=? WHERE id=?", (old, p1)
        )
        storage._db.commit()
        removed = expression.decay(storage, decay_days=15, max_count=300)
        assert removed == 1

    def test_decay_nothing_to_remove(self, storage):
        assert expression.decay(storage, decay_days=15, max_count=300) == 0


def expression_storage_insert(storage):
    """辅助：插入一条 approved 表达模式"""
    pid = storage.insert_pattern("g1", "chat", "旧表达")
    storage.update_status("expression_pattern", [pid], "approved")
    return pid

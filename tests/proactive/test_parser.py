"""proactive.parser 容错解析测试

覆盖：markdown 围栏剥离、Python True/False 字面量、括号配平扫描、
旧字段名兼容、完全非法输入，以及 action/cooldown 等字段的规范化。
"""

from iris_memory.proactive.parser import extract_json, parse_decision


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"action": "speak"}') == {"action": "speak"}

    def test_markdown_fence_with_lang(self):
        text = '```json\n{"action": "speak"}\n```'
        assert extract_json(text) == {"action": "speak"}

    def test_markdown_fence_without_lang(self):
        text = '```\n{"action": "none"}\n```'
        assert extract_json(text) == {"action": "none"}

    def test_python_bool_literals(self):
        text = '{"drifted": True, "action": "none", "x": False}'
        obj = extract_json(text)
        assert obj == {"drifted": True, "action": "none", "x": False}

    def test_prose_around_json(self):
        text = '好的，这是结果：{"action": "speak", "obs": "test"} 以上。'
        assert extract_json(text) == {"action": "speak", "obs": "test"}

    def test_braces_inside_strings(self):
        text = '前缀 {"message": "用 {curly} 括起来"} 后缀'
        obj = extract_json(text)
        assert obj == {"message": "用 {curly} 括起来"}

    def test_no_json_returns_none(self):
        assert extract_json("完全没有任何 JSON") is None

    def test_non_dict_json_returns_none(self):
        assert extract_json("[1, 2, 3]") is None

    def test_unbalanced_braces_returns_none(self):
        assert extract_json('{"action": "speak"') is None


class TestParseDecision:
    def test_full_fields(self):
        text = (
            '{"action": "speak", "message": "大家好", "obs": "在聊周末", '
            '"watch": ["u1", "u2"], "watch_keywords": ["周末"], '
            '"why": "感兴趣", "drifted": false, "cooldown": 0}'
        )
        d = parse_decision(text, mode="chime_in")
        assert d.should_speak is True
        assert d.action == "speak"
        assert d.mode == "chime_in"
        assert d.message == "大家好"
        assert d.observation == "在聊周末"
        assert d.watch == ["u1", "u2"]
        assert d.watch_keywords == ["周末"]
        assert d.why == "感兴趣"
        assert d.drifted is False
        assert d.cooldown_minutes == 0
        assert d.parse_failed is False

    def test_old_field_names_compat(self):
        text = (
            '{"reply": true, "observation": "旧观察", '
            '"follow_up_users": ["u1"], "follow_up_keywords": ["k1"], '
            '"interest_reason": "旧原因", "topic_drifted": false}'
        )
        d = parse_decision(text, mode="follow_up")
        assert d.should_speak is True
        assert d.observation == "旧观察"
        assert d.watch == ["u1"]
        assert d.watch_keywords == ["k1"]
        assert d.why == "旧原因"
        assert d.drifted is False

    def test_old_reply_false_means_none(self):
        d = parse_decision('{"reply": false}')
        assert d.should_speak is False
        assert d.action == "none"

    def test_completely_invalid_input(self):
        d = parse_decision("这不是 JSON，随便说点什么", mode="initiate")
        assert d.parse_failed is True
        assert d.mode == "initiate"
        assert d.should_speak is False
        assert d.action == "none"
        assert d.message == ""

    def test_empty_input(self):
        d = parse_decision("")
        assert d.parse_failed is True

    def test_drifted_overrides_speak(self):
        text = '{"action": "speak", "message": "x", "drifted": true}'
        d = parse_decision(text)
        assert d.drifted is True
        assert d.action == "none"
        assert d.should_speak is False
        assert d.message == ""

    def test_message_cleared_when_not_speak(self):
        d = parse_decision('{"action": "none", "message": "不该出现"}')
        assert d.message == ""

    def test_action_synonyms(self):
        assert parse_decision('{"action": "reply"}').should_speak is True
        assert parse_decision('{"action": "yes"}').should_speak is True
        assert parse_decision('{"action": "skip"}').should_speak is False
        assert parse_decision('{"action": "no"}').should_speak is False

    def test_cooldown_parsing(self):
        assert parse_decision('{"cooldown": 10}').cooldown_minutes == 10
        assert parse_decision('{"cooldown": "15"}').cooldown_minutes == 15
        assert parse_decision('{"cooldown": -5}').cooldown_minutes == 0
        assert parse_decision('{"cooldown": 999}').cooldown_minutes == 120
        assert parse_decision('{"cooldown": "abc"}').cooldown_minutes == 0

    def test_watch_list_truncated_to_10(self):
        users = [f"u{i}" for i in range(15)]
        d = parse_decision('{"watch": ' + str(users).replace("'", '"') + "}")
        assert len(d.watch) == 10

    def test_watch_filters_empty_strings(self):
        d = parse_decision('{"watch": ["u1", "", "  ", "u2"]}')
        assert d.watch == ["u1", "u2"]

    def test_new_fields_beat_old_fields(self):
        text = '{"action": "speak", "obs": "新", "observation": "旧"}'
        assert parse_decision(text).observation == "新"

    def test_markdown_wrapped_decision(self):
        text = '```json\n{"action": "speak", "obs": "ok"}\n```'
        d = parse_decision(text)
        assert d.should_speak is True
        assert d.parse_failed is False

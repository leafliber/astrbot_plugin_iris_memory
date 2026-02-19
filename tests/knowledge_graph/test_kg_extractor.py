"""
三元组提取器（规则模式）单元测试
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from iris_memory.knowledge_graph.kg_extractor import (
    KGExtractor,
    _guess_node_type,
)
from iris_memory.knowledge_graph.kg_models import (
    KGNodeType,
    KGRelationType,
)
from iris_memory.knowledge_graph.kg_storage import KGStorage


def run(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture
def storage_and_extractor():
    """创建 KGStorage + KGExtractor (rule 模式)"""
    s = KGStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_kg.db"
        run(s.initialize(db_path))
        ext = KGExtractor(storage=s, mode="rule")
        yield s, ext
        run(s.close())


class TestGuessNodeType:
    """实体类型推断测试"""

    def test_person_surname(self):
        assert _guess_node_type("张三") == KGNodeType.PERSON
        assert _guess_node_type("李明") == KGNodeType.PERSON

    def test_person_nickname(self):
        assert _guess_node_type("小王") == KGNodeType.PERSON
        assert _guess_node_type("老张") == KGNodeType.PERSON

    def test_person_english(self):
        assert _guess_node_type("Alice") == KGNodeType.PERSON
        assert _guess_node_type("John Smith") == KGNodeType.PERSON

    def test_location_suffix(self):
        assert _guess_node_type("北京市") == KGNodeType.LOCATION
        assert _guess_node_type("浦东新区") == KGNodeType.LOCATION

    def test_organization_suffix(self):
        assert _guess_node_type("腾讯公司") == KGNodeType.ORGANIZATION

    def test_person_occupation_suffix(self):
        assert _guess_node_type("程序员") == KGNodeType.PERSON
        assert _guess_node_type("工程师") == KGNodeType.PERSON

    def test_unknown(self):
        assert _guess_node_type("快乐") == KGNodeType.UNKNOWN
        assert _guess_node_type("今天") == KGNodeType.UNKNOWN

    def test_concept_keywords(self):
        """概念关键词应识别为 CONCEPT"""
        assert _guess_node_type("编程") == KGNodeType.CONCEPT
        assert _guess_node_type("机器学习") == KGNodeType.CONCEPT
        assert _guess_node_type("深度学习") == KGNodeType.CONCEPT
        assert _guess_node_type("音乐") == KGNodeType.CONCEPT
        assert _guess_node_type("运动") == KGNodeType.CONCEPT

    def test_event_keywords(self):
        """事件关键词应识别为 EVENT"""
        from iris_memory.knowledge_graph.kg_models import KGNodeType
        assert _guess_node_type("会议") == KGNodeType.EVENT
        assert _guess_node_type("毕业") == KGNodeType.EVENT

    def test_object_keywords(self):
        """物品关键词应识别为 OBJECT"""
        from iris_memory.knowledge_graph.kg_models import KGNodeType
        assert _guess_node_type("手机") == KGNodeType.OBJECT
        assert _guess_node_type("猫") == KGNodeType.OBJECT

    def test_person_internet_slang(self):
        """网络用语身份识别"""
        assert _guess_node_type("博主") == KGNodeType.PERSON
        assert _guess_node_type("大佬") == KGNodeType.PERSON
        assert _guess_node_type("萌新") == KGNodeType.PERSON


class TestRuleExtraction:
    """规则提取测试"""

    def test_extract_friend_relation_cn(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="张三和李四是好朋友",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.FRIEND_OF for t in triples)

    def test_extract_colleague_relation(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="张三和李四是同事",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.COLLEAGUE_OF for t in triples)

    def test_extract_likes_relation(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="我喜欢编程",
            user_id="u1",
            sender_name="小明",
        ))
        assert len(triples) >= 1
        # 代词 "我" 应被替换为 "小明"
        assert any(t.subject == "小明" for t in triples)
        assert any(t.relation_type == KGRelationType.LIKES for t in triples)

    def test_extract_dislikes_relation(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="我讨厌加班",
            user_id="u1",
            sender_name="小明",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.DISLIKES for t in triples)

    def test_extract_lives_in(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="张三住在北京",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.LIVES_IN for t in triples)

    def test_extract_works_at(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="小明在腾讯工作",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.WORKS_AT for t in triples)

    def test_extract_boss_relation(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="张三是李四的老板",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.BOSS_OF for t in triples)

    def test_extract_has_relation(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="我有一只猫",
            user_id="u1",
            sender_name="小红",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.HAS for t in triples)

    def test_extract_is_relation(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="小明是一名程序员",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.IS for t in triples)

    def test_extract_en_likes(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="Alice likes programming",
            user_id="u1",
            sender_name="Alice",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.LIKES for t in triples)

    def test_short_text_skipped(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="嗯",
            user_id="u1",
            sender_name="x",
        ))
        assert len(triples) == 0

    def test_empty_text(self, storage_and_extractor):
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="",
            user_id="u1",
            sender_name="x",
        ))
        assert len(triples) == 0

    def test_nodes_created_in_storage(self, storage_and_extractor):
        """提取后节点应已写入存储"""
        s, ext = storage_and_extractor
        run(ext.extract_and_store(
            text="张三和李四是好朋友",
            user_id="u1",
            sender_name="我",
        ))
        stats = run(s.get_stats())
        assert stats["nodes"] >= 2
        assert stats["edges"] >= 1

    def test_edges_with_memory_id(self, storage_and_extractor):
        """边应关联 memory_id"""
        s, ext = storage_and_extractor
        run(ext.extract_and_store(
            text="张三喜欢编程",
            user_id="u1",
            memory_id="mem_123",
            sender_name="我",
        ))
        # 查找张三节点
        nodes = run(s.search_nodes("张三", user_id="u1"))
        assert len(nodes) >= 1
        edges = run(s.get_edges_from(nodes[0].id))
        if edges:
            assert any(e.memory_id == "mem_123" for e in edges)


class TestPronounResolution:
    """代词替换测试"""

    def test_resolve_chinese_pronoun(self, storage_and_extractor):
        _, ext = storage_and_extractor
        assert ext._resolve_pronoun("我", "小明") == "小明"
        assert ext._resolve_pronoun("本人", "小明") == "小明"

    def test_resolve_english_pronoun(self, storage_and_extractor):
        _, ext = storage_and_extractor
        assert ext._resolve_pronoun("I", "Alice") == "Alice"
        assert ext._resolve_pronoun("me", "Alice") == "Alice"

    def test_no_resolve_non_pronoun(self, storage_and_extractor):
        _, ext = storage_and_extractor
        assert ext._resolve_pronoun("张三", "小明") == "张三"

    def test_resolve_without_sender_name(self, storage_and_extractor):
        _, ext = storage_and_extractor
        assert ext._resolve_pronoun("我", None) == "我"


class TestImplicitTriples:
    """隐含关系构建测试"""

    def test_build_implicit(self, storage_and_extractor):
        _, ext = storage_and_extractor
        triples = ext._build_implicit_triples(
            text="今天去找张三玩了",
            sender_name="小明",
            entities=["张三"],
        )
        assert len(triples) == 1
        assert triples[0].subject == "小明"
        assert triples[0].object == "张三"
        assert triples[0].relation_type == KGRelationType.RELATED_TO

    def test_implicit_excludes_sender(self, storage_and_extractor):
        _, ext = storage_and_extractor
        triples = ext._build_implicit_triples(
            text="我是小明",
            sender_name="小明",
            entities=["小明"],
        )
        assert len(triples) == 0

    def test_implicit_max_3(self, storage_and_extractor):
        _, ext = storage_and_extractor
        triples = ext._build_implicit_triples(
            text="...",
            sender_name="小明",
            entities=["a", "b", "c", "d", "e"],
        )
        assert len(triples) <= 3


class TestExpandedRelationPatterns:
    """新增关系模式提取测试"""

    def test_is_friend_of_pattern(self, storage_and_extractor):
        """'XXX是YYY的朋友' 模式"""
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="李四是张三的朋友",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.FRIEND_OF for t in triples)

    def test_dating_relationship(self, storage_and_extractor):
        """'XXX和YYY谈恋爱' 模式"""
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="张三和李四在谈恋爱",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.RELATED_TO for t in triples)

    def test_mentor_relationship(self, storage_and_extractor):
        """'XXX是YYY的导师' 模式"""
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="王教授是张三的导师",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1

    def test_student_relationship(self, storage_and_extractor):
        """'XXX是YYY的学生' 模式"""
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="张三是王教授的学生",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1

    def test_family_of_pattern(self, storage_and_extractor):
        """'XXX是YYY的家人' 补充模式"""
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="李四是张三的爸爸",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.FAMILY_OF for t in triples)

    def test_colleague_of_pattern(self, storage_and_extractor):
        """'XXX是YYY的同事' 补充模式"""
        s, ext = storage_and_extractor
        triples = run(ext.extract_and_store(
            text="李四是张三的同事",
            user_id="u1",
            sender_name="我",
        ))
        assert len(triples) >= 1
        assert any(t.relation_type == KGRelationType.COLLEAGUE_OF for t in triples)

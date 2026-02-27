"""
知识图谱一致性检测器单元测试
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGRelationType,
)
from iris_memory.knowledge_graph.kg_storage import KGStorage
from iris_memory.knowledge_graph.kg_consistency import (
    KGConsistencyDetector,
    ContradictionIssue,
    DanglingEdgeIssue,
    SelfReferenceIssue,
    CycleIssue,
    DuplicateRelationIssue,
    ConsistencyReport,
    CONTRADICTORY_RELATIONS,
    UNIQUE_RELATIONS,
)


@pytest.fixture
def storage():
    """创建临时数据库的 KGStorage 实例"""
    s = KGStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_kg_consistency.db"
        asyncio.get_event_loop().run_until_complete(s.initialize(db_path))
        yield s
        asyncio.get_event_loop().run_until_complete(s.close())


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_node(name: str, user_id: str = "u1", **kwargs) -> KGNode:
    return KGNode(name=name, display_name=name, user_id=user_id, **kwargs)


def _make_edge(
    source_id: str,
    target_id: str,
    user_id: str = "u1",
    relation_type: KGRelationType = KGRelationType.RELATED_TO,
    **kwargs,
) -> KGEdge:
    return KGEdge(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        user_id=user_id,
        **kwargs,
    )


def _insert_edge_raw(storage: KGStorage, edge: KGEdge) -> None:
    """直接向数据库插入边（临时禁用外键约束以模拟悬空边）"""
    assert storage._conn is not None
    storage._conn.execute("PRAGMA foreign_keys=OFF")
    d = edge.to_dict()
    storage._conn.execute(
        """INSERT INTO kg_edges
           (id, source_id, target_id, relation_type, relation_label,
            memory_id, user_id, group_id, persona_id, confidence, weight,
            properties, created_time, updated_time)
           VALUES (:id, :source_id, :target_id, :relation_type, :relation_label,
                   :memory_id, :user_id, :group_id, :persona_id, :confidence, :weight,
                   :properties, :created_time, :updated_time)""",
        d,
    )
    storage._conn.commit()
    storage._conn.execute("PRAGMA foreign_keys=ON")


class TestDetectContradictions:
    """矛盾关系检测测试"""

    def test_no_contradictions_in_clean_graph(self, storage):
        """无矛盾的正常图"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.LIKES)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_contradictions())
        assert len(issues) == 0

    def test_detect_likes_dislikes_contradiction(self, storage):
        """检测喜欢vs讨厌矛盾"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.LIKES)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.DISLIKES)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_contradictions())
        assert len(issues) == 1
        assert issues[0].relation_a in ("likes", "dislikes")
        assert issues[0].relation_b in ("likes", "dislikes")

    def test_detect_boss_subordinate_contradiction(self, storage):
        """检测A既是B的上司又是B的下属"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.BOSS_OF)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.SUBORDINATE_OF)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_contradictions())
        assert len(issues) == 1

    def test_no_contradiction_different_targets(self, storage):
        """不同目标节点的关系不是矛盾"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        n3 = run(storage.upsert_node(_make_node("Charlie")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.LIKES)))
        run(storage.upsert_edge(_make_edge(n1.id, n3.id, relation_type=KGRelationType.DISLIKES)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_contradictions())
        assert len(issues) == 0

    def test_detect_friend_dislikes_contradiction(self, storage):
        """检测朋友关系与讨厌的矛盾"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.FRIEND_OF)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.DISLIKES)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_contradictions())
        assert len(issues) == 1

    def test_empty_graph_no_contradictions(self, storage):
        """空图无矛盾"""
        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_contradictions())
        assert len(issues) == 0


class TestValidateEdgeReferences:
    """悬空边检测测试"""

    def test_valid_edges(self, storage):
        """所有边引用有效节点"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.validate_edge_references())
        assert len(issues) == 0

    def test_detect_missing_source(self, storage):
        """检测源节点缺失"""
        n2 = run(storage.upsert_node(_make_node("Bob")))
        _insert_edge_raw(storage, _make_edge("missing_source", n2.id))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.validate_edge_references())
        assert len(issues) >= 1
        source_missing = [i for i in issues if i.is_source_missing]
        assert len(source_missing) >= 1

    def test_detect_missing_target(self, storage):
        """检测目标节点缺失"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        _insert_edge_raw(storage, _make_edge(n1.id, "missing_target"))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.validate_edge_references())
        assert len(issues) >= 1
        target_missing = [i for i in issues if not i.is_source_missing]
        assert len(target_missing) >= 1

    def test_detect_both_missing(self, storage):
        """检测源和目标都缺失"""
        _insert_edge_raw(storage, _make_edge("missing_a", "missing_b"))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.validate_edge_references())
        assert len(issues) == 2  # 源和目标各一个 issue

    def test_empty_graph_no_dangling(self, storage):
        """空图无悬空边"""
        detector = KGConsistencyDetector(storage)
        issues = run(detector.validate_edge_references())
        assert len(issues) == 0


class TestDetectSelfReferences:
    """自引用检测测试"""

    def test_no_self_references(self, storage):
        """正常图无自引用"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_self_references())
        assert len(issues) == 0

    def test_detect_self_reference(self, storage):
        """检测自引用边"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        run(storage.upsert_edge(_make_edge(n1.id, n1.id, relation_type=KGRelationType.KNOWS)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_self_references())
        assert len(issues) == 1
        assert issues[0].node_id == n1.id
        assert issues[0].relation_type == "knows"

    def test_detect_multiple_self_references(self, storage):
        """检测多个自引用边"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n1.id, relation_type=KGRelationType.KNOWS)))
        run(storage.upsert_edge(_make_edge(n2.id, n2.id, relation_type=KGRelationType.LIKES)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_self_references())
        assert len(issues) == 2


class TestDetectCycles:
    """循环检测测试"""

    def test_no_cycles_in_linear_graph(self, storage):
        """线性图无循环"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        n3 = run(storage.upsert_node(_make_node("Charlie")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))
        run(storage.upsert_edge(_make_edge(n2.id, n3.id)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_cycles())
        assert len(issues) == 0

    def test_detect_2_node_cycle(self, storage):
        """检测 2 节点循环 (A→B→A)"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.LIKES)))
        run(storage.upsert_edge(_make_edge(n2.id, n1.id, relation_type=KGRelationType.LIKES)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_cycles())
        assert len(issues) >= 1

    def test_detect_3_node_cycle(self, storage):
        """检测 3 节点循环 (A→B→C→A)"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        n3 = run(storage.upsert_node(_make_node("Charlie")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))
        run(storage.upsert_edge(_make_edge(n2.id, n3.id)))
        run(storage.upsert_edge(_make_edge(n3.id, n1.id)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_cycles())
        assert len(issues) >= 1

    def test_self_reference_not_detected_as_cycle(self, storage):
        """自引用由 detect_self_references 处理，不在循环检测中"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        run(storage.upsert_edge(_make_edge(n1.id, n1.id)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_cycles())
        assert len(issues) == 0

    def test_empty_graph_no_cycles(self, storage):
        """空图无循环"""
        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_cycles())
        assert len(issues) == 0

    def test_custom_max_cycle_length(self, storage):
        """自定义最大循环长度"""
        n1 = run(storage.upsert_node(_make_node("A")))
        n2 = run(storage.upsert_node(_make_node("B")))
        n3 = run(storage.upsert_node(_make_node("C")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))
        run(storage.upsert_edge(_make_edge(n2.id, n3.id)))
        run(storage.upsert_edge(_make_edge(n3.id, n1.id)))

        detector = KGConsistencyDetector(storage)
        # max_cycle_length=2 不会检测 3 节点循环
        issues = run(detector.detect_cycles(max_cycle_length=2))
        assert len(issues) == 0

        # max_cycle_length=3 会检测到
        issues = run(detector.detect_cycles(max_cycle_length=3))
        assert len(issues) >= 1


class TestRunAllChecks:
    """完整一致性检查测试"""

    def test_clean_graph(self, storage):
        """干净图通过所有检查"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.FRIEND_OF)))

        detector = KGConsistencyDetector(storage)
        report = run(detector.run_all_checks())

        assert isinstance(report, ConsistencyReport)
        assert report.is_consistent
        assert report.total_issues == 0

    def test_mixed_issues(self, storage):
        """混合问题的一致性检查"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))

        # 矛盾关系
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.LIKES)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.DISLIKES)))

        # 自引用
        run(storage.upsert_edge(_make_edge(n1.id, n1.id, relation_type=KGRelationType.KNOWS)))

        detector = KGConsistencyDetector(storage)
        report = run(detector.run_all_checks())

        assert not report.is_consistent
        assert report.total_issues >= 2

    def test_empty_graph_consistent(self, storage):
        """空图一致"""
        detector = KGConsistencyDetector(storage)
        report = run(detector.run_all_checks())
        assert report.is_consistent

    def test_report_summary(self, storage):
        """报告摘要可生成"""
        detector = KGConsistencyDetector(storage)
        report = run(detector.run_all_checks())
        summary = report.summary()
        assert "一致性检测" in summary


class TestContradictoryRelationsMap:
    """矛盾关系映射配置测试"""

    def test_likes_dislikes_are_contradictory(self):
        """LIKES 和 DISLIKES 互为矛盾"""
        assert KGRelationType.DISLIKES in CONTRADICTORY_RELATIONS[KGRelationType.LIKES]
        assert KGRelationType.LIKES in CONTRADICTORY_RELATIONS[KGRelationType.DISLIKES]

    def test_boss_subordinate_are_contradictory(self):
        """BOSS_OF 和 SUBORDINATE_OF 互为矛盾"""
        assert KGRelationType.SUBORDINATE_OF in CONTRADICTORY_RELATIONS[KGRelationType.BOSS_OF]
        assert KGRelationType.BOSS_OF in CONTRADICTORY_RELATIONS[KGRelationType.SUBORDINATE_OF]

    def test_friend_dislikes_contradiction(self):
        """FRIEND_OF 与 DISLIKES 矛盾"""
        assert KGRelationType.DISLIKES in CONTRADICTORY_RELATIONS[KGRelationType.FRIEND_OF]


class TestIssueDataclasses:
    """问题数据类测试"""

    def test_contradiction_issue_auto_description(self):
        issue = ContradictionIssue(
            edge_a_id="e1", edge_b_id="e2",
            source_id="s1234567890", target_id="t1234567890",
            relation_a="likes", relation_b="dislikes",
        )
        assert "矛盾关系" in issue.description
        assert "likes" in issue.description

    def test_dangling_edge_issue_auto_description(self):
        issue = DanglingEdgeIssue(
            edge_id="e1234567890",
            missing_node_id="n1234567890",
            is_source_missing=True,
        )
        assert "悬空边" in issue.description
        assert "源" in issue.description

    def test_self_reference_issue_auto_description(self):
        issue = SelfReferenceIssue(
            edge_id="e1234567890",
            node_id="n1234567890",
            relation_type="knows",
        )
        assert "自引用" in issue.description

    def test_cycle_issue_auto_description(self):
        issue = CycleIssue(
            node_ids=["n1234567890", "n2345678901"],
            edge_ids=["e1", "e2"],
            cycle_length=2,
        )
        assert "循环依赖" in issue.description

    def test_duplicate_relation_issue_auto_description(self):
        issue = DuplicateRelationIssue(
            source_id="s1234567890",
            relation_type="lives_in",
            edge_ids=["e1", "e2"],
            target_ids=["t1", "t2"],
        )
        assert "唯一性关系重复" in issue.description
        assert "lives_in" in issue.description


class TestContradictoryRelationsExtended:
    """扩展矛盾关系映射测试"""

    def test_lives_in_works_at_contradictory(self):
        """LIVES_IN 与 WORKS_AT 矛盾"""
        assert KGRelationType.WORKS_AT in CONTRADICTORY_RELATIONS[KGRelationType.LIVES_IN]

    def test_lives_in_studies_at_contradictory(self):
        """LIVES_IN 与 STUDIES_AT 矛盾"""
        assert KGRelationType.STUDIES_AT in CONTRADICTORY_RELATIONS[KGRelationType.LIVES_IN]

    def test_works_at_studies_at_contradictory(self):
        """WORKS_AT 与 STUDIES_AT 互为矛盾"""
        assert KGRelationType.STUDIES_AT in CONTRADICTORY_RELATIONS[KGRelationType.WORKS_AT]
        assert KGRelationType.WORKS_AT in CONTRADICTORY_RELATIONS[KGRelationType.STUDIES_AT]

    def test_wants_dislikes_contradictory(self):
        """WANTS 与 DISLIKES 矛盾"""
        assert KGRelationType.DISLIKES in CONTRADICTORY_RELATIONS[KGRelationType.WANTS]

    def test_unique_relations_defined(self):
        """唯一性关系集合已定义"""
        assert KGRelationType.LIVES_IN in UNIQUE_RELATIONS
        assert KGRelationType.WORKS_AT in UNIQUE_RELATIONS
        assert KGRelationType.STUDIES_AT in UNIQUE_RELATIONS


class TestDetectDuplicateUniqueRelations:
    """唯一性关系重复检测测试"""

    def test_no_duplicates_in_clean_graph(self, storage):
        """正常图无重复"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Beijing")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.LIVES_IN)))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_duplicate_unique_relations())
        assert len(issues) == 0

    def test_detect_duplicate_lives_in(self, storage):
        """检测同一人住在两个地方"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Beijing")))
        n3 = run(storage.upsert_node(_make_node("Shanghai")))

        _insert_edge_raw(storage, _make_edge(n1.id, n2.id, relation_type=KGRelationType.LIVES_IN))
        _insert_edge_raw(storage, _make_edge(n1.id, n3.id, relation_type=KGRelationType.LIVES_IN))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_duplicate_unique_relations())
        assert len(issues) == 1
        assert issues[0].source_id == n1.id
        assert issues[0].relation_type == "lives_in"
        assert len(issues[0].edge_ids) == 2

    def test_different_sources_no_duplicate(self, storage):
        """不同来源的相同关系类型不构成重复"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        n3 = run(storage.upsert_node(_make_node("Beijing")))

        _insert_edge_raw(storage, _make_edge(n1.id, n3.id, relation_type=KGRelationType.LIVES_IN))
        _insert_edge_raw(storage, _make_edge(n2.id, n3.id, relation_type=KGRelationType.LIVES_IN))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_duplicate_unique_relations())
        assert len(issues) == 0


class TestCycleDetectionOptimized:
    """循环检测优化后的测试"""

    def test_max_issues_limit(self, storage):
        """max_issues 参数限制结果数"""
        # 创建多个短循环
        nodes = []
        for i in range(6):
            nodes.append(run(storage.upsert_node(_make_node(f"N{i}"))))

        # 创建 3 个独立短循环: (0->1->0), (2->3->2), (4->5->4)
        for i in range(0, 6, 2):
            _insert_edge_raw(storage, _make_edge(
                nodes[i].id, nodes[i+1].id,
                relation_type=KGRelationType.RELATED_TO,
            ))
            _insert_edge_raw(storage, _make_edge(
                nodes[i+1].id, nodes[i].id,
                relation_type=KGRelationType.RELATED_TO,
            ))

        detector = KGConsistencyDetector(storage)
        issues = run(detector.detect_cycles(max_cycle_length=3, max_issues=1))
        assert len(issues) <= 1

    def test_run_all_checks_includes_duplicates(self, storage):
        """完整检查包含唯一性关系重复"""
        detector = KGConsistencyDetector(storage)
        report = run(detector.run_all_checks())
        assert hasattr(report, 'duplicate_relations')
        assert isinstance(report.duplicate_relations, list)

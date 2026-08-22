"""L3 按用户/按群删除的回归测试

历史 bug:delete_by_user 仅按 name == user_id 匹配,properties.user_id
标记形态与跨群共享节点(group_id 列被覆盖)均漏删;delete_by_group 同样
不覆盖 properties.group_ids。修复后两类标记均可命中(彻底删除语义)。
"""

import sqlite3

import pytest

from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.l3_kg.models import GraphNode


@pytest.fixture
def l3_adapter():
    adapter = L3KGAdapter()
    adapter._db = sqlite3.connect(":memory:")
    adapter._db.row_factory = sqlite3.Row
    adapter._db.execute("PRAGMA foreign_keys=ON")
    adapter._create_schema_unlocked()
    adapter._is_available = True
    return adapter


async def _add_node(
    adapter: L3KGAdapter,
    *,
    label: str = "Person",
    name: str,
    properties: dict | None = None,
    group_id: str = "g1",
    persona_id: str = "default",
) -> str:
    node = GraphNode(
        id="",
        label=label,
        name=name,
        content="测试节点",
        confidence=0.9,
        group_id=group_id,
        persona_id=persona_id,
        properties=dict(properties or {}),
    )
    node.id = node.generate_id()
    assert await adapter.add_node(node)
    return node.id


def _node_count(adapter: L3KGAdapter) -> int:
    return adapter._db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]


def _names(adapter: L3KGAdapter) -> set:
    return {r[0] for r in adapter._db.execute("SELECT name FROM nodes").fetchall()}


class TestDeleteByUser:
    @pytest.mark.asyncio
    async def test_name_match_still_works(self, l3_adapter):
        """归一化形态(name=user_id)命中"""
        await _add_node(l3_adapter, name="10001")
        assert await l3_adapter.delete_by_user("10001", "g1") == 1
        assert _node_count(l3_adapter) == 0

    @pytest.mark.asyncio
    async def test_properties_user_id_match(self, l3_adapter):
        """name 为昵称但 properties.user_id 已标记的节点命中"""
        await _add_node(
            l3_adapter, name="小张", properties={"user_id": "10001"}
        )
        assert await l3_adapter.delete_by_user("10001", "g1") == 1
        assert _node_count(l3_adapter) == 0

    @pytest.mark.asyncio
    async def test_degraded_node_not_matched(self, l3_adapter):
        """昵称命名且无任何 user_id 标记的节点不命中(需归一化修复先行)"""
        await _add_node(l3_adapter, name="小张", properties={})
        assert await l3_adapter.delete_by_user("10001", "g1") == 0
        assert _node_count(l3_adapter) == 1

    @pytest.mark.asyncio
    async def test_cross_group_shared_node_matched_via_group_ids(self, l3_adapter):
        """跨群共享节点:group_id 列被群 B 覆盖,当前群 A 仅存在于
        properties.group_ids CSV,带群删除仍命中"""
        node_id = await _add_node(
            l3_adapter,
            name="10001",
            group_id="gB",
            properties={"user_id": "10001"},
        )
        # add_node 新建路径会用 node.group_id 覆盖 properties.group_ids,
        # 合并产生的跨群 CSV 直接用 SQL 模拟
        l3_adapter._db_write(
            "UPDATE nodes SET properties = ? WHERE id = ?",
            ('{"user_id": "10001", "group_ids": "gA,gB"}', node_id),
        )

        assert await l3_adapter.delete_by_user("10001", "gA") == 1
        assert _node_count(l3_adapter) == 0

    @pytest.mark.asyncio
    async def test_group_ids_csv_no_substring_false_positive(self, l3_adapter):
        """group_ids CSV 精确匹配:g11 不被 g1 误删"""
        await _add_node(
            l3_adapter,
            name="10001",
            group_id="gB",
            properties={"group_ids": "g11,gB"},
        )

        assert await l3_adapter.delete_by_user("10001", "g1") == 0
        assert _node_count(l3_adapter) == 1

    @pytest.mark.asyncio
    async def test_corrupted_properties_json_tolerated(self, l3_adapter):
        """损坏 JSON 节点不导致整次删除抛错(json_valid 守卫)"""
        node = GraphNode(
            id="person_bad",
            label="Person",
            name="10002",
            content="x",
            confidence=0.9,
            group_id="g1",
            properties={},
        )
        assert await l3_adapter.add_node(node)
        # 直接写入损坏 JSON 绕过 add_node 的序列化
        l3_adapter._db_write(
            "UPDATE nodes SET properties = ? WHERE id = ?",
            ("{invalid json", "person_bad"),
        )
        await _add_node(l3_adapter, name="10001")

        removed = await l3_adapter.delete_by_user("10001", "g1")
        assert removed == 1
        # 损坏节点因 name 不匹配保留,未被误删也未阻断删除
        assert _names(l3_adapter) == {"10002"}

    @pytest.mark.asyncio
    async def test_persona_isolation(self, l3_adapter):
        """persona 过滤只删指定 persona 的节点"""
        await _add_node(l3_adapter, name="10001", persona_id="default")
        await _add_node(l3_adapter, name="10001", persona_id="persona_b")

        assert await l3_adapter.delete_by_user("10001", "g1", persona_id="default") == 1
        remaining = {
            r[0]
            for r in l3_adapter._db.execute(
                "SELECT persona_id FROM nodes"
            ).fetchall()
        }
        assert remaining == {"persona_b"}


class TestDeleteByGroup:
    @pytest.mark.asyncio
    async def test_column_match(self, l3_adapter):
        await _add_node(l3_adapter, name="n1", group_id="g1")
        await _add_node(l3_adapter, name="n2", group_id="g2")
        assert await l3_adapter.delete_by_group("g1") == 1
        assert _names(l3_adapter) == {"n2"}

    @pytest.mark.asyncio
    async def test_group_ids_csv_match(self, l3_adapter):
        """跨群共享节点仅存于 properties.group_ids 时也命中"""
        node_id = await _add_node(l3_adapter, name="shared", group_id="gB")
        l3_adapter._db_write(
            "UPDATE nodes SET properties = ? WHERE id = ?",
            ('{"group_ids": "gA,gB"}', node_id),
        )
        assert await l3_adapter.delete_by_group("gA") == 1
        assert _node_count(l3_adapter) == 0

    @pytest.mark.asyncio
    async def test_persona_isolation(self, l3_adapter):
        await _add_node(l3_adapter, name="n1", group_id="g1", persona_id="default")
        await _add_node(l3_adapter, name="n2", group_id="g1", persona_id="p2")
        assert await l3_adapter.delete_by_group("g1", persona_id="p2") == 1
        assert _names(l3_adapter) == {"n1"}

    @pytest.mark.asyncio
    async def test_edges_cascade(self, l3_adapter):
        """删除节点时关联边级联删除"""
        a = await _add_node(l3_adapter, name="a", label="Person")
        b = await _add_node(l3_adapter, name="b", label="Preference")
        from iris_memory.l3_kg.models import GraphEdge

        edge = GraphEdge(
            source_id=a, target_id=b, relation_type="HAS_PREFERENCE", confidence=0.9
        )
        assert await l3_adapter.add_edge(edge)

        assert await l3_adapter.delete_by_group("g1") == 2
        edge_count = l3_adapter._db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert edge_count == 0

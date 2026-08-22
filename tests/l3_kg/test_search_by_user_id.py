"""复现用户报告:知识图谱界面按用户 ID 搜索

验证 search_nodes 的匹配范围(name/content/properties)在不同数据形态下
能否命中用户 ID:
- 归一化形态(Person name = user_id):命中
- 降级形态(name 为昵称、properties 无 ID 标记):不命中
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
    adapter._create_schema_unlocked()
    adapter._is_available = True
    return adapter


async def _add_person(adapter: L3KGAdapter, name: str, properties: dict) -> str:
    person = GraphNode(
        id="",
        label="Person",
        name=name,
        content="群成员,喜欢猫",
        confidence=0.9,
        group_id="g1",
        properties=dict(properties),
    )
    person.id = person.generate_id()
    assert await adapter.add_node(person)
    return person.id


class TestSearchByUserId:
    @pytest.mark.asyncio
    async def test_canonicalized_person_searchable_by_id(self, l3_adapter):
        """归一化后 name 即 user_id,properties 带 user_id/aliases"""
        await _add_person(
            l3_adapter, "10001", {"user_id": "10001", "aliases": "小张"}
        )

        assert len(await l3_adapter.search_nodes("10001")) == 1
        assert len(await l3_adapter.search_nodes("小张")) == 1

    @pytest.mark.asyncio
    async def test_partial_id_substring_match(self, l3_adapter):
        """部分 ID 子串也能命中"""
        await _add_person(l3_adapter, "123456789", {"user_id": "123456789"})

        assert len(await l3_adapter.search_nodes("12345")) == 1

    @pytest.mark.asyncio
    async def test_degraded_person_not_searchable_by_id(self, l3_adapter):
        """降级形态:name 为昵称且 properties 无任何 ID 标记 → 按 ID 搜索不命中"""
        await _add_person(l3_adapter, "小张", {})

        assert len(await l3_adapter.search_nodes("10001")) == 0
        assert len(await l3_adapter.search_nodes("小张")) == 1

    @pytest.mark.asyncio
    async def test_active_users_property_searchable(self, l3_adapter):
        """properties.active_users 含 user_id 时按 ID 命中"""
        await _add_person(
            l3_adapter, "小张", {"active_users": "10001,10002"}
        )

        assert len(await l3_adapter.search_nodes("10001")) == 1

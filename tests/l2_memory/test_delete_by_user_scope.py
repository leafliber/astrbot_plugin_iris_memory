"""用户级 clear/delete 的作用域回归测试

历史 bug:delete_by_user 只匹配 metadata.active_users,漏删只有 user_id 的
记忆(如 save_memory 工具写入),导致群内 clear 后记忆仍被检索注入。
修复后命中条件为 metadata.user_id == user 或 active_users CSV 包含 user。
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import numpy as np
import pytest

from iris_memory.l2_memory.adapter import L2MemoryAdapter


def _make_adapter() -> L2MemoryAdapter:
    """构建带 mock FAISS + 真实 SQLite 的适配器"""
    adapter = L2MemoryAdapter()
    adapter._is_available = True
    adapter._persist_dir = Path(tempfile.mkdtemp())
    adapter._embedding_dimensions = 8

    mock_index = Mock()
    mock_index.ntotal = 0
    mock_index.d = 8

    def fake_add_with_ids(vectors, ids):
        mock_index.ntotal += len(ids)

    mock_index.add_with_ids = fake_add_with_ids
    mock_index.search = Mock(
        return_value=(np.array([[0.95]]), np.array([[0]]))
    )
    mock_index.remove_ids = Mock()
    adapter._index = mock_index

    adapter._db = adapter._open_db(adapter._persist_dir / "metadata.db")
    adapter._free_list = []
    adapter._dirty = False
    adapter._actual_embedding_model = "test-model"
    adapter._embedding_source = "provider"
    adapter._embedding_provider = None
    return adapter


async def _add(adapter: L2MemoryAdapter, content: str, metadata: dict) -> str:
    adapter._find_similar_unlocked = Mock(return_value=None)
    adapter._embed = AsyncMock(return_value=[[0.1] * 8])
    memory_id = await adapter.add_memory(content, metadata=dict(metadata))
    assert memory_id
    return memory_id


def _remaining_ids(adapter: L2MemoryAdapter) -> set:
    return {
        r[0]
        for r in adapter._db.execute("SELECT memory_id FROM memories").fetchall()
    }


class TestDeleteByUserScopes:
    @pytest.mark.asyncio
    async def test_delete_by_user_removes_tool_saved_memory(self):
        """save_memory 工具记忆(仅 user_id 无 active_users)应被用户级清除删除"""
        adapter = _make_adapter()
        user_id = "10001"
        group_id = "g1"

        tool_mid = await _add(
            adapter,
            "工具保存的记忆:用户喜欢猫",
            {"user_id": user_id, "user_name": "测试用户",
             "group_id": group_id, "source": "tool"},
        )
        summary_mid = await _add(
            adapter,
            "总结记忆:用户养了一只猫",
            {"group_id": group_id, "source": "l1_summary",
             "active_users": user_id},
        )

        removed = await adapter.delete_by_user(user_id, group_id)

        remaining = _remaining_ids(adapter)
        assert removed == 2
        assert tool_mid not in remaining
        assert summary_mid not in remaining

    @pytest.mark.asyncio
    async def test_delete_by_user_hits_multi_user_scene_memory(self):
        """场景记忆 active_users 含多个用户时,删除其中一人即命中"""
        adapter = _make_adapter()
        await _add(
            adapter,
            "场景记忆:两人讨论了搬家",
            {"group_id": "g1", "source": "l1_summary",
             "active_users": "10001,10002"},
        )

        removed = await adapter.delete_by_user("10001", "g1")
        assert removed == 1
        assert _remaining_ids(adapter) == set()

    @pytest.mark.asyncio
    async def test_delete_by_user_skips_global_scope(self):
        """全局共享记忆不属于用户资产,用户级清理跳过"""
        adapter = _make_adapter()
        global_mid = await _add(
            adapter,
            "全局知识",
            {"user_id": "10001", "group_id": "g1", "scope": "global"},
        )

        removed = await adapter.delete_by_user("10001", "g1")
        assert removed == 0
        assert global_mid in _remaining_ids(adapter)

    @pytest.mark.asyncio
    async def test_delete_by_user_no_substring_false_positive(self):
        """active_users 为 CSV 精确匹配,子串用户号不误删"""
        adapter = _make_adapter()
        await _add(
            adapter,
            "他人记忆",
            {"group_id": "g1", "source": "l1_summary",
             "active_users": "100011,10002"},
        )

        removed = await adapter.delete_by_user("10001", "g1")
        assert removed == 0
        assert _remaining_ids(adapter)

    @pytest.mark.asyncio
    async def test_delete_entries_removes_tool_saved_memory(self):
        """web 端按 ID 硬删不受 active_users 影响"""
        adapter = _make_adapter()
        tool_mid = await _add(
            adapter,
            "工具保存的记忆",
            {"user_id": "10001", "group_id": "g1", "source": "tool"},
        )

        assert await adapter.delete_entries([tool_mid])
        assert _remaining_ids(adapter) == set()

    @pytest.mark.asyncio
    async def test_delete_by_group_covers_tool_saved_memory(self):
        """群级清除按 group_id 列匹配,可删除工具记忆"""
        adapter = _make_adapter()
        await _add(
            adapter,
            "工具保存的记忆",
            {"user_id": "10001", "group_id": "g1", "source": "tool"},
        )

        removed = await adapter.delete_by_group("g1")
        assert removed == 1
        assert _remaining_ids(adapter) == set()

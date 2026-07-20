"""
PruningPhase 遗忘清洗测试

测试核心功能：
- L2 记忆淘汰
- L3 图谱节点淘汰
- L3 重复节点合并
- 低置信度标记
- LLM 兜底确认（默认保留策略）
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from iris_memory.dream.pruning import PruningPhase


def _mock_config():
    mock = Mock()
    mock.get = Mock(
        side_effect=lambda key, default=None: {
            "eviction_batch_size": 100,
            "node_confidence_threshold": 0.3,
            "forgetting_threshold": 0.3,
            "forgetting_threshold_kg": 0.3,
            "kg_retention_days": 30,
            "forgetting_llm_confirm_enable": False,
            "forgetting_llm_confirm_threshold": 0.5,
            "forgetting_llm_confirm_provider": "",
            "isolation_config.enable_group_memory_isolation": False,
        }.get(key, default)
    )
    return mock


class TestPruningPhase:
    @pytest.fixture
    def phase(self):
        return PruningPhase()

    @pytest.mark.asyncio
    async def test_execute_l2_unavailable(self, phase):
        l2 = Mock()
        l2.is_available = False
        l3 = None
        llm = None

        with patch("iris_memory.dream.pruning.get_config", return_value=_mock_config()):
            result = await phase.execute(l2, l3, llm)

        assert result["l2_evicted"] == 0
        assert result["l3_evicted"] == 0

    @pytest.mark.asyncio
    async def test_execute_no_l2_entries(self, phase):
        l2 = Mock()
        l2.is_available = True
        l2.get_all_entries = AsyncMock(return_value=[])
        l3 = Mock()
        l3.is_available = False
        llm = None

        with patch("iris_memory.dream.pruning.get_config", return_value=_mock_config()):
            result = await phase.execute(l2, l3, llm)

        assert result["l2_evicted"] == 0

    @pytest.mark.asyncio
    async def test_llm_confirm_eviction_disabled(self, phase):
        with patch("iris_memory.dream.pruning.get_config") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.get = Mock(
                side_effect=lambda key, default=None: {
                    "forgetting_llm_confirm_enable": False,
                }.get(key, default)
            )
            mock_config.return_value = mock_config_instance

            entries = [("id1", "content1", 0.05), ("id2", "content2", 0.08)]
            result = await phase._llm_confirm_eviction(entries, None, source="l2")

            assert len(result) == 2
            assert "id1" in result
            assert "id2" in result

    @pytest.mark.asyncio
    async def test_llm_confirm_eviction_no_llm_defaults_to_keep(self, phase):
        with patch("iris_memory.dream.pruning.get_config") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.get = Mock(
                side_effect=lambda key, default=None: {
                    "forgetting_llm_confirm_enable": True,
                }.get(key, default)
            )
            mock_config.return_value = mock_config_instance

            entries = [("id1", "content1", 0.05)]
            result = await phase._llm_confirm_eviction(entries, None, source="l2")

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_llm_confirm_eviction_llm_failure_defaults_to_keep(self, phase):
        with patch("iris_memory.dream.pruning.get_config") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.get = Mock(
                side_effect=lambda key, default=None: {
                    "forgetting_llm_confirm_enable": True,
                    "forgetting_llm_confirm_threshold": 0.5,
                    "forgetting_llm_confirm_provider": None,
                }.get(key, default)
            )
            mock_config.return_value = mock_config_instance

            llm = Mock()
            llm.generate_direct = AsyncMock(side_effect=Exception("LLM error"))

            entries = [("id1", "content1", 0.05)]
            result = await phase._llm_confirm_eviction(entries, llm, source="l2")

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_llm_confirm_eviction_high_score_auto_confirm(self, phase):
        with patch("iris_memory.dream.pruning.get_config") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.get = Mock(
                side_effect=lambda key, default=None: {
                    "forgetting_llm_confirm_enable": True,
                    "forgetting_llm_confirm_threshold": 0.5,
                    "forgetting_llm_confirm_provider": None,
                }.get(key, default)
            )
            mock_config.return_value = mock_config_instance

            llm = Mock()
            llm.generate_direct = AsyncMock(return_value="YES")

            entries = [("id1", "content1", 0.8)]
            result = await phase._llm_confirm_eviction(entries, llm, source="l2")

            assert "id1" in result

    @pytest.mark.asyncio
    async def test_execute_with_entries_parameter(self, phase):
        l2 = Mock()
        l2.is_available = True
        l3 = Mock()
        l3.is_available = False
        llm = None

        entry = Mock()
        entry.id = "mem1"
        entry.content = "test"
        entry.confidence = 0.9
        entry.metadata = {"confidence": 0.9}

        with patch("iris_memory.dream.pruning.get_config", return_value=_mock_config()):
            result = await phase.execute(l2, l3, llm, entries=[entry])

        assert result["l2_low_confidence_marked"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_subject_nodes(self, phase):
        """回归：_cleanup_orphaned_subject_nodes 清理无 Person 关联的节点

        无主节点（如"有特定角色偏好"但不知道是谁的偏好）对用户没有价值，
        应当在遗忘淘汰之前被定向清除。该方法通过 find_orphaned_subject_nodes
        查找，再调用 evict_nodes 批量删除。
        """
        l3 = Mock()
        l3.find_orphaned_subject_nodes = AsyncMock(
            return_value=[
                {
                    "id": "preference_abc123",
                    "label": "Preference",
                    "name": "角色偏好",
                    "content": "有特定角色偏好",
                    "confidence": 0.7,
                },
                {
                    "id": "trait_def456",
                    "label": "Trait",
                    "name": "性格特点",
                    "content": "某种性格特点",
                    "confidence": 0.6,
                },
            ]
        )
        l3.evict_nodes = AsyncMock(return_value=2)

        removed = await phase._cleanup_orphaned_subject_nodes(l3)

        assert removed == 2
        l3.find_orphaned_subject_nodes.assert_called_once()
        l3.evict_nodes.assert_called_once()

        # evict_nodes 应收到正确的孤儿节点 ID 列表
        evicted_ids = l3.evict_nodes.call_args.args[0]
        assert "preference_abc123" in evicted_ids
        assert "trait_def456" in evicted_ids
        assert len(evicted_ids) == 2

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_subject_nodes_empty(self, phase):
        """回归：_cleanup_orphaned_subject_nodes 无孤儿节点时返回 0 且不调用 evict_nodes"""
        l3 = Mock()
        l3.find_orphaned_subject_nodes = AsyncMock(return_value=[])
        l3.evict_nodes = AsyncMock(return_value=0)

        removed = await phase._cleanup_orphaned_subject_nodes(l3)

        assert removed == 0
        l3.find_orphaned_subject_nodes.assert_called_once()
        l3.evict_nodes.assert_not_called()

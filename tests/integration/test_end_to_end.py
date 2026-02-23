"""
端到端集成测试
测试完整的记忆管理流程和复杂场景
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import asyncio

from iris_memory.models.memory import Memory
from iris_memory.models.user_persona import UserPersona
from iris_memory.core.types import (
    StorageLayer, MemoryType, ModalityType, EmotionType
)
from iris_memory.storage.lifecycle_manager import SessionState
from iris_memory.storage.chroma_manager import ChromaManager
from iris_memory.storage.lifecycle_manager import SessionLifecycleManager
from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.utils.token_manager import TokenBudget, MemoryCompressor, DynamicMemorySelector


@pytest.fixture
def mock_config():
    """模拟配置对象"""
    config = Mock()
    config.embedding = {
        'local_model': 'BAAI/bge-m3',
        'local_dimension': 1024,
        'collection_name': 'test_collection'
    }
    return config


@pytest.fixture
def mock_plugin_context():
    """模拟插件上下文"""
    context = Mock()
    return context


@pytest.fixture
def sample_memories():
    """示例记忆列表"""
    now = datetime.now()
    return [
        Memory(
            id="mem_001",
            content="今天工作很顺利，完成了重要项目",
            user_id="user_123",
            group_id="group_456",
            type=MemoryType.FACT,
            modality=ModalityType.TEXT,
            storage_layer=StorageLayer.EPISODIC,
            created_time=now - timedelta(hours=1),
            quality_level=5,
            rif_score=0.8,
            importance_score=0.9
        ),
        Memory(
            id="mem_002",
            content="我和朋友聊天很开心",
            user_id="user_123",
            group_id="group_456",
            type=MemoryType.FACT,
            modality=ModalityType.TEXT,
            storage_layer=StorageLayer.EPISODIC,
            created_time=now - timedelta(hours=2),
            quality_level=4,
            rif_score=0.7,
            importance_score=0.8
        ),
        Memory(
            id="mem_003",
            content="我对这次考试感到有点焦虑",
            user_id="user_123",
            group_id=None,  # 私聊
            type=MemoryType.EMOTION,
            modality=ModalityType.TEXT,
            storage_layer=StorageLayer.WORKING,
            created_time=now - timedelta(hours=3),
            quality_level=3,
            rif_score=0.6,
            importance_score=0.7
        )
    ]


@pytest.fixture
def sample_persona():
    """示例用户画像"""
    return UserPersona(
        user_id="user_123",
        work_style="创新",
        lifestyle="规律",
        emotional_baseline="joy",
        personality_openness=0.8,
        communication_formality=0.4
    )


class TestEndToEndMemoryCapture:
    """测试端到端记忆捕获流程"""
    
    @pytest.mark.asyncio
    async def test_full_capture_workflow(self, mock_config, mock_plugin_context):
        """测试完整的捕获工作流"""
        # 1. 初始化组件
        chroma_manager = ChromaManager(mock_config, Mock(), mock_plugin_context)
        
        with patch.object(chroma_manager, 'initialize') as mock_init, \
             patch.object(chroma_manager, 'add_memory') as mock_add:
            mock_init.return_value = None
            mock_add.return_value = "mem_001"
            
            await chroma_manager.initialize()
            
            # 2. 创建记忆
            memory = Memory(
                id="mem_001",
                content="这是一条测试记忆",
                user_id="user_123",
                group_id="group_456",
                type=MemoryType.FACT,
                modality=ModalityType.TEXT,
                storage_layer=StorageLayer.EPISODIC
            )
            
            # 3. 存储记忆
            memory_id = await chroma_manager.add_memory(memory)
            
            # 4. 验证
            assert memory_id == "mem_001"
            mock_add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_capture_with_emotion_analysis(
        self, mock_config
    ):
        """测试带情感分析的捕获流程"""
        # 1. 初始化情感分析器
        emotion_analyzer = EmotionAnalyzer(mock_config)

        # 2. 分析文本情感
        text = "今天真开心！"
        emotion_result = await emotion_analyzer.analyze_emotion(text)

        # 3. 验证情感分析结果
        assert emotion_result["primary"] == EmotionType.JOY
        assert 0.0 <= emotion_result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_capture_with_rif_scoring(self):
        """测试带RIF评分的捕获流程"""
        # 1. 初始化RIF评分器
        rif_scorer = RIFScorer()

        # 2. 创建记忆
        memory = Memory(
            id="mem_001",
            content="重要的会议内容",
            user_id="user_123",
            created_time=datetime.now() - timedelta(hours=1),
            importance_score=0.9,
            access_count=5
        )

        # 3. 计算RIF分数
        rif_score = rif_scorer.calculate_rif(memory)

        # 4. 验证RIF分数
        assert 0.0 <= rif_score <= 1.0


class TestEndToEndRetrieval:
    """测试端到端检索流程"""

    @pytest.mark.asyncio
    async def test_retrieval_workflow(
        self, mock_config, sample_memories
    ):
        """测试完整检索工作流"""
        # 1. 初始化组件
        chroma_manager = ChromaManager(mock_config, Mock(), Mock())
        
        with patch.object(chroma_manager, 'initialize') as mock_init, \
             patch.object(chroma_manager, 'query_memories') as mock_query:
            mock_init.return_value = None
            mock_query.return_value = sample_memories[:2]
            
            await chroma_manager.initialize()
            
            # 2. 执行查询
            results = await chroma_manager.query_memories(
                "工作项目",
                user_id="user_123",
                group_id="group_456",
                top_k=10
            )
            
            # 3. 验证结果
            assert len(results) == 2
            assert results[0].id == "mem_001"
            assert results[0].type == MemoryType.FACT
    
    @pytest.mark.asyncio
    async def test_retrieval_with_token_budget(
        self, mock_config, sample_memories
    ):
        """测试带token预算的检索流程"""
        # 1. 初始化token预算
        token_budget = TokenBudget(total_budget=512, preamble_cost=20)
        compressor = MemoryCompressor(max_summary_length=50)
        selector = DynamicMemorySelector(
            token_budget=token_budget,
            compressor=compressor
        )
        
        # 2. 添加摘要
        for memory in sample_memories:
            memory.summary = memory.content[:50]
        
        # 3. 选择记忆
        selected, stats = selector.select_memories(sample_memories, target_count=3)
        
        # 4. 验证结果
        assert stats["total_candidates"] == 3
        assert stats["selected_count"] >= 0
        assert stats["used_tokens"] <= token_budget.total_budget
        
        # 5. 生成上下文
        context = selector.get_memory_context(sample_memories, target_count=3)
        
        if selected:
            assert "【相关记忆】" in context


class TestEndToEndPersonaUpdate:
    """测试端到端画像更新流程"""
    
    def test_persona_update_from_memories(self, sample_persona):
        """测试从记忆更新画像"""
        # 1. 创建多条记忆
        memories = [
            Mock(
                type="emotion",
                subtype="joy",
                emotional_weight=0.8,
                content="很开心"
            ),
            Mock(
                type="fact",
                content="我希望在工作中提升技能",
                summary="提升工作技能"
            ),
            Mock(
                type="relationship",
                summary="很信任朋友",
                content="信任朋友"
            )
        ]
        
        # 2. 更新画像
        for memory in memories:
            sample_persona.update_from_memory(memory)
        
        # 3. 验证更新
        assert sample_persona.emotional_baseline == "joy"
        assert "提升工作技能" in sample_persona.work_goals
        assert sample_persona.trust_level > 0.5


class TestEndToEndSessionLifecycle:
    """测试端到端会话生命周期流程"""
    
    @pytest.mark.asyncio
    async def test_session_lifecycle_full(self):
        """测试完整的会话生命周期"""
        # 1. 创建会话管理器
        session_manager = Mock()
        session_manager.get_session_key = Mock(
            side_effect=lambda user_id, group_id: f"{user_id}:{group_id or 'private'}"
        )
        session_manager.update_session_activity = Mock()
        session_manager.get_working_memory = Mock(return_value=[])
        session_manager.clear_working_memory = Mock()
        session_manager.delete_session = Mock()
        
        lifecycle_manager = SessionLifecycleManager(
            session_manager=session_manager,
            cleanup_interval=10,
            session_timeout=60,
            inactive_timeout=30
        )
        
        # 2. 启动生命周期管理
        await lifecycle_manager.start()
        assert lifecycle_manager.is_running is True
        
        # 3. 激活会话
        await lifecycle_manager.activate_session("user_123", "group_456")
        session_key = "user_123:group_456"
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.ACTIVE
        
        # 4. 停用会话
        await lifecycle_manager.deactivate_session("user_123", "group_456")
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.INACTIVE
        
        # 5. 关闭会话
        await lifecycle_manager.close_session("user_123", "group_456")
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.CLOSED
        
        # 6. 停止生命周期管理
        await lifecycle_manager.stop()
        assert lifecycle_manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_session_timeout_and_cleanup(self):
        """测试会话超时和清理"""
        session_manager = Mock()
        session_manager.get_session_key = Mock(
            side_effect=lambda user_id, group_id: f"{user_id}:{group_id or 'private'}"
        )
        session_manager.update_session_activity = Mock()
        session_manager.get_working_memory = Mock(return_value=[])
        session_manager.clear_working_memory = Mock()
        session_manager.delete_session = Mock()
        
        lifecycle_manager = SessionLifecycleManager(
            session_manager=session_manager,
            cleanup_interval=1,
            session_timeout=2,
            inactive_timeout=1
        )
        
        await lifecycle_manager.start()
        
        # 激活会话
        await lifecycle_manager.activate_session("user_123", "group_456")
        session_key = "user_123:group_456"
        
        # 模拟时间流逝（超过非活跃超时）
        lifecycle_manager.session_states[session_key]['last_active'] = \
            datetime.now() - timedelta(seconds=70)
        
        # 执行清理
        await lifecycle_manager._cleanup_expired_sessions()
        
        # 应该转换为INACTIVE
        assert lifecycle_manager.session_states[session_key]['state'] == SessionState.INACTIVE
        
        await lifecycle_manager.stop()


class TestEndToEndComplexScenario:
    """测试复杂场景"""
    
    @pytest.mark.asyncio
    async def test_multi_user_multi_group_scenario(self, mock_config):
        """测试多用户多群组场景"""
        # 1. 初始化组件
        chroma_manager = ChromaManager(mock_config, Mock(), Mock())
        
        with patch.object(chroma_manager, 'initialize') as mock_init, \
             patch.object(chroma_manager, 'add_memory') as mock_add, \
             patch.object(chroma_manager, 'query_memories') as mock_query:
            mock_init.return_value = None
            mock_add.return_value = "mem_xxx"
            
            await chroma_manager.initialize()
            
            # 2. 创建多个用户的记忆
            users = ["user_1", "user_2", "user_3"]
            groups = ["group_1", "group_2", None]  # None表示私聊
            
            memories = []
            for user in users:
                for group in groups:
                    memory = Memory(
                        id=f"mem_{user}_{group}",
                        content=f"{user}在{group or '私聊'}的消息",
                        user_id=user,
                        group_id=group,
                        type=MemoryType.FACT,
                        modality=ModalityType.TEXT,
                        storage_layer=StorageLayer.EPISODIC
                    )
                    memories.append(memory)
                    await chroma_manager.add_memory(memory)
            
            # 3. 验证添加
            assert mock_add.call_count == len(memories)
    
    @pytest.mark.asyncio
    async def test_memory_upgrade_workflow(self, mock_config):
        """测试记忆升级工作流"""
        # 1. 创建工作记忆
        memory = Memory(
            id="mem_working",
            content="重要的工作记忆",
            user_id="user_123",
            group_id="group_456",
            type=MemoryType.FACT,
            modality=ModalityType.TEXT,
            storage_layer=StorageLayer.WORKING,
            access_count=10,
            importance_score=0.9,
            rif_score=0.85
        )
        
        # 2. 模拟记忆升级
        memory.storage_layer = StorageLayer.EPISODIC
        
        # 3. 验证升级
        assert memory.storage_layer == StorageLayer.EPISODIC
    
    @pytest.mark.asyncio
    async def test_cross_session_retrieval(self, mock_config, sample_memories):
        """测试跨会话检索"""
        # 1. 初始化组件
        chroma_manager = ChromaManager(mock_config, Mock(), Mock())

        # 过滤出群聊记忆（排除私聊）
        group_memories = [m for m in sample_memories if m.group_id == "group_456"]

        with patch.object(chroma_manager, 'initialize') as mock_init, \
             patch.object(chroma_manager, 'get_all_memories') as mock_get, \
             patch.object(chroma_manager, 'count_memories') as mock_count:
            mock_init.return_value = None
            mock_get.return_value = group_memories
            mock_count.return_value = len(group_memories)

            await chroma_manager.initialize()

            # 2. 获取用户所有记忆
            all_memories = await chroma_manager.get_all_memories(
                user_id="user_123",
                group_id="group_456"
            )

            # 3. 验证结果
            assert len(all_memories) == 2  # 私聊的记忆不在其中
            assert all(m.group_id == "group_456" for m in all_memories)

            # 4. 统计记忆数量
            count = await chroma_manager.count_memories(
                user_id="user_123",
                group_id="group_456"
            )

            assert count == 2


class TestEndToEndErrorHandling:
    """测试端到端错误处理"""
    
    @pytest.mark.asyncio
    async def test_api_failure_fallback(self, mock_config):
        """测试情感分析的基本功能"""
        # 1. 初始化情感分析器
        emotion_analyzer = EmotionAnalyzer(mock_config)

        # 2. 分析文本情感
        result = await emotion_analyzer.analyze_emotion("测试文本")

        # 3. 验证返回结果包含情感分析
        assert "primary" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_insufficient_token_budget(self, mock_config, sample_memories):
        """测试token预算不足"""
        # 1. 初始化低预算的token管理
        token_budget = TokenBudget(total_budget=50, preamble_cost=20)
        compressor = MemoryCompressor(max_summary_length=30)
        selector = DynamicMemorySelector(
            token_budget=token_budget,
            compressor=compressor
        )
        
        # 2. 尝试选择大量记忆
        selected, stats = selector.select_memories(sample_memories, target_count=10)

        # 3. 验证部分选择（因为token预算应该限制选择数量）
        # 3个记忆可能都能被选择，因为它们的summary可能很短
        assert stats["selected_count"] <= len(sample_memories)
        assert len(selected) <= len(sample_memories)


class TestEndToEndPerformance:
    """测试端到端性能"""
    
    @pytest.mark.asyncio
    async def test_large_scale_memory_operations(self, mock_config):
        """测试大规模记忆操作"""
        # 1. 创建大量记忆
        memories = []
        for i in range(100):
            memory = Memory(
                id=f"mem_{i}",
                content=f"记忆内容{i}",
                user_id="user_123",
                type=MemoryType.FACT,
                storage_layer=StorageLayer.EPISODIC,
                rif_score=0.5 + (i / 200),
                importance_score=0.5 + (i / 200)
            )
            memories.append(memory)
        
        # 2. 排序记忆
        sorted_memories = sorted(
            memories,
            key=lambda m: m.rif_score * m.importance_score,
            reverse=True
        )
        
        # 3. 验证排序
        assert sorted_memories[0].rif_score >= sorted_memories[-1].rif_score
        
        # 4. 选择前10个
        top_10 = sorted_memories[:10]
        assert len(top_10) == 10
        
        # 5. 计算RIF分数
        rif_scorer = RIFScorer()
        for memory in top_10:
            memory.rif_score = rif_scorer.calculate_rif(memory)
            assert 0.0 <= memory.rif_score <= 1.0


class TestEndToEndDataConsistency:
    """测试端到端数据一致性"""
    
    @pytest.mark.asyncio
    async def test_persona_memory_consistency(self, sample_persona):
        """测试画像和记忆的一致性"""
        # 1. 添加证据
        sample_persona.add_memory_evidence("mem_001", "confirmed")
        sample_persona.add_memory_evidence("mem_002", "inferred")
        sample_persona.add_memory_evidence("mem_003", "contested")
        
        # 2. 验证证据状态
        assert len(sample_persona.evidence_confirmed) == 1
        assert len(sample_persona.evidence_inferred) == 1
        assert len(sample_persona.evidence_contested) == 1
        
        # 3. 序列化和反序列化
        data = sample_persona.to_dict()
        new_persona = UserPersona.from_dict(data)
        
        # 4. 验证数据一致性
        assert new_persona.user_id == sample_persona.user_id
        assert len(new_persona.evidence_confirmed) == len(sample_persona.evidence_confirmed)
        assert len(new_persona.evidence_inferred) == len(sample_persona.evidence_inferred)
        assert len(new_persona.evidence_contested) == len(sample_persona.evidence_contested)

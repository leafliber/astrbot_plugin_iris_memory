#!/usr/bin/env python
"""
本地测试脚本 - 独立测试核心组件
不依赖 AstrBot 框架，仅测试业务逻辑
"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from iris_memory.models.memory import Memory, MemoryType, StorageLayer
from iris_memory.models.user_persona import UserPersona
from iris_memory.storage.chroma_manager import ChromaManager
from iris_memory.capture.capture_engine import MemoryCaptureEngine
from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.storage.session_manager import SessionManager
from iris_memory.storage.cache import CacheManager


class TestConfig:
    """简单的测试配置类"""
    def __init__(self):
        self.embedding_model = "BAAI/bge-m3"
        self.embedding_dimension = 1024
    
    def get(self, key, default=None):
        return getattr(self, key, default)


async def test_basic_capture():
    """测试记忆捕获"""
    print("=== 测试记忆捕获 ===")
    
    # 初始化组件
    config = TestConfig()
    emotion_analyzer = EmotionAnalyzer(config)
    rif_scorer = RIFScorer()
    capture_engine = MemoryCaptureEngine(
        chroma_manager=None,  # 不使用实际存储
        emotion_analyzer=emotion_analyzer,
        rif_scorer=rif_scorer
    )
    
    # 测试捕获
    test_messages = [
        "我是Cassia，我喜欢编程",
        "我觉得今天心情很好",
        "我有一个想法"
    ]
    
    for msg in test_messages:
        memory = await capture_engine.capture_memory(
            message=msg,
            user_id="test_user",
            group_id=None,
            is_user_requested=True
        )
        if memory:
            print(f"✓ 捕获成功: {memory.type.value} - {memory.content[:30]}...")
        else:
            print(f"✗ 未捕获: {msg}")
    
    print()


async def test_session_manager():
    """测试会话管理"""
    print("=== 测试会话管理 ===")
    
    session_manager = SessionManager()
    
    # 测试工作记忆
    memory = Memory(
        content="测试记忆",
        user_id="test_user",
        group_id=None,
        type=MemoryType.FACT,
        storage_layer=StorageLayer.WORKING
    )
    
    session_manager.add_working_memory(memory)
    memories = session_manager.get_working_memory("test_user", None)
    
    print(f"✓ 工作记忆数量: {len(memories)}")
    
    # 测试会话激活
    session_manager.update_session_activity("test_user", None)
    session_key = session_manager.get_session_key("test_user", None)
    session = session_manager.get_session(session_key)
    
    if session:
        print(f"✓ 会话消息数: {session['message_count']}")
    
    print()


async def test_emotion_analysis():
    """测试情感分析"""
    print("=== 测试情感分析 ===")
    
    config = TestConfig()
    emotion_analyzer = EmotionAnalyzer(config)
    
    test_messages = [
        "我非常开心！",
        "这让我感到很沮丧",
        "我觉得还可以"
    ]
    
    for msg in test_messages:
        result = await emotion_analyzer.analyze_emotion(msg)
        print(f"✓ '{msg}' -> {result['primary']} (强度: {result['intensity']:.2f})")
    
    print()


async def test_rif_scoring():
    """测试RIF评分"""
    print("=== 测试RIF评分 ===")

    rif_scorer = RIFScorer()

    # 创建测试记忆
    memory = Memory(
        content="测试记忆",
        user_id="test_user",
        group_id=None,
        type=MemoryType.FACT,
        storage_layer=StorageLayer.EPISODIC
    )

    # 模拟多次访问
    for i in range(5):
        memory.access_count += 1

    # 计算RIF评分
    score = rif_scorer.calculate_rif(memory)
    print(f"✓ 访问 {memory.access_count} 次后，RIF评分: {score:.3f}")
    print(f"✓ 记忆RIF评分已更新: {memory.rif_score:.3f}")

    print()


async def main():
    """运行所有测试"""
    print("Iris Memory 本地测试\n" + "="*50 + "\n")
    
    try:
        await test_basic_capture()
        await test_session_manager()
        await test_emotion_analysis()
        await test_rif_scoring()
        
        print("="*50)
        print("✓ 所有测试完成！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

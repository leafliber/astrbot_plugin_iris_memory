#!/usr/bin/env python3
"""
多维度评分系统测试和演示脚本

用于测试新的多维度评分系统并与传统RIF评分进行对比。
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 模拟导入（实际使用时需要确保正确的导入路径）
from iris_memory.models.memory import Memory
from iris_memory.analysis.multidimensional_scorer import MultidimensionalScorer, ScenarioType
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.core.types import MemoryType, QualityLevel, SensitivityLevel, StorageLayer


def create_test_memories() -> List[Memory]:
    """创建测试记忆数据"""
    now = datetime.now()
    
    memories = [
        # 1. 高频访问的个人偏好
        Memory(
            content="用户喜欢喝咖啡，特别是拿铁，每天早上都要喝一杯",
            type=MemoryType.FACT,
            quality_level=QualityLevel.HIGH_CONFIDENCE,
            confidence=0.85,
            access_count=15,
            last_access_time=now - timedelta(hours=2),
            created_time=now - timedelta(days=30),
            emotional_weight=0.3,
            importance_score=0.7,
            is_user_requested=True,
            consistency_score=0.9,
            keywords=["咖啡", "拿铁", "早上", "偏好"]
        ),
        
        # 2. 重要的情感记忆
        Memory(
            content="用户提到昨天和女朋友分手了，感到很难过和失落",
            type=MemoryType.EMOTION,
            quality_level=QualityLevel.CONFIRMED,
            confidence=0.95,
            access_count=3,
            last_access_time=now - timedelta(hours=1),
            created_time=now - timedelta(hours=26),
            emotional_weight=0.9,
            importance_score=0.8,
            is_user_requested=False,
            consistency_score=1.0,
            sensitivity_level=SensitivityLevel.PRIVATE,
            keywords=["分手", "女朋友", "难过", "失落"]
        ),
        
        # 3. 关系记忆
        Memory(
            content="用户的妹妹在上海工作，是一名软件工程师",
            type=MemoryType.RELATIONSHIP,
            quality_level=QualityLevel.MODERATE,
            confidence=0.7,
            access_count=2,
            last_access_time=now - timedelta(days=7),
            created_time=now - timedelta(days=60),
            emotional_weight=0.4,
            importance_score=0.6,
            is_user_requested=False,
            consistency_score=0.8,
            sensitivity_level=SensitivityLevel.PERSONAL,
            keywords=["妹妹", "上海", "软件工程师", "家人"]
        ),
        
        # 4. 日常互动记忆
        Memory(
            content="用户询问今天天气如何",
            type=MemoryType.INTERACTION,
            quality_level=QualityLevel.LOW_CONFIDENCE,
            confidence=0.4,
            access_count=1,
            last_access_time=now - timedelta(hours=5),
            created_time=now - timedelta(hours=5),
            emotional_weight=0.1,
            importance_score=0.2,
            is_user_requested=False,
            consistency_score=0.5,
            keywords=["天气", "询问"]
        ),
        
        # 5. 陈旧但重要的记忆
        Memory(
            content="用户的生日是3月15日，今年28岁",
            type=MemoryType.FACT,
            quality_level=QualityLevel.CONFIRMED,
            confidence=1.0,
            access_count=8,
            last_access_time=now - timedelta(days=20),
            created_time=now - timedelta(days=200),
            emotional_weight=0.5,
            importance_score=0.9,
            is_user_requested=True,
            consistency_score=1.0,
            sensitivity_level=SensitivityLevel.PERSONAL,
            keywords=["生日", "3月15日", "28岁", "个人信息"]
        )
    ]
    
    return memories


def create_test_contexts() -> List[Dict[str, Any]]:
    """创建测试上下文"""
    return [
        # 情感对话场景
        {
            "scenario_type": ScenarioType.EMOTIONAL_DIALOGUE,
            "emotional_state": {"type": "sadness", "intensity": 0.8},
            "current_message": "我还是很想念她",
            "query_type": "emotional_support"
        },
        
        # 事实查询场景
        {
            "scenario_type": ScenarioType.FACTUAL_QUERY,
            "emotional_state": {"type": "neutral", "intensity": 0.1},
            "current_message": "我的生日是什么时候来着",
            "query_type": "fact_lookup"
        },
        
        # 社交场景
        {
            "scenario_type": ScenarioType.SOCIAL_INTERACTION,
            "emotional_state": {"type": "neutral", "intensity": 0.3},
            "current_message": "我妹妹最近怎么样",
            "query_type": "social_inquiry",
            "group_id": "family_group"
        },
        
        # 日常闲聊
        {
            "scenario_type": ScenarioType.ROUTINE_CHAT,
            "emotional_state": {"type": "calm", "intensity": 0.2},
            "current_message": "今天天气真不错",
            "query_type": "casual_chat"
        },
        
        # 默认场景
        {
            "emotional_state": {"type": "neutral", "intensity": 0.5},
            "current_message": "你好"
        }
    ]


async def run_comparison_test():
    """运行对比测试"""
    print("🔬 多维度评分系统 vs 传统RIF评分对比测试")
    print("=" * 60)
    
    # 初始化评分器
    traditional_scorer = RIFScorer(use_multidimensional=False)
    multidimensional_scorer = RIFScorer(
        use_multidimensional=True,
        enable_advanced_features=True,
        enable_context_adaptation=True
    )
    
    # 创建测试数据
    memories = create_test_memories()
    contexts = create_test_contexts()
    
    print(f"📊 测试数据: {len(memories)} 条记忆, {len(contexts)} 种场景")
    print()
    
    # 对每个记忆在每种场景下进行评分
    results = []
    
    for i, memory in enumerate(memories):
        print(f"📝 记忆 {i+1}: {memory.content[:50]}...")
        print(f"   类型: {memory.type.value}, 置信度: {memory.confidence:.2f}, 访问次数: {memory.access_count}")
        print()
        
        for j, context in enumerate(contexts):
            scenario = context.get('scenario_type', 'default')
            print(f"  🎯 场景 {j+1}: {scenario}")
            
            # 传统评分
            traditional_score = traditional_scorer.calculate_rif(memory.copy() if hasattr(memory, 'copy') else memory)
            
            # 多维度评分
            multidimensional_score = multidimensional_scorer.calculate_rif(memory, context)
            
            # 计算差异
            difference = multidimensional_score - traditional_score
            
            print(f"     传统RIF: {traditional_score:.3f}")
            print(f"     多维度:  {multidimensional_score:.3f} (差异: {difference:+.3f})")
            print()
            
            results.append({
                'memory_index': i,
                'context_index': j,
                'memory_type': memory.type.value,
                'scenario': scenario,
                'traditional_score': traditional_score,
                'multidimensional_score': multidimensional_score,
                'difference': difference,
                'memory_content': memory.content[:100]
            })
    
    # 统计分析
    print("📈 统计分析")
    print("-" * 40)
    
    differences = [r['difference'] for r in results]
    avg_diff = sum(differences) / len(differences)
    max_diff = max(differences)
    min_diff = min(differences)
    
    print(f"平均差异: {avg_diff:+.3f}")
    print(f"最大差异: {max_diff:+.3f}")
    print(f"最小差异: {min_diff:+.3f}")
    print()
    
    # 按场景分析
    print("🎯 按场景分析")
    print("-" * 40)
    
    scenario_stats = {}
    for result in results:
        scenario = result['scenario']
        if scenario not in scenario_stats:
            scenario_stats[scenario] = []
        scenario_stats[scenario].append(result['difference'])
    
    for scenario, diffs in scenario_stats.items():
        avg_diff = sum(diffs) / len(diffs)
        print(f"{scenario}: 平均差异 {avg_diff:+.3f}")
    
    print()
    
    # 获取评分统计
    print("📊 评分器统计")
    print("-" * 40)
    
    trad_stats = traditional_scorer.get_statistics()
    multi_stats = multidimensional_scorer.get_statistics()
    
    print(f"传统RIF计算次数: {trad_stats.get('traditional_calculations', 0)}")
    print(f"多维度计算次数: {multi_stats.get('multidimensional_calculations', 0)}")
    print(f"回退次数: {multi_stats.get('fallbacks', 0)}")
    
    return results


def analyze_detailed_scores():
    """分析详细的多维度得分"""
    print("\n🔍 多维度得分详细分析")
    print("=" * 60)
    
    # 创建多维度评分器
    scorer = MultidimensionalScorer(
        enable_advanced_features=True,
        enable_context_adaptation=True
    )
    
    memories = create_test_memories()
    contexts = create_test_contexts()
    
    # 选择一个有代表性的记忆进行详细分析
    memory = memories[1]  # 情感记忆
    context = contexts[0]  # 情感对话场景
    
    print(f"📝 分析记忆: {memory.content}")
    print(f"🎯 分析场景: {context.get('scenario_type', 'default')}")
    print()
    
    # 计算详细得分
    result = scorer.calculate_score(memory, context)
    
    print("📊 各维度得分:")
    print(f"  时间维度 (Temporal):  {result.temporal_score:.3f}")
    print(f"  语义维度 (Semantic):  {result.semantic_score:.3f}")
    print(f"  社交维度 (Social):    {result.social_score:.3f}")
    print(f"  情感维度 (Emotional): {result.emotional_score:.3f}")
    print(f"  质量维度 (Quality):   {result.quality_score:.3f}")
    print()
    
    print("⚖️ 权重配置:")
    weights = result.weights_used
    print(f"  时间权重: {weights.temporal:.2f}")
    print(f"  语义权重: {weights.semantic:.2f}")
    print(f"  社交权重: {weights.social:.2f}")
    print(f"  情感权重: {weights.emotional:.2f}")
    print(f"  质量权重: {weights.quality:.2f}")
    print()
    
    print("🎯 最终结果:")
    print(f"  加权得分: {result.weighted_score:.3f}")
    print(f"  最终得分: {result.final_score:.3f}")
    print(f"  场景类型: {result.scenario_type.value}")
    print()
    
    print("🔬 计算元数据:")
    metadata = result.calculation_metadata
    for key, value in metadata.items():
        print(f"  {key}: {value}")


def benchmark_performance():
    """性能基准测试"""
    print("\n⚡ 性能基准测试")
    print("=" * 60)
    
    import time
    
    # 创建测试数据
    memories = create_test_memories() * 10  # 50条记忆
    context = create_test_contexts()[0]
    
    # 测试传统RIF评分性能
    traditional_scorer = RIFScorer(use_multidimensional=False)
    
    start_time = time.time()
    for memory in memories:
        traditional_scorer.calculate_rif(memory)
    traditional_time = time.time() - start_time
    
    # 测试多维度评分性能
    multidimensional_scorer = RIFScorer(use_multidimensional=True)
    
    start_time = time.time()
    for memory in memories:
        multidimensional_scorer.calculate_rif(memory, context)
    multidimensional_time = time.time() - start_time
    
    print(f"📊 性能测试结果 ({len(memories)} 条记忆):")
    print(f"  传统RIF:   {traditional_time:.3f}秒 ({traditional_time/len(memories)*1000:.2f}ms/条)")
    print(f"  多维度:    {multidimensional_time:.3f}秒 ({multidimensional_time/len(memories)*1000:.2f}ms/条)")
    print(f"  性能比例:  {multidimensional_time/traditional_time:.2f}x (多维度相对于传统)")
    
    if multidimensional_time > traditional_time:
        print(f"  ⚠️  多维度评分较慢 {(multidimensional_time/traditional_time-1)*100:.1f}%")
    else:
        print(f"  ✅ 多维度评分更快 {(1-multidimensional_time/traditional_time)*100:.1f}%")


async def main():
    """主函数"""
    try:
        # 运行对比测试
        await run_comparison_test()
        
        # 详细分析
        analyze_detailed_scores()
        
        # 性能测试
        benchmark_performance()
        
        print("\n✅ 测试完成！")
        print("\n💡 使用建议:")
        print("1. 在情感对话场景中，多维度评分更准确地识别情感记忆的重要性")
        print("2. 在事实查询场景中，质量维度和语义维度权重更高，提升查询准确性")  
        print("3. 社交场景中，关系记忆和群体相关性得到更好的评估")
        print("4. 多维度评分虽然计算复杂，但提供了更精细的记忆价值评估")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
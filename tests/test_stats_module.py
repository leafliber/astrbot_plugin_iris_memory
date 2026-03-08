"""
stats 模块全面测试脚本

测试内容：
1. 模块导入测试
2. 数据模型测试
3. KV 存储适配器测试
4. 注册表测试
5. 与 llm_helper 的集成测试
6. 查询功能测试
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
    
    def add_pass(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append(f"{name}: {error}")
        print(f"  ✗ {name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"测试结果: {self.passed}/{total} 通过")
        if self.errors:
            print("\n失败详情:")
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'='*60}")
        return self.failed == 0


result = TestResult()


def test_imports():
    """测试 1: 模块导入"""
    print("\n[测试 1] 模块导入测试")
    
    try:
        from iris_memory.stats import get_stats_registry
        result.add_pass("get_stats_registry 导入")
    except Exception as e:
        result.add_fail("get_stats_registry 导入", str(e))
        return False
    
    try:
        from iris_memory.stats import LLMStatsRegistry
        result.add_pass("LLMStatsRegistry 导入")
    except Exception as e:
        result.add_fail("LLMStatsRegistry 导入", str(e))
    
    try:
        from iris_memory.stats import LLMCallRecord, LLMAggregatedStats, StatsQuery, StatsSummary
        result.add_pass("数据模型导入")
    except Exception as e:
        result.add_fail("数据模型导入", str(e))
    
    try:
        from iris_memory.stats import StatsKVStore
        result.add_pass("StatsKVStore 导入")
    except Exception as e:
        result.add_fail("StatsKVStore 导入", str(e))
    
    try:
        from iris_memory.stats import SOURCE_ALIASES
        result.add_pass("SOURCE_ALIASES 导入")
    except Exception as e:
        result.add_fail("SOURCE_ALIASES 导入", str(e))
    
    return True


def test_models():
    """测试 2: 数据模型"""
    print("\n[测试 2] 数据模型测试")
    
    from iris_memory.stats.models import LLMCallRecord, LLMAggregatedStats, StatsQuery, StatsSummary
    
    # LLMCallRecord
    try:
        record = LLMCallRecord(
            record_id="test-001",
            timestamp=time.time(),
            provider_id="test-provider",
            source_module="test_module",
            source_class="TestClass",
            success=True,
            tokens_used=100,
            duration_ms=500.0,
        )
        assert record.record_id == "test-001"
        assert record.success == True
        assert record.tokens_used == 100
        
        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["record_id"] == "test-001"
        
        record2 = LLMCallRecord.from_dict(d)
        assert record2.record_id == record.record_id
        assert record2.tokens_used == record.tokens_used
        
        result.add_pass("LLMCallRecord 创建/序列化/反序列化")
    except Exception as e:
        result.add_fail("LLMCallRecord", str(e))
    
    # LLMAggregatedStats
    try:
        stats = LLMAggregatedStats()
        stats.total_calls = 10
        stats.successful_calls = 9
        stats.failed_calls = 1
        stats.total_tokens = 1000
        stats.total_duration_ms = 5000.0
        stats.calls_by_provider["provider-a"] = 5
        stats.calls_by_provider["provider-b"] = 5
        
        assert stats.success_rate == 0.9
        assert stats.avg_duration_ms == 500.0
        assert stats.avg_tokens_per_call == 100.0
        
        d = stats.to_dict()
        stats2 = LLMAggregatedStats.from_dict(d)
        assert stats2.total_calls == 10
        assert stats2.success_rate == 0.9
        
        stats2.reset()
        assert stats2.total_calls == 0
        assert len(stats2.calls_by_provider) == 0
        
        result.add_pass("LLMAggregatedStats 创建/计算/重置")
    except Exception as e:
        result.add_fail("LLMAggregatedStats", str(e))
    
    # StatsQuery
    try:
        query = StatsQuery(
            provider_id="test-provider",
            limit=50,
        )
        assert query.provider_id == "test-provider"
        assert query.limit == 50
        
        d = query.to_dict()
        query2 = StatsQuery.from_dict(d)
        assert query2.provider_id == "test-provider"
        
        result.add_pass("StatsQuery 创建/序列化")
    except Exception as e:
        result.add_fail("StatsQuery", str(e))
    
    # StatsSummary
    try:
        summary = StatsSummary(
            total_calls=100,
            success_rate=0.95,
            total_tokens=10000,
            avg_duration_ms=300.0,
            top_providers=[{"provider_id": "p1", "calls": 50}],
            top_sources=[{"source": "s1", "calls": 30}],
            recent_errors=5,
        )
        assert summary.total_calls == 100
        d = summary.to_dict()
        assert d["total_calls"] == 100
        
        result.add_pass("StatsSummary 创建/序列化")
    except Exception as e:
        result.add_fail("StatsSummary", str(e))


def test_kv_store():
    """测试 3: KV 存储适配器"""
    print("\n[测试 3] KV 存储适配器测试")
    
    from iris_memory.stats.store import StatsKVStore
    
    store = StatsKVStore()
    
    # 初始状态
    try:
        assert not store.is_ready()
        result.add_pass("StatsKVStore 初始状态检查")
    except Exception as e:
        result.add_fail("StatsKVStore 初始状态", str(e))
    
    # Mock KV 接口
    mock_data: Dict[str, Any] = {}
    
    async def mock_get_kv(key: str, default: Any) -> Any:
        return mock_data.get(key, default)
    
    async def mock_put_kv(key: str, value: Any) -> None:
        mock_data[key] = value
    
    # 设置接口
    try:
        store.set_kv_interface(mock_get_kv, mock_put_kv)
        assert store.is_ready()
        result.add_pass("StatsKVStore 设置 KV 接口")
    except Exception as e:
        result.add_fail("StatsKVStore 设置 KV 接口", str(e))
    
    # 测试保存和加载
    async def test_save_load():
        try:
            await store.save_aggregated({"total_calls": 100})
            loaded = await store.load_aggregated()
            assert loaded.get("total_calls") == 100
            result.add_pass("StatsKVStore 聚合统计保存/加载")
        except Exception as e:
            result.add_fail("StatsKVStore 聚合统计保存/加载", str(e))
        
        try:
            await store.save_records([
                {"record_id": "r1", "timestamp": 1.0},
                {"record_id": "r2", "timestamp": 2.0},
            ])
            loaded = await store.load_records()
            assert len(loaded) == 2
            assert loaded[0]["record_id"] == "r1"
            result.add_pass("StatsKVStore 记录保存/加载")
        except Exception as e:
            result.add_fail("StatsKVStore 记录保存/加载", str(e))
        
        # 测试截断
        try:
            large_records = [{"record_id": f"r{i}"} for i in range(1500)]
            await store.save_records(large_records)
            loaded = await store.load_records()
            assert len(loaded) == 1000  # MAX_RECORDS
            result.add_pass("StatsKVStore 记录截断")
        except Exception as e:
            result.add_fail("StatsKVStore 记录截断", str(e))
        
        # 测试清除
        try:
            await store.clear_all()
            loaded = await store.load_aggregated()
            assert loaded == {}
            loaded_records = await store.load_records()
            assert loaded_records == []
            result.add_pass("StatsKVStore 清除数据")
        except Exception as e:
            result.add_fail("StatsKVStore 清除数据", str(e))
    
    asyncio.run(test_save_load())


def test_registry():
    """测试 4: 注册表"""
    print("\n[测试 4] 注册表测试")
    
    from iris_memory.stats import get_stats_registry, LLMStatsRegistry, StatsQuery
    
    # 单例测试
    try:
        r1 = get_stats_registry()
        r2 = LLMStatsRegistry()
        assert r1 is r2
        result.add_pass("LLMStatsRegistry 单例模式")
    except Exception as e:
        result.add_fail("LLMStatsRegistry 单例模式", str(e))
    
    registry = get_stats_registry()
    
    # Mock KV 接口
    mock_data: Dict[str, Any] = {}
    
    async def mock_get_kv(key: str, default: Any) -> Any:
        return mock_data.get(key, default)
    
    async def mock_put_kv(key: str, value: Any) -> None:
        mock_data[key] = value
    
    async def test_registry_async():
        # 设置 KV 接口
        try:
            registry.set_kv_interface(mock_get_kv, mock_put_kv)
            await registry.initialize()
            result.add_pass("LLMStatsRegistry 初始化")
        except Exception as e:
            result.add_fail("LLMStatsRegistry 初始化", str(e))
        
        # 记录调用
        try:
            record_id = await registry.record_call(
                provider_id="test-provider",
                success=True,
                tokens_used=100,
                duration_ms=500.0,
                prompt="test prompt",
                response="test response",
            )
            assert record_id is not None
            
            aggregated = registry.get_aggregated()
            assert aggregated.total_calls == 1
            assert aggregated.successful_calls == 1
            assert aggregated.total_tokens == 100
            
            result.add_pass("LLMStatsRegistry record_call")
        except Exception as e:
            result.add_fail("LLMStatsRegistry record_call", str(e))
        
        # 多次调用
        try:
            for i in range(5):
                await registry.record_call(
                    provider_id=f"provider-{i % 2}",
                    success=i % 3 != 0,
                    tokens_used=50 + i * 10,
                    duration_ms=100.0 + i * 50,
                    prompt=f"prompt {i}",
                    response=f"response {i}",
                )
            
            aggregated = registry.get_aggregated()
            assert aggregated.total_calls == 6  # 1 + 5
            
            result.add_pass("LLMStatsRegistry 多次调用统计")
        except Exception as e:
            result.add_fail("LLMStatsRegistry 多次调用统计", str(e))
        
        # 查询测试
        try:
            query = StatsQuery(provider_id="provider-0", limit=10)
            records = registry.query(query)
            assert len(records) >= 1
            for r in records:
                assert r.provider_id == "provider-0"
            
            result.add_pass("LLMStatsRegistry 查询功能")
        except Exception as e:
            result.add_fail("LLMStatsRegistry 查询功能", str(e))
        
        # get_by_provider
        try:
            provider_stats = registry.get_by_provider("provider-0")
            assert provider_stats["provider_id"] == "provider-0"
            assert "total_calls" in provider_stats
            
            result.add_pass("LLMStatsRegistry get_by_provider")
        except Exception as e:
            result.add_fail("LLMStatsRegistry get_by_provider", str(e))
        
        # get_summary
        try:
            summary = registry.get_summary()
            assert summary.total_calls == 6
            assert len(summary.top_providers) <= 5
            assert len(summary.top_sources) <= 5
            
            result.add_pass("LLMStatsRegistry get_summary")
        except Exception as e:
            result.add_fail("LLMStatsRegistry get_summary", str(e))
        
        # 持久化测试
        try:
            await registry.flush()
            
            assert "llm_stats_aggregated" in mock_data
            assert "llm_stats_records" in mock_data
            
            result.add_pass("LLMStatsRegistry 持久化")
        except Exception as e:
            result.add_fail("LLMStatsRegistry 持久化", str(e))
        
        # 重置测试
        try:
            await registry.reset()
            aggregated = registry.get_aggregated()
            assert aggregated.total_calls == 0
            
            result.add_pass("LLMStatsRegistry 重置")
        except Exception as e:
            result.add_fail("LLMStatsRegistry 重置", str(e))
    
    asyncio.run(test_registry_async())


def test_source_inference():
    """测试 5: 来源推断"""
    print("\n[测试 5] 来源推断测试")
    
    from iris_memory.stats import get_stats_registry, SOURCE_ALIASES
    
    registry = get_stats_registry()
    
    # 检查 SOURCE_ALIASES
    try:
        assert "iris_memory.knowledge_graph.kg_extractor.KGExtractor" in SOURCE_ALIASES
        assert SOURCE_ALIASES["iris_memory.knowledge_graph.kg_extractor.KGExtractor"] == "kg_extraction"
        result.add_pass("SOURCE_ALIASES 配置检查")
    except Exception as e:
        result.add_fail("SOURCE_ALIASES 配置检查", str(e))
    
    # 测试 _infer_source 方法
    try:
        source_module, source_class = registry._infer_source()
        # 在测试脚本中调用，应该返回 unknown 或脚本相关名称
        assert isinstance(source_module, str)
        assert isinstance(source_class, str)
        result.add_pass("_infer_source 方法调用")
    except Exception as e:
        result.add_fail("_infer_source 方法调用", str(e))


def test_integration_with_llm_helper():
    """测试 6: 与 llm_helper 集成"""
    print("\n[测试 6] 与 llm_helper 集成测试")
    
    from iris_memory.stats import get_stats_registry
    from iris_memory.utils import llm_helper
    
    try:
        _record_stats = llm_helper._record_stats
        LLMCallResult = llm_helper.LLMCallResult
        result.add_pass("llm_helper._record_stats 导入")
    except Exception as e:
        result.add_fail("llm_helper._record_stats 导入", str(e))
        return
    
    registry = get_stats_registry()
    
    # Mock KV 接口
    mock_data: Dict[str, Any] = {}
    
    async def mock_get_kv(key: str, default: Any) -> Any:
        return mock_data.get(key, default)
    
    async def mock_put_kv(key: str, value: Any) -> None:
        mock_data[key] = value
    
    async def test_integration():
        try:
            registry.set_kv_interface(mock_get_kv, mock_put_kv)
            await registry.initialize()
            
            # 重置
            await registry.reset()
            
            # 模拟调用 _record_stats
            test_result = LLMCallResult(
                success=True,
                content="test response",
                tokens_used=150,
            )
            
            _record_stats(
                provider_id="integration-test-provider",
                result=test_result,
                duration_ms=300.0,
                prompt="integration test prompt",
                is_multimodal=False,
                image_count=0,
            )
            
            await asyncio.sleep(0.1)
            
            aggregated = registry.get_aggregated()
            assert aggregated.total_calls >= 1
            
            result.add_pass("_record_stats 调用集成")
        except Exception as e:
            result.add_fail("_record_stats 调用集成", str(e))
    
    asyncio.run(test_integration())


def test_concurrent_access():
    """测试 7: 并发访问"""
    print("\n[测试 7] 并发访问测试")
    
    from iris_memory.stats import get_stats_registry
    
    registry = get_stats_registry()
    
    # Mock KV 接口
    mock_data: Dict[str, Any] = {}
    
    async def mock_get_kv(key: str, default: Any) -> Any:
        return mock_data.get(key, default)
    
    async def mock_put_kv(key: str, value: Any) -> None:
        mock_data[key] = value
    
    async def test_concurrent():
        try:
            registry.set_kv_interface(mock_get_kv, mock_put_kv)
            await registry.initialize()
            await registry.reset()
            
            async def record_call(i: int):
                await registry.record_call(
                    provider_id=f"concurrent-{i % 3}",
                    success=True,
                    tokens_used=100,
                    duration_ms=100.0,
                    prompt=f"concurrent prompt {i}",
                    response=f"concurrent response {i}",
                )
            
            tasks = [record_call(i) for i in range(20)]
            await asyncio.gather(*tasks)
            
            aggregated = registry.get_aggregated()
            assert aggregated.total_calls == 20
            
            result.add_pass("并发 record_call 测试")
        except Exception as e:
            result.add_fail("并发 record_call 测试", str(e))
    
    asyncio.run(test_concurrent())


def test_edge_cases():
    """测试 8: 边界情况"""
    print("\n[测试 8] 边界情况测试")
    
    from iris_memory.stats import get_stats_registry, StatsQuery
    from iris_memory.stats.models import LLMCallRecord, LLMAggregatedStats
    
    # 空记录查询
    try:
        registry = get_stats_registry()
        query = StatsQuery(provider_id="non-existent-provider")
        records = registry.query(query)
        assert records == []
        result.add_pass("空记录查询")
    except Exception as e:
        result.add_fail("空记录查询", str(e))
    
    # 零值统计
    try:
        stats = LLMAggregatedStats()
        assert stats.success_rate == 0.0
        assert stats.avg_duration_ms == 0.0
        assert stats.avg_tokens_per_call == 0.0
        result.add_pass("零值统计计算")
    except Exception as e:
        result.add_fail("零值统计计算", str(e))
    
    # 空字符串处理
    try:
        record = LLMCallRecord(
            record_id="test",
            timestamp=0.0,
            provider_id="",
            source_module="",
            source_class="",
            success=False,
            tokens_used=0,
            duration_ms=0.0,
            prompt_preview="",
            response_preview="",
        )
        d = record.to_dict()
        record2 = LLMCallRecord.from_dict(d)
        assert record2.provider_id == ""
        result.add_pass("空字符串处理")
    except Exception as e:
        result.add_fail("空字符串处理", str(e))
    
    # 超长字符串截断
    try:
        long_prompt = "x" * 1000
        long_response = "y" * 1000
        
        async def test_truncate():
            registry = get_stats_registry()
            record_id = await registry.record_call(
                provider_id="test",
                success=True,
                tokens_used=100,
                duration_ms=100.0,
                prompt=long_prompt,
                response=long_response,
            )
            
            records = registry.get_recent(1)
            assert len(records) == 1
            assert len(records[0].prompt_preview) == 100
            assert len(records[0].response_preview) == 100
        
        asyncio.run(test_truncate())
        result.add_pass("超长字符串截断")
    except Exception as e:
        result.add_fail("超长字符串截断", str(e))


def main():
    print("=" * 60)
    print("stats 模块全面测试")
    print("=" * 60)
    
    test_imports()
    test_models()
    test_kv_store()
    test_registry()
    test_source_inference()
    test_integration_with_llm_helper()
    test_concurrent_access()
    test_edge_cases()
    
    success = result.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

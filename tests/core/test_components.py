"""组件管理器测试"""

import pytest
from iris_memory.core.components import (
    Component,
    ComponentManager,
    SystemStatus,
    ComponentInitResult,
)


class MockComponent(Component):
    """模拟组件"""

    def __init__(self, name: str, should_fail: bool = False):
        super().__init__()
        self._name = name
        self._should_fail = should_fail
        self.initialize_called = False
        self.shutdown_called = False

    @property
    def name(self) -> str:
        return self._name

    async def initialize(self) -> None:
        self.initialize_called = True
        if self._should_fail:
            raise RuntimeError(f"组件 {self._name} 初始化失败")
        self._is_available = True

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self._is_available = False


class TestSystemStatus:
    """SystemStatus 测试"""

    def test_register_module(self):
        """测试模块注册"""
        status = SystemStatus()

        status.register_module("l1_buffer", default_available=False)
        status.register_module("l2_memory", default_available=True)

        assert not status.is_module_available("l1_buffer")
        assert status.is_module_available("l2_memory")

    def test_set_available(self):
        """测试设置可用性"""
        status = SystemStatus()
        status.register_module("l2_memory", default_available=False)

        # 设置为可用
        status.set_available("l2_memory", True)
        assert status.is_module_available("l2_memory")

        # 设置为不可用
        status.set_available("l2_memory", False)
        assert not status.is_module_available("l2_memory")

    def test_get_available_modules(self):
        """测试获取可用模块列表"""
        status = SystemStatus()
        status.register_module("l1_buffer", default_available=True)
        status.register_module("l2_memory", default_available=False)
        status.register_module("l3_kg", default_available=True)

        available = status.get_available_modules()

        assert "l1_buffer" in available
        assert "l2_memory" not in available
        assert "l3_kg" in available

    def test_get_unavailable_modules(self):
        """测试获取不可用模块列表"""
        status = SystemStatus()
        status.register_module("l1_buffer", default_available=True)
        status.register_module("l2_memory", default_available=False)

        unavailable = status.get_unavailable_modules()

        assert "l1_buffer" not in unavailable
        assert "l2_memory" in unavailable

    def test_to_dict(self):
        """测试转换为字典"""
        status = SystemStatus()
        status.register_module("l1_buffer", default_available=True)
        status.register_module("l2_memory", default_available=False)

        result = status.to_dict()

        assert result["l1_buffer_available"]
        assert not result["l2_memory_available"]

    def test_is_module_available_nonexistent(self):
        """测试查询不存在的模块"""
        status = SystemStatus()

        # 不存在的模块应该返回 False
        assert not status.is_module_available("nonexistent")


class TestComponentManager:
    """ComponentManager 测试"""

    @pytest.mark.asyncio
    async def test_initialize_all_success(self):
        """测试所有组件初始化成功"""
        component1 = MockComponent("l1_buffer")
        component2 = MockComponent("l2_memory")

        manager = ComponentManager((component1, component2))

        # 验证模块已注册
        assert not manager.status.is_module_available("l1_buffer")
        assert not manager.status.is_module_available("l2_memory")

        # 初始化
        results = await manager.initialize_all()

        # 验证结果
        assert len(results) == 2
        assert all(r.success for r in results)

        # 验证组件状态
        assert component1.is_available
        assert component2.is_available
        assert component1.initialize_called
        assert component2.initialize_called

        # 验证系统状态
        assert manager.status.is_module_available("l1_buffer")
        assert manager.status.is_module_available("l2_memory")

    @pytest.mark.asyncio
    async def test_initialize_partial_failure(self):
        """测试部分组件初始化失败"""
        component1 = MockComponent("l1_buffer", should_fail=False)
        component2 = MockComponent("l2_memory", should_fail=True)

        manager = ComponentManager((component1, component2))
        results = await manager.initialize_all()

        # 验证结果
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success
        assert "初始化失败" in results[1].error_message

        # 验证组件状态
        assert component1.is_available
        assert not component2.is_available

        # 验证系统状态
        assert manager.status.is_module_available("l1_buffer")
        assert not manager.status.is_module_available("l2_memory")

    @pytest.mark.asyncio
    async def test_initialize_all_failure(self):
        """测试所有组件初始化失败"""
        component1 = MockComponent("l1_buffer", should_fail=True)
        component2 = MockComponent("l2_memory", should_fail=True)

        manager = ComponentManager((component1, component2))
        results = await manager.initialize_all()

        # 验证结果
        assert len(results) == 2
        assert all(not r.success for r in results)

        # 验证系统状态
        assert not manager.status.is_module_available("l1_buffer")
        assert not manager.status.is_module_available("l2_memory")

    @pytest.mark.asyncio
    async def test_initialize_twice_raises_error(self):
        """测试重复初始化抛出异常"""
        component = MockComponent("l1_buffer")
        manager = ComponentManager((component,))

        await manager.initialize_all()

        # 第二次初始化应该抛出异常
        with pytest.raises(RuntimeError, match="组件已初始化"):
            await manager.initialize_all()

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """测试关闭所有组件"""
        component1 = MockComponent("l1_buffer")
        component2 = MockComponent("l2_memory")

        manager = ComponentManager((component1, component2))
        await manager.initialize_all()

        # 关闭
        await manager.shutdown_all()

        # 验证组件状态
        assert component1.shutdown_called
        assert component2.shutdown_called
        assert not component1.is_available
        assert not component2.is_available

        # 验证系统状态已重置
        assert not manager.status.is_module_available("l1_buffer")
        assert not manager.status.is_module_available("l2_memory")

    @pytest.mark.asyncio
    async def test_shutdown_without_initialize(self):
        """测试未初始化时关闭"""
        component = MockComponent("l1_buffer")
        manager = ComponentManager((component,))

        # 未初始化时关闭应该正常执行
        await manager.shutdown_all()

        assert not component.shutdown_called

    @pytest.mark.asyncio
    async def test_shutdown_partial_failure(self):
        """测试部分组件关闭失败"""
        component1 = MockComponent("l1_buffer")
        component2 = MockComponent("l2_memory")

        # 让第二个组件关闭时失败
        async def failing_shutdown():
            raise RuntimeError("关闭失败")

        component2.shutdown = failing_shutdown

        manager = ComponentManager((component1, component2))
        await manager.initialize_all()

        # 关闭应该继续执行，不抛出异常
        await manager.shutdown_all()

        # 第一个组件应该已关闭
        assert component1.shutdown_called

    def test_get_component(self):
        """测试按名称获取组件"""
        component1 = MockComponent("l1_buffer")
        component2 = MockComponent("l2_memory")

        manager = ComponentManager((component1, component2))

        # 获取存在的组件
        found = manager.get_component("l2_memory")
        assert found is component2

        # 获取不存在的组件
        not_found = manager.get_component("l3_kg")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_available_components(self):
        """测试获取可用组件列表"""
        component1 = MockComponent("l1_buffer", should_fail=False)
        component2 = MockComponent("l2_memory", should_fail=True)

        manager = ComponentManager((component1, component2))
        await manager.initialize_all()

        available = manager.get_available_components()

        assert len(available) == 1
        assert component1 in available
        assert component2 not in available

    @pytest.mark.asyncio
    async def test_get_failed_components(self):
        """测试获取失败组件列表"""
        component1 = MockComponent("l1_buffer", should_fail=False)
        component2 = MockComponent("l2_memory", should_fail=True)

        manager = ComponentManager((component1, component2))
        await manager.initialize_all()

        failed = manager.get_failed_components()

        assert len(failed) == 1
        assert component1 not in failed
        assert component2 in failed


class TestComponentInitResult:
    """ComponentInitResult 测试"""

    def test_success_result(self):
        """测试成功结果"""
        result = ComponentInitResult("l1_buffer", True)

        assert result.name == "l1_buffer"
        assert result.success
        assert result.error_message is None

    def test_failure_result(self):
        """测试失败结果"""
        result = ComponentInitResult("l2_memory", False, "初始化失败")

        assert result.name == "l2_memory"
        assert not result.success
        assert result.error_message == "初始化失败"

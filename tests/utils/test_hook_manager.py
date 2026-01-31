"""
测试Hook管理器
测试 iris_memory.utils.hook_manager 中的 MemoryInjector 和 HookCoordinator 类
"""

import pytest

from iris_memory.utils.hook_manager import (
    InjectionMode,
    HookPriority,
    MemoryInjector,
    HookCoordinator
)


class TestInjectionMode:
    """测试InjectionMode枚举"""

    def test_injection_mode_values(self):
        """测试注入模式的值"""
        assert InjectionMode.PREFIX.value == "prefix"
        assert InjectionMode.SUFFIX.value == "suffix"
        assert InjectionMode.EMBEDDED.value == "embedded"
        assert InjectionMode.HYBRID.value == "hybrid"

    def test_injection_mode_count(self):
        """测试注入模式的数量"""
        assert len(InjectionMode) == 4


class TestHookPriority:
    """测试HookPriority枚举"""

    def test_hook_priority_values(self):
        """测试Hook优先级的值"""
        assert HookPriority.CRITICAL.value == "critical"
        assert HookPriority.HIGH.value == "high"
        assert HookPriority.NORMAL.value == "normal"
        assert HookPriority.LOW.value == "low"

    def test_hook_priority_count(self):
        """测试Hook优先级的数量"""
        assert len(HookPriority) == 4


class TestMemoryInjectorInit:
    """测试MemoryInjector初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        injector = MemoryInjector()

        assert injector.injection_mode == InjectionMode.SUFFIX
        assert injector.priority == HookPriority.NORMAL
        assert injector.namespace == "iris_memory"
        assert injector.enable_injection is True
        assert injector.enable_structured is True
        assert injector.max_injection_length == 1000

    def test_init_custom_mode(self):
        """测试自定义注入模式"""
        injector = MemoryInjector(injection_mode=InjectionMode.PREFIX)

        assert injector.injection_mode == InjectionMode.PREFIX

    def test_init_custom_priority(self):
        """测试自定义优先级"""
        injector = MemoryInjector(priority=HookPriority.HIGH)

        assert injector.priority == HookPriority.HIGH

    def test_init_custom_namespace(self):
        """测试自定义命名空间"""
        injector = MemoryInjector(namespace="custom_namespace")

        assert injector.namespace == "custom_namespace"


class TestMemoryInjectorInject:
    """测试inject方法"""

    def test_inject_disabled(self):
        """测试禁用注入"""
        injector = MemoryInjector()
        injector.enable_injection = False

        result = injector.inject(
            memory_context="测试上下文",
            system_prompt="系统提示"
        )

        assert result == "系统提示"

    def test_inject_empty_context(self):
        """测试空上下文注入"""
        injector = MemoryInjector()

        result = injector.inject(
            memory_context="",
            system_prompt="系统提示"
        )

        assert result == "系统提示"

    def test_inject_none_context(self):
        """测试None上下文注入"""
        injector = MemoryInjector()

        result = injector.inject(
            memory_context=None,
            system_prompt="系统提示"
        )

        assert result == "系统提示"

    def test_inject_truncate_long_context(self):
        """测试截断过长上下文"""
        injector = MemoryInjector()
        long_context = "a" * 1500

        result = injector.inject(
            memory_context=long_context,
            system_prompt="系统提示"
        )

        # 应该被截断到max_injection_length + "..."
        # 注意：structured模式会添加额外格式化内容：[NAMESPACE CONTEXT]\n\n
        extra_length = len(f"[{injector.namespace.upper()} CONTEXT]\\n\\n") if injector.enable_structured else 0
        assert len(result) <= len("系统提示") + 1000 + 3 + extra_length
        assert result.endswith("...")

    def test_inject_prefix_mode(self):
        """测试前置模式注入"""
        injector = MemoryInjector(injection_mode=InjectionMode.PREFIX)

        result = injector.inject(
            memory_context="测试上下文",
            system_prompt="系统提示"
        )

        assert result.startswith("[IRIS_MEMORY CONTEXT]")
        assert "测试上下文" in result
        assert result.endswith("系统提示")

    def test_inject_suffix_mode(self):
        """测试后置模式注入"""
        injector = MemoryInjector(injection_mode=InjectionMode.SUFFIX)

        result = injector.inject(
            memory_context="测试上下文",
            system_prompt="系统提示"
        )

        assert result.startswith("系统提示")
        assert "[IRIS_MEMORY CONTEXT]" in result
        assert "测试上下文" in result

    def test_inject_embedded_mode_with_context_keyword(self):
        """测试嵌入式注入（带上下文关键词）"""
        injector = MemoryInjector(injection_mode=InjectionMode.EMBEDDED)

        result = injector.inject(
            memory_context="测试上下文",
            system_prompt="系统上下文提示"
        )

        # 应该在"上下文"附近插入
        assert "测试上下文" in result

    def test_inject_embedded_mode_without_keyword(self):
        """测试嵌入式注入（无关键词，回退到后置）"""
        injector = MemoryInjector(injection_mode=InjectionMode.EMBEDDED)

        result = injector.inject(
            memory_context="测试上下文",
            system_prompt="系统提示"
        )

        # 应该回退到后置模式
        assert result.startswith("系统提示")

    def test_inject_hybrid_mode(self):
        """测试混合模式注入"""
        injector = MemoryInjector(injection_mode=InjectionMode.HYBRID)

        result = injector.inject(
            memory_context="测试上下文",
            system_prompt="系统提示"
        )

        # 应该同时包含前置和后置
        assert "[IRIS_MEMORY REFERENCE]" in result
        assert "[IRIS_MEMORY DETAILS]" in result
        assert "测试上下文" in result
        assert "系统提示" in result

    def test_inject_unstructured(self):
        """测试非结构化注入"""
        injector = MemoryInjector()
        injector.enable_structured = False

        result = injector.inject(
            memory_context="测试上下文",
            system_prompt="系统提示"
        )

        # 不应该包含方括号标记
        assert "[IRIS_MEMORY" not in result
        assert "测试上下文" in result


class TestMemoryInjectorHelperMethods:
    """测试MemoryInjector辅助方法"""

    def test_get_priority_hint(self):
        """测试获取优先级提示"""
        injector = MemoryInjector(
            injection_mode=InjectionMode.PREFIX,
            priority=HookPriority.HIGH,
            namespace="test_namespace"
        )

        hint = injector.get_priority_hint()

        assert "test_namespace" in hint
        assert "prefix" in hint
        assert "high" in hint

    def test_parse_existing_context_empty(self):
        """测试解析空上下文"""
        injector = MemoryInjector()

        contexts = injector.parse_existing_context("系统提示")

        assert contexts == []

    def test_parse_existing_context_with_namespace(self):
        """测试解析带命名空间的上下文"""
        injector = MemoryInjector()

        prompt = "系统提示\n\n[TEST CONTEXT]\n测试内容\n\n其他内容"
        contexts = injector.parse_existing_context(prompt)

        assert len(contexts) == 1
        assert contexts[0]["namespace"] == "test"
        assert contexts[0]["content"] == "测试内容"

    def test_parse_existing_context_multiple(self):
        """测试解析多个上下文"""
        injector = MemoryInjector()

        prompt = (
            "系统提示\n\n"
            "[TEST1 CONTEXT]\n内容1\n\n"
            "[TEST2 CONTEXT]\n内容2\n\n"
            "其他内容"
        )
        contexts = injector.parse_existing_context(prompt)

        assert len(contexts) == 2
        assert contexts[0]["namespace"] == "test1"
        assert contexts[1]["namespace"] == "test2"


class TestMemoryInjectorConflictDetection:
    """测试冲突检测"""

    def test_detect_conflicts_no_conflicts(self):
        """测试无冲突情况"""
        injector = MemoryInjector(namespace="test_namespace")

        conflicts = injector.detect_conflicts(
            system_prompt="系统提示",
            my_content="新内容"
        )

        assert conflicts["namespace_conflict"] is False
        assert conflicts["content_conflict"] is False
        assert conflicts["suggestions"] == []

    def test_detect_namespace_conflict(self):
        """测试命名空间冲突"""
        injector = MemoryInjector(namespace="test_namespace")

        prompt = "系统提示\n\n[TEST_NAMESPACE CONTEXT]\n已有内容"
        conflicts = injector.detect_conflicts(
            system_prompt=prompt,
            my_content="新内容"
        )

        assert conflicts["namespace_conflict"] is True
        assert len(conflicts["suggestions"]) > 0

    def test_detect_content_conflict(self):
        """测试内容冲突"""
        injector = MemoryInjector(namespace="namespace1")

        prompt = "系统提示\n\n[NAMESPACE2 CONTEXT]\n相似的内容"
        conflicts = injector.detect_conflicts(
            system_prompt=prompt,
            my_content="相似的内容"
        )

        # 相似度应该大于0.8
        assert conflicts["content_conflict"] is True

    def test_detect_content_no_conflict(self):
        """测试内容不冲突"""
        injector = MemoryInjector(namespace="namespace1")

        prompt = "系统提示\n\n[NAMESPACE2 CONTEXT]\n完全不同的内容"
        conflicts = injector.detect_conflicts(
            system_prompt=prompt,
            my_content="我的内容"
        )

        # 相似度应该小于0.8
        assert conflicts["content_conflict"] is False

    def test_calculate_similarity_identical(self):
        """测试计算相同文本的相似度"""
        injector = MemoryInjector()

        similarity = injector._calculate_similarity("相同文本", "相同文本")

        assert similarity == 1.0

    def test_calculate_similarity_completely_different(self):
        """测试计算完全不同文本的相似度"""
        injector = MemoryInjector()

        # 使用完全不同的文本，确保无任何重叠的词汇
        similarity = injector._calculate_similarity("first A", "second B")

        # 完全不同的文本应该相似度为0
        assert similarity == 0.0

    def test_calculate_similarity_partial(self):
        """测试计算部分相似文本的相似度"""
        injector = MemoryInjector()

        similarity = injector._calculate_similarity("测试内容A", "测试内容B")

        # "测试内容"是相同的
        assert 0 < similarity < 1

    def test_calculate_similarity_case_insensitive(self):
        """测试计算相似度时忽略大小写"""
        injector = MemoryInjector()

        similarity = injector._calculate_similarity("Test Content", "test content")

        assert similarity == 1.0


class TestHookCoordinator:
    """测试HookCoordinator类"""

    def test_init(self):
        """测试初始化"""
        coordinator = HookCoordinator()

        assert coordinator.registered_hooks == {}

    def test_register_hook_single(self):
        """测试注册单个Hook"""
        coordinator = HookCoordinator()

        def handler(*args, **kwargs):
            return "result"

        coordinator.register_hook(handler, HookPriority.NORMAL)

        assert HookPriority.NORMAL in coordinator.registered_hooks
        assert handler in coordinator.registered_hooks[HookPriority.NORMAL]

    def test_register_hook_multiple(self):
        """测试注册多个Hook"""
        coordinator = HookCoordinator()

        handler1 = lambda: "result1"
        handler2 = lambda: "result2"

        coordinator.register_hook(handler1, HookPriority.HIGH)
        coordinator.register_hook(handler2, HookPriority.HIGH)

        assert len(coordinator.registered_hooks[HookPriority.HIGH]) == 2

    def test_register_hook_different_priorities(self):
        """测试注册不同优先级的Hook"""
        coordinator = HookCoordinator()

        handler1 = lambda: "result1"
        handler2 = lambda: "result2"

        coordinator.register_hook(handler1, HookPriority.HIGH)
        coordinator.register_hook(handler2, HookPriority.LOW)

        assert HookPriority.HIGH in coordinator.registered_hooks
        assert HookPriority.LOW in coordinator.registered_hooks

    def test_execute_hooks_empty(self):
        """测试执行空Hook列表"""
        coordinator = HookCoordinator()

        results = coordinator.execute_hooks()

        assert results == []

    def test_execute_hooks_single(self):
        """测试执行单个Hook"""
        coordinator = HookCoordinator()

        def handler(x):
            return x * 2

        coordinator.register_hook(handler, HookPriority.NORMAL)

        results = coordinator.execute_hooks(5)

        assert len(results) == 1
        assert results[0] == 10

    def test_execute_hooks_multiple(self):
        """测试执行多个Hook"""
        coordinator = HookCoordinator()

        handler1 = lambda x: x * 2
        handler2 = lambda x: x + 1
        handler3 = lambda x: x / 2

        coordinator.register_hook(handler1, HookPriority.HIGH)
        coordinator.register_hook(handler2, HookPriority.NORMAL)
        coordinator.register_hook(handler3, HookPriority.LOW)

        results = coordinator.execute_hooks(10)

        assert len(results) == 3
        assert 20 in results
        assert 11 in results
        assert 5.0 in results

    def test_execute_hooks_priority_order(self):
        """测试Hook按优先级顺序执行"""
        coordinator = HookCoordinator()

        execution_order = []

        def handler_critical():
            execution_order.append("critical")

        def handler_high():
            execution_order.append("high")

        def handler_normal():
            execution_order.append("normal")

        def handler_low():
            execution_order.append("low")

        coordinator.register_hook(handler_normal, HookPriority.NORMAL)
        coordinator.register_hook(handler_critical, HookPriority.CRITICAL)
        coordinator.register_hook(handler_low, HookPriority.LOW)
        coordinator.register_hook(handler_high, HookPriority.HIGH)

        coordinator.execute_hooks()

        # 验证执行顺序
        assert execution_order[0] == "critical"
        assert execution_order[1] == "high"
        assert execution_order[2] == "normal"
        assert execution_order[3] == "low"

    def test_execute_hooks_exception_handling(self):
        """测试Hook异常处理"""
        coordinator = HookCoordinator()

        def handler_error():
            raise ValueError("Test error")

        def handler_success():
            return "success"

        coordinator.register_hook(handler_error, HookPriority.HIGH)
        coordinator.register_hook(handler_success, HookPriority.NORMAL)

        # 不应该抛出异常
        results = coordinator.execute_hooks()

        # 应该只返回成功的Hook结果
        assert len(results) == 1
        assert "success" in results


class TestIntegration:
    """测试集成场景"""

    def test_full_injection_workflow(self):
        """测试完整注入工作流"""
        # 创建注入器
        injector = MemoryInjector(
            injection_mode=InjectionMode.SUFFIX,
            namespace="test_plugin"
        )

        # 检测冲突
        conflicts = injector.detect_conflicts(
            system_prompt="系统提示",
            my_content="测试内容"
        )

        # 应该没有冲突
        assert conflicts["namespace_conflict"] is False

        # 注入内容
        result = injector.inject(
            memory_context="测试内容",
            system_prompt="系统提示"
        )

        # 验证注入结果
        assert "系统提示" in result
        assert "测试内容" in result
        assert "[TEST_PLUGIN CONTEXT]" in result

    def test_coordinator_with_injector(self):
        """测试协调器与注入器配合使用"""
        coordinator = HookCoordinator()
        injector = MemoryInjector()

        def inject_handler(system_prompt):
            return injector.inject(
                memory_context="上下文",
                system_prompt=system_prompt
            )

        coordinator.register_hook(inject_handler, HookPriority.HIGH)

        results = coordinator.execute_hooks("原始提示")

        assert len(results) == 1
        assert "上下文" in results[0]
        assert "[IRIS_MEMORY CONTEXT]" in results[0]

    def test_multiple_namespace_isolation(self):
        """测试多命名空间隔离"""
        injector1 = MemoryInjector(namespace="plugin1")
        injector2 = MemoryInjector(namespace="plugin2")

        # Plugin1注入内容
        prompt1 = "系统提示"
        result1 = injector1.inject("内容1", prompt1)

        # Plugin2注入到已有内容的prompt
        result2 = injector2.inject("内容2", result1)

        # 应该包含两个命名空间的内容
        assert "[PLUGIN1 CONTEXT]" in result2
        assert "[PLUGIN2 CONTEXT]" in result2
        assert "内容1" in result2
        assert "内容2" in result2

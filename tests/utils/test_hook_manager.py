"""
测试Hook管理器
测试 iris_memory.utils.hook_manager 中的 MemoryInjector 类
"""

import pytest

from iris_memory.utils.hook_manager import (
    InjectionMode,
    HookPriority,
    MemoryInjector
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



"""
Hook管理器
管理LLM Hook的执行和记忆注入策略
"""

from enum import Enum
from typing import List, Dict, Any, Optional


class InjectionMode(str, Enum):
    """记忆注入模式"""
    
    # 前置模式：在system_prompt最开始注入
    PREFIX = "prefix"
    
    # 后置模式：在system_prompt末尾注入
    SUFFIX = "suffix"
    
    # 嵌入模式：嵌入到system_prompt的特定位置
    EMBEDDED = "embedded"
    
    # 混合模式：结合前置和后置
    HYBRID = "hybrid"


class HookPriority(str, Enum):
    """Hook优先级"""
    
    # 最高优先级：应该最先执行
    CRITICAL = "critical"
    
    # 高优先级：早期执行
    HIGH = "high"
    
    # 正常优先级：默认执行顺序
    NORMAL = "normal"
    
    # 低优先级：后期执行
    LOW = "low"


class MemoryInjector:
    """记忆注入器
    
    管理记忆注入到LLM prompt的方式和时机
    """
    
    def __init__(
        self,
        injection_mode: InjectionMode = InjectionMode.SUFFIX,
        priority: HookPriority = HookPriority.NORMAL,
        namespace: str = "iris_memory"
    ):
        """初始化注入器
        
        Args:
            injection_mode: 注入模式
            priority: Hook优先级
            namespace: 命名空间（用于内容隔离）
        """
        self.injection_mode = injection_mode
        self.priority = priority
        self.namespace = namespace
        
        # 注入配置
        self.enable_injection = True
        self.enable_structured = True
        self.max_injection_length = 1000  # 最大注入长度（字符）
    
    def inject(
        self,
        memory_context: str,
        system_prompt: str,
        req: Any = None
    ) -> str:
        """注入记忆上下文
        
        Args:
            memory_context: 记忆上下文文本
            system_prompt: 原始system_prompt
            req: LLM请求对象（可选）
            
        Returns:
            str: 注入后的system_prompt
        """
        if not memory_context or not self.enable_injection:
            return system_prompt
        
        # 检查长度限制
        if len(memory_context) > self.max_injection_length:
            from astrbot.api import logger
            logger.warning(
                f"Memory context too long ({len(memory_context)} chars), "
                f"truncating to {self.max_injection_length} chars"
            )
            memory_context = memory_context[:self.max_injection_length] + "..."
        
        # 根据模式注入
        if self.injection_mode == InjectionMode.PREFIX:
            return self._inject_prefix(memory_context, system_prompt)
        elif self.injection_mode == InjectionMode.SUFFIX:
            return self._inject_suffix(memory_context, system_prompt)
        elif self.injection_mode == InjectionMode.EMBEDDED:
            return self._inject_embedded(memory_context, system_prompt)
        elif self.injection_mode == InjectionMode.HYBRID:
            return self._inject_hybrid(memory_context, system_prompt)
        else:
            # 默认使用后置模式
            return self._inject_suffix(memory_context, system_prompt)
    
    def _inject_prefix(self, memory_context: str, system_prompt: str) -> str:
        """前置模式注入
        
        记忆上下文放在system_prompt的最开始
        """
        if self.enable_structured:
            return f"[{self.namespace.upper()} CONTEXT]\n{memory_context}\n\n{system_prompt}"
        else:
            return f"{memory_context}\n\n{system_prompt}"
    
    def _inject_suffix(self, memory_context: str, system_prompt: str) -> str:
        """后置模式注入
        
        记忆上下文放在system_prompt的末尾
        """
        if self.enable_structured:
            return f"{system_prompt}\n\n[{self.namespace.upper()} CONTEXT]\n{memory_context}"
        else:
            return f"{system_prompt}\n\n{memory_context}"
    
    def _inject_embedded(self, memory_context: str, system_prompt: str) -> str:
        """嵌入式注入
        
        尝试在system_prompt中找到合适位置嵌入
        简化版本：如果包含"Context"或"上下文"关键词，则在其附近插入
        """
        # 查找合适的位置
        import re
        pattern = r'(上下文|context|Context|BACKGROUND|背景)'
        match = re.search(pattern, system_prompt, re.IGNORECASE)
        
        if match:
            # 在匹配位置附近插入
            pos = match.start()
            if self.enable_structured:
                return (
                    system_prompt[:pos] +
                    f"[{self.namespace.upper()}]\n{memory_context}\n" +
                    system_prompt[pos:]
                )
            else:
                return (
                    system_prompt[:pos] +
                    f"{memory_context}\n" +
                    system_prompt[pos:]
                )
        
        # 如果没找到合适位置，使用后置模式
        return self._inject_suffix(memory_context, system_prompt)
    
    def _inject_hybrid(self, memory_context: str, system_prompt: str) -> str:
        """混合模式注入
        
        前置：简短提示（如"参考以下记忆"）
        后置：完整的记忆上下文
        """
        if self.enable_structured:
            prefix = f"[{self.namespace.upper()} REFERENCE]\n"
            suffix = f"\n\n[{self.namespace.upper()} DETAILS]\n{memory_context}"
            return prefix + system_prompt + suffix
        else:
            prefix = "参考以下记忆：\n"
            suffix = f"\n\n{memory_context}"
            return prefix + system_prompt + suffix
    
    def get_priority_hint(self) -> str:
        """获取优先级提示
        
        用于调试和日志
        
        Returns:
            str: 优先级描述
        """
        return f"[{self.namespace}] Mode: {self.injection_mode.value}, Priority: {self.priority.value}"
    
    def parse_existing_context(
        self,
        system_prompt: str
    ) -> List[Dict[str, Any]]:
        """解析system_prompt中已有的上下文
        
        Args:
            system_prompt: system_prompt文本
            
        Returns:
            List[Dict]: 找到的上下文信息列表
            [{"namespace": str, "content": str, "mode": str}]
        """
        import re
        
        # 匹配结构化上下文
        pattern = r'\[([A-Z_]+)\s*(CONTEXT|REFERENCE|DETAILS)?\]\n(.*?)(?=\n\n\[|$)'
        matches = re.findall(pattern, system_prompt, re.DOTALL)
        
        contexts = []
        for match in matches:
            contexts.append({
                "namespace": match[0].lower(),
                "type": match[1] or "CONTEXT",
                "content": match[2].strip()
            })
        
        return contexts
    
    def detect_conflicts(
        self,
        system_prompt: str,
        my_content: str
    ) -> Dict[str, Any]:
        """检测内容冲突
        
        Args:
            system_prompt: system_prompt文本
            my_content: 我的注入内容
            
        Returns:
            Dict: 冲突检测结果
        """
        existing_contexts = self.parse_existing_context(system_prompt)
        
        conflicts = {
            "namespace_conflict": False,
            "content_conflict": False,
            "suggestions": []
        }
        
        # 检查命名空间冲突
        for ctx in existing_contexts:
            if ctx["namespace"] == self.namespace and ctx["content"] != my_content.strip():
                conflicts["namespace_conflict"] = True
                conflicts["suggestions"].append(
                    f"命名空间'{self.namespace}'已被其他插件使用，考虑修改命名空间"
                )
        
        # 检查内容相似性
        for ctx in existing_contexts:
            similarity = self._calculate_similarity(my_content, ctx["content"])
            if similarity > 0.8:  # 高相似度
                conflicts["content_conflict"] = True
                conflicts["suggestions"].append(
                    f"与'{ctx['namespace']}'的内容高度相似（{similarity:.2f}），"
                    "可能产生冗余"
                )
        
        return conflicts
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版Jaccard）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度（0-1）
        """
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union


class HookCoordinator:
    """Hook协调器
    
    协调多个插件的Hook执行
    """
    
    def __init__(self):
        """初始化协调器"""
        self.registered_hooks = {}  # {priority: [handlers]}
    
    def register_hook(self, handler, priority: HookPriority):
        """注册Hook
        
        Args:
            handler: 处理器函数
            priority: 优先级
        """
        if priority not in self.registered_hooks:
            self.registered_hooks[priority] = []
        
        self.registered_hooks[priority].append(handler)
    
    def execute_hooks(self, *args, **kwargs) -> List[Any]:
        """按优先级执行Hook
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            List[Any]: 所有Hook的返回值
        """
        results = []
        
        # 按优先级顺序执行
        priority_order = [
            HookPriority.CRITICAL,
            HookPriority.HIGH,
            HookPriority.NORMAL,
            HookPriority.LOW
        ]
        
        for priority in priority_order:
            if priority in self.registered_hooks:
                for handler in self.registered_hooks[priority]:
                    try:
                        result = handler(*args, **kwargs)
                        results.append(result)
                    except Exception as e:
                        from astrbot.api import logger
                        logger.error(
                            f"Hook execution failed (priority: {priority.value}): {e}"
                        )
        
        return results

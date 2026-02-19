"""
指令解析工具模块 - 统一处理指令解析逻辑
"""
from __future__ import annotations

from typing import Optional, Set, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from iris_memory.core.constants import DeleteMainScope


@dataclass(frozen=True)
class ParsedCommand:
    """解析后的指令结果"""
    raw_message: str
    content: str  # 去除指令前缀后的内容
    command: Optional[str]  # 识别到的指令名
    args: List[str]  # 参数列表
    
    @property
    def has_content(self) -> bool:
        """是否有有效内容"""
        return bool(self.content.strip())
    
    @property
    def first_arg(self) -> Optional[str]:
        """获取第一个参数"""
        return self.args[0] if self.args else None


class CommandParser:
    """指令解析器"""
    
    @staticmethod
    def parse(message: str, prefixes: Set[str]) -> ParsedCommand:
        """
        解析指令消息
        
        Args:
            message: 原始消息
            prefixes: 指令前缀集合
            
        Returns:
            ParsedCommand: 解析结果
        """
        stripped = message.strip()
        command: Optional[str] = None
        content: str = stripped
        args: List[str] = []
        
        # 检测并移除指令前缀
        for prefix in prefixes:
            prefix_stripped = prefix.lstrip('/')
            if stripped.startswith(prefix):
                command = prefix_stripped
                content = stripped[len(prefix):].strip()
                break
            elif stripped.startswith(prefix_stripped):
                command = prefix_stripped
                content = stripped[len(prefix_stripped):].strip()
                break
        
        # 解析参数
        if content:
            args = content.split()
        
        return ParsedCommand(
            raw_message=message,
            content=content,
            command=command,
            args=args
        )
    
    @staticmethod
    def parse_with_slash(message: str, command_name: str) -> ParsedCommand:
        """
        解析带斜杠的指令
        
        Args:
            message: 原始消息
            command_name: 指令名（不含斜杠）
            
        Returns:
            ParsedCommand: 解析结果
        """
        prefixes = {f"/{command_name}", command_name}
        return CommandParser.parse(message, prefixes)


class DeleteScopeParser:
    """删除范围解析器（群聊子范围）"""

    SCOPE_MAP = {
        "shared": "group_shared",
        "private": "group_private",
        "all": None,
    }

    @classmethod
    def parse(cls, param: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """
        解析删除范围参数

        Args:
            param: 用户输入的参数

        Returns:
            tuple: (scope_filter, scope_desc)
                - scope_filter: 用于数据库过滤的值
                - scope_desc: 用于显示的中文描述
        """
        if not param:
            return None, "所有"

        normalized = param.lower()
        scope_filter = cls.SCOPE_MAP.get(normalized)

        scope_desc_map = {
            "shared": "共享",
            "private": "个人",
            "all": "所有",
        }
        scope_desc = scope_desc_map.get(normalized, "所有")

        return scope_filter, scope_desc

    @classmethod
    def is_valid(cls, param: str) -> bool:
        """检查参数是否有效"""
        return param.lower() in cls.SCOPE_MAP


@dataclass(frozen=True)
class UnifiedDeleteResult:
    """统一删除指令解析结果"""
    main_scope: DeleteMainScope        # 主范围
    group_sub_scope: Optional[str]     # 群聊子范围 (shared/private/all)
    scope_filter: Optional[str]        # 数据库过滤值
    scope_desc: str                    # 中文描述
    is_valid: bool                     # 是否有效
    error_message: Optional[str]       # 错误消息


class UnifiedDeleteScopeParser:
    """统一删除范围解析器"""

    MAIN_SCOPES = {"current", "private", "group", "all"}
    GROUP_SUB_SCOPES = {"shared", "private", "all"}

    @classmethod
    def parse(
        cls,
        args: List[str],
        has_confirm: bool = False
    ) -> UnifiedDeleteResult:
        """
        解析统一删除指令参数

        Args:
            args: 参数列表
            has_confirm: 是否已包含 confirm 参数

        Returns:
            UnifiedDeleteResult: 解析结果
        """
        from iris_memory.core.constants import DeleteMainScope  # lazy import

        if not args:
            # 默认删除当前会话
            return UnifiedDeleteResult(
                main_scope=DeleteMainScope.CURRENT,
                group_sub_scope=None,
                scope_filter=None,
                scope_desc="当前会话",
                is_valid=True,
                error_message=None
            )

        first_arg = args[0].lower()

        # 检查主范围是否有效
        if first_arg not in cls.MAIN_SCOPES:
            return UnifiedDeleteResult(
                main_scope=DeleteMainScope.CURRENT,
                group_sub_scope=None,
                scope_filter=None,
                scope_desc="",
                is_valid=False,
                error_message="参数错误，可用范围: current, private, group [shared|private|all], all confirm"
            )

        # current: 当前会话
        if first_arg == "current":
            return UnifiedDeleteResult(
                main_scope=DeleteMainScope.CURRENT,
                group_sub_scope=None,
                scope_filter=None,
                scope_desc="当前会话",
                is_valid=True,
                error_message=None
            )

        # private: 私聊记忆
        if first_arg == "private":
            return UnifiedDeleteResult(
                main_scope=DeleteMainScope.PRIVATE,
                group_sub_scope=None,
                scope_filter=None,
                scope_desc="私聊",
                is_valid=True,
                error_message=None
            )

        # group: 群聊记忆
        if first_arg == "group":
            sub_scope = args[1].lower() if len(args) > 1 else "all"

            if sub_scope not in cls.GROUP_SUB_SCOPES:
                return UnifiedDeleteResult(
                    main_scope=DeleteMainScope.GROUP,
                    group_sub_scope=sub_scope,
                    scope_filter=None,
                    scope_desc="",
                    is_valid=False,
                    error_message="参数错误，请使用: shared, private 或 all"
                )

            scope_filter, scope_desc = DeleteScopeParser.parse(sub_scope)
            return UnifiedDeleteResult(
                main_scope=DeleteMainScope.GROUP,
                group_sub_scope=sub_scope,
                scope_filter=scope_filter,
                scope_desc=scope_desc,
                is_valid=True,
                error_message=None
            )

        # all: 所有记忆（需要 confirm）
        if first_arg == "all":
            return UnifiedDeleteResult(
                main_scope=DeleteMainScope.ALL,
                group_sub_scope=None,
                scope_filter=None,
                scope_desc="所有",
                is_valid=True,
                error_message=None
            )

        # 不应该到达这里
        return UnifiedDeleteResult(
            main_scope=DeleteMainScope.CURRENT,
            group_sub_scope=None,
            scope_filter=None,
            scope_desc="",
            is_valid=False,
            error_message="未知错误"
        )


class StatsFormatter:
    """统计信息格式化器"""
    
    @staticmethod
    def format_memory_stats(
        working_count: int,
        episodic_count: int,
        image_analyzed: int = 0,
        cache_hits: int = 0
    ) -> str:
        """
        格式化记忆统计信息
        
        Args:
            working_count: 工作记忆数量
            episodic_count: 情景记忆数量
            image_analyzed: 图片分析数量
            cache_hits: 缓存命中次数
            
        Returns:
            str: 格式化后的统计文本
        """
        lines = [
            "记忆统计：",
            f"- 工作记忆：{working_count} 条",
            f"- 情景记忆：{episodic_count} 条",
        ]
        
        if image_analyzed > 0:
            lines.append(f"- 图片分析：{image_analyzed} 张")
        if cache_hits > 0:
            lines.append(f"- 缓存命中：{cache_hits} 次")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_search_results(memories: List) -> str:
        """
        格式化搜索结果
        
        Args:
            memories: 记忆对象列表
            
        Returns:
            str: 格式化后的搜索结果
        """
        if not memories:
            return "未找到相关记忆"
        
        lines = [f"找到 {len(memories)} 条相关记忆：\n"]
        
        for i, memory in enumerate(memories, 1):
            time_str = memory.created_time.strftime("%m-%d %H:%M")
            lines.append(f"{i}. [{memory.type.value.upper()}] {time_str}")
            lines.append(f"   {memory.content}\n")
        
        return "\n".join(lines)


class SessionKeyBuilder:
    """会话键构建器"""
    
    SEPARATOR: str = ":"
    PRIVATE: str = "private"
    
    @classmethod
    def build(cls, user_id: str, group_id: Optional[str]) -> str:
        """
        构建会话键
        
        Args:
            user_id: 用户ID
            group_id: 群聊ID（私聊为None）
            
        Returns:
            str: 会话键
        """
        if group_id:
            return f"{user_id}{cls.SEPARATOR}{group_id}"
        return f"{user_id}{cls.SEPARATOR}{cls.PRIVATE}"
    
    @classmethod
    def build_for_kv(cls, user_id: str, group_id: Optional[str]) -> str:
        """
        构建用于KV存储的键
        
        Args:
            user_id: 用户ID
            group_id: 群聊ID
            
        Returns:
            str: KV存储键
        """
        group_suffix = group_id if group_id else "private"
        return f"last_save_{user_id}_{group_suffix}"


class MessageFilter:
    """消息过滤器"""

    KNOWN_COMMANDS: Set[str] = frozenset([
        "memory_save", "memory_search", "memory_clear", "memory_stats",
        "memory_delete", "proactive_reply"
    ])
    
    @classmethod
    def is_command(cls, message: str) -> bool:
        """检查是否为指令消息"""
        stripped = message.strip()
        
        # 检查斜杠开头
        if stripped.startswith('/'):
            return True
        
        # 检查已知指令
        for cmd in cls.KNOWN_COMMANDS:
            if stripped.startswith(cmd):
                return True
        
        return False
    
    @classmethod
    def should_skip_message_processing(cls, message: str) -> bool:
        """
        检查是否应跳过消息处理
        
        用于批量消息处理器过滤指令
        """
        return cls.is_command(message)

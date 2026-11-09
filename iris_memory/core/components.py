"""
Iris Chat Memory - 组件初始化框架

提供统一的组件生命周期管理，支持：
- 抽象基类定义
- 细粒度状态追踪
- 独立初始化与故障隔离
- 异步生命周期管理
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, TypeVar, overload
import asyncio

from .logger import get_logger

logger = get_logger("components")


# ============================================================================
# 枚举定义
# ============================================================================


class ComponentStatus(Enum):
    """组件初始化状态"""

    PENDING = "pending"
    INITIALIZING = "initializing"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class InitMode(Enum):
    """组件初始化模式

    EAGER: 同步初始化，阻塞主流程，适用于轻量组件（L1Buffer、LLMManager 等）
    BACKGROUND: 后台初始化，不阻塞主流程，适用于重型组件（ChromaDB 等）
    """

    EAGER = "eager"
    BACKGROUND = "background"


class ErrorType(Enum):
    """错误类型分类"""

    DISABLED = "disabled"
    DEPENDENCY_MISSING = "dependency_missing"
    CONNECTION_FAILED = "connection_failed"
    OTHER = "other"


# ============================================================================
# 数据类定义
# ============================================================================


@dataclass
class ComponentState:
    """单个组件的详细状态"""

    status: ComponentStatus = ComponentStatus.PENDING
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None

    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
        }


@dataclass
class SystemStatus:
    """系统各模块可用性状态

    细粒度追踪各层功能模块的可用性，替代单一布尔值，
    便于上层逻辑根据实际状态灵活降级。

    使用字典集中管理模块可用性，由 ComponentManager 动态注册。

    Attributes:
        _availability: 模块可用性字典 {module_name: bool}
        _states: 组件详细状态字典 {module_name: ComponentState}
        _global_status: 全局初始化状态

    Examples:
        >>> status = SystemStatus()
        >>> status.register_module("l2_memory", False)
        >>> status.set_available("l2_memory")
        >>> status.get_available_modules()
        ['l2_memory']
        >>> status.is_module_available("l2_memory")
        True
    """

    _availability: Dict[str, bool] = field(default_factory=dict)
    _states: Dict[str, ComponentState] = field(default_factory=dict)
    _global_status: ComponentStatus = ComponentStatus.PENDING

    def register_module(
        self, module_name: str, default_available: bool = False
    ) -> None:
        """注册模块

        Args:
            module_name: 模块名称
            default_available: 默认是否可用
        """
        self._availability[module_name] = default_available
        if module_name not in self._states:
            self._states[module_name] = ComponentState()

    def set_available(self, module_name: str, available: bool = True) -> None:
        """设置模块可用性

        Args:
            module_name: 模块名称
            available: 是否可用
        """
        if module_name in self._availability:
            self._availability[module_name] = available
        if module_name in self._states:
            self._states[module_name].status = (
                ComponentStatus.AVAILABLE if available else ComponentStatus.UNAVAILABLE
            )

    def set_state(self, module_name: str, state: ComponentState) -> None:
        """设置组件详细状态

        Args:
            module_name: 模块名称
            state: 组件状态
        """
        self._states[module_name] = state

    def get_state(self, module_name: str) -> ComponentState:
        """获取组件状态

        Args:
            module_name: 模块名称

        Returns:
            组件状态
        """
        return self._states.get(module_name, ComponentState())

    def is_module_available(self, module_name: str) -> bool:
        """检查指定模块是否可用

        Args:
            module_name: 模块名称

        Returns:
            模块是否可用

        Examples:
            >>> status.is_module_available("l2")
            False
            >>> status.is_module_available("l1")
            True
        """
        return self._availability.get(module_name, False)

    def get_available_modules(self) -> List[str]:
        """获取可用模块名称列表

        Returns:
            可用模块的名称列表

        Examples:
            >>> status.get_available_modules()
            ['l1']
        """
        return [m for m, available in self._availability.items() if available]

    def get_unavailable_modules(self) -> List[str]:
        """获取不可用模块名称列表

        Returns:
            不可用模块的名称列表

        Examples:
            >>> status.get_unavailable_modules()
            ['l2', 'l3', 'profile', 'scheduler', 'image_quota']
        """
        return [m for m, available in self._availability.items() if not available]

    def set_global_status(self, status: ComponentStatus) -> None:
        """设置全局初始化状态

        Args:
            status: 全局状态
        """
        self._global_status = status

    @property
    def global_status(self) -> ComponentStatus:
        """获取全局初始化状态"""
        return self._global_status

    def to_dict(self) -> Dict[str, bool]:
        """转换为字典

        Returns:
            状态字典 {module_name_available: bool}
        """
        return {
            f"{m}_available": available for m, available in self._availability.items()
        }

    def to_detailed_dict(self) -> Dict:
        """转换为详细字典

        Returns:
            详细状态字典，包含所有组件状态
        """
        return {
            "components": {
                name: state.to_dict() for name, state in self._states.items()
            },
            "global_status": self._global_status.value,
        }


@dataclass
class ComponentInitResult:
    """组件初始化结果

    记录单个组件的初始化状态，用于日志追踪和状态汇总。

    Attributes:
        name: 组件名称
        success: 是否成功
        error_message: 错误信息（失败时）

    Examples:
        >>> result = ComponentInitResult("l2_memory", True)
        >>> result.success
        True
    """

    name: str
    success: bool
    error_message: Optional[str] = None


# ============================================================================
# 工具函数
# ============================================================================


def classify_error(error_msg: str) -> ErrorType:
    """根据错误信息分类错误类型

    Args:
        error_msg: 错误信息字符串

    Returns:
        错误类型枚举
    """
    error_lower = error_msg.lower()

    disabled_keywords = [
        "disabled",
        "not enabled",
        "未启用",
        "已禁用",
        "未开启",
    ]
    if any(keyword in error_lower for keyword in disabled_keywords):
        return ErrorType.DISABLED

    dependency_keywords = [
        "no module named",
        "importerror",
        "modulenotfounderror",
        "missing",
        "not found",
        "dependency",
        "required",
        "未安装",
        "找不到模块",
        "依赖缺失",
        "缺少依赖",
    ]
    if any(keyword in error_lower for keyword in dependency_keywords):
        return ErrorType.DEPENDENCY_MISSING

    connection_keywords = [
        "connection",
        "connect",
        "timeout",
        "refused",
        "unreachable",
        "network",
        "socket",
        "无法连接",
        "连接失败",
        "超时",
        "网络错误",
    ]
    if any(keyword in error_lower for keyword in connection_keywords):
        return ErrorType.CONNECTION_FAILED

    return ErrorType.OTHER


# ============================================================================
# 抽象基类
# ============================================================================

_C = TypeVar("_C", bound="Component")


class Component(ABC):
    """组件抽象基类

    所有组件（ChromaDB、画像存储等）需继承此类，
    实现统一的初始化和关闭接口。

    Attributes:
        _is_available: 组件是否可用
        _init_error: 初始化错误信息

    Examples:
        >>> class ExampleComponent(Component):
        ...     @property
        ...     def name(self) -> str:
        ...         return "example"
        ...
        ...     async def initialize(self) -> None:
        ...         self._is_available = True
        ...
        ...     async def shutdown(self) -> None:
        ...         self._is_available = False
    """

    def __init__(self):
        self._is_available: bool = False
        self._init_error: Optional[str] = None
        self._status: ComponentStatus = ComponentStatus.PENDING
        self._init_mode: InitMode = InitMode.EAGER

    @property
    @abstractmethod
    def name(self) -> str:
        """组件名称，用于日志标识

        Returns:
            组件名称字符串
        """
        pass

    @property
    def init_mode(self) -> InitMode:
        """组件初始化模式

        Returns:
            初始化模式枚举，默认 EAGER
        """
        return self._init_mode

    @property
    def is_background_init(self) -> bool:
        """是否为后台初始化组件

        Returns:
            是否后台初始化
        """
        return self._init_mode == InitMode.BACKGROUND

    @property
    def is_available(self) -> bool:
        """组件是否可用

        Returns:
            可用状态
        """
        return self._is_available

    @property
    def status(self) -> ComponentStatus:
        """组件初始化状态

        Returns:
            组件状态
        """
        return self._status

    @property
    def init_error(self) -> Optional[str]:
        """初始化错误信息

        Returns:
            错误信息字符串，无错误时为 None
        """
        return self._init_error

    @property
    def error_type(self) -> Optional[ErrorType]:
        """错误类型分类

        Returns:
            错误类型枚举
        """
        if self._init_error:
            return classify_error(self._init_error)
        return None

    @property
    def is_initializing(self) -> bool:
        """组件是否正在初始化中

        Returns:
            是否正在初始化
        """
        return self._status == ComponentStatus.INITIALIZING

    @property
    def is_disabled(self) -> bool:
        """组件是否被禁用（配置未启用）

        Returns:
            是否被禁用
        """
        return (
            not self._is_available
            and self._init_error is not None
            and self.error_type == ErrorType.DISABLED
        )

    def get_state(self) -> ComponentState:
        """获取组件详细状态

        Returns:
            组件状态对象
        """
        state = ComponentState(
            status=self._status, error=self._init_error, error_type=self.error_type
        )
        return state

    @abstractmethod
    async def initialize(self) -> None:
        """初始化组件

        子类需实现此方法，完成资源初始化（如数据库连接、文件创建等）。
        成功时设置 `_is_available = True`，失败时设置 `_init_error`。

        Raises:
            Exception: 初始化失败时抛出异常，由 ComponentManager 捕获
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """关闭组件

        子类需实现此方法，完成资源释放（如关闭连接，保存状态等）。

        Note:
            子类实现应在最后调用 self._reset_state() 重置状态。
        """
        pass

    def _reset_state(self) -> None:
        """重置组件状态

        在 shutdown 时调用，确保状态一致性。
        """
        self._is_available = False
        self._status = ComponentStatus.PENDING
        self._init_error = None


# ============================================================================
# 组件管理器
# ============================================================================


class ComponentManager:
    """组件生命周期管理器

    负责协调多个组件的初始化和关闭，提供故障隔离和状态追踪。

    Features:
        - 独立初始化：单组件失败不影响其他组件
        - 状态追踪：细粒度追踪各层可用性
        - 线程安全：使用 asyncio.Lock 保护关键操作
        - 组件查询：按名称或可用性获取组件

    Attributes:
        _components: 组件元组（不可变）
        _initialized: 是否已初始化
        _lock: 异步锁
        _status: 系统状态

    Examples:
        >>> from iris_memory.storage import ChromaAdapter, KuzuAdapter
        >>>
        >>> components = (ChromaAdapter(), KuzuAdapter())
        >>> manager = ComponentManager(components)
        >>>
        >>> # 初始化所有组件
        >>> results = await manager.initialize_all()
        >>>
        >>> # 查询状态
        >>> status = manager.status
        >>> print(status.to_dict())
    """

    def __init__(self, components: Tuple[Component, ...]):
        """初始化组件管理器

        Args:
            components: 组件元组，按初始化顺序排列
        """
        self._components = components
        self._initialized = False
        self._initializing = False
        self._lock = asyncio.Lock()
        self._status = SystemStatus()
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._background_results: Dict[str, ComponentInitResult] = {}

        for component in components:
            self._status.register_module(component.name, default_available=False)

        logger.info(
            f"组件管理器已创建，共 {len(components)} 个组件，已注册模块：{list(self._status._availability.keys())}"
        )

    @property
    def status(self) -> SystemStatus:
        """获取系统状态

        Returns:
            系统状态对象
        """
        return self._status

    @property
    def is_initialized(self) -> bool:
        """是否已完成初始化"""
        return self._initialized

    @property
    def is_initializing(self) -> bool:
        """是否正在初始化"""
        return self._initializing

    async def initialize_all(self) -> List[ComponentInitResult]:
        """初始化所有组件

        EAGER 组件按顺序同步初始化（阻塞主流程），
        BACKGROUND 组件在后台异步初始化（不阻塞主流程）。
        单组件失败不影响其他组件。

        Returns:
            同步初始化组件的结果列表（后台组件结果通过
            get_background_init_results 获取）

        Raises:
            RuntimeError: 已初始化时抛出

        Notes:
            - 使用 asyncio.Lock 保护初始化过程
            - 每个组件的初始化错误会被捕获并记录
            - 后台组件初始化完成后自动更新系统状态
        """
        async with self._lock:
            if self._initialized:
                raise RuntimeError("组件已初始化，请勿重复调用")

            self._initializing = True
            self._status.set_global_status(ComponentStatus.INITIALIZING)

            for component in self._components:
                component._status = ComponentStatus.INITIALIZING

            logger.info("开始初始化组件...")

            eager_results: List[ComponentInitResult] = []
            background_components: List[Component] = []

            for component in self._components:
                if component.init_mode == InitMode.BACKGROUND:
                    background_components.append(component)
                else:
                    result = await self._init_single_component(component)
                    eager_results.append(result)

            for component in background_components:
                task = asyncio.create_task(
                    self._init_background_component(component),
                    name=f"init_{component.name}",
                )
                self._background_tasks[component.name] = task
                logger.info(f"组件 {component.name} 已提交后台初始化")

            self._update_status()
            self._initialized = True
            self._initializing = False
            self._status.set_global_status(ComponentStatus.AVAILABLE)

            eager_success = sum(1 for r in eager_results if r.success)
            bg_count = len(background_components)
            logger.info(
                f"同步组件初始化完成：{eager_success}/{len(eager_results)} 成功，"
                f"{bg_count} 个后台组件初始化中，"
                f"当前可用模块：{', '.join(self._status.get_available_modules())}"
            )

            return eager_results

    async def _init_background_component(
        self, component: Component
    ) -> ComponentInitResult:
        """后台初始化单个组件

        Args:
            component: 组件实例

        Returns:
            初始化结果
        """
        result = await self._init_single_component(component)
        self._background_results[component.name] = result

        self._update_status()

        if result.success:
            logger.info(
                f"后台组件 {component.name} 初始化完成，"
                f"当前可用模块：{', '.join(self._status.get_available_modules())}"
            )
        else:
            logger.warning(
                f"后台组件 {component.name} 初始化失败：{result.error_message}"
            )

        return result

    @property
    def background_init_pending(self) -> List[str]:
        """仍在后台初始化中的组件名称列表"""
        return [
            name for name, task in self._background_tasks.items() if not task.done()
        ]

    @property
    def all_background_init_done(self) -> bool:
        """所有后台组件是否已完成初始化（成功或失败）"""
        return all(task.done() for task in self._background_tasks.values())

    def get_background_init_results(self) -> Dict[str, ComponentInitResult]:
        """获取后台组件初始化结果

        Returns:
            组件名称到初始化结果的映射
        """
        return dict(self._background_results)

    async def wait_for_background_init(self, timeout: Optional[float] = None) -> None:
        """等待所有后台组件初始化完成

        Args:
            timeout: 超时时间（秒），None 表示无限等待
        """
        if not self._background_tasks:
            return

        tasks = list(self._background_tasks.values())
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pending = self.background_init_pending
            if pending:
                logger.warning(f"等待后台组件初始化超时，未完成：{pending}")

    async def _init_single_component(self, component: Component) -> ComponentInitResult:
        """初始化单个组件

        Args:
            component: 组件实例

        Returns:
            初始化结果
        """
        try:
            logger.debug(f"正在初始化组件：{component.name}")
            await component.initialize()

            if component.is_available:
                component._status = ComponentStatus.AVAILABLE
                logger.info(f"组件 {component.name} 初始化成功")
                return ComponentInitResult(component.name, True)
            else:
                error_msg = component.init_error or "初始化后组件不可用"
                component._status = ComponentStatus.UNAVAILABLE
                logger.warning(f"组件 {component.name} 初始化失败：{error_msg}")
                return ComponentInitResult(component.name, False, error_msg)

        except Exception as e:
            error_msg = str(e)
            component._init_error = error_msg
            component._status = ComponentStatus.UNAVAILABLE
            logger.error(
                f"组件 {component.name} 初始化异常：{error_msg}", exc_info=True
            )
            return ComponentInitResult(component.name, False, error_msg)

    def _update_status(self) -> None:
        """更新系统状态

        根据各组件的可用性更新 SystemStatus。先在局部构建完整的新状态对象，
        再原子替换 self._status，避免后台组件并发完成初始化时读到半重建的
        中间状态（如某组件刚初始化成功却被报为不可用）。
        """
        previous_global = self._status.global_status
        new_status = SystemStatus()
        for component in self._components:
            new_status.register_module(component.name, default_available=False)
            new_status.set_state(component.name, component.get_state())

        for component in self._components:
            if component.is_available:
                new_status.set_available(component.name, True)

        new_status.set_global_status(previous_global)
        self._status = new_status

    async def shutdown_all(self) -> None:
        """关闭所有组件

        按组件在元组中的逆序依次关闭。
        即使部分组件关闭失败，也会继续关闭其他组件。

        Notes:
            - 使用 asyncio.Lock 保护关闭过程
            - 关闭后重置初始化状态
            - 取消所有后台初始化任务
        """
        async with self._lock:
            if not self._initialized:
                logger.debug("组件未初始化，无需关闭")
                return

            logger.info("开始关闭组件...")

            for name, task in self._background_tasks.items():
                if not task.done():
                    task.cancel()
                    logger.debug(f"已取消后台初始化任务：{name}")

            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks.values(), return_exceptions=True
                )

            for component in reversed(self._components):
                try:
                    logger.debug(f"正在关闭组件：{component.name}")
                    await component.shutdown()
                    logger.info(f"组件 {component.name} 已关闭")
                except Exception as e:
                    logger.error(f"组件 {component.name} 关闭失败：{e}", exc_info=True)

            self._status = SystemStatus()
            for component in self._components:
                self._status.register_module(component.name, default_available=False)
            self._initialized = False
            self._status.set_global_status(ComponentStatus.PENDING)
            self._background_tasks.clear()
            self._background_results.clear()

            logger.info("所有组件已关闭")

    @overload
    def get_component(self, name: str) -> Optional[Component]: ...

    @overload
    def get_component(self, name: str, expected_type: type[_C]) -> Optional[_C]: ...

    def get_component(
        self, name: str, expected_type: type[_C] | None = None
    ) -> Optional[Component] | Optional[_C]:
        """按名称获取组件

        Args:
            name: 组件名称
            expected_type: 期望的组件类型（可选），传入后返回值类型为该类型的 Optional

        Returns:
            组件实例，找不到时返回 None
        """
        for component in self._components:
            if component.name == name:
                if expected_type is not None:
                    return component if isinstance(component, expected_type) else None
                return component
        return None

    @overload
    def get_available_component(self, name: str) -> Optional[Component]: ...

    @overload
    def get_available_component(
        self, name: str, expected_type: type[_C]
    ) -> Optional[_C]: ...

    def get_available_component(
        self, name: str, expected_type: type[_C] | None = None
    ) -> Optional[Component] | Optional[_C]:
        """获取已可用的组件实例

        便捷方法，仅当组件存在且 is_available=True 时返回。
        适用于业务调用点需要"可用或 None"的简化判断。

        Args:
            name: 组件名称
            expected_type: 期望的组件类型（可选），传入后返回值类型为该类型的 Optional

        Returns:
            可用的组件实例，不可用时返回 None
        """
        component = self.get_component(name, expected_type)  # type: ignore[arg-type]
        if component and component.is_available:
            return component
        return None

    def check_component(self, name: str) -> str:
        """检查组件状态，返回状态标识

        适用于业务调用点需要区分三种退化状态的场景：
        - "available": 组件可用
        - "initializing": 组件正在后台初始化中
        - "disabled": 组件被配置禁用
        - "unavailable": 组件初始化失败或不存在

        Args:
            name: 组件名称

        Returns:
            状态标识字符串
        """
        component = self.get_component(name)
        if component is None:
            return "unavailable"
        if component.is_available:
            return "available"
        if component.is_initializing:
            return "initializing"
        if component.is_disabled:
            return "disabled"
        return "unavailable"

    def get_available_components(self) -> List[Component]:
        """获取所有可用组件

        Returns:
            可用组件列表
        """
        return [c for c in self._components if c.is_available]

    def get_failed_components(self) -> List[Component]:
        """获取所有失败的组件

        Returns:
            失败组件列表
        """
        return [c for c in self._components if not c.is_available]

    def get_all_states(self) -> Dict[str, Dict]:
        """获取所有组件的详细状态

        Returns:
            组件状态字典
        """
        return {
            component.name: component.get_state().to_dict()
            for component in self._components
        }

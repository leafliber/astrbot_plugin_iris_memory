"""
图片分析缓存和预算控制模块

将缓存和预算控制逻辑从 ImageAnalyzer 中拆分出来，提高代码可维护性。
"""

import hashlib
import time
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from collections import deque
from dataclasses import dataclass

from iris_memory.utils.logger import get_logger

logger = get_logger("image_cache")


@dataclass
class ImageInfo:
    """图片信息"""
    url: str = ""
    file: str = ""
    is_sticker: bool = False
    width: Optional[int] = None
    height: Optional[int] = None


class ImageAnalysisLevel:
    """图片分析层级"""
    SKIP = "skip"
    BRIEF = "brief"
    DETAILED = "detailed"


@dataclass
class ImageAnalysisResult:
    """图片分析结果"""
    level: str = ImageAnalysisLevel.SKIP
    description: str = ""
    emotions: List[str] = None
    objects: List[str] = None
    context_relevance: float = 0.0
    token_cost: int = 0
    cached: bool = False
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.emotions is None:
            self.emotions = []
        if self.objects is None:
            self.objects = []


class ImageCacheManager:
    """图片分析缓存管理器
    
    职责：
    1. 分析结果缓存（基于URL哈希）
    2. 缓存TTL管理
    3. 缓存大小控制
    """
    
    def __init__(
        self,
        cache_ttl: int = 3600,
        max_cache_size: int = 200
    ):
        """初始化缓存管理器
        
        Args:
            cache_ttl: 缓存过期时间（秒）
            max_cache_size: 最大缓存数量
        """
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._analysis_cache: Dict[str, Tuple[ImageAnalysisResult, float]] = {}
    
    def get_image_hash(self, image_info: ImageInfo) -> Optional[str]:
        """获取图片哈希用于缓存
        
        Args:
            image_info: 图片信息
            
        Returns:
            Optional[str]: 哈希值或None
        """
        identifier = image_info.url or image_info.file
        if identifier:
            return hashlib.md5(identifier.encode()).hexdigest()[:16]
        return None
    
    def get_from_cache(self, image_hash: Optional[str]) -> Optional[ImageAnalysisResult]:
        """从缓存获取分析结果
        
        Args:
            image_hash: 图片哈希
            
        Returns:
            Optional[ImageAnalysisResult]: 缓存的结果或None
        """
        if not image_hash:
            return None
        
        cached = self._analysis_cache.get(image_hash)
        if cached:
            result, timestamp = cached
            if time.time() - timestamp < self.cache_ttl:
                return ImageAnalysisResult(
                    level=result.level,
                    description=result.description,
                    emotions=result.emotions.copy() if result.emotions else [],
                    objects=result.objects.copy() if result.objects else [],
                    context_relevance=result.context_relevance,
                    token_cost=0,
                    cached=True
                )
            else:
                del self._analysis_cache[image_hash]
        
        return None
    
    def add_to_cache(self, image_hash: str, result: ImageAnalysisResult) -> None:
        """添加到缓存
        
        Args:
            image_hash: 图片哈希
            result: 分析结果
        """
        if len(self._analysis_cache) >= self.max_cache_size:
            self._cleanup_cache()
        
        self._analysis_cache[image_hash] = (result, time.time())
    
    def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, ts) in self._analysis_cache.items()
            if current_time - ts > self.cache_ttl
        ]
        for k in expired_keys:
            del self._analysis_cache[k]
        
        if len(self._analysis_cache) >= self.max_cache_size:
            sorted_items = sorted(
                self._analysis_cache.items(),
                key=lambda x: x[1][1]
            )
            for k, _ in sorted_items[:len(sorted_items) // 2]:
                del self._analysis_cache[k]
    
    def clear(self) -> None:
        """清空缓存"""
        self._analysis_cache.clear()
    
    @property
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._analysis_cache)


class ImageBudgetManager:
    """图片分析预算管理器
    
    职责：
    1. 每日分析预算控制
    2. 会话分析预算控制
    3. 用户冷却时间管理
    """
    
    def __init__(
        self,
        daily_budget: int = 100,
        session_budget: int = 20,
        cooldown: float = 3.0
    ):
        """初始化预算管理器
        
        Args:
            daily_budget: 每日最大分析次数
            session_budget: 每会话最大分析次数
            cooldown: 用户分析冷却时间（秒）
        """
        self.daily_budget = daily_budget
        self.session_budget = session_budget
        self.cooldown = cooldown
        
        self._daily_count: Dict[str, int] = {}
        self._session_count: Dict[str, int] = {}
        self._last_analysis_time: Dict[str, float] = {}
    
    def check_budget(
        self,
        user_id: str,
        session_id: str = "",
        daily_budget_override: Optional[int] = None
    ) -> bool:
        """检查是否超出分析预算
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            daily_budget_override: 每日预算覆盖值
            
        Returns:
            bool: True表示预算内可以分析
        """
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = self._daily_count.get(today, 0)
        budget_limit = (
            daily_budget_override
            if daily_budget_override is not None
            else self.daily_budget
        )
        if daily_count >= budget_limit:
            logger.debug(f"Daily analysis budget exhausted: {daily_count}/{budget_limit}")
            return False
        
        if session_id and self.session_budget > 0:
            session_count = self._session_count.get(session_id, 0)
            if session_count >= self.session_budget:
                logger.debug(f"Session analysis budget exhausted: {session_count}/{self.session_budget}")
                return False
        
        return True
    
    def check_cooldown(self, user_id: str) -> bool:
        """检查用户是否在冷却期
        
        Args:
            user_id: 用户ID
            
        Returns:
            bool: True表示可以分析
        """
        last_time = self._last_analysis_time.get(user_id, 0)
        return time.time() - last_time >= self.cooldown
    
    def increment(self, user_id: str, session_id: str = "") -> None:
        """增加预算计数
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
        """
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_count[today] = self._daily_count.get(today, 0) + 1
        self._cleanup_daily_counts()
        
        if session_id:
            self._session_count[session_id] = self._session_count.get(session_id, 0) + 1
        
        self._last_analysis_time[user_id] = time.time()
    
    def _cleanup_daily_counts(self) -> None:
        """清理过期的每日计数"""
        today = datetime.now()
        keys_to_remove = []
        for date_str in self._daily_count:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if (today - date).days > 3:
                    keys_to_remove.append(date_str)
            except ValueError:
                keys_to_remove.append(date_str)
        
        for key in keys_to_remove:
            del self._daily_count[key]
    
    def reset_session(self, session_id: str) -> None:
        """重置会话预算"""
        if session_id in self._session_count:
            del self._session_count[session_id]
    
    def get_status(self, session_id: str = "") -> Dict:
        """获取预算状态"""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_used = self._daily_count.get(today, 0)
        
        status = {
            "daily_used": daily_used,
            "daily_budget": self.daily_budget,
            "daily_remaining": max(0, self.daily_budget - daily_used),
            "daily_exhausted": daily_used >= self.daily_budget
        }
        
        if session_id:
            session_used = self._session_count.get(session_id, 0)
            status.update({
                "session_used": session_used,
                "session_budget": self.session_budget,
                "session_remaining": max(0, self.session_budget - session_used),
                "session_exhausted": session_used >= self.session_budget
            })
        
        return status
    
    def clear_all(self) -> None:
        """清空所有预算计数"""
        self._daily_count.clear()
        self._session_count.clear()
        self._last_analysis_time.clear()


class SimilarImageDetector:
    """相似图片检测器
    
    职责：
    1. 短时间内重复图片检测
    2. 最近图片追踪
    """
    
    def __init__(
        self,
        time_window: int = 60,
        limit: int = 20
    ):
        """初始化检测器
        
        Args:
            time_window: 检测时间窗口（秒）
            limit: 保留的最近图片数量
        """
        self.time_window = time_window
        self.limit = limit
        self._recent_images: deque = deque(maxlen=limit)
    
    def is_similar(self, image_hash: Optional[str]) -> bool:
        """检查是否为最近分析过的相似图片
        
        Args:
            image_hash: 图片哈希
            
        Returns:
            bool: True表示是相似/重复图片
        """
        if not image_hash:
            return False
        
        current_time = time.time()
        
        for stored_hash, timestamp in self._recent_images:
            if current_time - timestamp > self.time_window:
                continue
            
            if stored_hash == image_hash:
                return True
        
        return False
    
    def add(self, image_hash: str) -> None:
        """添加到最近图片列表"""
        self._recent_images.append((image_hash, time.time()))
    
    def clear(self) -> None:
        """清空追踪列表"""
        self._recent_images.clear()
    
    @property
    def tracked_count(self) -> int:
        """获取追踪的图片数量"""
        return len(self._recent_images)

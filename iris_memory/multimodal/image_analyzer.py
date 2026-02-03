"""
图片智能分析模块
利用 AstrBot 的 Vision LLM 接口实现图片理解

采用分层策略平衡 Token 消耗和上下文理解：
- SKIP: 跳过分析（表情包、连续图片等）
- BRIEF: 简要描述（普通图片，低token消耗）
- DETAILED: 深度分析（用户询问、情感表达等）

优化策略：
- 缓存机制：基于URL哈希避免重复分析
- 时间冷却：用户级别的分析间隔控制
- 相似去重：短时间内相似图片只分析一次
- 分析预算：每日/会话分析次数限制
- 上下文过滤：无关上下文跳过分析
"""

import asyncio
import hashlib
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from iris_memory.utils.logger import get_logger

# 模块logger
logger = get_logger("image_analyzer")


class ImageAnalysisLevel(str, Enum):
    """图片分析层级"""
    SKIP = "skip"           # 跳过分析
    BRIEF = "brief"         # 简要描述
    DETAILED = "detailed"   # 深度分析


@dataclass
class ImageInfo:
    """图片信息"""
    url: str = ""                       # 图片URL或路径
    file: str = ""                      # 本地文件路径
    is_sticker: bool = False            # 是否为表情包
    width: Optional[int] = None         # 宽度（如果可获取）
    height: Optional[int] = None        # 高度（如果可获取）


@dataclass
class ImageAnalysisResult:
    """图片分析结果"""
    level: ImageAnalysisLevel = ImageAnalysisLevel.SKIP
    description: str = ""               # 图片描述
    emotions: List[str] = field(default_factory=list)    # 检测到的情感
    objects: List[str] = field(default_factory=list)     # 检测到的物体/元素
    context_relevance: float = 0.0      # 与对话的相关性 (0-1)
    token_cost: int = 0                 # Token消耗估计
    cached: bool = False                # 是否来自缓存
    error: Optional[str] = None         # 错误信息


class ImageAnalyzer:
    """图片智能分析器
    
    利用 AstrBot 的 LLM Vision 接口分析图片内容，
    采用分层策略平衡 Token 消耗和理解质量
    
    Features:
    - 智能分层：根据上下文决定分析深度
    - 缓存机制：避免重复分析相同图片
    - 表情包检测：跳过低价值图片
    - Token预算：控制分析消耗
    """
    
    # 分析提示词模板
    BRIEF_PROMPT = "用一句话简洁描述这张图片的主要内容（不超过30字）："
    
    DETAILED_PROMPT = """分析这张图片，请简洁回答：
1. 图片主要内容是什么？
2. 图中传达的情感或氛围？

用中文回复，不超过80字。"""
    
    # 表情包/贴纸URL特征
    STICKER_PATTERNS = [
        'sticker', 'emoji', 'face/', 'marketface', 
        'gif', 'emoticon', 'meme', 'qq_face'
    ]
    
    # 询问性关键词（触发深度分析）
    QUESTION_KEYWORDS = [
        "这是什么", "看看", "帮我看", "是不是", "怎么样", 
        "什么意思", "分析", "识别", "看一下", "看下"
    ]
    
    # 情感表达关键词（触发深度分析）
    EMOTION_KEYWORDS = [
        "喜欢", "讨厌", "开心", "难过", "生气", "害怕", 
        "感动", "惊讶", "好看", "丑", "可爱", "帅", "美"
    ]
    
    def __init__(
        self,
        astrbot_context,
        config: Optional[Dict[str, Any]] = None
    ):
        """初始化图片分析器
        
        Args:
            astrbot_context: AstrBot 上下文对象
            config: 配置选项
        """
        self.context = astrbot_context
        self.config = config or {}
        
        # 配置参数
        self.enable_analysis = self.config.get("enable_image_analysis", True)
        self.max_images_per_message = self.config.get("max_images_per_message", 2)
        self.analysis_cooldown = self.config.get("analysis_cooldown", 3.0)  # 秒
        self.skip_sticker = self.config.get("skip_sticker", True)
        self.default_level = self.config.get("default_level", "auto")  # auto/brief/detailed/skip
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1小时
        self.max_cache_size = self.config.get("max_cache_size", 200)
        
        # 新增：分析预算配置
        self.daily_analysis_budget = self.config.get("daily_analysis_budget", 100)  # 每日最大分析次数
        self.session_analysis_budget = self.config.get("session_analysis_budget", 20)  # 每会话最大分析次数
        
        # 新增：相似图片去重配置
        self.similar_image_window = self.config.get("similar_image_window", 60)  # 秒，相似图片检测时间窗口
        self.recent_image_limit = self.config.get("recent_image_limit", 20)  # 保留的最近图片数量
        
        # 新增：上下文相关性配置
        self.require_context_relevance = self.config.get("require_context_relevance", True)  # 是否要求上下文相关
        
        # 缓存：避免重复分析
        self._analysis_cache: Dict[str, Tuple[ImageAnalysisResult, float]] = {}
        self._last_analysis_time: Dict[str, float] = {}  # {user_id: timestamp}
        
        # 新增：分析预算追踪
        self._daily_analysis_count: Dict[str, int] = {}  # {date_str: count}
        self._session_analysis_count: Dict[str, int] = {}  # {session_key: count}
        
        # 新增：相似图片去重 - 存储最近的图片哈希
        self._recent_images: deque = deque(maxlen=self.recent_image_limit)  # [(hash, timestamp), ...]
        
        # 统计信息
        self._stats = {
            "total_analyzed": 0,
            "cache_hits": 0,
            "skipped": 0,
            "brief_analyses": 0,
            "detailed_analyses": 0,
            "errors": 0,
            "budget_exceeded": 0,
            "similar_skipped": 0,
            "context_skipped": 0
        }
        
        logger.info(f"ImageAnalyzer initialized: enable={self.enable_analysis}, "
                   f"max_images={self.max_images_per_message}, "
                   f"default_level={self.default_level}")
    
    async def analyze_message_images(
        self,
        message_chain: List,
        user_id: str,
        context_text: str = "",
        umo: str = "",
        session_id: str = ""
    ) -> List[ImageAnalysisResult]:
        """分析消息中的所有图片
        
        Args:
            message_chain: 消息链 (event.message_obj.message)
            user_id: 用户ID
            context_text: 伴随的文字内容
            umo: unified_msg_origin，用于获取 LLM provider
            session_id: 会话ID，用于会话预算追踪
            
        Returns:
            List[ImageAnalysisResult]: 分析结果列表
        """
        if not self.enable_analysis:
            return []
        
        # 提取图片
        images = self._extract_images(message_chain)
        if not images:
            return []
        
        logger.debug(f"Found {len(images)} images in message from user {user_id}")
        
        results = []
        for i, image_info in enumerate(images[:self.max_images_per_message]):
            # 确定分析层级
            level = self._determine_analysis_level(
                image_info=image_info,
                image_index=i,
                total_images=len(images),
                context_text=context_text,
                user_id=user_id
            )
            
            if level == ImageAnalysisLevel.SKIP:
                self._stats["skipped"] += 1
                results.append(ImageAnalysisResult(
                    level=level,
                    description="[图片]",
                    token_cost=0
                ))
                continue
            
            # 检查缓存
            image_hash = self._get_image_hash(image_info)
            cached_result = self._get_from_cache(image_hash)
            if cached_result:
                self._stats["cache_hits"] += 1
                cached_result.cached = True
                results.append(cached_result)
                continue
            
            # 新增：检查分析预算
            if not self._check_budget(user_id, session_id):
                logger.debug(f"Analysis budget exceeded for user {user_id}")
                self._stats["budget_exceeded"] += 1
                results.append(ImageAnalysisResult(
                    level=ImageAnalysisLevel.SKIP,
                    description="[图片]",
                    token_cost=0
                ))
                continue
            
            # 新增：检查相似图片（短时间内重复发送）
            if self._is_similar_recent_image(image_hash):
                logger.debug(f"Similar image detected recently, skipping analysis")
                self._stats["similar_skipped"] += 1
                results.append(ImageAnalysisResult(
                    level=ImageAnalysisLevel.SKIP,
                    description="[图片]",
                    token_cost=0
                ))
                continue
            
            # 新增：检查上下文相关性
            if self.require_context_relevance and not self._check_context_relevance(context_text, image_info):
                logger.debug(f"Image not relevant to context, skipping detailed analysis")
                self._stats["context_skipped"] += 1
                # 降级为跳过，不做分析
                results.append(ImageAnalysisResult(
                    level=ImageAnalysisLevel.SKIP,
                    description="[图片]",
                    token_cost=0
                ))
                continue
            
            # 检查冷却时间
            if not self._check_cooldown(user_id):
                logger.debug(f"Analysis cooldown active for user {user_id}")
                results.append(ImageAnalysisResult(
                    level=ImageAnalysisLevel.SKIP,
                    description="[图片]",
                    token_cost=0
                ))
                continue
            
            # 执行分析
            result = await self._analyze_single_image(
                image_info=image_info,
                level=level,
                context_text=context_text,
                umo=umo
            )
            
            # 更新统计
            self._stats["total_analyzed"] += 1
            if level == ImageAnalysisLevel.BRIEF:
                self._stats["brief_analyses"] += 1
            else:
                self._stats["detailed_analyses"] += 1
            
            # 缓存结果
            if result and image_hash and not result.error:
                self._add_to_cache(image_hash, result)
            
            # 更新冷却时间
            self._last_analysis_time[user_id] = time.time()
            
            # 新增：更新预算计数
            self._increment_budget(user_id, session_id)
            
            # 新增：记录到最近图片列表
            if image_hash:
                self._add_recent_image(image_hash)
            
            results.append(result)
        
        return results
    
    def _extract_images(self, message_chain: List) -> List[ImageInfo]:
        """从消息链提取图片信息
        
        Args:
            message_chain: AstrBot消息链
            
        Returns:
            List[ImageInfo]: 图片信息列表
        """
        images = []
        
        for component in message_chain:
            # 检查是否为Image类型
            comp_type = type(component).__name__
            if comp_type == 'Image':
                # 提取URL和文件路径
                url = getattr(component, 'url', '') or ''
                file_path = getattr(component, 'file', '') or ''
                
                # 判断是否为表情包
                is_sticker = self._is_sticker_url(url) or self._is_sticker_url(file_path)
                
                images.append(ImageInfo(
                    url=url,
                    file=file_path,
                    is_sticker=is_sticker
                ))
        
        return images
    
    def _is_sticker_url(self, url: str) -> bool:
        """判断URL是否指向表情包/贴纸"""
        if not url:
            return False
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in self.STICKER_PATTERNS)
    
    def _determine_analysis_level(
        self,
        image_info: ImageInfo,
        image_index: int,
        total_images: int,
        context_text: str,
        user_id: str
    ) -> ImageAnalysisLevel:
        """确定图片分析层级
        
        分层策略：
        1. 配置为skip -> 跳过
        2. 表情包/贴纸 -> 跳过
        3. 非首张图片 -> 跳过（多图只分析第一张）
        4. 有询问词 -> 深度分析
        5. 有情感表达 -> 深度分析
        6. 默认 -> 简要分析
        
        Args:
            image_info: 图片信息
            image_index: 图片索引
            total_images: 总图片数
            context_text: 伴随文字
            user_id: 用户ID
            
        Returns:
            ImageAnalysisLevel: 分析层级
        """
        # 配置强制跳过
        if self.default_level == "skip":
            return ImageAnalysisLevel.SKIP
        
        # 配置强制层级
        if self.default_level == "brief":
            if image_index > 0 or image_info.is_sticker:
                return ImageAnalysisLevel.SKIP
            return ImageAnalysisLevel.BRIEF
        
        if self.default_level == "detailed":
            if image_index > 0 or image_info.is_sticker:
                return ImageAnalysisLevel.SKIP
            return ImageAnalysisLevel.DETAILED
        
        # auto模式：智能判断
        
        # 表情包跳过
        if self.skip_sticker and image_info.is_sticker:
            return ImageAnalysisLevel.SKIP
        
        # 非首张图片跳过
        if image_index > 0:
            return ImageAnalysisLevel.SKIP
        
        # 无伴随文字 -> 简要分析
        if not context_text.strip():
            return ImageAnalysisLevel.BRIEF
        
        # 有询问性文字 -> 深度分析
        if any(kw in context_text for kw in self.QUESTION_KEYWORDS):
            return ImageAnalysisLevel.DETAILED
        
        # 情感表达类文字 -> 深度分析
        if any(kw in context_text for kw in self.EMOTION_KEYWORDS):
            return ImageAnalysisLevel.DETAILED
        
        # 默认简要分析
        return ImageAnalysisLevel.BRIEF
    
    def _get_image_hash(self, image_info: ImageInfo) -> Optional[str]:
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
    
    def _get_from_cache(self, image_hash: Optional[str]) -> Optional[ImageAnalysisResult]:
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
            # 检查TTL
            if time.time() - timestamp < self.cache_ttl:
                return ImageAnalysisResult(
                    level=result.level,
                    description=result.description,
                    emotions=result.emotions.copy() if result.emotions else [],
                    objects=result.objects.copy() if result.objects else [],
                    context_relevance=result.context_relevance,
                    token_cost=0,  # 缓存命中不消耗token
                    cached=True
                )
            else:
                # 过期，删除
                del self._analysis_cache[image_hash]
        
        return None
    
    def _add_to_cache(self, image_hash: str, result: ImageAnalysisResult):
        """添加到缓存
        
        Args:
            image_hash: 图片哈希
            result: 分析结果
        """
        # 清理过期缓存
        if len(self._analysis_cache) >= self.max_cache_size:
            self._cleanup_cache()
        
        self._analysis_cache[image_hash] = (result, time.time())
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, ts) in self._analysis_cache.items()
            if current_time - ts > self.cache_ttl
        ]
        for k in expired_keys:
            del self._analysis_cache[k]
        
        # 如果仍然过大，删除最旧的
        if len(self._analysis_cache) >= self.max_cache_size:
            sorted_items = sorted(
                self._analysis_cache.items(),
                key=lambda x: x[1][1]  # 按时间戳排序
            )
            # 保留一半
            for k, _ in sorted_items[:len(sorted_items) // 2]:
                del self._analysis_cache[k]
    
    def _check_cooldown(self, user_id: str) -> bool:
        """检查用户是否在冷却期
        
        Args:
            user_id: 用户ID
            
        Returns:
            bool: True表示可以分析，False表示在冷却中
        """
        last_time = self._last_analysis_time.get(user_id, 0)
        return time.time() - last_time >= self.analysis_cooldown
    
    # ==================== 新增：预算控制方法 ====================
    
    def _check_budget(self, user_id: str, session_id: str = "") -> bool:
        """检查是否超出分析预算
        
        同时检查每日预算和会话预算
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            bool: True表示预算内可以分析，False表示预算已耗尽
        """
        # 检查每日预算
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = self._daily_analysis_count.get(today, 0)
        if daily_count >= self.daily_analysis_budget:
            logger.debug(f"Daily analysis budget exhausted: {daily_count}/{self.daily_analysis_budget}")
            return False
        
        # 检查会话预算（如果提供了session_id）
        if session_id and self.session_analysis_budget > 0:
            session_count = self._session_analysis_count.get(session_id, 0)
            if session_count >= self.session_analysis_budget:
                logger.debug(f"Session analysis budget exhausted: {session_count}/{self.session_analysis_budget}")
                return False
        
        return True
    
    def _increment_budget(self, user_id: str, session_id: str = ""):
        """增加预算计数
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
        """
        # 更新每日计数
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_analysis_count[today] = self._daily_analysis_count.get(today, 0) + 1
        
        # 清理旧日期的计数（保留最近3天）
        self._cleanup_daily_counts()
        
        # 更新会话计数
        if session_id:
            self._session_analysis_count[session_id] = self._session_analysis_count.get(session_id, 0) + 1
    
    def _cleanup_daily_counts(self):
        """清理过期的每日计数"""
        today = datetime.now()
        keys_to_remove = []
        for date_str in self._daily_analysis_count:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if (today - date).days > 3:
                    keys_to_remove.append(date_str)
            except ValueError:
                keys_to_remove.append(date_str)
        
        for key in keys_to_remove:
            del self._daily_analysis_count[key]
    
    # ==================== 新增：相似图片检测方法 ====================
    
    def _is_similar_recent_image(self, image_hash: Optional[str]) -> bool:
        """检查是否为最近分析过的相似图片
        
        基于短时间窗口内的图片哈希检测重复
        
        Args:
            image_hash: 图片哈希
            
        Returns:
            bool: True表示是相似/重复图片
        """
        if not image_hash:
            return False
        
        current_time = time.time()
        
        for stored_hash, timestamp in self._recent_images:
            # 检查时间窗口
            if current_time - timestamp > self.similar_image_window:
                continue
            
            # 完全匹配（相同图片）
            if stored_hash == image_hash:
                return True
        
        return False
    
    def _add_recent_image(self, image_hash: str):
        """添加到最近图片列表
        
        Args:
            image_hash: 图片哈希
        """
        self._recent_images.append((image_hash, time.time()))
    
    # ==================== 新增：上下文相关性检测方法 ====================
    
    # 需要分析的上下文关键词
    CONTEXT_KEYWORDS = [
        # 询问图片内容
        "这是什么", "这个是", "什么东西", "什么意思",
        # 分享类
        "给你看", "分享", "拍的", "我的", "送你",
        # 情感类
        "好看", "可爱", "漂亮", "帅", "美",
        # 请求反馈
        "怎么样", "好不好", "觉得",
        # 表达类
        "开心", "难过", "生气", "高兴", "伤心",
        # 直接引用
        "这张", "这图", "图片", "照片"
    ]
    
    def _check_context_relevance(self, context_text: str, image_info: ImageInfo) -> bool:
        """检查图片是否与上下文相关，值得分析
        
        过滤掉纯图片刷屏、无意义转发等场景
        
        Args:
            context_text: 伴随文字
            image_info: 图片信息
            
        Returns:
            bool: True表示相关值得分析，False表示不相关可跳过
        """
        # 有伴随文字的一般都相关
        if context_text.strip():
            # 检查是否包含相关关键词
            for keyword in self.CONTEXT_KEYWORDS:
                if keyword in context_text:
                    return True
            
            # 有文字但没有明确关键词，也认为相关（用户可能在描述）
            if len(context_text.strip()) > 2:
                return True
        
        # 没有伴随文字，检查是否为表情包
        if image_info.is_sticker:
            return False  # 纯表情包不分析
        
        # 单独发送的图片，可能有意义（如分享照片）
        # 但为了节省API，默认返回False，除非配置允许
        return not self.require_context_relevance
    
    def reset_session_budget(self, session_id: str):
        """重置指定会话的预算计数
        
        适用于会话重新开始时调用
        
        Args:
            session_id: 会话ID
        """
        if session_id in self._session_analysis_count:
            del self._session_analysis_count[session_id]
            logger.debug(f"Reset session budget for: {session_id}")

    async def _analyze_single_image(
        self,
        image_info: ImageInfo,
        level: ImageAnalysisLevel,
        context_text: str,
        umo: str
    ) -> ImageAnalysisResult:
        """分析单张图片
        
        使用 AstrBot 的 Provider.text_chat(image_urls=...) 接口
        
        Args:
            image_info: 图片信息
            level: 分析层级
            context_text: 伴随文字
            umo: unified_msg_origin
            
        Returns:
            ImageAnalysisResult: 分析结果
        """
        try:
            # 获取图片URL
            image_url = image_info.url or image_info.file
            if not image_url:
                return ImageAnalysisResult(
                    level=level,
                    description="[无法获取图片]",
                    token_cost=0,
                    error="No image URL"
                )
            
            # 获取 LLM Provider
            provider = self.context.get_using_provider(umo=umo)
            if not provider:
                logger.warning("No LLM provider available for image analysis")
                return ImageAnalysisResult(
                    level=level,
                    description="[图片]",
                    token_cost=0,
                    error="No LLM provider"
                )
            
            # 选择提示词
            if level == ImageAnalysisLevel.BRIEF:
                prompt = self.BRIEF_PROMPT
            else:
                prompt = self.DETAILED_PROMPT
            
            # 如果有上下文文字，添加到提示中
            if context_text.strip():
                context_hint = context_text[:100]  # 限制长度
                prompt = f"用户发送图片时说：「{context_hint}」\n\n{prompt}"
            
            # 调用 Vision LLM
            logger.debug(f"Calling LLM for image analysis, level={level.value}")
            
            llm_resp = await provider.text_chat(
                prompt=prompt,
                image_urls=[image_url],
                context=[]
            )
            
            if not llm_resp or not llm_resp.completion_text:
                self._stats["errors"] += 1
                return ImageAnalysisResult(
                    level=level,
                    description="[图片分析失败]",
                    token_cost=0,
                    error="Empty LLM response"
                )
            
            description = llm_resp.completion_text.strip()
            
            # 清理描述（移除可能的前缀）
            description = self._clean_description(description)
            
            # 估算 Token 消耗
            token_cost = 100 if level == ImageAnalysisLevel.BRIEF else 300
            
            logger.debug(f"Image analysis completed: {description[:50]}...")
            
            return ImageAnalysisResult(
                level=level,
                description=description,
                token_cost=token_cost
            )
            
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            self._stats["errors"] += 1
            return ImageAnalysisResult(
                level=level,
                description="[图片]",
                token_cost=0,
                error=str(e)
            )
    
    def _clean_description(self, description: str) -> str:
        """清理LLM返回的描述
        
        Args:
            description: 原始描述
            
        Returns:
            str: 清理后的描述
        """
        # 移除常见的前缀
        prefixes_to_remove = [
            "这张图片", "图片中", "图片显示", "图中",
            "这是", "画面中", "照片中", "照片显示"
        ]
        
        cleaned = description.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                # 移除可能的标点
                if cleaned.startswith(('：', ':', '，', ',')):
                    cleaned = cleaned[1:].strip()
                break
        
        return cleaned if cleaned else description
    
    def format_for_memory(
        self,
        results: List[ImageAnalysisResult],
        include_level: bool = False
    ) -> str:
        """将分析结果格式化为记忆存储格式
        
        Args:
            results: 分析结果列表
            include_level: 是否包含分析层级信息
            
        Returns:
            str: 格式化的描述字符串
        """
        if not results:
            return ""
        
        descriptions = []
        for r in results:
            if r.error:
                descriptions.append("[图片]")
            elif r.description and r.description not in ["[图片]", "[无法获取图片]"]:
                if include_level:
                    descriptions.append(f"[图片({r.level.value}): {r.description}]")
                else:
                    descriptions.append(f"[图片: {r.description}]")
            else:
                descriptions.append("[图片]")
        
        return " ".join(descriptions)
    
    def format_for_llm_context(self, results: List[ImageAnalysisResult]) -> str:
        """格式化为LLM上下文注入格式
        
        Args:
            results: 分析结果列表
            
        Returns:
            str: LLM上下文格式的描述
        """
        if not results:
            return ""
        
        valid_results = [r for r in results if r.description and not r.error]
        if not valid_results:
            return ""
        
        if len(valid_results) == 1:
            return f"（用户发送的图片内容：{valid_results[0].description}）"
        
        desc_list = [f"{i+1}. {r.description}" for i, r in enumerate(valid_results)]
        return f"（用户发送的图片内容：\n" + "\n".join(desc_list) + "）"
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取分析统计信息
        
        Returns:
            Dict[str, Any]: 统计数据
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            **self._stats,
            "cache_size": len(self._analysis_cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(1, self._stats["total_analyzed"] + self._stats["cache_hits"])
            ),
            "daily_analysis_count": self._daily_analysis_count.get(today, 0),
            "daily_budget_remaining": max(0, self.daily_analysis_budget - self._daily_analysis_count.get(today, 0)),
            "recent_images_tracked": len(self._recent_images),
            "active_sessions": len(self._session_analysis_count)
        }
    
    def get_budget_status(self, session_id: str = "") -> Dict[str, Any]:
        """获取预算状态
        
        Args:
            session_id: 可选的会话ID
            
        Returns:
            Dict[str, Any]: 预算状态信息
        """
        today = datetime.now().strftime("%Y-%m-%d")
        daily_used = self._daily_analysis_count.get(today, 0)
        
        status = {
            "daily_used": daily_used,
            "daily_budget": self.daily_analysis_budget,
            "daily_remaining": max(0, self.daily_analysis_budget - daily_used),
            "daily_exhausted": daily_used >= self.daily_analysis_budget
        }
        
        if session_id:
            session_used = self._session_analysis_count.get(session_id, 0)
            status.update({
                "session_used": session_used,
                "session_budget": self.session_analysis_budget,
                "session_remaining": max(0, self.session_analysis_budget - session_used),
                "session_exhausted": session_used >= self.session_analysis_budget
            })
        
        return status
    
    def clear_cache(self):
        """清空缓存"""
        self._analysis_cache.clear()
        self._last_analysis_time.clear()
        self._recent_images.clear()
        logger.info("Image analysis cache cleared")
    
    def clear_all_budgets(self):
        """清空所有预算计数（用于调试或管理）"""
        self._daily_analysis_count.clear()
        self._session_analysis_count.clear()
        logger.info("All image analysis budgets cleared")

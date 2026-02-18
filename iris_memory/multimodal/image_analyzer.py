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

架构：
- 使用组合模式拆分功能模块
- image_cache.py: 缓存和预算控制
"""

import time
from typing import Optional, Dict, Any, List

from iris_memory.utils.logger import get_logger
from iris_memory.utils.provider_utils import (
    get_default_provider,
    get_provider_by_id,
    normalize_provider_id,
)
from iris_memory.multimodal.image_cache import (
    ImageInfo,
    ImageAnalysisResult,
    ImageAnalysisLevel,
    ImageCacheManager,
    ImageBudgetManager,
    SimilarImageDetector
)

logger = get_logger("image_analyzer")


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
    
    BRIEF_PROMPT = "用一句话简洁描述这张图片的主要内容（不超过30字）："
    
    DETAILED_PROMPT = """分析这张图片，请简洁回答：
1. 图片主要内容是什么？
2. 图中传达的情感或氛围？

用中文回复，不超过80字。"""
    
    STICKER_PATTERNS = [
        'sticker', 'emoji', 'face/', 'marketface', 
        'gif', 'emoticon', 'meme', 'qq_face'
    ]
    
    QUESTION_KEYWORDS = [
        "这是什么", "看看", "帮我看", "是不是", "怎么样", 
        "什么意思", "分析", "识别", "看一下", "看下"
    ]
    
    EMOTION_KEYWORDS = [
        "喜欢", "讨厌", "开心", "难过", "生气", "害怕", 
        "感动", "惊讶", "好看", "丑", "可爱", "帅", "美"
    ]
    
    CONTEXT_KEYWORDS = [
        "这是什么", "这个是", "什么东西", "什么意思",
        "给你看", "分享", "拍的", "我的", "送你",
        "好看", "可爱", "漂亮", "帅", "美",
        "怎么样", "好不好", "觉得",
        "开心", "难过", "生气", "高兴", "伤心",
        "这张", "这图", "图片", "照片"
    ]
    
    def __init__(
        self,
        astrbot_context,
        config: Optional[Dict[str, Any]] = None,
        provider_id: Optional[str] = None
    ):
        """初始化图片分析器
        
        Args:
            astrbot_context: AstrBot 上下文对象
            config: 配置选项
            provider_id: 指定的 LLM 提供者 ID（用于 Vision 模型）
        """
        self.context = astrbot_context
        self.config = config or {}
        self._configured_provider_id = normalize_provider_id(provider_id)
        
        self.enable_analysis = self.config.get("enable_image_analysis", True)
        self.max_images_per_message = self.config.get("max_images_per_message", 2)
        self.skip_sticker = self.config.get("skip_sticker", True)
        self.default_level = self.config.get("default_level", "auto")
        self.require_context_relevance = self.config.get("require_context_relevance", True)
        
        self._cache_manager = ImageCacheManager(
            cache_ttl=self.config.get("cache_ttl", 3600),
            max_cache_size=self.config.get("max_cache_size", 200)
        )
        
        self._budget_manager = ImageBudgetManager(
            daily_budget=self.config.get("daily_analysis_budget", 100),
            session_budget=self.config.get("session_analysis_budget", 20),
            cooldown=self.config.get("analysis_cooldown", 3.0)
        )
        
        self._similar_detector = SimilarImageDetector(
            time_window=self.config.get("similar_image_window", 60),
            limit=self.config.get("recent_image_limit", 20)
        )
        
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
    
    @property
    def _recent_images(self):
        """向后兼容属性"""
        return self._similar_detector._recent_images
    
    @property
    def _daily_analysis_count(self):
        """向后兼容属性"""
        return self._budget_manager._daily_count
    
    @property
    def _session_analysis_count(self):
        """向后兼容属性"""
        return self._budget_manager._session_count
    
    async def analyze_message_images(
        self,
        message_chain: List,
        user_id: str,
        context_text: str = "",
        umo: str = "",
        session_id: str = "",
        daily_analysis_budget: Optional[int] = None,
    ) -> List[ImageAnalysisResult]:
        """分析消息中的所有图片"""
        if not self.enable_analysis:
            return []
        
        images = self._extract_images(message_chain)
        if not images:
            return []
        
        logger.debug(f"Found {len(images)} images in message from user {user_id}")
        
        results = []
        for i, image_info in enumerate(images[:self.max_images_per_message]):
            result = await self._analyze_single_image_with_checks(
                image_info=image_info,
                image_index=i,
                total_images=len(images),
                context_text=context_text,
                user_id=user_id,
                session_id=session_id,
                umo=umo,
                daily_budget=daily_analysis_budget
            )
            results.append(result)
        
        return results
    
    async def _analyze_single_image_with_checks(
        self,
        image_info: ImageInfo,
        image_index: int,
        total_images: int,
        context_text: str,
        user_id: str,
        session_id: str,
        umo: str,
        daily_budget: Optional[int]
    ) -> ImageAnalysisResult:
        """执行带各项检查的图片分析"""
        
        level = self._determine_analysis_level(
            image_info=image_info,
            image_index=image_index,
            total_images=total_images,
            context_text=context_text,
            user_id=user_id
        )
        
        if level == ImageAnalysisLevel.SKIP:
            self._stats["skipped"] += 1
            return ImageAnalysisResult(
                level=level,
                description="[图片]",
                token_cost=0
            )
        
        image_hash = self._cache_manager.get_image_hash(image_info)
        cached_result = self._cache_manager.get_from_cache(image_hash)
        if cached_result:
            self._stats["cache_hits"] += 1
            cached_result.cached = True
            return cached_result
        
        if not self._budget_manager.check_budget(user_id, session_id, daily_budget):
            logger.debug(f"Analysis budget exceeded for user {user_id}")
            self._stats["budget_exceeded"] += 1
            return ImageAnalysisResult(level=ImageAnalysisLevel.SKIP, description="[图片]", token_cost=0)
        
        if self._similar_detector.is_similar(image_hash):
            logger.debug(f"Similar image detected recently, skipping analysis")
            self._stats["similar_skipped"] += 1
            return ImageAnalysisResult(level=ImageAnalysisLevel.SKIP, description="[图片]", token_cost=0)
        
        if self.require_context_relevance and not self._check_context_relevance(context_text, image_info):
            logger.debug(f"Image not relevant to context, skipping detailed analysis")
            self._stats["context_skipped"] += 1
            return ImageAnalysisResult(level=ImageAnalysisLevel.SKIP, description="[图片]", token_cost=0)
        
        if not self._budget_manager.check_cooldown(user_id):
            logger.debug(f"Analysis cooldown active for user {user_id}")
            return ImageAnalysisResult(level=ImageAnalysisLevel.SKIP, description="[图片]", token_cost=0)
        
        result = await self._analyze_single_image(
            image_info=image_info,
            level=level,
            context_text=context_text,
            umo=umo
        )
        
        self._update_stats_and_state(result, level, image_hash, user_id, session_id)
        
        return result
    
    def _update_stats_and_state(
        self,
        result: ImageAnalysisResult,
        level: str,
        image_hash: Optional[str],
        user_id: str,
        session_id: str
    ) -> None:
        """更新统计和状态"""
        self._stats["total_analyzed"] += 1
        if level == ImageAnalysisLevel.BRIEF:
            self._stats["brief_analyses"] += 1
        else:
            self._stats["detailed_analyses"] += 1
        
        if result and image_hash and not result.error:
            self._cache_manager.add_to_cache(image_hash, result)
        
        self._budget_manager.increment(user_id, session_id)
        
        if image_hash:
            self._similar_detector.add(image_hash)
    
    def _extract_images(self, message_chain: List) -> List[ImageInfo]:
        """从消息链提取图片信息"""
        images = []
        
        for component in message_chain:
            comp_type = type(component).__name__
            if comp_type == 'Image':
                url = getattr(component, 'url', '') or ''
                file_path = getattr(component, 'file', '') or ''
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
    ) -> str:
        """确定图片分析层级"""
        if self.default_level == "skip":
            return ImageAnalysisLevel.SKIP
        
        if self.default_level == "brief":
            if image_index > 0 or image_info.is_sticker:
                return ImageAnalysisLevel.SKIP
            return ImageAnalysisLevel.BRIEF
        
        if self.default_level == "detailed":
            if image_index > 0 or image_info.is_sticker:
                return ImageAnalysisLevel.SKIP
            return ImageAnalysisLevel.DETAILED
        
        if self.skip_sticker and image_info.is_sticker:
            return ImageAnalysisLevel.SKIP
        
        if image_index > 0:
            return ImageAnalysisLevel.SKIP
        
        if not context_text.strip():
            return ImageAnalysisLevel.BRIEF
        
        if any(kw in context_text for kw in self.QUESTION_KEYWORDS):
            return ImageAnalysisLevel.DETAILED
        
        if any(kw in context_text for kw in self.EMOTION_KEYWORDS):
            return ImageAnalysisLevel.DETAILED
        
        return ImageAnalysisLevel.BRIEF
    
    def _check_context_relevance(self, context_text: str, image_info: ImageInfo) -> bool:
        """检查图片是否与上下文相关"""
        if context_text.strip():
            for keyword in self.CONTEXT_KEYWORDS:
                if keyword in context_text:
                    return True
            
            if len(context_text.strip()) > 2:
                return True
        
        if image_info.is_sticker:
            return False
        
        return not self.require_context_relevance
    
    async def _analyze_single_image(
        self,
        image_info: ImageInfo,
        level: str,
        context_text: str,
        umo: str
    ) -> ImageAnalysisResult:
        """分析单张图片"""
        try:
            image_url = image_info.url or image_info.file
            if not image_url:
                return ImageAnalysisResult(
                    level=level,
                    description="[无法获取图片]",
                    token_cost=0,
                    error="No image URL"
                )
            
            provider = await self._resolve_provider(umo)
            if not provider:
                logger.warning("No LLM provider available for image analysis")
                return ImageAnalysisResult(
                    level=level,
                    description="[图片]",
                    token_cost=0,
                    error="No LLM provider"
                )
            
            if level == ImageAnalysisLevel.BRIEF:
                prompt = self.BRIEF_PROMPT
            else:
                prompt = self.DETAILED_PROMPT
            
            if context_text.strip():
                context_hint = context_text[:100]
                prompt = f"用户发送图片时说：「{context_hint}」\n\n{prompt}"
            
            logger.debug(f"Calling LLM for image analysis, level={level}")
            
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
            description = self._clean_description(description)
            
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
    
    async def _resolve_provider(self, umo: str = ""):
        """解析 LLM 提供者"""
        if not self.context:
            return None
        
        provider_id = normalize_provider_id(self._configured_provider_id)
        
        if provider_id and provider_id not in ("", "default"):
            try:
                provider, resolved_id = get_provider_by_id(self.context, provider_id)
                if provider:
                    logger.debug(f"Using configured provider for image analysis: {resolved_id or provider_id}")
                    return provider
                logger.warning(f"Provider not found: {provider_id}, falling back to default")
            except Exception as e:
                logger.warning(f"Failed to get provider list: {e}")
        
        provider, _ = get_default_provider(self.context, umo=umo)
        return provider
    
    def _clean_description(self, description: str) -> str:
        """清理LLM返回的描述"""
        prefixes_to_remove = [
            "这张图片", "图片中", "图片显示", "图中",
            "这是", "画面中", "照片中", "照片显示"
        ]
        
        cleaned = description.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                if cleaned.startswith(('：', ':', '，', ',')):
                    cleaned = cleaned[1:].strip()
                break
        
        return cleaned if cleaned else description
    
    def format_for_memory(
        self,
        results: List[ImageAnalysisResult],
        include_level: bool = False
    ) -> str:
        """将分析结果格式化为记忆存储格式"""
        if not results:
            return ""
        
        descriptions = []
        for r in results:
            if r.error:
                descriptions.append("[图片]")
            elif r.description and r.description not in ["[图片]", "[无法获取图片]"]:
                if include_level:
                    descriptions.append(f"[图片({r.level}): {r.description}]")
                else:
                    descriptions.append(f"[图片: {r.description}]")
            else:
                descriptions.append("[图片]")
        
        return " ".join(descriptions)
    
    def format_for_llm_context(self, results: List[ImageAnalysisResult]) -> str:
        """格式化为LLM上下文注入格式"""
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
        """获取分析统计信息"""
        budget_status = self._budget_manager.get_status()
        return {
            **self._stats,
            "cache_size": self._cache_manager.size,
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(1, self._stats["total_analyzed"] + self._stats["cache_hits"])
            ),
            "daily_analysis_count": budget_status["daily_used"],
            "daily_budget_remaining": budget_status["daily_remaining"],
            "recent_images_tracked": self._similar_detector.tracked_count,
            "active_sessions": len(self._budget_manager._session_count)
        }
    
    def get_budget_status(self, session_id: str = "") -> Dict[str, Any]:
        """获取预算状态"""
        return self._budget_manager.get_status(session_id)
    
    def reset_session_budget(self, session_id: str) -> None:
        """重置指定会话的预算计数"""
        self._budget_manager.reset_session(session_id)
        logger.debug(f"Reset session budget for: {session_id}")
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache_manager.clear()
        self._similar_detector.clear()
        logger.info("Image analysis cache cleared")
    
    def clear_all_budgets(self) -> None:
        """清空所有预算计数"""
        self._budget_manager.clear_all()
        logger.info("All image analysis budgets cleared")
    
    def _increment_budget(self, user_id: str, session_id: str = "") -> None:
        """向后兼容方法：增加预算计数"""
        self._budget_manager.increment(user_id, session_id)
    
    def _check_budget(
        self,
        user_id: str,
        session_id: str = "",
        daily_analysis_budget: Optional[int] = None,
    ) -> bool:
        """向后兼容方法：检查预算"""
        return self._budget_manager.check_budget(user_id, session_id, daily_analysis_budget)
    
    def _is_similar_recent_image(self, image_hash: Optional[str]) -> bool:
        """向后兼容方法：检查相似图片"""
        return self._similar_detector.is_similar(image_hash)
    
    def _add_recent_image(self, image_hash: str) -> None:
        """向后兼容方法：添加最近图片"""
        self._similar_detector.add(image_hash)

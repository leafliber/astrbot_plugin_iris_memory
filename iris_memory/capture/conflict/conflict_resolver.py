"""
去重与冲突检测模块

实现记忆去重检查和语义冲突检测功能。
"""

import re
from datetime import datetime, timedelta
from typing import List, Optional, TYPE_CHECKING

from iris_memory.utils.logger import get_logger
from iris_memory.models.memory import Memory
from iris_memory.core.types import QualityLevel
from iris_memory.capture.conflict.similarity_calculator import SimilarityCalculator

if TYPE_CHECKING:
    from iris_memory.knowledge_graph.kg_storage import KGStorage

logger = get_logger("conflict_resolver")


class ConflictResolver:
    """记忆冲突解决器
    
    负责：
    - 记忆去重检测
    - 语义冲突检测
    - 冲突解决策略执行
    """
    
    # 反义词对列表
    ANTONYM_PAIRS = [
        ("喜欢", "讨厌"), ("喜欢", "恨"), ("爱", "恨"),
        ("开心", "难过"), ("高兴", "伤心"), ("快乐", "痛苦"),
        ("好", "坏"), ("优秀", "差劲"), ("成功", "失败"),
        ("支持", "反对"), ("同意", "拒绝"),
        ("有", "没有"), ("能", "不能"), ("会", "不会"),
        ("大", "小"), ("多", "少"), ("高", "低"),
        ("喜欢", "dislike"), ("讨厌", "like"), ("love", "hate"),
        ("happy", "sad"), ("good", "bad"), ("success", "failure")
    ]
    
    # 否定词列表
    NEGATION_WORDS = [
        "不", "没", "无", "非", "别", "不是", "don't", "not", "no", "never",
        "不喜欢", "讨厌", "喜欢"
    ]
    
    def __init__(self, similarity_calculator: Optional[SimilarityCalculator] = None):
        """初始化冲突解决器
        
        Args:
            similarity_calculator: 相似度计算器实例（可选，默认创建新实例）
        """
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()
    
    async def check_duplicate_by_vector(
        self,
        memory: Memory,
        user_id: str,
        group_id: Optional[str],
        chroma_manager,
        similarity_threshold: float = 0.95
    ) -> Optional[Memory]:
        """使用向量相似度检查重复记忆
        
        性能优化版本：使用ChromaDB的向量查询直接找到最相似的记忆，
        避免加载全部记忆。
        
        Args:
            memory: 新记忆
            user_id: 用户ID
            group_id: 群组ID
            chroma_manager: Chroma管理器
            similarity_threshold: 向量相似度阈值（默认0.95，越高越严格）
            
        Returns:
            Optional[Memory]: 如果找到重复记忆则返回，否则返回None
        """
        if not chroma_manager:
            return None
        
        try:
            # 使用向量查询找到最相似的记忆（只查询5条）
            similar_memories = await chroma_manager.query_memories(
                query_text=memory.content,
                user_id=user_id,
                group_id=group_id,
                top_k=5
            )
            
            if not similar_memories:
                return None
            
            # 检查是否有高相似度的记忆
            for existing in similar_memories:
                # 跳过自己（如果已经存在）
                if existing.id == memory.id:
                    continue
                
                # 使用文本相似度进行精确验证
                text_sim = self.similarity_calculator.calculate_similarity(
                    memory.content, existing.content
                )
                if text_sim >= similarity_threshold:
                    logger.debug(f"Found duplicate via vector search: {existing.id} (text_sim={text_sim:.3f})")
                    return existing
            
            return None
            
        except Exception as e:
            logger.debug(f"Vector-based duplicate check failed: {e}, falling back to text-based")
            return None

    async def check_conflicts_by_vector(
        self,
        memory: Memory,
        user_id: str,
        group_id: Optional[str],
        chroma_manager
    ) -> List[Memory]:
        """使用向量相似度检查记忆冲突
        
        性能优化版本：使用ChromaDB的向量查询找到语义相似的记忆，
        然后检查是否存在语义冲突。
        
        Args:
            memory: 新记忆
            user_id: 用户ID
            group_id: 群组ID
            chroma_manager: Chroma管理器
            
        Returns:
            List[Memory]: 冲突的记忆列表
        """
        conflicts = []
        
        if not chroma_manager:
            return conflicts
        
        try:
            # 使用向量查询找到语义相关的记忆
            similar_memories = await chroma_manager.query_memories(
                query_text=memory.content,
                user_id=user_id,
                group_id=group_id,
                top_k=10
            )
            
            if not similar_memories:
                return conflicts
            
            # 检查语义冲突
            for existing in similar_memories:
                if existing.id == memory.id:
                    continue
                
                # 只检查相同类型的记忆
                if existing.type != memory.type:
                    continue
                
                # 检查内容相似度
                content_sim = self.similarity_calculator.calculate_content_similarity(
                    memory.content, existing.content
                )
                if content_sim < 0.3:
                    continue
                
                # 检查是否为相反内容
                if self.is_opposite(memory.content, existing.content):
                    conflicts.append(existing)
                    memory.add_conflict(existing.id)
                    logger.debug(f"Conflict detected via vector search: {memory.id} vs {existing.id}")
            
            return conflicts
            
        except Exception as e:
            logger.warning(f"Vector-based conflict check failed: {e}")
            return conflicts

    def find_duplicate_from_results(
        self,
        memory: Memory,
        similar_memories: List[Memory],
        similarity_threshold: float = 0.95
    ) -> Optional[Memory]:
        """从查询结果中查找重复记忆
        
        用于共享查询结果的优化版本，避免重复查询。
        
        Args:
            memory: 新记忆
            similar_memories: 相似记忆列表（来自 query_memories 结果）
            similarity_threshold: 文本相似度阈值
            
        Returns:
            Optional[Memory]: 如果找到重复记忆则返回，否则返回None
        """
        if not similar_memories:
            return None
        
        # 只检查前5条（与原始逻辑一致）
        for existing in similar_memories[:5]:
            # 跳过自己
            if existing.id == memory.id:
                continue
            
            # 使用文本相似度进行精确验证
            text_sim = self.similarity_calculator.calculate_similarity(
                memory.content, existing.content
            )
            if text_sim >= similarity_threshold:
                logger.debug(f"Found duplicate via vector search: {existing.id} (text_sim={text_sim:.3f})")
                return existing
        
        return None

    def find_conflicts_from_results(
        self,
        memory: Memory,
        similar_memories: List[Memory]
    ) -> List[Memory]:
        """从查询结果中查找冲突记忆
        
        用于共享查询结果的优化版本，避免重复查询。
        
        Args:
            memory: 新记忆
            similar_memories: 相似记忆列表（来自 query_memories 结果）
            
        Returns:
            List[Memory]: 冲突的记忆列表
        """
        conflicts = []
        
        if not similar_memories:
            return conflicts
        
        # 检查语义冲突
        for existing in similar_memories:
            if existing.id == memory.id:
                continue
            
            # 只检查相同类型的记忆
            if existing.type != memory.type:
                continue
            
            # 检查内容相似度
            content_sim = self.similarity_calculator.calculate_content_similarity(
                memory.content, existing.content
            )
            if content_sim < 0.3:
                continue
            
            # 检查是否为相反内容
            if self.is_opposite(memory.content, existing.content):
                conflicts.append(existing)
                memory.add_conflict(existing.id)
                logger.debug(f"Conflict detected via vector search: {memory.id} vs {existing.id}")
        
        return conflicts

    async def resolve_conflicts(
        self,
        new_memory: Memory,
        conflicting_memories: List[Memory],
        chroma_manager,
        kg_storage: Optional["KGStorage"] = None,
    ) -> bool:
        """解决记忆冲突

        冲突解决策略：
        1. 如果新记忆是用户显式请求的，优先采用新记忆
        2. 如果新记忆置信度更高，更新旧记忆
        3. 如果旧记忆质量等级更高，保留旧记忆
        4. 否则标记为需要用户确认

        Args:
            new_memory: 新记忆
            conflicting_memories: 冲突的记忆列表
            chroma_manager: Chroma管理器
            kg_storage: 知识图谱存储层（可选，用于同步删除关联边）

        Returns:
            bool: True 表示新记忆仍需存储，False 表示新记忆已合并到旧记忆、无需再存储
        """
        if not conflicting_memories:
            return True

        resolved_count = 0

        for old_memory in conflicting_memories:
            resolution = self._determine_conflict_resolution(new_memory, old_memory)

            if resolution == "replace":
                # 用新记忆替换旧记忆
                try:
                    if chroma_manager:
                        await chroma_manager.delete_memory(old_memory.id)
                        # 同步删除知识图谱关联边
                        if kg_storage:
                            try:
                                edge_count = await kg_storage.delete_by_memory_id(old_memory.id)
                                if edge_count > 0:
                                    logger.debug(
                                        f"Deleted {edge_count} KG edges for replaced memory {old_memory.id}"
                                    )
                            except Exception as kg_err:
                                logger.warning(
                                    f"Failed to delete KG edges for memory {old_memory.id}: {kg_err}"
                                )
                        logger.debug(f"Conflict resolved: replaced {old_memory.id} with {new_memory.id}")
                        resolved_count += 1
                except Exception as e:
                    logger.error(f"Failed to replace conflicting memory: {e}")
                    
            elif resolution == "keep_old":
                # 保留旧记忆，标记新记忆为低质量
                new_memory.quality_level = QualityLevel.LOW_CONFIDENCE
                new_memory.metadata["conflict_resolution"] = "kept_old"
                logger.debug(f"Conflict resolved: keeping {old_memory.id}, lowered {new_memory.id} quality")
                resolved_count += 1
                
            elif resolution == "merge":
                # 合并两条记忆（增加旧记忆的置信度）
                try:
                    if chroma_manager:
                        old_memory.confidence = min(1.0, old_memory.confidence + 0.1)
                        old_memory.access_count += 1
                        await chroma_manager.update_memory(old_memory)
                        logger.debug(f"Conflict resolved: merged into {old_memory.id}")
                        resolved_count += 1
                        # 合并成功，新记忆无需再存储
                        return False
                except Exception as e:
                    logger.error(f"Failed to merge memories: {e}")
                    
            else:  # "pending"
                # 标记为待确认
                new_memory.metadata["conflict_status"] = "pending_user_confirmation"
                new_memory.metadata["conflicting_memory_id"] = old_memory.id
                logger.debug(f"Conflict pending: {new_memory.id} vs {old_memory.id}")
        
        # 所有冲突均已解决且无 merge，新记忆仍需存储
        return True

    def _determine_conflict_resolution(
        self,
        new_memory: Memory,
        old_memory: Memory
    ) -> str:
        """确定冲突解决策略
        
        Args:
            new_memory: 新记忆
            old_memory: 旧记忆
            
        Returns:
            str: 解决策略 ("replace", "keep_old", "merge", "pending")
        """
        # 策略1：用户显式请求的新记忆优先
        if new_memory.is_user_requested:
            return "replace"
        
        # 策略2：高质量等级的记忆优先
        if new_memory.quality_level.value > old_memory.quality_level.value + 1:
            return "replace"
        if old_memory.quality_level.value > new_memory.quality_level.value + 1:
            return "keep_old"
        
        # 策略3：置信度差异较大时
        confidence_diff = new_memory.confidence - old_memory.confidence
        if confidence_diff > 0.3:
            return "replace"
        if confidence_diff < -0.3:
            return "keep_old"
        
        # 策略4：如果内容非常相似但不完全相反，可能是更新
        if new_memory.created_time > old_memory.created_time:
            # 更新的信息，检查是否是细微修正
            content_sim = self.similarity_calculator.calculate_similarity(
                new_memory.content, old_memory.content
            )
            if content_sim > 0.7:
                return "replace"  # 可能是用户纠正旧信息
        
        # 默认：需要用户确认
        return "pending"

    def is_opposite(self, text1: str, text2: str) -> bool:
        """判断两个文本是否相反（语义冲突检测）

        基于 companion-memory framework 第12节的冲突检测要求，实现多策略语义冲突检测：

        策略1 - 否定词检测（权重最高）：
            检测逻辑：text1包含否定词而text2不包含，且核心内容相似度>0.6
            示例："我喜欢咖啡" vs "我不喜欢咖啡" → 冲突

        策略2 - 反义词检测：
            预定义反义词库覆盖常见对立概念（喜欢/讨厌、开心/难过等）
            要求两个文本有共同主题（≥2个共同非停用词）
            示例："工作很开心" vs "工作很痛苦" → 冲突

        策略3 - 数值/时间冲突：
            检测相同描述框架下的数值差异
            示例："我有3个苹果" vs "我有5个苹果" → 冲突

        复杂度：O(N + M)，N/M为文本长度

        Args:
            text1: 待比较的文本1
            text2: 待比较的文本2

        Returns:
            bool: 是否存在语义冲突
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # 策略1: 否定词检测
        for neg in self.NEGATION_WORDS:
            # 情况1: 否定词在text1但不在text2
            if neg in text1_lower and neg not in text2_lower:
                text1_clean = text1_lower.replace(neg, "").strip()
                # 计算相似度，如果核心内容相似则可能是冲突
                similarity = self.similarity_calculator.calculate_content_similarity(
                    text1_clean, text2_lower
                )
                if similarity > 0.6:
                    return True
            # 情况2: 否定词在text2但不在text1
            elif neg in text2_lower and neg not in text1_lower:
                text2_clean = text2_lower.replace(neg, "").strip()
                similarity = self.similarity_calculator.calculate_content_similarity(
                    text1_lower, text2_clean
                )
                if similarity > 0.6:
                    return True

        # 策略2: 反义词检测
        for word1, word2 in self.ANTONYM_PAIRS:
            if (word1 in text1_lower and word2 in text2_lower) or \
               (word1 in text2_lower and word2 in text1_lower):
                # 检查是否有相同的主题/对象
                if self.similarity_calculator.have_common_subject(text1_lower, text2_lower):
                    return True

        # 策略3: 数值冲突检测
        # 提取数值并检查是否冲突
        numbers1 = re.findall(r'\d+', text1)
        numbers2 = re.findall(r'\d+', text2)
        if numbers1 and numbers2 and numbers1 != numbers2:
            # 如果有相同的非数字部分，但数值不同，可能是冲突
            non_num1 = re.sub(r'\d+', '{NUM}', text1)
            non_num2 = re.sub(r'\d+', '{NUM}', text2)
            if non_num1 == non_num2 and set(numbers1) != set(numbers2):
                return True

        return False

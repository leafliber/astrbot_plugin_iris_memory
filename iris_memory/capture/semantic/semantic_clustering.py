"""
语义聚类模块

将 EPISODIC 记忆按实体/主题进行聚类，为语义提取做准备。

流程：
1. 预筛选：排除低质量、过新、已提取的记忆
2. 实体/主题聚类：基于关键词和记忆类型
3. （可选）向量相似度增强：用嵌入向量做 DBSCAN 聚类
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.utils.logger import get_logger

logger = get_logger("semantic_clustering")

# ── 停用词 ──
_STOP_WORDS: frozenset[str] = frozenset([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "那", "吗", "呢", "吧", "啊", "呀", "嗯", "哦", "哈",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
])


@dataclass
class MemoryCluster:
    """记忆聚类结果"""
    cluster_id: str
    cluster_key: str  # 聚类键（实体名 / 主题标签）
    cluster_type: str  # "entity" / "topic" / "type" / "vector"
    memories: List[Memory] = field(default_factory=list)
    user_id: str = ""  # 所属用户 ID（聚类按用户隔离）

    @property
    def size(self) -> int:
        return len(self.memories)

    @property
    def memory_ids(self) -> List[str]:
        return [m.id for m in self.memories]


class SemanticClustering:
    """语义聚类引擎

    将 EPISODIC 记忆按实体/主题聚类，输出可供 LLM 语义提取的记忆组。
    """

    def __init__(
        self,
        min_confidence: float = 0.4,
        min_age_days: int = 30,
        min_cluster_size: int = 3,
        cluster_time_window_days: int = 90,
        similarity_threshold: float = 0.75,
        max_clusters_per_run: int = 20,
        max_memories_per_cluster: int = 15,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_age_days = min_age_days
        self.min_cluster_size = min_cluster_size
        self.cluster_time_window_days = cluster_time_window_days
        self.similarity_threshold = similarity_threshold
        self.max_clusters_per_run = max_clusters_per_run
        self.max_memories_per_cluster = max_memories_per_cluster

    # ── 主入口 ──

    def cluster(self, memories: List[Memory]) -> List[MemoryCluster]:
        """对记忆列表执行聚类（按用户隔离）

        Args:
            memories: EPISODIC 层记忆列表

        Returns:
            聚类结果列表（已按大小降序排序，截断到 max_clusters_per_run）
        """
        filtered = self._prefilter(memories)
        if not filtered:
            logger.debug("No memories passed pre-filtering")
            return []

        logger.debug(f"Pre-filtering: {len(memories)} -> {len(filtered)} memories")

        # 按用户分组后再聚类
        user_groups = self._group_by_user(filtered)
        all_clusters: List[MemoryCluster] = []
        for user_id, user_memories in user_groups.items():
            clusters = self._entity_topic_clustering(user_memories)
            for cluster in clusters:
                cluster.user_id = user_id
            all_clusters.extend(clusters)

        # 截断每个聚类的记忆数量
        for cluster in all_clusters:
            if len(cluster.memories) > self.max_memories_per_cluster:
                # 保留置信度最高的
                cluster.memories.sort(key=lambda m: m.confidence, reverse=True)
                cluster.memories = cluster.memories[:self.max_memories_per_cluster]

        # 按 size 降序，截断总数
        all_clusters.sort(key=lambda c: c.size, reverse=True)
        all_clusters = all_clusters[:self.max_clusters_per_run]

        logger.debug(f"Clustering produced {len(all_clusters)} clusters")
        return all_clusters

    def cluster_with_vectors(
        self,
        memories: List[Memory],
    ) -> List[MemoryCluster]:
        """使用向量相似度增强聚类（按用户隔离）

        先做实体/主题聚类，再对未被归入任何簇的记忆
        做基于嵌入向量的相似度聚类。

        Args:
            memories: EPISODIC 层记忆列表

        Returns:
            聚类结果列表
        """
        filtered = self._prefilter(memories)
        if not filtered:
            return []

        # 按用户分组后再聚类
        user_groups = self._group_by_user(filtered)
        all_clusters: List[MemoryCluster] = []

        for user_id, user_memories in user_groups.items():
            # 第一轮：实体/主题聚类
            entity_clusters = self._entity_topic_clustering(user_memories)
            for cluster in entity_clusters:
                cluster.user_id = user_id
            clustered_ids: Set[str] = set()
            for cluster in entity_clusters:
                clustered_ids.update(m.id for m in cluster.memories)

            # 第二轮：向量聚类（仅处理未被归入实体簇的记忆）
            unclustered = [m for m in user_memories if m.id not in clustered_ids]
            vector_clusters = self._vector_clustering(unclustered)
            for cluster in vector_clusters:
                cluster.user_id = user_id

            all_clusters.extend(entity_clusters)
            all_clusters.extend(vector_clusters)

        for cluster in all_clusters:
            if len(cluster.memories) > self.max_memories_per_cluster:
                cluster.memories.sort(key=lambda m: m.confidence, reverse=True)
                cluster.memories = cluster.memories[:self.max_memories_per_cluster]

        all_clusters.sort(key=lambda c: c.size, reverse=True)
        return all_clusters[:self.max_clusters_per_run]

    # ── 阶段 1：预筛选 ──

    def _prefilter(self, memories: List[Memory]) -> List[Memory]:
        """预筛选记忆

        排除条件：
        - 非 EPISODIC 层
        - 已被语义提取 (summarized=True)
        - 置信度低于阈值
        - 创建时间不足 min_age_days
        """
        now = datetime.now()
        min_created = now - timedelta(days=self.min_age_days)
        result = []

        for memory in memories:
            if memory.storage_layer != StorageLayer.EPISODIC:
                continue
            if memory.summarized:
                continue
            if memory.confidence < self.min_confidence:
                continue
            if memory.created_time > min_created:
                continue
            result.append(memory)

        return result

    # ── 阶段 2：实体/主题聚类 ──

    def _entity_topic_clustering(self, memories: List[Memory]) -> List[MemoryCluster]:
        """基于关键词和记忆类型的聚类"""
        # 构建倒排索引：keyword -> [memory, ...]
        keyword_index: Dict[str, List[Memory]] = defaultdict(list)
        type_index: Dict[str, List[Memory]] = defaultdict(list)

        now = datetime.now()
        window_start = now - timedelta(days=self.cluster_time_window_days)

        for memory in memories:
            # 按记忆类型分组
            type_key = memory.type.value
            type_index[type_key].append(memory)

            # 按关键词分组
            keywords = self._extract_keywords(memory)
            for kw in keywords:
                keyword_index[kw].append(memory)

        clusters: List[MemoryCluster] = []
        used_memory_ids: Set[str] = set()
        cluster_counter = 0

        # 关键词聚类（实体/主题）
        for keyword, mems in sorted(keyword_index.items(), key=lambda x: -len(x[1])):
            # 时间窗口过滤
            windowed = [
                m for m in mems
                if m.created_time >= window_start and m.id not in used_memory_ids
            ]
            if len(windowed) < self.min_cluster_size:
                continue

            cluster_counter += 1
            cluster = MemoryCluster(
                cluster_id=f"kw_{cluster_counter}",
                cluster_key=keyword,
                cluster_type="entity",
                memories=windowed,
            )
            clusters.append(cluster)
            used_memory_ids.update(m.id for m in windowed)

        # 类型聚类（对于未被关键词聚类捕获的记忆）
        for type_key, mems in type_index.items():
            remaining = [
                m for m in mems
                if m.created_time >= window_start and m.id not in used_memory_ids
            ]
            if len(remaining) < self.min_cluster_size:
                continue

            cluster_counter += 1
            cluster = MemoryCluster(
                cluster_id=f"type_{cluster_counter}",
                cluster_key=type_key,
                cluster_type="type",
                memories=remaining,
            )
            clusters.append(cluster)
            used_memory_ids.update(m.id for m in remaining)

        return clusters

    # ── 阶段 3：向量相似度聚类 ──

    def _vector_clustering(self, memories: List[Memory]) -> List[MemoryCluster]:
        """基于嵌入向量的相似度聚类

        使用简单的层次聚类（无需 sklearn 依赖）：
        1. 计算两两余弦相似度
        2. 贪心合并相似度 >= threshold 的记忆对
        """
        # 只处理有嵌入向量的记忆
        with_embedding = [m for m in memories if m.embedding is not None]
        if len(with_embedding) < self.min_cluster_size:
            return []

        # 构建相似度矩阵（上三角）
        n = len(with_embedding)
        adjacency: Dict[int, Set[int]] = defaultdict(set)

        for i in range(n):
            emb_i = with_embedding[i].embedding
            norm_i = np.linalg.norm(emb_i)
            if norm_i == 0:
                continue
            for j in range(i + 1, n):
                emb_j = with_embedding[j].embedding
                norm_j = np.linalg.norm(emb_j)
                if norm_j == 0:
                    continue
                sim = float(np.dot(emb_i, emb_j) / (norm_i * norm_j))
                if sim >= self.similarity_threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # 连通分量（贪心 BFS）
        visited: Set[int] = set()
        clusters: List[MemoryCluster] = []
        cluster_counter = 0

        for start in range(n):
            if start in visited or start not in adjacency:
                continue
            component: List[int] = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) < self.min_cluster_size:
                continue

            cluster_counter += 1
            cluster_memories = [with_embedding[i] for i in component]
            cluster = MemoryCluster(
                cluster_id=f"vec_{cluster_counter}",
                cluster_key=f"vector_group_{cluster_counter}",
                cluster_type="vector",
                memories=cluster_memories,
            )
            clusters.append(cluster)

        return clusters

    # ── 用户分组 ──

    @staticmethod
    def _group_by_user(memories: List[Memory]) -> Dict[str, List[Memory]]:
        """按 user_id 分组记忆

        Args:
            memories: 记忆列表

        Returns:
            user_id -> 记忆列表 的映射
        """
        groups: Dict[str, List[Memory]] = defaultdict(list)
        for m in memories:
            groups[m.user_id].append(m)
        return groups

    # ── 关键词提取 ──

    @staticmethod
    def _extract_keywords(memory: Memory) -> List[str]:
        """从记忆中提取关键词

        优先使用记忆自带的 keywords 字段，否则从 content 中提取。
        """
        if memory.keywords:
            return [kw.lower().strip() for kw in memory.keywords if kw.strip()]

        # 简单中文/英文分词
        content = memory.content
        # 提取中文词（2-4字）
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', content)
        # 提取英文词
        english_words = re.findall(r'[a-zA-Z]{3,}', content)

        all_words = [w.lower() for w in chinese_words + english_words]
        # 过滤停用词
        return [w for w in all_words if w not in _STOP_WORDS]

"""
场景向量存储（ChromaDB）

管理 proactive_scenes 向量集合，提供场景检索和管理功能。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from iris_memory.proactive.core.models import ProactiveScene
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.scene_store")


class SceneStore:
    """场景向量存储（基于 ChromaDB）

    管理 proactive_scenes 集合：
    - 向量检索相似场景
    - 场景增删改查
    - 批量初始化预定义场景
    """

    COLLECTION_NAME = "proactive_scenes"

    def __init__(self, chroma_manager: Any) -> None:
        """
        Args:
            chroma_manager: ChromaManager 实例，提供 ChromaDB 连接
        """
        self._chroma_manager = chroma_manager
        self._collection: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化 ChromaDB 集合"""
        if self._initialized:
            return
        try:
            client = self._chroma_manager.client
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            count = self._collection.count()
            logger.info(f"SceneStore initialized: {count} scenes in collection")
        except Exception as e:
            logger.error(f"Failed to initialize SceneStore: {e}")
            raise

    async def query_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ProactiveScene]:
        """向量检索相似场景

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            where_filter: ChromaDB 过滤条件

        Returns:
            按相似度排序的场景列表（包含相似度分数）
        """
        if not self._collection:
            logger.warning("SceneStore not initialized")
            return []

        try:
            # 默认只查活跃场景
            if where_filter is None:
                where_filter = {"is_active": True}

            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "distances"],
            )

            scenes: List[ProactiveScene] = []
            if not results or not results.get("ids") or not results["ids"][0]:
                return scenes

            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

            for scene_id, metadata, distance in zip(ids, metadatas, distances):
                # ChromaDB cosine 返回的 distance = 1 - similarity
                similarity = max(0.0, 1.0 - distance)

                keywords_raw = metadata.get("keywords", "[]")
                try:
                    keywords = json.loads(keywords_raw) if isinstance(keywords_raw, str) else keywords_raw
                except (json.JSONDecodeError, TypeError):
                    keywords = []

                scene = ProactiveScene(
                    scene_id=scene_id,
                    description=metadata.get("trigger_pattern", ""),
                    keywords=keywords if isinstance(keywords, list) else [],
                    scene_type=metadata.get("scene_type", "chat"),
                    target_emotion=metadata.get("target_emotion"),
                    time_pattern=metadata.get("time_pattern"),
                    is_active=metadata.get("is_active", True),
                    # similarity stored in success_rate temporarily for sorting
                    # 注意：真实的 success_rate 来自 SQLite，在 VectorDetector 中加载
                )
                # 使用 success_rate 临时存储原始相似度
                scene.success_rate = similarity
                scenes.append(scene)

            return scenes
        except Exception as e:
            logger.error(f"Scene query failed: {e}")
            return []

    async def add_scene(
        self,
        scene: ProactiveScene,
        embedding: List[float],
    ) -> bool:
        """添加单个场景

        Args:
            scene: 场景数据
            embedding: 场景描述的嵌入向量

        Returns:
            是否添加成功
        """
        if not self._collection:
            return False
        try:
            metadata = self._build_metadata(scene)
            self._collection.add(
                ids=[scene.scene_id],
                embeddings=[embedding],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add scene {scene.scene_id}: {e}")
            return False

    async def add_scenes_batch(
        self,
        scenes: List[ProactiveScene],
        embeddings: List[List[float]],
    ) -> int:
        """批量添加场景

        Args:
            scenes: 场景列表
            embeddings: 对应的嵌入向量列表

        Returns:
            成功添加的数量
        """
        if not self._collection or not scenes:
            return 0
        try:
            ids = [s.scene_id for s in scenes]
            metadatas = [self._build_metadata(s) for s in scenes]
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Batch added {len(scenes)} scenes")
            return len(scenes)
        except Exception as e:
            logger.error(f"Batch add scenes failed: {e}")
            return 0

    async def update_scene_status(
        self, scene_id: str, is_active: bool
    ) -> bool:
        """更新场景激活状态"""
        if not self._collection:
            return False
        try:
            self._collection.update(
                ids=[scene_id],
                metadatas=[{"is_active": is_active}],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update scene {scene_id}: {e}")
            return False

    async def deactivate(self, scene_id: str) -> bool:
        """停用场景"""
        return await self.update_scene_status(scene_id, False)

    async def get_scene(self, scene_id: str) -> Optional[ProactiveScene]:
        """获取单个场景"""
        if not self._collection:
            return None
        try:
            result = self._collection.get(
                ids=[scene_id],
                include=["metadatas"],
            )
            if not result or not result.get("ids") or not result["ids"]:
                return None
            metadata = result["metadatas"][0] if result.get("metadatas") else {}
            return self._metadata_to_scene(scene_id, metadata)
        except Exception as e:
            logger.error(f"Failed to get scene {scene_id}: {e}")
            return None

    async def list_scenes(
        self,
        active_only: bool = False,
        scene_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[ProactiveScene]:
        """列出场景"""
        if not self._collection:
            return []
        try:
            where_filter: Optional[Dict[str, Any]] = None
            conditions = []
            if active_only:
                conditions.append({"is_active": True})
            if scene_type:
                conditions.append({"scene_type": scene_type})

            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}

            kwargs: Dict[str, Any] = {"include": ["metadatas"]}
            if where_filter:
                kwargs["where"] = where_filter

            result = self._collection.get(**kwargs)
            if not result or not result.get("ids"):
                return []

            scenes = []
            for i, scene_id in enumerate(result["ids"][:limit]):
                metadata = result["metadatas"][i] if result.get("metadatas") else {}
                scenes.append(self._metadata_to_scene(scene_id, metadata))
            return scenes
        except Exception as e:
            logger.error(f"Failed to list scenes: {e}")
            return []

    async def count(self) -> int:
        """返回场景总数"""
        if not self._collection:
            return 0
        return self._collection.count()

    async def close(self) -> None:
        """释放资源"""
        self._collection = None
        self._initialized = False

    # ========== 内部方法 ==========

    @staticmethod
    def _build_metadata(scene: ProactiveScene) -> Dict[str, Any]:
        """构建 ChromaDB metadata"""
        return {
            "trigger_pattern": scene.description,
            "scene_type": scene.scene_type,
            "keywords": json.dumps(scene.keywords, ensure_ascii=False),
            "target_emotion": scene.target_emotion or "any",
            "time_pattern": scene.time_pattern or "any",
            "is_active": scene.is_active,
            "created_at": scene.created_at.isoformat(),
        }

    @staticmethod
    def _metadata_to_scene(scene_id: str, metadata: Dict[str, Any]) -> ProactiveScene:
        """从 metadata 构建 ProactiveScene"""
        keywords_raw = metadata.get("keywords", "[]")
        try:
            keywords = json.loads(keywords_raw) if isinstance(keywords_raw, str) else keywords_raw
        except (json.JSONDecodeError, TypeError):
            keywords = []

        created_at_raw = metadata.get("created_at")
        try:
            created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else datetime.now()
        except (ValueError, TypeError):
            created_at = datetime.now()

        return ProactiveScene(
            scene_id=scene_id,
            description=metadata.get("trigger_pattern", ""),
            keywords=keywords if isinstance(keywords, list) else [],
            scene_type=metadata.get("scene_type", "chat"),
            target_emotion=metadata.get("target_emotion"),
            time_pattern=metadata.get("time_pattern"),
            is_active=metadata.get("is_active", True),
            created_at=created_at,
        )

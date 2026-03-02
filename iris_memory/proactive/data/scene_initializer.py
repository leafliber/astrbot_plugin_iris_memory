"""
场景初始化器

从 YAML 文件加载预定义场景并导入到 SceneStore 和 FeedbackStore。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris_memory.proactive.core.cold_start import ColdStartStrategy
from iris_memory.proactive.core.models import ProactiveScene
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.scene_initializer")

# YAML 文件路径
_SCENES_FILE = Path(__file__).parent / "predefined_scenes.yaml"


class SceneInitializer:
    """场景初始化器

    负责：
    1. 从 YAML 加载预定义场景
    2. 向量化场景描述
    3. 批量导入到 ChromaDB (SceneStore)
    4. 初始化 SQLite 中的权重 (FeedbackStore)
    """

    def __init__(
        self,
        scene_store: Optional[Any] = None,
        feedback_store: Optional[Any] = None,
        embedding_manager: Optional[Any] = None,
    ) -> None:
        self._scene_store = scene_store
        self._feedback_store = feedback_store
        self._embedding_manager = embedding_manager

    async def initialize_scenes(
        self,
        force_reinit: bool = False,
    ) -> int:
        """初始化预定义场景

        Args:
            force_reinit: 是否强制重新初始化

        Returns:
            导入的场景数量
        """
        if not self._scene_store:
            logger.warning("SceneStore not available, skip scene init")
            return 0

        # 检查是否已有场景
        existing_count = await self._scene_store.count()
        if existing_count > 0 and not force_reinit:
            logger.info(
                f"SceneStore already has {existing_count} scenes, "
                f"skip initialization"
            )
            return existing_count

        # 加载 YAML
        scenes = self._load_scenes_from_yaml()
        if not scenes:
            logger.warning("No scenes loaded from YAML")
            return 0

        # 向量化
        embeddings = await self._vectorize_scenes(scenes)
        if len(embeddings) != len(scenes):
            logger.error("Embedding count mismatch, aborting")
            return 0

        # 批量导入 ChromaDB
        count = await self._scene_store.add_scenes_batch(scenes, embeddings)

        # 初始化 SQLite 权重
        if self._feedback_store:
            await self._init_scene_weights(scenes)

        logger.info(f"Initialized {count} predefined scenes")
        return count

    def _load_scenes_from_yaml(self) -> List[ProactiveScene]:
        """从 YAML 文件加载场景"""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not available, using fallback loader")
            return self._load_scenes_fallback()

        if not _SCENES_FILE.exists():
            logger.error(f"Scenes file not found: {_SCENES_FILE}")
            return []

        try:
            with open(_SCENES_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            scenes: List[ProactiveScene] = []
            for scene_id, info in data.items():
                scene = ProactiveScene(
                    scene_id=scene_id,
                    description=info.get("description", ""),
                    keywords=info.get("keywords", []),
                    scene_type=info.get("type", "chat"),
                    target_emotion=info.get("target_emotion"),
                    time_pattern=info.get("time_pattern"),
                    is_active=True,
                    created_at=datetime.now(),
                )
                scenes.append(scene)

            return scenes
        except Exception as e:
            logger.error(f"Failed to load scenes YAML: {e}")
            return []

    def _load_scenes_fallback(self) -> List[ProactiveScene]:
        """简易 YAML 解析回退（当 PyYAML 不可用时）"""
        if not _SCENES_FILE.exists():
            return []

        scenes: List[ProactiveScene] = []
        try:
            content = _SCENES_FILE.read_text(encoding="utf-8")
            # 简单按 scene_XXX: 分割
            import re
            blocks = re.split(r"\n(scene_\d+):", content)
            # blocks[0] 是头部注释，之后每对 (id, body)
            for i in range(1, len(blocks), 2):
                scene_id = blocks[i].strip()
                body = blocks[i + 1] if i + 1 < len(blocks) else ""

                desc_match = re.search(r'description:\s*"([^"]*)"', body)
                type_match = re.search(r'type:\s*"([^"]*)"', body)
                emotion_match = re.search(r'target_emotion:\s*"([^"]*)"', body)
                time_match = re.search(r'time_pattern:\s*"([^"]*)"', body)
                rate_match = re.search(r'initial_success_rate:\s*([\d.]+)', body)

                # 解析 keywords
                kw_match = re.search(r'keywords:\s*\[([^\]]*)\]', body)
                keywords = []
                if kw_match:
                    keywords = [
                        k.strip().strip('"').strip("'")
                        for k in kw_match.group(1).split(",")
                        if k.strip()
                    ]

                scenes.append(ProactiveScene(
                    scene_id=scene_id,
                    description=desc_match.group(1) if desc_match else "",
                    keywords=keywords,
                    scene_type=type_match.group(1) if type_match else "chat",
                    target_emotion=emotion_match.group(1) if emotion_match else None,
                    time_pattern=time_match.group(1) if time_match else None,
                    is_active=True,
                ))
        except Exception as e:
            logger.error(f"Fallback scene loading failed: {e}")

        return scenes

    async def _vectorize_scenes(
        self, scenes: List[ProactiveScene]
    ) -> List[List[float]]:
        """向量化场景描述"""
        if not self._embedding_manager:
            logger.error("Embedding manager not available")
            return []

        texts = [s.description for s in scenes]
        try:
            if hasattr(self._embedding_manager, "embed_batch"):
                embeddings = await self._embedding_manager.embed_batch(texts)
            else:
                embeddings = []
                for text in texts:
                    vec = await self._embedding_manager.embed(text)
                    embeddings.append(vec)
            return embeddings
        except Exception as e:
            logger.error(f"Scene vectorization failed: {e}")
            return []

    async def _init_scene_weights(
        self, scenes: List[ProactiveScene]
    ) -> None:
        """初始化场景权重"""
        for scene in scenes:
            initial_rate = ColdStartStrategy.get_initial_success_rate(
                scene.scene_type
            )
            try:
                await self._feedback_store.upsert_scene_weight(
                    scene_id=scene.scene_id,
                    success_rate=initial_rate,
                    usage_count=0,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to init weight for {scene.scene_id}: {e}"
                )

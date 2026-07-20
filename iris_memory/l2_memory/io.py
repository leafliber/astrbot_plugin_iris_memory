"""
Iris Chat Memory - L2 记忆导入导出

提供记忆的导入导出功能，支持：
- JSON 格式导出/导入
- 增量导出（按时间、群聊ID筛选）
- 去重导入
- 导入导出统计
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict

from iris_memory.core import get_logger
from iris_memory.utils import atomic_write_json
from .models import MemoryEntry
from .adapter import L2MemoryAdapter

logger = get_logger("l2_memory.io")


# ============================================================================
# 数据类定义
# ============================================================================


@dataclass
class ExportStats:
    """导出统计信息

    Attributes:
        total_count: 总条目数
        exported_count: 成功导出数
        skipped_count: 跳过数
        file_size_bytes: 文件大小（字节）
        export_time: 导出时间戳
    """

    total_count: int
    exported_count: int
    skipped_count: int
    file_size_bytes: int
    export_time: str


@dataclass
class ImportStats:
    """导入统计信息

    Attributes:
        total_count: 文件中总条目数
        imported_count: 成功导入数
        skipped_count: 跳过数（重复/无效）
        error_count: 错误数
        import_time: 导入时间戳
    """

    total_count: int
    imported_count: int
    skipped_count: int
    error_count: int
    import_time: str


@dataclass
class MemoryExport:
    """记忆导出数据结构

    Attributes:
        version: 导出版本
        persona_id: 人格ID
        export_time: 导出时间
        entries: 记忆条目列表
        stats: 导出统计
    """

    version: str
    persona_id: str
    export_time: str
    entries: List[Dict[str, Any]]
    stats: Optional[Dict[str, Any]] = None


# ============================================================================
# 导出功能
# ============================================================================


class MemoryExporter:
    """记忆导出器

    提供记忆导出功能，支持筛选和格式化。

    Examples:
        >>> exporter = MemoryExporter(adapter)
        >>> stats = await exporter.export_to_file(
        ...     Path("memories.json"),
        ...     group_id="group_123"
        ... )
    """

    VERSION = "1.0"

    def __init__(self, adapter: L2MemoryAdapter):
        """初始化导出器

        Args:
            adapter: L2 记忆适配器
        """
        self._adapter = adapter

    async def export_to_file(
        self,
        file_path: Path,
        group_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filter_func: Optional[Callable[[MemoryEntry], bool]] = None,
        indent: int = 2,
    ) -> ExportStats:
        """导出记忆到文件

        Args:
            file_path: 导出文件路径
            group_id: 群聊ID筛选（可选）
            start_time: 起始时间筛选（可选）
            end_time: 结束时间筛选（可选）
            filter_func: 自定义筛选函数（可选）
            indent: JSON 缩进

        Returns:
            导出统计信息

        Examples:
            >>> stats = await exporter.export_to_file(
            ...     Path("backup.json"),
            ...     group_id="group_123"
            ... )
        """
        if not self._adapter.is_available:
            logger.error("L2 记忆库不可用，无法导出")
            return ExportStats(
                total_count=0,
                exported_count=0,
                skipped_count=0,
                file_size_bytes=0,
                export_time=datetime.now().isoformat(),
            )

        logger.info(f"开始导出记忆到 {file_path}")

        # 获取所有条目
        all_entries = await self._adapter.get_all_entries()
        total_count = len(all_entries)

        # 筛选条目
        filtered_entries = []
        for entry in all_entries:
            # 群聊筛选
            if group_id and entry.group_id != group_id:
                continue

            # 时间筛选
            if start_time or end_time:
                entry_time = entry.timestamp
                if entry_time:
                    try:
                        entry_dt = datetime.fromisoformat(entry_time)
                        if start_time and entry_dt < start_time:
                            continue
                        if end_time and entry_dt > end_time:
                            continue
                    except ValueError:
                        pass

            # 自定义筛选
            if filter_func and not filter_func(entry):
                continue

            filtered_entries.append(entry)

        exported_count = len(filtered_entries)
        skipped_count = total_count - exported_count

        # 构建导出数据
        export_data = MemoryExport(
            version=self.VERSION,
            persona_id=self._adapter._persona_id,
            export_time=datetime.now().isoformat(),
            entries=[entry.to_dict() for entry in filtered_entries],
            stats={
                "total_count": total_count,
                "exported_count": exported_count,
                "skipped_count": skipped_count,
            },
        )

        # 原子写入：避免写入中途崩溃导致导出文件截断损坏
        atomic_write_json(
            file_path, asdict(export_data), ensure_ascii=False, indent=indent
        )

        file_size = file_path.stat().st_size

        logger.info(
            f"导出完成：共 {total_count} 条，"
            f"导出 {exported_count} 条，跳过 {skipped_count} 条，"
            f"文件大小 {file_size} 字节"
        )

        return ExportStats(
            total_count=total_count,
            exported_count=exported_count,
            skipped_count=skipped_count,
            file_size_bytes=file_size,
            export_time=export_data.export_time,
        )

    async def export_all(self, file_path: Path) -> ExportStats:
        """导出所有记忆

        Args:
            file_path: 导出文件路径

        Returns:
            导出统计信息
        """
        return await self.export_to_file(file_path)

    def export_to_json(self, entries: List[MemoryEntry]) -> str:
        """导出记忆条目为 JSON 字符串

        Args:
            entries: 记忆条目列表

        Returns:
            JSON 字符串
        """
        export_data = MemoryExport(
            version=self.VERSION,
            persona_id=self._adapter._persona_id,
            export_time=datetime.now().isoformat(),
            entries=[entry.to_dict() for entry in entries],
        )
        return json.dumps(asdict(export_data), ensure_ascii=False, indent=2)


# ============================================================================
# 导入功能
# ============================================================================


class MemoryImporter:
    """记忆导入器

    提供记忆导入功能，支持去重和错误处理。

    Examples:
        >>> importer = MemoryImporter(adapter)
        >>> stats = await importer.import_from_file(Path("memories.json"))
    """

    def __init__(self, adapter: L2MemoryAdapter):
        """初始化导入器

        Args:
            adapter: L2 记忆适配器
        """
        self._adapter = adapter

    async def import_from_file(
        self,
        file_path: Path,
        skip_duplicates: bool = True,
        update_metadata: bool = False,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> ImportStats:
        """从文件导入记忆

        Args:
            file_path: 导入文件路径
            skip_duplicates: 是否跳过重复记忆
            update_metadata: 是否更新元数据
            metadata_updates: 要更新的元数据字段

        Returns:
            导入统计信息

        Examples:
            >>> stats = await importer.import_from_file(
            ...     Path("backup.json"),
            ...     metadata_updates={"imported_from": "backup"}
            ... )
        """
        if not self._adapter.is_available:
            logger.error("L2 记忆库不可用，无法导入")
            return ImportStats(
                total_count=0,
                imported_count=0,
                skipped_count=0,
                error_count=0,
                import_time=datetime.now().isoformat(),
            )

        if not file_path.exists():
            logger.error(f"导入文件不存在：{file_path}")
            return ImportStats(
                total_count=0,
                imported_count=0,
                skipped_count=0,
                error_count=1,
                import_time=datetime.now().isoformat(),
            )

        logger.info(f"开始从 {file_path} 导入记忆")

        # 读取文件
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败：{e}")
            return ImportStats(
                total_count=0,
                imported_count=0,
                skipped_count=0,
                error_count=1,
                import_time=datetime.now().isoformat(),
            )

        # 解析导出数据
        # 文件级 persona_id：旧版导出文件无逐条 persona_id 时作为回退，
        # 避免历史数据导入时全部塌缩为 "default"（仍是原适配器人格）。
        file_persona_id = "default"
        if isinstance(data, dict) and "entries" in data:
            # 新格式：MemoryExport 结构
            entries_data = data["entries"]
            file_persona_id = data.get("persona_id", "default")
            logger.debug(f"导入格式版本：{data.get('version', 'unknown')}")
        elif isinstance(data, list):
            # 旧格式：直接是条目列表
            entries_data = data
        else:
            logger.error("无法识别的导入文件格式")
            return ImportStats(
                total_count=0,
                imported_count=0,
                skipped_count=0,
                error_count=1,
                import_time=datetime.now().isoformat(),
            )

        total_count = len(entries_data)
        imported_count = 0
        skipped_count = 0
        error_count = 0

        # 导入每个条目
        for entry_data in entries_data:
            try:
                # 解析条目
                content = entry_data.get("content")
                if not content:
                    skipped_count += 1
                    continue

                metadata = entry_data.get("metadata", {})

                # 更新元数据
                if update_metadata and metadata_updates:
                    metadata.update(metadata_updates)

                # 透传 persona_id：逐条优先，缺失时回退到文件级 persona_id，
                # 再回退 "default"。这是模型迁移保留人格隔离的关键——
                # 否则 add_memory 默认 "default" 会把所有人格的记忆混入同一命名空间。
                persona_id = entry_data.get("persona_id", file_persona_id)

                # skip_duplicates 映射为 skip_dedup：skip_duplicates=False 时
                # 必须传 skip_dedup=True 关闭去重，否则相似度≥阈值的条目被
                # add_memory 静默丢弃（返回已存在 id），迁移时丢失近重复记忆。
                skip_dedup = not skip_duplicates

                # 添加记忆
                memory_id = await self._adapter.add_memory(
                    content, metadata, skip_dedup=skip_dedup, persona_id=persona_id
                )

                if memory_id:
                    imported_count += 1
                else:
                    # add_memory 返回 None 表示跳过（重复）或失败
                    if skip_duplicates:
                        skipped_count += 1
                    else:
                        error_count += 1

            except Exception as e:
                logger.error(f"导入条目失败：{e}")
                error_count += 1

        logger.info(
            f"导入完成：共 {total_count} 条，"
            f"导入 {imported_count} 条，跳过 {skipped_count} 条，错误 {error_count} 条"
        )

        return ImportStats(
            total_count=total_count,
            imported_count=imported_count,
            skipped_count=skipped_count,
            error_count=error_count,
            import_time=datetime.now().isoformat(),
        )

    async def import_from_json(
        self, json_str: str, skip_duplicates: bool = True
    ) -> ImportStats:
        """从 JSON 字符串导入记忆

        Args:
            json_str: JSON 字符串
            skip_duplicates: 是否跳过重复记忆

        Returns:
            导入统计信息
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败：{e}")
            return ImportStats(
                total_count=0,
                imported_count=0,
                skipped_count=0,
                error_count=1,
                import_time=datetime.now().isoformat(),
            )

        # 复用文件导入逻辑，使用唯一临时文件避免并发导入互相覆盖
        fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix="iris_import_")
        temp_file = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            stats = await self.import_from_file(temp_file, skip_duplicates)
        finally:
            temp_file.unlink(missing_ok=True)

        return stats

    async def import_entries(
        self, entries: List[MemoryEntry], skip_duplicates: bool = True
    ) -> ImportStats:
        """直接导入记忆条目

        Args:
            entries: 记忆条目列表
            skip_duplicates: 是否跳过重复记忆

        Returns:
            导入统计信息
        """
        total_count = len(entries)
        imported_count = 0
        skipped_count = 0
        error_count = 0

        for entry in entries:
            try:
                memory_id = await self._adapter.add_memory(
                    entry.content,
                    entry.metadata,
                    skip_dedup=not skip_duplicates,
                    persona_id=entry.persona_id,
                )

                if memory_id:
                    imported_count += 1
                else:
                    if skip_duplicates:
                        skipped_count += 1
                    else:
                        error_count += 1

            except Exception as e:
                logger.error(f"导入条目失败：{e}")
                error_count += 1

        return ImportStats(
            total_count=total_count,
            imported_count=imported_count,
            skipped_count=skipped_count,
            error_count=error_count,
            import_time=datetime.now().isoformat(),
        )


# ============================================================================
# 便捷函数
# ============================================================================


async def export_memories(
    adapter: L2MemoryAdapter, file_path: Path, **kwargs
) -> ExportStats:
    """便捷导出函数

    Args:
        adapter: L2 记忆适配器
        file_path: 导出文件路径
        **kwargs: 传递给 export_to_file 的参数

    Returns:
        导出统计信息
    """
    exporter = MemoryExporter(adapter)
    return await exporter.export_to_file(file_path, **kwargs)


async def import_memories(
    adapter: L2MemoryAdapter, file_path: Path, **kwargs
) -> ImportStats:
    """便捷导入函数

    Args:
        adapter: L2 记忆适配器
        file_path: 导入文件路径
        **kwargs: 传递给 import_from_file 的参数

    Returns:
        导入统计信息
    """
    importer = MemoryImporter(adapter)
    return await importer.import_from_file(file_path, **kwargs)

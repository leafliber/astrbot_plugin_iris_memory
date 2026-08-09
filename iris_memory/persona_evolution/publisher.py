"""
Iris Chat Memory - 人格自迭代发布器

与 AstrBot PersonaManager 的全部交互集中在此：
- 受控区块标记（文档 §6.1）解析与首次追加；
- 发布流程（文档 §11.1）：candidate Revision → per-persona 发布锁 →
  重读核 base_hash → update_persona(只传 persona_id+system_prompt) →
  回读验证 → 原子置 applied；
- PersonaManager 在本仓库零写先例：全部 getattr 探测 + try/except 降级。

persona 哈希约定：sha256(system_prompt 的 UTF-8 编码) 十六进制。
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from iris_memory.core import get_logger
from .models import ErrorCode, EvolutionJob, PersonaRevision, RevisionStatus

logger = get_logger("persona_evolution.publisher")

# 受控区块统一标记（文档 §6.1）
MANAGED_BLOCK_BEGIN = "<!-- IRIS_EVOLUTION:BEGIN v1 -->"
MANAGED_BLOCK_END = "<!-- IRIS_EVOLUTION:END -->"

# 区块初始内容（首次追加时写入）
_BLOCK_SEED = "[自动迭代生成的表达风格、语气、长度、节奏与互动习惯]"

# 标记解析结果状态
MARKERS_OK = "ok"  # 恰好一对标记，顺序正确，无嵌套
MARKERS_ABSENT = "absent"  # 完全没有标记（首次运行，允许尾部追加）
MARKERS_INVALID = "invalid"  # 重复 / 单侧缺失 / 嵌套 → conflict


def persona_hash(system_prompt: str) -> str:
    """persona 哈希 = sha256(system_prompt)"""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


@dataclass
class BlockSplit:
    """受控区块解析结果

    Attributes:
        status: MARKERS_OK / MARKERS_ABSENT / MARKERS_INVALID
        before: BEGIN 标记前内容（含标记行本身之前的全部字节）
        inner: 标记之间的内容
        after: END 标记后内容
    """

    status: str
    before: str = ""
    inner: str = ""
    after: str = ""


def split_managed_block(prompt: str) -> BlockSplit:
    """解析受控区块标记

    规则（文档 §6.1）：
    - BEGIN/END 各恰好一次、BEGIN 在 END 前、区块内不含任何标记 → OK；
    - 两个标记都不存在 → ABSENT；
    - 其余（重复 / 单侧缺失 / 嵌套）→ INVALID，禁止自动修复。
    """
    begin_count = prompt.count(MANAGED_BLOCK_BEGIN)
    end_count = prompt.count(MANAGED_BLOCK_END)
    if begin_count == 0 and end_count == 0:
        return BlockSplit(status=MARKERS_ABSENT)
    if begin_count != 1 or end_count != 1:
        return BlockSplit(status=MARKERS_INVALID)

    begin_at = prompt.index(MANAGED_BLOCK_BEGIN)
    end_at = prompt.index(MANAGED_BLOCK_END)
    if begin_at > end_at:
        return BlockSplit(status=MARKERS_INVALID)

    inner_start = begin_at + len(MANAGED_BLOCK_BEGIN)
    inner = prompt[inner_start:end_at]
    # 嵌套检查：区块内不得出现任何标记片段
    if MANAGED_BLOCK_BEGIN in inner or MANAGED_BLOCK_END in inner:
        return BlockSplit(status=MARKERS_INVALID)

    return BlockSplit(
        status=MARKERS_OK,
        before=prompt[:inner_start],
        inner=inner,
        after=prompt[end_at:],
    )


def append_managed_block(prompt: str) -> str:
    """首次运行时在原人格尾部追加受控区块"""
    suffix = f"\n\n{MANAGED_BLOCK_BEGIN}\n{_BLOCK_SEED}\n{MANAGED_BLOCK_END}\n"
    if not prompt:
        return suffix.lstrip("\n")
    return prompt.rstrip("\n") + suffix


def get_persona_manager(context: Any) -> Any:
    """从 AstrBot Context 探测 PersonaManager（不存在返回 None）"""
    if context is None:
        return None
    return getattr(context, "persona_manager", None)


async def read_persona_prompt(
    persona_manager: Any, persona_id: str
) -> Tuple[Optional[str], Optional[ErrorCode]]:
    """读取 Persona 的 system_prompt

    Returns:
        (system_prompt, None)；Persona 不存在 / 接口不可用返回错误码
    """
    if persona_manager is None:
        return None, ErrorCode.PERSONA_NOT_FOUND
    get_persona = getattr(persona_manager, "get_persona", None)
    if get_persona is None:
        return None, ErrorCode.PERSONA_NOT_FOUND
    try:
        persona = await get_persona(persona_id)
    except Exception as e:
        logger.warning(f"get_persona({persona_id}) 失败：{e}")
        return None, ErrorCode.PERSONA_NOT_FOUND
    if persona is None:
        return None, ErrorCode.PERSONA_NOT_FOUND
    system_prompt = getattr(persona, "system_prompt", None)
    if not isinstance(system_prompt, str):
        return None, ErrorCode.PERSONA_NOT_FOUND
    return system_prompt, None


class PersonaPublisher:
    """PersonaManager 发布器

    持有 per-persona 发布锁，保证同一 Persona 的发布串行。
    全部 PersonaManager 调用 getattr 探测 + try/except 降级。
    """

    def __init__(self, context: Any, storage: Any):
        self._context = context
        self._storage = storage
        self._publish_locks: Dict[str, asyncio.Lock] = {}

    def _publish_lock(self, persona_id: str) -> asyncio.Lock:
        lock = self._publish_locks.get(persona_id)
        if lock is None:
            lock = asyncio.Lock()
            self._publish_locks[persona_id] = lock
        return lock

    async def read_current_prompt(
        self, persona_id: str
    ) -> Tuple[Optional[str], Optional[ErrorCode]]:
        """读取当前 Persona 的 system_prompt（无锁）"""
        return await read_persona_prompt(
            get_persona_manager(self._context), persona_id
        )

    async def publish(
        self,
        job: EvolutionJob,
        revision: PersonaRevision,
        final_status: RevisionStatus = RevisionStatus.APPLIED,
    ) -> Optional[ErrorCode]:
        """执行发布流程（文档 §11.1）

        前提：revision 已以 candidate 状态入库，确定性闸门全部通过。

        Args:
            final_status: 发布成功后的 Revision 状态；常规发布为 applied，
                回滚发布传 rollback 以保留时间线上的回滚标识（§13.3）

        Returns:
            None 表示发布成功；否则返回错误码
            （Revision 状态已在内部相应更新）
        """
        persona_manager = get_persona_manager(self._context)
        update_persona = (
            getattr(persona_manager, "update_persona", None)
            if persona_manager is not None
            else None
        )
        if update_persona is None:
            logger.warning("PersonaManager 不可用，无法发布")
            self._storage.update_revision(
                revision.id, {"status": RevisionStatus.PUBLISH_FAILED.value}
            )
            return ErrorCode.PUBLISH_FAILED

        async with self._publish_lock(job.persona_id):
            # 记录发布意图：先于 PersonaManager 调用落库，供崩溃恢复对账
            self._storage.update_revision(
                revision.id, {"status": RevisionStatus.PUBLISHING.value}
            )

            # 重读并核对 base_hash（发布锁内防并发写）
            current, err = await self.read_current_prompt(job.persona_id)
            if err is not None or current is None:
                self._storage.update_revision(
                    revision.id, {"status": RevisionStatus.PUBLISH_FAILED.value}
                )
                return err or ErrorCode.PERSONA_NOT_FOUND
            if persona_hash(current) != revision.base_hash:
                # 生成后 Persona 被外部编辑：不覆盖，交由冲突流程处理
                logger.warning(
                    f"发布前哈希校验失败（Job {job.id}）：外部修改，停止发布"
                )
                self._storage.update_revision(
                    revision.id, {"status": RevisionStatus.PUBLISH_FAILED.value}
                )
                return ErrorCode.BASE_HASH_MISMATCH

            try:
                # 只传 persona_id 与 system_prompt（文档 §2.1）：
                # 不重新传入 tools/skills/begin_dialogs 等未修改字段
                await update_persona(
                    job.persona_id, system_prompt=revision.result_prompt
                )
            except Exception as e:
                logger.warning(f"update_persona({job.persona_id}) 失败：{e}")
                self._storage.update_revision(
                    revision.id, {"status": RevisionStatus.PUBLISH_FAILED.value}
                )
                return ErrorCode.PUBLISH_FAILED

            # 回读验证结果哈希
            after, err = await self.read_current_prompt(job.persona_id)
            if err is not None or after is None:
                self._storage.update_revision(
                    revision.id, {"status": RevisionStatus.PUBLISH_FAILED.value}
                )
                return ErrorCode.PUBLISH_FAILED
            if persona_hash(after) != revision.result_hash:
                logger.warning("发布回读验证失败：结果哈希不一致")
                self._storage.update_revision(
                    revision.id, {"status": RevisionStatus.PUBLISH_FAILED.value}
                )
                return ErrorCode.PUBLISH_FAILED

            # 原子置已发布：Revision 状态 + Job 基线同事务更新
            applied_at = time.time()
            self._storage.mark_revision_applied(
                revision.id, job.id, applied_at, status=final_status.value
            )
            logger.info(
                f"Job {job.id} Revision v{revision.version} 发布成功"
                f"（persona={job.persona_id}）"
            )
            return None

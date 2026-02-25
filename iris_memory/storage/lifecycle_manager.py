"""
会话生命周期管理器
管理会话的状态转换和同步
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List

from iris_memory.utils.logger import get_logger
from iris_memory.core.types import StorageLayer
from iris_memory.core.upgrade_evaluator import UpgradeEvaluator, UpgradeMode

# 模块logger
logger = get_logger("lifecycle_manager")


class SessionState(str, Enum):
    """会话状态"""
    
    # 活跃状态：会话正在进行中
    ACTIVE = "active"
    
    # 非活跃状态：会话存在但暂时未活动
    INACTIVE = "inactive"
    
    # 关闭状态：会话已结束
    CLOSED = "closed"
    
    # 归档状态：会话数据已归档
    ARCHIVED = "archived"


class SessionLifecycleManager:
    """会话生命周期管理器
    
    管理会话的状态转换、清理和持久化
    """
    
    def __init__(
        self,
        session_manager,
        chroma_manager=None,  # 新增：Chroma管理器用于持久化升级
        cleanup_interval: int = 3600,  # 清理间隔（秒），默认1小时
        session_timeout: int = 86400,  # 会话超时（秒），默认24小时
        inactive_timeout: int = 1800,  # 非活跃超时（秒），默认30分钟
        promotion_interval: int = 3600,  # 记忆升级检查间隔（秒），默认1小时
        upgrade_mode: str = "rule",  # 升级判断模式: rule, llm, hybrid
        llm_upgrade_batch_size: int = 5,
        llm_upgrade_threshold: float = 0.7
    ):
        """初始化生命周期管理器
        
        Args:
            session_manager: 会话管理器实例
            chroma_manager: Chroma管理器实例（用于持久化记忆升级）
            cleanup_interval: 定时清理间隔
            session_timeout: 会话超时时间
            inactive_timeout: 非活跃超时时间
            promotion_interval: 记忆升级检查间隔
            upgrade_mode: 升级判断模式 (rule/llm/hybrid)
            llm_upgrade_batch_size: LLM批量评估大小
            llm_upgrade_threshold: LLM升级置信度阈值
        """
        self.session_manager = session_manager
        self.chroma_manager = chroma_manager  # 新增
        self.cleanup_interval = cleanup_interval
        self.session_timeout = session_timeout
        self.inactive_timeout = inactive_timeout
        self.promotion_interval = promotion_interval  # 新增
        
        # 会话状态缓存：{session_key: {"state": SessionState, "last_active": datetime}}
        self.session_states: Dict[str, Dict[str, Any]] = {}
        
        # 定时任务
        self.cleanup_task = None
        
        # 升级评估器
        self.upgrade_mode = UpgradeMode(upgrade_mode) if isinstance(upgrade_mode, str) else upgrade_mode
        self.upgrade_evaluator = UpgradeEvaluator(
            mode=self.upgrade_mode,
            batch_size=llm_upgrade_batch_size,
            confidence_threshold=llm_upgrade_threshold
        )
        self.promotion_task = None  # 新增：记忆升级任务
        self.is_running = False
    
    def set_chroma_manager(self, chroma_manager):
        """设置Chroma管理器（用于延迟注入）
        
        Args:
            chroma_manager: Chroma管理器实例
        """
        self.chroma_manager = chroma_manager
        logger.debug("ChromaManager injected into SessionLifecycleManager")
    
    def set_llm_provider(self, llm_provider):
        """设置LLM提供者（用于LLM升级判断）
        
        Args:
            llm_provider: LLM提供者实例
        """
        self.upgrade_evaluator.set_llm_provider(llm_provider)
        logger.debug(f"LLM provider set for upgrade evaluation (mode={self.upgrade_mode.value})")
    
    async def start(self):
        """启动生命周期管理器"""
        if self.is_running:
            logger.debug("SessionLifecycleManager is already running")
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # 新增：启动记忆升级定时任务
        if self.chroma_manager:
            self.promotion_task = asyncio.create_task(self._promotion_loop())
            logger.debug("Memory promotion task started")
        
        logger.debug("SessionLifecycleManager started")
    
    async def stop(self):
        """停止生命周期管理器（热更新友好）"""
        logger.debug("[Hot-Reload] Stopping SessionLifecycleManager...")
        self.is_running = False
        
        # 收集所有需要取消的任务
        tasks_to_cancel = []
        if self.cleanup_task:
            tasks_to_cancel.append(("cleanup", self.cleanup_task))
        if self.promotion_task:
            tasks_to_cancel.append(("promotion", self.promotion_task))
        
        # 取消并等待所有任务
        for name, task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error cancelling {name} task: {e}")
        
        self.cleanup_task = None
        self.promotion_task = None
        
        logger.debug("[Hot-Reload] SessionLifecycleManager stopped")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _promotion_loop(self):
        """记忆升级循环 - 定期检查并执行 EPISODIC → SEMANTIC 升级"""
        while self.is_running:
            try:
                await asyncio.sleep(self.promotion_interval)
                await self._promote_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Promotion loop error: {e}")
    
    async def _promote_memories(self):
        """定期执行记忆升级检查
        
        检查并执行升级：
        1. WORKING → EPISODIC（从工作记忆升级到情景记忆）
        2. EPISODIC → SEMANTIC（从情景记忆升级到语义记忆）
        使用升级评估器（支持规则/LLM/混合模式）
        """
        if not self.chroma_manager:
            logger.warning("Cannot promote memories: chroma_manager not available")
            return
        
        try:
            # ========== 阶段 1: WORKING → EPISODIC ==========
            working_promoted = 0
            failed_promotions = 0
            
            # 从所有会话中获取工作记忆
            all_sessions = self.session_manager.get_all_sessions()
            logger.debug(f"Checking {len(all_sessions)} sessions for memory promotion")
            
            for session_key, session_data in all_sessions.items():
                working_memories = session_data.get("working_memories", [])
                if not working_memories:
                    continue
                
                # 解析 session_key
                parts = session_key.split(":")
                user_id = parts[0] if parts else None
                group_id = parts[1] if len(parts) > 1 and parts[1] != "private" else None
                
                if not user_id:
                    continue
                
                for memory in working_memories:
                    # 检查是否符合升级条件
                    if self._should_promote_working_to_episodic(memory):
                        # 升级到 EPISODIC 并保存到 Chroma
                        original_layer = memory.storage_layer
                        memory.storage_layer = StorageLayer.EPISODIC
                        
                        try:
                            success = await self.chroma_manager.add_memory(memory)
                            if success:
                                working_promoted += 1
                                # 从工作记忆中移除
                                await self.session_manager.remove_working_memory(
                                    user_id, group_id, memory.id
                                )
                                logger.debug(
                                    f"Memory {memory.id} promoted WORKING→EPISODIC "
                                    f"(RIF={memory.rif_score:.3f}, confidence={memory.confidence:.2f})"
                                )
                            else:
                                failed_promotions += 1
                                # 恢复原存储层
                                memory.storage_layer = original_layer
                                logger.warning(f"Failed to add memory {memory.id} to Chroma")
                        except Exception as e:
                            failed_promotions += 1
                            memory.storage_layer = original_layer
                            logger.error(f"Error promoting memory {memory.id}: {e}")
            
            if working_promoted > 0 or failed_promotions > 0:
                logger.debug(
                    f"Working memory promotion: {working_promoted} promoted, "
                    f"{failed_promotions} failed"
                )
            
            # ========== 阶段 2: EPISODIC → SEMANTIC ==========
            try:
                episodic_memories = await self.chroma_manager.get_memories_by_storage_layer(
                    StorageLayer.EPISODIC
                )
            except Exception as e:
                logger.warning(f"Failed to get episodic memories: {e}")
                episodic_memories = []
            
            semantic_promoted = 0
            if episodic_memories:
                logger.debug(f"Evaluating {len(episodic_memories)} episodic memories for SEMANTIC upgrade")
                
                # 使用升级评估器进行判断
                evaluation_results = await self.upgrade_evaluator.evaluate_episodic_to_semantic(
                    episodic_memories
                )
                
                for memory in episodic_memories:
                    result = evaluation_results.get(memory.id)
                    if result and result[0]:  # should_upgrade = True
                        should_upgrade, confidence, reason = result
                        
                        # 更改存储层为语义记忆
                        memory.storage_layer = StorageLayer.SEMANTIC
                        
                        # 持久化到Chroma
                        try:
                            success = await self.chroma_manager.update_memory(memory)
                            if success:
                                semantic_promoted += 1
                                logger.debug(
                                    f"Memory {memory.id} promoted EPISODIC→SEMANTIC "
                                    f"(confidence={confidence:.2f}, reason={reason})"
                                )
                            else:
                                logger.warning(
                                    f"Failed to promote memory {memory.id} to SEMANTIC"
                                )
                        except Exception as e:
                            logger.error(f"Error updating memory {memory.id}: {e}")
            
            if semantic_promoted > 0:
                logger.debug(
                    f"Memory promotion completed: {semantic_promoted} memories "
                    f"promoted from EPISODIC to SEMANTIC (mode={self.upgrade_mode.value})"
                )
                
        except Exception as e:
            logger.error(f"Failed to promote memories: {e}", exc_info=True)
    
    def _should_promote_working_to_episodic(self, memory) -> bool:
        """判断工作记忆是否应该升级到情景记忆

        升级条件（与Memory.should_upgrade_to_episodic保持一致）：
        - 访问>=1次 且 重要性>0.5
        - 或 情感强度>0.6
        - 或 置信度>=0.7
        - 或 用户主动请求的记忆
        - 或 RIF评分较高(>=0.5)且有访问

        Args:
            memory: 记忆对象

        Returns:
            bool: 是否应该升级
        """
        # 首先使用Memory模型的标准判断
        if memory.should_upgrade_to_episodic():
            return True

        # 额外条件：RIF评分较高且有访问
        if memory.rif_score >= 0.5 and memory.access_count >= 1:
            return True

        # 质量等级检查（HIGH_CONFIDENCE或更高）
        if memory.quality_level.value >= 3:
            return True

        return False
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        now = datetime.now()
        cleaned_count = 0
        archived_count = 0
        
        # 遍历所有会话状态
        for session_key, state_info in self.session_states.copy().items():
            session_state = state_info.get("state", SessionState.INACTIVE)
            last_active = state_info.get("last_active", now)
            
            # 计算非活跃时间
            inactive_duration = (now - last_active).total_seconds()
            
            # 状态转换逻辑
            if session_state == SessionState.ACTIVE:
                # 检查是否转为非活跃
                if inactive_duration > self.inactive_timeout:
                    self._update_session_state(
                        session_key, 
                        SessionState.INACTIVE
                    )
                    logger.debug(
                        f"Session {session_key} changed from ACTIVE to INACTIVE"
                    )
            
            elif session_state == SessionState.INACTIVE:
                # 检查是否应该关闭或归档
                if inactive_duration > self.session_timeout:
                    # 尝试归档工作记忆
                    if await self._archive_session(session_key):
                        self._update_session_state(
                            session_key, 
                            SessionState.ARCHIVED
                        )
                        archived_count += 1
                    else:
                        # 无法归档，直接关闭
                        self._update_session_state(
                            session_key, 
                            SessionState.CLOSED
                        )
                        await self._cleanup_session_data(session_key)
                        cleaned_count += 1
                    
                    logger.debug(
                        f"Session {session_key} changed from INACTIVE to "
                        f"{self.session_states[session_key]['state'].value}"
                    )
            
            elif session_state in [SessionState.CLOSED, SessionState.ARCHIVED]:
                # 检查是否应该完全删除
                if inactive_duration > self.session_timeout * 2:  # 48小时后删除
                    self._delete_session(session_key)
                    del self.session_states[session_key]
                    cleaned_count += 1
        
        if cleaned_count > 0 or archived_count > 0:
            logger.debug(
                f"Session cleanup completed: {cleaned_count} closed, "
                f"{archived_count} archived"
            )
    
    def _update_session_state(
        self,
        session_key: str,
        new_state: SessionState
    ):
        """更新会话状态
        
        Args:
            session_key: 会话标识符
            new_state: 新状态
        """
        if session_key not in self.session_states:
            self.session_states[session_key] = {}
        
        self.session_states[session_key]["state"] = new_state
        self.session_states[session_key]["last_updated"] = datetime.now()
    
    async def _archive_session(self, session_key: str) -> bool:
        """归档会话
        
        将工作记忆提升到情景记忆并持久化到Chroma，或清除不重要的工作记忆
        使用升级评估器（支持规则/LLM/混合模式）
        
        Args:
            session_key: 会话标识符
            
        Returns:
            bool: 是否归档成功
        """
        # 获取工作记忆
        try:
            # 解析session_key获取user_id和group_id
            parts = session_key.split(":")
            if len(parts) == 2:
                user_id, group_id_str = parts
                group_id = group_id_str if group_id_str != "private" else None
            else:
                return False
            
            # 获取工作记忆
            working_memories = await self.session_manager.get_working_memory(
                user_id, group_id
            )
            
            if not working_memories:
                return False
            
            # 使用升级评估器进行判断
            evaluation_results = await self.upgrade_evaluator.evaluate_working_to_episodic(
                working_memories
            )
            
            # 收集需要升级的记忆
            upgraded_memories = []
            discarded_memories = []
            
            for memory in working_memories:
                result = evaluation_results.get(memory.id)
                if result and result[0]:  # should_upgrade = True
                    should_upgrade, confidence, reason = result
                    # 更改存储层
                    memory.storage_layer = StorageLayer.EPISODIC
                    upgraded_memories.append(memory)
                    logger.debug(
                        f"Memory {memory.id} marked for upgrade WORKING→EPISODIC "
                        f"(confidence={confidence:.2f}, reason={reason})"
                    )
                else:
                    # 不满足升级条件的工作记忆将被清除
                    discarded_memories.append(memory)
            
            # 持久化升级后的记忆到Chroma
            # 注意：工作记忆从未存入Chroma，所以需要使用add_memory而非update_memory
            if upgraded_memories and self.chroma_manager:
                for memory in upgraded_memories:
                    try:
                        # 添加到Chroma（工作记忆首次进入持久化存储）
                        await self.chroma_manager.add_memory(memory)
                        logger.debug(
                            f"Memory {memory.id} added to Chroma as EPISODIC"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error persisting memory {memory.id}: {e}"
                        )
            
            # 清除工作记忆缓存
            await self.session_manager.clear_working_memory(user_id, group_id)
            
            if upgraded_memories:
                logger.debug(
                    f"Archived session {session_key}: "
                    f"{len(upgraded_memories)} memories upgraded to EPISODIC, "
                    f"{len(discarded_memories)} memories discarded"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to archive session {session_key}: {e}")
            return False
    
    async def _cleanup_session_data(self, session_key: str):
        """清理会话数据
        
        Args:
            session_key: 会话标识符
        """
        try:
            # 解析session_key获取user_id和group_id
            parts = session_key.split(":")
            if len(parts) == 2:
                user_id, group_id_str = parts
                group_id = group_id_str if group_id_str != "private" else None
                
                # 清除工作记忆
                await self.session_manager.clear_working_memory(user_id, group_id)
                
                logger.debug(f"Cleaned up session data: {session_key}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup session {session_key}: {e}")
    
    def _delete_session(self, session_key: str):
        """删除会话
        
        Args:
            session_key: 会话标识符
        """
        try:
            # 解析session_key获取user_id和group_id
            parts = session_key.split(":")
            if len(parts) == 2:
                user_id, group_id_str = parts
                group_id = group_id_str if group_id_str != "private" else None
                
                # 删除会话
                self.session_manager.delete_session(user_id, group_id)
                
                logger.debug(f"Deleted session: {session_key}")
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_key}: {e}")
    
    async def activate_session(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ):
        """激活会话
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        session_key = self.session_manager.get_session_key(user_id, group_id)
        
        # 确保会话在 SessionManager 中存在
        session = self.session_manager.get_session(session_key)
        if session is None:
            # 如果会话不存在，创建新会话
            self.session_manager.create_session(user_id, group_id)
        
        # 更新活动时间
        self.session_manager.update_session_activity(user_id, group_id)
        
        # 更新状态
        self._update_session_state(session_key, SessionState.ACTIVE)
        
        logger.debug(f"Session activated: {session_key}")
    
    async def deactivate_session(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ):
        """停用会话（转为非活跃）
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        session_key = self.session_manager.get_session_key(user_id, group_id)
        
        # 仅更新状态为非活跃，不清理数据
        self._update_session_state(session_key, SessionState.INACTIVE)
        
        logger.debug(f"Session deactivated: {session_key}")
    
    async def close_session(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ):
        """关闭会话
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
        """
        session_key = self.session_manager.get_session_key(user_id, group_id)
        
        # 清理工作记忆
        await self._cleanup_session_data(session_key)
        
        # 更新状态为关闭
        self._update_session_state(session_key, SessionState.CLOSED)
        
        logger.debug(f"Session closed: {session_key}")
    
    def get_session_state(
        self,
        user_id: str,
        group_id: Optional[str] = None
    ) -> Optional[SessionState]:
        """获取会话状态
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            
        Returns:
            Optional[SessionState]: 会话状态
        """
        session_key = self.session_manager.get_session_key(user_id, group_id)
        state_info = self.session_states.get(session_key)
        
        if state_info:
            return state_info.get("state")
        
        return None
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """获取会话统计信息
        
        修复：同步 SessionManager 的会话状态，确保统计准确
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        # 同步 SessionManager 中的会话状态
        all_sm_sessions = self.session_manager.get_all_sessions()
        
        # 确保所有 SessionManager 中的会话都有对应的生命周期状态
        for session_key in all_sm_sessions.keys():
            if session_key not in self.session_states:
                # 从 SessionManager 获取会话信息
                session = self.session_manager.get_session(session_key)
                if session:
                    last_active_str = session.get("last_active")
                    try:
                        from datetime import datetime
                        last_active = datetime.fromisoformat(last_active_str) if last_active_str else datetime.now()
                    except:
                        last_active = datetime.now()
                    
                    self.session_states[session_key] = {
                        "state": SessionState.INACTIVE,  # 默认非活跃
                        "last_active": last_active,
                        "last_updated": datetime.now()
                    }
        
        # 统计状态
        stats = {
            "total_sessions": len(self.session_states),
            "active_sessions": 0,
            "inactive_sessions": 0,
            "closed_sessions": 0,
            "archived_sessions": 0,
            "session_manager_sessions": len(all_sm_sessions)  # 新增：SM 中的会话数
        }
        
        for state_info in self.session_states.values():
            state = state_info.get("state", SessionState.INACTIVE)
            if state == SessionState.ACTIVE:
                stats["active_sessions"] += 1
            elif state == SessionState.INACTIVE:
                stats["inactive_sessions"] += 1
            elif state == SessionState.CLOSED:
                stats["closed_sessions"] += 1
            elif state == SessionState.ARCHIVED:
                stats["archived_sessions"] += 1
        
        return stats
    
    async def serialize_state(self) -> Dict[str, Any]:
        """序列化会话状态用于持久化
        
        Returns:
            Dict[str, Any]: 序列化的状态数据
        """
        serialized = {}
        for session_key, state_info in self.session_states.items():
            serialized[session_key] = {
                "state": state_info["state"].value,
                "last_active": state_info["last_active"].isoformat(),
                "last_updated": state_info["last_updated"].isoformat()
            }
        
        return serialized
    
    async def deserialize_state(self, data: Dict[str, Any]):
        """反序列化会话状态

        Args:
            data: 序列化的状态数据
        """
        for session_key, state_data in data.items():
            try:
                state_str = state_data.get("state", "inactive")
                state = SessionState(state_str)
            except (ValueError, KeyError):
                # 如果状态无效，使用默认值
                logger.warning(f"Invalid state '{state_data.get('state')}' for session {session_key}, using default 'inactive'")
                state = SessionState.INACTIVE

            try:
                last_active = datetime.fromisoformat(
                    state_data.get("last_active", datetime.now().isoformat())
                )
            except (ValueError, TypeError):
                # 如果日期无效，使用当前时间
                logger.warning(f"Invalid last_active for session {session_key}, using current time")
                last_active = datetime.now()

            try:
                last_updated = datetime.fromisoformat(
                    state_data.get("last_updated", datetime.now().isoformat())
                )
            except (ValueError, TypeError):
                # 如果日期无效，使用当前时间
                logger.warning(f"Invalid last_updated for session {session_key}, using current time")
                last_updated = datetime.now()

            self.session_states[session_key] = {
                "state": state,
                "last_active": last_active,
                "last_updated": last_updated
            }

        logger.debug(f"Loaded {len(self.session_states)} session states")

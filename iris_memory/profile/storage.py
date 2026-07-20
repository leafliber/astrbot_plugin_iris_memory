"""
Iris Chat Memory - 画像存储组件

使用 AstrBot KV 存储 API 实现画像数据持久化。
支持群聊隔离和人格隔离。
"""

from typing import Optional, TYPE_CHECKING, Set
import asyncio
import functools
import inspect
from contextlib import asynccontextmanager

from iris_memory.core import Component, get_logger
from iris_memory.core.storage import KVStorage
from iris_memory.config import get_config
from .models import (
    GroupProfile,
    UserProfile,
    profile_to_dict,
    dict_to_group_profile,
    dict_to_user_profile,
)

if TYPE_CHECKING:
    pass

logger = get_logger("profile")

GROUP_PROFILE_WRITABLE_FIELDS: Set[str] = {
    "group_name",
    "interests",
    "atmosphere_tags",
    "long_term_tags",
    "blacklist_topics",
    "custom_fields",
}

USER_PROFILE_WRITABLE_FIELDS: Set[str] = {
    "user_name",
    "historical_names",
    "personality_tags",
    "interests",
    "occupation",
    "language_style",
    "communication_style",
    "emotional_baseline",
    "favorability",
    "bot_relationship",
    "important_dates",
    "taboo_topics",
    "important_events",
    "custom_fields",
}


def profile_lock(kind: str):
    """装饰器：为画像 read-modify-write 操作按命名空间串行化加锁。

    kind="user" 时按 (persona, group, user) 维度加锁，kind="group" 时按
    (persona, group) 维度加锁，确保同一画像的并发「读→改→写」不会交错、
    丢失彼此的更新。通过签名绑定提取参数，被装饰方法体无需任何改动。
    """

    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            persona_id = bound.arguments.get("persona_id", "default")
            if kind == "user":
                user_id = bound.arguments.get("user_id")
                group_id = bound.arguments.get("group_id", "default")
                async with self._storage.lock_user(user_id, group_id, persona_id):
                    return await func(self, *args, **kwargs)
            else:
                group_id = bound.arguments.get("group_id")
                async with self._storage.lock_group(group_id, persona_id):
                    return await func(self, *args, **kwargs)

        return wrapper

    return decorator


class ProfileStorage(Component):
    """画像存储组件

    使用 AstrBot KV 存储 API，支持群聊隔离和人格隔离。

    存储键格式：
        - 群聊画像：group_profile:{persona_id}:{group_id}
        - 用户画像：user_profile:{persona_id}:{group_id}:{user_id}

    Attributes:
        _storage: KV 存储适配器
        _is_available: 组件是否可用
    """

    def __init__(self, storage: KVStorage):
        """初始化画像存储组件

        Args:
            storage: KV 存储适配器（实现 KVStorage 协议的对象）
        """
        super().__init__()
        self._storage = storage
        # RMW 串行化锁：按画像命名空间（persona/group/user）分配，避免并发丢失更新
        self._locks: dict = {}
        self._locks_guard = asyncio.Lock()
        # 索引列表（user_index/group_index）读-改-写的全局锁
        self._index_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """组件名称"""
        return "profile"

    async def initialize(self) -> None:
        """初始化画像存储"""
        config = get_config()

        if not config.get("profile.enable"):
            self._is_available = False
            logger.info("画像系统未启用")
            return

        self._is_available = True
        logger.info("画像存储组件初始化完成")

    async def shutdown(self) -> None:
        """关闭存储"""
        self._reset_state()
        logger.info("画像存储组件已关闭")

    async def get_group_profile(
        self, group_id: str, persona_id: str = "default"
    ) -> Optional[GroupProfile]:
        """获取群聊画像

        Args:
            group_id: 群聊ID
            persona_id: 人格ID（默认为 "default"）

        Returns:
            群聊画像对象，不存在则返回 None
        """
        if not self._is_available:
            return None

        persona_id = self._effective_persona(persona_id)
        key = f"group_profile:{persona_id}:{group_id}"

        try:
            data = await self._storage.get_kv_data(key, None)

            if data:
                profile = dict_to_group_profile(data)
                logger.debug(f"获取群聊画像成功: {key}")
                return profile

            logger.debug(f"群聊画像不存在: {key}")
            return None

        except Exception as e:
            logger.error(f"获取群聊画像失败: {key}, 错误: {e}")
            return None

    async def save_group_profile(
        self,
        profile: GroupProfile,
        increment_version: bool = True,
        persona_id: str = "default",
    ) -> None:
        if not self._is_available:
            return

        if increment_version:
            profile.version += 1

        persona_id = self._effective_persona(persona_id)
        key = f"group_profile:{persona_id}:{profile.group_id}"

        try:
            data = profile_to_dict(profile)
            await self._storage.put_kv_data(key, data)
            await self._add_to_group_index(profile.group_id, persona_id)
            await self._add_to_persona_index(persona_id)
            logger.debug(f"保存群聊画像成功: {key}, version={profile.version}")

        except Exception as e:
            logger.error(f"保存群聊画像失败: {key}, 错误: {e}")

    async def get_user_profile(
        self, user_id: str, group_id: str = "default", persona_id: str = "default"
    ) -> Optional[UserProfile]:
        """获取用户画像

        Args:
            user_id: 用户ID
            group_id: 群聊ID（全局模式传 "default"）
            persona_id: 人格ID（默认为 "default"）

        Returns:
            用户画像对象，不存在则返回 None
        """
        if not self._is_available:
            return None

        persona_id = self._effective_persona(persona_id)
        key = f"user_profile:{persona_id}:{group_id}:{user_id}"

        try:
            data = await self._storage.get_kv_data(key, None)

            if data:
                profile = dict_to_user_profile(data)
                logger.debug(f"获取用户画像成功: {key}")
                return profile

            logger.debug(f"用户画像不存在: {key}")
            return None

        except Exception as e:
            logger.error(f"获取用户画像失败: {key}, 错误: {e}")
            return None

    async def save_user_profile(
        self,
        profile: UserProfile,
        group_id: str = "default",
        increment_version: bool = True,
        persona_id: str = "default",
    ) -> None:
        if not self._is_available:
            return

        if increment_version:
            profile.version += 1

        persona_id = self._effective_persona(persona_id)
        key = f"user_profile:{persona_id}:{group_id}:{profile.user_id}"

        try:
            data = profile_to_dict(profile)
            await self._storage.put_kv_data(key, data)
            await self._add_to_user_index(profile.user_id, group_id, persona_id)
            await self._add_to_persona_index(persona_id)
            logger.debug(f"保存用户画像成功: {key}, version={profile.version}")

        except Exception as e:
            logger.error(f"保存用户画像失败: {key}, 错误: {e}")

    def _effective_persona(self, persona_id: str) -> str:
        """规范化 persona_id

        隔离未启用时强制返回 "default"（即便调用方传入了具体 persona，
        也不应产生非 default 的存储键）。隔离启用时返回传入值，空值兜底 "default"。
        """
        if not persona_id:
            return "default"
        try:
            if not get_config().get("isolation_config.enable_persona_isolation"):
                return "default"
        except RuntimeError:
            return "default"
        return persona_id

    async def _get_lock(self, key: str) -> asyncio.Lock:
        """获取（或创建）指定命名空间的 RMW 锁。"""
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    @asynccontextmanager
    async def lock_group(self, group_id: str, persona_id: str = "default"):
        """群聊画像 RMW 串行化锁。同一 (persona, group) 的读-改-写不会被并发交错。"""
        persona_id = self._effective_persona(persona_id)
        lock = await self._get_lock(f"group:{persona_id}:{group_id}")
        async with lock:
            yield

    @asynccontextmanager
    async def lock_user(
        self, user_id: str, group_id: str = "default", persona_id: str = "default"
    ):
        """用户画像 RMW 串行化锁。同一 (persona, group, user) 的读-改-写不会被并发交错。"""
        persona_id = self._effective_persona(persona_id)
        lock = await self._get_lock(f"user:{persona_id}:{group_id}:{user_id}")
        async with lock:
            yield

    async def update_group_profile(
        self, group_id: str, updates: dict, persona_id: str = "default"
    ) -> bool:
        """更新群聊画像

        Args:
            group_id: 群聊ID
            updates: 更新字段字典
            persona_id: 人格ID

        Returns:
            是否更新成功
        """
        try:
            # 读-改-写必须持锁，否则与消息驱动 update_from_analysis 并发时
            # 互相覆盖（lost update）。Web 路由 /profile/*/update 直接调此方法，
            # 此前未加锁，而管理器同类更新都持命名空间锁。
            async with self.lock_group(group_id, persona_id):
                profile = await self.get_group_profile(group_id, persona_id)

                if not profile:
                    profile = GroupProfile(group_id=group_id)

                for key, value in updates.items():
                    if key in GROUP_PROFILE_WRITABLE_FIELDS:
                        setattr(profile, key, value)

                await self.save_group_profile(profile, persona_id=persona_id)

            logger.info(f"更新群聊画像成功: {group_id}")
            return True

        except Exception as e:
            logger.error(f"更新群聊画像失败: {e}", exc_info=True)
            return False

    async def update_user_profile(
        self, user_id: str, group_id: str, updates: dict, persona_id: str = "default"
    ) -> bool:
        """更新用户画像

        Args:
            user_id: 用户ID
            group_id: 群聊ID
            updates: 更新字段字典
            persona_id: 人格ID

        Returns:
            是否更新成功
        """
        try:
            # 读-改-写必须持锁，否则与消息驱动 update_from_analysis 并发时
            # 互相覆盖（lost update）。Web 路由 /profile/*/update 直接调此方法，
            # 此前未加锁，而管理器同类更新都持命名空间锁。
            async with self.lock_user(user_id, group_id, persona_id):
                profile = await self.get_user_profile(user_id, group_id, persona_id)

                if not profile:
                    profile = UserProfile(user_id=user_id)

                for key, value in updates.items():
                    if key in USER_PROFILE_WRITABLE_FIELDS:
                        setattr(profile, key, value)

                await self.save_user_profile(
                    profile, group_id=group_id, persona_id=persona_id
                )

            logger.info(f"更新用户画像成功: {user_id}@{group_id}")
            return True

        except Exception as e:
            logger.error(f"更新用户画像失败: {e}", exc_info=True)
            return False

    async def list_groups(self, persona_id: str = "default") -> list:
        persona_id = self._effective_persona(persona_id)
        index_key = f"group_index:{persona_id}"

        try:
            group_ids = await self._storage.get_kv_data(index_key, [])

            if not group_ids:
                return []

            tasks = [
                self.get_group_profile(group_id, persona_id) for group_id in group_ids
            ]
            profiles = await asyncio.gather(*tasks, return_exceptions=True)

            groups = []
            for group_id, profile in zip(group_ids, profiles):
                if isinstance(profile, Exception):
                    logger.warning(f"获取群聊画像失败: {group_id}, 错误: {profile}")
                    continue
                if profile and isinstance(profile, GroupProfile):
                    groups.append(
                        {
                            "group_id": group_id,
                            "group_name": profile.group_name or group_id,
                        }
                    )

            return groups

        except Exception as e:
            logger.error(f"获取群聊列表失败: {e}", exc_info=True)
            return []

    async def list_users(
        self, group_id: str = "default", persona_id: str = "default"
    ) -> list:
        persona_id = self._effective_persona(persona_id)
        index_key = f"user_index:{persona_id}:{group_id}"

        try:
            user_ids = await self._storage.get_kv_data(index_key, [])

            if not user_ids:
                return []

            tasks = [
                self.get_user_profile(user_id, group_id, persona_id)
                for user_id in user_ids
            ]
            profiles = await asyncio.gather(*tasks, return_exceptions=True)

            users = []
            for user_id, profile in zip(user_ids, profiles):
                if isinstance(profile, Exception):
                    logger.warning(f"获取用户画像失败: {user_id}, 错误: {profile}")
                    continue
                if profile and isinstance(profile, UserProfile):
                    users.append(
                        {
                            "user_id": user_id,
                            "nickname": profile.user_name or user_id,
                            "group_id": group_id,
                        }
                    )

            return users

        except Exception as e:
            logger.error(f"获取用户列表失败: {e}", exc_info=True)
            return []

    async def list_all_users(self, persona_id: str = "default") -> list:
        """列出所有群聊下的用户画像

        遍历 user_group_index 获取有用户画像的 group_id 列表，
        再逐个群聊拉取用户列表。用于 Web UI 无指定群聊时展示全部用户。

        Args:
            persona_id: 人格ID

        Returns:
            用户列表，每项包含 user_id / nickname / group_id
        """
        persona_id = self._effective_persona(persona_id)
        ug_index_key = f"user_group_index:{persona_id}"

        try:
            group_ids = await self._storage.get_kv_data(ug_index_key, [])
            if not group_ids:
                return []

            all_users = []
            for gid in group_ids:
                users = await self.list_users(gid, persona_id)
                all_users.extend(users)

            return all_users

        except Exception as e:
            logger.error(f"获取全部用户列表失败: {e}", exc_info=True)
            return []

    async def _add_to_group_index(self, group_id: str, persona_id: str) -> None:
        index_key = f"group_index:{persona_id}"
        try:
            async with self._index_lock:
                group_ids = await self._storage.get_kv_data(index_key, [])
                if group_id not in group_ids:
                    group_ids.append(group_id)
                    await self._storage.put_kv_data(index_key, group_ids)
        except Exception as e:
            logger.error(f"更新群聊索引失败: {e}")

    async def _add_to_user_index(
        self, user_id: str, group_id: str, persona_id: str
    ) -> None:
        index_key = f"user_index:{persona_id}:{group_id}"
        try:
            async with self._index_lock:
                user_ids = await self._storage.get_kv_data(index_key, [])
                if user_id not in user_ids:
                    user_ids.append(user_id)
                    await self._storage.put_kv_data(index_key, user_ids)
                # 同时维护 user_group_index：记录有用户画像的 group_id，
                # 供 delete_all_user_profiles / list_all_users 遍历。
                # group_index 只记录群聊画像的 group_id，当隔离关闭时
                # 用户画像存于 "default" 而 group_index 不含 "default"，
                # 导致按 group_index 遍历会漏删用户画像。
                ug_index_key = f"user_group_index:{persona_id}"
                group_ids = await self._storage.get_kv_data(ug_index_key, [])
                if group_id not in group_ids:
                    group_ids.append(group_id)
                    await self._storage.put_kv_data(ug_index_key, group_ids)
        except Exception as e:
            logger.error(f"更新用户索引失败: {e}")

    async def _add_to_persona_index(self, persona_id: str) -> None:
        """记录出现过的 persona_id，供 delete_all 遍历所有命名空间。"""
        index_key = "persona_index"
        try:
            async with self._index_lock:
                personas = await self._storage.get_kv_data(index_key, [])
                if persona_id not in personas:
                    personas.append(persona_id)
                    await self._storage.put_kv_data(index_key, personas)
        except Exception as e:
            logger.error(f"更新 persona 索引失败: {e}")

    async def _get_known_personas(self) -> list:
        """获取所有已知 persona_id（始终包含 default 兜底）。"""
        personas = await self._storage.get_kv_data("persona_index", [])
        if "default" not in personas:
            personas = ["default", *personas]
        return personas

    async def delete_user_profile(
        self, user_id: str, group_id: str = "default", persona_id: str = "default"
    ) -> bool:
        """删除用户画像

        Args:
            user_id: 用户ID
            group_id: 群聊ID
            persona_id: 人格ID

        Returns:
            是否删除成功
        """
        if not self._is_available:
            return False

        persona_id = self._effective_persona(persona_id)
        key = f"user_profile:{persona_id}:{group_id}:{user_id}"

        try:
            await self._storage.delete_kv_data(key)
            logger.info(f"已删除用户画像: {key}")
            return True

        except Exception as e:
            logger.error(f"删除用户画像失败: {key}, 错误: {e}")
            return False

    async def delete_group_profile(
        self, group_id: str, persona_id: str = "default"
    ) -> bool:
        """删除群聊画像

        Args:
            group_id: 群聊ID
            persona_id: 人格ID

        Returns:
            是否删除成功
        """
        if not self._is_available:
            return False

        persona_id = self._effective_persona(persona_id)
        key = f"group_profile:{persona_id}:{group_id}"

        try:
            await self._storage.delete_kv_data(key)
            logger.info(f"已删除群聊画像: {key}")
            return True

        except Exception as e:
            logger.error(f"删除群聊画像失败: {key}, 错误: {e}")
            return False

    async def delete_all_user_profiles_in_group(self, group_id: str) -> int:
        """删除群聊内所有用户画像

        通过 persona_index / user_index 遍历，无需 KV 列表功能。

        Args:
            group_id: 群聊ID

        Returns:
            删除的画像数量
        """
        if not self._is_available:
            return 0

        deleted = 0
        try:
            for persona_id in await self._get_known_personas():
                user_ids = await self._storage.get_kv_data(
                    f"user_index:{persona_id}:{group_id}", []
                )
                for user_id in user_ids:
                    await self._storage.delete_kv_data(
                        f"user_profile:{persona_id}:{group_id}:{user_id}"
                    )
                    deleted += 1
                if user_ids:
                    await self._storage.delete_kv_data(
                        f"user_index:{persona_id}:{group_id}"
                    )
        except Exception as e:
            logger.error(f"删除群聊内用户画像失败: {e}", exc_info=True)

        logger.info(f"已删除群聊 {group_id} 内 {deleted} 个用户画像")
        return deleted

    async def delete_all_user_profiles(self) -> int:
        """删除所有用户画像

        通过 persona_index / user_group_index / user_index 遍历，无需 KV 列表功能。
        使用 user_group_index（而非 group_index）是因为群聊画像和用户画像的
        group_id 可能不一致：隔离关闭时群聊画像用真实 group_id，用户画像用 "default"。

        Returns:
            删除的画像数量
        """
        if not self._is_available:
            return 0

        deleted = 0
        try:
            for persona_id in await self._get_known_personas():
                ug_index_key = f"user_group_index:{persona_id}"
                group_ids = await self._storage.get_kv_data(ug_index_key, [])
                for group_id in group_ids:
                    user_ids = await self._storage.get_kv_data(
                        f"user_index:{persona_id}:{group_id}", []
                    )
                    for user_id in user_ids:
                        await self._storage.delete_kv_data(
                            f"user_profile:{persona_id}:{group_id}:{user_id}"
                        )
                        deleted += 1
                    if user_ids:
                        await self._storage.delete_kv_data(
                            f"user_index:{persona_id}:{group_id}"
                        )
                if group_ids:
                    await self._storage.delete_kv_data(ug_index_key)
        except Exception as e:
            logger.error(f"删除所有用户画像失败: {e}", exc_info=True)

        logger.info(f"已删除 {deleted} 个用户画像")
        return deleted

    async def delete_all_group_profiles(self) -> int:
        """删除所有群聊画像

        通过 persona_index / group_index 遍历，无需 KV 列表功能。

        Returns:
            删除的画像数量
        """
        if not self._is_available:
            return 0

        deleted = 0
        try:
            for persona_id in await self._get_known_personas():
                group_ids = await self._storage.get_kv_data(
                    f"group_index:{persona_id}", []
                )
                for group_id in group_ids:
                    await self._storage.delete_kv_data(
                        f"group_profile:{persona_id}:{group_id}"
                    )
                    deleted += 1
                if group_ids:
                    await self._storage.delete_kv_data(f"group_index:{persona_id}")
        except Exception as e:
            logger.error(f"删除所有群聊画像失败: {e}", exc_info=True)

        logger.info(f"已删除 {deleted} 个群聊画像")
        return deleted

    async def delete_all_profiles(self) -> dict:
        """删除所有画像（用户画像 + 群聊画像）

        Returns:
            删除统计 {"user_profiles": int, "group_profiles": int}
        """
        user_count = await self.delete_all_user_profiles()
        group_count = await self.delete_all_group_profiles()

        # 清理 persona_index（所有命名空间已清空，索引不再有意义）
        try:
            await self._storage.delete_kv_data("persona_index")
        except Exception as e:
            logger.warning(f"清理 persona_index 失败: {e}")

        return {"user_profiles": user_count, "group_profiles": group_count}

    async def export_all(self, persona_id: str = "default") -> dict:
        """导出所有画像数据

        Args:
            persona_id: 人格ID，导出该 persona 命名空间下的画像

        Returns:
            包含群聊画像和用户画像的字典
        """
        if not self._is_available:
            return {
                "version": "1.0",
                "export_time": "",
                "groups": [],
                "users": [],
                "stats": {"group_count": 0, "user_count": 0},
            }

        try:
            from datetime import datetime as _dt

            groups = await self.list_groups(persona_id)
            group_profiles = []
            for g in groups:
                profile = await self.get_group_profile(g["group_id"], persona_id)
                if profile:
                    group_profiles.append(profile_to_dict(profile))

            all_users = []
            for g in groups:
                users = await self.list_users(g["group_id"], persona_id)
                for u in users:
                    profile = await self.get_user_profile(
                        u["user_id"], g["group_id"], persona_id
                    )
                    if profile:
                        all_users.append(
                            {
                                **profile_to_dict(profile),
                                "_group_id": g["group_id"],
                            }
                        )

            users_without_group = await self.list_users("default", persona_id)
            for u in users_without_group:
                profile = await self.get_user_profile(
                    u["user_id"], "default", persona_id
                )
                if profile:
                    already = any(
                        p.get("user_id") == u["user_id"]
                        and p.get("_group_id") == "default"
                        for p in all_users
                    )
                    if not already:
                        all_users.append(
                            {
                                **profile_to_dict(profile),
                                "_group_id": "default",
                            }
                        )

            export_time = _dt.now().isoformat()

            logger.info(
                f"画像导出完成：{len(group_profiles)} 个群聊，{len(all_users)} 个用户"
            )

            return {
                "version": "1.0",
                "export_time": export_time,
                "groups": group_profiles,
                "users": all_users,
                "stats": {
                    "group_count": len(group_profiles),
                    "user_count": len(all_users),
                },
            }

        except Exception as e:
            logger.error(f"导出画像失败：{e}", exc_info=True)
            return {
                "version": "1.0",
                "export_time": "",
                "groups": [],
                "users": [],
                "stats": {"group_count": 0, "user_count": 0},
            }

    async def import_from_data(
        self, data: dict, skip_duplicates: bool = True, persona_id: str = "default"
    ) -> dict:
        """从数据字典导入画像

        Args:
            data: 导出数据字典（包含 groups 和 users）
            skip_duplicates: 是否跳过已有画像（否则覆盖更新）
            persona_id: 人格ID，导入到该 persona 命名空间

        Returns:
            导入统计 {"imported_groups": int, "imported_users": int, "skipped": int, "error_count": int}
        """
        if not self._is_available:
            return {
                "imported_groups": 0,
                "imported_users": 0,
                "skipped": 0,
                "error_count": 0,
            }

        groups_data = data.get("groups", [])
        users_data = data.get("users", [])

        imported_groups = 0
        imported_users = 0
        skipped = 0
        error_count = 0

        for group_data in groups_data:
            try:
                group_id = group_data.get("group_id")
                if not group_id:
                    skipped += 1
                    continue

                if skip_duplicates:
                    existing = await self.get_group_profile(group_id, persona_id)
                    if existing:
                        skipped += 1
                        continue

                profile = dict_to_group_profile(group_data)
                await self.save_group_profile(
                    profile, increment_version=False, persona_id=persona_id
                )
                imported_groups += 1

            except Exception as e:
                logger.error(f"导入群聊画像失败：{e}")
                error_count += 1

        for user_data in users_data:
            try:
                user_id = user_data.get("user_id")
                group_id = user_data.pop("_group_id", "default")

                if not user_id:
                    skipped += 1
                    continue

                if skip_duplicates:
                    existing = await self.get_user_profile(
                        user_id, group_id, persona_id
                    )
                    if existing:
                        skipped += 1
                        continue

                profile = dict_to_user_profile(user_data)
                await self.save_user_profile(
                    profile,
                    group_id=group_id,
                    increment_version=False,
                    persona_id=persona_id,
                )
                imported_users += 1

            except Exception as e:
                logger.error(f"导入用户画像失败：{e}")
                error_count += 1

        logger.info(
            f"画像导入完成：群聊 {imported_groups}/{len(groups_data)}，"
            f"用户 {imported_users}/{len(users_data)}，"
            f"跳过 {skipped}，错误 {error_count}"
        )

        return {
            "imported_groups": imported_groups,
            "imported_users": imported_users,
            "skipped": skipped,
            "error_count": error_count,
        }

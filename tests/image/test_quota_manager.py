"""图片解析配额管理器测试"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import date

from iris_memory.image import ImageQuotaManager


class TestImageQuotaManager:
    """ImageQuotaManager 测试"""

    @pytest.fixture
    def mock_context(self):
        """模拟 AstrBot Context"""
        context = Mock()
        context.get_kv_data = AsyncMock(return_value={})
        context.put_kv_data = AsyncMock()
        return context

    @pytest.fixture
    def quota_manager(self, mock_context):
        """创建配额管理器实例"""
        return ImageQuotaManager(mock_context)

    @pytest.mark.asyncio
    async def test_initialize_disabled(self, mock_context):
        """测试禁用状态初始化"""
        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key: {"l1_buffer.image_parsing.enable": False}.get(
                    key
                )
            )
            mock_get_config.return_value = mock_config

            manager = ImageQuotaManager(mock_context)
            await manager.initialize()

            assert not manager.is_available

    @pytest.mark.asyncio
    async def test_initialize_success(self, quota_manager, mock_context):
        """测试成功初始化"""
        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            await quota_manager.initialize()

            assert quota_manager.is_available
            mock_context.put_kv_data.assert_called()

    @pytest.mark.asyncio
    async def test_load_quota_from_storage(self, mock_context):
        """测试从存储加载配额"""
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        stored_data = {
            "date": today,
            "used": 50,
            "total": 200,
            "last_reset_time": f"{today}T00:00:00",
        }
        mock_context.get_kv_data = AsyncMock(return_value=stored_data)

        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            manager = ImageQuotaManager(mock_context)
            await manager.initialize()

            status = await manager.get_status()
            assert status is not None
            assert status.used == 50
            assert status.total == 200

    @pytest.mark.asyncio
    async def test_check_quota_success(self, quota_manager, mock_context):
        """测试检查配额（充足）"""
        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            await quota_manager.initialize()

            has_quota = await quota_manager.check_quota()
            assert has_quota

    @pytest.mark.asyncio
    async def test_check_quota_exhausted(self, quota_manager, mock_context):
        """测试检查配额（耗尽）"""
        stored_data = {
            "date": date.today().isoformat(),
            "used": 200,
            "total": 200,
            "last_reset_time": "2026-03-29T00:00:00",
        }
        mock_context.get_kv_data = AsyncMock(return_value=stored_data)

        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            await quota_manager.initialize()

            has_quota = await quota_manager.check_quota()
            assert not has_quota

    @pytest.mark.asyncio
    async def test_use_quota_success(self, quota_manager, mock_context):
        """测试使用配额成功"""
        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            await quota_manager.initialize()

            success = await quota_manager.use_quota(5)
            assert success

            status = await quota_manager.get_status()
            assert status is not None
            assert status.used == 5

    @pytest.mark.asyncio
    async def test_use_quota_exhausted(self, quota_manager, mock_context):
        """测试使用配额失败（配额不足）"""
        stored_data = {
            "date": date.today().isoformat(),
            "used": 195,
            "total": 200,
            "last_reset_time": "2026-03-29T00:00:00",
        }
        mock_context.get_kv_data = AsyncMock(return_value=stored_data)

        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            await quota_manager.initialize()

            # 尝试使用 10 个配额（只有 5 个剩余）
            success = await quota_manager.use_quota(10)
            assert not success

    @pytest.mark.asyncio
    async def test_reset_quota(self, quota_manager, mock_context):
        """测试重置配额"""
        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            await quota_manager.initialize()

            # 使用一些配额
            await quota_manager.use_quota(50)

            # 重置
            await quota_manager.reset_quota()

            status = await quota_manager.get_status()
            assert status is not None
            assert status.used == 0
            assert status.date == date.today().isoformat()

    @pytest.mark.asyncio
    async def test_auto_reset_on_date_change(self, mock_context):
        """测试跨天自动重置"""
        # 存储昨天的数据
        yesterday = "2026-03-28"
        stored_data = {
            "date": yesterday,
            "used": 100,
            "total": 200,
            "last_reset_time": "2026-03-28T00:00:00",
        }
        mock_context.get_kv_data = AsyncMock(return_value=stored_data)

        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            with patch("iris_memory.image.quota_manager.date") as mock_date:
                # 模拟今天是 2026-03-29
                mock_date.today.return_value = date(2026, 3, 29)

                manager = ImageQuotaManager(mock_context)
                await manager.initialize()

                # 检查配额时应该自动重置
                await manager.check_quota()

                status = await manager.get_status()
                assert status is not None
                assert status.used == 0  # 已重置
                assert status.date == "2026-03-29"  # 更新为今天

    @pytest.mark.asyncio
    async def test_shutdown(self, quota_manager, mock_context):
        """测试关闭"""
        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            await quota_manager.initialize()
            await quota_manager.shutdown()

            assert not quota_manager.is_available

    @pytest.mark.asyncio
    async def test_cross_day_reset_no_deadlock(self, mock_context):
        """回归：跨天重置在持锁中不得重入 asyncio.Lock 导致死锁

        历史 bug：check_quota/use_quota/get_status 持 _lock 后调
        _check_and_reset_if_needed，跨天时内部 await reset_quota() 又
        async with self._lock。asyncio.Lock 不可重入，真实时钟跨过午夜后
        必然挂死。现有测试在 init 前固定 today 绕过了该路径。
        """
        import asyncio

        # 存储昨天的数据
        yesterday = "2026-03-28"
        stored_data = {
            "date": yesterday,
            "used": 100,
            "total": 200,
            "last_reset_time": "2026-03-28T00:00:00",
        }
        mock_context.get_kv_data = AsyncMock(return_value=stored_data)

        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            with patch("iris_memory.image.quota_manager.date") as mock_date:
                # init 时是 3-28（不触发重置）
                mock_date.today.return_value = date(2026, 3, 28)
                manager = ImageQuotaManager(mock_context)
                await manager.initialize()

                # 模拟跨天到 3-29
                mock_date.today.return_value = date(2026, 3, 29)

                # check_quota 持 _lock → _check_and_reset_if_needed →
                # 跨天 → _reset_quota_locked（不加锁，不重入）
                # 修复前会调 reset_quota() 重入 _lock → 死锁
                try:
                    result = await asyncio.wait_for(manager.check_quota(), timeout=2.0)
                    # 不死锁即通过
                    assert result is True
                except asyncio.TimeoutError:
                    pytest.fail("跨天重置死锁：check_quota 在持锁中重入 _lock")

                # 验证配额已重置
                status = manager._quota_status
                assert status.date == "2026-03-29"
                assert status.used == 0

    @pytest.mark.asyncio
    async def test_cross_day_use_quota_no_deadlock(self, mock_context):
        """use_quota 持锁跨天同样不得死锁"""
        import asyncio

        yesterday = "2026-03-28"
        stored_data = {
            "date": yesterday,
            "used": 50,
            "total": 200,
            "last_reset_time": "2026-03-28T00:00:00",
        }
        mock_context.get_kv_data = AsyncMock(return_value=stored_data)

        with patch("iris_memory.image.quota_manager.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get = Mock(
                side_effect=lambda key, default=None: {
                    "l1_buffer.image_parsing.enable": True,
                    "l1_buffer.image_parsing.daily_quota": 200,
                }.get(key, default)
            )
            mock_get_config.return_value = mock_config

            with patch("iris_memory.image.quota_manager.date") as mock_date:
                mock_date.today.return_value = date(2026, 3, 28)
                manager = ImageQuotaManager(mock_context)
                await manager.initialize()

                # 跨天
                mock_date.today.return_value = date(2026, 3, 29)

                try:
                    success = await asyncio.wait_for(manager.use_quota(5), timeout=2.0)
                    assert success
                except asyncio.TimeoutError:
                    pytest.fail("跨天重置死锁：use_quota 在持锁中重入 _lock")

                status = manager._quota_status
                assert status.date == "2026-03-29"
                assert status.used == 5

"""人格自迭代配置接线测试：默认值、schema 合法性、组件注册与消息旁路"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from iris_memory.config.defaults import Defaults

SCHEMA_PATH = Path(__file__).parent.parent.parent / "_conf_schema.json"


class TestSchemaJson:
    """_conf_schema.json 合法且 persona_evolution 组齐备"""

    def test_schema_is_valid_json(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert "persona_evolution" in schema

    def test_persona_evolution_group(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        group = schema["persona_evolution"]
        items = group["items"]
        assert set(items) == {
            "enable",
            "provider",
            "review_provider",
            "sample_retention_days",
            "sample_max_count",
        }
        assert items["enable"]["default"] is False
        assert items["provider"]["default"] == ""
        assert items["provider"]["_special"] == "select_provider"
        assert items["review_provider"]["_special"] == "select_provider"
        assert items["sample_retention_days"]["default"] == 30
        assert items["sample_max_count"]["default"] == 20000


class TestDefaults:
    """defaults.py 默认值读取"""

    def test_section_defaults(self):
        defaults = Defaults()
        pe = defaults.persona_evolution
        assert pe.enable is False
        assert pe.provider == ""
        assert pe.review_provider == ""
        assert pe.sample_retention_days == 30
        assert pe.sample_max_count == 20000
        # 扁平键访问
        assert defaults.get_by_flat_key("persona_evolution.enable") is False
        assert defaults.get_by_flat_key("persona_evolution.sample_max_count") == 20000

    def test_job_defaults_in_hidden_config(self):
        """文档 §15.1 的 Job 默认值（隐藏配置）"""
        hidden = Defaults().hidden
        assert hidden.persona_evolution_edit_mode == "managed_block"
        assert hidden.persona_evolution_approval_mode == "auto"
        assert hidden.persona_evolution_trigger_sample_count == 100
        assert hidden.persona_evolution_min_interval_hours == 24
        assert hidden.persona_evolution_manual_min_samples == 20
        assert hidden.persona_evolution_analysis_sample_size == 60
        assert hidden.persona_evolution_full_prompt_max_change_ratio == 0.20

    def test_advanced_params_in_hidden_config(self):
        """文档 §15.2 的隐藏高级参数"""
        hidden = Defaults().hidden
        assert hidden.persona_evolution_sample_store_chars == 500
        assert hidden.persona_evolution_sample_prompt_chars == 240
        assert hidden.persona_evolution_sample_group_max_ratio == 0.35
        assert hidden.persona_evolution_sample_user_max_ratio == 0.20
        assert hidden.persona_evolution_min_confidence == 0.65
        assert hidden.persona_evolution_block_max_chars == 1500
        assert hidden.persona_evolution_full_max_growth_ratio == 1.25
        assert hidden.persona_evolution_full_max_length > 0
        assert hidden.persona_evolution_max_reuse_chars == 16
        assert hidden.persona_evolution_retry_intervals_minutes == "30,120,360"
        assert hidden.persona_evolution_circuit_breaker_threshold == 3

    def test_hidden_params_via_config_get(self, disabled_config):
        """经 Config.get 按扁平键读取隐藏参数（走隐藏配置层）"""
        assert disabled_config.get("persona_evolution_trigger_sample_count") == 100
        assert disabled_config.get("persona_evolution_sample_store_chars") == 500
        assert disabled_config.get("persona_evolution.enable") is False


class TestLifecycleGuard:
    """create_components 按 persona_evolution.enable 守卫"""

    def test_component_registered_when_enabled(self, config):
        from iris_memory.core.lifecycle import create_components

        components = create_components(MagicMock(), MagicMock())
        names = [c.name for c in components]
        assert "persona_evolution" in names

    def test_component_not_registered_when_disabled(self, disabled_config):
        from iris_memory.core.lifecycle import create_components

        components = create_components(MagicMock(), MagicMock())
        names = [c.name for c in components]
        assert "persona_evolution" not in names


class TestMessageHookBypass:
    """message_hook 的 persona_evolution 采集旁路"""

    def _manager_with(self, component):
        manager = MagicMock()
        manager.get_available_component.side_effect = (
            lambda name: component if name == "persona_evolution" else None
        )
        return manager

    @pytest.mark.asyncio
    async def test_bypass_calls_on_message(self, config):
        from iris_memory.core.message_hook import handle_user_message

        component = MagicMock()
        component.on_message = AsyncMock()
        event = MagicMock()
        event.message_str = ""  # 空消息让 L1 路径提前返回，聚焦旁路
        await handle_user_message(event, self._manager_with(component))
        component.on_message.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_bypass_skips_unavailable(self, config):
        """组件未注册/未就绪时不报错"""
        from iris_memory.core.message_hook import handle_user_message

        event = MagicMock()
        event.message_str = ""
        await handle_user_message(event, self._manager_with(None))

    @pytest.mark.asyncio
    async def test_bypass_isolates_exception(self, config):
        """采集异常被隔离，不影响主流程"""
        from iris_memory.core.message_hook import handle_user_message

        component = MagicMock()
        component.on_message = AsyncMock(side_effect=RuntimeError("采集炸了"))
        event = MagicMock()
        event.message_str = ""
        await handle_user_message(event, self._manager_with(component))  # 不抛出

"""Web 学习路由辅助函数测试"""

import pytest

from iris_memory.web.routes.learning import (
    MAX_PAGE_SIZE,
    parse_pagination,
    validate_table,
    validate_table_status,
)


class TestValidateTable:
    """表名白名单校验"""

    @pytest.mark.parametrize("table", ["jargon", "expression_pattern", "few_shot"])
    def test_accepts_whitelisted_tables(self, table):
        assert validate_table(table) == table

    @pytest.mark.parametrize("table", [None, "", "users", "sqlite_master", "jargon; DROP TABLE jargon"])
    def test_rejects_invalid_tables(self, table):
        with pytest.raises(ValueError):
            validate_table(table)


class TestParsePagination:
    """分页参数解析"""

    def test_defaults(self):
        assert parse_pagination({}) == (1, 20)

    def test_normal_values(self):
        assert parse_pagination({"page": "3", "page_size": "50"}) == (3, 50)

    def test_clamps_page_size(self):
        assert parse_pagination({"page_size": str(MAX_PAGE_SIZE + 1)})[1] == MAX_PAGE_SIZE
        assert parse_pagination({"page_size": "0"})[1] == 1

    def test_clamps_page(self):
        assert parse_pagination({"page": "0"})[0] == 1
        assert parse_pagination({"page": "-5"})[0] == 1

    def test_invalid_values_fall_back(self):
        assert parse_pagination({"page": "abc", "page_size": None}) == (1, 20)


class TestValidateTableStatus:
    """目标状态按表校验（暗语无审查语义）"""

    @pytest.mark.parametrize(
        "table,status",
        [
            ("jargon", "active"),
            ("jargon", "disabled"),
            ("expression_pattern", "pending_review"),
            ("expression_pattern", "approved"),
            ("expression_pattern", "disabled"),
            ("few_shot", "pending_review"),
            ("few_shot", "approved"),
            ("few_shot", "disabled"),
        ],
    )
    def test_accepts_valid_status(self, table, status):
        assert validate_table_status(table, status) == status

    @pytest.mark.parametrize(
        "table,status",
        [
            ("jargon", "approved"),        # 暗语不可设为已通过
            ("jargon", "pending_review"),  # 暗语不可设为待审查
            ("few_shot", "active"),        # 样例不可设为生效中
            ("expression_pattern", "active"),
            ("jargon", None),
            ("jargon", ""),
            ("few_shot", "bogus"),
        ],
    )
    def test_rejects_invalid_status(self, table, status):
        with pytest.raises(ValueError):
            validate_table_status(table, status)


class TestAugmentDisabledComponents:
    """配置禁用的可选组件在系统统计中补报为 disabled"""

    @pytest.fixture
    def config(self, tmp_path):
        from iris_memory.config import init_config
        from iris_memory.config.config import reset_config

        cfg = init_config({}, tmp_path)
        yield cfg
        reset_config()

    def test_adds_disabled_entry_for_unregistered(self, config):
        from iris_memory.web.routes.stats import _augment_disabled_components

        states = _augment_disabled_components({})
        # learning.enable 默认 false 且组件未注册 → 补报 disabled
        assert states["learning"]["status"] == "unavailable"
        assert states["learning"]["error_type"] == "disabled"

    def test_keeps_registered_state(self, config):
        from iris_memory.web.routes.stats import _augment_disabled_components

        existing = {"learning": {"status": "available", "error": None, "error_type": None}}
        states = _augment_disabled_components(existing)
        assert states["learning"]["status"] == "available"

    def test_enabled_component_not_augmented(self, tmp_path):
        from iris_memory.config import init_config
        from iris_memory.config.config import reset_config
        from iris_memory.web.routes.stats import _augment_disabled_components

        init_config({"learning": {"enable": True}}, tmp_path)
        try:
            states = _augment_disabled_components({})
            assert "learning" not in states
        finally:
            reset_config()

"""BoundedDict 单元测试"""

import pytest
from iris_memory.utils.bounded_dict import BoundedDict


class TestBoundedDict:
    """BoundedDict LRU 有界字典测试"""

    def test_basic_operations(self):
        """基本读写操作"""
        d = BoundedDict[str, int](max_size=5)
        d["a"] = 1
        d["b"] = 2
        assert d["a"] == 1
        assert d["b"] == 2
        assert len(d) == 2

    def test_eviction_on_overflow(self):
        """超出容量时驱逐最旧条目"""
        d = BoundedDict[str, int](max_size=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        d["d"] = 4  # 应驱逐 "a"
        assert "a" not in d
        assert len(d) == 3
        assert list(d.keys()) == ["b", "c", "d"]

    def test_lru_access_order(self):
        """LRU: 访问过的条目不会被优先驱逐"""
        d = BoundedDict[str, int](max_size=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        _ = d["a"]  # 访问 "a"，使其成为最近使用
        d["d"] = 4  # 应驱逐 "b"（最旧未访问）
        assert "a" in d
        assert "b" not in d
        assert len(d) == 3

    def test_update_moves_to_end(self):
        """更新已有条目不增加大小，且移到末尾"""
        d = BoundedDict[str, int](max_size=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        d["a"] = 10  # 更新 "a"
        d["d"] = 4   # 应驱逐 "b"
        assert d["a"] == 10
        assert "b" not in d
        assert len(d) == 3

    def test_max_size_one(self):
        """最小容量=1"""
        d = BoundedDict[str, int](max_size=1)
        d["a"] = 1
        d["b"] = 2
        assert "a" not in d
        assert d["b"] == 2
        assert len(d) == 1

    def test_invalid_max_size(self):
        """非法容量应被钳住为 1"""
        d = BoundedDict[str, int](max_size=0)
        assert d.max_size == 1
        d2 = BoundedDict[str, int](max_size=-1)
        assert d2.max_size == 1

    def test_get_nonexistent_key(self):
        """访问不存在的键抛出 KeyError"""
        d = BoundedDict[str, int](max_size=5)
        with pytest.raises(KeyError):
            _ = d["missing"]

    def test_delete_key(self):
        """删除操作"""
        d = BoundedDict[str, int](max_size=5)
        d["a"] = 1
        del d["a"]
        assert "a" not in d
        assert len(d) == 0

    def test_iteration(self):
        """迭代顺序正确"""
        d = BoundedDict[str, int](max_size=5)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        assert list(d.items()) == [("a", 1), ("b", 2), ("c", 3)]

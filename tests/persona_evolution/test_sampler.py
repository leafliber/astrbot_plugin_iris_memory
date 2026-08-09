"""人格自迭代均衡抽样器测试（文档 §7.4）"""

from iris_memory.persona_evolution.sampler import stratified_sample


def _samples(group_user_counts: dict[tuple[str, str], int], start_id: int = 1):
    """按 (群, 用户) -> 条数 生成候选语料，created_at 随 id 递增"""
    samples = []
    sid = start_id
    for (group_id, user_id), count in group_user_counts.items():
        for i in range(count):
            samples.append(
                {
                    "id": sid,
                    "group_id": group_id,
                    "user_id": user_id,
                    "created_at": 1000000.0 + sid * 60,
                    "normalized_text": f"样本{sid}",
                }
            )
            sid += 1
    return samples


class TestRatioCaps:
    """多群单群 ≤35%、多用户单用户 ≤20%"""

    def test_group_and_user_caps(self):
        # 3 群 × 5 用户 × 15 条 = 225 条，抽 60
        samples = _samples(
            {(f"g{g}", f"u{u}"): 15 for g in range(3) for u in range(5)}
        )
        selected = stratified_sample(samples, max_count=60)
        assert len(selected) == 60

        group_counts: dict[str, int] = {}
        user_counts: dict[str, int] = {}
        for s in selected:
            group_counts[s["group_id"]] = group_counts.get(s["group_id"], 0) + 1
            user_counts[s["user_id"]] = user_counts.get(s["user_id"], 0) + 1
        # 单群不超过 35%（60*0.35=21），单用户不超过 20%（60*0.2=12）
        assert all(c <= 21 for c in group_counts.values())
        assert all(c <= 12 for c in user_counts.values())
        # 三群均被覆盖（不按消息量加权）
        assert len(group_counts) == 3

    def test_hot_group_not_dominating(self):
        """高活跃群贡献远超其他群时，占比仍受限（上限对所有群生效）"""
        samples = _samples({("hot", f"u{i}"): 40 for i in range(5)})
        samples += _samples({("cold", f"v{i}"): 5 for i in range(5)}, start_id=1000)
        selected = stratified_sample(samples, max_count=60)
        hot = sum(1 for s in selected if s["group_id"] == "hot")
        cold = sum(1 for s in selected if s["group_id"] == "cold")
        # 两个群都受 35%（21 条）上限约束；冷群语料不足时不会用热群补满
        assert hot == 21
        assert cold == 21
        assert len(selected) == 42


class TestScopeDegeneration:
    """单群/单用户场景取消相应比例限制"""

    def test_single_group_pool_no_group_cap(self):
        # 语料池只有一个群：群占比限制自动失效
        samples = _samples({("g1", f"u{i}"): 15 for i in range(10)})
        selected = stratified_sample(samples, max_count=60)
        assert len(selected) == 60
        assert all(s["group_id"] == "g1" for s in selected)

    def test_single_user_scope_cancels_user_cap(self):
        # Job 指定单用户：用户占比限制取消
        samples = _samples({("g1", "u1"): 100})
        selected = stratified_sample(
            samples, max_count=60, single_user_scope=True
        )
        assert len(selected) == 60
        assert all(s["user_id"] == "u1" for s in selected)

    def test_single_group_scope_cancels_group_cap(self):
        # Job 指定单群：即使传入多群语料也不受 35% 限制
        samples = _samples({("g1", f"u{i}"): 20 for i in range(10)})
        samples += _samples({("g2", "u99"): 20}, start_id=1000)
        selected = stratified_sample(
            samples, max_count=60, single_group_scope=True
        )
        g1 = sum(1 for s in selected if s["group_id"] == "g1")
        assert g1 > 21  # 不受 35% 群上限约束


class TestTimeCoverage:
    """覆盖不同时间段，不只学最近一次突发话题"""

    def test_even_time_coverage(self):
        samples = _samples({("g1", "u1"): 200})
        selected = stratified_sample(samples, max_count=20)
        assert len(selected) == 20
        times = sorted(s["created_at"] for s in selected)
        total_span = samples[-1]["created_at"] - samples[0]["created_at"]
        # 最早与最晚抽样的跨度覆盖语料整体时间跨度的绝大部分
        assert times[-1] - times[0] > total_span * 0.8
        # 不是简单地取最近 20 条
        latest_20_ids = {s["id"] for s in samples[-20:]}
        assert {s["id"] for s in selected} != latest_20_ids


class TestEdgeCases:
    def test_fewer_than_max_returns_all(self):
        samples = _samples({("g1", "u1"): 10})
        selected = stratified_sample(samples, max_count=60)
        assert len(selected) == 10

    def test_empty_and_zero(self):
        assert stratified_sample([], max_count=60) == []
        samples = _samples({("g1", "u1"): 10})
        assert stratified_sample(samples, max_count=0) == []

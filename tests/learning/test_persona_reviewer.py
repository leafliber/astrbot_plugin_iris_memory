"""Persona 修改检测与学习内容一致性复审测试。"""

import json
import hashlib
import time
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.learning import LearningComponent
from iris_memory.learning.persona_reviewer import PersonaLearningReviewer
from iris_memory.learning.storage import LEGACY_PERSONA_ID, LearningStorage


class TestPersonaReviewerParsing:
    @pytest.mark.asyncio
    async def test_requires_complete_batch(self):
        reviewer = PersonaLearningReviewer()
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value='[{"id":1,"type":"pair","compatible":true}]'
        )
        result = await reviewer.request_verdicts(
            llm,
            "你是温柔的助手",
            [{"id": 1, "user_text": "你好", "bot_text": "你好呀"}],
            [{"id": 2, "scene": "chat", "expression": "你好呀"}],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_parses_complete_batch(self):
        reviewer = PersonaLearningReviewer()
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value=json.dumps(
                [
                    {"id": 1, "type": "pair", "compatible": True},
                    {"id": 2, "type": "pattern", "compatible": False},
                ]
            )
        )
        result = await reviewer.request_verdicts(
            llm,
            "你是温柔的助手",
            [{"id": 1, "user_text": "你好", "bot_text": "你好呀"}],
            [{"id": 2, "scene": "chat", "expression": "滚开"}],
        )
        assert result == {("pair", 1): True, ("pattern", 2): False}


class TestPersonaStorageMigration:
    def test_failed_review_is_reclaimed_on_next_observation(self, tmp_path):
        storage = LearningStorage(tmp_path / "retry.db")
        storage.init_schema()
        try:
            assert storage.observe_persona_prompt("p1", "old") == "baseline"
            assert storage.observe_persona_prompt("p1", "new") == "changed"
            storage.fail_persona_review("p1", "new", "temporary error")
            assert storage.observe_persona_prompt("p1", "new") == "changed"
        finally:
            storage.close()

    def test_v2_rows_become_legacy_and_wait_for_review(self, tmp_path):
        db_path = tmp_path / "learning-v2.db"
        db = sqlite3.connect(db_path)
        db.executescript(
            """
            CREATE TABLE few_shot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL DEFAULT '', user_id TEXT NOT NULL DEFAULT '',
                user_text TEXT NOT NULL, bot_text TEXT NOT NULL, message_id TEXT,
                status TEXT DEFAULT 'pending_review', created_at REAL
            );
            CREATE TABLE expression_pattern (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL DEFAULT '', scene TEXT NOT NULL DEFAULT '',
                expression TEXT NOT NULL, source_pair_id INTEGER,
                hit_count INTEGER DEFAULT 0, status TEXT DEFAULT 'pending_review',
                created_at REAL, last_hit_at REAL
            );
            INSERT INTO few_shot(group_id,user_id,user_text,bot_text,status,created_at)
            VALUES('g1','u1','你好','旧回复','approved',1);
            INSERT INTO expression_pattern(group_id,scene,expression,status,created_at)
            VALUES('g1','chat','旧表达','approved',1);
            """
        )
        db.close()

        storage = LearningStorage(db_path)
        storage.init_schema()
        try:
            assert storage.list_rows("few_shot")[0]["persona_id"] == LEGACY_PERSONA_ID
            pairs, patterns = storage.get_persona_review_items("p1")
            assert len(pairs) == 1
            assert len(patterns) == 1
        finally:
            storage.close()


class TestPersonaRevalidationFlow:
    @staticmethod
    def _event():
        event = MagicMock()
        event.message_str = "你好"
        return event

    @staticmethod
    def _adapter():
        adapter = MagicMock()
        adapter.get_session_id.return_value = "s1"
        adapter.get_group_id.return_value = "g1"
        return adapter

    @pytest.mark.asyncio
    async def test_changed_persona_deletes_incompatible_items(self, config):
        comp = LearningComponent()
        await comp.initialize()
        event = self._event()
        try:
            # 首次观察时尚无学习数据，只建立人格基线。
            with patch(
                "iris_memory.learning.injector.get_adapter",
                return_value=self._adapter(),
            ):
                assert await comp.build_context(
                    event, {}, persona_id="p1", persona_prompt="旧人格"
                ) == ""

            pair_id = comp.storage.insert_pair(
                "g1", "u1", "你好", "旧人格回复", persona_id="p1"
            )
            pattern_id = comp.storage.insert_pattern(
                "g1", "chat", "旧人格口癖", persona_id="p1"
            )
            comp.storage.update_status("few_shot", [pair_id], "approved")
            comp.storage.update_status(
                "expression_pattern", [pattern_id], "approved"
            )

            llm = MagicMock()
            llm.generate_direct = AsyncMock(
                return_value=json.dumps(
                    [
                        {
                            "id": pair_id,
                            "type": "pair",
                            "compatible": False,
                            "reason": "与新人格冲突",
                        },
                        {
                            "id": pattern_id,
                            "type": "pattern",
                            "compatible": False,
                            "reason": "与新人格冲突",
                        },
                    ]
                )
            )
            with (
                patch.object(comp, "_get_llm_manager", return_value=llm),
                patch(
                    "iris_memory.learning.injector.get_adapter",
                    return_value=self._adapter(),
                ),
            ):
                meta = {}
                text = await comp.build_context(
                    event, meta, persona_id="p1", persona_prompt="全新人格"
                )
                assert text == ""
                assert meta["skipped"] == "persona_revalidation"
                task = comp._persona_review_tasks["p1"]
                await task

            assert comp.storage.count_rows("few_shot") == 0
            assert comp.storage.count_rows("expression_pattern") == 0
            assert llm.generate_direct.await_count == 1
        finally:
            await comp.shutdown()

    @pytest.mark.asyncio
    async def test_injection_is_scoped_to_current_persona(self, config):
        comp = LearningComponent()
        await comp.initialize()
        try:
            p1 = comp.storage.insert_pattern(
                "g1", "chat", "人格一表达", persona_id="p1"
            )
            p2 = comp.storage.insert_pattern(
                "g1", "chat", "人格二表达", persona_id="p2"
            )
            comp.storage.update_status("expression_pattern", [p1, p2], "approved")
            with patch(
                "iris_memory.learning.injector.get_adapter",
                return_value=self._adapter(),
            ):
                text = await comp.build_context(
                    self._event(), {}, persona_id="p1", persona_prompt=""
                )
            assert "人格一表达" in text
            assert "人格二表达" not in text
        finally:
            await comp.shutdown()

    @pytest.mark.asyncio
    async def test_large_review_is_sliced_and_resumes_from_persistent_cursor(
        self, config
    ):
        comp = LearningComponent()
        await comp.initialize()
        try:
            assert comp.storage.observe_persona_prompt("p1", "old") == "baseline"
            for index in range(60):
                row_id = comp.storage.insert_pair(
                    "g1",
                    f"u{index}",
                    f"问题 {index}",
                    f"回复 {index}",
                    persona_id="p1",
                )
                comp.storage.update_status("few_shot", [row_id], "approved")

            prompt = "全新人格"
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            assert comp.storage.observe_persona_prompt("p1", prompt_hash) == "changed"

            async def accept_all(_llm, _prompt, pairs, patterns):
                return {
                    **{("pair", int(row["id"])): True for row in pairs},
                    **{("pattern", int(row["id"])): True for row in patterns},
                }

            comp._persona_reviewer.request_verdicts = AsyncMock(
                side_effect=accept_all
            )
            with patch.object(comp, "_get_llm_manager", return_value=MagicMock()):
                comp._schedule_persona_review("p1", prompt, prompt_hash)
                await comp._persona_review_tasks["p1"]

                state = comp.storage.get_persona_review_state("p1")
                assert state["status"] == "reviewing"
                assert state["processed"] == 50
                assert state["next_run_at"] > time.time()
                assert comp._persona_reviewer.request_verdicts.await_count == 5

                comp.storage.advance_persona_review(
                    "p1",
                    prompt_hash,
                    pair_cursor=state["pair_cursor"],
                    pattern_cursor=state["pattern_cursor"],
                    processed=state["processed"],
                    next_run_at=time.time() - 1,
                )
                comp._schedule_persona_review("p1", prompt, prompt_hash)
                await comp._persona_review_tasks["p1"]

            state = comp.storage.get_persona_review_state("p1")
            assert state["status"] == "idle"
            assert comp._persona_reviewer.request_verdicts.await_count == 6
        finally:
            await comp.shutdown()

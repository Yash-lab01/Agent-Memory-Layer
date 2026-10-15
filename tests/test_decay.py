"""Unit and integration tests for MemoryDecayEngine, frequency reinforcement, and explicit forget API."""

from __future__ import annotations

import json
import math
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio

from memlayer.api import MemoryLayer
from memlayer.consolidate.decay import (
    MemoryDecayEngine,
    calculate_decayed_importance,
)
from memlayer.extract.extractor import FactExtractor
from memlayer.llm.embedder import MockEmbedder
from memlayer.models import MemoryItem, MemoryStatus
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore
from memlayer.store.vector_base import VectorPoint


@pytest_asyncio.fixture
async def sql_store() -> AsyncGenerator[SQLStore, None]:
    """In-memory SQLite store fixture."""
    store = SQLStore(db_url="sqlite+aiosqlite:///:memory:")
    await store.init_db()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def vector_store() -> AsyncGenerator[QdrantVectorStore, None]:
    """In-memory QdrantVectorStore fixture."""
    store = QdrantVectorStore(location=":memory:", vector_size=32)
    await store.init_store()
    yield store
    await store.close()


@pytest.fixture
def embedder() -> MockEmbedder:
    """MockEmbedder fixture."""
    return MockEmbedder(dimension=32)


class TestDecayFormula:
    """Unit tests for the mathematical calculate_decayed_importance function."""

    def test_zero_elapsed_time_no_accesses(self) -> None:
        """When memory was just created, importance is unchanged."""
        now = datetime.now(UTC)
        new_imp, elapsed_days = calculate_decayed_importance(
            importance=0.8,
            created_at=now,
            last_accessed_at=None,
            access_count=0,
            now=now,
            tau_days=30.0,
        )
        assert elapsed_days == 0.0
        assert math.isclose(new_imp, 0.8, abs_tol=1e-4)

    def test_one_half_life_decay(self) -> None:
        """After exactly 1 half-life (tau), unaccessed memory decays by 1/e (~0.3679)."""
        created = datetime.now(UTC) - timedelta(days=30)
        now = datetime.now(UTC)
        new_imp, elapsed_days = calculate_decayed_importance(
            importance=0.8,
            created_at=created,
            last_accessed_at=None,
            access_count=0,
            now=now,
            tau_days=30.0,
        )
        expected = 0.8 * math.exp(-1.0)
        assert math.isclose(elapsed_days, 30.0, abs_tol=0.1)
        assert math.isclose(new_imp, expected, abs_tol=1e-3)

    def test_retrieval_frequency_boost(self) -> None:
        """Frequently retrieved memories resist decay and have higher importance."""
        created = datetime.now(UTC) - timedelta(days=30)
        now = datetime.now(UTC)

        # Decayed without access
        unaccessed_imp, _ = calculate_decayed_importance(
            importance=0.8,
            created_at=created,
            last_accessed_at=None,
            access_count=0,
            now=now,
            tau_days=30.0,
        )

        # Decayed with 20 accesses
        accessed_imp, _ = calculate_decayed_importance(
            importance=0.8,
            created_at=created,
            last_accessed_at=None,
            access_count=20,
            now=now,
            tau_days=30.0,
            beta=0.15,
        )

        assert accessed_imp > unaccessed_imp

    def test_anchor_prefers_last_accessed_over_created(self) -> None:
        """If last_accessed_at is present, it acts as the temporal anchor."""
        created = datetime.now(UTC) - timedelta(days=60)
        last_accessed = datetime.now(UTC) - timedelta(days=5)
        now = datetime.now(UTC)

        new_imp, elapsed_days = calculate_decayed_importance(
            importance=0.8,
            created_at=created,
            last_accessed_at=last_accessed,
            access_count=1,
            now=now,
            tau_days=30.0,
        )
        assert math.isclose(elapsed_days, 5.0, abs_tol=0.1)
        assert new_imp > 0.7  # Only 5 days decayed, not 60 days

    def test_importance_bounds_clamping(self) -> None:
        """Importance is clamped between 0.01 and 1.0."""
        created = datetime.now(UTC) - timedelta(days=1000)
        now = datetime.now(UTC)
        low_imp, _ = calculate_decayed_importance(
            importance=0.05,
            created_at=created,
            now=now,
            tau_days=10.0,
        )
        assert low_imp == 0.01

        high_imp, _ = calculate_decayed_importance(
            importance=1.0,
            created_at=now,
            access_count=1000,
            now=now,
            beta=1.0,
        )
        assert high_imp == 1.0


class TestMemoryDecayEngine:
    """Integration tests for MemoryDecayEngine with SQLite and Vector stores."""

    @pytest.mark.asyncio
    async def test_decay_user_memories_lifecycle(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        """Verify unaccessed memories decay in SQL and vector store while accessed ones resist decay."""
        now = datetime.now(UTC)
        thirty_days_ago = now - timedelta(days=30)

        # 1. Unaccessed memory
        mem_unaccessed = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            entity="Location",
            importance=0.8,
            created_at=thirty_days_ago,
            access_count=0,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem_unaccessed)
        v1 = await embedder.embed_text(mem_unaccessed.content)
        await vector_store.upsert(
            sample_user_id,
            [
                VectorPoint(
                    id=mem_unaccessed.id,
                    vector=v1,
                    payload={
                        "content": mem_unaccessed.content,
                        "importance": 0.8,
                        "status": "active",
                    },
                )
            ],
        )

        # 2. Frequently accessed memory
        mem_accessed = MemoryItem(
            user_id=sample_user_id,
            content="User is a conversational AI researcher",
            entity="Career",
            importance=0.8,
            created_at=thirty_days_ago,
            last_accessed_at=now - timedelta(days=1),
            access_count=15,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem_accessed)
        v2 = await embedder.embed_text(mem_accessed.content)
        await vector_store.upsert(
            sample_user_id,
            [
                VectorPoint(
                    id=mem_accessed.id,
                    vector=v2,
                    payload={
                        "content": mem_accessed.content,
                        "importance": 0.8,
                        "status": "active",
                    },
                )
            ],
        )

        engine = MemoryDecayEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            tau_days=30.0,
        )

        res = await engine.decay_user_memories(user_id=sample_user_id, now=now)

        assert res.processed_count == 2
        assert res.updated_count == 2
        assert res.pruned_count == 0

        # Verify SQL store values
        m1_db = await sql_store.get_memory(sample_user_id, mem_unaccessed.id)
        m2_db = await sql_store.get_memory(sample_user_id, mem_accessed.id)
        assert m1_db is not None
        assert m2_db is not None

        # Unaccessed memory decayed from 0.8 to ~0.294
        assert m1_db.importance < 0.35
        # Frequently accessed memory kept high importance (> 0.75)
        assert m2_db.importance > 0.75

        # Verify Vector store payload values
        search_hits = await vector_store.search(
            sample_user_id, query_vector=v1, limit=5, status="active"
        )
        hit_map = {h.id: h.payload.get("importance") for h in search_hits}
        assert math.isclose(
            float(hit_map[mem_unaccessed.id] or 0.0), m1_db.importance, abs_tol=1e-3
        )
        assert math.isclose(float(hit_map[mem_accessed.id] or 0.0), m2_db.importance, abs_tol=1e-3)

    @pytest.mark.asyncio
    async def test_decay_soft_prune_neglected(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        """Verify severely neglected memories below prune_threshold are soft-pruned to SUPERSEDED."""
        now = datetime.now(UTC)
        neglected_time = now - timedelta(days=120)

        mem = MemoryItem(
            user_id=sample_user_id,
            content="User once mentioned liking apples",
            importance=0.10,
            created_at=neglected_time,
            access_count=0,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem)
        v = await embedder.embed_text(mem.content)
        await vector_store.upsert(
            sample_user_id,
            [VectorPoint(id=mem.id, vector=v, payload={"status": "active", "importance": 0.10})],
        )

        engine = MemoryDecayEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            tau_days=30.0,
            prune_threshold=0.05,
        )

        res = await engine.decay_user_memories(user_id=sample_user_id, now=now)
        assert res.pruned_count == 1

        # Invariant: Record still exists in SQL (never hard-deleted by decay)
        mem_db = await sql_store.get_memory(sample_user_id, mem.id)
        assert mem_db is not None
        assert mem_db.status == MemoryStatus.SUPERSEDED
        assert mem_db.superseded_by == "pruned_decay"

        # Invariant: Active vector search does NOT return soft-pruned memory
        active_hits = await vector_store.search(
            sample_user_id, query_vector=v, limit=5, status="active"
        )
        assert mem.id not in [h.id for h in active_hits]

    @pytest.mark.asyncio
    async def test_decay_user_isolation(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        """Verify decaying user A never affects user B's memories."""
        now = datetime.now(UTC)
        old_time = now - timedelta(days=45)

        mem_a = MemoryItem(
            user_id=sample_user_id,
            content="User A fact",
            importance=0.8,
            created_at=old_time,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem_a)

        mem_b = MemoryItem(
            user_id=alternate_user_id,
            content="User B fact",
            importance=0.8,
            created_at=old_time,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem_b)

        engine = MemoryDecayEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            tau_days=30.0,
        )

        # Run decay for user A only
        res = await engine.decay_user_memories(user_id=sample_user_id, now=now)
        assert res.processed_count == 1

        # User A's memory decayed
        db_a = await sql_store.get_memory(sample_user_id, mem_a.id)
        assert db_a is not None and db_a.importance < 0.5

        # User B's memory remains untouched at 0.8
        db_b = await sql_store.get_memory(alternate_user_id, mem_b.id)
        assert db_b is not None and db_b.importance == 0.8


def _make_mock_extractor(content: str = "User fact") -> FactExtractor:
    response_json = json.dumps(
        {
            "facts": [
                {
                    "content": content,
                    "entity": "User",
                    "importance": 0.8,
                    "confidence": 0.9,
                }
            ]
        }
    )

    async def mock_llm(**_kwargs: Any) -> str:
        return response_json

    return FactExtractor(completion_fn=mock_llm)


class TestExplicitForgettingAPI:
    """Tests for explicit hard-deletion APIs (forget and forget_all) on MemoryLayer."""

    @pytest.mark.asyncio
    async def test_forget_single_memory(self, sample_user_id: str) -> None:
        """Verify forget hard-deletes memory from SQL and vector stores."""
        extractor = _make_mock_extractor("I live in Pune.")
        mem_layer = await MemoryLayer.create_local(vector_dim=32, extractor=extractor)

        # Ingest memory
        mems = await mem_layer.add(
            user_id=sample_user_id,
            messages=[{"role": "user", "content": "I live in Pune."}],
        )
        assert len(mems) > 0
        target_id = mems[0].id

        # Verify it exists in SQL and Vector store
        stored_sql = await mem_layer.store.get_memory(sample_user_id, target_id)
        assert stored_sql is not None

        # Execute explicit forget
        success = await mem_layer.forget(sample_user_id, target_id)
        assert success is True

        # Invariant: Record is completely gone from SQL (hard delete)
        assert await mem_layer.store.get_memory(sample_user_id, target_id) is None

        # Invariant: Vector point is removed
        vec = await mem_layer.embedder.embed_text("I live in Pune.")
        hits = await mem_layer.vector_store.search(
            sample_user_id, query_vector=vec, limit=5, status=None
        )
        assert target_id not in [h.id for h in hits]

        await mem_layer.close()

    @pytest.mark.asyncio
    async def test_forget_multi_user_isolation(
        self,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        """Verify user A cannot delete user B's memory."""
        extractor = _make_mock_extractor("Confidential data for User B.")
        mem_layer = await MemoryLayer.create_local(vector_dim=32, extractor=extractor)

        mems_b = await mem_layer.add(
            user_id=alternate_user_id,
            messages=[{"role": "user", "content": "Confidential data for User B."}],
        )
        assert len(mems_b) > 0
        target_b_id = mems_b[0].id

        # User A tries to delete User B's memory
        deleted = await mem_layer.forget(user_id=sample_user_id, memory_id=target_b_id)
        assert deleted is False

        # User B's memory is still intact
        stored_b = await mem_layer.store.get_memory(alternate_user_id, target_b_id)
        assert stored_b is not None

        await mem_layer.close()

    @pytest.mark.asyncio
    async def test_forget_all_gdpr_purge(
        self,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        """Verify forget_all purges all user data while leaving other users unaffected."""
        extractor = _make_mock_extractor("User message.")
        mem_layer = await MemoryLayer.create_local(vector_dim=32, extractor=extractor)

        await mem_layer.add(
            user_id=sample_user_id,
            messages=[{"role": "user", "content": "User A message 1."}],
        )
        await mem_layer.add(
            user_id=alternate_user_id,
            messages=[{"role": "user", "content": "User B message 1."}],
        )

        purged_count = await mem_layer.forget_all(sample_user_id)
        assert purged_count > 0

        # User A has 0 memories and 0 episodes
        assert await mem_layer.store.list_memories(sample_user_id) == []
        assert await mem_layer.store.list_episodes(sample_user_id) == []

        # User B's data is completely intact
        mems_b = await mem_layer.store.list_memories(alternate_user_id)
        assert len(mems_b) > 0

        await mem_layer.close()

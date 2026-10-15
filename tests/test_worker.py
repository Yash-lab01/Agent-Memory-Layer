"""Unit and integration tests for ConsolidationWorker, APScheduler lifecycle, and multi-tenant sweeps."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from memlayer.api import MemoryLayer
from memlayer.consolidate.decay import MemoryDecayEngine
from memlayer.consolidate.dedup import DeduplicationEngine
from memlayer.consolidate.worker import ConsolidationWorker
from memlayer.llm.embedder import MockEmbedder
from memlayer.models import MemoryItem, MemoryStatus
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore


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
    """Deterministic MockEmbedder fixture."""
    return MockEmbedder(dimension=32)


@pytest.fixture
def dedup_engine(
    sql_store: SQLStore,
    vector_store: QdrantVectorStore,
    embedder: MockEmbedder,
) -> DeduplicationEngine:
    """DeduplicationEngine fixture."""
    return DeduplicationEngine(store=sql_store, vector_store=vector_store, embedder=embedder)


@pytest.fixture
def decay_engine(
    sql_store: SQLStore,
    vector_store: QdrantVectorStore,
    embedder: MockEmbedder,
) -> MemoryDecayEngine:
    """MemoryDecayEngine fixture."""
    return MemoryDecayEngine(
        store=sql_store,
        vector_store=vector_store,
        embedder=embedder,
        tau_days=30.0,
    )


class TestConsolidationWorkerLifecycle:
    """Unit tests for ConsolidationWorker lifecycle states and control methods."""

    def test_worker_initial_state(
        self,
        sql_store: SQLStore,
        dedup_engine: DeduplicationEngine,
        decay_engine: MemoryDecayEngine,
    ) -> None:
        worker = ConsolidationWorker(
            store=sql_store,
            dedup_engine=dedup_engine,
            decay_engine=decay_engine,
        )
        assert worker.is_running is False

    @pytest.mark.asyncio
    async def test_worker_start_pause_resume_stop(
        self,
        sql_store: SQLStore,
        dedup_engine: DeduplicationEngine,
        decay_engine: MemoryDecayEngine,
    ) -> None:
        worker = ConsolidationWorker(
            store=sql_store,
            dedup_engine=dedup_engine,
            decay_engine=decay_engine,
            dedup_interval_minutes=60.0,
            decay_interval_minutes=120.0,
        )

        try:
            worker.start()
            assert worker.is_running is True

            # Idempotent start
            worker.start()
            assert worker.is_running is True

            # Pause & resume
            worker.pause()
            worker.resume()
        finally:
            worker.stop(wait=False)
            assert worker.is_running is False


class TestConsolidationWorkerSweeps:
    """Integration tests for manual and periodic consolidation sweeps."""

    @pytest.mark.asyncio
    async def test_run_dedup_single_user(
        self,
        sql_store: SQLStore,
        dedup_engine: DeduplicationEngine,
        decay_engine: MemoryDecayEngine,
        sample_user_id: str,
    ) -> None:
        """Verify run_dedup executes deduplication on targeted user."""
        # Insert exact duplicate memories
        mem1 = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            status=MemoryStatus.ACTIVE,
        )
        mem2 = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memories([mem1, mem2])

        worker = ConsolidationWorker(
            store=sql_store,
            dedup_engine=dedup_engine,
            decay_engine=decay_engine,
        )

        results = await worker.run_dedup(user_id=sample_user_id)
        assert len(results) == 1
        assert results[0].user_id == sample_user_id
        assert results[0].merged_count == 1

    @pytest.mark.asyncio
    async def test_run_decay_single_user(
        self,
        sql_store: SQLStore,
        dedup_engine: DeduplicationEngine,
        decay_engine: MemoryDecayEngine,
        sample_user_id: str,
    ) -> None:
        """Verify run_decay decays stale memories for targeted user."""
        old_time = datetime.now(UTC) - timedelta(days=60)
        mem = MemoryItem(
            user_id=sample_user_id,
            content="User likes black coffee",
            importance=0.8,
            created_at=old_time,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem)

        worker = ConsolidationWorker(
            store=sql_store,
            dedup_engine=dedup_engine,
            decay_engine=decay_engine,
        )

        results = await worker.run_decay(user_id=sample_user_id)
        assert len(results) == 1
        assert results[0].user_id == sample_user_id
        assert results[0].updated_count == 1

    @pytest.mark.asyncio
    async def test_run_full_consolidation_multi_user(
        self,
        sql_store: SQLStore,
        dedup_engine: DeduplicationEngine,
        decay_engine: MemoryDecayEngine,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        """Verify full consolidation discovers all active users and runs sweeps across each."""
        # User A has duplicate memories
        await sql_store.add_memories(
            [
                MemoryItem(user_id=sample_user_id, content="Fact A", status=MemoryStatus.ACTIVE),
                MemoryItem(user_id=sample_user_id, content="Fact A", status=MemoryStatus.ACTIVE),
            ]
        )

        # User B has old memory
        old_time = datetime.now(UTC) - timedelta(days=45)
        await sql_store.add_memory(
            MemoryItem(
                user_id=alternate_user_id,
                content="Fact B",
                importance=0.8,
                created_at=old_time,
                status=MemoryStatus.ACTIVE,
            )
        )

        worker = ConsolidationWorker(
            store=sql_store,
            dedup_engine=dedup_engine,
            decay_engine=decay_engine,
        )

        full_res = await worker.run_full_consolidation(user_id=None)

        assert full_res.total_duplicates_merged >= 1
        assert full_res.total_memories_decayed >= 1
        assert len(full_res.dedup_results) == 2
        assert len(full_res.decay_results) == 2

    @pytest.mark.asyncio
    async def test_worker_fault_tolerance_on_user_error(
        self,
        sql_store: SQLStore,
        decay_engine: MemoryDecayEngine,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        """Verify an error during one user's sweep does not crash or abort sweeps for other users."""
        await sql_store.add_memories(
            [
                MemoryItem(
                    user_id=sample_user_id, content="User A mem", status=MemoryStatus.ACTIVE
                ),
                MemoryItem(
                    user_id=alternate_user_id, content="User B mem", status=MemoryStatus.ACTIVE
                ),
            ]
        )

        mock_dedup = AsyncMock()
        # Fail for User A, succeed for User B
        mock_dedup.deduplicate_user.side_effect = [
            RuntimeError("Simulated LLM crash for user A"),
            AsyncMock(user_id=alternate_user_id, merged_count=0, pairs=[]),
        ]

        worker = ConsolidationWorker(
            store=sql_store,
            dedup_engine=mock_dedup,
            decay_engine=decay_engine,
        )

        results = await worker.run_dedup(user_id=None)
        # Succeeded for user B despite failure on user A
        assert len(results) == 1


class TestMemoryLayerWorkerIntegration:
    """Integration tests verifying MemoryLayer manages ConsolidationWorker lifecycle."""

    @pytest.mark.asyncio
    async def test_memory_layer_worker_lifecycle(self) -> None:
        """Verify MemoryLayer starts and closes consolidation worker cleanly."""
        mem = await MemoryLayer.create_local(vector_dim=32)

        worker = mem.start_consolidation_worker(
            dedup_interval_minutes=10.0,
            decay_interval_minutes=30.0,
        )
        assert worker.is_running is True

        # Stop worker
        mem.stop_consolidation_worker()
        assert worker.is_running is False

        # Restart and close mem_layer
        mem.start_consolidation_worker()
        assert worker.is_running is True

        await mem.close()
        assert worker.is_running is False

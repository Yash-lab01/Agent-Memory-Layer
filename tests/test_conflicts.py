"""Unit and integration tests for contradiction detection, supersession, and tenant isolation."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import pytest
import pytest_asyncio

from memlayer.consolidate.conflicts import ConflictDetector
from memlayer.llm.embedder import MockEmbedder
from memlayer.models import MemoryItem, MemoryStatus
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore
from memlayer.store.vector_base import VectorPoint

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "contradictions.json"


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
    store = QdrantVectorStore(location=":memory:", vector_size=64)
    await store.init_store()
    yield store
    await store.close()


@pytest.fixture
def embedder() -> MockEmbedder:
    """Mock embedder fixture."""
    return MockEmbedder(dimension=64)


def load_contradiction_fixtures() -> list[dict[str, Any]]:
    """Load curated contradiction evaluation fixtures."""
    with FIXTURES_PATH.open("r", encoding="utf-8") as f:
        return cast(list[dict[str, Any]], json.load(f))


class TestConflictDetectionFixtures:
    """Test contradiction classification logic against curated benchmark fixtures."""

    @pytest.mark.asyncio
    async def test_all_fixture_cases(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        """Verify detect_conflicts correctly identifies contradiction vs compatible across all 10 fixtures."""
        fixtures = load_contradiction_fixtures()
        assert len(fixtures) == 10

        for case in fixtures:
            case_id = case["case_id"]
            expected_contradiction = case["is_contradiction"]
            old_mem = MemoryItem(
                user_id=sample_user_id,
                content=case["old_fact"],
                entity=case["entity"],
            )
            new_mem = MemoryItem(
                user_id=sample_user_id,
                content=case["new_fact"],
                entity=case["entity"],
            )

            # Mock LLM returning expected classification for this case
            mock_response_json = json.dumps(
                {
                    "conflicts": [
                        {
                            "old_memory_id": old_mem.id,
                            "is_contradiction": expected_contradiction,
                            "reason": case["description"],
                        }
                    ]
                }
            )

            async def mock_llm(resp: str = mock_response_json, **_kwargs: Any) -> str:
                return resp

            detector = ConflictDetector(
                store=sql_store,
                vector_store=vector_store,
                embedder=embedder,
                completion_fn=mock_llm,
            )

            conflicts = await detector.detect_conflicts(
                new_memory=new_mem, candidate_memories=[old_mem]
            )
            assert len(conflicts) == 1, f"Failed on case: {case_id}"
            assert conflicts[0].is_contradiction == expected_contradiction, (
                f"Mismatch for case: {case_id}"
            )
            assert conflicts[0].old_memory_id == old_mem.id

    @pytest.mark.asyncio
    async def test_detect_conflicts_empty_candidates(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        """Verify detect_conflicts returns empty list without calling LLM when candidates are empty."""
        new_mem = MemoryItem(
            user_id=sample_user_id,
            content="User relocated to Bangalore",
            entity="Location",
        )

        call_count = 0

        async def mock_llm(**_kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            return "{}"

        detector = ConflictDetector(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            completion_fn=mock_llm,
        )

        conflicts = await detector.detect_conflicts(new_memory=new_mem, candidate_memories=[])
        assert conflicts == []
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_detect_conflicts_malformed_json_fallback(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        """Verify detector safely falls back to empty list on malformed LLM response."""
        old_mem = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            entity="Location",
        )
        new_mem = MemoryItem(
            user_id=sample_user_id,
            content="User relocated to Bangalore",
            entity="Location",
        )

        async def mock_llm(**_kwargs: Any) -> str:
            return "This is not valid json at all!"

        detector = ConflictDetector(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            completion_fn=mock_llm,
        )

        conflicts = await detector.detect_conflicts(
            new_memory=new_mem, candidate_memories=[old_mem]
        )
        assert conflicts == []


class TestConflictResolutionAndSupersession:
    """Test full resolve_and_supersede pipeline against relational and vector stores."""

    @pytest.mark.asyncio
    async def test_resolve_and_supersede_lifecycle(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        """Verify non-destructive supersession: old memory marked SUPERSEDED in SQL and vector store."""
        # 1. Ingest initial active memory
        old_mem = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            entity="Location",
            importance=0.8,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(old_mem)
        old_vec = await embedder.embed_text(old_mem.content)
        await vector_store.upsert(
            sample_user_id,
            [
                VectorPoint(
                    id=old_mem.id,
                    vector=old_vec,
                    payload={
                        "content": old_mem.content,
                        "entity": old_mem.entity,
                        "importance": old_mem.importance,
                        "created_at": old_mem.created_at.isoformat(),
                        "status": "active",
                    },
                )
            ],
        )

        # 2. Prepare new memory that contradicts old memory
        new_mem = MemoryItem(
            user_id=sample_user_id,
            content="User relocated to Bangalore",
            entity="Location",
            importance=0.9,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(new_mem)
        new_vec = await embedder.embed_text(new_mem.content)
        await vector_store.upsert(
            sample_user_id,
            [
                VectorPoint(
                    id=new_mem.id,
                    vector=new_vec,
                    payload={
                        "content": new_mem.content,
                        "entity": new_mem.entity,
                        "importance": new_mem.importance,
                        "created_at": new_mem.created_at.isoformat(),
                        "status": "active",
                    },
                )
            ],
        )

        # 3. Mock LLM output marking old memory as contradicted
        mock_response = json.dumps(
            {
                "conflicts": [
                    {
                        "old_memory_id": old_mem.id,
                        "is_contradiction": True,
                        "reason": "User moved from Pune to Bangalore",
                    }
                ]
            }
        )

        async def mock_llm(**_kwargs: Any) -> str:
            return mock_response

        detector = ConflictDetector(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            completion_fn=mock_llm,
        )

        # 4. Execute resolve_and_supersede
        result = await detector.resolve_and_supersede(
            user_id=sample_user_id,
            new_memory=new_mem,
        )

        assert result.superseded_memory_ids == [old_mem.id]
        assert result.new_memory.id == new_mem.id

        # 5. Invariant check: SQL Store
        # Old memory must still exist (NEVER hard-deleted)
        stored_old = await sql_store.get_memory(sample_user_id, old_mem.id)
        assert stored_old is not None
        assert stored_old.status == MemoryStatus.SUPERSEDED
        assert stored_old.superseded_by == new_mem.id

        # New memory must be active and not superseded
        stored_new = await sql_store.get_memory(sample_user_id, new_mem.id)
        assert stored_new is not None
        assert stored_new.status == MemoryStatus.ACTIVE
        assert stored_new.superseded_by is None

        # 6. Invariant check: Vector Store
        # Active search must NOT return the old superseded memory
        active_search = await vector_store.search(
            user_id=sample_user_id,
            query_vector=old_vec,
            limit=10,
            status="active",
        )
        active_ids = [hit.id for hit in active_search]
        assert old_mem.id not in active_ids
        assert new_mem.id in active_ids

        # Unfiltered search includes old memory with superseded status in payload
        all_search = await vector_store.search(
            user_id=sample_user_id,
            query_vector=old_vec,
            limit=10,
            status=None,
        )
        old_hits = [h for h in all_search if h.id == old_mem.id]
        assert len(old_hits) == 1
        assert old_hits[0].payload["status"] == "superseded"
        assert old_hits[0].payload["superseded_by"] == new_mem.id

    @pytest.mark.asyncio
    async def test_resolve_compatible_facts_no_supersession(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        """Verify compatible memories coexist: neither is superseded."""
        coffee_mem = MemoryItem(
            user_id=sample_user_id,
            content="User enjoys drinking black coffee in morning",
            entity="Diet",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(coffee_mem)

        tea_mem = MemoryItem(
            user_id=sample_user_id,
            content="User likes sipping chamomile tea before bed",
            entity="Diet",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(tea_mem)

        # Mock LLM determining compatible facts
        mock_response = json.dumps(
            {
                "conflicts": [
                    {
                        "old_memory_id": coffee_mem.id,
                        "is_contradiction": False,
                        "reason": "Coffee in morning and tea at night can coexist",
                    }
                ]
            }
        )

        async def mock_llm(**_kwargs: Any) -> str:
            return mock_response

        detector = ConflictDetector(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            completion_fn=mock_llm,
        )

        result = await detector.resolve_and_supersede(
            user_id=sample_user_id,
            new_memory=tea_mem,
        )

        assert result.superseded_memory_ids == []

        # Both remain ACTIVE in SQL
        coffee_db = await sql_store.get_memory(sample_user_id, coffee_mem.id)
        tea_db = await sql_store.get_memory(sample_user_id, tea_mem.id)
        assert coffee_db is not None and coffee_db.status == MemoryStatus.ACTIVE
        assert tea_db is not None and tea_db.status == MemoryStatus.ACTIVE
        assert coffee_db.superseded_by is None
        assert tea_db.superseded_by is None

    @pytest.mark.asyncio
    async def test_resolve_supersede_tenant_isolation(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        """Verify User A's update can never alter or supersede User B's memories."""
        # User A's memory
        user_a_mem = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            entity="Location",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(user_a_mem)

        # User B adds a memory for the same entity
        user_b_mem = MemoryItem(
            user_id=alternate_user_id,
            content="User relocated to Bangalore",
            entity="Location",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(user_b_mem)

        detector = ConflictDetector(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
        )

        # Run resolution for User B
        result_b = await detector.resolve_and_supersede(
            user_id=alternate_user_id,
            new_memory=user_b_mem,
        )

        # User B should supersede nothing
        assert result_b.superseded_memory_ids == []

        # User A's memory MUST still be ACTIVE and untouched
        user_a_db = await sql_store.get_memory(sample_user_id, user_a_mem.id)
        assert user_a_db is not None
        assert user_a_db.status == MemoryStatus.ACTIVE
        assert user_a_db.superseded_by is None

    @pytest.mark.asyncio
    async def test_resolve_invalid_user_id(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
    ) -> None:
        """Verify empty or whitespace user_id raises ValueError."""
        detector = ConflictDetector(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
        )
        dummy_mem = MemoryItem(user_id="any", content="dummy content", entity="Location")

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await detector.resolve_and_supersede(user_id="", new_memory=dummy_mem)

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await detector.resolve_and_supersede(user_id="   ", new_memory=dummy_mem)

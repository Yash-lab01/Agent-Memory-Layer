"""Unit and integration tests for DeduplicationEngine, near-duplicate merging, and tenant isolation."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio

from memlayer.consolidate.dedup import (
    DeduplicationEngine,
    _cosine_similarity,
)
from memlayer.llm.embedder import BaseEmbedder, MockEmbedder
from memlayer.models import MemoryItem, MemoryStatus
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore
from memlayer.store.vector_base import VectorPoint


class ControlledEmbedder(BaseEmbedder):
    """Embedder allowing deterministic control of vector embeddings for testing."""

    def __init__(self, mapping: dict[str, list[float]], default_dim: int = 4) -> None:
        self.mapping = mapping
        self.default_dim = default_dim

    @property
    def dimension(self) -> int:
        return self.default_dim

    async def embed_text(self, text: str) -> list[float]:
        if text in self.mapping:
            return self.mapping[text]
        return [1.0] + [0.0] * (self.default_dim - 1)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]


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
    store = QdrantVectorStore(location=":memory:", vector_size=4)
    await store.init_store()
    yield store
    await store.close()


class TestCosineSimilarity:
    """Unit tests for mathematical cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0, 0.0]
        assert pytest.approx(_cosine_similarity(v, v)) == 1.0

    def test_orthogonal_vectors(self) -> None:
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]
        assert pytest.approx(_cosine_similarity(v1, v2)) == 0.0

    def test_opposite_vectors(self) -> None:
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0, 0.0]
        assert pytest.approx(_cosine_similarity(v1, v2)) == -1.0

    def test_zero_vector(self) -> None:
        v1 = [0.0, 0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0, 0.0]
        assert _cosine_similarity(v1, v2) == 0.0


class TestDeduplicationEngine:
    """Test suite for DeduplicationEngine lifecycle and invariants."""

    @pytest.mark.asyncio
    async def test_dedup_exact_text_match(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        sample_user_id: str,
    ) -> None:
        """Verify exact duplicate facts are merged into canonical memory without LLM call."""
        embedder = ControlledEmbedder(
            {
                "User lives in Pune": [1.0, 0.0, 0.0, 0.0],
            }
        )

        # Ingest first memory
        mem1 = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            entity="Location",
            importance=0.8,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem1)
        v1 = await embedder.embed_text(mem1.content)
        await vector_store.upsert(
            sample_user_id,
            [
                VectorPoint(
                    id=mem1.id,
                    vector=v1,
                    payload={"content": mem1.content, "status": "active"},
                )
            ],
        )

        # Ingest duplicate memory with slightly longer text
        mem2 = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune, Maharashtra",
            entity="Location",
            importance=0.85,
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem2)
        # Give mem2 identical vector to trigger exact_threshold
        embedder.mapping["User lives in Pune, Maharashtra"] = [1.0, 0.0, 0.0, 0.0]
        v2 = await embedder.embed_text(mem2.content)
        await vector_store.upsert(
            sample_user_id,
            [
                VectorPoint(
                    id=mem2.id,
                    vector=v2,
                    payload={"content": mem2.content, "status": "active"},
                )
            ],
        )

        engine = DeduplicationEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = await engine.deduplicate_user(sample_user_id)

        assert result.merged_count == 1
        assert len(result.pairs) == 1
        pair = result.pairs[0]

        # Longer text (mem2) should be selected as canonical
        assert pair.canonical_id == mem2.id
        assert pair.duplicate_id == mem1.id

        # Invariant: SQL Store
        # Duplicate is SUPERSEDED by canonical
        stored_dup = await sql_store.get_memory(sample_user_id, mem1.id)
        assert stored_dup is not None
        assert stored_dup.status == MemoryStatus.SUPERSEDED
        assert stored_dup.superseded_by == mem2.id

        # Canonical remains ACTIVE and has access_count reinforced
        stored_canonical = await sql_store.get_memory(sample_user_id, mem2.id)
        assert stored_canonical is not None
        assert stored_canonical.status == MemoryStatus.ACTIVE
        assert stored_canonical.superseded_by is None
        assert stored_canonical.access_count == 1

        # Invariant: Vector Store
        # Active vector search returns only canonical memory
        active_hits = await vector_store.search(
            user_id=sample_user_id,
            query_vector=v1,
            limit=10,
            status="active",
        )
        active_ids = [h.id for h in active_hits]
        assert mem1.id not in active_ids
        assert mem2.id in active_ids

    @pytest.mark.asyncio
    async def test_dedup_llm_verified_paraphrase(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        sample_user_id: str,
    ) -> None:
        """Verify borderline similarity pair is confirmed by LLM and merged."""
        embedder = ControlledEmbedder(
            {
                "User works at Google DeepMind as an AI researcher": [0.9, 0.435, 0.0, 0.0],
                "User is an AI researcher at Google DeepMind": [0.7, 0.714, 0.0, 0.0],
            }
        )

        mem1 = MemoryItem(
            user_id=sample_user_id,
            content="User works at Google DeepMind as an AI researcher",
            entity="Career",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem1)
        v1 = await embedder.embed_text(mem1.content)
        await vector_store.upsert(
            sample_user_id,
            [VectorPoint(id=mem1.id, vector=v1, payload={"status": "active"})],
        )

        mem2 = MemoryItem(
            user_id=sample_user_id,
            content="User is an AI researcher at Google DeepMind",
            entity="Career",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem2)
        v2 = await embedder.embed_text(mem2.content)
        await vector_store.upsert(
            sample_user_id,
            [VectorPoint(id=mem2.id, vector=v2, payload={"status": "active"})],
        )

        # Mock LLM returning confirmation of duplicate with mem1 as canonical
        mock_response = json.dumps(
            {
                "is_duplicate": True,
                "canonical_id": mem1.id,
                "reason": "Paraphrase of employment role",
            }
        )

        async def mock_llm(**_kwargs: Any) -> str:
            return mock_response

        engine = DeduplicationEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            completion_fn=mock_llm,
        )

        result = await engine.deduplicate_user(sample_user_id, similarity_threshold=0.85)

        assert result.merged_count == 1
        pair = result.pairs[0]
        assert pair.canonical_id == mem1.id
        assert pair.duplicate_id == mem2.id

        # Verify SQL states
        stored_dup = await sql_store.get_memory(sample_user_id, mem2.id)
        assert stored_dup is not None and stored_dup.status == MemoryStatus.SUPERSEDED
        assert stored_dup.superseded_by == mem1.id

    @pytest.mark.asyncio
    async def test_dedup_llm_rejected_coexisting(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        sample_user_id: str,
    ) -> None:
        """Verify coexisting facts with high topic similarity are preserved when LLM rejects duplication."""
        embedder = ControlledEmbedder(
            {
                "User drinks coffee in the morning": [0.9, 0.435, 0.0, 0.0],
                "User drinks tea in the evening": [0.7, 0.714, 0.0, 0.0],
            }
        )

        mem1 = MemoryItem(
            user_id=sample_user_id,
            content="User drinks coffee in the morning",
            entity="Diet",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem1)

        mem2 = MemoryItem(
            user_id=sample_user_id,
            content="User drinks tea in the evening",
            entity="Diet",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem2)

        # Mock LLM rejecting duplicate
        mock_response = json.dumps(
            {
                "is_duplicate": False,
                "canonical_id": "",
                "reason": "Different beverages for morning vs evening can coexist",
            }
        )

        async def mock_llm(**_kwargs: Any) -> str:
            return mock_response

        engine = DeduplicationEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            completion_fn=mock_llm,
        )

        result = await engine.deduplicate_user(sample_user_id, similarity_threshold=0.85)

        assert result.merged_count == 0
        assert len(result.pairs) == 0

        # Both remain ACTIVE
        m1 = await sql_store.get_memory(sample_user_id, mem1.id)
        m2 = await sql_store.get_memory(sample_user_id, mem2.id)
        assert m1 is not None and m1.status == MemoryStatus.ACTIVE
        assert m2 is not None and m2.status == MemoryStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_dedup_user_isolation(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        """Verify deduplication never merges or alters memories belonging to other users."""
        embedder = MockEmbedder(dimension=4)

        # User A has memory
        mem_a = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            entity="Location",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem_a)

        # User B has duplicate content
        mem_b = MemoryItem(
            user_id=alternate_user_id,
            content="User lives in Pune",
            entity="Location",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(mem_b)

        engine = DeduplicationEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
        )

        # Run dedup for User A
        res_a = await engine.deduplicate_user(sample_user_id)
        assert res_a.merged_count == 0

        # Run dedup for User B
        res_b = await engine.deduplicate_user(alternate_user_id)
        assert res_b.merged_count == 0

        # Both must still be ACTIVE and untouched
        a_db = await sql_store.get_memory(sample_user_id, mem_a.id)
        b_db = await sql_store.get_memory(alternate_user_id, mem_b.id)
        assert a_db is not None and a_db.status == MemoryStatus.ACTIVE
        assert b_db is not None and b_db.status == MemoryStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_dedup_insufficient_memories(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        sample_user_id: str,
    ) -> None:
        """Verify users with fewer than 2 active memories return empty result without errors."""
        embedder = MockEmbedder(dimension=4)
        engine = DeduplicationEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
        )

        res0 = await engine.deduplicate_user(sample_user_id)
        assert res0.merged_count == 0

        single_mem = MemoryItem(
            user_id=sample_user_id,
            content="User is learning French",
            status=MemoryStatus.ACTIVE,
        )
        await sql_store.add_memory(single_mem)

        res1 = await engine.deduplicate_user(sample_user_id)
        assert res1.merged_count == 0

    @pytest.mark.asyncio
    async def test_dedup_invalid_user_id(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
    ) -> None:
        """Verify empty or whitespace user_id raises ValueError."""
        engine = DeduplicationEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=MockEmbedder(dimension=4),
        )

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await engine.deduplicate_user("")

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await engine.deduplicate_user("   ")

    @pytest.mark.asyncio
    async def test_dedup_llm_error_fallback(
        self,
        sql_store: SQLStore,
        vector_store: QdrantVectorStore,
        sample_user_id: str,
    ) -> None:
        """Verify engine gracefully falls back to non-duplicate when LLM fails or returns invalid output."""
        embedder = ControlledEmbedder(
            {
                "Memory A": [0.9, 0.435, 0.0, 0.0],
                "Memory B": [0.7, 0.714, 0.0, 0.0],
            }
        )

        mem1 = MemoryItem(user_id=sample_user_id, content="Memory A", status=MemoryStatus.ACTIVE)
        mem2 = MemoryItem(user_id=sample_user_id, content="Memory B", status=MemoryStatus.ACTIVE)
        await sql_store.add_memory(mem1)
        await sql_store.add_memory(mem2)

        async def broken_llm(**_kwargs: Any) -> str:
            return "This is completely invalid JSON!"

        engine = DeduplicationEngine(
            store=sql_store,
            vector_store=vector_store,
            embedder=embedder,
            completion_fn=broken_llm,
        )

        result = await engine.deduplicate_user(sample_user_id, similarity_threshold=0.85)
        assert result.merged_count == 0

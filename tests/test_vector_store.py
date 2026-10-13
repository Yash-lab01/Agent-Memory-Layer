"""Tests for embedders, vector storage interface, and QdrantVectorStore isolation."""

import math
import uuid
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from memlayer.llm.embedder import MockEmbedder
from memlayer.store.vector import QdrantVectorStore
from memlayer.store.vector_base import VectorPoint


@pytest_asyncio.fixture
async def vector_store() -> AsyncGenerator[QdrantVectorStore, None]:
    """Fixture providing an initialized in-memory QdrantVectorStore."""
    store = QdrantVectorStore(location=":memory:", vector_size=384)
    await store.init_store()
    yield store
    await store.close()


@pytest.fixture
def embedder() -> MockEmbedder:
    """Fixture providing deterministic MockEmbedder."""
    return MockEmbedder(dimension=384)


class TestEmbedders:
    """Unit tests for embedding generation."""

    @pytest.mark.asyncio
    async def test_mock_embedder_dimension_and_norm(self, embedder: MockEmbedder) -> None:
        assert embedder.dimension == 384
        vec = await embedder.embed_text("Test fact about conversational AI")
        assert len(vec) == 384
        norm = math.sqrt(sum(v * v for v in vec))
        assert math.isclose(norm, 1.0, rel_tol=1e-5)

    @pytest.mark.asyncio
    async def test_mock_embedder_determinism(self, embedder: MockEmbedder) -> None:
        vec1 = await embedder.embed_text("User lives in Pune")
        vec2 = await embedder.embed_text("User lives in Pune")
        vec3 = await embedder.embed_text("User moved to Bangalore")

        assert vec1 == vec2
        assert vec1 != vec3

    @pytest.mark.asyncio
    async def test_mock_embedder_batch(self, embedder: MockEmbedder) -> None:
        texts = ["Fact A", "Fact B", "Fact C"]
        batch_vecs = await embedder.embed_texts(texts)
        assert len(batch_vecs) == 3

        for i, text in enumerate(texts):
            single_vec = await embedder.embed_text(text)
            assert batch_vecs[i] == single_vec


class TestQdrantVectorStoreCRUD:
    """Unit tests for QdrantVectorStore operations."""

    @pytest.mark.asyncio
    async def test_upsert_and_search(
        self,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        fact1_id = str(uuid.uuid4())
        fact2_id = str(uuid.uuid4())

        vec1 = await embedder.embed_text("User loves Python programming")
        vec2 = await embedder.embed_text("The capital of France is Paris")

        point1 = VectorPoint(
            id=fact1_id,
            vector=vec1,
            payload={"content": "User loves Python programming", "status": "active"},
        )
        point2 = VectorPoint(
            id=fact2_id,
            vector=vec2,
            payload={"content": "The capital of France is Paris", "status": "active"},
        )
        await vector_store.upsert(sample_user_id, [point1, point2])

        # Query closely matching point1
        query_vec = await embedder.embed_text("User loves Python programming")
        hits = await vector_store.search(sample_user_id, query_vec, limit=5)

        assert len(hits) == 2
        assert hits[0].id == fact1_id
        assert math.isclose(hits[0].score, 1.0, rel_tol=1e-4)
        assert hits[0].payload["content"] == "User loves Python programming"
        assert hits[0].payload["user_id"] == sample_user_id

    @pytest.mark.asyncio
    async def test_search_status_filter(
        self,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        active_id = str(uuid.uuid4())
        superseded_id = str(uuid.uuid4())

        vec = await embedder.embed_text("User address")
        point_active = VectorPoint(
            id=active_id,
            vector=vec,
            payload={"content": "Bangalore", "status": "active"},
        )
        point_superseded = VectorPoint(
            id=superseded_id,
            vector=vec,
            payload={"content": "Pune", "status": "superseded"},
        )
        await vector_store.upsert(sample_user_id, [point_active, point_superseded])

        # Active only search (default)
        active_hits = await vector_store.search(sample_user_id, vec, status="active")
        assert len(active_hits) == 1
        assert active_hits[0].id == active_id

        # Superseded only search
        superseded_hits = await vector_store.search(sample_user_id, vec, status="superseded")
        assert len(superseded_hits) == 1
        assert superseded_hits[0].id == superseded_id

        # All statuses search (status=None)
        all_hits = await vector_store.search(sample_user_id, vec, status=None)
        assert len(all_hits) == 2

    @pytest.mark.asyncio
    async def test_delete_vectors(
        self,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        point_id = str(uuid.uuid4())
        vec = await embedder.embed_text("Fact to remove")
        point = VectorPoint(id=point_id, vector=vec, payload={"status": "active"})
        await vector_store.upsert(sample_user_id, [point])

        # Verify present
        hits = await vector_store.search(sample_user_id, vec)
        assert len(hits) == 1

        # Delete
        await vector_store.delete(sample_user_id, [point_id])

        # Verify gone
        hits_after = await vector_store.search(sample_user_id, vec)
        assert len(hits_after) == 0

    @pytest.mark.asyncio
    async def test_delete_user_vectors(
        self,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
    ) -> None:
        vec = await embedder.embed_text("Sample fact")
        points = [
            VectorPoint(id=str(uuid.uuid4()), vector=vec, payload={"status": "active"}),
            VectorPoint(id=str(uuid.uuid4()), vector=vec, payload={"status": "active"}),
        ]
        await vector_store.upsert(sample_user_id, points)

        deleted_count = await vector_store.delete_user_vectors(sample_user_id)
        assert deleted_count == 2

        hits = await vector_store.search(sample_user_id, vec)
        assert len(hits) == 0


class TestVectorUserIsolation:
    """Rigorous tests verifying that vectors are isolated by user_id."""

    @pytest.mark.asyncio
    async def test_user_cannot_retrieve_other_users_vectors(
        self,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        user1_id = str(uuid.uuid4())
        vec = await embedder.embed_text("Private medical facts of User 1")

        point_user1 = VectorPoint(
            id=user1_id,
            vector=vec,
            payload={"content": "Private medical facts", "status": "active"},
        )
        await vector_store.upsert(sample_user_id, [point_user1])

        # User 1 searches: should find it
        user1_hits = await vector_store.search(sample_user_id, vec)
        assert len(user1_hits) == 1
        assert user1_hits[0].id == user1_id

        # User 2 searches with EXACT SAME query vector: MUST return zero hits
        user2_hits = await vector_store.search(alternate_user_id, vec)
        assert len(user2_hits) == 0

    @pytest.mark.asyncio
    async def test_user_cannot_delete_other_users_vectors(
        self,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        user1_point_id = str(uuid.uuid4())
        vec = await embedder.embed_text("User 1 confidential data")

        point_user1 = VectorPoint(
            id=user1_point_id,
            vector=vec,
            payload={"content": "User 1 data", "status": "active"},
        )
        await vector_store.upsert(sample_user_id, [point_user1])

        # User 2 attempts to delete User 1's vector ID
        await vector_store.delete(alternate_user_id, [user1_point_id])

        # User 1's vector MUST still exist
        user1_hits = await vector_store.search(sample_user_id, vec)
        assert len(user1_hits) == 1
        assert user1_hits[0].id == user1_point_id

        # User 2 purges their own vectors
        deleted_count = await vector_store.delete_user_vectors(alternate_user_id)
        assert deleted_count == 0

        # User 1's vector still intact
        user1_hits_again = await vector_store.search(sample_user_id, vec)
        assert len(user1_hits_again) == 1

from collections.abc import AsyncGenerator

import pytest

from memlayer.llm.embedder import MockEmbedder
from memlayer.models import MemoryItem, MemoryStatus
from memlayer.retrieve.bm25 import BM25Index, tokenize
from memlayer.retrieve.hybrid import HybridRetriever
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore
from memlayer.store.vector_base import VectorPoint

SetupEnv = tuple[SQLStore, QdrantVectorStore, MockEmbedder, HybridRetriever]


class TestBM25Index:
    """Tests for BM25 tokenizer and sparse search index."""

    def test_tokenize_clean_and_lowercase(self) -> None:
        text = "Hello, World! This is a test: Python-3.11."
        tokens = tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
        assert "3" in tokens
        assert "11" in tokens

    def test_tokenize_empty_and_whitespace(self) -> None:
        assert tokenize("") == []
        assert tokenize("    ") == []

    def test_bm25_empty_corpus(self) -> None:
        index = BM25Index([], [])
        hits = index.search("python")
        assert hits == []

    def test_bm25_search_ranking(self) -> None:
        doc_ids = ["doc_1", "doc_2", "doc_3"]
        texts = [
            "User prefers Python for data science and machine learning",
            "User lives in Pune and loves outdoor hiking",
            "User prefers Postgres and Redis for databases",
        ]
        index = BM25Index(doc_ids, texts)

        # Query matches doc_1 strongly
        hits_python = index.search("Python machine learning")
        assert len(hits_python) > 0
        assert hits_python[0].id == "doc_1"

        # Query matches doc_2
        hits_pune = index.search("Pune hiking")
        assert len(hits_pune) > 0
        assert hits_pune[0].id == "doc_2"

        # Query with non-existent words
        assert index.search("xylophone kangaroo") == []


class TestHybridRetriever:
    """Tests for HybridRetriever dual-stream search and fusion."""

    @pytest.fixture
    async def setup_env(self) -> AsyncGenerator[SetupEnv, None]:
        store = SQLStore(db_url="sqlite+aiosqlite:///:memory:")
        await store.init_db()

        embedder = MockEmbedder(dimension=32)
        vector_store = QdrantVectorStore(location=":memory:", vector_size=32)
        await vector_store.init_store()

        retriever = HybridRetriever(
            store=store,
            vector_store=vector_store,
            embedder=embedder,
        )

        yield store, vector_store, embedder, retriever

        await store.close()
        await vector_store.close()

    @pytest.mark.asyncio
    async def test_hybrid_search_with_rrf_and_convex(self, setup_env: SetupEnv) -> None:
        store, vector_store, embedder, retriever = setup_env

        user_id = "user_retrieval_test"

        mem1 = MemoryItem(
            user_id=user_id,
            content="User is highly proficient in PostgreSQL and relational modeling",
            entity="Database",
            importance=0.9,
            status=MemoryStatus.ACTIVE,
        )
        mem2 = MemoryItem(
            user_id=user_id,
            content="User enjoys cycling around Bangalore on weekends",
            entity="Lifestyle",
            importance=0.6,
            status=MemoryStatus.ACTIVE,
        )
        mem3 = MemoryItem(
            user_id=user_id,
            content="User designs high-performance microservices in Go",
            entity="Backend",
            importance=0.8,
            status=MemoryStatus.ACTIVE,
        )

        for m in [mem1, mem2, mem3]:
            await store.add_memory(m)
            vec = await embedder.embed_text(m.content)
            await vector_store.upsert(
                user_id,
                [
                    VectorPoint(
                        id=m.id,
                        vector=vec,
                        payload={"content": m.content, "status": m.status.value},
                    )
                ],
            )

        # Test RRF search for PostgreSQL keyword
        rrf_matches = await retriever.search(
            user_id, "PostgreSQL database modeling", limit=5, fusion_method="rrf"
        )
        assert len(rrf_matches) >= 1
        assert rrf_matches[0].memory.id == mem1.id
        assert rrf_matches[0].sparse_score > 0.0
        assert rrf_matches[0].combined_score > 0.0

        # Test Convex search for Bangalore cycling
        convex_matches = await retriever.search(
            user_id, "Bangalore cycling weekends", limit=5, fusion_method="convex"
        )
        assert len(convex_matches) >= 1
        assert convex_matches[0].memory.id == mem2.id
        assert convex_matches[0].sparse_score > 0.0
        assert convex_matches[0].combined_score > 0.0

    @pytest.mark.asyncio
    async def test_superseded_memories_filtered_by_default(self, setup_env: SetupEnv) -> None:
        store, vector_store, embedder, retriever = setup_env
        user_id = "user_supersede_test"

        mem_active = MemoryItem(
            user_id=user_id,
            content="User relocated to Bangalore recently",
            entity="Location",
            status=MemoryStatus.ACTIVE,
        )
        mem_superseded = MemoryItem(
            user_id=user_id,
            content="User lives in Pune city center",
            entity="Location",
            status=MemoryStatus.SUPERSEDED,
        )

        for m in [mem_active, mem_superseded]:
            await store.add_memory(m)
            vec = await embedder.embed_text(m.content)
            await vector_store.upsert(
                user_id,
                [
                    VectorPoint(
                        id=m.id,
                        vector=vec,
                        payload={"content": m.content, "status": m.status.value},
                    )
                ],
            )

        # Search with default (active only)
        active_matches = await retriever.search(user_id, "Pune city center", limit=5)
        matched_ids = [m.memory.id for m in active_matches]
        assert mem_superseded.id not in matched_ids

        # Search with status=None (include all)
        all_matches = await retriever.search(user_id, "Pune city center", limit=5, status=None)
        all_matched_ids = [m.memory.id for m in all_matches]
        assert mem_superseded.id in all_matched_ids

    @pytest.mark.asyncio
    async def test_strict_multi_user_isolation(self, setup_env: SetupEnv) -> None:
        store, vector_store, embedder, retriever = setup_env

        user_alice = "alice"
        user_bob = "bob"

        mem_alice = MemoryItem(
            user_id=user_alice,
            content="Secret project codename is Operation Phoenix",
            entity="Security",
            status=MemoryStatus.ACTIVE,
        )
        await store.add_memory(mem_alice)
        vec_alice = await embedder.embed_text(mem_alice.content)
        await vector_store.upsert(
            user_alice,
            [
                VectorPoint(
                    id=mem_alice.id,
                    vector=vec_alice,
                    payload={"content": mem_alice.content, "status": "active"},
                )
            ],
        )

        # Bob searches for Alice's secret
        bob_matches = await retriever.search(user_bob, "Operation Phoenix")
        assert len(bob_matches) == 0

        # Alice searches for her secret
        alice_matches = await retriever.search(user_alice, "Operation Phoenix")
        assert len(alice_matches) == 1
        assert alice_matches[0].memory.content == mem_alice.content

    @pytest.mark.asyncio
    async def test_boundary_and_empty_inputs(self, setup_env: SetupEnv) -> None:
        _, _, _, retriever = setup_env

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await retriever.search("", "query")

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await retriever.search("   ", "query")

        # Empty query
        assert await retriever.search("user_any", "") == []
        assert await retriever.search("user_any", "   ") == []

        # User with no memories
        assert await retriever.search("user_no_mems", "hello") == []

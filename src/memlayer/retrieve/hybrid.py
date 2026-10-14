"""Hybrid retrieval engine combining dense vector search and sparse BM25 scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from memlayer.llm.embedder import BaseEmbedder
from memlayer.models import MemoryItem, MemoryStatus
from memlayer.retrieve.bm25 import BM25Index
from memlayer.store.sql import SQLStore
from memlayer.store.vector_base import VectorStoreBase


@dataclass
class HybridMatch:
    """A retrieved memory item with individual and combined retrieval scores."""

    memory: MemoryItem
    combined_score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None


class HybridRetriever:
    """Coordinates hybrid retrieval across semantic vector and BM25 sparse indices."""

    def __init__(
        self,
        store: SQLStore,
        vector_store: VectorStoreBase,
        embedder: BaseEmbedder,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> None:
        """Initialize hybrid retriever with storage backends and fusion hyperparameters."""
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        fusion_method: Literal["rrf", "convex"] = "rrf",
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[HybridMatch]:
        """Perform hybrid search combining vector similarity and BM25 keywords.

        Args:
            user_id: Unique user identifier for strict isolation.
            query: Search query text.
            limit: Maximum number of matches to return.
            status: Optional memory status filter (default: ACTIVE).
            fusion_method: Fusion algorithm ('rrf' or 'convex').

        Returns:
            List of HybridMatch instances ranked by combined_score descending.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        clean_user_id = user_id.strip()

        if not query or not query.strip():
            return []

        # 1. Fetch active memories for BM25 indexing and verification
        user_memories = await self.store.list_memories(
            user_id=clean_user_id,
            status=status,
            limit=500,
        )
        if not user_memories:
            return []

        memory_by_id: dict[str, MemoryItem] = {m.id: m for m in user_memories}

        # 2. Sparse Search: BM25
        doc_ids = [m.id for m in user_memories]
        doc_texts = [f"{m.entity}: {m.content}" for m in user_memories]
        bm25_index = BM25Index(doc_ids=doc_ids, texts=doc_texts)
        bm25_hits = bm25_index.search(query, limit=limit * 2)

        sparse_scores: dict[str, float] = {hit.id: hit.score for hit in bm25_hits}
        sparse_ranks: dict[str, int] = {hit.id: rank + 1 for rank, hit in enumerate(bm25_hits)}

        # 3. Dense Search: Vector similarity
        query_vector = await self.embedder.embed_text(query)
        status_filter = status.value if status is not None else None
        vector_hits = await self.vector_store.search(
            user_id=clean_user_id,
            query_vector=query_vector,
            limit=limit * 2,
            status=status_filter,
        )

        dense_scores: dict[str, float] = {
            hit.id: hit.score for hit in vector_hits if hit.id in memory_by_id
        }
        dense_ranks: dict[str, int] = {
            hit.id: rank + 1 for rank, hit in enumerate(vector_hits) if hit.id in memory_by_id
        }

        # Candidate pool: union of IDs from dense and sparse results
        candidate_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        if not candidate_ids:
            return []

        matches: list[HybridMatch] = []

        # 4. Fusion
        w_dense = self.dense_weight if dense_weight is None else dense_weight
        w_sparse = self.sparse_weight if sparse_weight is None else sparse_weight

        if fusion_method == "rrf":
            for mem_id in candidate_ids:
                d_rank = dense_ranks.get(mem_id)
                s_rank = sparse_ranks.get(mem_id)

                rrf_dense = w_dense / (self.rrf_k + d_rank) if d_rank is not None else 0.0
                rrf_sparse = w_sparse / (self.rrf_k + s_rank) if s_rank is not None else 0.0
                combined = rrf_dense + rrf_sparse

                matches.append(
                    HybridMatch(
                        memory=memory_by_id[mem_id],
                        combined_score=combined,
                        dense_score=dense_scores.get(mem_id, 0.0),
                        sparse_score=sparse_scores.get(mem_id, 0.0),
                        dense_rank=d_rank,
                        sparse_rank=s_rank,
                    )
                )
        else:
            # Convex linear combination with natural zero-baseline normalization
            max_dense = max(dense_scores.values()) if dense_scores else 0.0
            max_sparse = max(sparse_scores.values()) if sparse_scores else 0.0

            for mem_id in candidate_ids:
                raw_d = dense_scores.get(mem_id, 0.0)
                norm_d = (max(0.0, raw_d) / max_dense) if max_dense > 0.0 else 0.0

                raw_s = sparse_scores.get(mem_id, 0.0)
                norm_s = (raw_s / max_sparse) if max_sparse > 0.0 else 0.0

                combined = (w_dense * norm_d) + (w_sparse * norm_s)

                matches.append(
                    HybridMatch(
                        memory=memory_by_id[mem_id],
                        combined_score=combined,
                        dense_score=raw_d,
                        sparse_score=raw_s,
                        dense_rank=dense_ranks.get(mem_id),
                        sparse_rank=sparse_ranks.get(mem_id),
                    )
                )

        # 5. Sort descending by combined_score
        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches[:limit]

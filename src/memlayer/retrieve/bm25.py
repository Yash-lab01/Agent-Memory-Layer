"""BM25 keyword search index using rank-bm25 for sparse textual retrieval."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


@dataclass(frozen=True)
class BM25Hit:
    """A matched document ID and its BM25 score."""

    id: str
    score: float


class BM25Index:
    """In-memory BM25 index over a collection of document texts."""

    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Build BM25 index from document IDs and corresponding text contents."""
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length.")

        self.doc_ids = list(doc_ids)
        self.texts = list(texts)
        self.tokenized_corpus = [tokenize(t) for t in self.texts]

        # Handle empty corpus gracefully
        if self.tokenized_corpus and any(len(t) > 0 for t in self.tokenized_corpus):
            self._bm25: BM25Okapi | None = BM25Okapi(self.tokenized_corpus)
        else:
            self._bm25 = None

    def search(self, query: str, limit: int = 10) -> list[BM25Hit]:
        """Search the indexed corpus with BM25 keyword matching."""
        if not self._bm25 or not self.doc_ids:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Pair document IDs with scores
        scored_pairs = [
            BM25Hit(id=doc_id, score=float(score))
            for doc_id, score in zip(self.doc_ids, scores, strict=True)
            if score > 0.0
        ]

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda hit: hit.score, reverse=True)
        return scored_pairs[:limit]

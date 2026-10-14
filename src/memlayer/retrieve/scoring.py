"""Multi-factor memory scoring engine combining relevance, recency decay, and importance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from memlayer.models import MemoryItem, ScoredMemory


@dataclass(frozen=True)
class ScoringWeights:
    """Hyperparameters for the multi-factor memory scoring formula.

    Formula: score = a * similarity + b * exp(-age / tau) + c * importance
    """

    a: float = 0.5  # Weight for similarity / relevance
    b: float = 0.3  # Weight for recency decay
    c: float = 0.2  # Weight for intrinsic importance
    tau: float = 72.0  # Exponential decay time-constant in hours (default: 3 days)

    def __post_init__(self) -> None:
        if self.a < 0.0 or self.b < 0.0 or self.c < 0.0:
            raise ValueError("Weights a, b, and c must be non-negative (>= 0.0).")
        if self.tau <= 0.0:
            raise ValueError("Decay time-constant tau must be strictly positive (> 0.0).")


def compute_recency_score(
    created_at: datetime,
    now: datetime | None = None,
    tau: float = 72.0,
) -> float:
    """Compute exponential recency decay factor: exp(-age / tau).

    Args:
        created_at: Timestamp when memory was observed or created.
        now: Current reference timestamp (defaults to datetime.now(UTC)).
        tau: Decay time-constant in hours.

    Returns:
        Recency factor between 0.0 (infinitely old) and 1.0 (brand new).
    """
    if tau <= 0.0:
        raise ValueError("Decay time-constant tau must be strictly positive (> 0.0).")

    ref_now = now or datetime.now(UTC)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=UTC)

    mem_time = created_at
    if mem_time.tzinfo is None:
        mem_time = mem_time.replace(tzinfo=UTC)

    age_seconds = (ref_now - mem_time).total_seconds()
    age_hours = max(0.0, age_seconds / 3600.0)

    return math.exp(-age_hours / tau)


def score_memory(
    memory: MemoryItem,
    similarity: float,
    weights: ScoringWeights | None = None,
    now: datetime | None = None,
) -> ScoredMemory:
    """Compute composite multi-factor score for a single memory item.

    Formula: score = a * similarity + b * exp(-age / tau) + c * importance

    Args:
        memory: The memory item to score.
        similarity: Query similarity score (0.0 to 1.0).
        weights: Hyperparameter weights (defaults to standard tuned weights).
        now: Reference timestamp for deterministic scoring.

    Returns:
        ScoredMemory with individual component scores and final composite score.
    """
    w = weights or ScoringWeights()

    recency = compute_recency_score(memory.created_at, now=now, tau=w.tau)
    importance = memory.importance if memory.importance is not None else 0.5

    composite = (w.a * similarity) + (w.b * recency) + (w.c * importance)

    return ScoredMemory(
        memory=memory,
        score=composite,
        similarity=similarity,
        recency_score=recency,
        importance_score=importance,
    )


def rank_memories(
    candidates: Sequence[tuple[MemoryItem, float]],
    weights: ScoringWeights | None = None,
    now: datetime | None = None,
    limit: int | None = None,
) -> list[ScoredMemory]:
    """Score and rank candidate (memory, similarity) pairs in descending score order.

    Args:
        candidates: Sequence of (MemoryItem, similarity) tuples.
        weights: Scoring weights configuration.
        now: Reference timestamp for deterministic scoring.
        limit: Optional maximum number of scored memories to return.

    Returns:
        List of ScoredMemory instances sorted by score descending.
    """
    scored = [
        score_memory(memory=mem, similarity=sim, weights=weights, now=now)
        for mem, sim in candidates
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    if limit is not None:
        return scored[:limit]
    return scored

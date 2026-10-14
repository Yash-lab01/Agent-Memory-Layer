"""Unit tests for multi-factor scoring formula, recency decay, and ranking."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from memlayer.models import MemoryItem, MemoryStatus
from memlayer.retrieve.scoring import (
    ScoringWeights,
    compute_recency_score,
    rank_memories,
    score_memory,
)


class TestScoringWeights:
    """Tests for ScoringWeights validation and defaults."""

    def test_default_weights(self) -> None:
        w = ScoringWeights()
        assert w.a == 0.5
        assert w.b == 0.3
        assert w.c == 0.2
        assert w.tau == 72.0

    def test_negative_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            ScoringWeights(a=-0.1)

        with pytest.raises(ValueError, match="must be non-negative"):
            ScoringWeights(b=-0.2)

        with pytest.raises(ValueError, match="must be non-negative"):
            ScoringWeights(c=-0.05)

    def test_invalid_tau_raises(self) -> None:
        with pytest.raises(ValueError, match="tau must be strictly positive"):
            ScoringWeights(tau=0.0)

        with pytest.raises(ValueError, match="tau must be strictly positive"):
            ScoringWeights(tau=-10.0)


class TestRecencyDecay:
    """Tests for compute_recency_score mathematical properties and edge cases."""

    def test_age_zero_returns_one(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
        created_at = now
        score = compute_recency_score(created_at, now=now, tau=72.0)
        assert pytest.approx(score, rel=1e-6) == 1.0

    def test_future_timestamp_clamped_to_one(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
        future_time = now + timedelta(hours=5)
        score = compute_recency_score(future_time, now=now, tau=72.0)
        assert pytest.approx(score, rel=1e-6) == 1.0

    def test_one_tau_decay(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
        tau = 72.0
        created_at = now - timedelta(hours=tau)
        score = compute_recency_score(created_at, now=now, tau=tau)
        assert pytest.approx(score, rel=1e-5) == math.exp(-1.0)  # ~0.367879

    def test_two_tau_decay(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
        tau = 72.0
        created_at = now - timedelta(hours=tau * 2)
        score = compute_recency_score(created_at, now=now, tau=tau)
        assert pytest.approx(score, rel=1e-5) == math.exp(-2.0)  # ~0.135335

    def test_very_old_memory_approaches_zero(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
        very_old = now - timedelta(days=365 * 5)
        score = compute_recency_score(very_old, now=now, tau=72.0)
        assert score >= 0.0
        assert score < 1e-10

    def test_naive_datetime_handling(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0)  # naive
        created_at = now - timedelta(hours=72)  # naive
        score = compute_recency_score(created_at, now=now, tau=72.0)
        assert pytest.approx(score, rel=1e-5) == math.exp(-1.0)


class TestScoreMemory:
    """Tests for score_memory composite calculation and ranking."""

    def test_score_memory_perfect_match(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
        mem = MemoryItem(
            user_id="user_1",
            content="User lives in Pune",
            importance=1.0,
            created_at=now,
            status=MemoryStatus.ACTIVE,
        )

        scored = score_memory(
            memory=mem,
            similarity=1.0,
            weights=ScoringWeights(a=0.5, b=0.3, c=0.2, tau=72.0),
            now=now,
        )

        assert pytest.approx(scored.score, rel=1e-6) == 1.0
        assert pytest.approx(scored.similarity, rel=1e-6) == 1.0
        assert pytest.approx(scored.recency_score, rel=1e-6) == 1.0
        assert pytest.approx(scored.importance_score, rel=1e-6) == 1.0

    def test_score_memory_missing_importance_defaults_to_neutral(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
        mem = MemoryItem(
            user_id="user_1",
            content="User prefers dark mode",
            importance=0.5,  # neutral
            created_at=now,
        )
        scored = score_memory(memory=mem, similarity=0.8, now=now)
        # score = 0.5 * 0.8 + 0.3 * 1.0 + 0.2 * 0.5 = 0.40 + 0.30 + 0.10 = 0.80
        assert pytest.approx(scored.score, rel=1e-6) == 0.80
        assert scored.importance_score == 0.5

    def test_rank_memories_descending_order(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)

        # Fresh and highly relevant
        mem_top = MemoryItem(
            user_id="u",
            content="Top memory",
            importance=0.9,
            created_at=now,
        )
        # Old but somewhat relevant
        mem_mid = MemoryItem(
            user_id="u",
            content="Mid memory",
            importance=0.7,
            created_at=now - timedelta(hours=100),
        )
        # Irrelevant and old
        mem_low = MemoryItem(
            user_id="u",
            content="Low memory",
            importance=0.2,
            created_at=now - timedelta(hours=500),
        )

        candidates = [
            (mem_mid, 0.7),
            (mem_low, 0.2),
            (mem_top, 0.95),
        ]

        ranked = rank_memories(candidates, now=now)
        assert len(ranked) == 3
        assert ranked[0].memory.id == mem_top.id
        assert ranked[1].memory.id == mem_mid.id
        assert ranked[2].memory.id == mem_low.id

    def test_rank_memories_with_limit(self) -> None:
        now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
        mems = [
            (
                MemoryItem(
                    user_id="u",
                    content=f"Mem {i}",
                    created_at=now,
                ),
                float(i) / 10.0,
            )
            for i in range(5)
        ]
        ranked = rank_memories(mems, now=now, limit=2)
        assert len(ranked) == 2
        assert ranked[0].similarity == 0.4
        assert ranked[1].similarity == 0.3

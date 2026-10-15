"""Memory decay engine modeling human-like retention and retrieval-frequency reinforcement."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from memlayer.llm.embedder import BaseEmbedder
from memlayer.models import MemoryStatus
from memlayer.store.sql import SQLStore
from memlayer.store.vector_base import VectorPoint, VectorStoreBase

logger = logging.getLogger(__name__)


def calculate_decayed_importance(
    importance: float,
    created_at: datetime,
    last_accessed_at: datetime | None = None,
    access_count: int = 0,
    now: datetime | None = None,
    tau_days: float = 30.0,
    beta: float = 0.15,
) -> tuple[float, float]:
    """Calculate decayed importance and elapsed days from temporal anchor.

    Args:
        importance: Baseline importance score in [0.0, 1.0].
        created_at: Memory creation timestamp.
        last_accessed_at: Most recent retrieval timestamp if accessed.
        access_count: Number of times memory has been retrieved.
        now: Current evaluation time (defaults to UTC now).
        tau_days: Half-life decay timescale in days.
        beta: Logarithmic frequency boost coefficient.

    Returns:
        tuple[float, float]: (new_importance in [0.01, 1.0], elapsed_days).
    """
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)

    anchor = last_accessed_at if last_accessed_at is not None else created_at
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)

    elapsed_seconds = max(0.0, (current_time - anchor).total_seconds())
    elapsed_days = elapsed_seconds / 86400.0

    # Inactivity decay factor
    decay_factor = math.exp(-elapsed_days / max(0.001, tau_days))

    # Retrieval frequency logarithmic reinforcement
    frequency_boost = 1.0 + beta * math.log(1.0 + max(0, access_count))

    raw_val = float(importance) * decay_factor * frequency_boost
    new_importance = min(1.0, max(0.01, round(raw_val, 4)))

    return new_importance, round(elapsed_days, 2)


class DecayItemResult(BaseModel):
    """Result of running decay evaluation on an individual memory."""

    memory_id: str
    old_importance: float
    new_importance: float
    access_count: int
    elapsed_days: float
    pruned: bool


class DecayResult(BaseModel):
    """Aggregated result of executing memory decay consolidation across a user."""

    user_id: str
    processed_count: int
    updated_count: int
    pruned_count: int
    items: list[DecayItemResult] = Field(default_factory=list)


class MemoryDecayEngine:
    """Manages memory lifecycle decay, frequency reinforcement, and soft-pruning.

    Lowers the importance of inactive memories over time while protecting and
    boosting memories with recurring retrieval citations.
    """

    def __init__(
        self,
        store: SQLStore,
        vector_store: VectorStoreBase,
        embedder: BaseEmbedder,
        tau_days: float = 30.0,
        beta: float = 0.15,
        prune_threshold: float = 0.05,
    ) -> None:
        """Initialize the memory decay engine."""
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder
        self.tau_days = tau_days
        self.beta = beta
        self.prune_threshold = prune_threshold

    async def decay_user_memories(
        self,
        user_id: str,
        now: datetime | None = None,
        prune_neglect_days: float | None = None,
    ) -> DecayResult:
        """Evaluate all active memories for user_id, applying decay and soft pruning.

        Args:
            user_id: Scoped owner identifier.
            now: Optional fixed evaluation timestamp (useful for testing).
            prune_neglect_days: Days of inactivity required before pruning (default: 2 * tau_days).

        Returns:
            DecayResult summarising evaluated, updated, and soft-pruned memories.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        clean_user_id = user_id.strip()

        eval_time = now or datetime.now(UTC)
        neglect_cutoff = (
            prune_neglect_days if prune_neglect_days is not None else (2.0 * self.tau_days)
        )

        active_memories = await self.store.list_memories(
            user_id=clean_user_id,
            status=MemoryStatus.ACTIVE,
            limit=500,
        )

        if not active_memories:
            return DecayResult(
                user_id=clean_user_id,
                processed_count=0,
                updated_count=0,
                pruned_count=0,
                items=[],
            )

        updated_items: list[DecayItemResult] = []
        updated_count = 0
        pruned_count = 0

        for mem in active_memories:
            new_importance, elapsed_days = calculate_decayed_importance(
                importance=mem.importance,
                created_at=mem.created_at,
                last_accessed_at=mem.last_accessed_at,
                access_count=mem.access_count,
                now=eval_time,
                tau_days=self.tau_days,
                beta=self.beta,
            )

            should_prune = new_importance < self.prune_threshold and elapsed_days >= neglect_cutoff

            if should_prune:
                # Soft prune: Transition status to SUPERSEDED with marker pointer
                await self.store.mark_superseded(
                    user_id=clean_user_id,
                    old_memory_id=mem.id,
                    new_memory_id="pruned_decay",
                )

                # Update Vector Store point payload
                vec = await self.embedder.embed_text(mem.content)
                payload = {
                    "content": mem.content,
                    "entity": mem.entity,
                    "importance": new_importance,
                    "created_at": mem.created_at.isoformat(),
                    "event_time": (mem.event_time.isoformat() if mem.event_time else None),
                    "status": MemoryStatus.SUPERSEDED.value,
                    "superseded_by": "pruned_decay",
                }
                await self.vector_store.upsert(
                    user_id=clean_user_id,
                    points=[VectorPoint(id=mem.id, vector=vec, payload=payload)],
                )

                pruned_count += 1
                updated_items.append(
                    DecayItemResult(
                        memory_id=mem.id,
                        old_importance=mem.importance,
                        new_importance=new_importance,
                        access_count=mem.access_count,
                        elapsed_days=elapsed_days,
                        pruned=True,
                    )
                )

            elif not math.isclose(new_importance, mem.importance, abs_tol=1e-4):
                # Update SQL importance
                await self.store.update_memory_importance(
                    user_id=clean_user_id,
                    memory_id=mem.id,
                    importance=new_importance,
                )

                # Update Vector Store payload importance
                vec = await self.embedder.embed_text(mem.content)
                payload = {
                    "content": mem.content,
                    "entity": mem.entity,
                    "importance": new_importance,
                    "created_at": mem.created_at.isoformat(),
                    "event_time": (mem.event_time.isoformat() if mem.event_time else None),
                    "status": MemoryStatus.ACTIVE.value,
                    "superseded_by": None,
                }
                await self.vector_store.upsert(
                    user_id=clean_user_id,
                    points=[VectorPoint(id=mem.id, vector=vec, payload=payload)],
                )

                updated_count += 1
                updated_items.append(
                    DecayItemResult(
                        memory_id=mem.id,
                        old_importance=mem.importance,
                        new_importance=new_importance,
                        access_count=mem.access_count,
                        elapsed_days=elapsed_days,
                        pruned=False,
                    )
                )

        return DecayResult(
            user_id=clean_user_id,
            processed_count=len(active_memories),
            updated_count=updated_count,
            pruned_count=pruned_count,
            items=updated_items,
        )

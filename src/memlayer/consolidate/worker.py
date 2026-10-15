"""Background consolidation worker coordinating periodic and on-demand sweeps via APScheduler."""

from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel, Field

from memlayer.consolidate.decay import DecayResult, MemoryDecayEngine
from memlayer.consolidate.dedup import DeduplicationEngine, DedupResult
from memlayer.store.sql import SQLStore

logger = logging.getLogger(__name__)


class FullConsolidationResult(BaseModel):
    """Aggregate outcome of executing full consolidation (deduplication and decay)."""

    dedup_results: list[DedupResult] = Field(default_factory=list)
    decay_results: list[DecayResult] = Field(default_factory=list)
    total_duplicates_merged: int = 0
    total_memories_decayed: int = 0
    total_memories_pruned: int = 0


class ConsolidationWorker:
    """Manages periodic asynchronous background maintenance sweeps for durable long-term memory.

    Coordinates deduplication of near-paraphrase memories and exponential age decay with
    frequency reinforcement, ensuring high recall quality without blocking interactive user turns.
    """

    def __init__(
        self,
        store: SQLStore,
        dedup_engine: DeduplicationEngine,
        decay_engine: MemoryDecayEngine,
        dedup_interval_minutes: float = 360.0,
        decay_interval_minutes: float = 1440.0,
        scheduler: AsyncIOScheduler | None = None,
    ) -> None:
        """Initialize the consolidation worker."""
        self.store = store
        self.dedup_engine = dedup_engine
        self.decay_engine = decay_engine
        self.dedup_interval_minutes = max(0.01, dedup_interval_minutes)
        self.decay_interval_minutes = max(0.01, decay_interval_minutes)
        self._scheduler = scheduler or AsyncIOScheduler()
        self._is_started = False

    @property
    def is_running(self) -> bool:
        """Return True if background scheduler is active and running."""
        return bool(self._scheduler and self._scheduler.running)

    def start(self) -> None:
        """Register scheduled jobs and start the background scheduler."""
        if self.is_running:
            return

        self._scheduler.add_job(
            self.run_dedup,
            "interval",
            minutes=self.dedup_interval_minutes,
            id="memlayer_consolidation_dedup",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self.run_decay,
            "interval",
            minutes=self.decay_interval_minutes,
            id="memlayer_consolidation_decay",
            replace_existing=True,
        )

        self._scheduler.start()
        self._is_started = True
        logger.info(
            "ConsolidationWorker started with dedup every %s min, decay every %s min",
            self.dedup_interval_minutes,
            self.decay_interval_minutes,
        )

    def stop(self, wait: bool = False) -> None:
        """Gracefully shutdown the scheduler."""
        if self.is_running:
            self._scheduler.shutdown(wait=wait)
            self._is_started = False
            logger.info("ConsolidationWorker stopped.")

    def pause(self) -> None:
        """Suspend execution of scheduled consolidation sweeps."""
        if self.is_running:
            self._scheduler.pause()

    def resume(self) -> None:
        """Resume execution of paused consolidation sweeps."""
        if self.is_running:
            self._scheduler.resume()

    async def _resolve_target_users(self, user_id: str | None = None) -> list[str]:
        """Resolve targeted user IDs, discovering active tenants if user_id is None."""
        if user_id and user_id.strip():
            return [user_id.strip()]
        return await self.store.list_user_ids()

    async def run_dedup(self, user_id: str | None = None) -> list[DedupResult]:
        """Execute near-duplicate fact merging across targeted or all active users."""
        targets = await self._resolve_target_users(user_id)
        results: list[DedupResult] = []

        for uid in targets:
            try:
                res = await self.dedup_engine.deduplicate_user(uid)
                results.append(res)
            except Exception as exc:
                logger.error("Deduplication sweep failed for user '%s': %s", uid, exc)

        return results

    async def run_decay(self, user_id: str | None = None) -> list[DecayResult]:
        """Execute memory decay and soft-pruning across targeted or all active users."""
        targets = await self._resolve_target_users(user_id)
        results: list[DecayResult] = []

        for uid in targets:
            try:
                res = await self.decay_engine.decay_user_memories(uid)
                results.append(res)
            except Exception as exc:
                logger.error("Decay sweep failed for user '%s': %s", uid, exc)

        return results

    async def run_full_consolidation(self, user_id: str | None = None) -> FullConsolidationResult:
        """Execute both deduplication and decay sweeps sequentially."""
        dedup_res = await self.run_dedup(user_id=user_id)
        decay_res = await self.run_decay(user_id=user_id)

        merged_total = sum(r.merged_count for r in dedup_res)
        decayed_total = sum(r.updated_count for r in decay_res)
        pruned_total = sum(r.pruned_count for r in decay_res)

        return FullConsolidationResult(
            dedup_results=dedup_res,
            decay_results=decay_res,
            total_duplicates_merged=merged_total,
            total_memories_decayed=decayed_total,
            total_memories_pruned=pruned_total,
        )

# Spec 008: Background Consolidation Worker

## 1. Goal
Implement the background consolidation worker for Milestone 4 (Phase 4.4) in `src/memlayer/consolidate/worker.py`. The worker coordinates asynchronous, periodic background sweeps for deduplication (`DeduplicationEngine`) and memory decay (`MemoryDecayEngine`) using `APScheduler` without blocking user-facing chat interactions.

---

## 2. Requirements & Design

### 2.1 Scheduler Engine Architecture (`ConsolidationWorker`)
- Built on top of `APScheduler`'s `AsyncIOScheduler`.
- Supports configurable intervals:
  - `dedup_interval_minutes: float` (default: 360.0 minutes / 6 hours).
  - `decay_interval_minutes: float` (default: 1440.0 minutes / 24 hours).
- Provides lifecycle control:
  - `start()`: Initializes and starts the background scheduler and registers jobs.
  - `stop()`: Gracefully halts background tasks (`wait=False`).
  - `pause()` / `resume()`: Suspends and reactivates scheduled jobs.
  - `is_running: bool`: Property inspecting current active scheduler state.
- Supports on-demand manual triggers:
  - `run_dedup(user_id: str | None = None) -> list[DedupResult]`
  - `run_decay(user_id: str | None = None) -> list[DecayResult]`
  - `run_full_consolidation(user_id: str | None = None) -> FullConsolidationResult`

### 2.2 User Discovery & Tenant Isolation
- If `user_id` is supplied to a consolidation run, execution is strictly confined to that user.
- If `user_id` is `None` (full periodic background sweep):
  - Discovers distinct user IDs with active records via `SQLStore.list_user_ids()`.
  - Executes consolidation on each user independently within their isolated namespace.
  - Errors during one user's consolidation are logged and isolated; they never abort the sweep for other users.

### 2.3 Public API Integration (`MemoryLayer`)
- Expose optional background worker controls on `MemoryLayer`:
  - `start_consolidation_worker(...)`
  - `stop_consolidation_worker()`
  - Automatically shuts down worker on `await mem.close()`.

---

## 3. Data Flow

```
APScheduler (Cron / Interval Trigger) OR Manual trigger_now()
                           │
                           ▼
               Discover Active User IDs
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      User A (Isolated)           User B (Isolated)
             │                           │
   1. Deduplication Sweep      1. Deduplication Sweep
   2. Decay & Pruning Sweep    2. Decay & Pruning Sweep
```

---

## 4. Acceptance Criteria
1. **Lifecycle Management:** Worker cleanly starts, pauses, resumes, and stops without hanging event loops.
2. **Periodic & Manual Sweeps:** Successfully executes dedup and decay sweeps via scheduled jobs and direct manual calls.
3. **Tenant Fault Tolerance:** An exception in one user's consolidation does not crash the background worker or affect other users.
4. **Public API Lifecycle:** `MemoryLayer.close()` gracefully terminates active background schedulers.
5. **Quality Gates:** 100% test coverage in `tests/test_worker.py`, passes `ruff`, `mypy`, and `pytest`.

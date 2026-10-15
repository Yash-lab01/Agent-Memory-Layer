# Spec 007: Memory Decay & Explicit Forgetting

## 1. Goal
Implement the memory decay and explicit forgetting lifecycle for Milestone 4 (Phase 4.3). This encompasses:
1. Retrieval-frequency-aware decay in `src/memlayer/consolidate/decay.py`, which decays stale, unaccessed facts over time while reinforcing frequently recalled memories.
2. Verified explicit deletion APIs (`MemoryLayer.forget` and `MemoryLayer.forget_all`) fulfilling privacy, GDPR, and hard-erasure requirements.

---

## 2. Requirements & Design

### 2.1 Retrieval-Frequency Memory Decay Model
Conversational agents accumulate observations that lose relevance if never re-referenced. The decay engine models human-like memory retention:
- **Elapsed Inactivity ($\Delta t$):**
  $$\Delta t_{\text{days}} = \frac{t_{\text{now}} - \max(t_{\text{last\_accessed}}, t_{\text{created}})}{86400}$$
- **Exponential Inactivity Decay:**
  $$\text{decay\_factor} = \exp\left(-\frac{\Delta t_{\text{days}}}{\tau_{\text{decay}}}\right)$$
  where $\tau_{\text{decay}}$ is the decay half-life in days (default: 30.0 days).
- **Retrieval Frequency Reinforcement:**
  Facts that are frequently retrieved across conversation sessions resist decay:
  $$\text{frequency\_boost} = 1.0 + \beta \cdot \log(1 + \text{access\_count})$$
  where $\beta$ is the retrieval reinforcement coefficient (default: 0.15).
- **Updated Importance Formula:**
  $$I_{\text{new}} = \min\left(1.0, \max(0.01, I_{\text{base}} \cdot \text{decay\_factor} \cdot \text{frequency\_boost})\right)$$
- **Pruning Policy:**
  Memories whose decayed importance falls below `prune_threshold` (default: 0.05) after prolonged neglect ($\Delta t_{\text{days}} \ge 2 \cdot \tau_{\text{decay}}$) are soft-pruned: marked as `MemoryStatus.SUPERSEDED` (never hard-deleted during background consolidation).

### 2.2 Storage & Vector Synchronization
- **Relational Store (`SQLStore`):**
  - Added method: `update_memory_importance(user_id: str, memory_id: str, new_importance: float) -> bool`.
  - Soft-pruned memories transitioned via `mark_superseded(user_id, memory_id, new_memory_id="pruned_decay")`.
- **Vector Store (`VectorStoreBase`):**
  - Updated payloads: `importance = new_importance` (and `status = "superseded"` if soft-pruned).

### 2.3 Explicit Forgetting (Privacy & GDPR Compliance)
- As specified in `AGENTS.md` Section 4.4: *Explicit deletion (`forget`) is the only hard delete.*
- `MemoryLayer.forget(user_id: str, memory_id: str) -> bool`:
  - Hard-deletes the row from `SQLStore.delete_memory(user_id, memory_id)`.
  - Purges the embedding point from `VectorStoreBase.delete(user_id, [memory_id])`.
  - Strictly scoped by `user_id`.
- `MemoryLayer.forget_all(user_id: str) -> int`:
  - Hard-deletes all episodes and memories for `user_id` across relational and vector stores.

---

## 3. Data Flow

```
Active Memories (user_id) ──> Calculate Elapsed Inactivity (days)
                                          │
                                          ▼
                      Compute Decayed Importance with Frequency Boost
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
              [I_new >= prune_threshold]         [I_new < prune_threshold]
                         │                                 │
            SQL: update_memory_importance       SQL: mark_superseded
            Vector: payload.importance          Vector: payload.status = "superseded"
```

---

## 4. Acceptance Criteria
1. **Mathematical Accuracy:** Decay formula matches exponential decay with logarithmic frequency boost; tested across boundary conditions ($\Delta t = 0$, large $\Delta t$, high access counts).
2. **Storage Invariants:** Non-destructive soft pruning during decay; hard delete strictly confined to explicit `forget` / `forget_all` calls.
3. **Multi-User Isolation:** Decay and forget operations strictly verify `user_id`; never affect other tenants.
4. **Quality Gates:** 100% test coverage in `tests/test_decay.py`, passes `ruff`, `mypy`, and `pytest`.

# Spec 006: Deduplication Engine (Near-Duplicate Merging)

## 1. Goal
Implement the deduplication engine for Milestone 4 (Phase 4.2) in `src/memlayer/consolidate/dedup.py`. The engine identifies near-duplicate or redundant active memories belonging to the same user, merges them into a canonical representative, and non-destructively marks redundant records as `MemoryStatus.SUPERSEDED` with `superseded_by` pointing to the canonical memory.

---

## 2. Requirements & Design

### 2.1 Near-Duplicate Definition & Distinction
- **Near-Duplicate:** Two active facts that state essentially the same information with syntactic or minor lexical variations:
  - *"User lives in Pune"* vs. *"User resides in Pune, India"*
  - *"User works at Google DeepMind as an AI researcher"* vs. *"User is an AI researcher at Google DeepMind"*
  - Exact duplicates ingested from repeated dialogue turns.
- **Distinction from Contradiction (Spec 005):**
  - Contradiction: New state invalidates old state (e.g., moved from Pune to Bangalore). Handled by `ConflictDetector`.
  - Deduplication: Both memories assert the same ongoing reality. Handled by `DeduplicationEngine`.
- **Distinction from Compatible Coexistence:**
  - Distinct facts with high topical similarity (e.g., "enjoys black coffee" vs. "enjoys chamomile tea") are NOT duplicates and must remain separate.

### 2.2 Deduplication Architecture (`DeduplicationEngine`)
- **Per-User Scoping:** Deduplication strictly executes within a specified `user_id`. Never compares across tenants.
- **Candidate Discovery:**
  - Retrieve all `MemoryStatus.ACTIVE` memories for the user.
  - Candidate clustering: Group memories by entity, or compare pairs with high embedding cosine similarity ($\ge \text{similarity\_threshold}$, default $0.88$).
- **Semantic Classification & Synthesis:**
  - For candidate pairs:
    - If normalized cosine similarity $\ge 0.98$ or exact textual match, classify as duplicate without LLM call.
    - Otherwise, use lightweight LLM verification (or pluggable `completion_fn`) to confirm semantic equivalence and distinguish from compatible coexisting details.
- **Canonical Selection Policy:**
  - Among duplicate candidates, select the canonical memory:
    - Prefers the more detailed/longer text, or the earlier memory with higher provenance / access count.
  - Reinforce canonical memory:
    - Increments access telemetry (`record_access`) on the canonical memory to reflect reinforcement.
  - Non-destructive supersession:
    - Redundant duplicate is marked `MemoryStatus.SUPERSEDED` in `SQLStore` with `superseded_by = canonical_memory.id`.
    - Redundant duplicate point in `vector_store` updated with `status="superseded"` and `superseded_by=canonical_memory.id`.

### 2.3 Non-Destructive Invariant
- Never hard-delete duplicate rows from SQL or vector store.
- Audit trail, source episodes, and historical timestamps remain preserved.
- Active searches (`status=MemoryStatus.ACTIVE`) will return only the canonical memory, eliminating redundant context injection and saving token budget.

---

## 3. Data Flow

```
User Active Memories ──> Pairwise Similarity Clustering (cosine >= 0.88)
                                    │
                                    ▼
                         Semantic Verification (LLM / Fast Heuristic)
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                   [Duplicate]            [Distinct]
                         │                     │
       1. Identify Canonical Memory        No changes
       2. Reinforce Canonical (record_access)
       3. SQL: duplicate.status = SUPERSEDED, superseded_by = canonical.id
       4. Vector: duplicate.status = "superseded", superseded_by = canonical.id
```

---

## 4. Acceptance Criteria
1. **Accurate Detection:** Correctly identifies exact and near-paraphrase duplicates while preserving distinct coexisting facts.
2. **Non-Destructive Invariant:** No memories are deleted from SQL during deduplication.
3. **Audit Trail:** Superseded duplicate maintains exact `superseded_by` link to the canonical memory.
4. **Tenant Isolation:** Deduplication strictly operates per `user_id`. Cannot merge or supersede across different users.
5. **Quality Gates:** 100% test coverage in `tests/test_dedup.py`, passes `ruff`, `mypy`, and `pytest`.

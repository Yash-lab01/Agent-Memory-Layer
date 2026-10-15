# Spec 005: Contradiction & Supersession Engine (Test-Driven)

## 1. Goal
Implement the contradiction and supersession resolution engine for Milestone 4 (Phase 4.1) in `src/memlayer/consolidate/conflicts.py`. The engine identifies when a newly ingested fact invalidates or updates an existing active memory, marking the outdated memory as `MemoryStatus.SUPERSEDED` with a pointer to `superseded_by` without ever hard-deleting records.

---

## 2. Requirements & Design

### 2.1 Test-Driven Fixtures (`tests/fixtures/contradictions.json`)
Before implementing detection logic, establish a standardized evaluation dataset containing:
- Direct location changes ("lives in Pune" vs. "moved to Bangalore").
- Employment/career changes ("works at Startup A" vs. "accepted an offer at Tech Giant B").
- Preference shifts & negations ("loves spicy food" vs. "cannot eat spicy food anymore due to allergies").
- Tooling/setup updates ("uses VS Code" vs. "switched to Cursor as daily driver").
- Coexisting traits / non-conflicts ("enjoys coffee" vs. "drinks green tea", "has a cat" vs. "has a dog").

### 2.2 Conflict Detection Architecture (`ConflictDetector`)
- **Candidate Retrieval:** Identifies candidate prior memories belonging to the same `user_id` using entity matching and semantic similarity filtering ($> 0.6$).
- **Structured NLI Prompting:** Uses LLM with Pydantic JSON validation to classify relationship:
  - `CONTRADICTION` / `SUPERSEDES`: New fact updates or invalidates old fact.
  - `COMPATIBLE`: Both facts remain true simultaneously.
- **Pluggable Completion Function:** Allows passing custom `completion_fn` for fast, offline unit testing without external network calls.

### 2.3 Non-Destructive Supersession Lifecycle
- When contradiction is confirmed:
  1. Relational store updates target: `status = MemoryStatus.SUPERSEDED`, `superseded_by = new_memory.id`.
  2. Vector store updates payload: `status = "superseded"`.
  3. Original records, timestamps, and episode provenance remain intact in the database.
  4. Search queries with default `status=MemoryStatus.ACTIVE` will no longer surface the superseded memory.

---

## 3. Data Flow

```
New MemoryItem ──> Candidate Lookup (Vector / Entity Filter) ──> Conflict Detector (LLM / NLI)
                                                                           │
                                              ┌────────────────────────────┴──────────────────────────┐
                                              ▼                                                       ▼
                                      [Contradiction]                                            [Compatible]
                                              │                                                       │
                           SQL: status=SUPERSEDED, superseded_by=new_id                            No changes
                           Vector: status=superseded
```

---

## 4. Acceptance Criteria
1. **Fixture Accuracy:** Accurately classifies all test cases in `tests/fixtures/contradictions.json`.
2. **Non-Destructive Invariant:** No memories are deleted from SQL during supersession.
3. **Audit Trail:** Superseded memory maintains exact `superseded_by` link to replacing memory.
4. **Tenant Isolation:** Conflicts can only be detected and resolved within the same `user_id`.
5. **Quality Gates:** 100% test coverage in `tests/test_conflicts.py`, passes `ruff`, `mypy`, and `pytest`.

# Spec 004: Retrieval Router, Context Formatter & Public Search API

## 1. Goal
Complete the Memory Read Path for Milestone 3 (Phases 3.3 & 3.4) by providing:
1. **Retrieval Router (`src/memlayer/retrieve/router.py`):** Fast, zero-latency turn classifier that skips expensive retrieval on trivial turns ("thanks!", "ok", "hello").
2. **Context Formatter (`src/memlayer/retrieve/format.py`):** Formats scored memories with timestamps, provenance, and entities within an explicit token budget.
3. **Public Search API (`MemoryLayer.search` in `src/memlayer/api.py`):** Coordinates turn gating, hybrid search, multi-factor scoring, token-budgeted prompt injection, and access tracking.

---

## 2. Requirements & Specifications

### 2.1 Retrieval Router (`RetrievalRouter`)
- Evaluates whether a user's dialogue turn necessitates memory retrieval.
- **Skip Conditions (`should_retrieve(query) -> False`):**
  - Greetings & pleasantries (e.g., "hi", "hello", "good morning", "hey there").
  - Gratitude & acknowledgements (e.g., "thanks", "thank you", "great", "ok", "got it", "cool").
  - Short confirmations / negations (e.g., "yes", "no", "sure", "yep", "nope").
  - Punctuation-only or blank queries.
- **Retrieve Conditions (`should_retrieve(query) -> True`):**
  - Informational questions ("what is my...", "where do I live?", "who is...").
  - Statements containing specific entities, complex queries, or requests.
- **Bypass Option:** `bypass_router: bool = True` overrides the router when the caller mandates retrieval.

### 2.2 Context Formatter (`format_context`)
- Converts a list of `ScoredMemory` objects into a clean prompt-injection context string.
- Each line formatted with date provenance:
  `[YYYY-MM-DD] Entity: Content`
- **Token Budget Packing:**
  - Given `token_budget: int | None`, includes top-ranked memories in descending order of composite score until adding another memory would exceed `token_budget`.
  - Estimates token count accurately (~4 characters per token heuristic / word splitting).
  - Returns tuple of `(context_str, included_memories, total_token_count)`.

### 2.3 Public Search API (`MemoryLayer.search`)
```python
async def search(
    self,
    user_id: str,
    query: str,
    limit: int = 10,
    token_budget: int | None = None,
    bypass_router: bool = False,
    status: MemoryStatus | None = MemoryStatus.ACTIVE,
) -> SearchResult:
```
- Returns `SearchResult(context=..., memories=..., token_count=..., skipped=...)`.
- Records retrieval access count and timestamp asynchronously on returned memories via `SQLStore.record_access`.
- Enforces strict tenant isolation: queries can never return another user's memories.

---

## 3. Acceptance Criteria
1. **Turn Gating:** Trivial turns return `SearchResult(..., skipped=True, context="", memories=[])` with 0 database queries.
2. **Token Budget Enforcement:** Context never exceeds `token_budget` when specified.
3. **End-to-End Pipeline:** User turns requiring memory retrieve relevant facts, format them with timestamps, and return valid `SearchResult`.
4. **Access Telemetry:** Retrieved memories update `access_count` in the database.
5. **Quality Gates:** 100% test coverage for router, formatter, and search API; passes `ruff`, `mypy`, and `pytest`.

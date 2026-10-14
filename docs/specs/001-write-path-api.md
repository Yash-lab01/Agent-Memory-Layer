# Spec 001: Public Write API and Asynchronous Execution Pipeline

## 1. Goal
Provide the primary developer-facing interface `MemoryLayer` in `src/memlayer/api.py`, implementing the write path `MemoryLayer.add(user_id, messages)` that coordinates:
1. Fast conversation logging for durable provenance (`EpisodeLogger`).
2. Atomic fact extraction with schema validation and retry (`FactExtractor`).
3. Embedding generation (`BaseEmbedder`).
4. Relational fact persistence (`SQLMemoryStore`).
5. Vector indexing (`VectorStoreBase`).
6. Optional background execution (`asyncio.create_task`) for non-blocking conversational flows.

---

## 2. Inputs & Outputs

### 2.1 Inputs to `MemoryLayer.add`
- `user_id: str`: Unique user identifier (enforces per-user tenant isolation).
- `messages: Sequence[dict[str, Any] | Episode]`: Conversation turns to ingest (e.g., `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`).
- `session_id: str | None = None`: Optional conversation session ID for grouping.
- `background: bool = False`: If `True`, runs fact extraction and vector indexing asynchronously in a background `asyncio.Task`, returning immediately after logging episodes. If `False` (default for synchronous awaiting and testing), awaits the full ingestion pipeline.

### 2.2 Outputs
- When `background=False`: Returns `list[MemoryItem]` containing the newly created and indexed memories.
- When `background=True`: Returns `asyncio.Task[list[MemoryItem]]` representing the ongoing ingestion job.

---

## 3. Architecture & Data Flow

```
[Agent] ───> mem.add(user_id, messages, background=False/True)
                │
                ├─ 1. Log Raw Messages (Fast Sync/Await) ──> EpisodeLogger (SQL episodes table)
                │
                ├─ [If background=True, launch asyncio.Task and return Task handle]
                │
                ├─ 2. Extract Atomic Facts ───────────────> FactExtractor (LiteLLM)
                │
                ├─ 3. Generate Vector Embeddings ─────────> BaseEmbedder
                │
                ├─ 4. Persist Structured Memories ────────> SQLMemoryStore (SQL memories table)
                │
                └─ 5. Index into Vector Store ────────────> VectorStoreBase (Qdrant / Postgres)
```

---

## 4. Acceptance Criteria

1. **Strict User Scoping:**
   - Episodes, SQL memories, and vector embeddings are stored strictly under `user_id`.
   - `add` will reject empty or whitespace-only `user_id` with `ValueError`.
2. **Provenance Tracking:**
   - Raw dialogue is stored in the `episodes` table before fact extraction.
   - Each created `MemoryItem` references the triggering episode ID (`source_episode_id`).
3. **Pluggable & Defaultable Components:**
   - Users can provide custom `store`, `vector_store`, `embedder`, `extractor`, `episode_logger`.
   - Sensible local-first defaults (`SQLMemoryStore.create_in_memory()`, `MockEmbedder()`, in-memory `QdrantVectorStore()`, `FactExtractor()`) are instantiated if not provided.
4. **Resilience & Graceful Handling:**
   - If extraction returns zero facts (e.g. empty message or chit-chat), `add()` logs episodes cleanly and returns an empty list without error.
5. **Testing & Verification:**
   - Unit and integration tests in `tests/test_write_pipeline.py` verifying:
     - End-to-end ingestion with mock LLM and mock embedder.
     - Provenance linking (`source_episode_id`).
     - Multi-user isolation (User A memories not visible to User B).
     - Background ingestion task completion.
     - All quality checks pass (`ruff check`, `ruff format --check`, `mypy`, `pytest`).

---

## 5. Out of Scope for Phase 2.2
- Read path (`search()`), scoring formula, and retrieval router (Milestone 3).
- Contradiction / supersession resolution and dedup merging (Milestone 4).
- APScheduler background sweep worker (Milestone 4).

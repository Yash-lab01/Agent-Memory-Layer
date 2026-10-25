# Python API Reference

## `MemoryLayer`

The central entrypoint coordinating storage, extraction, retrieval, and consolidation.

```python
from memlayer import MemoryLayer
```

---

### Constructor

```python
mem = MemoryLayer(
    store: SQLStore,
    vector_store: VectorStoreBase,
    embedder: BaseEmbedder,
    extractor: FactExtractor | None = None,
    episode_logger: EpisodeStore | None = None,
    router: RetrievalRouter | None = None,
    retriever: HybridRetriever | None = None,
    scoring_weights: ScoringWeights | None = None,
    tracer: BaseTracer | None = None,
)
```

#### Factory Method: `create_local`
Quickly initialize a self-contained local instance (SQLite in-memory + Qdrant in-memory):

```python
mem = await MemoryLayer.create_local(
    db_url: str = "sqlite+aiosqlite:///:memory:",
    vector_dim: int = 64,
    embedder: BaseEmbedder | None = None,
    tracer: BaseTracer | None = None,
) -> MemoryLayer
```

---

### Methods

#### `add`
Ingests dialogue turns, persists episodes, extracts facts asynchronously, and updates supersession links.

```python
await mem.add(
    user_id: str,
    messages: Sequence[dict[str, Any] | Episode],
    session_id: str | None = None,
    background: bool = False,
) -> list[MemoryItem] | asyncio.Task[list[MemoryItem]]
```

#### `search`
Searches relevant memories using hybrid vector+BM25 retrieval, multi-factor scoring, and token budgeting.

```python
result: SearchResult = await mem.search(
    user_id: str,
    query: str,
    limit: int = 10,
    token_budget: int | None = None,
    bypass_router: bool = False,
    status: MemoryStatus | None = MemoryStatus.ACTIVE,
)
```

`SearchResult` properties:
- `context: str`: Formatted markdown context ready for system prompt injection.
- `memories: list[ScoredMemory]`: Retrieved memory items with relevance scores.
- `token_count: int`: Number of tokens used by the context string.
- `skipped: bool`: True if the router deemed the query non-informative.

#### `forget`
Deletes a single memory by ID across relational and vector indexes.

```python
deleted: bool = await mem.forget(user_id: str, memory_id: str)
```

#### `forget_all`
Purges all memories and episodes for a user (GDPR compliance).

```python
purged_count: int = await mem.forget_all(user_id: str)
```

#### `close`
Gracefully stops consolidation workers, flushes telemetry tracers, and closes storage engines.

```python
await mem.close()
```

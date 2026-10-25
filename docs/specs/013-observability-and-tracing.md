# Spec 013: Observability & Tracing (Langfuse & OpenTelemetry)

## 1. Goal
Equip the **Agent Memory Layer (`memlayer`)** with observability and distributed tracing. Provide deep visibility into:
1. Memory retrieval cycles (routing decisions, hybrid BM25 vs dense scoring breakdown, recency decay impact, token budgeting).
2. Memory ingestion cycles (episode persistence, fact extraction latency, embedding generation, conflict detection, supersession events).
3. Lifecycle consolidation (deduplication merges, access frequency decay sweeps).

---

## 2. Architecture & Design (`src/memlayer/tracing.py`)

### 2.1 Pluggable Tracing Backend (`MemoryTracer`)
`MemoryTracer` provides a unified telemetry interface:
- **Langfuse Support:** When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are provided (or via local Langfuse instance), traces and spans are published directly to Langfuse with user tagging (`user_id`), session tagging (`session_id`), input/output payloads, and latency tracking.
- **OpenTelemetry Support:** Compatible with standard OTel tracers when OTLP endpoints are configured.
- **In-Memory / Test Mode:** Stores traces in an internal record buffer for unit testing and CI verification without requiring external network connections.
- **Null / No-op Mode:** If tracing is disabled or unconfigured, decorators and context managers act as transparent zero-overhead passthroughs.

### 2.2 Core Traced Operations

1. **Retrieval Traces (`memlayer.search`):**
   - Trace name: `memory_search`
   - Attributes: `user_id`, `query`, `token_budget`, `limit`
   - Child Spans:
     - `router`: Routing evaluation (`should_retrieve` decision and rationale).
     - `hybrid_retriever`: Dense vector search and BM25 sparse matching.
     - `scorer`: Multi-factor score computation ($\alpha \cdot sim + \beta \cdot recency + \gamma \cdot importance$).
     - `formatter`: Prompt context budgeting and token truncation.
   - Outputs: `token_count`, `retrieved_count`, `context_snippet`.

2. **Ingestion Traces (`memlayer.add`):**
   - Trace name: `memory_add`
   - Attributes: `user_id`, `session_id`, `message_count`
   - Child Spans:
     - `episode_logger`: Relational turn persistence.
     - `fact_extractor`: LLM atomic fact extraction.
     - `embedder`: Vector embedding generation.
     - `vector_upsert`: Vector database ingestion.
     - `conflict_detector`: Contradiction analysis and supersession updates.
   - Outputs: `extracted_facts_count`, `superseded_count`.

3. **Consolidation Traces (`memlayer.consolidate`):**
   - Trace name: `memory_consolidation`
   - Child Spans: `deduplication_sweep`, `decay_sweep`.

---

## 3. Docker Compose Langfuse Integration (`docker-compose.yml`)

Add a self-hosted Langfuse stack to `docker-compose.yml`:
- `langfuse-server`: Official Langfuse container exposed at port `3000`.
- Configured with environment variables linking to the existing Postgres service or dedicated database.
- `.env.example` updated with sample configuration (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST=http://localhost:3000`).

---

## 4. Acceptance Criteria

1. **Zero-Overhead Default:** When tracing is not configured, `MemoryLayer` runs with `NullTracer` incurring zero latency penalty.
2. **Captured Metrics:** Traces accurately record wall-clock duration, user isolation tags (`user_id`), query text, extracted facts, and token usage.
3. **End-to-End Tracing:** `MemoryLayer.add()` and `MemoryLayer.search()` produce structured traces and child spans.
4. **Resilience:** Any network error or telemetry failure in Langfuse/OTel is caught and logged as a warning, never failing the core memory read or write path.
5. **Quality & Packaging:**
   - Code passes `ruff`, `mypy`, and full pytest suite (`tests/test_tracing.py`).
   - GitHub Actions CI installs `.[dev,postgres,demo,mcp,tracing]` and runs all tests green.

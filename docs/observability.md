# Observability & Distributed Tracing

`memlayer` includes comprehensive telemetry designed for deep production visibility. Every memory cycle generates structured traces and child spans published to **Langfuse** or compatible OpenTelemetry collectors.

---

## 1. Zero-Overhead Default

When unconfigured, `memlayer` uses `NullTracer` with `NullSpan`, ensuring 0.00ms latency overhead and zero network requests.

---

## 2. Enabling Langfuse

Set the following variables in `.env`:

```bash
MEMLAYER_TRACER=langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

---

## 3. Trace Topology & Spans

### Retrieval Cycles (`memlayer.search`)
- **Root Trace:** `memory_search`
  - Attributes: `user_id`, `query`, `token_budget`, `limit`
- **Child Spans:**
  - `router`: Evaluates whether the turn requires memory injection (`should_retrieve: true/false`).
  - `hybrid_retriever`: Dense vector search and BM25 sparse matching.
  - `scorer`: Multi-factor score computation ($\alpha \cdot sim + \beta \cdot recency + \gamma \cdot importance$).
  - `formatter`: Prompt context budgeting and token truncation.
- **Outputs:** `token_count`, `retrieved_count`, `skipped`.

### Ingestion Cycles (`memlayer.add`)
- **Root Trace:** `memory_add`
  - Attributes: `user_id`, `session_id`, `message_count`, `background`
- **Child Spans:**
  - `episode_logger`: Relational turn persistence.
  - `fact_extractor`: LLM atomic fact extraction.
  - `embedder`: Vector embedding generation.
  - `vector_store`: Vector database upsertion.
- **Outputs:** `extracted_facts_count`.

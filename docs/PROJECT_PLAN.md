# Agent Memory Layer: Project Plan

A reusable long-term memory library for conversational agents, built and judged on measured benchmark results.

---

## 1. Project in one paragraph

Build an open-source Python library plus a small demo app that gives any chat agent durable memory across sessions. It extracts facts from conversations, stores them with time and provenance, retrieves the right ones at the right moment, and keeps the store clean by merging duplicates, decaying stale facts and resolving contradictions. The goal is not "a new product". The goal is a **measured** answer to: *how much does each memory design choice actually improve a conversational agent?* You will benchmark against a full-context baseline and naive RAG, and publish ablations.

**Working name ideas:** `recall`, `mnemo`, `engram-lite`, `longhand`. Check that the name is free on PyPI and GitHub before committing.

## 2. Why this project, for your goals

- Targets **conversational AI and LLM systems** roles directly.
- Breaks your current portfolio pattern (LangGraph + RAG + UI) with a **library + evaluation** story.
- Closes your known gaps: observability, measurable evaluation, test discipline, CI.
- Lets you reuse your **LLM Evaluation Platform** as the benchmark harness, which gives the two projects a joint story.
- Adds new tech: pgvector or Qdrant used for structured memory, background workers, temporal reasoning, optional graph storage, and an MCP server done the standard way.

**Positioning honestly:** Mem0, Letta (formerly MemGPT) and Zep/Graphiti already exist. Do not claim novelty. Check their current docs before you write the README so your comparison is accurate. Your angle is rigor: a transparent design, ablations and reproducible numbers, and a clean small codebase people can read.

---

## 3. Core features (must-have, v1)

### 3.1 Memory write path
- **Fact extraction:** after each turn (or every N turns), a small LLM call extracts atomic, self-contained facts (for example "User lives in Pune", "User is targeting conversational AI roles") as structured JSON validated with Pydantic.
- **Metadata per memory:** entity or subject, content, timestamp (when said), event time (when it happened, if stated), source message id, confidence, importance score, status (`active` or `superseded`).
- **Async by design:** extraction runs after the reply is sent, never blocking the user.

### 3.2 Storage
- **Vector store:** embeddings for semantic lookup (Qdrant, which you already know, or pgvector).
- **Structured store:** SQLite for local mode, Postgres for server mode; holds facts, status, timestamps and supersession links.
- **Raw episode log:** keep the original messages so any memory can be traced to its source (provenance).

### 3.3 Memory read path
- **Hybrid retrieval:** vector similarity + keyword/BM25 + metadata filters.
- **Scoring:** combine relevance, recency and importance, e.g.
  `score = a * similarity + b * exp(-age / tau) + c * importance`
  with the weights tuned on a dev split (a, b, c and tau are hyperparameters you report).
- **Budgeted injection:** retrieve top-k within a token budget and format memories into the prompt with timestamps.
- **Retrieval router:** a cheap check decides whether a turn needs memory at all (skip for "thanks!").

### 3.4 Consolidation and conflict handling
- **Deduplication:** merge near-duplicate facts.
- **Supersession:** when a new fact contradicts an old one ("I moved to Bangalore" vs "lives in Pune"), mark the old one `superseded`, link both, and keep history rather than deleting.
- **Decay:** lower the retrieval weight of old, unused memories; boost memories that keep being retrieved.
- **Background job:** runs on a schedule or after N writes.

### 3.5 Developer interface
- A small, clean Python API:
  ```python
  mem = MemoryLayer(store=..., llm=..., embedder=...)
  mem.add(user_id, messages)  # extract + store
  ctx = mem.search(user_id, query)  # retrieve
  mem.forget(user_id, memory_id)  # delete
  ```
- Pluggable backends: LLM (Ollama, Groq, Gemini via LiteLLM), embedder, store.
- Per-user isolation (namespaces) from day one.

### 3.6 Evaluation harness (the heart of the project)
- Benchmarks: **LongMemEval** and **LoCoMo**, both public long-conversation memory benchmarks. Read their papers and repos for the exact question categories and metrics; they include categories such as knowledge updates, temporal reasoning, multi-session reasoning and abstention.
- Baselines: (a) full-context (stuff the whole history in), (b) naive RAG over chunks, (c) your memory layer.
- Metrics: answer accuracy (judge model or exact match as the benchmark specifies), retrieval recall, tokens per query, latency, cost.
- Reproducible runs: config files, fixed seeds, saved outputs, one command to rerun.

---

## 4. Extra features (nice-to-have, v2+)

| Feature | What it adds | Difficulty |
|---|---|---|
| **MCP server wrapper** | Any MCP client (Claude Desktop, Cursor) gets persistent memory; shows standard MCP skills | Easy |
| **Knowledge graph layer** | Entities and relations (Kuzu, Neo4j or NetworkX) for multi-hop questions | Medium |
| **Reflection / summaries** | Periodic higher-level summaries ("user's career goals") stored as separate memories | Medium |
| **Memory inspector UI** | Next.js page: browse, edit, delete memories, view supersession timeline | Medium |
| **Privacy controls** | Per-memory delete, export, "forget everything", PII redaction before storage | Medium |
| **Procedural memory** | Remember how the user likes tasks done (style, formats) separately from facts | Medium |
| **Multi-agent shared memory** | Scoped memory across agents with access rules | Hard |
| **Langfuse / OpenTelemetry tracing** | Every extract/retrieve/inject step traced with latency and token counts | Easy |
| **Adaptive retrieval** | Learn weights a, b, c, tau from feedback instead of hand-tuning | Hard |
| **Plug into your WanderAI** | Travel agent remembers preferences across trips (a real integration demo) | Easy |

**Suggested v2 pick:** MCP server + tracing + the inspector UI. They give the biggest portfolio return for the least risk.

---

## 5. Architecture

```
          Chat agent / app
               |
   +-----------+-----------+
   |   MemoryLayer API     |   add() | search() | forget()
   +-----------+-----------+
       |                |
  WRITE PATH        READ PATH
  (async)           (in the request)
       |                |
  Extractor LLM     Router (needs memory?)
  Pydantic schema   Hybrid retrieval (vector + BM25 + filters)
       |            Scorer (relevance + recency + importance)
  Conflict check    Token-budgeted formatter
       |                |
  +----+----------------+----+
  |   Vector store | SQL store |   + raw episode log
  +----+----------------+----+
       |
  Background worker: dedup, decay, summaries
       |
  Tracing (Langfuse / OTel)    Eval harness (benchmarks)
```

### Core data model (SQL sketch)
- `episodes(id, user_id, session_id, role, text, ts)`
- `memories(id, user_id, content, entity, event_time, created_at, importance, confidence, status, superseded_by, source_episode_id, last_accessed_at, access_count)`
- Vectors live in the vector store with `memory_id` as the key.

---

## 6. Tech stack

| Layer | Recommended | Notes |
|---|---|---|
| Language | Python 3.11+ | Type hints throughout |
| Schemas | Pydantic v2 | You already use it |
| LLM access | LiteLLM | One interface for Ollama, Groq, Gemini |
| Local model | Ollama (Llama 3.x / Qwen) | For cheap extraction |
| Embeddings | `bge-small` / `bge-m3` or Ollama embedding model | Compare 2 and report |
| Vector store | pgvector (new for you) or Qdrant (known) | pgvector lets one DB hold everything |
| Relational | SQLite (local) / Postgres (server) | SQLAlchemy 2.0 |
| Keyword search | `rank-bm25` or Postgres full-text | For hybrid retrieval |
| Reranker (optional) | `bge-reranker-base` | Measure the latency cost |
| Background jobs | Celery + Redis (known) or APScheduler (lighter) | Start with APScheduler |
| Graph (v2) | Kuzu or NetworkX | Kuzu is embedded and easy |
| API (demo) | FastAPI | Endpoints plus an MCP server |
| MCP | Official Python MCP SDK | Standard MCP is a known gap |
| UI (v2) | Next.js | Memory inspector |
| Observability | Langfuse (self-hosted) or OpenTelemetry | Fills your observability gap |
| Eval | Your LLM Evaluation Platform + benchmark scripts | Reuse, then link both projects |
| Testing | pytest, hypothesis (optional) | Unit tests for scoring and conflict logic |
| Quality | ruff, mypy, pre-commit | Fills known hygiene gap |
| CI | GitHub Actions | Lint, test, small eval smoke test |
| Packaging | `uv` or Poetry, publish to PyPI | A real installable package is a strong signal |
| Containers | Docker Compose | One command to run the full demo |
| Docs | MkDocs Material | Architecture and API pages |

---

## 7. Evaluation plan

1. **Pick the benchmark subsets.** Full benchmark runs can be expensive in LLM calls. Start with a 50 to 100 question subset and scale later. Estimate the cost before you start.
2. **Fix the answer model and judge.** Use the same answer model across all systems so you compare memory, not the LLM. State the judge model and prompt.
3. **Baselines.**
   - Full-context: the whole history in the prompt (an upper bound on accuracy, high token cost).
   - Naive RAG: chunk the history, embed, retrieve top-k.
   - Your system.
4. **Ablations (the portfolio gold):**
   - without decay
   - without supersession / conflict resolution
   - without hybrid retrieval (vector only)
   - without recency in the score
   - extraction model size: small local vs larger hosted
   - different embedding models
5. **Report per question category** (knowledge update, temporal, multi-session, abstention), not just one overall number. Weaknesses reported honestly make the project more credible.
6. **Efficiency table:** accuracy vs tokens per query vs p50/p95 latency vs cost.
7. **Failure analysis:** show 5 to 10 real failure cases and what you learned.

Expectation setting: do not promise "beats full-context". A realistic and respectable result is "close to full-context accuracy at a fraction of the tokens". Whatever you get, report it as measured.

---

## 8. Phased implementation plan

Timeline assumes roughly part-time work. Adjust to your pace.

**Phase 0: Setup (2 to 3 days)**
- Repo, packaging, ruff/mypy/pytest/pre-commit, GitHub Actions, Docker Compose skeleton.
- Read the LongMemEval and LoCoMo papers and repos. Decide subsets and metrics.

**Phase 1: Data contracts and storage (3 to 4 days)**
- Pydantic models for episodes and memories; SQL schema and migrations.
- Vector store adapter; per-user namespaces.
- Unit tests for CRUD.

**Phase 2: Write path (4 to 5 days)**
- Extraction prompt and schema validation; retry on malformed JSON.
- Async execution after the reply.
- Tests with fixed conversation fixtures; measure extraction precision by hand on a small labeled set.

**Phase 3: Read path (4 to 5 days)**
- Hybrid retrieval, scoring function, token-budgeted formatter, retrieval router.
- Tune weights on a dev split.

**Phase 4: Consolidation (4 to 5 days)**
- Dedup, supersession logic, decay, background job.
- Dedicated tests for contradiction cases (the trickiest part).

**Phase 5: Evaluation harness (5 to 7 days)**
- Benchmark loaders, baselines, runner, result storage; connect to your eval platform.
- First full results run, then ablations.

**Phase 6: Demo and integrations (4 to 5 days)**
- FastAPI demo chat app with memory on/off toggle.
- MCP server wrapper.
- Tracing with Langfuse.

**Phase 7: Docs and portfolio (3 to 4 days)**
- README with architecture diagram and results table, MkDocs site, a 2 to 3 minute demo video.
- Portfolio page and write-up, PyPI release.

**Stretch:** knowledge graph layer, inspector UI, WanderAI integration.

---

## 9. Suggested repository structure

```
agent-memory-layer/
  src/memlayer/
    api.py              # MemoryLayer class
    models.py           # Pydantic schemas
    extract/            # prompts, extractor, validators
    store/              # sql.py, vector.py, episodes.py
    retrieve/           # hybrid.py, scoring.py, router.py, format.py
    consolidate/        # dedup.py, conflicts.py, decay.py
    llm/                # LiteLLM wrapper, embedders
    mcp_server.py
    tracing.py
  eval/
    datasets/           # loaders for LongMemEval, LoCoMo subsets
    baselines/          # full_context.py, naive_rag.py
    run.py              # one-command reproducible runs
    configs/
    results/            # saved outputs and tables
  demo/
    app/                # FastAPI + simple UI
  tests/
  docs/
  docker-compose.yml
  pyproject.toml
  README.md
```

---

## 10. Risks and how to handle them

| Risk | Mitigation |
|---|---|
| Benchmark runs cost too much | Start with small subsets; cache LLM outputs; use a local model for extraction |
| Small local models extract poorly | Strict JSON schema, few-shot examples, retries; compare against a hosted model in an ablation |
| Contradiction handling is subtle | Build a hand-written test set of update cases first |
| Results are not impressive | Report honestly; the ablations and analysis are still the value |
| Scope creep (graph, UI, multi-agent) | Finish v1 and evaluation before touching extras |
| Comparisons with Mem0/Letta/Zep become stale | Verify their current docs before publishing; keep claims narrow |
| LLM judge bias | Use the benchmark's recommended judge; spot-check a sample by hand |
| Privacy concerns about stored personal facts | Local-first default, delete/export functions, PII redaction option, never store sensitive data without a flag |

---

## 11. How to present it in your portfolio

- **Headline:** "Long-term memory for conversational agents, benchmarked on LongMemEval/LoCoMo."
- **Show:** the results table, the ablation chart, the architecture diagram, a short demo (memory on vs off, including an update such as a changed city).
- **Link projects:** "Evaluated with my own LLM Evaluation Platform" and "integrated into WanderAI".
- **Be specific on tech:** pgvector/Qdrant, hybrid retrieval, supersession logic, async workers, MCP, Langfuse tracing.
- **Resume bullet templates** (fill with your real numbers):
  - Built an open-source long-term memory library for LLM agents with hybrid retrieval, contradiction-aware consolidation and time-decay; reached **X%** accuracy on a LongMemEval subset using **Y%** fewer tokens than full-context.
  - Ran **N** ablations to isolate the effect of decay, supersession and hybrid retrieval; documented failure modes.
  - Shipped as a PyPI package with CI, typed API, MCP server and Langfuse tracing.

---

## 12. Definition of done (v1)

- [ ] Installable package with typed API and docs
- [ ] Write path, read path and consolidation working with tests
- [ ] Benchmark results reproducible with one command
- [ ] Baselines and at least 4 ablations reported
- [ ] Demo app and MCP server running via Docker Compose
- [ ] Tracing visible in Langfuse
- [ ] README with results, architecture and honest limitations
- [ ] Portfolio page and short demo video

---

## 13. Open decisions to settle before coding

1. pgvector or Qdrant as the primary vector store?
2. Which LLM for extraction (local Ollama vs hosted) and which for answering and judging?
3. Which benchmark subset sizes fit your budget?
4. Library name and license (MIT or Apache-2.0 are typical).
5. Do you ship the graph layer in v1 or keep it as v2?

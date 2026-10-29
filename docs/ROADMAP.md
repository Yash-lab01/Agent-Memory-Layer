# Project Roadmap: Agent Memory Layer (`memlayer`)

This roadmap tracks the development progress of `memlayer` across 8 core milestones and 25 granular phases. Check off tasks as they are implemented, verified, and committed.

---

## Milestone 0: Scaffolding, Governance & Developer Tooling
- [x] **Phase 0.1: Project Skeleton & Packaging Setup**
  - [x] Initialize `pyproject.toml` (Python 3.11+) with dependencies and dev tools
  - [x] Configure `ruff`, `mypy`, and `pytest`
  - [x] Set up `.gitignore`, `.env.example`, and `.pre-commit-config.yaml`
  - [x] Create repository directory tree (`src/memlayer/`, `eval/`, `demo/`, `tests/`, `docs/`)
- [x] **Phase 0.2: Project Governance & Architecture Documentation**
  - [x] Establish canonical `AGENTS.md`
  - [x] Create `docs/ROADMAP.md` (this file)
  - [x] Create `docs/DECISIONS.md` with ADR-001 (Architecture Choices & Open Decisions Framework)
  - [x] Create `docs/ARCHITECTURE.md` (System components, read/write flows, data model)
  - [x] Create `docs/PRD.md` (Product requirements, non-goals, benchmark targets)
- [x] **Phase 0.3: CI Pipeline & Containerization Skeleton**
  - [x] Create `.github/workflows/ci.yml` (linting, type checking, unit tests)
  - [x] Create baseline `docker-compose.yml` for local service orchestration

---

## Milestone 1: Domain Models & Pluggable Storage Layer
- [x] **Phase 1.1: Core Data Contracts & Schemas**
  - [x] Implement `src/memlayer/models.py` with Pydantic v2 schemas (`Episode`, `MemoryItem`, `MemoryStatus`, `MemoryMetadata`)
  - [x] Unit tests for model validation, serialization, and edge cases in `tests/test_models.py`
- [x] **Phase 1.2: Relational Store & Episode Logging (SQLite / SQLAlchemy 2.0)**
  - [x] Implement `src/memlayer/store/sql.py` (SQLAlchemy 2.0 declarative models & repository)
  - [x] Implement `src/memlayer/store/episodes.py` (Episode logger with provenance tracking)
  - [x] Enforce strict per-user isolation (`user_id` filtering on all operations)
  - [x] Unit tests for CRUD and multi-user isolation in `tests/test_sql_store.py`
- [x] **Phase 1.3: Embeddings & Vector Store Adapter (Abstract + Local)**
  - [x] Implement embedder abstraction in `src/memlayer/llm/embedder.py` (LiteLLM / sentence-transformers)
  - [x] Implement vector store interface in `src/memlayer/store/vector_base.py`
  - [x] Implement local vector adapter in `src/memlayer/store/vector.py` (Qdrant in-memory/local)
  - [x] Unit tests for vector insertion, search, and namespace filtering in `tests/test_vector_store.py`
- [x] **Phase 1.4: Database Migrations & Postgres Support**
  - [x] Set up Alembic migration scaffolding
  - [x] Implement Postgres + pgvector connection support in `src/memlayer/store/postgres.py`

---

## Milestone 2: Memory Write Path (Extraction & Ingestion)
- [x] **Phase 2.1: Extraction Prompt & Pydantic JSON Validator**
  - [x] Design extraction prompt in `src/memlayer/extract/prompts.py`
  - [x] Implement extractor engine in `src/memlayer/extract/extractor.py` (LiteLLM)
  - [x] Structured JSON validation with automatic retry on malformed outputs
  - [x] Fixed conversation fixtures and tests in `tests/test_extractor.py`
- [x] **Phase 2.2: Public Write API & Asynchronous Execution**
  - [x] Define `MemoryLayer` class in `src/memlayer/api.py`
  - [x] Implement `MemoryLayer.add(user_id, messages)`
  - [x] Async background processing for extraction and vector indexing
  - [x] Write path integration tests in `tests/test_write_pipeline.py`

---

## Milestone 3: Memory Read Path (Hybrid Retrieval & Scoring)
- [x] **Phase 3.1: Hybrid Retrieval Engine (Semantic + BM25)**
  - [x] Implement BM25 keyword search in `src/memlayer/retrieve/hybrid.py`
  - [x] Implement Reciprocal Rank Fusion (RRF) for semantic + keyword merging
  - [x] Filter out superseded memories by default
  - [x] Unit tests in `tests/test_hybrid_retrieval.py`
- [x] **Phase 3.2: Multi-Factor Scoring Engine**
  - [x] Implement scoring formula in `src/memlayer/retrieve/scoring.py`: `score = a * sim + b * exp(-age/tau) + c * importance`
  - [x] Configurable hyperparameters (`a`, `b`, `c`, `tau`)
  - [x] Mathematical unit tests for age decay and boundary conditions in `tests/test_scoring.py`
- [x] **Phase 3.3: Retrieval Router (Turn Gating)**
  - [x] Implement query classification in `src/memlayer/retrieve/router.py` to skip retrieval on trivial turns
  - [x] Router tests in `tests/test_router.py`
- [x] **Phase 3.4: Context Formatter & Public Search API**
  - [x] Implement `src/memlayer/retrieve/format.py` (timestamp formatting, provenance tags, token budgeting)
  - [x] Implement `MemoryLayer.search(user_id, query, token_budget)` in `src/memlayer/api.py`
  - [x] Search API tests in `tests/test_search_api.py`

---

## Milestone 4: Memory Consolidation & Lifecycle Management
- [x] **Phase 4.1: Contradiction & Supersession Engine (Test-Driven)**
  - [x] Create curated contradiction test fixtures in `tests/fixtures/contradictions.json`
  - [x] Implement conflict detection in `src/memlayer/consolidate/conflicts.py`
  - [x] Mark outdated memories as `superseded` with `superseded_by` pointer (never hard-delete)
  - [x] Contradiction unit tests in `tests/test_conflicts.py`
- [x] **Phase 4.2: Deduplication Engine**
  - [x] Implement near-duplicate fact merging in `src/memlayer/consolidate/dedup.py`
  - [x] Dedup tests in `tests/test_dedup.py`
- [x] **Phase 4.3: Memory Decay & Explicit Forgetting**
  - [x] Implement retrieval-frequency decay in `src/memlayer/consolidate/decay.py`
  - [x] Implement `MemoryLayer.forget(user_id, memory_id)` in `src/memlayer/api.py`
  - [x] Decay and explicit forget tests in `tests/test_decay.py`
- [x] **Phase 4.4: Background Consolidation Worker**
  - [x] Implement `APScheduler` background worker in `src/memlayer/consolidate/worker.py`
  - [x] Background worker lifecycle tests in `tests/test_worker.py`

---

## Milestone 5: Benchmark Evaluation Harness & Ablation Studies
- [x] **Phase 5.1: Benchmark Dataset Loaders & Splits**
  - [x] Dataset loaders for LongMemEval and LoCoMo subsets in `eval/datasets/`
  - [x] Subsets config in `eval/configs/subsets.yaml`
- [x] **Phase 5.2: Baseline Implementations**
  - [x] Full-context baseline in `eval/baselines/full_context.py`
  - [x] Naive RAG baseline in `eval/baselines/naive_rag.py`
  - [x] MemLayer baseline in `eval/baselines/memlayer_agent.py`
- [x] **Phase 5.3: Benchmark Runner & Output Caching**
  - [x] CLI runner in `eval/run.py` with reproducible seeds and disk caching
  - [x] Standardized LLM judge prompt in `eval/judge.py`
- [x] **Phase 5.4: Baseline vs MemLayer Benchmark Execution**
  - [x] Execute evaluation runs and save raw outputs to `eval/results/`
  - [x] Compute accuracy, latency (p50/p95), tokens/query, and cost
- [x] **Phase 5.5: Ablation Studies & Failure Analysis**
  - [x] Run 4+ ablations: without decay, without supersession, vector-only, without recency
  - [x] Compile failure analysis report in `docs/EVALUATION_REPORT.md`

---

## Milestone 6: Demo Application, MCP Server & Observability
- [x] **Phase 6.1: FastAPI Service & Interactive Web Demo**
  - [x] FastAPI backend in `demo/app/main.py`
  - [x] Interactive UI with Memory Toggle (ON / OFF) and live supersession visualizer
- [x] **Phase 6.2: Model Context Protocol (MCP) Server**
  - [x] Implement `src/memlayer/mcp_server.py` using Python MCP SDK
  - [x] Expose memory tools (`memlayer_add`, `memlayer_search`, `memlayer_forget`, `memlayer_list`, `memlayer_reset`)
  - [x] MCP tool tests in `tests/test_mcp_server.py`
- [x] **Phase 6.3: Observability & Tracing (Langfuse / OpenTelemetry)**
  - [x] Implement `src/memlayer/tracing.py`
  - [x] Docker Compose Langfuse service integration
  - [x] Distributed tracing tests in `tests/test_tracing.py`

---

## Milestone 7: Packaging, Documentation & Recruiter Portfolio
- [x] **Phase 7.1: Packaging & PyPI Release Readiness**
  - [x] Verify standalone `pip install .` without demo dependencies
  - [x] Test wheel build and packaging
- [x] **Phase 7.2: Recruiter-Ready README & MkDocs Documentation**
  - [x] Comprehensive `README.md` with benchmark numbers and architecture diagram
  - [x] MkDocs Material setup and documentation pages
- [x] **Phase 7.3: Portfolio Write-Up & Demo Walkthrough**
  - [x] Case study and interview prep notes in `docs/PORTFOLIO.md`

---

## Milestone 8 (v2 Stretch Items)
- [x] **Phase 8.1: Knowledge Graph Layer (`InMemoryGraphStore` / `NetworkX`, Spec 017)**
  - [x] Abstract `GraphStoreBase` interface in `src/memlayer/store/graph_base.py`
  - [x] Zero-dependency `InMemoryGraphStore` with multi-hop BFS pathfinding
  - [x] `NetworkXGraphStore` adapter with centrality analysis via `agent-memory-layer[graph]`
  - [x] `EntityNode`, `RelationEdge`, `GraphTriplet` Pydantic models in `src/memlayer/models.py`
  - [x] Subgraph context formatting in `src/memlayer/retrieve/format.py`
  - [x] `MemoryLayer` integration (`add_triplet`, `get_graph_context`, search injection, tenant-scoped purge)
  - [x] 12 comprehensive unit and integration tests in `tests/test_graph_store.py`
- [x] **Phase 8.2: Memory Inspector UI & Knowledge Graph Visualizer (Spec 018)**
  - [x] Interactive 3-tab Inspector (Active facts, Superseded history, Knowledge Graph)
  - [x] Zero-dependency Canvas force-directed graph layout with high-DPI scaling and node dragging
  - [x] Multi-hop path finding UI connecting entities with highlighted hops and breadcrumbs
  - [x] Triplet creation modal and tenant-scoped graph reset
  - [x] FastAPI graph endpoints (`/api/graph`, `/api/graph/triplets`, `/api/graph/paths`, `DELETE /api/graph`)
  - [x] End-to-end integration tests in `tests/test_demo.py`
- [x] **Phase 8.3: WanderAI Travel Companion Integration Demo (Spec 019)**
  - [x] `WanderAIAgent` encapsulating `MemoryLayer` with travel persona and itinerary synthesis
  - [x] 4-turn showcase scenarios (`demo/wanderai/scenarios.py`) demonstrating cross-trip recall, loyalty/budget updates, and graph traversal
  - [x] Executable CLI demo runner (`demo/wanderai/run_demo.py`)
  - [x] WanderAI REST endpoints in `demo/app/main.py` (`/api/wanderai/chat`, `/api/wanderai/profile`, `/api/wanderai/scenarios`, `/api/wanderai/seed`)
  - [x] Dedicated WanderAI profile tab and scenario triggers in web UI
  - [x] Comprehensive unit and integration tests in `tests/test_wanderai.py`

---

## Milestone 9: Real Evaluation, Release & Portfolio (Spec 020)
- [ ] **Phase 9.1: Harness Audit, Experimental Design Freeze & Cost Modeling**
  - [ ] Audit harness in `eval/README.md` (document real vs stub models, embedder dimension, dataset sampling)
  - [ ] Freeze models, dev/test splits, and seeds in `eval/configs/main.yaml`
  - [ ] Implement cost estimation script in `scripts/estimate_eval_cost.py` with 25% safety buffer
  - [ ] Implement query/completion disk caching keyed by `hash(model, prompt, params)`
- [ ] **Phase 9.2: Real Dataset Loaders & Chronological Ingestion Protocol**
  - [ ] Full/large-share LoCoMo dataset loader in `eval/datasets/locomo.py`
  - [ ] Stratified 100-150 question LongMemEval subset loader in `eval/datasets/longmemeval.py`
  - [ ] Implement `EvalItem` unified schema in `eval/datasets/models.py`
  - [ ] Enforce chronological replay with original timestamps
  - [ ] Pass question date as "now" anchor to prevent wall-clock drift in recency/decay
  - [ ] Isolate tenant namespaces per haystack with reusable SQLite memory dumps
- [ ] **Phase 9.3: Baseline Expansion**
  - [ ] Standardize identical answer model, judge, and prompt templates across all baselines
  - [ ] Implement Full-Context baseline with context window truncation telemetry
  - [ ] Implement Naive RAG baseline (chunked dense top-k)
  - [ ] Implement BM25 + Reranker baseline over raw chunks (evaluating value of atomic fact extraction)
- [ ] **Phase 9.4: Statistical Engine & Honest Cost Accounting**
  - [ ] Per-question JSONL logging (verdict, raw text, prompt/completion tokens, retrieval latency, total latency)
  - [ ] Retrieval recall@k tracking against evidence IDs
  - [ ] 95% bootstrap confidence intervals ($N=10,000$ resamples) per system and category
  - [ ] Paired bootstrap difference and McNemar's tests across systems
  - [ ] Amortized ingestion cost formula and break-even query count $Q^*$ analysis
- [ ] **Phase 9.5: Judge Calibration & Pilot Run**
  - [ ] Hand-grade 30-50 random judge verdicts and verify $\ge 90\%$ agreement rate
  - [ ] Tune category-specific judge prompts for temporal and abstention queries
  - [ ] Execute 10-question pilot run to verify end-to-end telemetry and token accounting
- [ ] **Phase 9.6: Hyperparameter Tuning (Dev Split) & Main Test Execution**
  - [ ] Tune $\alpha, \beta, \gamma, \tau, k$, and token budget solely on Dev split; freeze tagged `eval-v1`
  - [ ] Execute main LoCoMo test split
  - [ ] Execute main LongMemEval stratified test split
  - [ ] Run 8 systematic ablations (without supersession, without decay, without recency, vector-only, without router, without graph context, extraction model size, token budget sweep)
  - [ ] Classify 10+ failure cases into failure taxonomy
- [ ] **Phase 9.7: Automated Report Generation & README Overhaul**
  - [ ] Build `eval/generate_tables.py` to compile Markdown tables directly from JSONL results
  - [ ] Configure `make eval-smoke` (CI fast check) and `make eval-full` (reproducible run)
  - [ ] Update README with generated tables, 95% CIs, sample size $n$, break-even analysis, and CI smoke test note
- [ ] **Phase 9.8: PyPI Release (v0.1.0) & Repository Polish**
  - [ ] Finalize `pyproject.toml` metadata (classifiers, author, URLs, entry points)
  - [ ] Build wheels with `uv build` and verify TestPyPI installation
  - [ ] Publish v0.1.0 to PyPI using GitHub Actions trusted publishing
  - [ ] Deploy MkDocs documentation site (`mkdocs gh-deploy`)
  - [ ] Add `CONTRIBUTING.md`, GitHub issue templates, and 2-minute demo video/GIF walkthrough
- [ ] **Phase 9.9: Portfolio Case Study & Resume Bullets**
  - [ ] Comprehensive case study in `docs/CASE_STUDY.md` and `docs/PORTFOLIO.md`
  - [ ] Update resume bullet templates strictly with verified benchmark numbers



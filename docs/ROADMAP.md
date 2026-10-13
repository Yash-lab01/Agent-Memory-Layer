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
- [ ] **Phase 1.4: Database Migrations & Postgres Support**
  - [ ] Set up Alembic migration scaffolding
  - [ ] Implement Postgres + pgvector connection support in `src/memlayer/store/postgres.py`

---

## Milestone 2: Memory Write Path (Extraction & Ingestion)
- [ ] **Phase 2.1: Extraction Prompt & Pydantic JSON Validator**
  - [ ] Design extraction prompt in `src/memlayer/extract/prompts.py`
  - [ ] Implement extractor engine in `src/memlayer/extract/extractor.py` (LiteLLM)
  - [ ] Structured JSON validation with automatic retry on malformed outputs
  - [ ] Fixed conversation fixtures and tests in `tests/test_extractor.py`
- [ ] **Phase 2.2: Public Write API & Asynchronous Execution**
  - [ ] Define `MemoryLayer` class in `src/memlayer/api.py`
  - [ ] Implement `MemoryLayer.add(user_id, messages)`
  - [ ] Async background processing for extraction and vector indexing
  - [ ] Write path integration tests in `tests/test_write_pipeline.py`

---

## Milestone 3: Memory Read Path (Hybrid Retrieval & Scoring)
- [ ] **Phase 3.1: Hybrid Retrieval Engine (Semantic + BM25)**
  - [ ] Implement BM25 keyword search in `src/memlayer/retrieve/hybrid.py`
  - [ ] Implement Reciprocal Rank Fusion (RRF) for semantic + keyword merging
  - [ ] Filter out superseded memories by default
  - [ ] Unit tests in `tests/test_hybrid_retrieval.py`
- [ ] **Phase 3.2: Multi-Factor Scoring Engine**
  - [ ] Implement scoring formula in `src/memlayer/retrieve/scoring.py`: `score = a * sim + b * exp(-age/tau) + c * importance`
  - [ ] Configurable hyperparameters (`a`, `b`, `c`, `tau`)
  - [ ] Mathematical unit tests for age decay and boundary conditions in `tests/test_scoring.py`
- [ ] **Phase 3.3: Retrieval Router (Turn Gating)**
  - [ ] Implement query classification in `src/memlayer/retrieve/router.py` to skip retrieval on trivial turns
  - [ ] Router tests in `tests/test_router.py`
- [ ] **Phase 3.4: Context Formatter & Public Search API**
  - [ ] Implement `src/memlayer/retrieve/format.py` (timestamp formatting, provenance tags, token budgeting)
  - [ ] Implement `MemoryLayer.search(user_id, query, token_budget)` in `src/memlayer/api.py`
  - [ ] Search API tests in `tests/test_search_api.py`

---

## Milestone 4: Memory Consolidation & Lifecycle Management
- [ ] **Phase 4.1: Contradiction & Supersession Engine (Test-Driven)**
  - [ ] Create curated contradiction test fixtures in `tests/fixtures/contradictions.json`
  - [ ] Implement conflict detection in `src/memlayer/consolidate/conflicts.py`
  - [ ] Mark outdated memories as `superseded` with `superseded_by` pointer (never hard-delete)
  - [ ] Contradiction unit tests in `tests/test_conflicts.py`
- [ ] **Phase 4.2: Deduplication Engine**
  - [ ] Implement near-duplicate fact merging in `src/memlayer/consolidate/dedup.py`
  - [ ] Dedup tests in `tests/test_dedup.py`
- [ ] **Phase 4.3: Memory Decay & Explicit Forgetting**
  - [ ] Implement retrieval-frequency decay in `src/memlayer/consolidate/decay.py`
  - [ ] Implement `MemoryLayer.forget(user_id, memory_id)` in `src/memlayer/api.py`
  - [ ] Decay and explicit forget tests in `tests/test_decay.py`
- [ ] **Phase 4.4: Background Consolidation Worker**
  - [ ] Implement `APScheduler` background worker in `src/memlayer/consolidate/worker.py`
  - [ ] Background worker lifecycle tests in `tests/test_worker.py`

---

## Milestone 5: Benchmark Evaluation Harness & Ablation Studies
- [ ] **Phase 5.1: Benchmark Dataset Loaders & Splits**
  - [ ] Dataset loaders for LongMemEval and LoCoMo subsets in `eval/datasets/`
  - [ ] Subsets config in `eval/configs/subsets.yaml`
- [ ] **Phase 5.2: Baseline Implementations**
  - [ ] Full-context baseline in `eval/baselines/full_context.py`
  - [ ] Naive RAG baseline in `eval/baselines/naive_rag.py`
- [ ] **Phase 5.3: Benchmark Runner & Output Caching**
  - [ ] CLI runner in `eval/run.py` with reproducible seeds and disk caching
  - [ ] Standardized LLM judge prompt in `eval/judge.py`
- [ ] **Phase 5.4: Baseline vs MemLayer Benchmark Execution**
  - [ ] Execute evaluation runs and save raw outputs to `eval/results/`
  - [ ] Compute accuracy, latency (p50/p95), tokens/query, and cost
- [ ] **Phase 5.5: Ablation Studies & Failure Analysis**
  - [ ] Run 4+ ablations: without decay, without supersession, vector-only, without recency
  - [ ] Compile failure analysis report in `docs/EVALUATION_REPORT.md`

---

## Milestone 6: Demo Application, MCP Server & Observability
- [ ] **Phase 6.1: FastAPI Service & Interactive Web Demo**
  - [ ] FastAPI backend in `demo/app/main.py`
  - [ ] Interactive UI with Memory Toggle (ON / OFF) and live supersession visualizer
- [ ] **Phase 6.2: Model Context Protocol (MCP) Server**
  - [ ] Implement `src/memlayer/mcp_server.py` using Python MCP SDK
  - [ ] Expose memory tools (`memlayer_add`, `memlayer_search`, `memlayer_forget`)
  - [ ] MCP tool tests in `tests/test_mcp_server.py`
- [ ] **Phase 6.3: Observability & Tracing (Langfuse / OpenTelemetry)**
  - [ ] Implement `src/memlayer/tracing.py`
  - [ ] Docker Compose Langfuse service integration

---

## Milestone 7: Packaging, Documentation & Recruiter Portfolio
- [ ] **Phase 7.1: Packaging & PyPI Release Readiness**
  - [ ] Verify standalone `pip install .` without demo dependencies
  - [ ] Test wheel build and packaging
- [ ] **Phase 7.2: Recruiter-Ready README & MkDocs Documentation**
  - [ ] Comprehensive `README.md` with benchmark numbers and architecture diagram
  - [ ] MkDocs Material setup and documentation pages
- [ ] **Phase 7.3: Portfolio Write-Up & Demo Walkthrough**
  - [ ] Case study and interview prep notes in `docs/PORTFOLIO.md`

---

## Milestone 8 (v2 Stretch Items)
- [ ] **Phase 8.1:** Knowledge Graph Layer (`Kùzu` / `NetworkX`)
- [ ] **Phase 8.2:** Memory Inspector UI (Next.js)
- [ ] **Phase 8.3:** WanderAI Integration Demo

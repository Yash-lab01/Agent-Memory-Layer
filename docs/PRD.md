# Product Requirements Document (PRD): Agent Memory Layer (`memlayer`)

---

## 1. Executive Summary
`memlayer` is an open-source Python library and companion demo application that provides conversational AI agents with durable, long-term memory across sessions. It extracts atomic facts from conversation turns, maintains temporal provenance, resolves contradictions non-destructively through supersession, and surfaces relevant memories via hybrid retrieval.

The primary deliverable of this project is a **measured, reproducible benchmark** comparing memory design choices on public benchmarks (LongMemEval and LoCoMo) against full-context and naive RAG baselines.

---

## 2. Problem Statement
Current LLM conversational agents suffer from three major memory failure modes:
1. **Context Window Saturation:** Stuffing entire multi-session conversation histories into prompts is cost-prohibitive, introduces high latency, and degrades reasoning accuracy ("lost in the middle").
2. **Naive RAG Failures:** Chunk-based RAG retrieves disjointed message fragments lacking entity context, struggles with temporal updates (e.g. user moved cities), and cannot resolve direct factual contradictions.
3. **Lack of Transparent Ablations:** Existing memory frameworks claim performance boosts without published, reproducible ablation studies isolating the specific impact of decay, deduplication, supersession, or hybrid scoring.

---

## 3. Product Positioning & Boundaries
- **Market Context:** Frameworks such as Mem0, Letta (MemGPT), and Zep/Graphiti provide agent memory solutions. `memlayer` does not claim theoretical novelty; its core value proposition is **rigor, transparency, and reproducible evaluation**.
- **Portfolio Focus:** Designed for senior conversational AI and LLM systems engineering evaluation. Every design tradeoff is justified by measurable numbers.
- **Non-Goals for v1:**
  - Multi-agent shared swarm memory (deferred to future work).
  - Graph neural network embeddings (deferred to v2 stretch).
  - Proprietary cloud-hosted SaaS layer (remains open-source and local-first).

---

## 4. Functional Requirements

### 4.1 Memory Write Path
- **FR-W1 (Fact Extraction):** Extract atomic, self-contained facts from conversation turns using structured Pydantic schemas.
- **FR-W2 (Provenance):** Retain raw conversation turns in the episode log linked to extracted memory IDs.
- **FR-W3 (Asynchronous Processing):** Memory extraction and indexing must execute asynchronously without increasing chat response latency.

### 4.2 Storage Layer
- **FR-S1 (Relational Store):** Store structured facts, status, timestamps, and supersession links in SQLite (local) and PostgreSQL (server).
- **FR-S2 (Vector Store):** Store fact embeddings with metadata payloads in Qdrant or pgvector behind an abstract adapter.
- **FR-S3 (User Isolation):** Enforce strict `user_id` scoping across all storage queries and payload filters.

### 4.3 Memory Read Path
- **FR-R1 (Hybrid Retrieval):** Blend vector semantic search with BM25 keyword matching using Reciprocal Rank Fusion.
- **FR-R2 (Multi-Factor Scoring):** Rank memories based on semantic similarity, exponential recency decay, and importance score.
- **FR-R3 (Retrieval Router):** Bypass memory lookup for pleasantries and non-informative dialogue turns.
- **FR-R4 (Token Budgeting):** Truncate retrieved memories to fit strictly within a caller-specified token budget.

### 4.4 Consolidation & Lifecycle
- **FR-C1 (Supersession):** Detect factual contradictions and mark outdated memories as `superseded` with `superseded_by` links. No hard deletes on contradiction.
- **FR-C2 (Deduplication):** Merge semantically redundant facts without losing provenance.
- **FR-C3 (Decay & Forget):** Lower retrieval weight of unaccessed facts; support explicit user-requested deletion (`forget`).

---

## 5. Non-Functional Requirements
- **Performance:** Local memory retrieval latency under 50ms (p95); add endpoint acknowledgement under 20ms.
- **Type Safety & Code Quality:** Python 3.11+, 100% type annotations checked with `mypy`, formatted and linted with `ruff`.
- **Reproducibility:** Evaluation runs reproducible with fixed seeds, LiteLLM disk caching, and a single CLI command (`python -m eval.run`).
- **Privacy & Security:** Local-first defaults, zero cross-user data leakage, no hardcoded API keys.

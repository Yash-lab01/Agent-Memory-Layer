# Spec 002: Hybrid Retrieval Engine (Semantic + BM25)

## 1. Goal
Implement the hybrid retrieval engine for Milestone 3 (Phase 3.1) in `src/memlayer/retrieve/hybrid.py` (and supporting BM25 indexer in `src/memlayer/retrieve/bm25.py`). The engine merges dense semantic vector similarity with sparse BM25 keyword search, ensuring high recall for both exact entity/keyword queries (e.g., proper nouns, email addresses, numbers) and conceptual paraphrases.

---

## 2. Key Requirements & Design

### 2.1 BM25 Keyword Search (`src/memlayer/retrieve/bm25.py`)
- Uses `rank_bm25.BM25Okapi` over tokenized memory contents.
- Simple, robust tokenizer (lowercasing, punctuation stripping, token filtering).
- Computes normalized relevance scores for queries against a collection of memory items.

### 2.2 Hybrid Fusion (`src/memlayer/retrieve/hybrid.py`)
- **Dual Retrieval Streams:**
  1. **Dense Semantic Stream:** Vector embeddings matched via `VectorStoreBase.search(user_id, query_vector, limit)`.
  2. **Sparse Keyword Stream:** BM25 scoring over active user memories.
- **Fusion Strategies:**
  - **Reciprocal Rank Fusion (RRF):**
    $$RRF\_Score(d) = \frac{w_{dense}}{k + rank_{dense}(d)} + \frac{w_{bm25}}{k + rank_{bm25}(d)}$$
    Standard constant $k = 60$.
  - **Convex Linear Combination:**
    $$Score(d) = \alpha \cdot DenseScore(d) + (1 - \alpha) \cdot BM25Normalized(d)$$
- **Strict Filtering:**
  - Filter out `superseded` memories by default (only `MemoryStatus.ACTIVE` returned).
  - Strict tenant isolation: every query is scoped by `user_id`.

---

## 3. Data Flow

```
User Query ──┬──> Embedder ─────────> Vector Store (Dense Search) ──────┐
             │                                                          ├─> Fusion (RRF) ──> Ranked Candidates
             └──> Tokenizer ────────> BM25 Indexer (Active Memories) ───┘
```

---

## 4. Acceptance Criteria
1. **Keyword Exact Matches:** Queries with specific terms (e.g. "PostgreSQL", "Bangalore") rank facts containing those exact terms at the top via BM25 even if embeddings have low differentiation.
2. **Paraphrase Matches:** Semantic queries without exact keyword overlap retrieve relevant facts via vector similarity.
3. **Supersession Exclusion:** Outdated memories marked `status=MemoryStatus.SUPERSEDED` are excluded from results.
4. **Per-User Isolation:** Search strictly restricted to `user_id`; never returns memories of another user.
5. **Quality Gates:** Passes `ruff format --check`, `ruff check`, `mypy src tests`, and pytest suite.

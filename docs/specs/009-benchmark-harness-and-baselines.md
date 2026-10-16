# Spec 009: Benchmark Evaluation Harness & Baselines

## 1. Goal
Implement the benchmark evaluation harness and baseline agents for Milestone 5 (Phases 5.1, 5.2, 5.3) as specified in `AGENTS.md` and `docs/ROADMAP.md`. The harness enables reproducible, transparent, and ablation-ready benchmarking across LongMemEval and LoCoMo datasets comparing Full-Context, Naive RAG, and MemLayer agents.

---

## 2. Requirements & Architecture

### 2.1 Benchmark Dataset Loaders & Splits (Phase 5.1)
- **Data Models (`eval/datasets/models.py`)**:
  - `EvalTurn`: represents a message in background history (`role: str`, `content: str`, `timestamp: datetime | None`, `session_id: str | None`).
  - `EvalQuestion`: represents a benchmark evaluation item (`id: str`, `benchmark: str`, `category: str`, `user_id: str`, `conversation: list[EvalTurn]`, `question: str`, `ground_truth: str`, `split: str = "dev"`, `metadata: dict[str, Any]`).
  - Standard Categories:
    - `knowledge_update`: fact modification/supersession where agent must answer with updated information.
    - `temporal_reasoning`: queries requiring time math, sequence, or event timestamps.
    - `multi_session`: facts spread across separate sessions days/weeks apart.
    - `abstention`: queries asking about unmentioned or unknown facts where agent must abstain instead of hallucinating.
- **Curated Dataset Fixtures (`eval/datasets/fixtures/`)**:
  - `longmemeval_sample.json`: Representative test scenarios for LongMemEval across all 4 categories.
  - `locomo_sample.json`: Representative multi-session scenarios for LoCoMo across all 4 categories.
- **Dataset Loader (`eval/datasets/loader.py`)**:
  - `load_eval_dataset(benchmark: str, split: str = "dev", category: str | None = None, limit: int | None = None, fixtures_dir: Path | None = None) -> list[EvalQuestion]`
- **Subsets Configuration (`eval/configs/subsets.yaml`)**:
  - Configuration mapping benchmark names, dev/test split parameters, category lists, and default limits.

### 2.2 Baseline Implementations (Phase 5.2)
- **Base Baseline Interface (`eval/baselines/base.py`)**:
  - `BaselineOutput`: structured result (`prediction: str`, `prompt_tokens: int`, `completion_tokens: int`, `total_tokens: int`, `latency_ms: float`, `context_used: str`, `metadata: dict[str, Any]`).
  - `BaseBaseline(ABC)`:
    - `async def setup(self, item: EvalQuestion) -> None`: ingests background conversation.
    - `async def answer(self, item: EvalQuestion) -> BaselineOutput`: executes retrieval/prompt construction and calls answer LLM.
    - `async def teardown(self) -> None`: cleans up resources/stores.
- **Full-Context Baseline (`eval/baselines/full_context.py`)**:
  - Concatenates the entire conversation history verbatim into prompt context window.
  - Measures upper bound on context availability and token usage.
- **Naive RAG Baseline (`eval/baselines/naive_rag.py`)**:
  - Ingests conversation turns into a simple vector store.
  - Retrieves top-k chunks strictly by vector cosine similarity (no recency decay, no BM25 hybrid, no turn gating, no conflict supersession).
- **MemLayer Agent Baseline (`eval/baselines/memlayer_agent.py`)**:
  - Ingests conversation through `MemoryLayer.add(...)` (fact extraction, temporal extraction, deduplication, conflict supersession).
  - Retrieves through `MemoryLayer.search(...)` (hybrid BM25 + vector, multi-factor scoring `score = a*sim + b*exp(-age/tau) + c*importance`, turn gating router, token-budgeted formatting).

### 2.3 Benchmark Runner, Disk Caching & LLM Judge (Phase 5.3)
- **Deterministic Disk Cache (`eval/cache.py`)**:
  - Computes SHA256 key over `(model, prompt, kwargs)`.
  - Caches responses in `eval/cache/<hash>.json` to guarantee reproducible runs and prevent redundant API spend.
- **Standardized LLM Judge (`eval/judge.py`)**:
  - Standardized evaluation prompt taking Question, Ground Truth, Model Prediction, and Category.
  - Returns `JudgeResult(score: float, is_correct: bool, rationale: str)`.
  - Enforces category-specific criteria (e.g. abstinence verification for `abstention`, recency check for `knowledge_update`).
- **Benchmark Runner CLI (`eval/run.py`)**:
  - Command-line arguments: `--baseline`, `--benchmark`, `--split`, `--category`, `--limit`, `--seed`, `--output-dir`, `--no-cache`.
  - Aggregates and reports:
    - Overall accuracy and per-category accuracy (`knowledge_update`, `temporal_reasoning`, `multi_session`, `abstention`).
    - Latency distribution (mean, p50, p95 in ms).
    - Token metrics (mean prompt tokens, mean completion tokens, total tokens).
    - Cost estimation.
  - Persists full execution artifact to `eval/results/<run_id>.json`.

---

## 3. Acceptance Criteria
1. **Loaders & Schema Validation:** Dataset loader successfully parses fixtures and validates against Pydantic models.
2. **Baseline Parity:** All 3 baselines (`full_context`, `naive_rag`, `memlayer`) adhere to `BaseBaseline` contract and track tokens + latency.
3. **Deterministic Caching:** Disk cache returns saved responses on cache hit and avoids network round-trips.
4. **Standardized Judge:** LLM judge accurately evaluates correctness across all 4 categories with explanatory rationales.
5. **Execution & Metrics:** `eval/run.py` executes end-to-end and outputs structured JSON results and formatted summary tables.
6. **Quality Gates:** 100% test coverage in `tests/test_eval.py`, passes `ruff`, `mypy`, and `pytest`.

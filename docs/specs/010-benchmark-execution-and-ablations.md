# Spec 010: Baseline vs MemLayer Benchmark Execution & Ablation Studies

## 1. Goal
Execute rigorous, reproducible benchmark evaluations for Milestone 5 (Phases 5.4 and 5.5) as defined in `AGENTS.md` and `docs/ROADMAP.md`. Compare Full-Context, Naive RAG, and MemLayer agents across LongMemEval and LoCoMo benchmarks, execute 4 systematic ablation studies on MemLayer, analyze real failure cases, and publish the verified report in `docs/EVALUATION_REPORT.md`.

---

## 2. Requirements & Architecture

### 2.1 Baseline vs MemLayer Evaluation (Phase 5.4)
- **Datasets**:
  - `LongMemEval` (dev split: 4 multi-session scenarios across knowledge update, temporal reasoning, multi-session synthesis, and abstention).
  - `LoCoMo` (dev split: 4 multi-session scenarios across knowledge update, temporal reasoning, multi-session synthesis, and abstention).
- **Architectures Evaluated**:
  1. `Full-Context Baseline`: Verbatim concatenation of conversation turns into the LLM context window.
  2. `Naive RAG Baseline`: Fixed top-k dense vector retrieval (no recency decay, no BM25, no router gating, no conflict supersession).
  3. `MemLayer Agent`: Full Agent Memory Layer (fact extraction, conflict detection & supersession, hybrid BM25 + dense retrieval, multi-factor recency/importance scoring, turn router).
- **Execution & Storage**:
  - Run all baselines through `eval/run.py`.
  - Save full JSON run artifacts to `eval/results/<benchmark>_<baseline>_<timestamp>.json`.
  - Compute overall accuracy, per-category accuracy, latency distribution (mean, p50, p95), token efficiency (prompt tokens, completion tokens, total), and cost estimation.

### 2.2 Systematic Ablation Studies (Phase 5.5)
Execute 4 planned ablations isolating individual memory layer components:
1. `without_decay`: Memory importance decay is disabled (`MemoryDecayEngine` omitted / frequency decay inactive).
2. `without_supersession`: Conflict detection and supersession disabled (`ConflictDetector` inactive, older outdated memories remain active).
3. `vector_only`: Sparse BM25 keyword matching disabled in `HybridRetriever` (`dense_weight = 1.0, sparse_weight = 0.0`).
4. `without_recency`: Recency factor omitted in multi-factor scoring formula (`b = 0.0` in `ScoringWeights(a=0.7, b=0.0, c=0.3)`).

### 2.3 Failure Analysis & Evaluation Report
- Inspect and document 5 to 10 failure modes from benchmark runs and ablations.
- Detail why each failure occurred (e.g., semantic confusion in vector-only search, retrieval of outdated facts when supersession was ablated, temporal miscalculation).
- Publish comprehensive report in `docs/EVALUATION_REPORT.md` including:
  - Benchmark Results Summary Table.
  - Efficiency Table (Accuracy vs Tokens/Query vs p50/p95 Latency vs Cost).
  - Ablation Study Table (Accuracy and Category delta vs Full MemLayer).
  - Deep-dive Failure Analysis with architectural lessons learned.

---

## 3. Acceptance Criteria
1. **Zero Fabrication:** All numbers in `docs/EVALUATION_REPORT.md` must directly reference saved runs in `eval/results/`.
2. **Four Verified Ablations:** All 4 required ablations (`without_decay`, `without_supersession`, `vector_only`, `without_recency`) successfully executed and logged.
3. **Multi-Category Reporting:** Accuracy reported per category (`knowledge_update`, `temporal_reasoning`, `multi_session`, `abstention`) in addition to overall scores.
4. **Failure Analysis:** Minimum 5 concrete failure cases analyzed with root causes and mitigations documented.
5. **Quality Gates:** 100% test pass, 0 ruff errors, 0 mypy errors.

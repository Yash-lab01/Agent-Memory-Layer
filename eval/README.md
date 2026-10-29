# `memlayer` Evaluation Harness & Methodology

This document details the experimental methodology, scientific controls, dataset protocols, and historical audit for benchmarking the Agent Memory Layer (`memlayer`) against baselines on **LongMemEval** and **LoCoMo**.

---

## 1. Historical Audit of Initial Evaluation Runs

Before undertaking large-scale evaluation, an internal audit of the initial test harness (`eval/`) was conducted. The findings are documented here transparently:

### Audit Findings Matrix

| Question | Audit Finding | Root Cause & Implication |
| :--- | :--- | :--- |
| **1. Were LLMs real or stubs?** | **Stubs/Fallbacks in CI test runs.** | `BaseBaseline._generate_llm_response` and `LLMJudge.evaluate` catch unauthenticated exceptions and invoke rule-based string matching (`_fallback_generate`). The recorded latencies (2–30 ms) reflect local string operations, not network calls. |
| **2. Was the embedder real?** | **64-dimensional mock hash embedder.** | `MemLayerBaseline` instantiated `MemoryLayer.create_local()` with default `MockEmbedder(dimension=64)`, performing pseudo-random vector hashing rather than semantic dense representation. |
| **3. How were the 4 questions chosen?** | **Hand-written synthetic samples.** | `eval/datasets/fixtures/` contained exactly 4 synthetic questions (1 per category: `knowledge_update`, `temporal_reasoning`, `multi_session`, `abstention`), not samples from the official LongMemEval or LoCoMo datasets. |
| **4. Were outputs cached and saved?** | **Yes, cached and saved in JSON.** | `DiskCache` correctly keyed completions on SHA-256 hashes of `(model, prompt, kwargs)`. Results were saved to `eval/results/<run_id>.json`. However, retrieval IDs and evidence IDs were not saved for recall@k metrics. |
| **5. Were answer models and judges uniform?** | **Uniform in configuration, diverged in heuristics.** | Both baselines and judge specified uniform models (`gemini-1.5-flash` and `gemini-1.5-pro`), but under offline conditions, heuristic branches diverged based on context strings. |

> **Classification:** The initial 4-question runs served as **functional smoke tests** verifying pipeline wiring and CI execution. In Milestone 9, these smoke tests are preserved as fast CI checks (`make eval-smoke`), while the real scientific benchmarks (`make eval-full`) are established with real models, full datasets, and statistical confidence intervals.

---

## 2. Experimental Design & Methodology (Frozen)

All real benchmark runs are governed by strict controls frozen in `eval/configs/main.yaml`:

### 2.1 Model Selection
- **Answer Model (Unified):** One model across all systems (Full-Context, Naive RAG, BM25+Reranker, MemLayer). Default: hosted mid-size model (`gemini/gemini-1.5-flash` or `gpt-4o-mini`).
- **Judge Model (Standardized):** Benchmark-recommended judge model (`gemini/gemini-1.5-pro` or `gpt-4o`) operating with category-specific evaluation prompts.
- **Extraction Model (MemLayer only):** Lightweight, economical model (`gemini/gemini-1.5-flash` or local `ollama/llama3.2`) reported explicitly.
- **Embedding Model:** Real dense embedding model (`BAAI/bge-small-en-v1.5` or `text-embedding-3-small`), replacing the 64-dim mock embedder.
- **Sampling Parameters:** Temperature strictly `0.0` across answer generation, judging, and extraction. Fixed deterministic random seeds.

### 2.2 Dataset Splits & Hyperparameter Tuning
- **Split Ratio:** Questions are partitioned into **Dev** (20–30%) and **Test** (70–80%).
- **Strict Leakage Prevention:** Scoring weights ($\alpha, \beta, \gamma, \tau$), retrieval top-$k$, and token budgets are tuned **only on the Dev split**.
- **Test Set Execution:** The Test split is executed **once** per final configuration. No post-hoc tuning on test data is permitted.

### 2.3 Chronological Replay & Temporal Anchoring
- **Order of Replay:** Historical dialogue sessions are ingested in strict chronological order with original session timestamps.
- **Anchored "Now" Reference:** The question's stated date is explicitly passed into retrieval scoring as the temporal anchor. Using the runtime wall-clock on historical datasets produces invalid recency decay.
- **Tenant Isolation:** Every haystack/question operates within an isolated `user_id` namespace. No memory is shared across evaluation questions.

---

## 3. Evaluated Systems & Baselines

All systems receive the identical question prompt and answer model; only the injected context differs:

```
+-------------------------------------------------------------------------------+
| System                       | Context Injected into Prompt                   |
+==============================+================================================+
| 1. Full-Context Baseline     | Entire conversation haystack (verbatim)        |
| 2. Naive RAG Baseline        | Chunked dialogue turns, dense top-k similarity |
| 3. BM25 + Reranker Baseline  | Sparse BM25 + dense rerank over raw chunks     |
| 4. MemLayer (Full)           | Atomic facts + supersession + hybrid scoring   |
+-------------------------------------------------------------------------------+
```

The **BM25 + Reranker** baseline over raw conversation chunks is critical: it isolates whether extracting atomic facts provides measurable accuracy over standard chunk-level RAG.

---

## 4. Statistical Metrics & Cost Accounting

Every evaluation output is logged per question with complete telemetry:

### 4.1 Statistical Rigor
- **Bootstrap 95% Confidence Intervals:** Computed via 10,000 bootstrap resamples for all accuracy metrics.
- **Paired Comparisons:** Paired bootstrap differences and McNemar's tests across identical questions between systems. If the 95% CI of the difference crosses zero, it is reported as "no statistically significant difference."
- **Retrieval Recall@k:** Measured against annotated `evidence_ids` to decouple retrieval misses from answer-model generation errors.

### 4.2 Honest Amortized Cost Accounting
Rather than claiming token savings solely at query time, `memlayer` reports total cost:

$$\text{Total Cost}(Q) = \text{Ingestion Cost} + Q \times \text{Query Cost}$$

Where:
- $\text{Ingestion Cost}$: LLM extraction tokens + embedding tokens for ingesting the haystack.
- $\text{Query Cost}$: Retrieval prompt tokens + answer completion tokens.
- **Break-Even Query Count ($Q^*$):** The minimum number of questions asked over a single conversation history before `memlayer` becomes cheaper than Full-Context:
$$Q^* = \frac{\text{Ingestion Cost}}{\text{Full-Context Query Cost} - \text{MemLayer Query Cost}}$$

---

## 5. Judge Validation & Quality Control

Before large-scale benchmarking:
1. **Human Agreement Sampling:** 30–50 random judge verdicts are independently graded by hand.
2. **Agreement Threshold:** Agreement rate between human evaluation and LLM judge must meet or exceed **90%** before executing test runs.
3. **Failure Analysis:** At least 10 incorrect outputs are categorized into:
   - *Extraction Miss:* Fact was omitted during async ingestion.
   - *Supersession Failure:* Contradiction was not linked; old memory remained active.
   - *Retrieval Miss:* Fact was active in store but missed top-$k$ (low recall@k).
   - *Budget Truncation:* Scored fact was omitted due to token budget limit.
   - *Answer Model Error:* Fact was present in context, but model failed to answer.
   - *Judge Error:* Model answer was correct, but judge misclassified it.

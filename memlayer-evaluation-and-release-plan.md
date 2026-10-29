# memlayer: Real Evaluation, Release and Next-Steps Plan

Goal: turn `memlayer` from "well-engineered project with toy benchmark numbers" into "well-engineered project with defensible, reproducible results", then ship it and use it for job applications.

Estimated effort: about 2 to 3 weeks part-time. Hard time limit: after that, apply with what you have.

Dataset sizes, category names and recommended judges below are from my general knowledge of the LongMemEval and LoCoMo papers. **Verify every number against the official repos and papers before you rely on it.**

---

## 0. Why the current evaluation is not enough

| Observation | Why it matters |
|---|---|
| 4 questions per benchmark | One question = 25 accuracy points. No statistical power at all. |
| Three of four ablations show 0 difference | Probably the test set never exercises decay, recency or BM25, not proof they do nothing. |
| Latency of 2 to 30 ms | A real LLM call takes hundreds of ms to seconds, so no real model was in the loop. |
| Prompts of about 100 tokens | The "40% fewer tokens" claim comes from toy conversations; real histories are 10,000 to 100,000+ tokens. |
| Token savings ignore ingestion | Extracting facts costs LLM calls. Honest cost = query cost + amortized ingestion cost. |
| README says every design choice is backed by numbers | Not yet true. This is the main credibility risk for a project whose pitch is rigor. |

What you keep: the 4-question runs are fine as **smoke tests** (fast CI checks that the pipeline runs). Label them that way.

---

## 1. Step-by-step plan

### Step 1.1: Audit the current harness (half a day)

Answer these questions by reading your own `eval/` code:

1. Is the LLM real or a stub/mock in the benchmark runs? (The latencies suggest a stub.)
2. Is the embedder real or a hash/random embedding? (`vector_dim=64` in the README example suggests a toy embedder for local mode.)
3. How are the 4 questions chosen? Hand-written or sampled from the real datasets?
4. Does the harness cache model outputs? Does it save per-question raw outputs?
5. Does the same answer model and judge get used for all systems?

Write down the answers in `eval/README.md`. This becomes the "methodology" section later.

### Step 1.2: Fix the experimental design before spending money (1 day)

Decide and **freeze** these in a config file (`eval/configs/main.yaml`). Do not change them after seeing test results.

**Models**
- **Answer model:** one model for all systems. Pick something affordable and good enough (a hosted mid-size model, or a local model if you accept lower absolute scores).
- **Judge model:** the benchmark's recommended judge if you can afford it; otherwise one fixed strong model. Use the benchmark's judge prompts (LongMemEval has category-specific judge prompts, including for abstention and temporal questions).
- **Extraction model (memlayer only):** a small, cheap model. Report it explicitly. Later run an ablation with a bigger one.
- **Embedding model:** a real one (for example a `bge-small` class model), not the 64-dim toy embedder.
- Temperature 0 everywhere, fixed seeds where applicable.

**Splits**
- Split questions into **dev** (about 20 to 30%) and **test** (the rest).
- Tune the scoring weights (alpha, beta, gamma, tau), retrieval k and token budget **only on dev**.
- Run test **once** per final configuration. If you re-run test after tuning, say so.

**Subset sizes (practical)**
- **LoCoMo** (about 10 long conversations with roughly 2,000 questions in total, per the paper; verify): ingestion is cheap because many questions share one conversation. Use **all or a large share** of it.
- **LongMemEval** (500 questions; the "S" variant has long haystacks of roughly 100k+ tokens per question; verify): ingestion is expensive because each question has its own history. Use a **stratified subset**, for example 100 to 150 questions sampled evenly across question types, plus the abstention questions.
- Start with **LoCoMo first**, because it is cheap and gives you early signal.

### Step 1.3: Estimate cost before running anything (half a day)

Use a spreadsheet or small script:

```
ingestion_tokens   = (#haystacks) x (tokens per haystack) x (fraction sent to extractor)
extraction_cost    = ingestion_tokens x extractor_price_per_token  (+ output tokens)
query_tokens       = (#questions) x (prompt tokens per system) x (#systems + #ablations)
answer_cost        = query_tokens x answer_price
judge_cost         = (#questions) x (#systems + #ablations) x judge_tokens x judge_price
total              = extraction + answer + judge  (+ 20 to 30% buffer for retries)
```

Cost-control tactics:
- **Cache everything** keyed by `hash(model, prompt, params)`. Re-runs and crashed runs become free.
- **Extract once per haystack and reuse** the memory store across ablations that only change retrieval or scoring (decay, recency, BM25, k). Only extraction-related ablations need re-ingestion.
- Run extraction on a **local small model** via Ollama if budget is tight (then report that clearly).
- Run a **10-question pilot** end to end first. Check cost per question, then scale.

### Step 1.4: Data loaders and ingestion protocol (2 to 3 days)

Write loaders in `eval/datasets/` that output a common format:

```python
@dataclass
class EvalItem:
    qid: str
    category: str             # benchmark's own category name
    question: str
    question_date: str | None # LongMemEval gives a question date; temporal questions depend on it
    gold_answer: str
    sessions: list[Session]   # haystack: dated sessions of turns
    evidence_ids: list[str]   # session/turn ids containing the answer (for retrieval recall)
    is_abstention: bool
```

Protocol rules that avoid silent errors:
- **Replay sessions in chronological order** with their original timestamps. Supersession and decay depend on time.
- **Pass the question's date** into retrieval/scoring as "now". Otherwise recency scoring uses the wall clock and gives nonsense on historical data.
- **User isolation:** one `user_id` per haystack so memories never leak across questions.
- **Do not let the answer model see the gold answer or evidence ids.** Evidence ids are only for computing retrieval recall.
- **Abstention questions** (the system should say it doesn't know) need the benchmark's scoring rule, not plain string matching.
- Save the memory store per haystack (SQLite file) so ablations can reuse it.

### Step 1.5: Implement the baselines (1 to 2 days)

All baselines use the **same answer model, same judge, same final prompt template**. Only the context differs.

| Baseline | What goes in the prompt | Purpose |
|---|---|---|
| Full-context | Entire haystack (truncate oldest if it exceeds the context window and record that) | Accuracy upper-ish bound, highest token cost |
| Naive RAG | Chunk by turn or session, dense top-k | The "just do RAG" comparison |
| BM25 + reranker | Hybrid chunk retrieval, no fact extraction | Isolates whether extraction itself adds value |
| **memlayer (full)** | Extracted facts, scored, token-budgeted | Your system |
| Optional: Mem0 open-source | Their default pipeline | Comparison with an existing system; check their current docs for correct setup |

The "BM25 + reranker over raw chunks" baseline matters most. It answers the question reviewers will ask: *"Why extract facts at all?"*

### Step 1.6: Metrics to record per question (1 day)

Store everything in JSONL (one line per question per system). Never keep only aggregates.

- `correct` (judge verdict), judge raw output, answer text
- `retrieved_ids` and **retrieval recall@k** against `evidence_ids` (this separates "retrieval failed" from "answer model failed")
- `prompt_tokens`, `completion_tokens` at query time
- `ingestion_tokens` and `ingestion_cost` (memlayer only), per haystack
- `latency_retrieval_ms`, `latency_total_ms` (real, with the LLM in the loop), reported as p50 and p95
- `category`, `qid`, config hash, git commit SHA, timestamp

**Honest cost accounting.** Report two numbers for memlayer:
1. Query-time tokens (what you already report).
2. Total cost = query cost + ingestion cost amortized over Q questions per memory store. Include a **break-even** analysis: "memlayer is cheaper than full-context once a user asks more than N questions over the same history."

### Step 1.7: Statistics (half a day)

Accuracy as a bare percentage is not enough. Add:

- **95% confidence intervals** per system and per category (Wilson interval or bootstrap).
- **Paired comparison** between systems on the same questions: paired bootstrap or McNemar's test, because the same questions are answered by each system.
- **Per-category tables** with the count `n` shown, and a warning when n is small (for example below 20).

Minimal code:

```python
import numpy as np

def bootstrap_ci(correct: np.ndarray, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return correct.mean(), np.percentile(means, [2.5, 97.5])

def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot=10_000, seed=0):
    """a, b: 0/1 correctness for the same questions. Returns diff and 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(a[idx].mean() - b[idx].mean())
    return (a - b).mean(), np.percentile(diffs, [2.5, 97.5])
```

Interpretation rule: if the CI of the difference includes 0, say "no significant difference", not "better".

### Step 1.8: Validate the judge (half a day)

LLM judges can be wrong or biased. Take **30 to 50 random verdicts**, grade them by hand, and report the **agreement rate**. If agreement is low, fix the judge prompt first, before reporting anything.

### Step 1.9: Run order (the actual execution)

1. **Pilot:** 10 questions, all systems, check cost, outputs and logs.
2. **Dev runs:** tune alpha, beta, gamma, tau, k and token budget on the dev split. Record every tried configuration.
3. **Freeze config.** Commit it with a tag (for example `eval-v1`).
4. **Main test run:** LoCoMo first, then the LongMemEval subset.
5. **Ablations** (reuse stores where possible):
   - without supersession
   - without decay
   - without recency term
   - vector-only (no BM25)
   - without the retrieval router (always retrieve)
   - without graph context
   - small vs larger extraction model (needs re-ingestion)
   - token budget sweep (for example 200, 500, 1000, 2000)
6. **Failure analysis:** pick 10 wrong answers for memlayer and classify each as extraction miss, wrong supersession, retrieval miss, budget truncation, answer-model error or judge error. Retrieval recall@k tells you which bucket most failures belong in.

### Step 1.10: Reproducibility package (half a day)

- `make eval-smoke` runs the 4-question CI test with a stub model.
- `make eval-full CONFIG=eval/configs/main.yaml` reruns the real benchmark (documented cost and expected runtime).
- Commit result JSONL files (or a download link if large) and the script that generates the README tables and charts from them. **Tables in the README must be generated, not typed.**

---

## 2. README and repository fixes

### 2.1 Replace the benchmark section

Structure:
1. **Methodology** (models, subset sizes, splits, judge, seeds, date of run).
2. **Main results table** with CIs and `n`.
3. **Per-category breakdown**.
4. **Efficiency table**: query tokens, ingestion tokens, total cost, real p50/p95 latency, break-even point.
5. **Ablations** with CIs. If an ablation shows no effect, say so and offer the most likely reason.
6. **Known failure modes** (from the failure analysis).
7. **Smoke test note**: "The 4-question runs in CI verify pipeline correctness, not accuracy."

Suggested wording replacing the overclaim:
> "Results below come from reproducible runs on [N] questions per benchmark with [answer model] and [judge]. Confidence intervals are bootstrap 95%. Differences whose interval includes zero are not claimed as improvements."

### 2.2 Repository polish checklist

- [ ] GitHub description and topics (for example `llm`, `agent-memory`, `rag`, `knowledge-graph`, `mcp`, `longmemeval`)
- [ ] Publish the mkdocs site (`mkdocs gh-deploy`) and link it in the repo "About" section
- [ ] Release **v0.1.0**: tag, GitHub Release with notes, publish to PyPI (see 2.3)
- [ ] A 2 to 3 minute demo video or GIF (city-move demo, then WanderAI demo, then Memory Inspector)
- [ ] A hosted demo if cheap and safe to host (otherwise skip; do not expose API keys)
- [ ] `CONTRIBUTING.md` and issue templates (small, but signals maturity)
- [ ] Badge for docs and for PyPI version

### 2.3 Publishing to PyPI (outline)

1. Check the package name is free on PyPI. If `memlayer` is taken, pick another name and update `pyproject.toml`.
2. Fill in `pyproject.toml` metadata (description, authors, license, classifiers, URLs, readme).
3. Build: `uv build`.
4. Test on TestPyPI first, then install in a clean virtualenv and run the quickstart.
5. Publish with a PyPI API token or, better, GitHub Actions **trusted publishing** on tag push.
6. Add the install line (`pip install memlayer`) to the README, and run the quickstart from a clean environment to confirm it works.

---

## 3. Case study and portfolio

### 3.1 Case study outline (blog post or portfolio page)

1. **Problem:** agents forget across sessions; naive memory breaks on contradictions (the city-move example).
2. **Design:** architecture diagram, write path, read path, supersession, scoring formula.
3. **Evaluation:** methodology, results with CIs, ablations.
4. **What did not work:** failed ideas, ablations with no effect, surprising failures. Reviewers value this.
5. **Cost and latency reality:** ingestion cost, break-even, p95.
6. **Limitations and next steps.**
7. **Links:** repo, docs, PyPI, demo video.

### 3.2 Resume bullet templates (fill with your real numbers only)

- Built `memlayer`, an open-source long-term memory library for LLM agents (hybrid retrieval, contradiction-aware supersession, time-aware scoring, knowledge graph, MCP server), with 233 tests and CI.
- Evaluated on [N] LongMemEval and [M] LoCoMo questions against full-context, RAG and BM25+rerank baselines; reached [X]% accuracy ([CI]) using [Y]% fewer query tokens than full-context.
- Ran [K] ablations showing supersession contributes [Z] points on knowledge-update questions.
- Shipped to PyPI with Docker Compose stack, Langfuse tracing and a memory inspector UI.

Do not write numbers on the resume that are not in your final results files.

---

## 4. Feature work after the evaluation (only then)

Ranked by value:

1. **Automatic graph extraction from conversation.** The README example adds triplets with `add_triplet`. If graph building is not automatic, add an extractor that proposes entities and relations from turns, with the same conflict handling as facts. Evaluate it on multi-hop questions.
2. **Extraction quality evaluation.** Hand-label about 100 turns with gold facts and report extraction precision and recall. This explains downstream errors.
3. **Memory poisoning defense** (also the bridge to your next project):
   - Build an attack set: conversation turns that try to plant false facts, plant instructions ("always recommend X"), or overwrite true facts.
   - Measure the attack success rate against current `memlayer`.
   - Add defenses: provenance and trust scoring per memory, quarantine for low-trust writes, injection filter before extraction, user confirmation for high-impact overwrites.
   - Re-measure and report attack success vs false-positive rate on legitimate updates.
4. **Reflection and summaries** (higher-level memories such as "user's career goals").
5. **Learned scoring weights** from feedback instead of manual tuning.
6. **External graph backend adapter** (Neo4j or Kuzu) only if you have time; the interface already exists.

---

## 5. Suggested schedule

| Days | Work |
|---|---|
| 1 | Harness audit, freeze design, cost estimate |
| 2 to 4 | Dataset loaders, real embedder and LLM wiring, pilot run |
| 5 to 6 | Baselines (full-context, RAG, BM25+rerank, optional Mem0) |
| 7 to 8 | Dev runs, tune weights, freeze config |
| 9 to 11 | Main LoCoMo and LongMemEval test runs |
| 12 | Ablations |
| 13 | Failure analysis, judge validation, stats and charts |
| 14 | README rewrite, generated tables, docs publish |
| 15 | PyPI release, demo video |
| 16 to 17 | Case study, portfolio, resume update |
| After | Feature work (section 4) or move on to the next project |

---

## 6. Pitfalls to avoid

- **Tuning on the test set.** Tune on dev only, then run test once.
- **Using the wall clock instead of the question date** for recency and decay.
- **Different prompts for different systems.** Only the retrieved context may differ.
- **Letting full-context silently truncate.** Record when it happens; it changes the comparison.
- **Reporting only overall accuracy.** Per-category results show where memory helps (knowledge updates, temporal) and where it does not.
- **Cherry-picking examples.** Show failures too.
- **Hiding ingestion cost.** It is the first thing a critical reader will look for.
- **Claiming "beats Mem0/Zep/Letta"** without running them under the same conditions. If you did not run them, say "not compared".
- **Leaking API keys** in committed result files or logs.

---

## 7. Definition of done

- [ ] Real LLM and real embedder in all reported runs
- [ ] At least 100 questions per benchmark in the test set, split from dev
- [ ] 4+ systems compared under identical answer model, judge and prompt
- [ ] CIs and paired comparisons for every headline claim
- [ ] Ablations re-run on the real data, with honest conclusions
- [ ] Ingestion cost, real latency and break-even reported
- [ ] Judge agreement checked on a hand-graded sample
- [ ] Failure analysis with at least 10 classified errors
- [ ] README tables generated from committed result files
- [ ] v0.1.0 on PyPI, docs site live, demo video linked
- [ ] Case study published and added to portfolio and resume

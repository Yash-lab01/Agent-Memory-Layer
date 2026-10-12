# AGENTS.md: Agent Memory Layer

Instructions for any AI coding agent working in this repository (Antigravity, Claude, Gemini, Cursor, Copilot, etc.). Read this file fully before doing anything. If an instruction here conflicts with a user prompt, follow the user prompt for that task, then tell the user about the conflict.

---

## 1. Project Context

- **Name:** Agent Memory Layer (working name: `memlayer`; final name TBD, see Open Decisions)
- **What it is:** An open-source Python library plus a small demo app that gives any chat agent durable long-term memory across sessions. It extracts atomic facts from conversations, stores them with time and provenance, retrieves the right ones at the right moment, and keeps the store clean by merging duplicates, decaying stale facts, and resolving contradictions.
- **The real goal:** A **measured** answer to "how much does each memory design choice improve a conversational agent?" Benchmarks and ablations are the product. The library exists to make them possible.
- **Benchmarks:** LongMemEval and LoCoMo. Baselines: full-context and naive RAG.
- **Positioning:** Mem0, Letta (formerly MemGPT) and Zep/Graphiti exist. Never claim novelty. The angle is rigor, transparency, reproducible numbers, and a small readable codebase. Verify competitors' current docs before making any comparison claim.
- **Owner's goal:** Portfolio-quality project for conversational AI / LLM systems roles. Every design choice must be explainable in an interview.
- **Current phase:** see `docs/ROADMAP.md`.

## 2. Start-of-Session Protocol

Before writing code:

1. Read this file.
2. Read `docs/ROADMAP.md` to find the current phase and next unchecked task.
3. Read `docs/ARCHITECTURE.md` and `docs/DECISIONS.md` if the task touches design.
4. Read the relevant spec in `docs/specs/` if one exists.
5. Run `git status` and `git log --oneline -10`.
6. Summarize your understanding of the task in 3 to 5 lines and confirm before any large change.

Do not assume context from earlier chats. The repo and its docs are the only source of truth.

## 3. Repository Structure

```
agent-memory-layer/
  AGENTS.md
  README.md
  src/memlayer/
    api.py              # MemoryLayer class (public API)
    models.py           # Pydantic v2 schemas
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
  demo/app/             # FastAPI + simple UI
  tests/
  docs/
    PRD.md
    ARCHITECTURE.md
    DECISIONS.md
    ROADMAP.md
    specs/              # NNN-feature.md
  docker-compose.yml
  pyproject.toml
  .env.example
```

Keep this layout. Do not create new top-level folders or modules without asking.

## 4. Core Design (do not drift from this without logging a decision)

### 4.1 Public API

```python
mem = MemoryLayer(store=..., llm=..., embedder=...)
mem.add(user_id, messages)           # extract + store
ctx = mem.search(user_id, query)     # retrieve
mem.forget(user_id, memory_id)       # delete
```

Backends (LLM, embedder, store) are pluggable. **Per-user isolation (namespaces) from day one.** Every read and write is scoped by `user_id`. A query must never be able to return another user's memories.

### 4.2 Write path
- Extraction runs **async, after the reply is sent**, never blocking the user.
- Extractor returns atomic, self-contained facts as JSON validated with Pydantic. Retry on malformed JSON.
- Per-memory metadata: `entity`, `content`, `created_at` (when said), `event_time` (when it happened, if stated), `source_episode_id`, `confidence`, `importance`, `status` (`active` or `superseded`).
- Raw messages are always kept in the episode log for provenance.

### 4.3 Read path
- Hybrid retrieval: vector similarity + keyword/BM25 + metadata filters.
- Scoring: `score = a * similarity + b * exp(-age / tau) + c * importance`. The values a, b, c and tau are hyperparameters, tuned **only on a dev split**, and reported.
- Retrieval router: a cheap check skips memory lookup for turns that do not need it ("thanks!").
- Token-budgeted injection: top-k within a token budget, formatted with timestamps.

### 4.4 Consolidation
- **Never hard-delete on contradiction.** Mark the old memory `superseded`, set `superseded_by`, keep history.
- Dedup merges near-duplicates. Decay lowers weight of old unused memories and boosts frequently retrieved ones.
- Runs as a background job (APScheduler to start; Celery + Redis only if needed).
- Explicit deletion (`forget`) is the only hard delete.

### 4.5 Data model (SQL)
- `episodes(id, user_id, session_id, role, text, ts)`
- `memories(id, user_id, content, entity, event_time, created_at, importance, confidence, status, superseded_by, source_episode_id, last_accessed_at, access_count)`
- Vectors live in the vector store keyed by `memory_id`.

## 5. Tech Stack and Rules

| Layer | Choice |
|---|---|
| Language | Python 3.11+, full type hints |
| Schemas | Pydantic v2 |
| LLM access | LiteLLM (Ollama, Groq, Gemini behind one interface) |
| Relational | SQLite (local), Postgres (server), SQLAlchemy 2.0 |
| Vector store | pgvector or Qdrant (**open decision**, see section 12) |
| Keyword search | `rank-bm25` or Postgres full-text |
| API / MCP | FastAPI; official Python MCP SDK |
| Observability | Langfuse or OpenTelemetry |
| Quality | ruff, mypy, pytest, pre-commit |
| CI | GitHub Actions: lint, test, small eval smoke test |
| Packaging | `uv` or Poetry; publishable to PyPI |
| Containers | Docker Compose (one command to run the demo) |
| Docs | MkDocs Material |

- Do not add dependencies without asking. State why and what it replaces.
- Keep the core library free of demo-only dependencies (FastAPI, UI) so `pip install` stays light. Use optional extras.
- All model names, weights, and hyperparameters live in config files, never hard-coded.
- Never commit API keys. Add new variable names to `.env.example` only.

## 6. Working Method

For every feature:

1. **Spec first.** If none exists, draft `docs/specs/NNN-feature.md` (goal, inputs/outputs, acceptance criteria, out of scope). Wait for approval.
2. **Plan before code.** Present files to touch, approach, and risks. Wait for approval on anything non-trivial.
3. **Small vertical slices.** One task from the current phase per change, ending in something runnable.
4. **Tests alongside code.** Required for scoring, conflict resolution, dedup, and decay logic (see section 8).
5. **Verify.** Run tests, lint, mypy, and the code itself. Report actual output. Never claim something works without running it.
6. **Update docs.** Tick `docs/ROADMAP.md`, log choices in `docs/DECISIONS.md`, update `docs/ARCHITECTURE.md` if structure changed.
7. **Summarize.** What changed, what was verified, what remains, open questions.

Follow the phases in `docs/ROADMAP.md` in order:
Phase 0 Setup, 1 Data contracts and storage, 2 Write path, 3 Read path, 4 Consolidation, 5 Evaluation harness, 6 Demo and integrations (FastAPI, MCP, tracing), 7 Docs and portfolio. Do not start stretch items (knowledge graph, inspector UI, WanderAI integration) before v1 and its evaluation are finished.

## 7. Git and Commit Rules

### 7.1 Separate commits by area (mandatory)

**Never mix areas in a single commit.** Each commit touches exactly one scope:

| Scope | Covers |
|---|---|
| `core` | `src/memlayer/` top-level: `api.py`, `models.py` |
| `extract` | `src/memlayer/extract/` |
| `store` | `src/memlayer/store/`, schemas, migrations |
| `retrieve` | `src/memlayer/retrieve/` |
| `consolidate` | `src/memlayer/consolidate/` |
| `llm` | `src/memlayer/llm/` |
| `mcp` | `mcp_server.py` |
| `tracing` | `tracing.py`, Langfuse/OTel setup |
| `eval` | `eval/`: loaders, baselines, runner, configs, results |
| `demo` | `demo/` app and UI |
| `tests` | `tests/` |
| `docs` | `docs/`, `README.md`, `AGENTS.md`, diagrams |
| `config` | linters, formatters, pre-commit, `.env.example` |
| `build` | `pyproject.toml`, dependencies, Dockerfile, Compose, CI |

If a task spans scopes, split into several commits in this order when possible:

1. `docs` (spec or design update)
2. implementation scopes (`store`, `extract`, `retrieve`, `consolidate`, ...)
3. `tests` / `eval`
4. `docs` (roadmap, decisions, README)

A refactor and a feature are separate commits, even in the same scope. Benchmark result files (`eval/results/`) are committed separately from the code that produced them.

### 7.2 Message format (Conventional Commits)

```
<type>(<scope>): <imperative summary, max 72 chars>

<optional body: what and why>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`.

Examples:
```
feat(extract): add Pydantic schema and retry for fact extraction
feat(retrieve): add recency-weighted scoring function
fix(consolidate): keep supersession link when merging duplicates
test(tests): add contradiction cases for city-change updates
feat(eval): add LongMemEval subset loader
docs(docs): add ADR for pgvector vs Qdrant
build(build): add GitHub Actions workflow for lint and tests
```

No vague messages ("update", "wip", "fix stuff"). One logical change per commit, and each commit should leave the project in a working state.

### 7.3 Branching and safety
- Never commit to `main`. Use `feature/<name>`, `fix/<name>`, `docs/<name>`, `eval/<name>`.
- Commit after each working slice, not at the end of a session.
- Stage explicit paths. No blind `git add .`. Review `git diff --staged` first.
- Never force push, rewrite shared history, or delete branches without permission.
- Never commit secrets, `.env`, model weights, raw benchmark datasets, vector DB files, or LLM output caches. Check `.gitignore`.
- Do not push unless asked.

## 8. Testing Requirements

- **Scoring:** unit tests for relevance, recency decay, and importance weighting, including edge cases (age 0, very old, missing importance).
- **Conflict handling:** a hand-written test set of update and contradiction cases ("lives in Pune" then "moved to Bangalore"; negations; partial updates; unrelated facts that look similar). Write these **before** the conflict logic.
- **Extraction:** fixed conversation fixtures; measure precision by hand on a small labeled set and record it.
- **Isolation:** tests proving one user's search can never return another user's memories.
- **Storage:** CRUD tests for episodes and memories on SQLite; Postgres/vector adapter tests where configured.
- Mock LLM calls in unit tests. Keep integration tests clearly separated and marked.
- A task is not done until tests pass and you have actually run them.

## 9. Evaluation Integrity Rules

These protect the credibility of the whole project.

- **Never fabricate or estimate results.** Every number in docs, README, or resume bullets must come from a saved run in `eval/results/`.
- Use the **same answer model across all systems** (full-context, naive RAG, memory layer) so the comparison measures memory, not the LLM. Record the answer model, judge model, and judge prompt.
- Tune hyperparameters on a dev split only. Never tune on the test questions.
- Runs must be reproducible: config file, fixed seeds, saved raw outputs, one command to rerun.
- Cache LLM outputs to control cost. Estimate cost before launching large runs and ask first.
- Start with 50 to 100 question subsets; scale only after the pipeline is validated.
- Report **per category** (knowledge update, temporal reasoning, multi-session, abstention), not only one overall number.
- Required outputs: results table, efficiency table (accuracy vs tokens per query vs p50/p95 latency vs cost), at least 4 ablations, and a failure analysis of 5 to 10 real cases.
- Planned ablations: without decay, without supersession, vector-only retrieval, without recency in score, small local vs larger hosted extraction model, different embedding models.
- Report weak or negative results honestly. A realistic target is "close to full-context accuracy at a fraction of the tokens". Do not promise "beats full-context".

## 10. Documentation Rules

- Docs are the project's memory. Keep them short, accurate, and current. Delete outdated content.
- `DECISIONS.md` entry format:
  ```
  ## NNN. <Title> (YYYY-MM-DD)
  - Context:
  - Decision:
  - Alternatives rejected:
  - Consequences:
  ```
- Do not make architecture or major tech choices silently. Propose 2 to 3 options with tradeoffs, let the user decide, then log it.
- `README.md` must stay recruiter-ready: headline ("Long-term memory for conversational agents, benchmarked on LongMemEval/LoCoMo"), architecture diagram, results table, ablation chart, demo (memory on vs off, including an update such as a changed city), setup in five commands or fewer, honest limitations.
- Privacy section is required in the README: local-first default, delete/export, PII redaction option, no sensitive data stored without an explicit flag.

## 11. Code Standards

- Clear, typed, readable code. Simple over clever. The owner must be able to explain every line.
- Docstrings on public functions. Explicit error handling, no bare `except`. Log context, never secrets or full user messages by default.
- Small, single-purpose functions. No dead code or commented-out blocks.
- Do not modify unrelated files. Keep diffs minimal.
- Public API changes are breaking changes: ask first and log in `DECISIONS.md`.

## 12. Open Decisions (ask the user; do not choose silently)

1. pgvector or Qdrant as the primary vector store?
2. Extraction LLM (local Ollama vs hosted), and which models for answering and judging?
3. Benchmark subset sizes that fit the budget?
4. Library name (check PyPI and GitHub availability) and license (MIT or Apache-2.0)?
5. Knowledge graph layer in v1 or v2?

Until a decision is logged in `docs/DECISIONS.md`, code behind an adapter interface so the choice stays swappable.

## 13. Boundaries

**Always:** read the docs before coding; ask when requirements are ambiguous; report failures and uncertainty honestly; keep user data scoped by `user_id`.

**Ask first:** adding dependencies or top-level folders; changing the data model, public API, or scoring formula; deleting files or large rewrites; running paid or large LLM jobs; starting stretch features.

**Never:** commit secrets or push without permission; claim code works without running it; fabricate benchmark numbers, citations, or competitor claims; hard-delete memories on contradiction; leave the repo in a broken state.

## 14. End-of-Session Checklist

- [ ] Tests, ruff, and mypy pass (actual output checked)
- [ ] `docs/ROADMAP.md` updated
- [ ] `docs/DECISIONS.md` updated if a choice was made
- [ ] `docs/ARCHITECTURE.md` updated if structure changed
- [ ] Any new eval numbers come from a saved run in `eval/results/`
- [ ] Commits split by scope with proper messages
- [ ] No secrets, datasets, or caches staged
- [ ] Summary given: done, verified, remaining, open questions

## 15. v1 Definition of Done

- [ ] Installable package with typed API and docs
- [ ] Write path, read path, and consolidation working with tests
- [ ] Benchmark results reproducible with one command
- [ ] Baselines and at least 4 ablations reported
- [ ] Demo app and MCP server running via Docker Compose
- [ ] Tracing visible in Langfuse
- [ ] README with results, architecture, and honest limitations
- [ ] Portfolio page and short demo video

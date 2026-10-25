# Spec 015: Recruiter-Ready README & MkDocs Documentation

## 1. Goal
Elevate the project's documentation to portfolio and recruiter-grade standard for LLM Systems and Conversational AI roles. Ensure that:
1. `README.md` acts as an executive summary and technical deep dive featuring benchmark numbers, architecture diagrams, demo walkthroughs, and honest limitations.
2. An **MkDocs Material** documentation site is configured and buildable with comprehensive guides across architecture, benchmarks, MCP, and observability.
3. Every metric and benchmark claim cited directly corresponds to empirical records in `eval/results/` and `docs/EVALUATION_REPORT.md`.

---

## 2. README Architecture & Content Structure

1. **Header & Badges:** CI, Python 3.11+, Ruff, Mypy, License (MIT).
2. **One-Line Value Proposition:** Long-term memory for conversational agents with measured benchmarks on LongMemEval and LoCoMo.
3. **Architecture Diagram:** Comprehensive Mermaid flowchart illustrating episodic logging, asynchronous atomic fact extraction, hybrid dense/sparse retrieval with calibrated scoring ($\alpha \cdot sim + \beta \cdot recency + \gamma \cdot importance$), non-destructive supersession, and storage layers.
4. **Benchmark Results:**
   - LongMemEval and LoCoMo comparative tables (Full-Context vs Naive RAG vs MemLayer).
   - Token efficiency & cost savings table (29%–36% prompt token reduction, 34%–40% total token reduction).
   - Ablation matrix isolating the 25% accuracy collapse when conflict supersession is disabled.
5. **Interactive Demo Walkthrough:**
   - Code snippet showing Memory ON vs Memory OFF.
   - Live state transition demonstration (e.g. moving from Pune to Bangalore, updating language preference from Go to Rust).
6. **Quickstart:** Setup in 5 commands or fewer (`git clone`, `uv venv`, `uv pip install`, `cp .env`, `pytest`).
7. **MCP & Observability Integrations:** Quick instructions for connecting the MCP server to Claude Desktop / Cursor and streaming traces to Langfuse.
8. **Privacy & Data Isolation:** Local-first defaults, hard forget API, tenant isolation guarantees.
9. **Engineering Tradeoffs & Honest Limitations:** Explicit discussion of indexing latency, LLM extraction costs, and boundaries.

---

## 3. MkDocs Material Configuration (`mkdocs.yml`)

- Site Name: `Agent Memory Layer (memlayer)`
- Theme: `material` with slate/default palette toggles, code copy buttons, search, and navigation tabs.
- Navigation Structure:
  - Overview: `index.md`
  - Architecture & Design: `architecture.md`
  - Benchmarks & Evaluation: `benchmarks.md`
  - Model Context Protocol (MCP): `mcp.md`
  - Observability & Tracing: `observability.md`
  - API Reference: `api.md`

---

## 4. Acceptance Criteria
1. `README.md` is complete, recruiter-ready, contains zero placeholder text ("*TBD*"), and includes actual evaluation numbers from `docs/EVALUATION_REPORT.md`.
2. `mkdocs.yml` and documentation markdown pages build cleanly with `mkdocs build --strict`.
3. All code blocks in docs are syntactically valid and match current API signatures (`MemoryLayer.create_local`, `mem.add`, `mem.search`, `mem.forget`).

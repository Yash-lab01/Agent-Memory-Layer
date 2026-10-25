# Spec 016: Portfolio Write-Up & Interview Guide

## 1. Goal
Create a recruiter-ready case study and interview preparation document (`docs/PORTFOLIO.md`) tailored for **LLM Systems, Conversational AI, and Applied AI Engineering** interview loops.

---

## 2. Core Sections in `docs/PORTFOLIO.md`

1. **The 30-Second Elevator Pitch:** Concise summary of what the project is, why it was built, and the primary measured result (100% state-update accuracy vs 0% in naive RAG, 34-41% token savings).
2. **The Engineering Problem:** Why standard vector RAG fails for conversational memory (context collisions, temporal decay ignorance, lack of entity linking).
3. **Core Architectural Invariants:**
   - Asynchronous write path (zero blocking latency on chat turns).
   - Non-destructive conflict supersession (links `superseded_by`, never hard deletes).
   - Strict per-user isolation from day one.
4. **Interview Talking Points & Defense of Design Choices:**
   - *Why hybrid search over dense-only?*
   - *Why SQLite/Postgres + Qdrant over an all-in-one graph DB in v1?*
   - *What were the ablation study findings and their significance?*
   - *How were benchmarks conducted to prevent LLM variance from contaminating memory evaluations?*
5. **Tradeoffs & Honest Limitations:** Realistic self-assessment demonstrating engineering maturity.
6. **Demo Script:** Step-by-step dialogue walkthrough showcasing Memory OFF vs Memory ON with live supersession.

---

## 3. Acceptance Criteria
1. `docs/PORTFOLIO.md` authored with depth, clarity, and zero generic buzzwords.
2. Included in `mkdocs.yml` navigation under Project Governance or Portfolio.
3. `mkdocs build --strict` passes with 0 warnings.
4. Phase 7.3 checked off in `docs/ROADMAP.md`.

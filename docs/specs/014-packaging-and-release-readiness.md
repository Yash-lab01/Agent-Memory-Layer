# Spec 014: Packaging & PyPI Release Readiness

## 1. Goal
Prepare `agent-memory-layer` (`memlayer`) for production distribution and PyPI release readiness. Guarantee that:
1. The base library installs cleanly with zero optional/heavy dependencies (no FastAPI, Jinja2, pandas, etc.).
2. The package builds standard compliant source distributions (`sdist`) and binary wheels (`wheel`) containing all modules and type markers (`py.typed`).
3. CLI entry points (e.g. `memlayer-mcp` for MCP server) are registered and operational.
4. Project metadata, URLs, classifiers, and documentation references conform to modern PyPA standards.

---

## 2. Architecture & Configuration

### 2.1 Dependency Isolation (`pyproject.toml`)
- **Core Dependencies (Minimal footprint):**
  - `pydantic>=2.7.0`
  - `sqlalchemy[asyncio]>=2.0.0`
  - `greenlet>=3.0.0`
  - `aiosqlite>=0.20.0`
  - `litellm>=1.40.0`
  - `rank-bm25>=0.2.2`
  - `numpy>=1.26.0`
  - `qdrant-client>=1.9.0`
  - `apscheduler>=3.10.4`
  - `alembic>=1.13.0`
- **Optional Extras:**
  - `[project.optional-dependencies]`
    - `postgres`: `asyncpg`, `pgvector`
    - `demo`: `fastapi`, `uvicorn`, `jinja2`
    - `mcp`: `mcp`
    - `tracing`: `langfuse`
    - `eval`: `pandas`, `tabulate`
    - `all`: aggregate convenience bundle

### 2.2 CLI Entry Points
- `memlayer-mcp`: Launches the FastMCP server over stdio for Claude Desktop / Cursor / IDE integrations.

### 2.3 Build Configuration & Type Marker
- `hatchling` backend configured with explicit package inclusion:
  ```toml
  [tool.hatch.build.targets.wheel]
  packages = ["src/memlayer"]
  ```
- Package includes `src/memlayer/py.typed` (PEP 561 compliance).

---

## 3. Acceptance Criteria
1. **Zero-Dependency Core Operation:** A client installing `pip install agent-memory-layer` without extras can run `MemoryLayer.create_local()`, add memories, search, and delete without missing dependency errors.
2. **Build Success:** Running `uv build` or `python -m build` produces valid `.tar.gz` and `.whl` artifacts in `dist/`.
3. **Artifact Integrity:** The wheel contains all 6 submodules (`extract`, `store`, `retrieve`, `consolidate`, `llm`, `tracing`, `mcp_server`) and `py.typed`.
4. **Smoke Tests:** `tests/test_package.py` tests version resolution, public API exports, CLI entry point definition, and py.typed existence.
5. **Quality Standards:** All tests pass, ruff and mypy pass with 0 errors.

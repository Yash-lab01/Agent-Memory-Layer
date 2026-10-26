# Architecture Decision Records (ADRs): `memlayer`

This log documents key architectural choices, tradeoffs, rejected alternatives, and consequences.

---

## 001. Pluggable Storage & Vector Adapter Architecture (2026-10-12)
- **Context:** `memlayer` requires both structured metadata storage (with supersession pointers) and fast semantic vector search. We need to decide between pgvector and Qdrant, while supporting zero-setup local execution for unit testing and CI.
- **Decision:** Build an abstract `VectorStoreBase` interface. Implement **Qdrant** (using its in-memory / local disk client for tests and containerized Qdrant for deployments) as the default vector store, with a modular adapter pattern enabling **pgvector** as an alternate backend. Relational storage defaults to **SQLite** for local development and **PostgreSQL (SQLAlchemy 2.0)** for production server environments.
- **Alternatives rejected:**
  - *Hardcoding pgvector only:* Requires running Postgres in every unit test, complicating lightweight CI and local testing without Docker.
  - *Chroma / FAISS:* Lacks production-grade server clustering, robust payload filtering, and native MCP ecosystem support compared to Qdrant/pgvector.
- **Consequences:** All vector interactions must remain strictly behind the adapter interface (`src/memlayer/store/vector_base.py`) so users can swap vector backends via configuration.

---

## 002. Unified LLM Interface via LiteLLM (2026-10-12)
- **Context:** Memory extraction, contradiction checking, and benchmark evaluation need to run across diverse model providers (Ollama for free local development, Gemini / Groq / OpenAI for production speed and quality).
- **Decision:** Standardize all model invocations through `LiteLLM`. Provide configuration files (`eval/configs/`) to map model endpoints without hardcoded vendor dependencies in core code.
- **Alternatives rejected:**
  - *Direct SDKs (openai, google-genai, ollama):* Bloats dependencies and introduces divergent error-handling and parameter names across the codebase.
  - *LangChain core:* Heavy dependency surface with frequent breaking changes; contradicts the project's goal of a lean, readable library.
- **Consequences:** All extraction and judge calls use unified LiteLLM parameters and response structures.

---

## 003. Non-Destructive Contradiction Handling via Supersession (2026-10-12)
- **Context:** When conversational agents encounter updated knowledge (e.g., "I moved from Pune to Bangalore"), naive memory systems either overwrite the old record, create duplicate conflicting facts, or delete the old record entirely.
- **Decision:** Implement **supersession**. When a new fact directly contradicts an existing fact, the existing memory's status is updated to `superseded` and its `superseded_by` foreign key is set to the new memory's ID. Historical facts remain in the database for auditability and temporal queries, but are excluded by default from active retrieval.
- **Alternatives rejected:**
  - *Hard delete / in-place overwrite:* Erases conversation history, breaks temporal reasoning ("where did I live last year?"), and destroys provenance.
  - *Append without conflict resolution:* Leads to conflicting facts being injected into context, causing model hallucination.
- **Consequences:** All read queries must filter by `status == 'active'` unless a temporal history lookup is explicitly requested.

---

## 004. Strict User Namespace Isolation (2026-10-12)
- **Context:** Multi-tenant conversational agents must prevent cross-user data leakage.
- **Decision:** Enforce mandatory `user_id` parameters on every public API method, SQL table query, and vector store payload filter. Unit tests must explicitly verify that queries for `user_A` never return records belonging to `user_B`.
- **Alternatives rejected:**
  - *Optional / global namespaces:* High risk of accidental privacy leakage in shared agent environments.
- **Consequences:** Zero API calls can execute without a validated `user_id`.

---

## 005. Knowledge Graph Layer Architecture: Pluggable Adapter with Zero-Dependency Default (2026-10-26)
- **Context:** To support multi-hop compositional reasoning across entities (e.g. `(Alice)-[WORKS_ON]->(Project)-[MANAGED_BY]->(Bob)`), `memlayer` requires graph storage. We need to support local lightweight execution without requiring external graph database daemons (like Neo4j) or C++ binaries.
- **Decision:** Define an abstract `GraphStoreBase` interface in `src/memlayer/store/graph_base.py`. Provide an in-memory, pure-Python adjacency store (`InMemoryGraphStore`) with BFS/DFS multi-hop pathfinding as the zero-overhead default. Support `NetworkX` (`NetworkXGraphStore`) via an optional extra (`agent-memory-layer[graph]`).
- **Alternatives rejected:**
  - *Hardcoding Neo4j / Memgraph:* Requires heavyweight Docker containers or cloud services for basic unit tests and local execution.
  - *Hardcoding Kùzu:* While embedded and fast, C++ binary compilation can fail on non-standard architectures, violating minimal-dependency requirements.
- **Consequences:** Core `memlayer` remains lightweight and fast. Advanced graph analytics (centrality, clustering) can be enabled via optional extras. All graph reads and writes remain strictly scoped by `user_id`.

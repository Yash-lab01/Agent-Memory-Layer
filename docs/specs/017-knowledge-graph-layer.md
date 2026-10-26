# Spec 017: Knowledge Graph Layer (Multi-Hop Relational Memory)

## 1. Goal
Extend **Agent Memory Layer (`memlayer`)** with a modular **Knowledge Graph Layer** to enable multi-hop relational reasoning across multi-session dialogues.
Standard vector search can retrieve facts about single entities, but degrades on relational queries requiring compositional path traversal (e.g., *"Who manages the project that Alice is working on?"*).

---

## 2. Core Concepts & Data Model

### 2.1 Graph Triplet & Schemas (`src/memlayer/models.py`)
- **`EntityNode`**:
  - `id`: Unique node ID (typically normalized `user_id:entity_name`).
  - `user_id`: Tenant identifier for strict data isolation.
  - `name`: Entity name (e.g. `"Alice"`, `"Project Apollo"`, `"Bangalore"`).
  - `entity_type`: Category (e.g. `"person"`, `"project"`, `"location"`, `"organization"`, `"concept"`).
  - `properties`: Flexible metadata dictionary.
- **`RelationEdge`**:
  - `id`: Unique edge ID.
  - `user_id`: Tenant identifier.
  - `source`: Source entity name.
  - `target`: Target entity name.
  - `relation`: Directed relationship predicate (e.g. `"WORKS_ON"`, `"MANAGED_BY"`, `"LIVES_IN"`).
  - `weight`: Edge confidence/strength ($0.0 \dots 1.0$).
  - `source_memory_id`: Optional link to underlying atomic `MemoryItem` for provenance.
  - `properties`: Metadata dictionary.
- **`GraphTriplet`**:
  - Lightweight tuple `(source, relation, target)` extracted from propositions.

---

## 3. Storage Architecture (`src/memlayer/store/graph_base.py`)

### 3.1 Abstract Interface: `GraphStoreBase`
All graph storage backends must implement:
- `add_node(user_id: str, node: EntityNode) -> EntityNode`
- `add_edge(user_id: str, edge: RelationEdge) -> RelationEdge`
- `get_node(user_id: str, name: str) -> EntityNode | None`
- `get_neighbors(user_id: str, entity: str, direction: str = "both") -> list[RelationEdge]`
- `find_paths(user_id: str, source: str, target: str, max_depth: int = 3) -> list[list[RelationEdge]]`
- `get_subgraph(user_id: str, entities: list[str], max_depth: int = 2) -> tuple[list[EntityNode], list[RelationEdge]]`
- `delete_user_graph(user_id: str) -> int` (GDPR compliance)
- `close() -> None`

### 3.2 Implementations
1. **`InMemoryGraphStore` (`src/memlayer/store/graph_memory.py`)**:
   - Zero-dependency, pure-Python adjacency list implementation.
   - Built-in multi-hop BFS/DFS pathfinding and subgraph extraction.
   - Ideal for local development, fast unit tests, and minimal CI runs.
2. **`NetworkXGraphStore` (`src/memlayer/store/graph_networkx.py`)**:
   - Optional adapter wrapping `networkx.MultiDiGraph` with JSON graph export and advanced graph algorithms (centrality, community detection).
   - Enabled via optional extra: `pip install agent-memory-layer[graph]`.

---

## 4. Retrieval & Integration with `MemoryLayer`

1. **Extraction Integration (`src/memlayer/extract/`):**
   - Fact extraction can optionally yield relational triplets `(subject, relation, object)`.
2. **Read Path Augmentation (`MemoryLayer.search`):**
   - Query entities are extracted and matched against the knowledge graph.
   - Multi-hop subgraphs are formatted into a markdown section:
     ```markdown
     [Knowledge Graph Relationships]
     - Alice --(WORKS_ON)--> Project Apollo
     - Project Apollo --(MANAGED_BY)--> Bob
     ```
3. **Multi-Tenant Isolation Invariant:**
   - Every graph operation requires validated `user_id`.
   - Traversals can never traverse or inspect nodes/edges belonging to another tenant.

---

## 5. Acceptance Criteria
1. **Zero-Dependency Default:** Core library runs `InMemoryGraphStore` without requiring `networkx` or C-extensions.
2. **Multi-Hop Traversal:** `find_paths` successfully discovers 2-hop and 3-hop relationships between entities.
3. **Strict User Isolation:** Explicit tests demonstrate User A's graph is invisible and unreachable to User B.
4. **Clean Integration:** `MemoryLayer` accepts optional `graph_store` without breaking existing public API signatures.
5. **Quality Standards:** 100% test coverage for graph storage, ruff and mypy pass with zero errors.

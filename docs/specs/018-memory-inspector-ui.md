# Spec 018: Memory Inspector UI & Knowledge Graph Visualizer

## 1. Goal
Provide a comprehensive, interactive **Memory Inspector UI** for **Agent Memory Layer (`memlayer`)** allowing developers, researchers, and recruiters to inspect, debug, and visualize:
1. **Episodic Dialogue turns & provenance.**
2. **Active vs. Superseded memory items** (showing contradiction resolution and supersession pointers).
3. **Knowledge Graph relationships** with an interactive force-directed graph canvas (nodes, directed edges, multi-hop traversals).
4. **Real-time retrieval & router simulator** with token budgeting metrics and latency tracking.

---

## 2. Architecture & Design

### 2.1 Backend API Extensions (`demo/app/main.py`)
To power the visualizer, `demo/app/main.py` is equipped with an `InMemoryGraphStore` and provides the following REST endpoints:
- `GET /api/graph?user_id=...` -> `{ "nodes": [EntityNode], "edges": [RelationEdge] }`
- `POST /api/graph/triplets` -> `{ "source": str, "relation": str, "target": str, "weight": float }`
- `GET /api/graph/paths?user_id=...&source=...&target=...&max_depth=...` -> `{ "paths": list[list[RelationEdge]] }`
- `DELETE /api/graph?user_id=...` -> `{ "deleted": int }`

### 2.2 Frontend Visualizer Capabilities
1. **Chat & Search Playground:**
   - Memory Toggle (ON / OFF) demonstrating recall vs. amnesia.
   - Real-time token budget meter and retrieved context preview.
2. **Knowledge Graph Visualizer (Canvas / SVG):**
   - Interactive force-directed or circular network rendering of entities (nodes) and relations (directed labeled edges).
   - Node selection highlighting incoming/outgoing edges.
   - Multi-hop path finding finder (e.g. *Alice* -> *Bob*).
   - "Add Triplet" modal allowing manual knowledge expansion.
3. **Memory Store & Supersession Inspector:**
   - Tabular and timeline view of all stored memories.
   - Color-coded badges for `active` (green) and `superseded` (purple/amber).
   - Supersession pointers (`superseded_by -> memory_id`) showing historical provenance.
   - One-click forget/delete action for testing GDPR deletion.

---

## 3. Visual & Aesthetic Standards
- **Theme:** Premium dark-mode glassmorphism (`#0b0f19` deep slate background, translucent cards with subtle 1px border glows).
- **Typography:** Modern clean sans-serif (Inter / Outfit style).
- **Performance:** Instant local rendering with zero external CDN dependency breakages; runs out-of-the-box via `uvicorn demo.app.main:app` or Docker Compose.

---

## 4. Acceptance Criteria
1. `demo/app/main.py` exposes `/api/graph` endpoints with strict `user_id` validation.
2. The UI includes a dedicated **Knowledge Graph** tab with interactive entity-relationship rendering.
3. Users can add triplets via the UI and immediately see them rendered on the graph and retrieved in `/api/chat`.
4. Contradictions continue to be visually indicated with supersession badges.
5. All tests in `tests/test_demo.py` pass and new tests cover the graph endpoints.

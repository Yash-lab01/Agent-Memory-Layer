# Spec 011: FastAPI Service & Interactive Web Demo

## 1. Goal
Provide a production-ready, interactive web demonstration and REST API for the **Agent Memory Layer (`memlayer`)**. The demo showcases durable long-term memory across sessions, allows toggling memory ON/OFF to directly visualize memory impact, and features a live **Memory Inspector** that displays active facts, non-destructive supersessions, and real-time token metrics.

---

## 2. Architecture & Public Interface

### 2.1 Backend Architecture (`demo/app/main.py`)
Built with **FastAPI** and mounted with static assets (Vanilla CSS + Vanilla JS, zero build step required):
- Initializes a default `MemoryLayer` instance (local SQLite + local vector store / mock embedder for immediate out-of-the-box execution without mandatory remote keys).
- Asynchronous background extraction on chat turns (`BackgroundTasks`).

### 2.2 REST Endpoints

1. **`POST /api/chat`**
   - **Request:**
     ```json
     {
       "user_id": "demo_user",
       "session_id": "session_1",
       "message": "I just moved to Austin, Texas.",
       "memory_enabled": true
     }
     ```
   - **Response:**
     ```json
     {
       "reply": "Congratulations on the move to Austin!",
       "memory_enabled": true,
       "retrieved_memories": [
         {
           "id": "mem_1",
           "content": "User relocated to Austin, Texas.",
           "entity": "Location",
           "score": 0.88,
           "status": "active"
         }
       ],
       "prompt_tokens": 78,
       "latency_ms": 14.2
     }
     ```

2. **`GET /api/memories`**
   - Query parameter: `user_id` (string, default: `demo_user`)
   - Returns all stored memory items for the user, partitioned or marked by status (`active` vs. `superseded`), including `superseded_by`, `entity`, `importance`, and `created_at`.

3. **`DELETE /api/memories/{memory_id}`**
   - Query parameter: `user_id`
   - Invokes `memlayer.forget(user_id, memory_id)` to execute durable deletion.

4. **`POST /api/reset`**
   - Query parameter: `user_id`
   - Purges memories and episodes for the designated test user to restore clean demo state.

5. **`GET /`**
   - Serves the single-page application (`demo/app/static/index.html`).

---

## 3. Frontend & Interactive UI (`demo/app/static/`)

### 3.1 Design Aesthetics
- **Theme:** Modern dark mode with glassmorphism, subtle gradients, and typography using Google Fonts (Inter / Outfit).
- **Layout:**
  - **Header:** Project identity ("Agent Memory Layer Demo"), connection status badge, and global user switcher (`user_id`).
  - **Left Panel (Chat Interface):**
    - Memory Toggle Switch (`#memory-toggle`): Switch between "Memory ON" (with context injection & active extraction) and "Memory OFF" (standard stateless LLM behavior).
    - Session selector: Switch between sessions (`session_1`, `session_2`, etc.) to simulate cross-session memory recall.
    - Chat message stream with message bubbles and metadata pills showing prompt token cost.
    - Chat input box (`#chat-input`) and send button (`#btn-send`).
    - Quick-prompt scenario buttons (e.g. *"I live in Seattle"*, *"I relocated to Austin"*, *"Where do I live?"*).
  - **Right Panel (Live Memory Inspector):**
    - Active Memories list (`#active-memories-list`): Cards displaying extracted facts, entity tag, importance score, and access count.
    - Superseded Memories list (`#superseded-memories-list`): Collapsible history showing outdated facts, strikethrough styling, and a badge linking to the superseding memory (`superseded_by`).
    - Forget button on each memory card (`#btn-forget-{id}`) for testing explicit forgetting.
    - Reset Store button (`#btn-reset-store`).

---

## 4. Acceptance Criteria

1. **Standalone Execution:** The demo can be run directly with a single command:
   ```bash
   python -m demo.app.main
   # or: uvicorn demo.app.main:app --port 8000
   ```
2. **Memory Toggle Comparison:**
   - With Memory ON: Asking *"Where do I live?"* after stating *"I relocated to Austin"* answers Austin accurately using retrieved memory.
   - With Memory OFF: Asking *"Where do I live?"* in a new session returns no recollection, demonstrating stateless behavior.
3. **Live Supersession Inspection:** Stating a contradiction in a new session (e.g. Seattle $\to$ Austin) immediately reflects in the Memory Inspector, with the Seattle card transitioning to `superseded` and Austin appearing in `active`.
4. **User Isolation:** Memories created under `user_a` never appear in `user_b`'s inspector or chat context.
5. **Quality & Packaging:**
   - Core library dependencies in `pyproject.toml` remain minimal; FastAPI/Uvicorn/Jinja2 are kept in `[project.optional-dependencies] demo`.
   - All backend code passes `ruff`, `mypy`, and unit tests in `tests/test_demo.py`.

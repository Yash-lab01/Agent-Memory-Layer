# Spec 012: Model Context Protocol (MCP) Server

## 1. Goal
Provide a Model Context Protocol (MCP) server for the **Agent Memory Layer (`memlayer`)**. This enables external chat clients, IDEs, and agentic platforms (Claude Desktop, Cursor, Antigravity, LangChain, etc.) to connect to a local or remote `memlayer` instance over standard I/O (`stdio`) or SSE transports to persist, search, inspect, and delete long-term memories.

---

## 2. Architectural Design (`src/memlayer/mcp_server.py`)

The server wraps `MemoryLayer` and exposes standard MCP tools using the official Python MCP SDK (`mcp`). It provides compatibility across both `mcp 1.x` (`FastMCP`) and `mcp 2.x` (`MCPServer`).

### 2.1 Backend Initialization
- **Storage:** Configurable via `DATABASE_URL` (defaults to local SQLite `sqlite+aiosqlite:///memlayer_mcp.db`).
- **Embedding:** Configurable via `EMBEDDING_PROVIDER` (defaults to `MockEmbedder` for offline/local execution or FastEmbed/hosted embeddings when keys are provided).
- **Extraction:** Employs `FactExtractor` with offline fallback heuristics when external LLM API credentials are unconfigured.

### 2.2 Exposed MCP Tools

1. **`memlayer_add`**
   - **Description:** Ingest dialogue messages or statements into persistent long-term memory for a user. Extracts atomic facts, generates vector embeddings, and performs non-destructive conflict supersession.
   - **Parameters:**
     - `user_id` (string, required): Namespace identifier for strict privacy isolation.
     - `text` (string, required): Dialogue text or statement to extract facts from.
     - `role` (string, optional, default: `"user"`): Speaker role (`"user"` or `"assistant"`).
     - `session_id` (string, optional, default: `"default"`): Conversation session ID.
   - **Returns:** JSON object containing list of newly extracted and stored memory items with `id`, `content`, `entity`, `importance`, and `status`.

2. **`memlayer_search`**
   - **Description:** Retrieve relevant historical memories for a query within a defined token budget using hybrid retrieval (BM25 + dense vector) and recency decay scoring.
   - **Parameters:**
     - `user_id` (string, required): Target user namespace.
     - `query` (string, required): The query string or user message.
     - `limit` (integer, optional, default: `5`): Maximum number of memories to return.
     - `token_budget` (integer, optional, default: `500`): Maximum tokens for the injected context.
   - **Returns:** JSON object containing:
     - `context`: Formatted prompt injection string.
     - `memories`: List of retrieved memory items with scores.
     - `token_count`: Estimated token count.
     - `skipped`: Boolean indicating whether retrieval was skipped by the query router.

3. **`memlayer_forget`**
   - **Description:** Permanently delete an individual memory by its ID across relational and vector stores.
   - **Parameters:**
     - `user_id` (string, required): Target user namespace.
     - `memory_id` (string, required): UUID of the memory item to delete.
   - **Returns:** JSON object with `deleted: true` and `memory_id`.

4. **`memlayer_list`**
   - **Description:** List stored memories for a user, optionally filtering by status (`active` or `superseded`).
   - **Parameters:**
     - `user_id` (string, required): Target user namespace.
     - `status` (string, optional, default: `"all"`): Filter by `"active"`, `"superseded"`, or `"all"`.
     - `limit` (integer, optional, default: `50`): Maximum number of memories to return.
   - **Returns:** JSON array of memory records including metadata, access count, and supersession links.

5. **`memlayer_reset`**
   - **Description:** Purge all stored memories and episodes for a user (GDPR Right to Be Forgotten).
   - **Parameters:**
     - `user_id` (string, required): Target user namespace.
   - **Returns:** JSON object confirming purge and number of deleted records.

---

## 3. Client Configuration (e.g. Claude Desktop / Antigravity)

The server can be registered in an MCP host configuration (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "agent-memory-layer": {
      "command": "python",
      "args": ["-m", "memlayer.mcp_server"],
      "env": {
        "DATABASE_URL": "sqlite+aiosqlite:///memlayer.db"
      }
    }
  }
}
```

---

## 4. Acceptance Criteria

1. **Protocol Compliance:** Uses official Python MCP SDK (`MCPServer` in 2.x, `FastMCP` in 1.x) with stdio transport.
2. **Tool Coverage:** Implements `memlayer_add`, `memlayer_search`, `memlayer_forget`, `memlayer_list`, and `memlayer_reset`.
3. **Strict Isolation:** Every tool invocation requires `user_id`; cross-user memory leakage is strictly prevented.
4. **Offline Resilience:** Functions out-of-the-box without mandatory remote API credentials.
5. **Testing & CI:**
   - Unit tests in `tests/test_mcp_server.py` test all tools, isolation, and error conditions.
   - CI workflow installs `.[dev,postgres,demo,mcp]` and passes with 100% green status.

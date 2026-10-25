# Model Context Protocol (MCP) Server

`memlayer` includes a native implementation of the **Model Context Protocol (MCP)**, allowing any compliant LLM client (Claude Desktop, Cursor, Zed, custom agent runtimes) to seamlessly read, write, inspect, and prune memory using standard tool calls over `stdio`.

---

## 1. Running the MCP Server

### Via CLI
```bash
memlayer-mcp
```

### Via Python Module
```bash
python -m memlayer.mcp_server
```

---

## 2. Connecting to Claude Desktop

Add `memlayer` to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memlayer": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/Agent-Memory-Layer",
        "memlayer-mcp"
      ]
    }
  }
}
```

---

## 3. Exposed MCP Tools

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `memlayer_add` | `user_id`, `messages`, `session_id` | Ingests dialog turns, persists episodes, extracts facts, and updates supersession links. |
| `memlayer_search` | `user_id`, `query`, `token_budget`, `limit` | Performs hybrid search and multi-factor ranking, returning formatted prompt context. |
| `memlayer_list` | `user_id`, `status` (`active`/`superseded`), `limit` | Lists stored memories with access frequency and supersession references. |
| `memlayer_forget` | `user_id`, `memory_id` | Explicitly deletes a single memory across SQL and vector storage. |
| `memlayer_reset` | `user_id` | Full GDPR purge of all episodes and memories for a tenant. |

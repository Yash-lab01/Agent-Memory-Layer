# Agent Memory Layer (`memlayer`)

> **Durable, long-term memory for conversational AI agents — benchmarked on LongMemEval & LoCoMo.**

`memlayer` is a modular, transparent Python library and service designed to give chat agents durable memory across multi-session dialogues. It extracts atomic facts from conversational interactions, maintains temporal provenance, resolves contradictions non-destructively through supersession, and retrieves relevant context using hybrid scoring.

---

## Core Value Proposition

- **Empirically Measured:** Rather than claiming algorithmic novelty over existing systems (Mem0, Letta, Zep), `memlayer` emphasizes **architectural rigor, reproducible numbers, and systematic ablations**.
- **Non-Destructive Supersession:** When user state evolves (e.g. moving from Pune to Bangalore), `memlayer` marks the earlier fact as `superseded` rather than hard-deleting it, eliminating contradictory context collisions while preserving historical audit trails.
- **Token Efficiency:** Delivers **34%–41% total token reductions** over Full-Context dialogue history while maintaining 100% accuracy on knowledge updates and state transitions.
- **Production Ready:** Includes native **Model Context Protocol (MCP)** server for Claude Desktop / Cursor, zero-overhead **Langfuse distributed tracing**, and one-command **Docker Compose** deployment.
- **Strict Per-User Isolation:** Multi-tenant security guaranteed at the storage engine level from day one.

---

## Quickstart (5 Commands)

```bash
# 1. Clone repository
git clone https://github.com/Yash-lab01/Agent-Memory-Layer.git
cd Agent-Memory-Layer

# 2. Create virtual environment
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install with development & demo dependencies
uv pip install -e ".[dev,demo,mcp,tracing]"

# 4. Copy environment configuration
cp .env.example .env

# 5. Run test suite
pytest
```

---

## Basic Python Example

```python
import asyncio
from memlayer import MemoryLayer


async def main():
    # 1. Initialize MemoryLayer (local SQLite + in-memory Qdrant default)
    mem = await MemoryLayer.create_local(vector_dim=64)

    # 2. Ingest dialogue turn asynchronously
    messages = [
        {"role": "user", "content": "I recently relocated from Pune to Bangalore."},
        {"role": "assistant", "content": "Welcome to Bangalore! How is the new city?"},
    ]
    await mem.add(user_id="alice", messages=messages)

    # 3. Retrieve relevant context within an injection token budget
    result = await mem.search(user_id="alice", query="Where do I live right now?", token_budget=150)

    print(result.context)
    # Output:
    # [Memories]
    # - User relocated to Bangalore (Event: 2026-10-26)

    # 4. Cleanup
    await mem.close()


if __name__ == "__main__":
    asyncio.run(main())
```

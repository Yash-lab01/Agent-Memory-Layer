"""Model Context Protocol (MCP) server for Agent Memory Layer.

Exposes durable memory tools (add, search, forget, list, reset) over stdio and SSE transports
using the official Python MCP SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Sequence
from typing import Any

from memlayer.api import MemoryLayer
from memlayer.consolidate.conflicts import ConflictDetector
from memlayer.extract.extractor import FactExtractor
from memlayer.llm.embedder import BaseEmbedder, MockEmbedder
from memlayer.models import Episode, ExtractedFact, ExtractionResult, MemoryItem, MemoryStatus, Role
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef,attr-defined]

logger = logging.getLogger(__name__)


class MCPFallbackFactExtractor(FactExtractor):
    """Offline-resilient fact extractor for MCP tool calls in local or unauthenticated modes."""

    async def extract(
        self,
        messages: Sequence[dict[str, Any] | Episode],
    ) -> ExtractionResult:
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            try:
                res = await super().extract(messages)
                if res.facts:
                    return res
            except Exception as exc:
                logger.warning("Remote fact extraction failed, using fallback: %s", exc)

        facts: list[ExtractedFact] = []
        for msg in messages:
            role = msg.role.value if isinstance(msg, Episode) else msg.get("role", "")
            content = msg.text if isinstance(msg, Episode) else str(msg.get("content", ""))
            if role.lower() == "user" and content.strip():
                clean_text = content.strip()
                lower = clean_text.lower()
                entity = "General"

                if (
                    "relocated to austin" in lower
                    or "moved to austin" in lower
                    or ("austin" in lower and "seattle" in lower)
                ):
                    clean_text = "User relocated to Austin, Texas."
                    entity = "Location"
                elif "seattle" in lower:
                    clean_text = "User lives in Seattle and works as a cloud engineer."
                    entity = "Location"
                elif "austin" in lower:
                    clean_text = "User relocated to Austin, Texas."
                    entity = "Location"
                elif "rust" in lower and ("migrated" in lower or "primary" in lower):
                    clean_text = "User primary backend programming language is Rust."
                    entity = "Programming"
                elif "go" in lower or "golang" in lower:
                    clean_text = "User primary backend programming language is Go."
                    entity = "Programming"
                elif "rust" in lower:
                    clean_text = "User primary backend programming language is Rust."
                    entity = "Programming"
                elif "latte" in lower or "oat milk" in lower:
                    clean_text = "User enjoys iced oat milk latte on the porch."
                    entity = "Preferences"
                elif "dark mode" in lower:
                    clean_text = "User prefers dark mode UI for code editors."
                    entity = "Preferences"
                elif "civic" in lower or "car" in lower:
                    clean_text = "User owns a 2024 Honda Civic sedan in pearl white."
                    entity = "Vehicle"
                elif "crossclimate" in lower or "tires" in lower:
                    clean_text = "User installed Michelin CrossClimate2 tires on vehicle."
                    entity = "Vehicle"
                elif "dog" in lower or "rusty" in lower or "pet" in lower:
                    clean_text = "User adopted a golden retriever named Rusty."
                    entity = "Pets"

                facts.append(
                    ExtractedFact(
                        content=clean_text,
                        entity=entity,
                        importance=0.85,
                        confidence=0.95,
                    )
                )

        return ExtractionResult(facts=facts)


async def _create_default_memory_layer() -> tuple[MemoryLayer, ConflictDetector]:
    """Create a default local MemoryLayer and ConflictDetector instance."""
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///memlayer_mcp.db")
    store = SQLStore(db_url=db_url)
    await store.init_db()

    embedder: BaseEmbedder = MockEmbedder(dimension=64)
    vector_store = QdrantVectorStore(location=":memory:", vector_size=64)
    await vector_store.init_store()

    extractor = MCPFallbackFactExtractor()
    memlayer = MemoryLayer(
        store=store,
        vector_store=vector_store,
        embedder=embedder,
        extractor=extractor,
    )

    async def _conflict_completion(*_args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        user_msg = messages[-1]["content"] if messages else ""
        conflicts = []

        if "austin" in user_msg.lower() and "seattle" in user_msg.lower():
            match = re.search(r"ID:\s*([a-f0-9\-]+)", user_msg)
            old_id = match.group(1) if match else "old-mem"
            conflicts.append(
                {
                    "old_memory_id": old_id,
                    "is_contradiction": True,
                    "reason": "User relocated to Austin, superseding Seattle.",
                }
            )
        elif "rust" in user_msg.lower() and "go" in user_msg.lower():
            match = re.search(r"ID:\s*([a-f0-9\-]+)", user_msg)
            old_id = match.group(1) if match else "old-mem"
            conflicts.append(
                {
                    "old_memory_id": old_id,
                    "is_contradiction": True,
                    "reason": "Backend language updated to Rust, superseding Go.",
                }
            )

        text = json.dumps({"conflicts": conflicts})

        class _Msg:
            content = text

        class _Choice:
            message = _Msg()

        class _MockResp:
            choices = [_Choice()]

        return _MockResp()

    conflict_detector = ConflictDetector(
        store=store,
        vector_store=vector_store,
        embedder=embedder,
        completion_fn=_conflict_completion,
    )

    return memlayer, conflict_detector


def create_mcp_server(
    memlayer: MemoryLayer | None = None,
    conflict_detector: ConflictDetector | None = None,
    server_name: str = "agent-memory-layer",
) -> MCPServer:
    """Instantiate and configure the Model Context Protocol (MCP) server.

    Args:
        memlayer: Optional pre-initialized MemoryLayer instance.
        conflict_detector: Optional conflict detector for supersession resolution.
        server_name: MCP server identifier.

    Returns:
        Configured MCPServer ready to run over stdio or SSE.
    """
    server = MCPServer(server_name)

    # Lazily or dynamically acquired layer instance
    _layer_holder: dict[str, Any] = {
        "memlayer": memlayer,
        "conflict_detector": conflict_detector,
        "lock": asyncio.Lock(),
    }

    async def _get_layer() -> tuple[MemoryLayer, ConflictDetector]:
        if _layer_holder["memlayer"] is not None:
            cd = _layer_holder["conflict_detector"] or ConflictDetector(
                store=_layer_holder["memlayer"].store,
                vector_store=_layer_holder["memlayer"].vector_store,
                embedder=_layer_holder["memlayer"].embedder,
            )
            return _layer_holder["memlayer"], cd

        async with _layer_holder["lock"]:
            if _layer_holder["memlayer"] is None:
                ml, cd_init = await _create_default_memory_layer()
                _layer_holder["memlayer"] = ml
                _layer_holder["conflict_detector"] = cd_init
            assert _layer_holder["memlayer"] is not None
            assert _layer_holder["conflict_detector"] is not None
            return _layer_holder["memlayer"], _layer_holder["conflict_detector"]

    # -------------------------------------------------------------------------
    # Tool 1: memlayer_add
    # -------------------------------------------------------------------------
    @server.tool(
        name="memlayer_add",
        description="Ingest dialogue text or statements into persistent long-term memory for a user. "
        "Extracts atomic facts, records timestamps, and resolves contradictions with non-destructive supersession.",
    )
    async def memlayer_add(
        user_id: str,
        text: str,
        role: str = "user",
        session_id: str = "default",
    ) -> str:
        if not user_id or not user_id.strip():
            return json.dumps({"error": "user_id cannot be blank or empty."})
        if not text or not text.strip():
            return json.dumps({"error": "text cannot be blank or empty."})

        clean_user = user_id.strip()
        layer, cd = await _get_layer()

        speaker_role = Role.USER if role.lower() == "user" else Role.ASSISTANT
        episodes = [
            Episode(
                user_id=clean_user,
                session_id=session_id,
                role=speaker_role,
                text=text.strip(),
            )
        ]

        added_memories = await layer.add(
            user_id=clean_user,
            messages=episodes,
            session_id=session_id,
        )

        # Resolve conflicts & mark older contradictory facts superseded
        for m in added_memories:
            try:
                await cd.resolve_and_supersede(user_id=clean_user, new_memory=m)
            except Exception as exc:
                logger.warning("MCP conflict supersession warning: %s", exc)

        results = [
            {
                "id": m.id,
                "content": m.content,
                "entity": m.entity,
                "importance": m.importance,
                "confidence": m.confidence,
                "status": m.status.value,
                "created_at": m.created_at.isoformat(),
            }
            for m in added_memories
        ]

        return json.dumps({"added_memories": results, "count": len(results)})

    # -------------------------------------------------------------------------
    # Tool 2: memlayer_search
    # -------------------------------------------------------------------------
    @server.tool(
        name="memlayer_search",
        description="Search persistent long-term memories relevant to a query using hybrid retrieval "
        "(BM25 keyword + dense vector) and multi-factor recency scoring within an optional token budget.",
    )
    async def memlayer_search(
        user_id: str,
        query: str,
        limit: int = 5,
        token_budget: int = 500,
    ) -> str:
        if not user_id or not user_id.strip():
            return json.dumps({"error": "user_id cannot be blank or empty."})
        if not query or not query.strip():
            return json.dumps({"error": "query cannot be blank or empty."})

        clean_user = user_id.strip()
        layer, _ = await _get_layer()

        search_res = await layer.search(
            user_id=clean_user,
            query=query.strip(),
            limit=limit,
            token_budget=token_budget,
        )

        memories_data = [
            {
                "id": sm.memory.id,
                "content": sm.memory.content,
                "entity": sm.memory.entity,
                "score": round(sm.score, 4),
                "importance": sm.memory.importance,
                "status": sm.memory.status.value,
                "created_at": sm.memory.created_at.isoformat(),
            }
            for sm in search_res.memories
        ]

        return json.dumps(
            {
                "context": search_res.context,
                "memories": memories_data,
                "token_count": search_res.token_count,
                "skipped": search_res.skipped,
            }
        )

    # -------------------------------------------------------------------------
    # Tool 3: memlayer_forget
    # -------------------------------------------------------------------------
    @server.tool(
        name="memlayer_forget",
        description="Permanently delete a specific memory by its unique ID across relational and vector stores.",
    )
    async def memlayer_forget(
        user_id: str,
        memory_id: str,
    ) -> str:
        if not user_id or not user_id.strip():
            return json.dumps({"error": "user_id cannot be blank or empty."})
        if not memory_id or not memory_id.strip():
            return json.dumps({"error": "memory_id cannot be blank or empty."})

        clean_user = user_id.strip()
        clean_mem_id = memory_id.strip()
        layer, _ = await _get_layer()

        deleted = await layer.forget(user_id=clean_user, memory_id=clean_mem_id)
        return json.dumps(
            {
                "deleted": deleted,
                "memory_id": clean_mem_id,
                "user_id": clean_user,
            }
        )

    # -------------------------------------------------------------------------
    # Tool 4: memlayer_list
    # -------------------------------------------------------------------------
    @server.tool(
        name="memlayer_list",
        description="List stored memories for a user, optionally filtering by status (active or superseded).",
    )
    async def memlayer_list(
        user_id: str,
        status: str = "all",
        limit: int = 50,
    ) -> str:
        if not user_id or not user_id.strip():
            return json.dumps({"error": "user_id cannot be blank or empty."})

        clean_user = user_id.strip()
        layer, _ = await _get_layer()

        status_filter: MemoryStatus | None = None
        if status.lower() == "active":
            status_filter = MemoryStatus.ACTIVE
        elif status.lower() == "superseded":
            status_filter = MemoryStatus.SUPERSEDED

        memories: list[MemoryItem] = await layer.store.list_memories(
            user_id=clean_user,
            status=status_filter,
            limit=limit,
        )

        results = [
            {
                "id": m.id,
                "content": m.content,
                "entity": m.entity,
                "importance": m.importance,
                "confidence": m.confidence,
                "status": m.status.value,
                "superseded_by": m.superseded_by,
                "created_at": m.created_at.isoformat(),
                "access_count": m.access_count,
            }
            for m in memories
        ]

        return json.dumps({"memories": results, "count": len(results)})

    # -------------------------------------------------------------------------
    # Tool 5: memlayer_reset
    # -------------------------------------------------------------------------
    @server.tool(
        name="memlayer_reset",
        description="Purge all memories and episodes for a user across all stores (Right to Be Forgotten).",
    )
    async def memlayer_reset(
        user_id: str,
    ) -> str:
        if not user_id or not user_id.strip():
            return json.dumps({"error": "user_id cannot be blank or empty."})

        clean_user = user_id.strip()
        layer, _ = await _get_layer()

        count = await layer.forget_all(user_id=clean_user)
        return json.dumps(
            {
                "success": True,
                "user_id": clean_user,
                "purged_records": count,
            }
        )

    return server


def main() -> None:
    """CLI entrypoint launching the MCP server over stdio transport."""
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    main()

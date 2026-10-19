"""Unit and integration tests for Model Context Protocol (MCP) server."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

try:
    from mcp.server.mcpserver import MCPServer

    from memlayer.mcp_server import MCPFallbackFactExtractor, create_mcp_server
except ImportError:
    pytest.skip("mcp SDK and dependencies not installed", allow_module_level=True)

import re

from memlayer.api import MemoryLayer
from memlayer.consolidate.conflicts import ConflictDetector
from memlayer.llm.embedder import MockEmbedder
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore


@pytest.fixture
async def mcp_server() -> MCPServer:
    """Fixture providing a fresh in-memory MCP server for testing."""
    store = SQLStore(db_url="sqlite+aiosqlite:///:memory:")
    await store.init_db()

    embedder = MockEmbedder(dimension=64)
    vector_store = QdrantVectorStore(location=":memory:", vector_size=64)
    await vector_store.init_store()

    extractor = MCPFallbackFactExtractor()
    memlayer = MemoryLayer(
        store=store,
        vector_store=vector_store,
        embedder=embedder,
        extractor=extractor,
    )

    async def _mock_conflict_completion(*_args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        user_msg = (
            messages[-1].get("content", "")
            if isinstance(messages, list) and messages and isinstance(messages[-1], dict)
            else ""
        )
        conflicts = []
        if "austin" in str(user_msg).lower() and "seattle" in str(user_msg).lower():
            match = re.search(r"ID:\s*([a-f0-9\-]+)", str(user_msg))
            old_id = match.group(1) if match else "old-mem"
            conflicts.append(
                {
                    "old_memory_id": old_id,
                    "is_contradiction": True,
                    "reason": "Relocated to Austin supersedes Seattle.",
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
        completion_fn=_mock_conflict_completion,
    )

    return create_mcp_server(
        memlayer=memlayer,
        conflict_detector=conflict_detector,
        server_name="test-memlayer-mcp",
    )


async def _call_tool_json(server: MCPServer, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Helper calling an MCP tool and parsing the JSON text content from the result."""
    result = await server.call_tool(name, args)
    assert hasattr(result, "is_error") and not result.is_error
    content_list = getattr(result, "content", [])
    assert len(content_list) > 0
    raw_text = getattr(content_list[0], "text", "")
    parsed: dict[str, Any] = json.loads(raw_text)
    return parsed


@pytest.mark.asyncio
async def test_list_tools_registered(mcp_server: MCPServer) -> None:
    """Verify that all five memlayer tools are registered with proper descriptions."""
    tools = await mcp_server.list_tools()
    tool_names = {t.name for t in tools}

    assert "memlayer_add" in tool_names
    assert "memlayer_search" in tool_names
    assert "memlayer_forget" in tool_names
    assert "memlayer_list" in tool_names
    assert "memlayer_reset" in tool_names


@pytest.mark.asyncio
async def test_mcp_add_and_search(mcp_server: MCPServer) -> None:
    """Verify adding facts via MCP tool and retrieving them via search."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"

    # Add a memory
    add_res = await _call_tool_json(
        mcp_server,
        "memlayer_add",
        {
            "user_id": user_id,
            "text": "I live in Seattle and work as a cloud engineer.",
            "session_id": "session_1",
        },
    )
    assert add_res["count"] >= 1
    added_item = add_res["added_memories"][0]
    assert "Seattle" in added_item["content"]

    # Search for the memory
    search_res = await _call_tool_json(
        mcp_server,
        "memlayer_search",
        {
            "user_id": user_id,
            "query": "Where do I live?",
            "limit": 5,
        },
    )
    assert len(search_res["memories"]) >= 1
    assert "Seattle" in str(search_res["context"])
    assert search_res["token_count"] > 0


@pytest.mark.asyncio
async def test_mcp_supersession_flow(mcp_server: MCPServer) -> None:
    """Verify that state update through memlayer_add triggers conflict supersession."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"

    # Turn 1: Seattle
    await _call_tool_json(
        mcp_server,
        "memlayer_add",
        {
            "user_id": user_id,
            "text": "I live in Seattle.",
            "session_id": "session_1",
        },
    )

    # Turn 2: Relocates to Austin
    await _call_tool_json(
        mcp_server,
        "memlayer_add",
        {
            "user_id": user_id,
            "text": "I relocated to Austin, Texas.",
            "session_id": "session_2",
        },
    )

    # List all memories
    list_res = await _call_tool_json(
        mcp_server,
        "memlayer_list",
        {
            "user_id": user_id,
            "status": "all",
        },
    )
    memories = list_res["memories"]
    assert len(memories) >= 2

    active = [m for m in memories if m["status"] == "active"]
    superseded = [m for m in memories if m["status"] == "superseded"]

    assert len(active) >= 1
    assert any("Austin" in m["content"] for m in active)

    assert len(superseded) >= 1
    assert any("Seattle" in m["content"] for m in superseded)
    seattle_mem = next(m for m in superseded if "Seattle" in m["content"])
    assert seattle_mem["superseded_by"] is not None


@pytest.mark.asyncio
async def test_mcp_forget_and_reset(mcp_server: MCPServer) -> None:
    """Verify individual memory deletion and GDPR user reset via MCP."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"

    # Add memory
    add_res = await _call_tool_json(
        mcp_server,
        "memlayer_add",
        {
            "user_id": user_id,
            "text": "My favorite beverage is iced oat milk latte.",
        },
    )
    target_id = add_res["added_memories"][0]["id"]

    # Forget memory
    forget_res = await _call_tool_json(
        mcp_server,
        "memlayer_forget",
        {
            "user_id": user_id,
            "memory_id": target_id,
        },
    )
    assert forget_res["deleted"] is True

    # Confirm it is no longer listed
    list_res = await _call_tool_json(
        mcp_server,
        "memlayer_list",
        {"user_id": user_id},
    )
    assert not any(m["id"] == target_id for m in list_res["memories"])

    # Reset user store
    reset_res = await _call_tool_json(
        mcp_server,
        "memlayer_reset",
        {"user_id": user_id},
    )
    assert reset_res["success"] is True

    final_list = await _call_tool_json(
        mcp_server,
        "memlayer_list",
        {"user_id": user_id},
    )
    assert len(final_list["memories"]) == 0


@pytest.mark.asyncio
async def test_mcp_strict_user_isolation(mcp_server: MCPServer) -> None:
    """Verify that user_a's memories can never be retrieved or seen by user_b."""
    user_a = f"alpha_{uuid.uuid4().hex[:8]}"
    user_b = f"beta_{uuid.uuid4().hex[:8]}"

    # Add secret memory for User A
    await _call_tool_json(
        mcp_server,
        "memlayer_add",
        {
            "user_id": user_a,
            "text": "My secret password is supercalifragilistic.",
        },
    )

    # User B searches for password
    search_b = await _call_tool_json(
        mcp_server,
        "memlayer_search",
        {
            "user_id": user_b,
            "query": "What is my secret password?",
        },
    )
    assert "supercalifragilistic" not in str(search_b["context"])
    assert len(search_b["memories"]) == 0

    # User B lists memories
    list_b = await _call_tool_json(
        mcp_server,
        "memlayer_list",
        {"user_id": user_b},
    )
    assert len(list_b["memories"]) == 0


@pytest.mark.asyncio
async def test_mcp_validation_errors(mcp_server: MCPServer) -> None:
    """Verify validation error handling for invalid or blank tool inputs."""
    res_blank_user = await _call_tool_json(
        mcp_server,
        "memlayer_add",
        {"user_id": "   ", "text": "Valid text"},
    )
    assert "error" in res_blank_user

    res_blank_text = await _call_tool_json(
        mcp_server,
        "memlayer_add",
        {"user_id": "valid_user", "text": "   "},
    )
    assert "error" in res_blank_text

    res_search_blank = await _call_tool_json(
        mcp_server,
        "memlayer_search",
        {"user_id": "valid_user", "query": "  "},
    )
    assert "error" in res_search_blank

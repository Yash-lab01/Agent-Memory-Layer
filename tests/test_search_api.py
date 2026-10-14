"""Integration tests for MemoryLayer public search API, router gating, and token budgeting."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from memlayer.api import MemoryLayer
from memlayer.extract.extractor import FactExtractor
from memlayer.models import SearchResult


def _make_mock_extractor(facts: list[dict[str, Any]]) -> FactExtractor:
    response_json = json.dumps({"facts": facts})

    async def mock_llm(**_kwargs: Any) -> str:
        return response_json

    return FactExtractor(completion_fn=mock_llm)


class TestSearchAPI:
    """Test suite for MemoryLayer.search() public API."""

    @pytest.mark.asyncio
    async def test_search_turn_gating_skips_trivial_turn(self) -> None:
        mem = await MemoryLayer.create_local(vector_dim=16)

        # Gratitude turn should skip
        result = await mem.search("user_1", "thanks!")
        assert isinstance(result, SearchResult)
        assert result.skipped is True
        assert result.context == ""
        assert result.memories == []
        assert result.token_count == 0

        # Greeting turn should skip
        result_greeting = await mem.search("user_1", "hello there")
        assert result_greeting.skipped is True

        await mem.close()

    @pytest.mark.asyncio
    async def test_search_bypass_router_forces_retrieval(self) -> None:
        extractor = _make_mock_extractor(
            [{"content": "User prefers concise answers", "entity": "Style", "importance": 0.8}]
        )
        mem = await MemoryLayer.create_local(vector_dim=16, extractor=extractor)

        await mem.add("user_bypass", [{"role": "user", "content": "Keep replies short."}])

        # Without bypass: skips
        res_skipped = await mem.search("user_bypass", "ok")
        assert res_skipped.skipped is True

        # With bypass: runs retrieval
        res_forced = await mem.search("user_bypass", "ok", bypass_router=True)
        assert res_forced.skipped is False
        assert len(res_forced.memories) > 0
        assert "User prefers concise answers" in res_forced.context

        await mem.close()

    @pytest.mark.asyncio
    async def test_search_informational_query_retrieves_and_formats(self) -> None:
        extractor = _make_mock_extractor(
            [
                {
                    "content": "User lives in Pune",
                    "entity": "Location",
                    "importance": 0.9,
                    "confidence": 1.0,
                },
                {
                    "content": "User is building Agent Memory Layer",
                    "entity": "Project",
                    "importance": 0.85,
                    "confidence": 1.0,
                },
            ]
        )
        mem = await MemoryLayer.create_local(vector_dim=32, extractor=extractor)

        await mem.add(
            "user_info",
            [{"role": "user", "content": "I live in Pune and I am building Agent Memory Layer."}],
        )

        result = await mem.search("user_info", "Where do I live and what am I building?")
        assert result.skipped is False
        assert len(result.memories) == 2
        assert result.token_count > 0
        assert "User lives in Pune" in result.context
        assert "Agent Memory Layer" in result.context

        # Verify date format provenance in context
        assert "[" in result.context and "]" in result.context

        await mem.close()

    @pytest.mark.asyncio
    async def test_search_token_budget_packing(self) -> None:
        extractor = _make_mock_extractor(
            [
                {
                    "content": f"Detailed fact number {i} regarding system configuration",
                    "importance": 0.7,
                }
                for i in range(10)
            ]
        )
        mem = await MemoryLayer.create_local(vector_dim=16, extractor=extractor)

        await mem.add("user_budget", [{"role": "user", "content": "Configuration list"}])

        # Large budget gets all memories
        res_large = await mem.search(
            "user_budget", "system configuration details", token_budget=1000
        )
        assert len(res_large.memories) == 10

        # Small budget (e.g. 25 tokens) gets only top subset
        res_small = await mem.search("user_budget", "system configuration details", token_budget=25)
        assert len(res_small.memories) < 10
        assert res_small.token_count <= 25

        await mem.close()

    @pytest.mark.asyncio
    async def test_search_records_access_telemetry(self) -> None:
        extractor = _make_mock_extractor(
            [
                {
                    "content": "User favorite language is Python",
                    "entity": "Language",
                    "importance": 0.8,
                }
            ]
        )
        mem = await MemoryLayer.create_local(vector_dim=16, extractor=extractor)

        saved = await mem.add(
            "user_access", [{"role": "user", "content": "Python is my favorite."}]
        )
        mem_id = saved[0].id

        # Verify access count is initially 0
        record_before = await mem.store.get_memory("user_access", mem_id)
        assert record_before is not None
        assert record_before.access_count == 0

        # Perform search that retrieves the memory
        await mem.search("user_access", "What is my favorite language?")

        # Yield control briefly for the background telemetry task to complete
        await asyncio.sleep(0.05)

        record_after = await mem.store.get_memory("user_access", mem_id)
        assert record_after is not None
        assert record_after.access_count == 1
        assert record_after.last_accessed_at is not None

        await mem.close()

    @pytest.mark.asyncio
    async def test_search_multi_user_isolation(self) -> None:
        extractor = _make_mock_extractor(
            [{"content": "Alice's secret PIN is 9876", "entity": "Security", "importance": 1.0}]
        )
        mem = await MemoryLayer.create_local(vector_dim=16, extractor=extractor)

        await mem.add("alice", [{"role": "user", "content": "My PIN is 9876."}])

        # Bob queries for PIN
        bob_res = await mem.search("bob", "What is the secret PIN?")
        assert bob_res.skipped is False
        assert bob_res.memories == []
        assert bob_res.context == ""

        # Alice queries for PIN
        alice_res = await mem.search("alice", "What is the secret PIN?")
        assert len(alice_res.memories) == 1
        assert "9876" in alice_res.context

        await mem.close()

    @pytest.mark.asyncio
    async def test_search_invalid_user_id_raises(self) -> None:
        mem = await MemoryLayer.create_local(vector_dim=16)

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await mem.search("", "query")

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await mem.search("   ", "query")

        await mem.close()

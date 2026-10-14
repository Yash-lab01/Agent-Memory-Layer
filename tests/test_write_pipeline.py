"""Integration tests for MemoryLayer write pipeline, provenance, isolation, and forget API."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from memlayer.api import MemoryLayer
from memlayer.extract.extractor import FactExtractor
from memlayer.models import MemoryStatus, Role


def _make_mock_extractor(facts: list[dict[str, Any]]) -> FactExtractor:
    response_json = json.dumps({"facts": facts})

    async def mock_llm(**_kwargs: Any) -> str:
        return response_json

    return FactExtractor(completion_fn=mock_llm)


class TestMemoryLayerWritePipeline:
    """Test suite for MemoryLayer write path operations."""

    @pytest.mark.asyncio
    async def test_create_local_initialization(self) -> None:
        mem = await MemoryLayer.create_local(vector_dim=32)
        assert mem.store is not None
        assert mem.vector_store is not None
        assert mem.embedder is not None
        assert mem.extractor is not None
        assert mem.episode_logger is not None
        await mem.close()

    @pytest.mark.asyncio
    async def test_add_synchronous_write_and_provenance(self) -> None:
        extractor = _make_mock_extractor(
            [
                {
                    "content": "User lives in Pune",
                    "entity": "User",
                    "importance": 0.9,
                    "confidence": 0.95,
                },
                {
                    "content": "User is a conversational AI researcher",
                    "entity": "User",
                    "importance": 0.85,
                    "confidence": 0.9,
                },
            ]
        )
        mem = await MemoryLayer.create_local(vector_dim=32, extractor=extractor)

        messages = [
            {"role": "user", "content": "I live in Pune and work on conversational AI."},
            {"role": "assistant", "content": "That sounds exciting!"},
        ]

        saved_memories = await mem.add("user_42", messages, session_id="sess_abc")
        assert len(saved_memories) == 2
        assert saved_memories[0].content == "User lives in Pune"
        assert saved_memories[0].user_id == "user_42"
        assert saved_memories[0].status == MemoryStatus.ACTIVE

        # Verify provenance in SQL episode store
        episodes = await mem.episode_logger.get_history("user_42", session_id="sess_abc")
        assert len(episodes) == 2
        assert episodes[0].role == Role.USER
        assert episodes[0].text == "I live in Pune and work on conversational AI."
        # Verify source_episode_id points to the user episode
        assert saved_memories[0].source_episode_id == episodes[0].id

        # Verify SQL memory persistence
        sql_mem = await mem.store.get_memory("user_42", saved_memories[0].id)
        assert sql_mem is not None
        assert sql_mem.content == "User lives in Pune"

        # Verify vector store indexing
        search_vec = [0.1] * 32
        hits = await mem.vector_store.search("user_42", search_vec, limit=5)
        assert len(hits) == 2
        hit_ids = {h.id for h in hits}
        assert saved_memories[0].id in hit_ids
        assert saved_memories[1].id in hit_ids

        await mem.close()

    @pytest.mark.asyncio
    async def test_add_background_async_task(self) -> None:
        extractor = _make_mock_extractor(
            [
                {
                    "content": "User relocated to Bangalore",
                    "entity": "User",
                    "importance": 0.95,
                    "confidence": 1.0,
                }
            ]
        )
        mem = await MemoryLayer.create_local(vector_dim=32, extractor=extractor)

        messages = [{"role": "user", "content": "I just moved to Bangalore."}]
        task = await mem.add("user_async", messages, background=True)

        assert isinstance(task, asyncio.Task)
        saved_memories = await task
        assert len(saved_memories) == 1
        assert saved_memories[0].content == "User relocated to Bangalore"

        # Verify episodes were logged immediately
        episodes = await mem.episode_logger.get_history("user_async")
        assert len(episodes) == 1

        await mem.close()

    @pytest.mark.asyncio
    async def test_add_empty_messages(self) -> None:
        mem = await MemoryLayer.create_local(vector_dim=16)

        result_sync = await mem.add("user_empty", [])
        assert result_sync == []

        task = await mem.add("user_empty", [], background=True)
        assert isinstance(task, asyncio.Task)
        result_async = await task
        assert result_async == []

        await mem.close()

    @pytest.mark.asyncio
    async def test_add_invalid_user_id_raises(self) -> None:
        mem = await MemoryLayer.create_local(vector_dim=16)

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await mem.add("", [{"role": "user", "content": "Hello"}])

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await mem.add("   ", [{"role": "user", "content": "Hello"}])

        await mem.close()

    @pytest.mark.asyncio
    async def test_multi_user_strict_isolation(self) -> None:
        extractor_a = _make_mock_extractor([{"content": "Alpha secret fact", "entity": "Alpha"}])
        extractor_b = _make_mock_extractor([{"content": "Beta public fact", "entity": "Beta"}])

        mem = await MemoryLayer.create_local(vector_dim=16, extractor=extractor_a)

        # Ingest for user_alpha
        await mem.add("user_alpha", [{"role": "user", "content": "Alpha turn"}])

        # Swap extractor for user_beta
        mem.extractor = extractor_b
        await mem.add("user_beta", [{"role": "user", "content": "Beta turn"}])

        # Verify SQL isolation
        alpha_mems = await mem.store.list_memories("user_alpha")
        beta_mems = await mem.store.list_memories("user_beta")
        assert len(alpha_mems) == 1
        assert alpha_mems[0].content == "Alpha secret fact"
        assert len(beta_mems) == 1
        assert beta_mems[0].content == "Beta public fact"

        # Verify Vector store isolation
        dummy_vec = [0.1] * 16
        hits_alpha = await mem.vector_store.search("user_alpha", dummy_vec)
        hits_beta = await mem.vector_store.search("user_beta", dummy_vec)

        assert all(h.payload.get("content") == "Alpha secret fact" for h in hits_alpha)
        assert all(h.payload.get("content") == "Beta public fact" for h in hits_beta)

        await mem.close()

    @pytest.mark.asyncio
    async def test_forget_single_memory(self) -> None:
        extractor = _make_mock_extractor([{"content": "Fact to delete", "entity": "User"}])
        mem = await MemoryLayer.create_local(vector_dim=16, extractor=extractor)

        mems = await mem.add("user_forget", [{"role": "user", "content": "Delete me later"}])
        mem_id = mems[0].id

        # Verify exists before deletion
        assert await mem.store.get_memory("user_forget", mem_id) is not None

        # Execute forget
        success = await mem.forget("user_forget", mem_id)
        assert success is True

        # Verify deleted from SQL
        assert await mem.store.get_memory("user_forget", mem_id) is None

        # Verify deleted from Vector store
        hits = await mem.vector_store.search("user_forget", [0.1] * 16)
        assert len(hits) == 0

        await mem.close()

    @pytest.mark.asyncio
    async def test_forget_all_user_data(self) -> None:
        extractor = _make_mock_extractor([{"content": "Fact 1", "entity": "User"}])
        mem = await MemoryLayer.create_local(vector_dim=16, extractor=extractor)

        await mem.add("user_purge", [{"role": "user", "content": "Purge all"}])
        await mem.add("user_keep", [{"role": "user", "content": "Keep me"}])

        purged_count = await mem.forget_all("user_purge")
        assert purged_count > 0

        # user_purge has 0 memories and 0 episodes
        assert len(await mem.store.list_memories("user_purge")) == 0
        assert len(await mem.episode_logger.get_history("user_purge")) == 0

        # user_keep is intact
        assert len(await mem.store.list_memories("user_keep")) == 1
        assert len(await mem.episode_logger.get_history("user_keep")) == 1

        await mem.close()

    @pytest.mark.asyncio
    async def test_search_not_implemented_milestone_3(self) -> None:
        mem = await MemoryLayer.create_local(vector_dim=16)

        with pytest.raises(NotImplementedError, match="Milestone 3"):
            await mem.search("user_1", "test query")

        await mem.close()

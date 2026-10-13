"""Tests for SQLStore and EpisodeStore relational storage and user isolation."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from memlayer.models import Episode, MemoryItem, MemoryStatus, Role
from memlayer.store.episodes import EpisodeStore
from memlayer.store.sql import SQLStore


@pytest_asyncio.fixture
async def sql_store() -> AsyncGenerator[SQLStore, None]:
    """Fixture providing an initialized in-memory SQLite store."""
    store = SQLStore(db_url="sqlite+aiosqlite:///:memory:")
    await store.init_db()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def episode_store(sql_store: SQLStore) -> EpisodeStore:
    """Fixture providing an EpisodeStore instance."""
    return EpisodeStore(sql_store)


class TestEpisodeCRUD:
    """Unit tests for episode storage operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_episode(self, sql_store: SQLStore, sample_user_id: str) -> None:
        ep = Episode(
            user_id=sample_user_id,
            session_id="session_1",
            role=Role.USER,
            text="Hello world",
        )
        saved = await sql_store.add_episode(ep)
        assert saved.id == ep.id

        retrieved = await sql_store.get_episode(sample_user_id, ep.id)
        assert retrieved is not None
        assert retrieved.id == ep.id
        assert retrieved.user_id == sample_user_id
        assert retrieved.text == "Hello world"
        assert retrieved.role == Role.USER
        assert retrieved.session_id == "session_1"
        assert retrieved.ts.tzinfo is not None

    @pytest.mark.asyncio
    async def test_list_episodes_chronological(
        self, sql_store: SQLStore, sample_user_id: str
    ) -> None:
        ep1 = Episode(
            user_id=sample_user_id,
            session_id="s1",
            text="First message",
            ts=datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC),
        )
        ep2 = Episode(
            user_id=sample_user_id,
            session_id="s1",
            text="Second message",
            ts=datetime(2026, 10, 1, 10, 5, 0, tzinfo=UTC),
        )
        ep3 = Episode(
            user_id=sample_user_id,
            session_id="s2",
            text="Different session",
            ts=datetime(2026, 10, 1, 10, 10, 0, tzinfo=UTC),
        )
        await sql_store.add_episodes([ep2, ep1, ep3])

        all_episodes = await sql_store.list_episodes(sample_user_id)
        assert len(all_episodes) == 3
        assert all_episodes[0].text == "First message"
        assert all_episodes[1].text == "Second message"
        assert all_episodes[2].text == "Different session"

        session_1_episodes = await sql_store.list_episodes(sample_user_id, session_id="s1")
        assert len(session_1_episodes) == 2
        assert [e.text for e in session_1_episodes] == ["First message", "Second message"]


class TestMemoryCRUD:
    """Unit tests for memory item storage operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_memory(self, sql_store: SQLStore, sample_user_id: str) -> None:
        mem = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            entity="User",
            importance=0.9,
            confidence=0.95,
        )
        await sql_store.add_memory(mem)

        retrieved = await sql_store.get_memory(sample_user_id, mem.id)
        assert retrieved is not None
        assert retrieved.id == mem.id
        assert retrieved.content == "User lives in Pune"
        assert retrieved.entity == "User"
        assert retrieved.importance == 0.9
        assert retrieved.confidence == 0.95
        assert retrieved.status == MemoryStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_mark_superseded(self, sql_store: SQLStore, sample_user_id: str) -> None:
        old_mem = MemoryItem(user_id=sample_user_id, content="Lives in Pune")
        new_mem = MemoryItem(user_id=sample_user_id, content="Moved to Bangalore")
        await sql_store.add_memories([old_mem, new_mem])

        success = await sql_store.mark_superseded(
            user_id=sample_user_id,
            old_memory_id=old_mem.id,
            new_memory_id=new_mem.id,
        )
        assert success is True

        updated_old = await sql_store.get_memory(sample_user_id, old_mem.id)
        assert updated_old is not None
        assert updated_old.status == MemoryStatus.SUPERSEDED
        assert updated_old.superseded_by == new_mem.id

        # Verify active memories query excludes superseded
        active_memories = await sql_store.list_memories(
            user_id=sample_user_id, status=MemoryStatus.ACTIVE
        )
        assert len(active_memories) == 1
        assert active_memories[0].content == "Moved to Bangalore"

    @pytest.mark.asyncio
    async def test_record_access_increments(self, sql_store: SQLStore, sample_user_id: str) -> None:
        mem = MemoryItem(user_id=sample_user_id, content="User speaks English")
        await sql_store.add_memory(mem)

        access_ts = datetime(2026, 10, 13, 15, 30, 0, tzinfo=UTC)
        success = await sql_store.record_access(
            user_id=sample_user_id,
            memory_id=mem.id,
            access_time=access_ts,
        )
        assert success is True

        updated = await sql_store.get_memory(sample_user_id, mem.id)
        assert updated is not None
        assert updated.access_count == 1
        assert updated.last_accessed_at is not None
        assert updated.last_accessed_at == access_ts

    @pytest.mark.asyncio
    async def test_delete_memory(self, sql_store: SQLStore, sample_user_id: str) -> None:
        mem = MemoryItem(user_id=sample_user_id, content="Fact to forget")
        await sql_store.add_memory(mem)

        deleted = await sql_store.delete_memory(sample_user_id, mem.id)
        assert deleted is True

        retrieved = await sql_store.get_memory(sample_user_id, mem.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_user_data(self, sql_store: SQLStore, sample_user_id: str) -> None:
        ep = Episode(user_id=sample_user_id, text="My message")
        mem = MemoryItem(user_id=sample_user_id, content="My memory")
        await sql_store.add_episode(ep)
        await sql_store.add_memory(mem)

        count = await sql_store.delete_user_data(sample_user_id)
        assert count == 2

        assert await sql_store.get_episode(sample_user_id, ep.id) is None
        assert await sql_store.get_memory(sample_user_id, mem.id) is None


class TestUserIsolation:
    """Rigorous multi-tenant isolation tests preventing cross-user data leaks."""

    @pytest.mark.asyncio
    async def test_user_cannot_access_other_users_episodes(
        self,
        sql_store: SQLStore,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        ep_user1 = Episode(user_id=sample_user_id, text="Secret from User 1")
        await sql_store.add_episode(ep_user1)

        # User 2 tries to fetch User 1's episode by ID
        result = await sql_store.get_episode(alternate_user_id, ep_user1.id)
        assert result is None

        # User 2 lists episodes
        user2_episodes = await sql_store.list_episodes(alternate_user_id)
        assert len(user2_episodes) == 0

    @pytest.mark.asyncio
    async def test_user_cannot_access_or_modify_other_users_memories(
        self,
        sql_store: SQLStore,
        sample_user_id: str,
        alternate_user_id: str,
    ) -> None:
        mem_user1 = MemoryItem(
            user_id=sample_user_id,
            content="Confidential info of User 1",
        )
        await sql_store.add_memory(mem_user1)

        # User 2 cannot read
        assert await sql_store.get_memory(alternate_user_id, mem_user1.id) is None
        user2_mems = await sql_store.list_memories(alternate_user_id)
        assert len(user2_mems) == 0

        # User 2 cannot mark superseded
        superseded = await sql_store.mark_superseded(alternate_user_id, mem_user1.id, "fake_id")
        assert superseded is False

        # User 2 cannot record access
        accessed = await sql_store.record_access(alternate_user_id, mem_user1.id)
        assert accessed is False

        # User 2 cannot delete
        deleted = await sql_store.delete_memory(alternate_user_id, mem_user1.id)
        assert deleted is False

        # Verify User 1's memory was untouched
        original = await sql_store.get_memory(sample_user_id, mem_user1.id)
        assert original is not None
        assert original.status == MemoryStatus.ACTIVE
        assert original.access_count == 0


class TestEpisodeStoreHelper:
    """Tests for the high-level EpisodeStore helper."""

    @pytest.mark.asyncio
    async def test_append_turn(self, episode_store: EpisodeStore, sample_user_id: str) -> None:
        ep = await episode_store.append_turn(
            user_id=sample_user_id,
            role=Role.USER,
            text="Hello from helper",
            session_id="sess_abc",
        )
        assert ep.user_id == sample_user_id
        assert ep.text == "Hello from helper"
        assert ep.session_id == "sess_abc"

    @pytest.mark.asyncio
    async def test_append_turns_batch(
        self, episode_store: EpisodeStore, sample_user_id: str
    ) -> None:
        messages = [
            {"role": "user", "content": "Where is the library?"},
            {"role": "assistant", "content": "Down the hall."},
            {"role": "user", "content": "   "},  # Blank message should be ignored
        ]
        episodes = await episode_store.append_turns(
            user_id=sample_user_id,
            messages=messages,
            session_id="chat_001",
        )
        assert len(episodes) == 2
        assert episodes[0].text == "Where is the library?"
        assert episodes[1].text == "Down the hall."

        history = await episode_store.get_history(sample_user_id, session_id="chat_001")
        assert len(history) == 2

"""Episode store for recording dialogue history with provenance tracking."""

from datetime import UTC, datetime

from memlayer.models import Episode, Role
from memlayer.store.sql import SQLStore


class EpisodeStore:
    """Manages dialogue episodes and session history backed by relational storage."""

    def __init__(self, sql_store: SQLStore) -> None:
        self.sql_store = sql_store

    async def append_turn(
        self,
        user_id: str,
        role: Role | str,
        text: str,
        session_id: str = "default",
        ts: datetime | None = None,
    ) -> Episode:
        """Log a single dialogue turn to the episode store."""
        role_enum = Role(role) if isinstance(role, str) else role
        episode = Episode(
            user_id=user_id,
            session_id=session_id,
            role=role_enum,
            text=text,
            ts=ts or datetime.now(UTC),
        )
        return await self.sql_store.add_episode(episode)

    async def append_turns(
        self,
        user_id: str,
        messages: list[dict[str, str]],
        session_id: str = "default",
    ) -> list[Episode]:
        """Log a batch of dialogue messages (e.g. from a chat turn)."""
        episodes: list[Episode] = []
        now = datetime.now(UTC)
        for msg in messages:
            role_str = msg.get("role", "user")
            content = msg.get("content", "")
            if not content.strip():
                continue
            episodes.append(
                Episode(
                    user_id=user_id,
                    session_id=session_id,
                    role=Role(role_str),
                    text=content,
                    ts=now,
                )
            )
        return await self.sql_store.add_episodes(episodes)

    async def get_history(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """Fetch chronological dialogue history for a user and optional session."""
        return await self.sql_store.list_episodes(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )

    async def get_episode(self, user_id: str, episode_id: str) -> Episode | None:
        """Fetch a single episode by ID with user ownership check."""
        return await self.sql_store.get_episode(user_id=user_id, episode_id=episode_id)


# Alias for compatibility with PRD and public specifications
EpisodeLogger = EpisodeStore

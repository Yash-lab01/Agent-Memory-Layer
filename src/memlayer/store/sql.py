"""SQLAlchemy 2.0 relational store for episodes and memories with strict user isolation."""

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    delete,
    select,
    update,
)
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memlayer.models import Episode, MemoryItem, MemoryStatus, Role


def _ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure datetime has timezone UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""


class EpisodeRow(Base):
    """Raw dialogue turn for provenance tracking."""

    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(
        String(128), index=True, nullable=False, default="default"
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    def to_pydantic(self) -> Episode:
        """Convert row to Pydantic Episode model."""
        return Episode(
            id=self.id,
            user_id=self.user_id,
            session_id=self.session_id,
            role=Role(self.role),
            text=self.text,
            ts=_ensure_utc(self.ts) or datetime.now(UTC),
        )

    @classmethod
    def from_pydantic(cls, ep: Episode) -> "EpisodeRow":
        """Create row from Pydantic Episode model."""
        return cls(
            id=ep.id,
            user_id=ep.user_id,
            session_id=ep.session_id,
            role=ep.role.value,
            text=ep.text,
            ts=ep.ts,
        )


class MemoryRow(Base):
    """Extracted atomic fact with metadata and provenance links."""

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    entity: Mapped[str | None] = mapped_column(String(256), index=True, nullable=True)
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=MemoryStatus.ACTIVE.value, index=True
    )
    superseded_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("memories.id"), nullable=True
    )
    source_episode_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("episodes.id"), nullable=True
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def to_pydantic(self) -> MemoryItem:
        """Convert row to Pydantic MemoryItem model."""
        return MemoryItem(
            id=self.id,
            user_id=self.user_id,
            content=self.content,
            entity=self.entity,
            event_time=_ensure_utc(self.event_time),
            created_at=_ensure_utc(self.created_at) or datetime.now(UTC),
            importance=self.importance,
            confidence=self.confidence,
            status=MemoryStatus(self.status),
            superseded_by=self.superseded_by,
            source_episode_id=self.source_episode_id,
            last_accessed_at=_ensure_utc(self.last_accessed_at),
            access_count=self.access_count,
        )

    @classmethod
    def from_pydantic(cls, mem: MemoryItem) -> "MemoryRow":
        """Create row from Pydantic MemoryItem model."""
        return cls(
            id=mem.id,
            user_id=mem.user_id,
            content=mem.content,
            entity=mem.entity,
            event_time=mem.event_time,
            created_at=mem.created_at,
            importance=mem.importance,
            confidence=mem.confidence,
            status=mem.status.value,
            superseded_by=mem.superseded_by,
            source_episode_id=mem.source_episode_id,
            last_accessed_at=mem.last_accessed_at,
            access_count=mem.access_count,
        )


class SQLStore:
    """Async relational storage adapter enforcing per-user isolation."""

    def __init__(
        self,
        db_url: str = "sqlite+aiosqlite:///:memory:",
        echo: bool = False,
        **engine_kwargs: Any,
    ) -> None:
        self.db_url = db_url
        self.engine: AsyncEngine = create_async_engine(db_url, echo=echo, **engine_kwargs)
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Create tables if they do not exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Dispose of engine connections."""
        await self.engine.dispose()

    # -------------------------------------------------------------------------
    # Episode Operations (Scoped by user_id)
    # -------------------------------------------------------------------------

    async def add_episode(self, episode: Episode) -> Episode:
        """Add a single dialogue episode to the provenance store."""
        row = EpisodeRow.from_pydantic(episode)
        async with self.session_factory() as session:
            async with session.begin():
                session.add(row)
        return episode

    async def add_episodes(self, episodes: list[Episode]) -> list[Episode]:
        """Batch add episodes."""
        if not episodes:
            return []
        rows = [EpisodeRow.from_pydantic(ep) for ep in episodes]
        async with self.session_factory() as session:
            async with session.begin():
                session.add_all(rows)
        return episodes

    async def get_episode(self, user_id: str, episode_id: str) -> Episode | None:
        """Retrieve a specific episode, strictly verifying user ownership."""
        user_id_clean = user_id.strip()
        async with self.session_factory() as session:
            stmt = select(EpisodeRow).where(
                EpisodeRow.user_id == user_id_clean,
                EpisodeRow.id == episode_id,
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return row.to_pydantic() if row else None

    async def list_episodes(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """List episodes for a user, ordered chronologically."""
        user_id_clean = user_id.strip()
        async with self.session_factory() as session:
            stmt = select(EpisodeRow).where(EpisodeRow.user_id == user_id_clean)
            if session_id:
                stmt = stmt.where(EpisodeRow.session_id == session_id)
            stmt = stmt.order_by(EpisodeRow.ts.asc()).limit(limit)
            result = await session.execute(stmt)
            ep_rows = result.scalars().all()
            return [row.to_pydantic() for row in ep_rows]

    # -------------------------------------------------------------------------
    # Memory Operations (Scoped by user_id)
    # -------------------------------------------------------------------------

    async def add_memory(self, memory: MemoryItem) -> MemoryItem:
        """Insert a single extracted memory item."""
        row = MemoryRow.from_pydantic(memory)
        async with self.session_factory() as session:
            async with session.begin():
                session.add(row)
        return memory

    async def add_memories(self, memories: list[MemoryItem]) -> list[MemoryItem]:
        """Batch insert extracted memory items."""
        if not memories:
            return []
        rows = [MemoryRow.from_pydantic(mem) for mem in memories]
        async with self.session_factory() as session:
            async with session.begin():
                session.add_all(rows)
        return memories

    async def get_memory(self, user_id: str, memory_id: str) -> MemoryItem | None:
        """Retrieve a memory item, strictly verifying user ownership."""
        user_id_clean = user_id.strip()
        async with self.session_factory() as session:
            stmt = select(MemoryRow).where(
                MemoryRow.user_id == user_id_clean,
                MemoryRow.id == memory_id,
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return row.to_pydantic() if row else None

    async def list_memories(
        self,
        user_id: str,
        status: MemoryStatus | None = None,
        entity: str | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """List memory items for a specific user with optional filters."""
        user_id_clean = user_id.strip()
        async with self.session_factory() as session:
            stmt = select(MemoryRow).where(MemoryRow.user_id == user_id_clean)
            if status is not None:
                stmt = stmt.where(MemoryRow.status == status.value)
            if entity:
                stmt = stmt.where(MemoryRow.entity == entity)
            stmt = stmt.order_by(MemoryRow.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            mem_rows = result.scalars().all()
            return [row.to_pydantic() for row in mem_rows]

    async def mark_superseded(
        self,
        user_id: str,
        old_memory_id: str,
        new_memory_id: str,
    ) -> bool:
        """Mark an old memory as superseded by a new memory.

        Verifies that the target memory belongs to user_id.
        """
        user_id_clean = user_id.strip()
        async with self.session_factory() as session:
            async with session.begin():
                stmt = (
                    update(MemoryRow)
                    .where(
                        MemoryRow.user_id == user_id_clean,
                        MemoryRow.id == old_memory_id,
                    )
                    .values(
                        status=MemoryStatus.SUPERSEDED.value,
                        superseded_by=new_memory_id,
                    )
                )
                result = await session.execute(stmt)
                cursor_res = cast(CursorResult[Any], result)
                return bool(cursor_res.rowcount > 0)

    async def record_access(
        self,
        user_id: str,
        memory_id: str,
        access_time: datetime | None = None,
    ) -> bool:
        """Increment access count and update last_accessed_at."""
        user_id_clean = user_id.strip()
        ts = access_time or datetime.now(UTC)
        async with self.session_factory() as session:
            async with session.begin():
                stmt = (
                    update(MemoryRow)
                    .where(
                        MemoryRow.user_id == user_id_clean,
                        MemoryRow.id == memory_id,
                    )
                    .values(
                        access_count=MemoryRow.access_count + 1,
                        last_accessed_at=ts,
                    )
                )
                result = await session.execute(stmt)
                cursor_res = cast(CursorResult[Any], result)
                return bool(cursor_res.rowcount > 0)

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Explicitly delete a memory record (for GDPR/forget API)."""
        user_id_clean = user_id.strip()
        async with self.session_factory() as session:
            async with session.begin():
                stmt = delete(MemoryRow).where(
                    MemoryRow.user_id == user_id_clean,
                    MemoryRow.id == memory_id,
                )
                result = await session.execute(stmt)
                cursor_res = cast(CursorResult[Any], result)
                return bool(cursor_res.rowcount > 0)

    async def delete_user_data(self, user_id: str) -> int:
        """Hard-delete all memories and episodes for a user (GDPR forget-all)."""
        user_id_clean = user_id.strip()
        async with self.session_factory() as session:
            async with session.begin():
                stmt_mem = delete(MemoryRow).where(MemoryRow.user_id == user_id_clean)
                res_mem = cast(CursorResult[Any], await session.execute(stmt_mem))
                stmt_ep = delete(EpisodeRow).where(EpisodeRow.user_id == user_id_clean)
                res_ep = cast(CursorResult[Any], await session.execute(stmt_ep))
                return int((res_mem.rowcount or 0) + (res_ep.rowcount or 0))


# Alias for compatibility with PRD and public specifications
SQLMemoryStore = SQLStore

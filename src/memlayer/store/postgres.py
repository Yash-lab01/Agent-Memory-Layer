"""PostgreSQL relational store and pgvector adapter for server deployments."""

from typing import Any, cast

from sqlalchemy import (
    JSON,
    String,
    delete,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memlayer.store.sql import SQLStore
from memlayer.store.vector_base import VectorPoint, VectorSearchResult, VectorStoreBase

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # type: ignore[assignment,misc]


class PostgresSQLStore(SQLStore):
    """PostgreSQL relational store backed by asyncpg with connection pooling."""

    def __init__(
        self,
        db_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/memlayer",
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        **engine_kwargs: Any,
    ) -> None:
        super().__init__(
            db_url=db_url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            **engine_kwargs,
        )


class VectorBase(DeclarativeBase):
    """Declarative base for pgvector tables."""


class PgMemoryVector(VectorBase):
    """pgvector table storing memory embeddings and JSON payloads."""

    __tablename__ = "memory_vectors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="active")
    embedding: Mapped[Any] = mapped_column(
        Vector(384) if Vector is not None else String, nullable=False
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class PgVectorStore(VectorStoreBase):
    """Vector store adapter backed by PostgreSQL pgvector extension."""

    def __init__(
        self,
        db_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/memlayer",
        vector_size: int = 384,
        table_name: str = "memory_vectors",
        echo: bool = False,
        **engine_kwargs: Any,
    ) -> None:
        if Vector is None:
            raise ImportError(
                "pgvector is not installed. Install with `pip install agent-memory-layer[postgres]`"
            )
        self.db_url = db_url
        self.vector_size = vector_size
        self.table_name = table_name
        self.engine: AsyncEngine = create_async_engine(
            db_url,
            echo=echo,
            pool_pre_ping=True,
            **engine_kwargs,
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_store(self) -> None:
        """Enable pgvector extension and create memory_vectors table."""
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(VectorBase.metadata.create_all)

    async def upsert(self, user_id: str, points: list[VectorPoint]) -> None:
        """Insert or update vector points with user_id enforcement."""
        if not points:
            return

        clean_user_id = user_id.strip()
        async with self.session_factory() as session:
            async with session.begin():
                for p in points:
                    payload = dict(p.payload)
                    payload["user_id"] = clean_user_id
                    status = str(payload.get("status", "active"))

                    stmt = pg_insert(PgMemoryVector).values(
                        id=p.id,
                        user_id=clean_user_id,
                        status=status,
                        embedding=p.vector,
                        payload=payload,
                    )
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=[PgMemoryVector.id],
                        set_={
                            "status": status,
                            "embedding": p.vector,
                            "payload": payload,
                        },
                    )
                    await session.execute(upsert_stmt)

    async def search(
        self,
        user_id: str,
        query_vector: list[float],
        limit: int = 10,
        status: str | None = "active",
    ) -> list[VectorSearchResult]:
        """Perform cosine distance similarity search strictly filtered by user_id."""
        clean_user_id = user_id.strip()
        async with self.session_factory() as session:
            distance_expr = PgMemoryVector.embedding.cosine_distance(query_vector).label("distance")

            stmt = select(
                PgMemoryVector.id,
                PgMemoryVector.payload,
                distance_expr,
            ).where(PgMemoryVector.user_id == clean_user_id)

            if status is not None:
                stmt = stmt.where(PgMemoryVector.status == status)

            stmt = stmt.order_by(distance_expr.asc()).limit(limit)
            result = await session.execute(stmt)
            rows = result.all()

            results: list[VectorSearchResult] = []
            for row in rows:
                p_id, payload, dist = row[0], row[1], float(row[2])
                similarity = max(0.0, 1.0 - dist)
                results.append(
                    VectorSearchResult(
                        id=str(p_id),
                        score=similarity,
                        payload=payload if isinstance(payload, dict) else {},
                    )
                )
            return results

    async def delete(self, user_id: str, point_ids: list[str]) -> None:
        """Delete specific vector points belonging strictly to user_id."""
        if not point_ids:
            return

        clean_user_id = user_id.strip()
        async with self.session_factory() as session:
            async with session.begin():
                stmt = delete(PgMemoryVector).where(
                    PgMemoryVector.user_id == clean_user_id,
                    PgMemoryVector.id.in_(point_ids),
                )
                await session.execute(stmt)

    async def delete_user_vectors(self, user_id: str) -> int:
        """Purge all vector embeddings for user_id."""
        clean_user_id = user_id.strip()
        async with self.session_factory() as session:
            async with session.begin():
                stmt = delete(PgMemoryVector).where(PgMemoryVector.user_id == clean_user_id)
                res = cast(CursorResult[Any], await session.execute(stmt))
                return int(res.rowcount or 0)

    async def close(self) -> None:
        """Dispose of engine connections."""
        await self.engine.dispose()

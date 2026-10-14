"""Tests for PostgreSQL storage adapters and pgvector configuration."""

from memlayer.store.postgres import PgMemoryVector, PgVectorStore, PostgresSQLStore
from memlayer.store.vector_base import VectorPoint


def test_postgres_sql_store_initialization() -> None:
    """Verify PostgresSQLStore configures asyncpg connection parameters."""
    custom_url = "postgresql+asyncpg://myuser:mypass@localhost:5432/testdb"
    store = PostgresSQLStore(db_url=custom_url, pool_size=5, max_overflow=10)
    assert store.db_url == custom_url
    assert store.engine.url.render_as_string(hide_password=False) == custom_url
    assert store.engine.url.username == "myuser"
    assert store.engine.url.database == "testdb"


def test_pgvector_store_initialization() -> None:
    """Verify PgVectorStore attributes and vector dimensions."""
    store = PgVectorStore(
        db_url="postgresql+asyncpg://postgres:postgres@localhost:5432/memlayer",
        vector_size=384,
    )
    assert store.vector_size == 384
    assert store.table_name == "memory_vectors"


def test_pgvector_point_schema() -> None:
    """Verify PgMemoryVector ORM mappings and user isolation fields."""
    assert hasattr(PgMemoryVector, "id")
    assert hasattr(PgMemoryVector, "user_id")
    assert hasattr(PgMemoryVector, "status")
    assert hasattr(PgMemoryVector, "embedding")
    assert hasattr(PgMemoryVector, "payload")

    # Verify VectorPoint dataclass
    p = VectorPoint(id="p1", vector=[0.1] * 384, payload={"status": "active"})
    assert p.id == "p1"
    assert len(p.vector) == 384
    assert p.payload["status"] == "active"

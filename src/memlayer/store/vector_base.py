"""Abstract interface for pluggable vector storage backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorPoint:
    """A vector embedding with an ID and metadata payload."""

    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    """A matched vector search hit with similarity score and payload."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStoreBase(ABC):
    """Abstract interface defining required vector store operations."""

    @abstractmethod
    async def init_store(self) -> None:
        """Initialize collection, index schemas, and connection pools."""

    @abstractmethod
    async def upsert(self, user_id: str, points: list[VectorPoint]) -> None:
        """Insert or update vector points, ensuring payload includes user_id."""

    @abstractmethod
    async def search(
        self,
        user_id: str,
        query_vector: list[float],
        limit: int = 10,
        status: str | None = "active",
    ) -> list[VectorSearchResult]:
        """Perform semantic similarity search strictly filtered by user_id."""

    @abstractmethod
    async def delete(self, user_id: str, point_ids: list[str]) -> None:
        """Delete specific vector points belonging to user_id."""

    @abstractmethod
    async def delete_user_vectors(self, user_id: str) -> int:
        """Purge all vector points associated with user_id."""

    @abstractmethod
    async def close(self) -> None:
        """Dispose of client connections and resources."""

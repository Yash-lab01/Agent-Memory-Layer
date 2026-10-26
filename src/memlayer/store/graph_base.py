"""Abstract interface for pluggable knowledge graph storage backends."""

from abc import ABC, abstractmethod

from memlayer.models import EntityNode, RelationEdge


class GraphStoreBase(ABC):
    """Abstract interface defining required knowledge graph operations."""

    @abstractmethod
    async def init_store(self) -> None:
        """Initialize graph schemas, indices, or connections."""

    @abstractmethod
    async def add_node(self, user_id: str, node: EntityNode) -> EntityNode:
        """Add or update an entity vertex, strictly scoped to user_id."""

    @abstractmethod
    async def add_edge(self, user_id: str, edge: RelationEdge) -> RelationEdge:
        """Add a directed relation edge between entities, strictly scoped to user_id."""

    @abstractmethod
    async def get_node(self, user_id: str, name: str) -> EntityNode | None:
        """Fetch an entity node by canonical name for user_id."""

    @abstractmethod
    async def get_neighbors(
        self, user_id: str, entity: str, direction: str = "both"
    ) -> list[RelationEdge]:
        """Fetch relation edges connected to an entity ('out', 'in', or 'both')."""

    @abstractmethod
    async def find_paths(
        self, user_id: str, source: str, target: str, max_depth: int = 3
    ) -> list[list[RelationEdge]]:
        """Find all multi-hop relation paths between source and target up to max_depth."""

    @abstractmethod
    async def get_subgraph(
        self, user_id: str, entities: list[str], max_depth: int = 2
    ) -> tuple[list[EntityNode], list[RelationEdge]]:
        """Extract multi-hop neighborhood subgraph surrounding the specified entities."""

    @abstractmethod
    async def delete_user_graph(self, user_id: str) -> int:
        """Purge all nodes and edges belonging to user_id (tenant isolation/GDPR)."""

    @abstractmethod
    async def close(self) -> None:
        """Dispose of graph store connections and resources."""

"""In-memory, zero-dependency knowledge graph store implementation."""

from collections import deque
from collections.abc import Sequence

from memlayer.models import EntityNode, RelationEdge
from memlayer.store.graph_base import GraphStoreBase


def _normalize(name: str) -> str:
    """Normalize entity name for case-insensitive lookup."""
    return name.strip().lower()


class InMemoryGraphStore(GraphStoreBase):
    """Pure-Python, zero-overhead in-memory knowledge graph store with tenant isolation."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, EntityNode]] = {}
        self._edges: dict[str, list[RelationEdge]] = {}
        self._out_edges: dict[str, dict[str, list[RelationEdge]]] = {}
        self._in_edges: dict[str, dict[str, list[RelationEdge]]] = {}

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self._nodes:
            self._nodes[user_id] = {}
            self._edges[user_id] = []
            self._out_edges[user_id] = {}
            self._in_edges[user_id] = {}

    async def init_store(self) -> None:
        """Initialize in-memory structures (no-op)."""

    async def add_node(self, user_id: str, node: EntityNode) -> EntityNode:
        """Add or update an entity vertex for user_id."""
        clean_user = user_id.strip()
        if not clean_user:
            raise ValueError("user_id cannot be blank")
        if node.user_id != clean_user:
            raise ValueError(f"Node user_id '{node.user_id}' does not match '{clean_user}'")

        self._ensure_user(clean_user)
        norm_name = _normalize(node.name)
        self._nodes[clean_user][norm_name] = node
        return node

    async def add_edge(self, user_id: str, edge: RelationEdge) -> RelationEdge:
        """Add a directed relation edge between entities for user_id."""
        clean_user = user_id.strip()
        if not clean_user:
            raise ValueError("user_id cannot be blank")
        if edge.user_id != clean_user:
            raise ValueError(f"Edge user_id '{edge.user_id}' does not match '{clean_user}'")

        self._ensure_user(clean_user)

        # Auto-create source node if missing
        norm_source = _normalize(edge.source)
        if norm_source not in self._nodes[clean_user]:
            self._nodes[clean_user][norm_source] = EntityNode(
                user_id=clean_user,
                name=edge.source,
                entity_type="concept",
            )

        # Auto-create target node if missing
        norm_target = _normalize(edge.target)
        if norm_target not in self._nodes[clean_user]:
            self._nodes[clean_user][norm_target] = EntityNode(
                user_id=clean_user,
                name=edge.target,
                entity_type="concept",
            )

        # Check for existing matching edge to update or append
        existing_edges = self._edges[clean_user]
        for idx, existing in enumerate(existing_edges):
            if (
                _normalize(existing.source) == norm_source
                and _normalize(existing.target) == norm_target
                and existing.relation.strip().upper() == edge.relation.strip().upper()
            ):
                # Update existing edge
                existing_edges[idx] = edge
                # Update out_edges
                out_list = self._out_edges[clean_user].get(norm_source, [])
                for o_idx, o_edge in enumerate(out_list):
                    if o_edge.id == existing.id:
                        out_list[o_idx] = edge
                # Update in_edges
                in_list = self._in_edges[clean_user].get(norm_target, [])
                for i_idx, i_edge in enumerate(in_list):
                    if i_edge.id == existing.id:
                        in_list[i_idx] = edge
                return edge

        # Append new edge
        self._edges[clean_user].append(edge)
        self._out_edges[clean_user].setdefault(norm_source, []).append(edge)
        self._in_edges[clean_user].setdefault(norm_target, []).append(edge)
        return edge

    async def get_node(self, user_id: str, name: str) -> EntityNode | None:
        """Fetch an entity node by canonical name."""
        clean_user = user_id.strip()
        norm_name = _normalize(name)
        return self._nodes.get(clean_user, {}).get(norm_name)

    async def get_neighbors(
        self, user_id: str, entity: str, direction: str = "both"
    ) -> list[RelationEdge]:
        """Fetch relation edges connected to an entity ('out', 'in', or 'both')."""
        clean_user = user_id.strip()
        norm_name = _normalize(entity)
        out_list = self._out_edges.get(clean_user, {}).get(norm_name, [])
        in_list = self._in_edges.get(clean_user, {}).get(norm_name, [])

        if direction == "out":
            return list(out_list)
        if direction == "in":
            return list(in_list)
        return list(out_list) + list(in_list)

    async def find_paths(
        self, user_id: str, source: str, target: str, max_depth: int = 3
    ) -> list[list[RelationEdge]]:
        """Find all multi-hop relation paths between source and target up to max_depth."""
        clean_user = user_id.strip()
        norm_source = _normalize(source)
        norm_target = _normalize(target)

        if norm_source == norm_target:
            return []

        out_map = self._out_edges.get(clean_user, {})
        if norm_source not in out_map:
            return []

        results: list[list[RelationEdge]] = []
        # Queue item: (current_node_norm, path: list[RelationEdge], visited_nodes: set[str])
        queue: deque[tuple[str, list[RelationEdge], set[str]]] = deque(
            [(norm_source, [], {norm_source})]
        )

        while queue:
            curr_node, path, visited = queue.popleft()
            if len(path) >= max_depth:
                continue

            for edge in out_map.get(curr_node, []):
                next_norm = _normalize(edge.target)
                if next_norm == norm_target:
                    results.append(path + [edge])
                elif next_norm not in visited and len(path) + 1 < max_depth:
                    queue.append((next_norm, path + [edge], visited | {next_norm}))

        return results

    async def get_subgraph(
        self, user_id: str, entities: Sequence[str], max_depth: int = 2
    ) -> tuple[list[EntityNode], list[RelationEdge]]:
        """Extract multi-hop neighborhood subgraph surrounding the specified entities."""
        clean_user = user_id.strip()
        user_nodes = self._nodes.get(clean_user, {})
        out_map = self._out_edges.get(clean_user, {})
        in_map = self._in_edges.get(clean_user, {})

        visited_nodes: dict[str, EntityNode] = {}
        visited_edges: dict[str, RelationEdge] = {}

        # Add initial seed entities if present
        current_frontier: set[str] = set()
        for ent in entities:
            norm_ent = _normalize(ent)
            if norm_ent in user_nodes:
                visited_nodes[norm_ent] = user_nodes[norm_ent]
                current_frontier.add(norm_ent)

        for _ in range(max_depth):
            if not current_frontier:
                break
            next_frontier: set[str] = set()

            for node_norm in current_frontier:
                # Explore outbound edges
                for edge in out_map.get(node_norm, []):
                    visited_edges[edge.id] = edge
                    t_norm = _normalize(edge.target)
                    if t_norm not in visited_nodes:
                        if t_norm in user_nodes:
                            visited_nodes[t_norm] = user_nodes[t_norm]
                        next_frontier.add(t_norm)

                # Explore inbound edges
                for edge in in_map.get(node_norm, []):
                    visited_edges[edge.id] = edge
                    s_norm = _normalize(edge.source)
                    if s_norm not in visited_nodes:
                        if s_norm in user_nodes:
                            visited_nodes[s_norm] = user_nodes[s_norm]
                        next_frontier.add(s_norm)

            current_frontier = next_frontier

        return list(visited_nodes.values()), list(visited_edges.values())

    async def delete_user_graph(self, user_id: str) -> int:
        """Purge all nodes and edges belonging to user_id."""
        clean_user = user_id.strip()
        node_count = len(self._nodes.get(clean_user, {}))
        edge_count = len(self._edges.get(clean_user, []))

        self._nodes.pop(clean_user, None)
        self._edges.pop(clean_user, None)
        self._out_edges.pop(clean_user, None)
        self._in_edges.pop(clean_user, None)

        return node_count + edge_count

    async def close(self) -> None:
        """Dispose of in-memory resources."""
        self._nodes.clear()
        self._edges.clear()
        self._out_edges.clear()
        self._in_edges.clear()

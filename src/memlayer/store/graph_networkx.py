"""NetworkX knowledge graph adapter for advanced graph algorithms."""

from collections.abc import Sequence
from typing import Any, cast

from memlayer.models import EntityNode, RelationEdge
from memlayer.store.graph_base import GraphStoreBase

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False


def _normalize(name: str) -> str:
    """Normalize entity name for case-insensitive lookup."""
    return name.strip().lower()


class NetworkXGraphStore(GraphStoreBase):
    """NetworkX-backed knowledge graph store for multi-hop reasoning and centrality analysis.

    Requires `agent-memory-layer[graph]` extra or `networkx>=3.0`.
    """

    def __init__(self) -> None:
        if not HAS_NETWORKX or nx is None:
            raise ImportError(
                "NetworkX is not installed. Install with `pip install agent-memory-layer[graph]` "
                "or `pip install networkx` to use NetworkXGraphStore."
            )
        self._graphs: dict[str, Any] = {}

    def _ensure_graph(self, user_id: str) -> Any:
        if user_id not in self._graphs:
            self._graphs[user_id] = nx.MultiDiGraph()
        return self._graphs[user_id]

    async def init_store(self) -> None:
        """Initialize store (no-op for NetworkX)."""

    async def add_node(self, user_id: str, node: EntityNode) -> EntityNode:
        """Add or update an entity vertex in the user's graph."""
        clean_user = user_id.strip()
        if not clean_user:
            raise ValueError("user_id cannot be blank")
        if node.user_id != clean_user:
            raise ValueError(f"Node user_id '{node.user_id}' does not match '{clean_user}'")

        graph = self._ensure_graph(clean_user)
        norm_name = _normalize(node.name)
        graph.add_node(
            norm_name,
            node=node,
            name=node.name,
            entity_type=node.entity_type,
            properties=node.properties,
        )
        return node

    async def add_edge(self, user_id: str, edge: RelationEdge) -> RelationEdge:
        """Add a directed relation edge between entities in the user's graph."""
        clean_user = user_id.strip()
        if not clean_user:
            raise ValueError("user_id cannot be blank")
        if edge.user_id != clean_user:
            raise ValueError(f"Edge user_id '{edge.user_id}' does not match '{clean_user}'")

        graph = self._ensure_graph(clean_user)
        norm_source = _normalize(edge.source)
        norm_target = _normalize(edge.target)

        # Auto-create source node if missing
        if not graph.has_node(norm_source):
            src_node = EntityNode(user_id=clean_user, name=edge.source, entity_type="concept")
            graph.add_node(
                norm_source,
                node=src_node,
                name=edge.source,
                entity_type="concept",
                properties={},
            )

        # Auto-create target node if missing
        if not graph.has_node(norm_target):
            tgt_node = EntityNode(user_id=clean_user, name=edge.target, entity_type="concept")
            graph.add_node(
                norm_target,
                node=tgt_node,
                name=edge.target,
                entity_type="concept",
                properties={},
            )

        # Check for existing matching edge between these two nodes with same relation
        if graph.has_edge(norm_source, norm_target):
            edge_dict = graph.get_edge_data(norm_source, norm_target)
            if edge_dict:
                for k, data in edge_dict.items():
                    if data.get("relation", "").strip().upper() == edge.relation.strip().upper():
                        graph[norm_source][norm_target][k]["edge"] = edge
                        graph[norm_source][norm_target][k]["weight"] = edge.weight
                        graph[norm_source][norm_target][k]["properties"] = edge.properties
                        return edge

        graph.add_edge(
            norm_source,
            norm_target,
            key=edge.id,
            edge=edge,
            relation=edge.relation,
            weight=edge.weight,
            properties=edge.properties,
        )
        return edge

    async def get_node(self, user_id: str, name: str) -> EntityNode | None:
        """Fetch an entity node by canonical name."""
        clean_user = user_id.strip()
        graph = self._graphs.get(clean_user)
        if graph is None:
            return None
        norm_name = _normalize(name)
        if not graph.has_node(norm_name):
            return None
        node_obj = graph.nodes[norm_name].get("node")
        if isinstance(node_obj, EntityNode):
            return node_obj
        return None

    async def get_neighbors(
        self, user_id: str, entity: str, direction: str = "both"
    ) -> list[RelationEdge]:
        """Fetch relation edges connected to an entity ('out', 'in', or 'both')."""
        clean_user = user_id.strip()
        graph = self._graphs.get(clean_user)
        if graph is None:
            return []

        norm_name = _normalize(entity)
        if not graph.has_node(norm_name):
            return []

        edges: list[RelationEdge] = []
        if direction in ("out", "both"):
            for _, _, data in graph.out_edges(norm_name, data=True):
                edge_val = data.get("edge")
                if isinstance(edge_val, RelationEdge):
                    edges.append(edge_val)

        if direction in ("in", "both"):
            for _, _, data in graph.in_edges(norm_name, data=True):
                edge_val = data.get("edge")
                if isinstance(edge_val, RelationEdge):
                    edges.append(edge_val)

        return edges

    async def find_paths(
        self, user_id: str, source: str, target: str, max_depth: int = 3
    ) -> list[list[RelationEdge]]:
        """Find all multi-hop relation paths between source and target up to max_depth."""
        clean_user = user_id.strip()
        graph = self._graphs.get(clean_user)
        if graph is None:
            return []

        norm_source = _normalize(source)
        norm_target = _normalize(target)

        if (
            norm_source == norm_target
            or not graph.has_node(norm_source)
            or not graph.has_node(norm_target)
        ):
            return []

        results: list[list[RelationEdge]] = []
        try:
            for edge_path in nx.all_simple_edge_paths(
                graph, norm_source, norm_target, cutoff=max_depth
            ):
                path_edges: list[RelationEdge] = []
                for u, v, k in edge_path:
                    data = graph[u][v][k]
                    edge_val = data.get("edge")
                    if isinstance(edge_val, RelationEdge):
                        path_edges.append(edge_val)
                if path_edges:
                    results.append(path_edges)
        except Exception:
            # Node not in graph or no path exists
            return []

        return results

    async def get_subgraph(
        self, user_id: str, entities: Sequence[str], max_depth: int = 2
    ) -> tuple[list[EntityNode], list[RelationEdge]]:
        """Extract multi-hop neighborhood subgraph surrounding the specified entities."""
        clean_user = user_id.strip()
        graph = self._graphs.get(clean_user)
        if graph is None:
            return [], []

        visited_nodes: dict[str, EntityNode] = {}
        visited_edges: dict[str, RelationEdge] = {}

        current_frontier: set[str] = set()
        for ent in entities:
            norm_ent = _normalize(ent)
            if graph.has_node(norm_ent):
                node_obj = graph.nodes[norm_ent].get("node")
                if isinstance(node_obj, EntityNode):
                    visited_nodes[norm_ent] = node_obj
                current_frontier.add(norm_ent)

        for _ in range(max_depth):
            if not current_frontier:
                break
            next_frontier: set[str] = set()

            for node_norm in current_frontier:
                for _, target, data in graph.out_edges(node_norm, data=True):
                    edge_val = data.get("edge")
                    if isinstance(edge_val, RelationEdge):
                        visited_edges[edge_val.id] = edge_val
                    if target not in visited_nodes:
                        target_obj = graph.nodes[target].get("node")
                        if isinstance(target_obj, EntityNode):
                            visited_nodes[target] = target_obj
                        next_frontier.add(target)

                for source, _, data in graph.in_edges(node_norm, data=True):
                    edge_val = data.get("edge")
                    if isinstance(edge_val, RelationEdge):
                        visited_edges[edge_val.id] = edge_val
                    if source not in visited_nodes:
                        source_obj = graph.nodes[source].get("node")
                        if isinstance(source_obj, EntityNode):
                            visited_nodes[source] = source_obj
                        next_frontier.add(source)

            current_frontier = next_frontier

        return list(visited_nodes.values()), list(visited_edges.values())

    async def delete_user_graph(self, user_id: str) -> int:
        """Purge all nodes and edges belonging to user_id."""
        clean_user = user_id.strip()
        graph = self._graphs.pop(clean_user, None)
        if graph is None:
            return 0
        count: int = int(graph.number_of_nodes()) + int(graph.number_of_edges())
        graph.clear()
        return count

    def get_centrality(self, user_id: str) -> dict[str, float]:
        """Compute degree centrality for all entities in the user's graph."""
        clean_user = user_id.strip()
        graph = self._graphs.get(clean_user)
        if graph is None or graph.number_of_nodes() == 0:
            return {}
        return cast(dict[str, float], nx.degree_centrality(graph))

    async def close(self) -> None:
        """Dispose of in-memory NetworkX graphs."""
        for g in self._graphs.values():
            g.clear()
        self._graphs.clear()

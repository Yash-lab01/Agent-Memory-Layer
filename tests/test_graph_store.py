import json
from typing import Any
from unittest.mock import patch

import pytest

from memlayer.api import MemoryLayer
from memlayer.extract.extractor import FactExtractor
from memlayer.models import EntityNode, RelationEdge
from memlayer.retrieve.format import format_subgraph_context
from memlayer.store.graph_memory import InMemoryGraphStore
from memlayer.store.graph_networkx import NetworkXGraphStore

# -----------------------------------------------------------------------------
# Pydantic Model Tests
# -----------------------------------------------------------------------------


def test_entity_node_validation() -> None:
    node = EntityNode(user_id="user_1", name="Alice", entity_type="person")
    assert node.user_id == "user_1"
    assert node.name == "Alice"
    assert node.entity_type == "person"
    assert node.id is not None

    # Blank user_id or name raises ValueError
    with pytest.raises(ValueError, match="Field cannot be blank"):
        EntityNode(user_id="   ", name="Alice")
    with pytest.raises(ValueError, match="Field cannot be blank"):
        EntityNode(user_id="user_1", name="   ")


def test_relation_edge_validation() -> None:
    edge = RelationEdge(
        user_id="user_1",
        source="Alice",
        target="Project Apollo",
        relation="WORKS_ON",
        weight=0.9,
    )
    assert edge.source == "Alice"
    assert edge.target == "Project Apollo"
    assert edge.relation == "WORKS_ON"
    assert edge.weight == 0.9

    # Blank checks
    with pytest.raises(ValueError, match="Field cannot be blank"):
        RelationEdge(user_id="user_1", source="   ", target="B", relation="REL")
    with pytest.raises(ValueError, match="Field cannot be blank"):
        RelationEdge(user_id="   ", source="A", target="B", relation="REL")


# -----------------------------------------------------------------------------
# InMemoryGraphStore Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_node_crud() -> None:
    store = InMemoryGraphStore()
    node = EntityNode(
        user_id="user_alpha",
        name="Alice",
        entity_type="person",
        properties={"role": "Engineer"},
    )
    saved = await store.add_node("user_alpha", node)
    assert saved.name == "Alice"

    # Case-insensitive fetch
    fetched = await store.get_node("user_alpha", "alice")
    assert fetched is not None
    assert fetched.name == "Alice"
    assert fetched.properties.get("role") == "Engineer"

    # Non-existent node
    assert await store.get_node("user_alpha", "Bob") is None

    # Mismatched user_id raises error
    with pytest.raises(ValueError, match="does not match"):
        await store.add_node("user_beta", node)

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_edge_crud_and_neighbors() -> None:
    store = InMemoryGraphStore()

    edge1 = RelationEdge(
        user_id="user_alpha",
        source="Alice",
        target="Project Apollo",
        relation="WORKS_ON",
        weight=0.8,
    )
    await store.add_edge("user_alpha", edge1)

    # Missing nodes are automatically created
    alice_node = await store.get_node("user_alpha", "Alice")
    apollo_node = await store.get_node("user_alpha", "Project Apollo")
    assert alice_node is not None
    assert apollo_node is not None

    # Duplicate edge updates properties/weight
    edge1_update = RelationEdge(
        user_id="user_alpha",
        source="Alice",
        target="Project Apollo",
        relation="WORKS_ON",
        weight=1.0,
        properties={"active": True},
    )
    await store.add_edge("user_alpha", edge1_update)

    out_neighbors = await store.get_neighbors("user_alpha", "Alice", direction="out")
    assert len(out_neighbors) == 1
    assert out_neighbors[0].weight == 1.0
    assert out_neighbors[0].properties.get("active") is True

    in_neighbors = await store.get_neighbors("user_alpha", "Project Apollo", direction="in")
    assert len(in_neighbors) == 1
    assert in_neighbors[0].source == "Alice"

    both_neighbors = await store.get_neighbors("user_alpha", "Project Apollo", direction="both")
    assert len(both_neighbors) == 1

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_find_paths_multihop() -> None:
    store = InMemoryGraphStore()
    user = "user_hop"

    # Build chain: Alice -> Apollo -> AcmeCorp -> Bangalore
    # Plus branching: Apollo -> Bob
    edges = [
        RelationEdge(user_id=user, source="Alice", target="Project Apollo", relation="WORKS_ON"),
        RelationEdge(user_id=user, source="Project Apollo", target="Bob", relation="MANAGED_BY"),
        RelationEdge(
            user_id=user,
            source="Project Apollo",
            target="AcmeCorp",
            relation="SPONSORED_BY",
        ),
        RelationEdge(user_id=user, source="AcmeCorp", target="Bangalore", relation="LOCATED_IN"),
    ]
    for edge in edges:
        await store.add_edge(user, edge)

    # 1-hop path
    paths_1hop = await store.find_paths(user, "Alice", "Project Apollo", max_depth=1)
    assert len(paths_1hop) == 1
    assert len(paths_1hop[0]) == 1
    assert paths_1hop[0][0].relation == "WORKS_ON"

    # 2-hop path: Alice -> Apollo -> Bob
    paths_2hop = await store.find_paths(user, "Alice", "Bob", max_depth=2)
    assert len(paths_2hop) == 1
    assert len(paths_2hop[0]) == 2
    assert [e.relation for e in paths_2hop[0]] == ["WORKS_ON", "MANAGED_BY"]

    # 3-hop path: Alice -> Apollo -> AcmeCorp -> Bangalore
    paths_3hop = await store.find_paths(user, "Alice", "Bangalore", max_depth=3)
    assert len(paths_3hop) == 1
    assert len(paths_3hop[0]) == 3
    assert [e.relation for e in paths_3hop[0]] == [
        "WORKS_ON",
        "SPONSORED_BY",
        "LOCATED_IN",
    ]

    # Depth cutoff: Alice -> Bangalore with max_depth=2 should find nothing
    paths_cutoff = await store.find_paths(user, "Alice", "Bangalore", max_depth=2)
    assert paths_cutoff == []

    # Non-existent path
    assert await store.find_paths(user, "Alice", "UnknownEntity") == []

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_cycle_handling() -> None:
    store = InMemoryGraphStore()
    user = "user_cycle"

    # Create cycle: A -> B -> C -> A
    await store.add_edge(user, RelationEdge(user_id=user, source="A", target="B", relation="TO_B"))
    await store.add_edge(user, RelationEdge(user_id=user, source="B", target="C", relation="TO_C"))
    await store.add_edge(user, RelationEdge(user_id=user, source="C", target="A", relation="TO_A"))

    # Finding path from A to C within cycle
    paths = await store.find_paths(user, "A", "C", max_depth=4)
    assert len(paths) == 1
    assert [e.relation for e in paths[0]] == ["TO_B", "TO_C"]

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_subgraph() -> None:
    store = InMemoryGraphStore()
    user = "user_subgraph"

    await store.add_edge(
        user,
        RelationEdge(user_id=user, source="Alice", target="Project Apollo", relation="WORKS_ON"),
    )
    await store.add_edge(
        user,
        RelationEdge(user_id=user, source="Project Apollo", target="Bob", relation="MANAGED_BY"),
    )
    await store.add_edge(
        user,
        RelationEdge(
            user_id=user,
            source="Charlie",
            target="Project Gemini",
            relation="LEADS",
        ),
    )

    # Subgraph seeded with Alice and 2 hops
    nodes, edges = await store.get_subgraph(user, ["Alice"], max_depth=2)
    node_names = {n.name for n in nodes}
    assert "Alice" in node_names
    assert "Project Apollo" in node_names
    assert "Bob" in node_names
    # Disconnected Charlie/Gemini must not be included
    assert "Charlie" not in node_names
    assert len(edges) == 2

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_tenant_isolation_and_delete() -> None:
    store = InMemoryGraphStore()

    await store.add_edge(
        "tenant_A",
        RelationEdge(user_id="tenant_A", source="Alice", target="SecretProject", relation="OWNS"),
    )
    await store.add_edge(
        "tenant_B",
        RelationEdge(user_id="tenant_B", source="Bob", target="PublicProject", relation="OWNS"),
    )

    # Tenant B cannot find Tenant A's paths or nodes
    assert await store.get_node("tenant_B", "Alice") is None
    assert await store.find_paths("tenant_B", "Alice", "SecretProject") == []
    assert await store.get_neighbors("tenant_B", "Alice") == []

    # Purging Tenant A does not affect Tenant B
    deleted = await store.delete_user_graph("tenant_A")
    assert deleted > 0
    assert await store.get_node("tenant_A", "Alice") is None

    # Tenant B data is intact
    assert await store.get_node("tenant_B", "Bob") is not None

    await store.close()


# -----------------------------------------------------------------------------
# NetworkXGraphStore Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_networkx_graph_store_end_to_end() -> None:
    store = NetworkXGraphStore()
    user = "nx_user"

    node = EntityNode(user_id=user, name="Alice", entity_type="person")
    await store.add_node(user, node)

    # Add edges
    await store.add_edge(
        user,
        RelationEdge(user_id=user, source="Alice", target="Project Apollo", relation="WORKS_ON"),
    )
    await store.add_edge(
        user,
        RelationEdge(user_id=user, source="Project Apollo", target="Bob", relation="MANAGED_BY"),
    )

    # Node fetch
    fetched = await store.get_node(user, "alice")
    assert fetched is not None
    assert fetched.name == "Alice"

    # Multi-hop find_paths
    paths = await store.find_paths(user, "Alice", "Bob", max_depth=3)
    assert len(paths) >= 1
    assert [e.relation for e in paths[0]] == ["WORKS_ON", "MANAGED_BY"]

    # Neighbors
    neighbors = await store.get_neighbors(user, "Project Apollo", direction="both")
    assert len(neighbors) == 2

    # Centrality calculation
    centrality = store.get_centrality(user)
    assert "project apollo" in centrality or "Project Apollo" in centrality
    assert centrality[list(centrality.keys())[0]] >= 0.0

    # Subgraph
    nodes, edges = await store.get_subgraph(user, ["Alice"], max_depth=2)
    assert len(nodes) >= 2
    assert len(edges) >= 1

    # Delete
    del_count = await store.delete_user_graph(user)
    assert del_count > 0
    assert await store.get_node(user, "Alice") is None

    await store.close()


def test_networkx_import_error_fallback() -> None:
    with (
        patch("memlayer.store.graph_networkx.HAS_NETWORKX", False),
        patch("memlayer.store.graph_networkx.nx", None),
    ):
        with pytest.raises(ImportError, match="NetworkX is not installed"):
            NetworkXGraphStore()


# -----------------------------------------------------------------------------
# Prompt Injection Formatter Tests
# -----------------------------------------------------------------------------


def test_format_subgraph_context() -> None:
    edges = [
        RelationEdge(
            user_id="u",
            source="Alice",
            target="Project Apollo",
            relation="WORKS_ON",
        ),
        RelationEdge(
            user_id="u",
            source="Project Apollo",
            target="Bob",
            relation="MANAGED_BY",
        ),
    ]
    nodes = [
        EntityNode(user_id="u", name="Alice"),
        EntityNode(user_id="u", name="Project Apollo"),
        EntityNode(user_id="u", name="Bob"),
    ]

    context, tokens = format_subgraph_context(nodes, edges)
    assert "[Knowledge Graph Relationships]" in context
    assert "- Alice --(WORKS_ON)--> Project Apollo" in context
    assert "- Project Apollo --(MANAGED_BY)--> Bob" in context
    assert tokens > 0

    # Empty edges returns empty string
    empty_ctx, empty_tokens = format_subgraph_context([], [])
    assert empty_ctx == ""
    assert empty_tokens == 0


# -----------------------------------------------------------------------------
# MemoryLayer Integration Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_layer_graph_integration() -> None:
    graph_store = InMemoryGraphStore()

    response_json = json.dumps(
        {
            "facts": [
                {
                    "content": "Alice works on Project Apollo",
                    "entity": "Alice",
                    "importance": 0.9,
                }
            ]
        }
    )

    async def mock_llm(**_kwargs: Any) -> str:
        return response_json

    extractor = FactExtractor(completion_fn=mock_llm)
    mem = await MemoryLayer.create_local(graph_store=graph_store, extractor=extractor)

    user_id = "user_kg_test"

    # Add atomic conversation turns
    await mem.add(
        user_id,
        [
            {
                "role": "user",
                "content": "Alice works on Project Apollo under Bob's leadership.",
            }
        ],
    )

    # Add structured relational triplets to the knowledge graph
    await mem.add_triplet(user_id, "Alice", "WORKS_ON", "Project Apollo")
    await mem.add_triplet(user_id, "Project Apollo", "MANAGED_BY", "Bob")

    # Verify get_graph_context
    graph_ctx, tokens = await mem.get_graph_context(user_id, ["Alice"])
    assert "[Knowledge Graph Relationships]" in graph_ctx
    assert "Alice --(WORKS_ON)--> Project Apollo" in graph_ctx
    assert tokens > 0

    # Search with graph context inclusion
    res = await mem.search(user_id, "What is Alice working on?", bypass_router=True)
    assert "[Knowledge Graph Relationships]" in res.context
    assert "WORKS_ON" in res.context

    # forget_all cleans up knowledge graph
    await mem.forget_all(user_id)
    assert await graph_store.get_node(user_id, "Alice") is None

    await mem.close()

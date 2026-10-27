"""Unit and integration tests for FastAPI interactive demo service."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

try:
    from starlette.testclient import TestClient

    from demo.app.main import app
except ImportError:
    pytest.skip("fastapi and demo dependencies not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Fixture providing TestClient with FastAPI lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verify health check returns status ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_serve_static_index(client: TestClient) -> None:
    """Verify root route serves HTML demo page."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Agent Memory Layer" in resp.text
    assert "Live Memory Inspector" in resp.text


def test_chat_memory_augmented_and_stateless(client: TestClient) -> None:
    """Verify chat interaction with memory enabled vs stateless baseline."""
    user_id = f"test_toggle_{uuid.uuid4().hex[:8]}"

    # Step 1: Memory ON - user introduces location
    resp_on = client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": "session_1",
            "message": "I live in Seattle and work as a cloud engineer.",
            "memory_enabled": True,
        },
    )
    assert resp_on.status_code == 200
    data_on = resp_on.json()
    assert data_on["memory_enabled"] is True
    assert data_on["prompt_tokens"] > 0
    assert data_on["latency_ms"] >= 0

    # Step 2: Memory ON - session 2 queries location
    resp_query = client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": "session_2",
            "message": "Where do I currently live?",
            "memory_enabled": True,
        },
    )
    assert resp_query.status_code == 200
    data_query = resp_query.json()
    assert "Seattle" in data_query["reply"] or "Seattle" in data_query["context_used"]
    assert len(data_query["retrieved_memories"]) > 0

    # Step 3: Memory OFF - stateless turn
    resp_off = client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": "session_3",
            "message": "Where do I currently live?",
            "memory_enabled": False,
        },
    )
    assert resp_off.status_code == 200
    data_off = resp_off.json()
    assert data_off["memory_enabled"] is False
    assert len(data_off["retrieved_memories"]) == 0
    assert data_off["context_used"] == ""


def test_live_supersession_in_demo(client: TestClient) -> None:
    """Verify that updating a fact in a subsequent session supersedes older memory."""
    user_id = f"test_supersede_{uuid.uuid4().hex[:8]}"

    # Turn 1: Lives in Seattle
    client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": "session_1",
            "message": "I live in Seattle.",
            "memory_enabled": True,
        },
    )

    # Turn 2: Relocates to Austin
    client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": "session_2",
            "message": "I relocated to Austin, Texas.",
            "memory_enabled": True,
        },
    )

    # Fetch memories for inspector
    resp_mem = client.get(f"/api/memories?user_id={user_id}")
    assert resp_mem.status_code == 200
    memories = resp_mem.json()

    active = [m for m in memories if m["status"] == "active"]
    superseded = [m for m in memories if m["status"] == "superseded"]

    assert len(active) >= 1
    assert any("Austin" in m["content"] for m in active)

    # Verify that the older Seattle memory transitioned to superseded
    assert len(superseded) >= 1
    assert any("Seattle" in m["content"] for m in superseded)
    seattle_mem = next(m for m in superseded if "Seattle" in m["content"])
    assert seattle_mem["superseded_by"] is not None


def test_forget_and_reset_endpoints(client: TestClient) -> None:
    """Verify individual memory forget and full store reset."""
    user_id = f"test_forget_{uuid.uuid4().hex[:8]}"

    client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": "session_1",
            "message": "I prefer dark mode UI.",
            "memory_enabled": True,
        },
    )

    mems_before = client.get(f"/api/memories?user_id={user_id}").json()
    assert len(mems_before) >= 1
    target_id = mems_before[0]["id"]

    # Delete specific memory
    resp_del = client.delete(f"/api/memories/{target_id}?user_id={user_id}")
    assert resp_del.status_code == 200
    assert resp_del.json()["deleted"] is True

    # Confirm deleted
    mems_after = client.get(f"/api/memories?user_id={user_id}").json()
    assert not any(m["id"] == target_id for m in mems_after)

    # Reset entire store for user
    resp_reset = client.post(f"/api/reset?user_id={user_id}")
    assert resp_reset.status_code == 200
    assert resp_reset.json()["success"] is True

    mems_final = client.get(f"/api/memories?user_id={user_id}").json()
    assert len(mems_final) == 0


def test_strict_user_isolation_in_demo(client: TestClient) -> None:
    """Verify that user_a's memories are never visible to user_b."""
    user_a = f"user_alpha_{uuid.uuid4().hex[:8]}"
    user_b = f"user_beta_{uuid.uuid4().hex[:8]}"

    client.post(
        "/api/chat",
        json={
            "user_id": user_a,
            "session_id": "session_1",
            "message": "My adopted dog is named Rusty.",
            "memory_enabled": True,
        },
    )

    # User B inspector should be completely empty
    resp_b = client.get(f"/api/memories?user_id={user_b}")
    assert resp_b.status_code == 200
    assert len(resp_b.json()) == 0

    # User B chat querying Rusty should return no recollection
    resp_b_chat = client.post(
        "/api/chat",
        json={
            "user_id": user_b,
            "session_id": "session_1",
            "message": "What is my pet's name?",
            "memory_enabled": True,
        },
    )
    assert resp_b_chat.status_code == 200
    assert "Rusty" not in resp_b_chat.json()["context_used"]


def test_graph_endpoints(client: TestClient) -> None:
    """Verify graph inspection, triplet ingestion, pathfinding, and deletion endpoints."""
    user_id = f"test_graph_{uuid.uuid4().hex[:8]}"

    # Initially empty for new random user
    init_res = client.get(f"/api/graph?user_id={user_id}")
    assert init_res.status_code == 200
    assert init_res.json()["nodes"] == []
    assert init_res.json()["edges"] == []

    # Ingest triplet 1: Alice -> Apollo
    t1 = client.post(
        "/api/graph/triplets",
        json={
            "user_id": user_id,
            "source": "Alice",
            "relation": "WORKS_ON",
            "target": "Apollo",
            "weight": 0.9,
        },
    )
    assert t1.status_code == 200
    assert t1.json()["relation"] == "WORKS_ON"

    # Ingest triplet 2: Apollo -> Bob
    t2 = client.post(
        "/api/graph/triplets",
        json={
            "user_id": user_id,
            "source": "Apollo",
            "relation": "MANAGED_BY",
            "target": "Bob",
            "weight": 1.0,
        },
    )
    assert t2.status_code == 200

    # Verify graph returns nodes and edges
    graph_res = client.get(f"/api/graph?user_id={user_id}")
    assert graph_res.status_code == 200
    data = graph_res.json()
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2

    # Verify pathfinding: Alice -> Bob (2-hop path)
    paths_res = client.get(
        f"/api/graph/paths?user_id={user_id}&source=Alice&target=Bob&max_depth=3"
    )
    assert paths_res.status_code == 200
    paths = paths_res.json()
    assert len(paths) >= 1
    assert len(paths[0]["path"]) == 2
    assert [e["relation"] for e in paths[0]["path"]] == ["WORKS_ON", "MANAGED_BY"]

    # Delete user graph
    del_res = client.delete(f"/api/graph?user_id={user_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] >= 5

    # Confirm purged
    final_res = client.get(f"/api/graph?user_id={user_id}")
    assert len(final_res.json()["nodes"]) == 0
    assert len(final_res.json()["edges"]) == 0


def test_graph_tenant_isolation_in_demo(client: TestClient) -> None:
    """Verify knowledge graph tenant isolation in demo endpoints."""
    user_a = f"graph_iso_a_{uuid.uuid4().hex[:8]}"
    user_b = f"graph_iso_b_{uuid.uuid4().hex[:8]}"

    client.post(
        "/api/graph/triplets",
        json={
            "user_id": user_a,
            "source": "SecretOrg",
            "relation": "OWNS",
            "target": "SecretTech",
        },
    )

    # User B should see 0 nodes and 0 edges
    res_b = client.get(f"/api/graph?user_id={user_b}")
    assert res_b.status_code == 200
    assert len(res_b.json()["nodes"]) == 0
    assert len(res_b.json()["edges"]) == 0

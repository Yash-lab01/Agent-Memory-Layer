"""Unit and integration tests for WanderAI travel companion demo integration."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

try:
    from starlette.testclient import TestClient

    from demo.app.main import app
    from demo.wanderai import (
        SEED_TRAVEL_TRIPLETS,
        WANDERAI_SCENARIOS,
        TravelerProfile,
        WanderAIAgent,
    )
except ImportError:
    pytest.skip("fastapi and demo dependencies not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Fixture providing TestClient with FastAPI lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_wanderai_scenarios_contract() -> None:
    """Verify predefined scenarios structure and content."""
    assert len(WANDERAI_SCENARIOS) == 4
    for sc in WANDERAI_SCENARIOS:
        assert "turn" in sc
        assert "title" in sc
        assert "session_id" in sc
        assert "user_message" in sc
        assert "expected_learnings" in sc
        assert len(sc["user_message"]) > 10


def test_wanderai_seed_knowledge_triplets() -> None:
    """Verify destination knowledge topology triplets."""
    assert len(SEED_TRAVEL_TRIPLETS) >= 15
    destinations = {t[0] for t in SEED_TRAVEL_TRIPLETS}
    assert "Paris" in destinations
    assert "Tokyo" in destinations


def test_wanderai_api_scenarios_endpoint(client: TestClient) -> None:
    """Verify GET /api/wanderai/scenarios returns the showcase scenarios."""
    resp = client.get("/api/wanderai/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 4
    assert data[0]["session_id"] == "trip_tokyo_spring"
    assert data[2]["session_id"] == "trip_paris_autumn"


def test_wanderai_chat_flow_and_cross_trip_recall(client: TestClient) -> None:
    """Verify multi-session cross-trip memory persistence in WanderAI."""
    traveler_id = f"traveler_{uuid.uuid4().hex[:8]}"

    # Turn 1: Establish vegan & Star Alliance preferences in Tokyo trip session
    resp1 = client.post(
        "/api/wanderai/chat",
        json={
            "user_id": traveler_id,
            "session_id": "trip_tokyo_2025",
            "message": "I am strictly vegan and hold Star Alliance Gold status. Planning a Tokyo trip.",
            "memory_enabled": True,
        },
    )
    assert resp1.status_code == 200
    d1 = resp1.json()
    assert d1["memory_enabled"] is True
    assert "Tokyo" in d1["reply"] or "vegan" in d1["reply"].lower()

    # Turn 2: Switch to Paris session 6 months later. Verify cross-trip recall.
    resp2 = client.post(
        "/api/wanderai/chat",
        json={
            "user_id": traveler_id,
            "session_id": "trip_paris_2025",
            "message": "Help me plan 4 days in Paris. Where should I eat dinner?",
            "memory_enabled": True,
        },
    )
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert d2["memory_enabled"] is True
    # The active vegan preference from Tokyo trip should be retrieved for Paris trip
    assert len(d2["retrieved_memories"]) > 0
    vegan_retrieved = any("vegan" in m["content"].lower() for m in d2["retrieved_memories"])
    assert vegan_retrieved, "Active vegan preference from Tokyo trip must be recalled for Paris trip"


def test_wanderai_preference_update_supersession(client: TestClient) -> None:
    """Verify that updating airline loyalty marks previous loyalty superseded."""
    traveler_id = f"traveler_update_{uuid.uuid4().hex[:8]}"

    # Turn 1: Initial loyalty program
    resp1 = client.post(
        "/api/wanderai/chat",
        json={
            "user_id": traveler_id,
            "session_id": "trip_setup",
            "message": "I fly frequently and hold Star Alliance Gold status.",
            "memory_enabled": True,
        },
    )
    assert resp1.status_code == 200

    # Turn 2: Switch loyalty to OneWorld
    resp2 = client.post(
        "/api/wanderai/chat",
        json={
            "user_id": traveler_id,
            "session_id": "trip_setup",
            "message": "I switched loyalty and now hold OneWorld Sapphire status instead of Star Alliance.",
            "memory_enabled": True,
        },
    )
    assert resp2.status_code == 200

    # Check traveler profile
    prof_resp = client.get(f"/api/wanderai/profile?user_id={traveler_id}")
    assert prof_resp.status_code == 200
    prof = prof_resp.json()
    assert prof["user_id"] == traveler_id

    # Verify superseded history contains Star Alliance
    superseded_contents = [s["content"].lower() for s in prof["superseded_preferences"]]
    assert any("star alliance" in c for c in superseded_contents), "Star Alliance Gold must be superseded"


def test_wanderai_tenant_isolation(client: TestClient) -> None:
    """Verify strict tenant isolation between different travelers."""
    traveler_a = f"traveler_a_{uuid.uuid4().hex[:8]}"
    traveler_b = f"traveler_b_{uuid.uuid4().hex[:8]}"

    # Traveler A is vegan
    client.post(
        "/api/wanderai/chat",
        json={
            "user_id": traveler_a,
            "session_id": "trip_a",
            "message": "I am strictly vegan.",
            "memory_enabled": True,
        },
    )

    # Traveler B asks where to eat in Paris
    resp_b = client.post(
        "/api/wanderai/chat",
        json={
            "user_id": traveler_b,
            "session_id": "trip_b",
            "message": "Where should I eat in Paris?",
            "memory_enabled": True,
        },
    )
    assert resp_b.status_code == 200
    d_b = resp_b.json()

    # Traveler B must NOT receive Traveler A's vegan memories
    for m in d_b["retrieved_memories"]:
        assert traveler_a not in str(m)


def test_wanderai_seed_endpoint(client: TestClient) -> None:
    """Verify POST /api/wanderai/seed adds knowledge graph relations."""
    traveler_id = f"traveler_seed_{uuid.uuid4().hex[:8]}"
    resp = client.post(f"/api/wanderai/seed?user_id={traveler_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "seeded"
    assert data["relations_added"] >= 15
    assert data["user_id"] == traveler_id

"""Shared pytest fixtures for memlayer tests."""

import pytest


@pytest.fixture
def sample_user_id() -> str:
    """Standard test user ID."""
    return "test_user_001"


@pytest.fixture
def alternate_user_id() -> str:
    """Alternate user ID for multi-user isolation tests."""
    return "test_user_002"


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Sample dialogue messages for testing."""
    return [
        {"role": "user", "content": "I live in Pune and work on conversational AI agents."},
        {"role": "assistant", "content": "That sounds like exciting work in Pune!"},
    ]

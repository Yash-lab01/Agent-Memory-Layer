"""Smoke tests for package initialization."""

import memlayer


def test_package_version() -> None:
    """Verify package version is defined and non-empty."""
    assert hasattr(memlayer, "__version__")
    assert memlayer.__version__ == "0.1.0"

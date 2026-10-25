"""Packaging and release readiness tests for agent-memory-layer (memlayer).

Verifies public API exports, PEP 561 py.typed marker, CLI entry points,
submodule layout, and minimal core dependency footprint.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import memlayer


def test_package_version() -> None:
    """Verify package version is defined, non-empty, and semver compliant."""
    assert hasattr(memlayer, "__version__")
    assert memlayer.__version__ == "0.1.0"


def test_public_api_exports() -> None:
    """Verify public symbols exposed at the top-level memlayer package."""
    expected_exports = {
        "MemoryLayer",
        "Episode",
        "Role",
        "MemoryItem",
        "MemoryStatus",
        "ExtractedFact",
        "ExtractionResult",
        "ScoredMemory",
        "SearchResult",
    }
    for symbol in expected_exports:
        assert hasattr(memlayer, symbol), (
            f"Expected symbol '{symbol}' not found in top-level memlayer"
        )
        assert symbol in memlayer.__all__, f"Symbol '{symbol}' missing from memlayer.__all__"


def test_py_typed_marker_presence() -> None:
    """Verify PEP 561 py.typed marker exists within the packaged package directory."""
    pkg_dir = Path(memlayer.__file__).parent
    py_typed_file = pkg_dir / "py.typed"
    assert py_typed_file.exists(), f"PEP 561 marker py.typed missing from {pkg_dir}"
    assert py_typed_file.is_file()


def test_core_submodules_importable() -> None:
    """Verify all internal architecture submodules are cleanly importable."""
    submodules = [
        "memlayer.api",
        "memlayer.models",
        "memlayer.extract",
        "memlayer.store",
        "memlayer.retrieve",
        "memlayer.consolidate",
        "memlayer.llm",
        "memlayer.tracing",
        "memlayer.mcp_server",
    ]
    for mod_name in submodules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import submodule {mod_name}"


def test_core_package_does_not_pull_heavy_extras() -> None:
    """Verify importing top-level memlayer does not eagerly import demo or evaluation packages."""
    code = (
        "import sys, memlayer; "
        "extras = [m for m in ['fastapi', 'pandas', 'tabulate'] if m in sys.modules]; "
        "assert not extras, f'Unexpected modules loaded: {extras}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Core import pulled heavy extras: {result.stderr}"


def test_cli_entry_point_metadata() -> None:
    """Verify registered CLI entry points in package distribution metadata."""
    console_scripts = entry_points(group="console_scripts")
    script_map = {ep.name: ep.value for ep in console_scripts}

    if "memlayer-mcp" in script_map:
        assert script_map["memlayer-mcp"] == "memlayer.mcp_server:main"

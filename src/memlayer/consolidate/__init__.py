"""Consolidation module for deduplication, conflict resolution, and memory decay."""

from memlayer.consolidate.conflicts import (
    ConflictDetectionResult,
    ConflictDetector,
    ConflictItem,
    ConflictResolutionResult,
)

__all__ = [
    "ConflictDetectionResult",
    "ConflictDetector",
    "ConflictItem",
    "ConflictResolutionResult",
]

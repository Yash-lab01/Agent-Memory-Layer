"""Consolidation module for deduplication, conflict resolution, and memory decay."""

from memlayer.consolidate.conflicts import (
    ConflictDetectionResult,
    ConflictDetector,
    ConflictItem,
    ConflictResolutionResult,
)
from memlayer.consolidate.dedup import (
    DeduplicationEngine,
    DedupResult,
    DedupVerificationResult,
    DuplicatePair,
)

__all__ = [
    "ConflictDetectionResult",
    "ConflictDetector",
    "ConflictItem",
    "ConflictResolutionResult",
    "DedupResult",
    "DedupVerificationResult",
    "DeduplicationEngine",
    "DuplicatePair",
]

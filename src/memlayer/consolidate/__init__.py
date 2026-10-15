"""Consolidation module for deduplication, conflict resolution, and memory decay."""

from memlayer.consolidate.conflicts import (
    ConflictDetectionResult,
    ConflictDetector,
    ConflictItem,
    ConflictResolutionResult,
)
from memlayer.consolidate.decay import (
    DecayItemResult,
    DecayResult,
    MemoryDecayEngine,
    calculate_decayed_importance,
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
    "DecayItemResult",
    "DecayResult",
    "DedupResult",
    "DedupVerificationResult",
    "DeduplicationEngine",
    "DuplicatePair",
    "MemoryDecayEngine",
    "calculate_decayed_importance",
]

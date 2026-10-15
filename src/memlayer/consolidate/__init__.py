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
from memlayer.consolidate.worker import (
    ConsolidationWorker,
    FullConsolidationResult,
)

__all__ = [
    "ConflictDetectionResult",
    "ConflictDetector",
    "ConflictItem",
    "ConflictResolutionResult",
    "ConsolidationWorker",
    "DecayItemResult",
    "DecayResult",
    "DedupResult",
    "DedupVerificationResult",
    "DeduplicationEngine",
    "DuplicatePair",
    "FullConsolidationResult",
    "MemoryDecayEngine",
    "calculate_decayed_importance",
]

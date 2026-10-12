"""Agent Memory Layer (memlayer)

A modular, durable long-term memory library for conversational AI agents.
"""

from memlayer.models import (
    Episode,
    ExtractedFact,
    ExtractionResult,
    MemoryItem,
    MemoryStatus,
    Role,
    ScoredMemory,
    SearchResult,
)

__all__ = [
    "Episode",
    "ExtractedFact",
    "ExtractionResult",
    "MemoryItem",
    "MemoryStatus",
    "Role",
    "ScoredMemory",
    "SearchResult",
]

__version__ = "0.1.0"

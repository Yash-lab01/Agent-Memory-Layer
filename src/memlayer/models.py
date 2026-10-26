"""Core Pydantic v2 data models and contracts for memlayer."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryStatus(StrEnum):
    """Lifecycle status of a memory record."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"


class Role(StrEnum):
    """Speaker role in a dialogue episode."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(UTC)


def generate_uuid() -> str:
    """Generate a standard UUID4 hex string."""
    return str(uuid4())


class Episode(BaseModel):
    """Raw dialogue turn stored for provenance and historical auditing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=generate_uuid, description="Unique episode ID")
    user_id: str = Field(..., min_length=1, description="Owner identifier (for isolation)")
    session_id: str = Field(default="default", description="Conversation session ID")
    role: Role = Field(default=Role.USER, description="Speaker role")
    text: str = Field(..., min_length=1, description="Raw message text")
    ts: datetime = Field(default_factory=utc_now, description="Timestamp message occurred")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id cannot be blank or whitespace")
        return v.strip()


class ExtractedFact(BaseModel):
    """An individual atomic fact extracted from dialogue."""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(..., min_length=1, description="Self-contained atomic statement")
    entity: str | None = Field(default=None, description="Entity or subject of the fact")
    event_time: datetime | None = Field(
        default=None, description="When the event occurred if specified in dialogue"
    )
    importance: Annotated[float, Field(ge=0.0, le=1.0, default=0.5)] = 0.5
    confidence: Annotated[float, Field(ge=0.0, le=1.0, default=1.0)] = 1.0


class ExtractionResult(BaseModel):
    """Structured extraction output from LLM."""

    model_config = ConfigDict(extra="ignore")

    facts: list[ExtractedFact] = Field(default_factory=list, description="Extracted atomic facts")


class MemoryItem(BaseModel):
    """A durable long-term memory record with metadata and provenance."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_uuid, description="Unique memory ID")
    user_id: str = Field(..., min_length=1, description="Owner identifier (for isolation)")
    content: str = Field(..., min_length=1, description="Atomic fact text")
    entity: str | None = Field(default=None, description="Entity or subject of the fact")
    event_time: datetime | None = Field(
        default=None, description="When the fact event occurred, if known"
    )
    created_at: datetime = Field(
        default_factory=utc_now, description="When memory was extracted/stored"
    )
    importance: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Importance score [0.0, 1.0]"
    )
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0, description="Extraction confidence score [0.0, 1.0]"
    )
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE, description="Active or superseded")
    superseded_by: str | None = Field(
        default=None, description="ID of memory that superseded this one"
    )
    source_episode_id: str | None = Field(
        default=None, description="Source episode ID providing provenance"
    )
    last_accessed_at: datetime | None = Field(
        default=None, description="Timestamp of most recent retrieval"
    )
    access_count: int = Field(default=0, ge=0, description="Number of times retrieved")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id cannot be blank or whitespace")
        return v.strip()

    def mark_superseded(self, newer_memory_id: str) -> None:
        """Mark memory as superseded by a newer memory without hard deleting."""
        self.status = MemoryStatus.SUPERSEDED
        self.superseded_by = newer_memory_id

    def record_access(self, access_time: datetime | None = None) -> None:
        """Record a retrieval hit for decay and importance weighting."""
        self.access_count += 1
        self.last_accessed_at = access_time or utc_now()


class ScoredMemory(BaseModel):
    """Memory item annotated with retrieval scores."""

    model_config = ConfigDict(frozen=True)

    memory: MemoryItem
    score: float = Field(..., description="Combined composite score")
    similarity: float = Field(default=0.0, description="Vector similarity component")
    recency_score: float = Field(default=0.0, description="Recency decay component")
    importance_score: float = Field(default=0.0, description="Importance component")


class SearchResult(BaseModel):
    """Public search output containing injected context and metadata."""

    model_config = ConfigDict(frozen=True)

    context: str = Field(..., description="Token-budgeted formatted memory string")
    memories: list[ScoredMemory] = Field(
        default_factory=list, description="List of scored memories included"
    )
    token_count: int = Field(default=0, ge=0, description="Estimated token count of context")
    skipped: bool = Field(default=False, description="True if query router skipped memory lookup")


class EntityNode(BaseModel):
    """An entity vertex in the knowledge graph."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_uuid, description="Unique node ID")
    user_id: str = Field(..., min_length=1, description="Owner identifier (for isolation)")
    name: str = Field(..., min_length=1, description="Canonical entity name")
    entity_type: str = Field(default="concept", description="Entity category/type")
    properties: dict[str, Any] = Field(default_factory=dict, description="Metadata key-value pairs")
    created_at: datetime = Field(default_factory=utc_now, description="Node creation timestamp")

    @field_validator("user_id", "name")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be blank or whitespace")
        return v.strip()


class RelationEdge(BaseModel):
    """A directed relationship edge in the knowledge graph."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_uuid, description="Unique edge ID")
    user_id: str = Field(..., min_length=1, description="Owner identifier (for isolation)")
    source: str = Field(..., min_length=1, description="Source entity name")
    target: str = Field(..., min_length=1, description="Target entity name")
    relation: str = Field(..., min_length=1, description="Relationship predicate")
    weight: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0, description="Edge weight/confidence"
    )
    source_memory_id: str | None = Field(default=None, description="Optional provenance memory ID")
    properties: dict[str, Any] = Field(default_factory=dict, description="Metadata key-value pairs")
    created_at: datetime = Field(default_factory=utc_now, description="Edge creation timestamp")

    @field_validator("user_id", "source", "target", "relation")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be blank or whitespace")
        return v.strip()


class GraphTriplet(BaseModel):
    """A semantic relationship triplet extracted from dialogue."""

    model_config = ConfigDict(extra="ignore")

    source: str = Field(..., min_length=1, description="Subject entity name")
    relation: str = Field(..., min_length=1, description="Predicate/relationship")
    target: str = Field(..., min_length=1, description="Object entity name")
    weight: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=1.0, description="Confidence")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")

    @field_validator("source", "relation", "target")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be blank or whitespace")
        return v.strip()

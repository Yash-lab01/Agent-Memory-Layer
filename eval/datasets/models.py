"""Data models for benchmark evaluation datasets and splits."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvalTurn(BaseModel):
    """Represents a single conversational turn in the background history."""

    role: str
    content: str
    timestamp: datetime | None = None
    session_id: str | None = None


class EvalQuestion(BaseModel):
    """Represents an evaluation item in a memory benchmark dataset."""

    id: str
    benchmark: str  # e.g., "longmemeval", "locomo"
    category: str  # "knowledge_update", "temporal_reasoning", "multi_session", "abstention"
    user_id: str
    conversation: list[EvalTurn] = Field(default_factory=list)
    question: str
    ground_truth: str
    split: str = "dev"  # "dev", "test"
    metadata: dict[str, Any] = Field(default_factory=dict)

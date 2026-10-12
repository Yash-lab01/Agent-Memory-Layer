"""Unit tests for Pydantic v2 data models in memlayer.models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

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


class TestEpisodeModel:
    """Tests for Episode model."""

    def test_create_valid_episode(self, sample_user_id: str) -> None:
        ep = Episode(user_id=sample_user_id, text="Hello world")
        assert ep.user_id == sample_user_id
        assert ep.text == "Hello world"
        assert ep.role == Role.USER
        assert ep.session_id == "default"
        assert isinstance(ep.id, str)
        assert len(ep.id) > 0
        assert isinstance(ep.ts, datetime)

    def test_episode_empty_user_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Episode(user_id="", text="Hello")

    def test_episode_whitespace_user_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Episode(user_id="   ", text="Hello")

    def test_episode_empty_text_rejected(self, sample_user_id: str) -> None:
        with pytest.raises(ValidationError):
            Episode(user_id=sample_user_id, text="")

    def test_episode_is_frozen(self, sample_user_id: str) -> None:
        ep = Episode(user_id=sample_user_id, text="Hello")
        with pytest.raises(ValidationError):
            ep.text = "Modified"


class TestMemoryItemModel:
    """Tests for MemoryItem model."""

    def test_create_valid_memory_item(self, sample_user_id: str) -> None:
        mem = MemoryItem(
            user_id=sample_user_id,
            content="User lives in Pune",
            entity="User",
            importance=0.8,
            confidence=0.95,
        )
        assert mem.user_id == sample_user_id
        assert mem.content == "User lives in Pune"
        assert mem.entity == "User"
        assert mem.status == MemoryStatus.ACTIVE
        assert mem.superseded_by is None
        assert mem.access_count == 0
        assert mem.last_accessed_at is None

    def test_importance_bounds_validation(self, sample_user_id: str) -> None:
        with pytest.raises(ValidationError):
            MemoryItem(user_id=sample_user_id, content="Fact", importance=1.5)

        with pytest.raises(ValidationError):
            MemoryItem(user_id=sample_user_id, content="Fact", importance=-0.1)

    def test_confidence_bounds_validation(self, sample_user_id: str) -> None:
        with pytest.raises(ValidationError):
            MemoryItem(user_id=sample_user_id, content="Fact", confidence=1.2)

        with pytest.raises(ValidationError):
            MemoryItem(user_id=sample_user_id, content="Fact", confidence=-0.5)

    def test_mark_superseded(self, sample_user_id: str) -> None:
        old_mem = MemoryItem(user_id=sample_user_id, content="User lives in Pune")
        new_mem_id = "mem_bangalore_456"

        assert old_mem.status == MemoryStatus.ACTIVE
        assert old_mem.superseded_by is None

        old_mem.mark_superseded(new_mem_id)

        current_status: MemoryStatus = old_mem.status
        assert current_status == MemoryStatus.SUPERSEDED
        assert old_mem.superseded_by == new_mem_id

    def test_record_access(self, sample_user_id: str) -> None:
        mem = MemoryItem(user_id=sample_user_id, content="User prefers Python")
        assert mem.access_count == 0
        assert mem.last_accessed_at is None

        access_ts = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)
        mem.record_access(access_ts)

        assert mem.access_count == 1
        assert mem.last_accessed_at == access_ts

        mem.record_access()
        assert mem.access_count == 2
        assert mem.last_accessed_at is not None

    def test_empty_user_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MemoryItem(user_id="", content="Fact")


class TestExtractionModels:
    """Tests for ExtractedFact and ExtractionResult models."""

    def test_extracted_fact_defaults(self) -> None:
        fact = ExtractedFact(content="User works on LLM memory")
        assert fact.content == "User works on LLM memory"
        assert fact.importance == 0.5
        assert fact.confidence == 1.0
        assert fact.entity is None
        assert fact.event_time is None

    def test_extraction_result_json_roundtrip(self) -> None:
        json_data = {
            "facts": [
                {
                    "content": "User lives in Pune",
                    "entity": "User",
                    "importance": 0.9,
                    "confidence": 0.95,
                    "extra_unneeded_field": "ignored",
                },
                {
                    "content": "User prefers remote work",
                    "entity": "User",
                    "importance": 0.7,
                },
            ]
        }
        result = ExtractionResult.model_validate(json_data)
        assert len(result.facts) == 2
        assert result.facts[0].content == "User lives in Pune"
        assert result.facts[0].importance == 0.9
        assert result.facts[1].content == "User prefers remote work"
        assert result.facts[1].confidence == 1.0


class TestSearchModels:
    """Tests for ScoredMemory and SearchResult models."""

    def test_scored_memory_and_search_result(self, sample_user_id: str) -> None:
        mem = MemoryItem(user_id=sample_user_id, content="Fact A")
        scored = ScoredMemory(
            memory=mem,
            score=0.88,
            similarity=0.92,
            recency_score=0.85,
            importance_score=0.8,
        )
        assert scored.score == 0.88

        result = SearchResult(
            context="[Memories]\n- Fact A",
            memories=[scored],
            token_count=15,
            skipped=False,
        )
        assert result.token_count == 15
        assert len(result.memories) == 1
        assert result.skipped is False

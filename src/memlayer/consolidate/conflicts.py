"""Contradiction and supersession resolution engine for memory lifecycle management."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import litellm
from pydantic import BaseModel, Field, ValidationError

from memlayer.consolidate.prompts import CONFLICT_DETECTION_SYSTEM_PROMPT
from memlayer.extract.extractor import clean_json_output
from memlayer.llm.embedder import BaseEmbedder
from memlayer.models import MemoryItem, MemoryStatus
from memlayer.store.sql import SQLStore
from memlayer.store.vector_base import VectorPoint, VectorStoreBase

logger = logging.getLogger(__name__)

CompletionFn = Callable[..., Awaitable[Any]]


class ConflictItem(BaseModel):
    """Result of analyzing an individual candidate memory for contradiction."""

    old_memory_id: str
    is_contradiction: bool
    reason: str = ""


class ConflictDetectionResult(BaseModel):
    """Collection of detected conflict evaluations."""

    conflicts: list[ConflictItem] = Field(default_factory=list)


@dataclass(frozen=True)
class ConflictResolutionResult:
    """Outcome of resolving conflicts for a newly added memory."""

    new_memory: MemoryItem
    superseded_memory_ids: list[str]


class ConflictDetector:
    """Identifies and resolves contradictory facts, ensuring non-destructive supersession."""

    def __init__(
        self,
        store: SQLStore,
        vector_store: VectorStoreBase,
        embedder: BaseEmbedder,
        model: str = "gemini/gemini-1.5-flash",
        completion_fn: CompletionFn | None = None,
    ) -> None:
        """Initialize the conflict detector with storage and model backends."""
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder
        self.model = model
        self.completion_fn = completion_fn or litellm.acompletion

    async def detect_conflicts(
        self,
        new_memory: MemoryItem,
        candidate_memories: list[MemoryItem],
    ) -> list[ConflictItem]:
        """Classify whether new_memory contradicts any candidate_memories using LLM NLI."""
        if not candidate_memories:
            return []

        candidates_block = "\n".join(
            f"- ID: {m.id} | Entity: {m.entity} | Content: {m.content}" for m in candidate_memories
        )

        user_prompt = (
            f"NEW FACT TO EVALUATE:\n"
            f"- Entity: {new_memory.entity}\n"
            f"- Content: {new_memory.content}\n\n"
            f"PRIOR ACTIVE MEMORIES:\n"
            f"{candidates_block}\n\n"
            f"Analyze each prior memory above. Return a JSON object matching the schema."
        )

        messages = [
            {"role": "system", "content": CONFLICT_DETECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.completion_fn(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            raw_text = (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else str(response)
            )

            cleaned_json = clean_json_output(raw_text)
            parsed = ConflictDetectionResult.model_validate_json(cleaned_json)
            return parsed.conflicts
        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            logger.warning("Conflict detection failed or returned invalid JSON: %s", exc)
            return []

    async def resolve_and_supersede(
        self,
        user_id: str,
        new_memory: MemoryItem,
        similarity_threshold: float = 0.65,
    ) -> ConflictResolutionResult:
        """Find candidate memories for user_id, detect conflicts, and mark superseded ones.

        Never hard-deletes memories. Outdated facts are marked MemoryStatus.SUPERSEDED
        with superseded_by pointing to new_memory.id.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        clean_user_id = user_id.strip()

        # Step 1: Candidate retrieval (Active memories only)
        # Strategy A: Same entity active memories
        candidate_map: dict[str, MemoryItem] = {}
        if new_memory.entity and new_memory.entity.lower() not in {"user", "none"}:
            entity_memories = await self.store.list_memories(
                user_id=clean_user_id,
                status=MemoryStatus.ACTIVE,
                entity=new_memory.entity,
                limit=50,
            )
            for m in entity_memories:
                if m.id != new_memory.id:
                    candidate_map[m.id] = m

        # Strategy B: Semantic similarity lookup
        query_vector = await self.embedder.embed_text(new_memory.content)
        vector_hits = await self.vector_store.search(
            user_id=clean_user_id,
            query_vector=query_vector,
            limit=10,
            status="active",
        )
        for hit in vector_hits:
            if hit.id != new_memory.id and hit.score >= similarity_threshold:
                if hit.id not in candidate_map:
                    mem = await self.store.get_memory(clean_user_id, hit.id)
                    if mem and mem.status == MemoryStatus.ACTIVE:
                        candidate_map[mem.id] = mem

        candidates = list(candidate_map.values())
        if not candidates:
            return ConflictResolutionResult(new_memory=new_memory, superseded_memory_ids=[])

        # Step 2: Detect conflicts via NLI
        conflicts = await self.detect_conflicts(
            new_memory=new_memory,
            candidate_memories=candidates,
        )

        # Step 3: Mark confirmed contradictions as superseded
        superseded_ids: list[str] = []
        for conflict in conflicts:
            if conflict.is_contradiction and conflict.old_memory_id in candidate_map:
                old_id = conflict.old_memory_id
                # 3a. Update SQL status to SUPERSEDED and link superseded_by
                success = await self.store.mark_superseded(
                    user_id=clean_user_id,
                    old_memory_id=old_id,
                    new_memory_id=new_memory.id,
                )
                if success:
                    superseded_ids.append(old_id)
                    # 3b. Update Vector point payload status to superseded
                    old_mem = candidate_map[old_id]
                    old_vec = await self.embedder.embed_text(old_mem.content)
                    payload = {
                        "content": old_mem.content,
                        "entity": old_mem.entity,
                        "importance": old_mem.importance,
                        "created_at": old_mem.created_at.isoformat(),
                        "event_time": (
                            old_mem.event_time.isoformat() if old_mem.event_time else None
                        ),
                        "status": MemoryStatus.SUPERSEDED.value,
                        "superseded_by": new_memory.id,
                    }
                    await self.vector_store.upsert(
                        user_id=clean_user_id,
                        points=[VectorPoint(id=old_id, vector=old_vec, payload=payload)],
                    )

        return ConflictResolutionResult(
            new_memory=new_memory,
            superseded_memory_ids=superseded_ids,
        )

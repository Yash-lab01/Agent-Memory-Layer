"""Deduplication and near-duplicate fact merging engine for memory consolidation."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any

import litellm
from pydantic import BaseModel, Field, ValidationError

from memlayer.consolidate.prompts import DEDUP_VERIFICATION_SYSTEM_PROMPT
from memlayer.extract.extractor import clean_json_output
from memlayer.llm.embedder import BaseEmbedder
from memlayer.models import MemoryItem, MemoryStatus
from memlayer.store.sql import SQLStore
from memlayer.store.vector_base import VectorPoint, VectorStoreBase

logger = logging.getLogger(__name__)

CompletionFn = Callable[..., Awaitable[Any]]


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    val = dot / (norm1 * norm2)
    return max(-1.0, min(1.0, val))


class DuplicatePair(BaseModel):
    """Details of an identified and merged duplicate pair."""

    canonical_id: str
    duplicate_id: str
    similarity: float
    reason: str = ""


class DedupVerificationResult(BaseModel):
    """Model for LLM verification of duplicate relationship."""

    is_duplicate: bool
    canonical_id: str = ""
    reason: str = ""


class DedupResult(BaseModel):
    """Result of running deduplication consolidation for a user."""

    user_id: str
    merged_count: int
    pairs: list[DuplicatePair] = Field(default_factory=list)


class DeduplicationEngine:
    """Identifies and consolidates near-duplicate facts within a user namespace.

    Merges redundant active memories into a canonical representative without ever
    hard-deleting records, preserving provenance and audit history.
    """

    def __init__(
        self,
        store: SQLStore,
        vector_store: VectorStoreBase,
        embedder: BaseEmbedder,
        model: str = "gemini/gemini-1.5-flash",
        completion_fn: CompletionFn | None = None,
    ) -> None:
        """Initialize the deduplication engine."""
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder
        self.model = model
        self.completion_fn = completion_fn or litellm.acompletion

    async def verify_duplicate(
        self,
        mem1: MemoryItem,
        mem2: MemoryItem,
        similarity: float,
        exact_threshold: float = 0.98,
    ) -> DedupVerificationResult:
        """Determine whether two candidate memories are duplicates and choose the canonical."""
        # 1. Fast heuristic: Identical or near-exact match
        is_exact_text = mem1.content.strip().lower() == mem2.content.strip().lower()
        if is_exact_text or similarity >= exact_threshold:
            # Canonical selection policy: prefer longer/more descriptive, or higher access/importance
            if len(mem1.content) > len(mem2.content):
                canonical = mem1
            elif len(mem2.content) > len(mem1.content):
                canonical = mem2
            elif (mem1.access_count, mem1.importance) >= (mem2.access_count, mem2.importance):
                canonical = mem1
            else:
                canonical = mem2

            return DedupVerificationResult(
                is_duplicate=True,
                canonical_id=canonical.id,
                reason="High semantic similarity or exact textual match",
            )

        # 2. LLM verification for borderline similarities
        user_prompt = (
            f"MEMORY 1:\n- ID: {mem1.id}\n- Entity: {mem1.entity}\n- Content: {mem1.content}\n\n"
            f"MEMORY 2:\n- ID: {mem2.id}\n- Entity: {mem2.entity}\n- Content: {mem2.content}\n\n"
            f"Analyze if these two memories are near-duplicates or distinct facts. Return JSON matching schema."
        )

        messages = [
            {"role": "system", "content": DEDUP_VERIFICATION_SYSTEM_PROMPT},
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
            parsed = DedupVerificationResult.model_validate_json(cleaned_json)
            # Ensure valid canonical_id returned
            if parsed.is_duplicate and parsed.canonical_id not in (mem1.id, mem2.id):
                parsed.canonical_id = mem1.id
            return parsed
        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            logger.warning("Dedup verification failed or returned invalid JSON: %s", exc)
            return DedupVerificationResult(
                is_duplicate=False,
                canonical_id="",
                reason=f"Verification fallback: {exc}",
            )

    async def deduplicate_user(
        self,
        user_id: str,
        similarity_threshold: float = 0.88,
        exact_threshold: float = 0.98,
    ) -> DedupResult:
        """Scan active memories for user_id, cluster near-duplicates, and merge non-destructively."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        clean_user_id = user_id.strip()

        # Step 1: Retrieve all active memories for the user
        active_memories = await self.store.list_memories(
            user_id=clean_user_id,
            status=MemoryStatus.ACTIVE,
            limit=500,
        )

        if len(active_memories) < 2:
            return DedupResult(user_id=clean_user_id, merged_count=0, pairs=[])

        # Step 2: Compute embeddings
        contents = [m.content for m in active_memories]
        vectors = await self.embedder.embed_texts(contents)

        superseded_in_run: set[str] = set()
        merged_pairs: list[DuplicatePair] = []

        # Step 3: Pairwise candidate comparison
        num_mems = len(active_memories)
        for i in range(num_mems):
            mem_i = active_memories[i]
            if mem_i.id in superseded_in_run:
                continue

            for j in range(i + 1, num_mems):
                mem_j = active_memories[j]
                if mem_j.id in superseded_in_run:
                    continue

                # Skip if entities explicitly differ (unless generic 'User' or missing)
                if (
                    mem_i.entity
                    and mem_j.entity
                    and mem_i.entity.lower() != mem_j.entity.lower()
                    and "user" not in (mem_i.entity.lower(), mem_j.entity.lower())
                ):
                    continue

                sim = _cosine_similarity(vectors[i], vectors[j])
                if sim >= similarity_threshold:
                    verification = await self.verify_duplicate(
                        mem_i, mem_j, similarity=sim, exact_threshold=exact_threshold
                    )
                    if verification.is_duplicate:
                        if verification.canonical_id == mem_i.id:
                            canonical, duplicate = mem_i, mem_j
                            dup_index = j
                        else:
                            canonical, duplicate = mem_j, mem_i
                            dup_index = i

                        # 3a. Mark duplicate as superseded in SQL (never delete)
                        success = await self.store.mark_superseded(
                            user_id=clean_user_id,
                            old_memory_id=duplicate.id,
                            new_memory_id=canonical.id,
                        )
                        if success:
                            # 3b. Reinforce canonical memory in SQL
                            await self.store.record_access(clean_user_id, canonical.id)

                            # 3c. Update duplicate vector point payload
                            dup_vec = vectors[dup_index]
                            payload = {
                                "content": duplicate.content,
                                "entity": duplicate.entity,
                                "importance": duplicate.importance,
                                "created_at": duplicate.created_at.isoformat(),
                                "event_time": (
                                    duplicate.event_time.isoformat()
                                    if duplicate.event_time
                                    else None
                                ),
                                "status": MemoryStatus.SUPERSEDED.value,
                                "superseded_by": canonical.id,
                            }
                            await self.vector_store.upsert(
                                user_id=clean_user_id,
                                points=[
                                    VectorPoint(
                                        id=duplicate.id,
                                        vector=dup_vec,
                                        payload=payload,
                                    )
                                ],
                            )

                            superseded_in_run.add(duplicate.id)
                            merged_pairs.append(
                                DuplicatePair(
                                    canonical_id=canonical.id,
                                    duplicate_id=duplicate.id,
                                    similarity=sim,
                                    reason=verification.reason,
                                )
                            )

                            # If mem_i was the duplicate that got superseded, stop testing it against others
                            if duplicate.id == mem_i.id:
                                break

        return DedupResult(
            user_id=clean_user_id,
            merged_count=len(merged_pairs),
            pairs=merged_pairs,
        )

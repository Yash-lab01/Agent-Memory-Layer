"""WanderAI Agent: Travel planning assistant powered by Agent Memory Layer.

Demonstrates durable memory across trips, non-destructive preference updates,
and multi-hop knowledge graph reasoning.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from memlayer.api import MemoryLayer
from memlayer.consolidate.conflicts import ConflictDetector
from memlayer.models import Episode, Role

from .scenarios import SEED_TRAVEL_TRIPLETS

logger = logging.getLogger("memlayer.wanderai")


class TravelerProfile(BaseModel):
    """Structured synthesis of a traveler's durable preferences across sessions."""

    user_id: str
    active_preferences: list[dict[str, Any]] = Field(default_factory=list)
    superseded_preferences: list[dict[str, Any]] = Field(default_factory=list)
    dietary: list[str] = Field(default_factory=list)
    airline_loyalty: list[str] = Field(default_factory=list)
    budget_tier: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    destinations: list[str] = Field(default_factory=list)
    graph_entities_count: int = 0
    graph_relations_count: int = 0


class WanderAIChatResult(BaseModel):
    """Result of a conversational turn with WanderAI."""

    reply: str
    memory_enabled: bool
    retrieved_memories: list[dict[str, Any]] = Field(default_factory=list)
    context_used: str = ""
    graph_context: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    session_id: str = "default_session"


class WanderAIAgent:
    """Conversational Travel Agent integrating MemoryLayer for durable cross-trip memory."""

    def __init__(
        self,
        memlayer: MemoryLayer,
        conflict_detector: ConflictDetector | None = None,
    ) -> None:
        self.memlayer = memlayer
        self.conflict_detector = conflict_detector

    async def seed_travel_knowledge(self, user_id: str) -> int:
        """Seed destination topology triplets for the traveler's knowledge graph."""
        count = 0
        if self.memlayer.graph_store:
            for src, rel, tgt in SEED_TRAVEL_TRIPLETS:
                await self.memlayer.add_triplet(
                    user_id=user_id,
                    source=src,
                    relation=rel,
                    target=tgt,
                )
                count += 1
        return count

    async def chat(
        self,
        user_id: str,
        message: str,
        session_id: str = "default_trip",
        memory_enabled: bool = True,
    ) -> WanderAIChatResult:
        """Handle a dialogue turn with memory retrieval and async fact ingestion."""
        start_time = time.perf_counter()
        retrieved_memories: list[dict[str, Any]] = []
        context_used = ""
        graph_context = ""

        # Step 1: Memory Retrieval (if Memory is ON)
        if memory_enabled:
            search_res = await self.memlayer.search(
                user_id=user_id,
                query=message,
                limit=5,
                token_budget=600,
            )
            context_used = search_res.context
            for sm in search_res.memories:
                retrieved_memories.append(
                    {
                        "id": sm.memory.id,
                        "content": sm.memory.content,
                        "entity": sm.memory.entity,
                        "score": round(sm.score, 4),
                        "status": sm.memory.status.value,
                        "created_at": sm.memory.created_at.isoformat(),
                    }
                )

            if "[Knowledge Graph Relationships]" in context_used:
                parts = context_used.split("[Knowledge Graph Relationships]")
                if len(parts) > 1:
                    graph_context = "[Knowledge Graph Relationships]" + parts[1]

        # Step 2: Formulate Prompt & Generate Response
        system_prompt = (
            "You are WanderAI, an expert personalized travel planning assistant. "
            "You craft curated itineraries and hotel/dining recommendations tailored strictly "
            "to the traveler's personal preferences, dietary restrictions, budget, and loyalty status.\n"
            "If the traveler has stated dietary needs (e.g. vegan), NEVER recommend non-compliant dining. "
            "If they have budget limits (e.g. under $150/night), recommend hotels fitting that tier. "
            "Be encouraging, concise, and helpful."
        )
        if context_used:
            system_prompt += (
                f"\n\nTRAVELER PROFILE & RETRIEVED MEMORIES FROM PAST SESSIONS:\n{context_used}"
            )

        reply = ""
        prompt_tokens = int(len(system_prompt.split()) * 1.3) + int(len(message.split()) * 1.3)
        completion_tokens = 0

        # Use LiteLLM if API keys are available, otherwise generate contextual travel reply
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            try:
                import litellm

                model = os.environ.get("LLM_MODEL", "gemini/gemini-1.5-flash")
                resp = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                    temperature=0.2,
                )
                reply = resp.choices[0].message.content or ""
                if hasattr(resp, "usage") and resp.usage:
                    prompt_tokens = getattr(resp.usage, "prompt_tokens", prompt_tokens)
                    completion_tokens = getattr(resp.usage, "completion_tokens", 0)
            except Exception as exc:
                logger.warning("LiteLLM completion error, falling back to contextual generator: %s", exc)
                reply = self._generate_contextual_reply(message, context_used)
        else:
            reply = self._generate_contextual_reply(message, context_used)

        if not completion_tokens:
            completion_tokens = int(len(reply.split()) * 1.3)

        # Step 3: Ingest Episodes & Check for Preference Updates
        if memory_enabled:
            turn_episodes = [
                Episode(
                    user_id=user_id,
                    session_id=session_id,
                    role=Role.USER,
                    text=message,
                ),
                Episode(
                    user_id=user_id,
                    session_id=session_id,
                    role=Role.ASSISTANT,
                    text=reply,
                ),
            ]
            added_memories = await self.memlayer.add(
                user_id=user_id,
                messages=turn_episodes,
                session_id=session_id,
            )

            # Resolve contradictions and preference updates
            if self.conflict_detector:
                for new_mem in added_memories:
                    try:
                        await self.conflict_detector.resolve_and_supersede(
                            user_id=user_id,
                            new_memory=new_mem,
                        )
                    except Exception as c_exc:
                        logger.warning("WanderAI conflict resolution warning: %s", c_exc)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return WanderAIChatResult(
            reply=reply,
            memory_enabled=memory_enabled,
            retrieved_memories=retrieved_memories,
            context_used=context_used,
            graph_context=graph_context,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            session_id=session_id,
        )

    def _generate_contextual_reply(self, message: str, context: str) -> str:
        """Context-aware fallback travel response for offline execution and tests."""
        msg_lower = message.lower()
        ctx_lower = context.lower()

        is_vegan = "vegan" in ctx_lower or "vegan" in msg_lower
        is_oneworld = "oneworld" in ctx_lower or "oneworld" in msg_lower
        is_star_alliance = "star alliance" in ctx_lower and not is_oneworld
        is_budget = "150" in ctx_lower or "budget" in ctx_lower or "150" in msg_lower

        if "tokyo" in msg_lower:
            parts = [
                "Tokyo is an extraordinary destination! Based on your preferences, here is a tailored plan:",
            ]
            if is_vegan:
                parts.append(
                    "- **Dining:** Curated 100% plant-based gourmet spots including Ain Soph Journey in Shinjuku and T's TanTan."
                )
            if is_oneworld:
                parts.append(
                    "- **Flights:** Japan Airlines (JAL) flights via your OneWorld Sapphire status for priority boarding and lounge access."
                )
            elif is_star_alliance:
                parts.append(
                    "- **Flights:** ANA flights with your Star Alliance Gold lounge benefits."
                )
            if "art" in msg_lower or "art" in ctx_lower:
                parts.append(
                    "- **Culture:** The Mori Art Museum in Roppongi and 21_21 Design Sight in Akasaka."
                )
            return "\n".join(parts)

        if "paris" in msg_lower:
            parts = [
                "I would love to help you plan your autumn getaway to Paris! Here are personalized recommendations:",
            ]
            if is_budget:
                parts.append(
                    "- **Neighborhood & Stay:** Le Marais district at the CitizenM Boutique Hotel (~$130/night, fitting your <$150/night budget)."
                )
            else:
                parts.append("- **Neighborhood:** Le Marais district with vibrant art galleries.")

            if is_vegan:
                parts.append(
                    "- **Dining:** Wild & The Moon in Le Marais for delicious organic plant-based dishes, followed by Land&Monkeys vegan bakery."
                )
            if is_oneworld:
                parts.append(
                    "- **Flight Note:** Booking via British Airways / American Airlines utilizing your OneWorld Sapphire privileges."
                )
            return "\n".join(parts)

        if "art museum" in msg_lower and "marais" in msg_lower:
            return (
                "Right in Le Marais, you have the world-renowned **Centre Pompidou** featuring an incredible "
                "contemporary art collection! It is just a short 7-minute walk from CitizenM Boutique Hotel. "
                "For breakfast nearby, **Wild & The Moon** on Rue Charlot offers fantastic 100% vegan pastries and matcha."
            )

        if "switch" in msg_lower or "budget" in msg_lower or "oneworld" in msg_lower:
            return (
                "Noted! I have updated your traveler profile: your primary airline loyalty is now "
                "OneWorld Sapphire, and your hotel budget is capped under $150/night. "
                "Your previous Star Alliance preference has been archived."
            )

        return (
            f"I have recorded your travel preferences regarding '{message}'. "
            "I'll ensure all future trip recommendations honor these constraints!"
        )

    async def get_traveler_profile(self, user_id: str) -> TravelerProfile:
        """Synthesize a traveler's complete profile with active facts and superseded history."""
        memories = await self.memlayer.store.list_memories(user_id=user_id, limit=200)

        active = [m for m in memories if m.status.value == "active"]
        superseded = [m for m in memories if m.status.value == "superseded"]

        dietary: list[str] = []
        airline_loyalty: list[str] = []
        budget_tier: list[str] = []
        interests: list[str] = []
        destinations: list[str] = []

        for m in active:
            c_lower = m.content.lower()
            ent = (m.entity or "").lower()

            if "vegan" in c_lower or "vegetarian" in c_lower or "gluten" in c_lower or "diet" in ent:
                dietary.append(m.content)
            elif "alliance" in c_lower or "oneworld" in c_lower or "airline" in c_lower or "loyalty" in ent:
                airline_loyalty.append(m.content)
            elif "budget" in c_lower or "$" in c_lower or "price" in ent:
                budget_tier.append(m.content)
            elif "art" in c_lower or "museum" in c_lower or "interest" in ent:
                interests.append(m.content)
            elif "tokyo" in c_lower or "paris" in c_lower or "destination" in ent:
                destinations.append(m.content)

        graph_nodes_count = 0
        graph_edges_count = 0
        if self.memlayer.graph_store and hasattr(self.memlayer.graph_store, "_nodes"):
            user_nodes = self.memlayer.graph_store._nodes.get(user_id, {})  # type: ignore[attr-defined]
            user_edges = self.memlayer.graph_store._edges.get(user_id, [])  # type: ignore[attr-defined]
            graph_nodes_count = len(user_nodes)
            graph_edges_count = len(user_edges)

        return TravelerProfile(
            user_id=user_id,
            active_preferences=[
                {
                    "id": m.id,
                    "content": m.content,
                    "entity": m.entity,
                    "importance": m.importance,
                    "created_at": m.created_at.isoformat(),
                }
                for m in active
            ],
            superseded_preferences=[
                {
                    "id": m.id,
                    "content": m.content,
                    "entity": m.entity,
                    "superseded_by": m.superseded_by,
                }
                for m in superseded
            ],
            dietary=list(set(dietary)),
            airline_loyalty=list(set(airline_loyalty)),
            budget_tier=list(set(budget_tier)),
            interests=list(set(interests)),
            destinations=list(set(destinations)),
            graph_entities_count=graph_nodes_count,
            graph_relations_count=graph_edges_count,
        )

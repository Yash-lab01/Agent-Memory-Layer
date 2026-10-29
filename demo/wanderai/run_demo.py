"""Standalone executable CLI demo of WanderAI Travel Companion with Agent Memory Layer.

Demonstrates:
- Multi-session preference retention across separate trips (Tokyo vs. Paris)
- Automatic supersession when airline loyalty and budget preferences are updated
- Knowledge graph multi-hop itinerary reasoning
"""

import asyncio
import json
import logging
import sys
from typing import Any

from memlayer.api import MemoryLayer
from memlayer.consolidate.conflicts import ConflictDetector
from memlayer.extract.extractor import FactExtractor
from memlayer.llm.embedder import MockEmbedder
from memlayer.models import Episode, ExtractedFact, ExtractionResult
from memlayer.store.graph_memory import InMemoryGraphStore
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore

try:
    from .agent import WanderAIAgent
    from .scenarios import WANDERAI_SCENARIOS
except ImportError:
    from demo.wanderai.agent import WanderAIAgent
    from demo.wanderai.scenarios import WANDERAI_SCENARIOS

# Configure logging
logging.basicConfig(level=logging.WARNING)


class WanderAIDemoExtractor(FactExtractor):
    """Deterministic extractor for WanderAI travel preferences."""

    async def extract(
        self,
        messages: Any,
    ) -> ExtractionResult:
        # Extract combined text from dialogue messages
        lines = []
        for msg in messages:
            if isinstance(msg, Episode):
                lines.append(msg.text)
            elif isinstance(msg, dict):
                lines.append(str(msg.get("content", "")))
            else:
                lines.append(str(msg))
        t_lower = " ".join(lines).lower()
        facts: list[ExtractedFact] = []

        if "vegan" in t_lower:
            facts.append(
                ExtractedFact(
                    entity="dietary",
                    content="Traveler maintains a strict vegan diet.",
                    importance=0.95,
                    confidence=0.98,
                )
            )
        if "star alliance" in t_lower:
            facts.append(
                ExtractedFact(
                    entity="airline_loyalty",
                    content="Traveler holds Star Alliance Gold loyalty status.",
                    importance=0.85,
                    confidence=0.95,
                )
            )
        if "oneworld" in t_lower:
            facts.append(
                ExtractedFact(
                    entity="airline_loyalty",
                    content="Traveler holds OneWorld Sapphire loyalty status.",
                    importance=0.90,
                    confidence=0.98,
                )
            )
        if "contemporary art" in t_lower:
            facts.append(
                ExtractedFact(
                    entity="interests",
                    content="Traveler loves contemporary art museums.",
                    importance=0.80,
                    confidence=0.90,
                )
            )
        if "150" in t_lower or "budget" in t_lower:
            facts.append(
                ExtractedFact(
                    entity="budget",
                    content="Traveler accommodation budget is strictly under $150 per night.",
                    importance=0.88,
                    confidence=0.95,
                )
            )
        if "tokyo" in t_lower:
            facts.append(
                ExtractedFact(
                    entity="destination",
                    content="Traveler is planning a 7-day trip to Tokyo in spring.",
                    importance=0.75,
                    confidence=0.90,
                )
            )
        if "paris" in t_lower:
            facts.append(
                ExtractedFact(
                    entity="destination",
                    content="Traveler is planning a 4-day autumn trip to Paris.",
                    importance=0.75,
                    confidence=0.90,
                )
            )

        return ExtractionResult(facts=facts)


async def run_wanderai_showcase() -> int:
    """Execute the full 4-turn WanderAI travel companion showcase."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 80)
    print("[WanderAI] TRAVEL COMPANION - POWERED BY AGENT MEMORY LAYER")
    print("=" * 80)
    print("Demonstrating durable cross-trip memory, preference updates, and graph traversal.\n")

    user_id = "wanderer_sarah"

    # 1. Initialize Storage & Memory Layer
    store = SQLStore(db_url="sqlite+aiosqlite:///:memory:")
    await store.init_db()

    embedder = MockEmbedder(dimension=64)
    vector_store = QdrantVectorStore(location=":memory:", vector_size=64)
    await vector_store.init_store()

    graph_store = InMemoryGraphStore()
    extractor = WanderAIDemoExtractor()

    memlayer = MemoryLayer(
        store=store,
        vector_store=vector_store,
        graph_store=graph_store,
        embedder=embedder,
        extractor=extractor,
    )

    async def _conflict_completion(*_args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        user_msg = messages[-1]["content"] if messages else ""
        conflicts = []

        # Detect airline loyalty switch: OneWorld supersedes Star Alliance
        if "oneworld" in user_msg.lower() and "star alliance" in user_msg.lower():
            import re

            match = re.search(r"ID:\s*([a-f0-9\-]+)", user_msg)
            old_id = match.group(1) if match else "old-mem"
            conflicts.append(
                {
                    "old_memory_id": old_id,
                    "is_contradiction": True,
                    "reason": "Traveler updated primary loyalty program from Star Alliance to OneWorld.",
                }
            )

        text = json.dumps({"conflicts": conflicts})

        class _Msg:
            content = text

        class _Choice:
            message = _Msg()

        class _MockResp:
            choices = [_Choice()]

        return _MockResp()

    conflict_detector = ConflictDetector(
        store=store,
        vector_store=vector_store,
        embedder=embedder,
        completion_fn=_conflict_completion,
    )

    agent = WanderAIAgent(memlayer=memlayer, conflict_detector=conflict_detector)

    # 2. Seed Destination Topology Knowledge Graph
    seed_count = await agent.seed_travel_knowledge(user_id=user_id)
    print(
        f"[Knowledge] Seeded {seed_count} travel topology relations into Knowledge Graph (Paris, Tokyo districts & amenities).\n"
    )

    # 3. Execute Showcase Dialogue Turns
    for scenario in WANDERAI_SCENARIOS:
        turn = scenario["turn"]
        title = scenario["title"]
        session_id = scenario["session_id"]
        user_message = scenario["user_message"]

        print("-" * 80)
        print(f">> TURN {turn}: {title}")
        print(f"   [Session: {session_id}]")
        print(f'   [Traveler]: "{user_message}"')

        result = await agent.chat(
            user_id=user_id,
            message=user_message,
            session_id=session_id,
            memory_enabled=True,
        )

        print(f"\n   [WanderAI]:\n      {result.reply.replace(chr(10), chr(10) + '      ')}\n")

        if result.retrieved_memories:
            print(f"   [Memory Context Injected] ({len(result.retrieved_memories)} facts):")
            for m in result.retrieved_memories:
                print(f"      * [{m['entity']}] {m['content']} (score: {m['score']})")

        if result.graph_context:
            print("   [Graph Subgraph Traversed]:")
            for line in result.graph_context.strip().split("\n"):
                if line.strip():
                    print(f"      {line.strip()}")

        print(f"   [Telemetry]: {result.prompt_tokens} tokens | Latency: {result.latency_ms} ms")
        print()

    # 4. Display Final Traveler Profile & Provenance
    print("=" * 80)
    print("[SUMMARY] FINAL SYNTHESIZED TRAVELER PROFILE")
    print("=" * 80)
    profile = await agent.get_traveler_profile(user_id=user_id)

    print(f"User Namespace : {profile.user_id}")
    print(f"Dietary Needs  : {', '.join(profile.dietary) if profile.dietary else 'None'}")
    print(
        f"Airline Loyalty: {', '.join(profile.airline_loyalty) if profile.airline_loyalty else 'None'}"
    )
    print(f"Budget Tier    : {', '.join(profile.budget_tier) if profile.budget_tier else 'None'}")
    print(f"Interests      : {', '.join(profile.interests) if profile.interests else 'None'}")
    print(f"Destinations   : {len(profile.destinations)} planned trips")
    print(
        f"Graph Topology : {profile.graph_entities_count} entities, {profile.graph_relations_count} relations\n"
    )

    print("[NON-DESTRUCTIVE SUPERSESSION PROVENANCE]:")
    if profile.superseded_preferences:
        for s in profile.superseded_preferences:
            print(f"   * [SUPERSEDED] {s['content']}")
            print(f"     -> Superseded By Memory ID: #{s['superseded_by']}")
    else:
        print("   No superseded items.")

    print("\n[SUCCESS] WanderAI integration demo completed successfully with zero amnesia!")
    await memlayer.close()
    return 0


def main() -> None:
    """CLI entrypoint."""
    sys.exit(asyncio.run(run_wanderai_showcase()))


if __name__ == "__main__":
    main()

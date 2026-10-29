"""FastAPI backend service for Agent Memory Layer interactive demonstration."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import litellm
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from memlayer.api import MemoryLayer
from memlayer.consolidate.conflicts import ConflictDetector
from memlayer.extract.extractor import FactExtractor
from memlayer.llm.embedder import BaseEmbedder, MockEmbedder
from memlayer.models import Episode, ExtractedFact, ExtractionResult, MemoryItem, Role
from memlayer.store.graph_memory import InMemoryGraphStore
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore

from demo.wanderai import SEED_TRAVEL_TRIPLETS, TravelerProfile, WANDERAI_SCENARIOS, WanderAIAgent

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Offline & Rule-Based Fallback Components
# -----------------------------------------------------------------------------


class DemoFactExtractor(FactExtractor):
    """Fact extractor with rule-based heuristics fallback for unauthenticated environments."""

    async def extract(
        self,
        messages: Sequence[dict[str, Any] | Episode],
    ) -> ExtractionResult:
        # First attempt real LLM extraction if API key configured
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            try:
                res = await super().extract(messages)
                if res.facts:
                    return res
            except Exception as exc:
                logger.warning("Remote extraction failed, falling back to rule extraction: %s", exc)

        facts: list[ExtractedFact] = []
        for msg in messages:
            role = msg.role.value if isinstance(msg, Episode) else msg.get("role", "")
            content = msg.text if isinstance(msg, Episode) else str(msg.get("content", ""))
            if role.lower() == "user" and content.strip():
                clean_text = content.strip()
                lower = clean_text.lower()
                matched = False

                # Travel Domain Facts
                if "vegan" in lower:
                    facts.append(
                        ExtractedFact(
                            content="Traveler maintains a strict vegan diet.",
                            entity="Dietary",
                            importance=0.95,
                            confidence=0.98,
                        )
                    )
                    matched = True
                if "star alliance" in lower:
                    facts.append(
                        ExtractedFact(
                            content="Traveler holds Star Alliance Gold loyalty status.",
                            entity="Loyalty",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True
                if "oneworld" in lower:
                    facts.append(
                        ExtractedFact(
                            content="Traveler holds OneWorld Sapphire loyalty status.",
                            entity="Loyalty",
                            importance=0.90,
                            confidence=0.98,
                        )
                    )
                    matched = True
                if "150" in lower or ("budget" in lower and ("hotel" in lower or "night" in lower)):
                    facts.append(
                        ExtractedFact(
                            content="Traveler accommodation budget is strictly under $150 per night.",
                            entity="Budget",
                            importance=0.88,
                            confidence=0.95,
                        )
                    )
                    matched = True
                if "contemporary art" in lower:
                    facts.append(
                        ExtractedFact(
                            content="Traveler loves contemporary art museums.",
                            entity="Interests",
                            importance=0.80,
                            confidence=0.90,
                        )
                    )
                    matched = True
                if "tokyo" in lower and ("spring" in lower or "trip" in lower or "planning" in lower):
                    facts.append(
                        ExtractedFact(
                            content="Traveler is planning a 7-day trip to Tokyo in spring.",
                            entity="Destination",
                            importance=0.75,
                            confidence=0.90,
                        )
                    )
                    matched = True
                if "paris" in lower and ("autumn" in lower or "trip" in lower or "plan" in lower or "eat" in lower):
                    facts.append(
                        ExtractedFact(
                            content="Traveler is planning a 4-day autumn trip to Paris.",
                            entity="Destination",
                            importance=0.75,
                            confidence=0.90,
                        )
                    )
                    matched = True

                # General & Developer Facts
                if (
                    "relocated to austin" in lower
                    or "moved to austin" in lower
                    or ("austin" in lower and "seattle" in lower)
                ):
                    facts.append(
                        ExtractedFact(
                            content="User relocated to Austin, Texas.",
                            entity="Location",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True
                elif "seattle" in lower and "tokyo" not in lower:
                    facts.append(
                        ExtractedFact(
                            content="User lives in Seattle and works as a cloud engineer.",
                            entity="Location",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True
                elif "austin" in lower and "paris" not in lower:
                    facts.append(
                        ExtractedFact(
                            content="User relocated to Austin, Texas.",
                            entity="Location",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True

                if "rust" in lower and ("migrated" in lower or "primary" in lower or "backend" in lower):
                    facts.append(
                        ExtractedFact(
                            content="User primary backend programming language is Rust.",
                            entity="Programming",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True
                elif re.search(r"\b(go|golang)\b", lower) and ("backend" in lower or "programming" in lower or "language" in lower):
                    facts.append(
                        ExtractedFact(
                            content="User primary backend programming language is Go.",
                            entity="Programming",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True

                if "latte" in lower or "oat milk" in lower:
                    facts.append(
                        ExtractedFact(
                            content="User enjoys iced oat milk latte on the porch.",
                            entity="Preferences",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True
                if "dark mode" in lower:
                    facts.append(
                        ExtractedFact(
                            content="User prefers dark mode UI for code editors.",
                            entity="Preferences",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True
                if "civic" in lower or "car" in lower:
                    facts.append(
                        ExtractedFact(
                            content="User owns a 2024 Honda Civic sedan in pearl white.",
                            entity="Vehicle",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True
                if "crossclimate" in lower or "tires" in lower:
                    facts.append(
                        ExtractedFact(
                            content="User installed Michelin CrossClimate2 tires on vehicle.",
                            entity="Vehicle",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True
                if "dog" in lower or "rusty" in lower or "pet" in lower:
                    facts.append(
                        ExtractedFact(
                            content="User adopted a golden retriever named Rusty.",
                            entity="Pets",
                            importance=0.85,
                            confidence=0.95,
                        )
                    )
                    matched = True

                if not matched and not any(k in lower for k in ["where", "what", "how", "who", "help", "recommend"]):
                    facts.append(
                        ExtractedFact(
                            content=clean_text,
                            entity="General",
                            importance=0.5,
                            confidence=0.8,
                        )
                    )

        return ExtractionResult(facts=facts)


def _offline_generate_reply(prompt: str, context: str) -> str:
    """Generate contextual reply when LLM is unauthenticated."""
    lower_prompt = prompt.lower()
    lower_context = context.lower()

    if "where do i live" in lower_prompt or "current city" in lower_prompt:
        if "austin" in lower_context:
            return "Based on your updated records, you currently live in Austin, Texas."
        if "seattle" in lower_context:
            return "According to your records, you currently live in Seattle."
        return "I do not have a record of where you currently live."

    if "backend" in lower_prompt or "programming language" in lower_prompt:
        if "rust" in lower_context:
            return "Your primary backend programming language is Rust."
        if "go" in lower_context:
            return "Your primary backend programming language is Go."
        return "I don't have a record of your primary backend programming language."

    if "car" in lower_prompt or "vehicle" in lower_prompt or "tires" in lower_prompt:
        if "civic" in lower_context and "crossclimate" in lower_context:
            return (
                "You own a 2024 Honda Civic sedan in pearl white with Michelin CrossClimate2 tires."
            )
        if "civic" in lower_context:
            return "You own a 2024 Honda Civic sedan."
        return "I have no vehicle details recorded in your memory."

    if "dog" in lower_prompt or "pet" in lower_prompt or "rusty" in lower_prompt:
        if "rusty" in lower_context:
            return "You have a golden retriever named Rusty!"
        return "I don't have any pets recorded in your memory."

    if "relocated" in lower_prompt or "moved to" in lower_prompt:
        return f"Noted! I've recorded your relocation in your persistent memory: '{prompt}'."

    if "hello" in lower_prompt or "hi" in lower_prompt:
        if context.strip():
            return "Hello! I remember our past conversations. How can I help you today?"
        return "Hello! How can I assist you today?"

    return f"Acknowledged. I have processed: '{prompt}'."


# -----------------------------------------------------------------------------
# Lifespan & Global Application State
# -----------------------------------------------------------------------------

GLOBAL_STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize persistent local SQLite storage, vector store, and MemoryLayer."""
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///demo_memlayer.db")
    store = SQLStore(db_url=db_url)
    await store.init_db()

    embedder: BaseEmbedder = MockEmbedder(dimension=64)
    vector_store = QdrantVectorStore(location=":memory:", vector_size=64)
    await vector_store.init_store()

    graph_store = InMemoryGraphStore()

    extractor = DemoFactExtractor()
    memlayer = MemoryLayer(
        store=store,
        vector_store=vector_store,
        graph_store=graph_store,
        embedder=embedder,
        extractor=extractor,
    )

    # Seed default sample triplets for demo_user to showcase multi-hop graph inspector
    demo_triplets = [
        ("Alice", "WORKS_ON", "Project Apollo"),
        ("Project Apollo", "MANAGED_BY", "Bob"),
        ("Bob", "REPORTS_TO", "Carol"),
        ("Project Apollo", "USES_TECH", "Rust"),
        ("Carol", "LOCATED_IN", "Seattle"),
    ]
    for src, rel, tgt in demo_triplets:
        await memlayer.add_triplet("demo_user", src, rel, tgt)

    async def _conflict_completion(*_args: Any, **kwargs: Any) -> Any:
        # Heuristic fallback for offline conflict detection
        messages = kwargs.get("messages", [])
        user_msg = messages[-1]["content"] if messages else ""
        conflicts = []

        if "austin" in user_msg.lower() and "seattle" in user_msg.lower():
            match = re.search(r"ID:\s*([a-f0-9\-]+)", user_msg)
            old_id = match.group(1) if match else "old-mem"
            conflicts.append(
                {
                    "old_memory_id": old_id,
                    "is_contradiction": True,
                    "reason": "User relocated to Austin, superseding Seattle.",
                }
            )
        elif "rust" in user_msg.lower() and "go" in user_msg.lower():
            match = re.search(r"ID:\s*([a-f0-9\-]+)", user_msg)
            old_id = match.group(1) if match else "old-mem"
            conflicts.append(
                {
                    "old_memory_id": old_id,
                    "is_contradiction": True,
                    "reason": "Backend language updated to Rust, superseding Go.",
                }
            )
        elif "oneworld" in user_msg.lower() and "star alliance" in user_msg.lower():
            match = re.search(r"ID:\s*([a-f0-9\-]+)", user_msg)
            old_id = match.group(1) if match else "old-mem"
            conflicts.append(
                {
                    "old_memory_id": old_id,
                    "is_contradiction": True,
                    "reason": "Airline loyalty updated to OneWorld Sapphire, superseding Star Alliance Gold.",
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

    wanderai_agent = WanderAIAgent(memlayer=memlayer, conflict_detector=conflict_detector)
    await wanderai_agent.seed_travel_knowledge("demo_user")
    await wanderai_agent.seed_travel_knowledge("wanderer_sarah")

    GLOBAL_STATE["store"] = store
    GLOBAL_STATE["vector_store"] = vector_store
    GLOBAL_STATE["graph_store"] = graph_store
    GLOBAL_STATE["memlayer"] = memlayer
    GLOBAL_STATE["conflict_detector"] = conflict_detector
    GLOBAL_STATE["wanderai_agent"] = wanderai_agent

    logger.info("Agent Memory Layer Demo initialized successfully.")
    yield

    await memlayer.close()
    GLOBAL_STATE.clear()
    logger.info("Agent Memory Layer Demo shutdown completed.")


# -----------------------------------------------------------------------------
# FastAPI App Definition & Middleware
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Agent Memory Layer Demo",
    description="Interactive demonstration and inspection dashboard for durable long-term agent memory.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Payload for conversational interaction with the memory-augmented agent."""

    user_id: str = Field(default="demo_user", min_length=1)
    session_id: str = Field(default="session_1", min_length=1)
    message: str = Field(..., min_length=1)
    memory_enabled: bool = Field(default=True)


class RetrievedMemoryItem(BaseModel):
    """Information on an individual memory retrieved for prompt context."""

    id: str
    content: str
    entity: str | None
    score: float
    status: str
    created_at: str


class ChatResponse(BaseModel):
    """Response returned from chat turn, including memory provenance and telemetry."""

    reply: str
    memory_enabled: bool
    retrieved_memories: list[RetrievedMemoryItem]
    context_used: str
    graph_context: str = ""
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class MemoryItemDTO(BaseModel):
    """Full detail of an active or superseded memory item for the inspector."""

    id: str
    user_id: str
    content: str
    entity: str | None
    importance: float
    confidence: float
    status: str
    superseded_by: str | None
    created_at: str
    access_count: int


class GraphNodeDTO(BaseModel):
    """Knowledge graph vertex representation."""

    id: str
    name: str
    entity_type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeDTO(BaseModel):
    """Knowledge graph directed relationship edge."""

    id: str
    source: str
    target: str
    relation: str
    weight: float
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphDataResponse(BaseModel):
    """Topology payload for the knowledge graph visualizer."""

    nodes: list[GraphNodeDTO]
    edges: list[GraphEdgeDTO]


class CreateTripletRequest(BaseModel):
    """Payload to add a relationship triplet to the user's graph."""

    user_id: str = Field(default="demo_user", min_length=1)
    source: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)


class PathResultDTO(BaseModel):
    """A discovered multi-hop path between entities."""

    path: list[GraphEdgeDTO]


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint confirming service readiness."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a dialogue turn to the agent with optional durable memory augmentation."""
    memlayer: MemoryLayer = GLOBAL_STATE["memlayer"]
    conflict_detector: ConflictDetector = GLOBAL_STATE["conflict_detector"]

    start_time = time.perf_counter()
    retrieved_items: list[RetrievedMemoryItem] = []
    context_used = ""

    # Step 1: Memory Retrieval (if Memory Toggle is ON)
    if req.memory_enabled:
        search_res = await memlayer.search(
            user_id=req.user_id,
            query=req.message,
            limit=5,
            token_budget=500,
        )
        context_used = search_res.context
        for sm in search_res.memories:
            retrieved_items.append(
                RetrievedMemoryItem(
                    id=sm.memory.id,
                    content=sm.memory.content,
                    entity=sm.memory.entity,
                    score=round(sm.score, 4),
                    status=sm.memory.status.value,
                    created_at=sm.memory.created_at.isoformat(),
                )
            )

    # Step 2: LLM Generation
    system_prompt = (
        "You are a helpful conversational assistant equipped with durable long-term memory across sessions. "
        "Answer the user's inquiry concisely and accurately."
    )
    if context_used:
        system_prompt += (
            f"\n\nRELEVANT USER MEMORIES (RETRIEVED FROM PRIOR SESSIONS):\n{context_used}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.message},
    ]

    reply = ""
    prompt_tokens = int(len(system_prompt.split()) * 1.3) + int(len(req.message.split()) * 1.3)
    completion_tokens = 0

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        try:
            model = os.environ.get("LLM_MODEL", "gemini/gemini-1.5-flash")
            resp = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            reply = resp.choices[0].message.content or ""
            if hasattr(resp, "usage") and resp.usage:
                prompt_tokens = getattr(resp.usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(resp.usage, "completion_tokens", 0)
        except Exception as exc:
            logger.warning("LLM acompletion error, using contextual fallback: %s", exc)
            reply = _offline_generate_reply(req.message, context_used)
    else:
        reply = _offline_generate_reply(req.message, context_used)

    if not completion_tokens:
        completion_tokens = int(len(reply.split()) * 1.3)

    # Step 3: Asynchronous Fact Ingestion & Supersession (if Memory Toggle is ON)
    if req.memory_enabled:
        turn_episodes = [
            Episode(
                user_id=req.user_id,
                session_id=req.session_id,
                role=Role.USER,
                text=req.message,
            ),
            Episode(
                user_id=req.user_id,
                session_id=req.session_id,
                role=Role.ASSISTANT,
                text=reply,
            ),
        ]
        added_memories = await memlayer.add(
            user_id=req.user_id,
            messages=turn_episodes,
            session_id=req.session_id,
        )

        # Check and resolve any contradictions for newly ingested memories
        for new_mem in added_memories:
            try:
                await conflict_detector.resolve_and_supersede(
                    user_id=req.user_id,
                    new_memory=new_mem,
                )
            except Exception as c_exc:
                logger.warning("Conflict resolution warning: %s", c_exc)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    graph_context = ""
    if "[Knowledge Graph Relationships]" in context_used:
        parts = context_used.split("[Knowledge Graph Relationships]")
        if len(parts) > 1:
            graph_context = "[Knowledge Graph Relationships]" + parts[1]

    return ChatResponse(
        reply=reply,
        memory_enabled=req.memory_enabled,
        retrieved_memories=retrieved_items,
        context_used=context_used,
        graph_context=graph_context,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


@app.get("/api/memories", response_model=list[MemoryItemDTO])
async def list_memories(
    user_id: str = Query(default="demo_user", min_length=1),
) -> list[MemoryItemDTO]:
    """Retrieve all active and superseded memories for a given user for visual inspection."""
    memlayer: MemoryLayer = GLOBAL_STATE["memlayer"]
    memories: list[MemoryItem] = await memlayer.store.list_memories(user_id=user_id, limit=200)

    return [
        MemoryItemDTO(
            id=m.id,
            user_id=m.user_id,
            content=m.content,
            entity=m.entity,
            importance=m.importance,
            confidence=m.confidence,
            status=m.status.value,
            superseded_by=m.superseded_by,
            created_at=m.created_at.isoformat(),
            access_count=m.access_count,
        )
        for m in memories
    ]


@app.delete("/api/memories/{memory_id}")
async def forget_memory(
    memory_id: str,
    user_id: str = Query(default="demo_user", min_length=1),
) -> dict[str, Any]:
    """Explicitly forget/delete a single memory from relational and vector stores."""
    memlayer: MemoryLayer = GLOBAL_STATE["memlayer"]
    deleted = await memlayer.forget(user_id=user_id, memory_id=memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found for user {user_id}.",
        )
    return {"deleted": True, "memory_id": memory_id, "user_id": user_id}


@app.post("/api/reset")
async def reset_store(
    user_id: str = Query(default="demo_user", min_length=1),
) -> dict[str, Any]:
    """Purge all episodes and memories for a given user to restore clean demo state."""
    memlayer: MemoryLayer = GLOBAL_STATE["memlayer"]
    count = await memlayer.forget_all(user_id=user_id)
    return {"success": True, "user_id": user_id, "purged_records": count}


@app.get("/api/graph", response_model=GraphDataResponse)
async def get_graph(
    user_id: str = Query(default="demo_user", min_length=1),
) -> GraphDataResponse:
    """Retrieve all entity vertices and relation edges in the user's knowledge graph."""
    graph_store: InMemoryGraphStore = GLOBAL_STATE["graph_store"]
    clean_user = user_id.strip()
    user_nodes = graph_store._nodes.get(clean_user, {})
    user_edges = graph_store._edges.get(clean_user, [])

    return GraphDataResponse(
        nodes=[
            GraphNodeDTO(
                id=n.id,
                name=n.name,
                entity_type=n.entity_type,
                properties=n.properties,
            )
            for n in user_nodes.values()
        ],
        edges=[
            GraphEdgeDTO(
                id=e.id,
                source=e.source,
                target=e.target,
                relation=e.relation,
                weight=e.weight,
                properties=e.properties,
            )
            for e in user_edges
        ],
    )


@app.post("/api/graph/triplets", response_model=GraphEdgeDTO)
async def add_triplet(req: CreateTripletRequest) -> GraphEdgeDTO:
    """Insert a new relationship triplet into the user's knowledge graph."""
    memlayer: MemoryLayer = GLOBAL_STATE["memlayer"]
    clean_user = req.user_id.strip()
    edge = await memlayer.add_triplet(
        user_id=clean_user,
        source=req.source,
        relation=req.relation,
        target=req.target,
        weight=req.weight,
        properties=req.properties,
    )
    return GraphEdgeDTO(
        id=edge.id,
        source=edge.source,
        target=edge.target,
        relation=edge.relation,
        weight=edge.weight,
        properties=edge.properties,
    )


@app.get("/api/graph/paths", response_model=list[PathResultDTO])
async def find_graph_paths(
    user_id: str = Query(default="demo_user", min_length=1),
    source: str = Query(..., min_length=1),
    target: str = Query(..., min_length=1),
    max_depth: int = Query(default=3, ge=1, le=5),
) -> list[PathResultDTO]:
    """Find multi-hop relation paths connecting two entities within max_depth."""
    graph_store: InMemoryGraphStore = GLOBAL_STATE["graph_store"]
    clean_user = user_id.strip()
    paths = await graph_store.find_paths(clean_user, source, target, max_depth=max_depth)
    return [
        PathResultDTO(
            path=[
                GraphEdgeDTO(
                    id=e.id,
                    source=e.source,
                    target=e.target,
                    relation=e.relation,
                    weight=e.weight,
                    properties=e.properties,
                )
                for e in p
            ]
        )
        for p in paths
    ]


@app.delete("/api/graph")
async def delete_graph(
    user_id: str = Query(default="demo_user", min_length=1),
) -> dict[str, Any]:
    """Purge all entity nodes and relation edges in the user's knowledge graph."""
    graph_store: InMemoryGraphStore = GLOBAL_STATE["graph_store"]
    clean_user = user_id.strip()
    count = await graph_store.delete_user_graph(clean_user)
    return {"deleted": count, "user_id": clean_user}


# -----------------------------------------------------------------------------
# WanderAI Travel Companion Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/wanderai/scenarios")
async def get_wanderai_scenarios() -> list[dict[str, Any]]:
    """Return the predefined showcase dialogue turns for WanderAI."""
    return WANDERAI_SCENARIOS


@app.post("/api/wanderai/chat", response_model=ChatResponse)
async def wanderai_chat(req: ChatRequest) -> ChatResponse:
    """Execute a conversational travel turn with WanderAI."""
    agent: WanderAIAgent = GLOBAL_STATE["wanderai_agent"]
    result = await agent.chat(
        user_id=req.user_id,
        message=req.message,
        session_id=req.session_id,
        memory_enabled=req.memory_enabled,
    )
    return ChatResponse(
        reply=result.reply,
        memory_enabled=result.memory_enabled,
        retrieved_memories=[
            RetrievedMemoryItem(
                id=m["id"],
                content=m["content"],
                entity=m.get("entity"),
                score=m["score"],
                status=m["status"],
                created_at=m["created_at"],
            )
            for m in result.retrieved_memories
        ],
        context_used=result.context_used,
        graph_context=result.graph_context,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
    )


@app.get("/api/wanderai/profile", response_model=TravelerProfile)
async def get_traveler_profile(
    user_id: str = Query(default="wanderer_sarah", min_length=1),
) -> TravelerProfile:
    """Retrieve synthesized traveler profile including active facts and superseded history."""
    agent: WanderAIAgent = GLOBAL_STATE["wanderai_agent"]
    clean_user = user_id.strip()
    return await agent.get_traveler_profile(user_id=clean_user)


@app.post("/api/wanderai/seed")
async def seed_wanderai_knowledge(
    user_id: str = Query(default="wanderer_sarah", min_length=1),
) -> dict[str, Any]:
    """Seed destination knowledge graph triplets for a traveler."""
    agent: WanderAIAgent = GLOBAL_STATE["wanderai_agent"]
    clean_user = user_id.strip()
    count = await agent.seed_travel_knowledge(user_id=clean_user)
    return {"status": "seeded", "relations_added": count, "user_id": clean_user}


# -----------------------------------------------------------------------------
# Static UI Mounting & Root Route
# -----------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    """Serve the single-page interactive demo UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="UI frontend static assets not found.",
        )
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("demo.app.main:app", host="0.0.0.0", port=8000, reload=True)

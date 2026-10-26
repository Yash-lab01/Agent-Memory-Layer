"""Public API for the Agent Memory Layer (memlayer).

Defines the central MemoryLayer class coordinating episodic logging,
fact extraction, hybrid retrieval, and consolidation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, overload

from memlayer.consolidate.decay import MemoryDecayEngine
from memlayer.consolidate.dedup import DeduplicationEngine
from memlayer.consolidate.worker import ConsolidationWorker
from memlayer.extract.extractor import FactExtractor
from memlayer.llm.embedder import BaseEmbedder, MockEmbedder
from memlayer.models import Episode, MemoryItem, MemoryStatus, RelationEdge, SearchResult
from memlayer.retrieve.format import format_context, format_subgraph_context
from memlayer.retrieve.hybrid import HybridRetriever
from memlayer.retrieve.router import RetrievalRouter
from memlayer.retrieve.scoring import ScoringWeights, rank_memories
from memlayer.store.episodes import EpisodeStore
from memlayer.store.graph_base import GraphStoreBase
from memlayer.store.sql import SQLStore
from memlayer.store.vector import QdrantVectorStore
from memlayer.store.vector_base import VectorPoint, VectorStoreBase
from memlayer.tracing import BaseTracer, get_tracer


class MemoryLayer:
    """Central public interface for Agent Memory Layer.

    Coordinates durable episodic logging, atomic fact extraction,
    vector storage, and memory retrieval with strict per-user isolation.
    """

    def __init__(
        self,
        store: SQLStore,
        vector_store: VectorStoreBase,
        embedder: BaseEmbedder,
        extractor: FactExtractor | None = None,
        episode_logger: EpisodeStore | None = None,
        router: RetrievalRouter | None = None,
        retriever: HybridRetriever | None = None,
        scoring_weights: ScoringWeights | None = None,
        graph_store: GraphStoreBase | None = None,
        tracer: BaseTracer | None = None,
    ) -> None:
        """Initialize MemoryLayer with explicit storage and model backends."""
        self.store = store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.extractor = extractor or FactExtractor()
        self.episode_logger = episode_logger or EpisodeStore(store)
        self.router = router or RetrievalRouter()
        self.retriever = retriever or HybridRetriever(
            store=store,
            vector_store=vector_store,
            embedder=embedder,
        )
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.tracer = tracer or get_tracer()
        self._worker: ConsolidationWorker | None = None

    @classmethod
    async def create_local(
        cls,
        db_url: str = "sqlite+aiosqlite:///:memory:",
        vector_dim: int = 64,
        embedder: BaseEmbedder | None = None,
        extractor: FactExtractor | None = None,
        router: RetrievalRouter | None = None,
        scoring_weights: ScoringWeights | None = None,
        graph_store: GraphStoreBase | None = None,
        tracer: BaseTracer | None = None,
    ) -> MemoryLayer:
        """Create a self-contained in-memory MemoryLayer instance for testing and local usage."""
        store = SQLStore(db_url=db_url)
        await store.init_db()

        actual_embedder = embedder or MockEmbedder(dimension=vector_dim)
        dim = getattr(actual_embedder, "dimension", vector_dim)

        vector_store = QdrantVectorStore(
            location=":memory:",
            vector_size=dim,
        )
        await vector_store.init_store()

        return cls(
            store=store,
            vector_store=vector_store,
            graph_store=graph_store,
            embedder=actual_embedder,
            extractor=extractor or FactExtractor(),
            router=router,
            scoring_weights=scoring_weights,
            tracer=tracer,
        )

    # -------------------------------------------------------------------------
    # Write Path: Episode Logging & Fact Ingestion
    # -------------------------------------------------------------------------

    @overload
    async def add(
        self,
        user_id: str,
        messages: Sequence[dict[str, Any] | Episode],
        session_id: str | None = None,
        background: Literal[False] = False,
    ) -> list[MemoryItem]: ...

    @overload
    async def add(
        self,
        user_id: str,
        messages: Sequence[dict[str, Any] | Episode],
        session_id: str | None = None,
        background: Literal[True] = ...,
    ) -> asyncio.Task[list[MemoryItem]]: ...

    @overload
    async def add(
        self,
        user_id: str,
        messages: Sequence[dict[str, Any] | Episode],
        session_id: str | None = None,
        background: bool = False,
    ) -> list[MemoryItem] | asyncio.Task[list[MemoryItem]]: ...

    async def add(
        self,
        user_id: str,
        messages: Sequence[dict[str, Any] | Episode],
        session_id: str | None = None,
        background: bool = False,
    ) -> list[MemoryItem] | asyncio.Task[list[MemoryItem]]:
        """Log dialogue episodes, extract atomic facts, and index them into stores.

        Args:
            user_id: Unique user identifier for strict isolation.
            messages: List of turns (dicts with role/content or Episode models).
            session_id: Optional dialogue session identifier.
            background: If True, extraction and vector indexing run in an async Task.

        Returns:
            List of saved MemoryItem instances if background=False,
            or the asyncio.Task handle if background=True.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        clean_user_id = user_id.strip()

        # Normalize messages to dict[str, str] for episode storage
        dict_messages: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, Episode):
                dict_messages.append({"role": msg.role.value, "content": msg.text})
            elif isinstance(msg, dict):
                role_val = msg.get("role", "user")
                role_str = role_val.value if hasattr(role_val, "value") else str(role_val)
                dict_messages.append({"role": role_str, "content": str(msg.get("content", ""))})

        trace = self.tracer.trace(
            "memory_add",
            user_id=clean_user_id,
            session_id=session_id or "default",
            metadata={"message_count": len(dict_messages), "background": background},
        )

        try:
            # Step 1: Log episodes immediately for durable provenance
            with self.tracer.span(
                "episode_logger", parent=trace, metadata={"message_count": len(dict_messages)}
            ) as ep_span:
                logged_episodes = await self.episode_logger.append_turns(
                    user_id=clean_user_id,
                    messages=dict_messages,
                    session_id=session_id or "default",
                )
                ep_span.set_output({"logged_count": len(logged_episodes)})

            if not logged_episodes:
                trace.set_output({"extracted_facts_count": 0, "status": "no_episodes"})
                trace.end(status="ok")
                if background:

                    async def _empty() -> list[MemoryItem]:
                        return []

                    return asyncio.create_task(_empty())
                return []

            primary_source_id = logged_episodes[0].id

            async def _ingest_pipeline() -> list[MemoryItem]:
                try:
                    with self.tracer.span("fact_extractor", parent=trace) as ext_span:
                        extraction_result = await self.extractor.extract(logged_episodes)
                        ext_span.set_output({"facts_extracted": len(extraction_result.facts)})

                    if not extraction_result.facts:
                        trace.set_output({"extracted_facts_count": 0})
                        trace.end(status="ok")
                        return []

                    # Generate embeddings for extracted facts
                    contents = [f.content for f in extraction_result.facts]
                    with self.tracer.span(
                        "embedder", parent=trace, metadata={"fact_count": len(contents)}
                    ) as emb_span:
                        vectors = await self.embedder.embed_texts(contents)
                        emb_span.set_output({"vectors_generated": len(vectors)})

                    saved_items: list[MemoryItem] = []
                    vector_points: list[VectorPoint] = []

                    for fact, vec in zip(extraction_result.facts, vectors, strict=True):
                        item = MemoryItem(
                            user_id=clean_user_id,
                            content=fact.content,
                            entity=fact.entity,
                            importance=fact.importance,
                            confidence=fact.confidence,
                            event_time=fact.event_time,
                            source_episode_id=primary_source_id,
                            status=MemoryStatus.ACTIVE,
                        )
                        saved_item = await self.store.add_memory(item)
                        saved_items.append(saved_item)

                        payload = {
                            "content": saved_item.content,
                            "entity": saved_item.entity,
                            "importance": saved_item.importance,
                            "created_at": saved_item.created_at.isoformat(),
                            "event_time": (
                                saved_item.event_time.isoformat() if saved_item.event_time else None
                            ),
                            "status": saved_item.status.value,
                        }
                        vector_points.append(
                            VectorPoint(
                                id=saved_item.id,
                                vector=vec,
                                payload=payload,
                            )
                        )

                    if vector_points:
                        with self.tracer.span(
                            "vector_store",
                            parent=trace,
                            metadata={"point_count": len(vector_points)},
                        ) as vs_span:
                            await self.vector_store.upsert(clean_user_id, vector_points)
                            vs_span.set_output({"upserted_points": len(vector_points)})

                    trace.set_output({"extracted_facts_count": len(saved_items)})
                    trace.end(status="ok")
                    return saved_items
                except Exception as exc:
                    trace.end(status="error", error=exc)
                    raise

            if background:
                return asyncio.create_task(_ingest_pipeline())

            return await _ingest_pipeline()
        except Exception as exc:
            trace.end(status="error", error=exc)
            raise

    # -------------------------------------------------------------------------
    # Read Path: Search (Milestone 3)
    # -------------------------------------------------------------------------

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        token_budget: int | None = None,
        bypass_router: bool = False,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        include_graph: bool = True,
    ) -> SearchResult:
        """Search memories relevant to query within an optional token budget.

        Args:
            user_id: Unique user identifier for strict isolation.
            query: The user query string.
            limit: Maximum number of memories to retrieve.
            token_budget: Maximum tokens allowed for formatted prompt context.
            bypass_router: If True, forces retrieval even on trivial turns.
            status: Optional memory status filter (default: ACTIVE).
            include_graph: If True and graph_store is present, augments context with subgraphs.

        Returns:
            SearchResult with formatted context string, scored memories, token count, and skipped status.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        clean_user_id = user_id.strip()

        trace = self.tracer.trace(
            "memory_search",
            user_id=clean_user_id,
            metadata={"query": query, "limit": limit, "token_budget": token_budget},
        )

        try:
            # Step 1: Retrieval Router Turn Gating
            with self.tracer.span("router", parent=trace) as router_span:
                should_retrieve = bypass_router or self.router.should_retrieve(query)
                router_span.set_output(
                    {"should_retrieve": should_retrieve, "bypassed": bypass_router}
                )

            if not should_retrieve:
                trace.set_output({"skipped": True, "token_count": 0, "retrieved_count": 0})
                trace.end(status="ok")
                return SearchResult(
                    context="",
                    memories=[],
                    token_count=0,
                    skipped=True,
                )

            # Step 2: Hybrid Retrieval (Dense Vector + Sparse BM25)
            with self.tracer.span("hybrid_retriever", parent=trace) as retriever_span:
                matches = await self.retriever.search(
                    user_id=clean_user_id,
                    query=query,
                    limit=limit,
                    status=status,
                )
                retriever_span.set_output({"match_count": len(matches)})

            if not matches:
                graph_ctx = ""
                graph_tokens = 0
                if self.graph_store is not None and include_graph:
                    candidate_entities = list(dict.fromkeys(query.split()))
                    nodes, edges = await self.graph_store.get_subgraph(
                        clean_user_id, candidate_entities, max_depth=2
                    )
                    graph_ctx, graph_tokens = format_subgraph_context(
                        nodes, edges, token_budget=token_budget
                    )

                if not graph_ctx:
                    trace.set_output({"skipped": False, "token_count": 0, "retrieved_count": 0})
                    trace.end(status="ok")
                    return SearchResult(
                        context="",
                        memories=[],
                        token_count=0,
                        skipped=False,
                    )

                trace.set_output(
                    {
                        "skipped": False,
                        "token_count": graph_tokens,
                        "retrieved_count": 0,
                    }
                )
                trace.end(status="ok")
                return SearchResult(
                    context=graph_ctx,
                    memories=[],
                    token_count=graph_tokens,
                    skipped=False,
                )

            # Step 3: Multi-Factor Scoring (Similarity + Recency Decay + Importance)
            with self.tracer.span("scorer", parent=trace) as scorer_span:
                candidates = [(m.memory, m.combined_score) for m in matches]
                scored_memories = rank_memories(
                    candidates,
                    weights=self.scoring_weights,
                    limit=limit,
                )
                scorer_span.set_output(
                    {"candidate_count": len(candidates), "scored_count": len(scored_memories)}
                )

            # Step 4: Context Formatting and Token Budget Packing
            with self.tracer.span("formatter", parent=trace) as formatter_span:
                context_str, included_memories, token_count = format_context(
                    scored_memories,
                    token_budget=token_budget,
                )
                if self.graph_store is not None and include_graph:
                    query_entities = [
                        sm.memory.entity
                        for sm in included_memories
                        if sm.memory.entity and sm.memory.entity.lower() not in {"user", "none"}
                    ]
                    candidate_entities = list(dict.fromkeys(query_entities + query.split()))
                    remaining_budget = (
                        (token_budget - token_count) if token_budget is not None else None
                    )
                    if remaining_budget is None or remaining_budget > 10:
                        nodes, edges = await self.graph_store.get_subgraph(
                            clean_user_id, candidate_entities, max_depth=2
                        )
                        graph_ctx, graph_tokens = format_subgraph_context(
                            nodes, edges, token_budget=remaining_budget
                        )
                        if graph_ctx:
                            context_str = (
                                f"{context_str}\n\n{graph_ctx}" if context_str else graph_ctx
                            )
                            token_count += graph_tokens

                formatter_span.set_output(
                    {"included_count": len(included_memories), "token_count": token_count}
                )

            # Step 5: Record Access Telemetry
            for sm in included_memories:
                asyncio.create_task(self.store.record_access(clean_user_id, sm.memory.id))

            trace.set_output(
                {
                    "skipped": False,
                    "token_count": token_count,
                    "retrieved_count": len(included_memories),
                }
            )
            trace.end(status="ok")

            return SearchResult(
                context=context_str,
                memories=included_memories,
                token_count=token_count,
                skipped=False,
            )
        except Exception as exc:
            trace.end(status="error", error=exc)
            raise

    # -------------------------------------------------------------------------
    # Knowledge Graph Operations
    # -------------------------------------------------------------------------

    async def add_triplet(
        self,
        user_id: str,
        source: str,
        relation: str,
        target: str,
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> RelationEdge:
        """Add a semantic relationship triplet to the user's knowledge graph."""
        if self.graph_store is None:
            raise RuntimeError("Cannot add triplet: no graph_store configured on MemoryLayer.")
        clean_user_id = user_id.strip()
        if not clean_user_id:
            raise ValueError("user_id must be a non-empty string.")
        edge = RelationEdge(
            user_id=clean_user_id,
            source=source,
            relation=relation,
            target=target,
            weight=weight,
            properties=properties or {},
        )
        return await self.graph_store.add_edge(clean_user_id, edge)

    async def get_graph_context(
        self,
        user_id: str,
        entities: Sequence[str],
        max_depth: int = 2,
        token_budget: int | None = None,
    ) -> tuple[str, int]:
        """Retrieve and format a multi-hop subgraph context surrounding the specified entities."""
        if self.graph_store is None or not entities:
            return "", 0
        clean_user_id = user_id.strip()
        if not clean_user_id:
            raise ValueError("user_id must be a non-empty string.")
        nodes, edges = await self.graph_store.get_subgraph(
            clean_user_id, entities, max_depth=max_depth
        )
        return format_subgraph_context(nodes, edges, token_budget=token_budget)

    # -------------------------------------------------------------------------
    # Deletion: Privacy & Forget API
    # -------------------------------------------------------------------------

    async def forget(self, user_id: str, memory_id: str) -> bool:
        """Explicitly delete a single memory by ID from both SQL and Vector stores."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        clean_user_id = user_id.strip()

        sql_deleted = bool(await self.store.delete_memory(clean_user_id, memory_id))
        await self.vector_store.delete(clean_user_id, [memory_id])
        return sql_deleted

    async def forget_all(self, user_id: str) -> int:
        """Purge all episodes, memories, and graph data for a user across all stores (GDPR)."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        clean_user_id = user_id.strip()

        sql_count = int(await self.store.delete_user_data(clean_user_id))
        await self.vector_store.delete_user_vectors(clean_user_id)
        if self.graph_store is not None:
            await self.graph_store.delete_user_graph(clean_user_id)
        return sql_count

    # -------------------------------------------------------------------------
    # Consolidation: Background Worker Management
    # -------------------------------------------------------------------------

    def start_consolidation_worker(
        self,
        dedup_interval_minutes: float = 360.0,
        decay_interval_minutes: float = 1440.0,
    ) -> ConsolidationWorker:
        """Initialize and start the background consolidation worker."""
        if self._worker is None:
            dedup_engine = DeduplicationEngine(
                store=self.store,
                vector_store=self.vector_store,
                embedder=self.embedder,
            )
            decay_engine = MemoryDecayEngine(
                store=self.store,
                vector_store=self.vector_store,
                embedder=self.embedder,
            )
            self._worker = ConsolidationWorker(
                store=self.store,
                dedup_engine=dedup_engine,
                decay_engine=decay_engine,
                dedup_interval_minutes=dedup_interval_minutes,
                decay_interval_minutes=decay_interval_minutes,
            )

        if not self._worker.is_running:
            self._worker.start()

        return self._worker

    def stop_consolidation_worker(self) -> None:
        """Stop active background consolidation worker if running."""
        if self._worker is not None and self._worker.is_running:
            self._worker.stop(wait=False)

    async def close(self) -> None:
        """Close database engine, vector store, graph store, background worker, and flush tracer."""
        if self._worker is not None and self._worker.is_running:
            self._worker.stop(wait=False)
        self.tracer.flush()
        await self.store.close()
        await self.vector_store.close()
        if self.graph_store is not None:
            await self.graph_store.close()

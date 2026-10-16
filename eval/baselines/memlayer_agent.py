import json
from typing import Any

import litellm

from eval.baselines.base import BaseBaseline, BaselineOutput
from eval.datasets.models import EvalQuestion
from memlayer.api import MemoryLayer
from memlayer.consolidate.conflicts import ConflictDetector
from memlayer.extract.extractor import FactExtractor
from memlayer.llm.embedder import BaseEmbedder
from memlayer.models import MemoryStatus
from memlayer.retrieve.scoring import ScoringWeights


class MemLayerBaseline(BaseBaseline):
    """MemoryLayer baseline with atomic fact extraction, supersession, BM25 hybrid search, and decay."""

    def __init__(
        self,
        name: str = "memlayer",
        model: str = "gemini/gemini-1.5-flash",
        token_budget: int = 1500,
        memlayer: MemoryLayer | None = None,
        embedder: BaseEmbedder | None = None,
        extractor: FactExtractor | None = None,
        enable_supersession: bool = True,
        enable_decay: bool = True,
        vector_only: bool = False,
        scoring_weights: ScoringWeights | None = None,
        enable_router: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)
        self.token_budget = token_budget
        self.memlayer = memlayer
        self.embedder = embedder
        self.extractor = extractor
        self.enable_supersession = enable_supersession
        self.enable_decay = enable_decay
        self.vector_only = vector_only
        self.scoring_weights = scoring_weights
        self.enable_router = enable_router
        self._owns_memlayer = memlayer is None
        self.conflict_detector: ConflictDetector | None = None

    async def setup(self, item: EvalQuestion) -> None:
        """Ingest background conversation into MemoryLayer with fact extraction & consolidation."""
        if self.memlayer is None:
            self.memlayer = await MemoryLayer.create_local(
                embedder=self.embedder,
                extractor=self.extractor,
            )
            self._owns_memlayer = True

        if self.enable_supersession and self.conflict_detector is None:

            async def _conflict_completion(*args: Any, **kwargs: Any) -> Any:
                if self.llm_client is not None:
                    messages = kwargs.get("messages", [])
                    user_msg = messages[-1]["content"] if messages else ""
                    sys_msg = messages[0]["content"] if len(messages) > 1 else None
                    res = self.llm_client(prompt=user_msg, system_prompt=sys_msg, model=self.model)
                    if hasattr(res, "__await__"):
                        res = await res
                    text = str(res[0]) if isinstance(res, tuple) else str(res)
                    if not text.strip().startswith("{"):
                        text = json.dumps({"conflicts": []})

                    class _MockChoice:
                        message = type("Msg", (), {"content": text})()

                    return type("Resp", (), {"choices": [_MockChoice()]})()
                return await litellm.acompletion(*args, **kwargs)

            self.conflict_detector = ConflictDetector(
                store=self.memlayer.store,
                vector_store=self.memlayer.vector_store,
                embedder=self.memlayer.embedder,
                model=self.model,
                completion_fn=_conflict_completion,
            )

        if not item.conversation:
            return

        # Group turns by session_id to maintain realistic conversational sessions
        sessions: dict[str, list[dict[str, Any]]] = {}
        for idx, turn in enumerate(item.conversation):
            sess_id = turn.session_id or f"session_{idx // 2 + 1}"
            if sess_id not in sessions:
                sessions[sess_id] = []
            sessions[sess_id].append({"role": turn.role, "content": turn.content})

        for sess_id, messages in sessions.items():
            new_items = await self.memlayer.add(
                user_id=item.user_id,
                messages=messages,
                session_id=sess_id,
                background=False,
            )
            if self.enable_supersession and self.conflict_detector and new_items:
                for new_item in new_items:
                    await self.conflict_detector.resolve_and_supersede(
                        user_id=item.user_id,
                        new_memory=new_item,
                    )

    async def answer(self, item: EvalQuestion) -> BaselineOutput:
        """Retrieve memories within token budget and generate prompt answer."""
        if self.memlayer is None:
            self.memlayer = await MemoryLayer.create_local(
                embedder=self.embedder,
                extractor=self.extractor,
            )
            self._owns_memlayer = True

        # Apply ablation configurations
        if self.vector_only:
            self.memlayer.retriever.dense_weight = 1.0
            self.memlayer.retriever.sparse_weight = 0.0

        if self.scoring_weights is not None:
            self.memlayer.scoring_weights = self.scoring_weights
        elif not self.enable_decay:
            self.memlayer.scoring_weights = ScoringWeights(a=0.7, b=0.0, c=0.3)

        status_filter = MemoryStatus.ACTIVE if self.enable_supersession else None
        bypass_router = not self.enable_router

        search_result = await self.memlayer.search(
            user_id=item.user_id,
            query=item.question,
            token_budget=self.token_budget,
            bypass_router=bypass_router,
            status=status_filter,
        )

        retrieved_context = search_result.context or "No relevant memories found."

        system_prompt = (
            "You are a helpful conversational assistant with durable long-term memory. "
            "Answer the user's question accurately using only the recalled memories provided below. "
            "If the memories do not contain the answer, explicitly state that it was not "
            "mentioned or is unknown."
        )

        user_prompt = (
            f"--- RECALLED MEMORIES ---\n"
            f"{retrieved_context}\n"
            f"--- END RECALLED MEMORIES ---\n\n"
            f"Question: {item.question}\n"
            f"Answer:"
        )

        text, prompt_tokens, comp_tokens, latency_ms = await self._generate_llm_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.kwargs.get("temperature", 0.0),
            max_tokens=self.kwargs.get("max_tokens", 512),
        )

        return BaselineOutput(
            prediction=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            total_tokens=prompt_tokens + comp_tokens,
            latency_ms=latency_ms,
            context_used=retrieved_context,
            metadata={
                "recalled_memories_count": len(search_result.memories),
                "context_token_count": search_result.token_count,
                "router_skipped": search_result.skipped,
            },
        )

    async def teardown(self) -> None:
        if self.memlayer and self._owns_memlayer:
            await self.memlayer.close()
            self.memlayer = None

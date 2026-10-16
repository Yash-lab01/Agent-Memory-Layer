"""MemLayer baseline agent incorporating the complete Agent Memory Layer architecture."""

from typing import Any

from eval.baselines.base import BaseBaseline, BaselineOutput
from eval.datasets.models import EvalQuestion
from memlayer.api import MemoryLayer
from memlayer.extract.extractor import FactExtractor
from memlayer.llm.embedder import BaseEmbedder


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
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)
        self.token_budget = token_budget
        self.memlayer = memlayer
        self.embedder = embedder
        self.extractor = extractor
        self._owns_memlayer = memlayer is None

    async def setup(self, item: EvalQuestion) -> None:
        """Ingest background conversation into MemoryLayer with fact extraction & consolidation."""
        if self.memlayer is None:
            self.memlayer = await MemoryLayer.create_local(
                embedder=self.embedder,
                extractor=self.extractor,
            )
            self._owns_memlayer = True

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
            await self.memlayer.add(
                user_id=item.user_id,
                messages=messages,
                session_id=sess_id,
                background=False,
            )

    async def answer(self, item: EvalQuestion) -> BaselineOutput:
        """Retrieve memories within token budget and generate prompt answer."""
        if self.memlayer is None:
            self.memlayer = await MemoryLayer.create_local(
                embedder=self.embedder,
                extractor=self.extractor,
            )
            self._owns_memlayer = True

        search_result = await self.memlayer.search(
            user_id=item.user_id,
            query=item.question,
            token_budget=self.token_budget,
            bypass_router=False,
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

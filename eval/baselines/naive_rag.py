"""Naive RAG baseline for memory benchmarking."""

import math
from typing import Any

from eval.baselines.base import BaseBaseline, BaselineOutput
from eval.datasets.models import EvalQuestion
from memlayer.llm.embedder import BaseEmbedder, MockEmbedder


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class NaiveRAGBaseline(BaseBaseline):
    """Naive RAG baseline retrieving top-k chunks strictly by vector cosine similarity."""

    def __init__(
        self,
        name: str = "naive_rag",
        model: str = "gemini/gemini-1.5-flash",
        top_k: int = 3,
        embedder: BaseEmbedder | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)
        self.top_k = top_k
        self.embedder = embedder or MockEmbedder()
        self._chunks: list[str] = []
        self._vectors: list[list[float]] = []

    async def setup(self, item: EvalQuestion) -> None:
        """Embed and index each conversation turn as a vector chunk."""
        self._chunks = []
        for turn in item.conversation:
            ts_str = turn.timestamp.isoformat() if turn.timestamp else ""
            chunk = f"{turn.role.capitalize()}: {turn.content}"
            if ts_str:
                chunk = f"[{ts_str}] {chunk}"
            self._chunks.append(chunk)

        if self._chunks:
            self._vectors = await self.embedder.embed_texts(self._chunks)
        else:
            self._vectors = []

    async def answer(self, item: EvalQuestion) -> BaselineOutput:
        """Retrieve top-k chunks by vector similarity and prompt LLM."""
        if not self._chunks:
            retrieved_context = "No conversation history available."
        else:
            query_vec = await self.embedder.embed_text(item.question)
            scores = [
                (_cosine_similarity(query_vec, chunk_vec), chunk)
                for chunk, chunk_vec in zip(self._chunks, self._vectors, strict=False)
            ]
            scores.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [chunk for _, chunk in scores[: self.top_k]]
            retrieved_context = "\n".join(top_chunks)

        system_prompt = (
            "You are a helpful conversational assistant. "
            "Answer the user's question accurately using only the retrieved context snippets. "
            "If the context does not contain the answer, explicitly state that it was not "
            "mentioned or is unknown."
        )

        user_prompt = (
            f"--- RETRIEVED CONTEXT ---\n"
            f"{retrieved_context}\n"
            f"--- END RETRIEVED CONTEXT ---\n\n"
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
            metadata={"top_k": self.top_k, "total_chunks": len(self._chunks)},
        )

    async def teardown(self) -> None:
        self._chunks = []
        self._vectors = []

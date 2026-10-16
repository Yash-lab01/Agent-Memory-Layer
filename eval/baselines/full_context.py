"""Full-context baseline for memory benchmarking."""

from typing import Any

from eval.baselines.base import BaseBaseline, BaselineOutput
from eval.datasets.models import EvalQuestion


class FullContextBaseline(BaseBaseline):
    """Full-context baseline concatenating verbatim conversation history into prompt."""

    def __init__(
        self,
        name: str = "full_context",
        model: str = "gemini/gemini-1.5-flash",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)
        self._current_history: str = ""

    async def setup(self, item: EvalQuestion) -> None:
        """Format the entire conversation history into context string."""
        formatted_turns = []
        for idx, turn in enumerate(item.conversation, 1):
            ts_str = turn.timestamp.isoformat() if turn.timestamp else "N/A"
            sess_str = turn.session_id or f"session_{idx}"
            formatted_turns.append(
                f"[Session: {sess_str} | Time: {ts_str}] {turn.role.capitalize()}: {turn.content}"
            )
        self._current_history = "\n".join(formatted_turns)

    async def answer(self, item: EvalQuestion) -> BaselineOutput:
        """Generate answer using full verbatim conversation history."""
        system_prompt = (
            "You are a helpful conversational assistant with access to complete chat history. "
            "Answer the user's question accurately using only the provided conversation history. "
            "If the question asks about something not mentioned in the history, explicitly state "
            "that it was not mentioned or is unknown."
        )

        user_prompt = (
            f"--- CONVERSATION HISTORY ---\n"
            f"{self._current_history}\n"
            f"--- END CONVERSATION HISTORY ---\n\n"
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
            context_used=self._current_history,
            metadata={"history_turns": len(item.conversation)},
        )

    async def teardown(self) -> None:
        self._current_history = ""

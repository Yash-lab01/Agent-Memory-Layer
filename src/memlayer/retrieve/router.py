"""Retrieval router for conversational turn gating to skip retrieval on trivial turns."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    """The result of a query routing evaluation."""

    needs_retrieval: bool
    reason: str


class RetrievalRouter:
    """Classifies user queries to decide whether long-term memory retrieval is needed."""

    # Exact matches after stripping punctuation and lowercasing
    DEFAULT_SKIP_EXACT: frozenset[str] = frozenset(
        {
            # Greetings
            "hi",
            "hello",
            "hey",
            "heya",
            "howdy",
            "sup",
            "hello there",
            "hi there",
            "hey there",
            "good morning",
            "good afternoon",
            "good evening",
            "good night",
            # Gratitude & Acknowledgements
            "thanks",
            "thank you",
            "thanks a lot",
            "thank you so much",
            "thank you very much",
            "thx",
            "ty",
            "cheers",
            "great",
            "ok",
            "okay",
            "k",
            "got it",
            "cool",
            "understood",
            "sure",
            "alright",
            "all right",
            "perfect",
            "awesome",
            "nice",
            "sounds good",
            "no problem",
            "you're welcome",
            "you are welcome",
            # Short confirmations & negations
            "yes",
            "yeah",
            "yep",
            "yup",
            "no",
            "nope",
            "nah",
            "bye",
            "goodbye",
            "see you",
            "cya",
        }
    )

    def __init__(
        self,
        skip_phrases: set[str] | frozenset[str] | None = None,
        min_query_length: int = 3,
    ) -> None:
        """Initialize the router with configurable skip phrases and thresholds."""
        self.skip_phrases = (
            frozenset(s.lower().strip() for s in skip_phrases)
            if skip_phrases is not None
            else self.DEFAULT_SKIP_EXACT
        )
        self.min_query_length = min_query_length

    def _normalize(self, text: str) -> str:
        """Clean and normalize query string for pattern matching."""
        # Remove surrounding whitespace and non-alphanumeric punctuation
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(cleaned.split())

    def route(self, query: str) -> RoutingDecision:
        """Evaluate if the query needs memory retrieval."""
        raw_clean = query.strip()
        if not raw_clean:
            return RoutingDecision(needs_retrieval=False, reason="empty query")

        # Punctuation-only check
        normalized = self._normalize(query)
        if not normalized:
            return RoutingDecision(needs_retrieval=False, reason="punctuation only")

        # Exact match with skip phrases
        if normalized in self.skip_phrases:
            return RoutingDecision(needs_retrieval=False, reason=f"trivial phrase: '{normalized}'")

        # Very short utterances (e.g. 'ok', 'hi')
        if len(normalized) <= self.min_query_length:
            return RoutingDecision(
                needs_retrieval=False, reason=f"too short (len <= {self.min_query_length})"
            )

        return RoutingDecision(needs_retrieval=True, reason="informational query")

    def should_retrieve(self, query: str) -> bool:
        """Return True if the query should trigger memory lookup, False otherwise."""
        return self.route(query).needs_retrieval

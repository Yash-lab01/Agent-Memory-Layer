"""Unit tests for RetrievalRouter turn gating and query classification."""

from __future__ import annotations

import pytest

from memlayer.retrieve.router import RetrievalRouter, RoutingDecision


class TestRetrievalRouter:
    """Test suite for turn-gating classification."""

    @pytest.fixture
    def router(self) -> RetrievalRouter:
        return RetrievalRouter()

    @pytest.mark.parametrize(
        "query",
        [
            "hi",
            "hello",
            "Hey!",
            "good morning",
            "good afternoon",
            "Good evening",
            "howdy",
            "sup",
        ],
    )
    def test_greetings_skip_retrieval(self, router: RetrievalRouter, query: str) -> None:
        decision = router.route(query)
        assert isinstance(decision, RoutingDecision)
        assert decision.needs_retrieval is False
        assert router.should_retrieve(query) is False

    @pytest.mark.parametrize(
        "query",
        [
            "thanks",
            "thank you",
            "thx",
            "ty",
            "ok",
            "okay",
            "k",
            "got it",
            "cool",
            "understood",
            "sure",
            "alright",
            "perfect",
            "awesome!",
            "nice",
            "sounds good",
        ],
    )
    def test_acknowledgements_skip_retrieval(self, router: RetrievalRouter, query: str) -> None:
        assert router.should_retrieve(query) is False

    @pytest.mark.parametrize(
        "query",
        [
            "yes",
            "yeah",
            "yep",
            "no",
            "nope",
            "nah",
        ],
    )
    def test_confirmations_skip_retrieval(self, router: RetrievalRouter, query: str) -> None:
        assert router.should_retrieve(query) is False

    @pytest.mark.parametrize(
        "query",
        [
            "",
            "   ",
            "...",
            "?!",
            "---",
        ],
    )
    def test_empty_and_punctuation_skip_retrieval(
        self, router: RetrievalRouter, query: str
    ) -> None:
        assert router.should_retrieve(query) is False

    @pytest.mark.parametrize(
        "query",
        [
            "Where do I live?",
            "What is my tech stack?",
            "Who is my direct manager?",
            "When did I relocate to Bangalore?",
            "Tell me about my career goals",
            "Recommend a relational database based on my preferences",
            "I need details on project Apollo",
        ],
    )
    def test_informational_queries_trigger_retrieval(
        self, router: RetrievalRouter, query: str
    ) -> None:
        decision = router.route(query)
        assert decision.needs_retrieval is True
        assert router.should_retrieve(query) is True

    def test_custom_skip_phrases(self) -> None:
        custom_router = RetrievalRouter(skip_phrases={"custom skip phrase", "ping"})
        assert custom_router.should_retrieve("custom skip phrase") is False
        assert custom_router.should_retrieve("ping") is False
        # Default phrases shouldn't apply if overridden
        assert custom_router.should_retrieve("What is Python?") is True

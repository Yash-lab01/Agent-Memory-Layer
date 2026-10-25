"""Tests for observability and distributed tracing support.

Verifies NullTracer, InMemoryTracer, LangfuseTracer resilience, and
end-to-end trace collection inside MemoryLayer.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from memlayer.api import MemoryLayer
from memlayer.models import ExtractedFact, ExtractionResult
from memlayer.tracing import (
    InMemoryTracer,
    LangfuseSpan,
    LangfuseTracer,
    NullSpan,
    NullTracer,
    get_tracer,
)


class TestNullTracer:
    """Tests for zero-overhead NullTracer and NullSpan."""

    def test_null_span_operations(self) -> None:
        span = NullSpan()
        # Ensure fluent chaining
        assert span.set_attribute("key", "val") is span
        assert span.set_output({"result": 42}) is span
        span.end(status="ok")
        span.end(status="error", error=RuntimeError("fail"))

    def test_null_span_context_managers(self) -> None:
        span = NullSpan()
        with span as s:
            assert s is span

        with pytest.raises(ValueError, match="test error"):
            with NullSpan():
                raise ValueError("test error")

    @pytest.mark.asyncio
    async def test_null_span_async_context_manager(self) -> None:
        async with NullSpan() as span:
            assert isinstance(span, NullSpan)

        with pytest.raises(RuntimeError, match="async boom"):
            async with NullSpan():
                raise RuntimeError("async boom")

    def test_null_tracer_methods(self) -> None:
        tracer = NullTracer()
        trace = tracer.trace("test_op", user_id="u1", session_id="s1", metadata={"foo": "bar"})
        assert isinstance(trace, NullSpan)

        child = tracer.span("child_op", parent=trace, metadata={"a": 1})
        assert isinstance(child, NullSpan)

        tracer.flush()


class TestInMemoryTracer:
    """Tests for in-memory telemetry recording and assertion utilities."""

    def test_trace_and_child_span_lifecycle(self) -> None:
        tracer = InMemoryTracer()
        with tracer.trace("search_flow", user_id="u_123", session_id="sess_1") as root:
            root.set_attribute("top_k", 5)
            with tracer.span("routing", parent=root) as child:
                child.set_attribute("skip", False)
                child.set_output({"decision": "retrieve"})

            root.set_output({"count": 1})

        assert len(tracer.traces) == 1
        trace = tracer.traces[0]
        assert trace.name == "search_flow"
        assert trace.attributes["user_id"] == "u_123"
        assert trace.attributes["session_id"] == "sess_1"
        assert trace.attributes["top_k"] == 5
        assert trace.output == {"count": 1}
        assert trace.status == "ok"
        assert trace.duration_ms >= 0.0

        assert len(trace.children) == 1
        child_span = trace.children[0]
        assert child_span.name == "routing"
        assert child_span.attributes["skip"] is False
        assert child_span.output == {"decision": "retrieve"}
        assert child_span.status == "ok"

    def test_trace_error_capture(self) -> None:
        tracer = InMemoryTracer()
        with pytest.raises(ValueError, match="flow failed"):
            with tracer.trace("failing_flow", user_id="u_err"):
                raise ValueError("flow failed")

        assert len(tracer.traces) == 1
        recorded = tracer.traces[0]
        assert recorded.status == "error"
        assert "flow failed" in str(recorded.error)

    def test_find_and_clear_traces(self) -> None:
        tracer = InMemoryTracer()
        tracer.trace("add", user_id="u1").end()
        tracer.trace("search", user_id="u1").end()
        tracer.trace("add", user_id="u2").end()

        assert len(tracer.find_traces("add")) == 2
        assert len(tracer.find_traces("search")) == 1

        tracer.clear()
        assert len(tracer.traces) == 0


class TestLangfuseTracer:
    """Tests for Langfuse adapter and fault-tolerant fallbacks."""

    def test_unconfigured_langfuse_degrades_to_null(self) -> None:
        tracer = LangfuseTracer(public_key=None, secret_key=None)
        assert tracer.client is None

        trace = tracer.trace("test", user_id="u1")
        assert isinstance(trace, NullSpan)

        span = tracer.span("child")
        assert isinstance(span, NullSpan)
        tracer.flush()

    def test_langfuse_span_exception_resilience(self) -> None:
        mock_lf_obj = MagicMock()
        mock_lf_obj.update.side_effect = RuntimeError("network timeout")
        mock_lf_obj.end.side_effect = RuntimeError("network timeout")

        span = LangfuseSpan(mock_lf_obj)
        # Should not raise any exception
        span.set_attribute("attr", "val")
        span.set_output({"res": 1})
        span.end(status="error", error=Exception("something"))


class TestGetTracerFactory:
    """Tests for tracer factory configuration."""

    def test_factory_types(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("MEMLAYER_TRACER", raising=False)

        assert isinstance(get_tracer("null"), NullTracer)
        assert isinstance(get_tracer("memory"), InMemoryTracer)
        assert isinstance(get_tracer("langfuse"), LangfuseTracer)
        assert isinstance(get_tracer(), NullTracer)

        monkeypatch.setenv("MEMLAYER_TRACER", "memory")
        assert isinstance(get_tracer(), InMemoryTracer)


class _MockFactExtractor:
    """Deterministic extractor for tracing tests."""

    def __init__(self, facts: list[dict[str, Any]]) -> None:
        self.facts = [
            ExtractedFact(
                content=f["content"],
                entity=f.get("entity", "User"),
                importance=f.get("importance", 0.7),
                confidence=f.get("confidence", 0.9),
            )
            for f in facts
        ]

    async def extract(self, _episodes: Any) -> ExtractionResult:
        return ExtractionResult(facts=self.facts)


class TestMemoryLayerTracingIntegration:
    """End-to-end integration tests of MemoryLayer using InMemoryTracer."""

    @pytest.mark.asyncio
    async def test_add_traces_and_spans(self) -> None:
        tracer = InMemoryTracer()
        extractor = _MockFactExtractor(
            [{"content": "Yash is researching distributed memory systems.", "entity": "Yash"}]
        )

        mem = await MemoryLayer.create_local(
            vector_dim=16,
            extractor=extractor,  # type: ignore[arg-type]
            tracer=tracer,
        )

        try:
            await mem.add(
                user_id="user_alice",
                messages=[{"role": "user", "content": "I am researching distributed memory."}],
                session_id="session_42",
            )

            traces = tracer.find_traces("memory_add")
            assert len(traces) == 1
            root = traces[0]

            assert root.attributes["user_id"] == "user_alice"
            assert root.attributes["session_id"] == "session_42"
            assert root.attributes["message_count"] == 1
            assert root.output == {"extracted_facts_count": 1}
            assert root.status == "ok"
            assert root.duration_ms > 0

            child_names = [c.name for c in root.children]
            assert "episode_logger" in child_names
            assert "fact_extractor" in child_names
            assert "embedder" in child_names
            assert "vector_store" in child_names
        finally:
            await mem.close()

    @pytest.mark.asyncio
    async def test_search_traces_and_spans(self) -> None:
        tracer = InMemoryTracer()
        extractor = _MockFactExtractor(
            [{"content": "Alice prefers Python over Javascript.", "entity": "Alice"}]
        )

        mem = await MemoryLayer.create_local(
            vector_dim=16,
            extractor=extractor,  # type: ignore[arg-type]
            tracer=tracer,
        )

        try:
            await mem.add(
                user_id="user_bob",
                messages=[{"role": "user", "content": "I prefer Python over Javascript."}],
            )
            tracer.clear()

            # Search informational query
            res = await mem.search(
                user_id="user_bob",
                query="What programming language does Alice prefer?",
                bypass_router=True,
            )
            assert len(res.memories) >= 1

            traces = tracer.find_traces("memory_search")
            assert len(traces) == 1
            root = traces[0]

            assert root.attributes["user_id"] == "user_bob"
            assert root.status == "ok"
            assert root.output["skipped"] is False
            assert root.output["retrieved_count"] >= 1

            child_names = [c.name for c in root.children]
            assert "router" in child_names
            assert "hybrid_retriever" in child_names
            assert "scorer" in child_names
            assert "formatter" in child_names
        finally:
            await mem.close()

    @pytest.mark.asyncio
    async def test_search_router_skip_trace(self) -> None:
        tracer = InMemoryTracer()
        mem = await MemoryLayer.create_local(vector_dim=16, tracer=tracer)

        try:
            res = await mem.search(user_id="user_bob", query="thanks!")
            assert res.skipped is True

            traces = tracer.find_traces("memory_search")
            assert len(traces) == 1
            root = traces[0]
            assert root.output["skipped"] is True
            assert root.output["retrieved_count"] == 0
            assert len(root.children) == 1
            assert root.children[0].name == "router"
        finally:
            await mem.close()

    @pytest.mark.asyncio
    async def test_tracing_strict_tenant_isolation(self) -> None:
        tracer = InMemoryTracer()
        extractor = _MockFactExtractor([{"content": "Secret note for User A", "entity": "UserA"}])

        mem = await MemoryLayer.create_local(
            vector_dim=16,
            extractor=extractor,  # type: ignore[arg-type]
            tracer=tracer,
        )

        try:
            await mem.add("user_A", [{"role": "user", "content": "My secret note."}])
            tracer.clear()

            # User B searches
            res = await mem.search("user_B", "secret note", bypass_router=True)
            assert len(res.memories) == 0

            traces = tracer.find_traces("memory_search")
            assert len(traces) == 1
            root = traces[0]
            assert root.attributes["user_id"] == "user_B"
            assert root.output["retrieved_count"] == 0
        finally:
            await mem.close()

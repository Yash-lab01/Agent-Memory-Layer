"""Observability and distributed tracing for Agent Memory Layer.

Provides unified telemetry support across Langfuse, OpenTelemetry, and in-memory trace recorders.
Gracefully degrades to zero-overhead no-op when tracing is unconfigured.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

# Try optional imports
try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment,misc]


class SpanContext(
    AbstractContextManager["SpanContext"], AbstractAsyncContextManager["SpanContext"], ABC
):
    """Abstract context manager representing a trace or child span."""

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> SpanContext:
        """Set a single metadata attribute on the active span."""
        ...

    @abstractmethod
    def set_output(self, output: Any) -> SpanContext:
        """Record output data on the active span."""
        ...

    @abstractmethod
    def end(self, status: str = "ok", error: Exception | None = None) -> None:
        """Mark span completion with execution status and optional error."""
        ...

    def __enter__(self) -> SpanContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is not None:
            self.end(status="error", error=exc_val if isinstance(exc_val, Exception) else None)
        else:
            self.end(status="ok")

    async def __aenter__(self) -> SpanContext:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is not None:
            self.end(status="error", error=exc_val if isinstance(exc_val, Exception) else None)
        else:
            self.end(status="ok")


class NullSpan(SpanContext):
    """Zero-overhead no-op span for when tracing is disabled."""

    def set_attribute(self, _key: str, _value: Any) -> SpanContext:
        return self

    def set_output(self, _output: Any) -> SpanContext:
        return self

    def end(self, _status: str = "ok", _error: Exception | None = None) -> None:
        pass


class RecordedSpan(SpanContext):
    """In-memory trace span capturing timings, metadata attributes, and output payloads."""

    def __init__(
        self,
        name: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.parent_id = parent_id
        self.start_time: float = time.perf_counter()
        self.end_time: float | None = None
        self.duration_ms: float = 0.0
        self.attributes: dict[str, Any] = dict(metadata or {})
        self.output: Any = None
        self.status: str = "running"
        self.error: str | None = None
        self.children: list[RecordedSpan] = []

    def set_attribute(self, key: str, value: Any) -> SpanContext:
        self.attributes[key] = value
        return self

    def set_output(self, output: Any) -> SpanContext:
        self.output = output
        return self

    def end(self, status: str = "ok", error: Exception | None = None) -> None:
        if self.end_time is None:
            self.end_time = time.perf_counter()
            self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
            self.status = status
            if error is not None:
                self.error = str(error)


class BaseTracer(ABC):
    """Abstract telemetry tracer interface for Agent Memory Layer."""

    @abstractmethod
    def trace(
        self,
        name: str,
        user_id: str,
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        """Start a top-level execution trace scoped by user_id."""
        ...

    @abstractmethod
    def span(
        self,
        name: str,
        parent: SpanContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        """Start a child operation span under a parent trace or span."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending telemetry payloads to remote storage."""
        ...


class NullTracer(BaseTracer):
    """Default no-op tracer running with zero overhead when tracing is disabled."""

    def trace(
        self,
        _name: str,
        _user_id: str,
        _session_id: str = "default",
        _metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        return NullSpan()

    def span(
        self,
        _name: str,
        _parent: SpanContext | None = None,
        _metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        return NullSpan()

    def flush(self) -> None:
        pass


class InMemoryTracer(BaseTracer):
    """In-memory telemetry tracer recording traces and spans for testing and local audits."""

    def __init__(self) -> None:
        self.traces: list[RecordedSpan] = []

    def trace(
        self,
        name: str,
        user_id: str,
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        attrs = dict(metadata or {})
        attrs["user_id"] = user_id
        attrs["session_id"] = session_id
        span = RecordedSpan(name=name, metadata=attrs)
        self.traces.append(span)
        return span

    def span(
        self,
        name: str,
        parent: SpanContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        span = RecordedSpan(name=name, metadata=metadata)
        if isinstance(parent, RecordedSpan):
            parent.children.append(span)
        return span

    def flush(self) -> None:
        pass

    def clear(self) -> None:
        """Clear recorded traces."""
        self.traces.clear()

    def find_traces(self, name: str) -> list[RecordedSpan]:
        """Find recorded traces matching a given name."""
        return [t for t in self.traces if t.name == name]


class LangfuseSpan(SpanContext):
    """Span context publishing events directly to Langfuse."""

    def __init__(self, langfuse_obj: Any) -> None:
        self.langfuse_obj = langfuse_obj

    def set_attribute(self, key: str, value: Any) -> SpanContext:
        try:
            if hasattr(self.langfuse_obj, "update"):
                self.langfuse_obj.update(metadata={key: value})
        except Exception as exc:
            logger.warning("Failed to set Langfuse attribute: %s", exc)
        return self

    def set_output(self, output: Any) -> SpanContext:
        try:
            if hasattr(self.langfuse_obj, "update"):
                self.langfuse_obj.update(output=output)
        except Exception as exc:
            logger.warning("Failed to set Langfuse output: %s", exc)
        return self

    def end(self, status: str = "ok", error: Exception | None = None) -> None:
        try:
            if hasattr(self.langfuse_obj, "end"):
                self.langfuse_obj.end(
                    status_message=str(error) if error else status,
                    level="ERROR" if status == "error" else "DEFAULT",
                )
        except Exception as exc:
            logger.warning("Failed to end Langfuse span: %s", exc)


class LangfuseTracer(BaseTracer):
    """Observability tracer publishing structured traces to a Langfuse server."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        self.public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        self.host = host or os.environ.get("LANGFUSE_HOST", "http://localhost:3000")

        self.client: Any = None
        if Langfuse is not None and self.public_key and self.secret_key:
            try:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
            except Exception as exc:
                logger.warning("Could not initialize Langfuse client: %s", exc)
                self.client = None

    def trace(
        self,
        name: str,
        user_id: str,
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        if self.client is None:
            return NullSpan()

        try:
            lf_trace = self.client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
            return LangfuseSpan(lf_trace)
        except Exception as exc:
            logger.warning("Langfuse trace creation failed: %s", exc)
            return NullSpan()

    def span(
        self,
        name: str,
        parent: SpanContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        if self.client is None:
            return NullSpan()

        try:
            parent_obj = getattr(parent, "langfuse_obj", None)
            if parent_obj is not None and hasattr(parent_obj, "span"):
                lf_span = parent_obj.span(name=name, metadata=metadata or {})
                return LangfuseSpan(lf_span)
            lf_span = self.client.span(name=name, metadata=metadata or {})
            return LangfuseSpan(lf_span)
        except Exception as exc:
            logger.warning("Langfuse span creation failed: %s", exc)
            return NullSpan()

    def flush(self) -> None:
        if self.client is not None and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception as exc:
                logger.warning("Langfuse flush warning: %s", exc)


def get_tracer(tracer_type: str | None = None) -> BaseTracer:
    """Factory creating configured tracer based on environment or explicit type.

    Args:
        tracer_type: Optional explicit tracer type ('langfuse', 'memory', 'null').

    Returns:
        Configured BaseTracer instance.
    """
    selected = (
        tracer_type
        or os.environ.get("MEMLAYER_TRACER")
        or ("langfuse" if os.environ.get("LANGFUSE_PUBLIC_KEY") else "null")
    ).lower()

    if selected == "langfuse":
        return LangfuseTracer()
    if selected == "memory":
        return InMemoryTracer()
    return NullTracer()

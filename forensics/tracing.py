"""Tracing layer: Trace and Span objects plus the @traced decorator.

Usage:
    @traced("extraction")
    def extract(doc: IntakeResult, llm: BaseLLMClient) -> ExtractedEntities: ...

    with start_trace("document_pipeline", trace_input=raw_doc) as trace:
        extract(doc, llm=llm)

Any LLM call made while a span is open is attached to that span automatically
(see record_llm_call), so step functions never touch the tracer directly.
IDs follow OpenTelemetry's shape (32-hex trace id, 16-hex span id) so these
spans can be exported to an OTel backend later without reshaping them.
"""

from __future__ import annotations

import functools
import inspect
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from pydantic import BaseModel, Field

LOW_CONFIDENCE = 2  # spans at or below this mark the trace as degraded


def _now() -> datetime:
    return datetime.now(timezone.utc)


def to_jsonable(value: Any) -> Any:
    """Turn step inputs/outputs into plain JSON-safe data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class LLMCall(BaseModel):
    provider: str
    model: str
    messages: list[dict[str, str]]
    response_schema: str
    raw_response: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    error: str | None = None


class SpanError(BaseModel):
    type: str
    message: str
    traceback: str


class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str
    parent_span_id: str | None = None
    name: str
    status: Literal["ok", "error"] = "ok"
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    latency_ms: float | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    llm_calls: list[LLMCall] = Field(default_factory=list)
    confidence: int | None = None
    confidence_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: SpanError | None = None

    @property
    def tokens(self) -> int:
        return sum((c.input_tokens or 0) + (c.output_tokens or 0) for c in self.llm_calls)

    def record_output(self, result: Any) -> None:
        """Store the step's output and lift out confidence/warnings if the output has them."""
        self.output = to_jsonable(result)
        confidence = getattr(result, "confidence", None)
        if isinstance(confidence, int):
            self.confidence = confidence
            self.confidence_reason = getattr(result, "confidence_reason", None)
        warnings = getattr(result, "warnings", None)
        if isinstance(warnings, list):
            self.warnings = [str(w) for w in warnings]


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    pipeline: str
    status: Literal["running", "success", "failure", "degraded"] = "running"
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    latency_ms: float | None = None
    input: Any = None
    spans: list[Span] = Field(default_factory=list)
    final_output: Any = None
    final_score: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.spans)

    @property
    def failed_step(self) -> str | None:
        return next((s.name for s in self.spans if s.status == "error"), None)

    def finalize(self) -> None:
        self.ended_at = _now()
        self.latency_ms = round((self.ended_at - self.started_at).total_seconds() * 1000, 1)

        scores = [s.confidence for s in self.spans if s.confidence is not None]
        # Placeholder until a real evaluator exists: the weakest step's confidence.
        self.final_score = float(min(scores)) if scores else None

        if self.error or any(s.status == "error" for s in self.spans):
            self.status = "failure"
        elif any(s.warnings for s in self.spans) or any(c <= LOW_CONFIDENCE for c in scores):
            self.status = "degraded"
        else:
            self.status = "success"


_current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


def current_trace() -> Trace | None:
    return _current_trace.get()


def record_llm_call(call: LLMCall) -> None:
    """Called by the LLM client after every request; attaches it to the open span."""
    span = _current_span.get()
    if span is not None:
        span.llm_calls.append(call)


@contextmanager
def start_trace(pipeline: str, trace_input: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[Trace]:
    trace = Trace(pipeline=pipeline, input=to_jsonable(trace_input), metadata=metadata or {})
    token = _current_trace.set(trace)
    try:
        yield trace
    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace.finalize()
        _current_trace.reset(token)


@contextmanager
def span(name: str, span_input: dict[str, Any] | None = None) -> Iterator[Span]:
    trace = _current_trace.get()
    parent = _current_span.get()
    s = Span(
        trace_id=trace.trace_id if trace else "untraced",
        parent_span_id=parent.span_id if parent else None,
        name=name,
        input=span_input or {},
    )
    if trace is not None:
        trace.spans.append(s)

    token = _current_span.set(s)
    start = time.perf_counter()
    try:
        yield s
    except Exception as exc:
        s.status = "error"
        s.error = SpanError(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        s.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        s.ended_at = _now()
        _current_span.reset(token)


def traced(name: str | None = None, ignore_args: tuple[str, ...] = ("llm",)) -> Callable:
    """Decorator that wraps a step function in a span.

    Captures the step's arguments as input (skipping `ignore_args`, e.g. the LLM
    client), its return value as output, and any exception as an error.
    """

    def decorator(fn: Callable) -> Callable:
        step_name = name or fn.__name__
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            step_input = {k: to_jsonable(v) for k, v in bound.arguments.items() if k not in ignore_args}
            with span(step_name, step_input) as s:
                result = fn(*args, **kwargs)
                s.record_output(result)
                return result

        return wrapper

    return decorator

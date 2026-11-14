"""LLM clients. Every call is timed and recorded on the current span.

OpenAILLMClient  - real calls with structured outputs (Pydantic schemas).
MockLLMClient    - free, offline, deterministic fake outputs for testing the plumbing.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
import types
from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

from .tracing import LLMCall, record_llm_call

T = TypeVar("T", bound=BaseModel)

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class LLMOutputError(RuntimeError):
    """The model refused or returned nothing usable."""


class BaseLLMClient(ABC):
    provider: str = "base"
    model: str = "unknown"

    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        call = LLMCall(provider=self.provider, model=self.model, messages=messages, response_schema=schema.__name__)
        start = time.perf_counter()
        try:
            parsed, raw, input_tokens, output_tokens, model_used = self._complete(messages, schema)
        except Exception as exc:
            call.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            call.raw_response = raw
            call.input_tokens = input_tokens
            call.output_tokens = output_tokens
            call.model = model_used
            return parsed
        finally:
            call.latency_ms = round((time.perf_counter() - start) * 1000, 1)
            record_llm_call(call)

    @abstractmethod
    def _complete(self, messages: list[dict[str, str]], schema: type[T]) -> tuple[T, str, int | None, int | None, str]:
        """Return (parsed, raw_text, input_tokens, output_tokens, model_used)."""


class OpenAILLMClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self._client = OpenAI()  # reads OPENAI_API_KEY

    def _complete(self, messages, schema):
        completion = self._client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema,
        )
        choice = completion.choices[0]
        message = choice.message
        if message.refusal:
            raise LLMOutputError(f"Model refused: {message.refusal}")
        if message.parsed is None:
            raise LLMOutputError(f"No parsed output (finish_reason={choice.finish_reason})")
        usage = completion.usage
        return (
            message.parsed,
            message.content or "",
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            completion.model,
        )


class MockLLMClient(BaseLLMClient):
    """Fills any schema with fake but valid values, seeded by the prompt so runs repeat."""

    provider = "mock"
    model = "mock"

    def _complete(self, messages, schema):
        prompt = "\n".join(m["content"] for m in messages)
        rng = random.Random(int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 2**32)
        parsed = _fake_model(schema, rng)
        return parsed, parsed.model_dump_json(), len(prompt) // 4, 150, self.model


def _fake_model(model: type[T], rng: random.Random) -> T:
    return model(**{name: _fake_value(f.annotation, name, rng) for name, f in model.model_fields.items()})


def _fake_value(annotation: Any, name: str, rng: random.Random) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _fake_value(args[0], name, rng)
    if origin is Literal:
        return rng.choice(args)
    if origin in (Union, types.UnionType):
        return _fake_value(next(a for a in args if a is not type(None)), name, rng)
    if origin is list:
        return [_fake_value(args[0], name, rng) for _ in range(rng.randint(1, 2))]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _fake_model(annotation, rng)
    if annotation is str:
        return f"mock {name}"
    if annotation is int:
        return rng.randint(1, 10)
    if annotation is float:
        return round(rng.uniform(10, 5000), 2)
    if annotation is bool:
        return rng.random() < 0.5
    raise TypeError(f"MockLLMClient can't fake {annotation!r} for field '{name}'")

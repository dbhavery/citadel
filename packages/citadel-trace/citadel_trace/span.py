"""Trace and Span data models for LLM observability."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Span:
    """Represents a single unit of work (e.g., one LLM call, one tool execution)."""

    id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    kind: str  # "llm", "tool", "agent", "chain"
    model: Optional[str]
    provider: Optional[str]  # "anthropic", "openai", "google", "ollama"

    # Timing
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None

    # LLM-specific
    input_messages: Optional[list[dict]] = None
    output_content: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Cost
    cost_usd: Optional[float] = None

    # Status
    status: str = "ok"
    error_message: Optional[str] = None

    # Metadata
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def finish(self, end_time: Optional[float] = None) -> None:
        """Mark this span as finished, computing duration."""
        self.end_time = end_time or time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0

    def set_tokens(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """Set token counts and compute total."""
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens
        inp = self.input_tokens or 0
        out = self.output_tokens or 0
        self.total_tokens = inp + out

    @staticmethod
    def new(
        name: str,
        kind: str = "llm",
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ) -> "Span":
        """Create a new span with generated IDs and current timestamp."""
        return Span(
            id=uuid.uuid4().hex,
            trace_id=trace_id or uuid.uuid4().hex,
            parent_id=parent_id,
            name=name,
            kind=kind,
            model=model,
            provider=provider,
            start_time=time.time(),
            **kwargs,
        )


@dataclass
class Trace:
    """A collection of related spans forming a single logical operation."""

    id: str
    spans: list[Span]
    start_time: float
    end_time: Optional[float]
    total_cost_usd: float
    total_tokens: int
    root_span: Optional[Span]

    @staticmethod
    def from_spans(trace_id: str, spans: list[Span]) -> "Trace":
        """Build a Trace from a list of spans."""
        if not spans:
            return Trace(
                id=trace_id,
                spans=[],
                start_time=0.0,
                end_time=None,
                total_cost_usd=0.0,
                total_tokens=0,
                root_span=None,
            )

        sorted_spans = sorted(spans, key=lambda s: s.start_time)
        start_time = sorted_spans[0].start_time

        end_times = [s.end_time for s in sorted_spans if s.end_time is not None]
        end_time = max(end_times) if end_times else None

        total_cost = sum(s.cost_usd for s in spans if s.cost_usd is not None)
        total_tokens = sum(s.total_tokens for s in spans if s.total_tokens is not None)

        root_span = None
        for s in sorted_spans:
            if s.parent_id is None:
                root_span = s
                break

        return Trace(
            id=trace_id,
            spans=sorted_spans,
            start_time=start_time,
            end_time=end_time,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            root_span=root_span,
        )

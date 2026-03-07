"""Tests for Span and Trace data models."""

import time

from citadel_trace.span import Span, Trace


class TestSpanCreation:
    """Test span creation with all fields."""

    def test_span_creation_all_fields(self) -> None:
        """Span can be created with all fields populated."""
        span = Span(
            id="span-001",
            trace_id="trace-001",
            parent_id="span-000",
            name="chat.completion",
            kind="llm",
            model="claude-sonnet-4-6",
            provider="anthropic",
            start_time=1000.0,
            end_time=1002.5,
            duration_ms=2500.0,
            input_messages=[{"role": "user", "content": "Hello"}],
            output_content="Hi there!",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.000105,
            status="ok",
            error_message=None,
            metadata={"session": "abc"},
            tags=["test", "v1"],
        )

        assert span.id == "span-001"
        assert span.trace_id == "trace-001"
        assert span.parent_id == "span-000"
        assert span.name == "chat.completion"
        assert span.kind == "llm"
        assert span.model == "claude-sonnet-4-6"
        assert span.provider == "anthropic"
        assert span.start_time == 1000.0
        assert span.end_time == 1002.5
        assert span.duration_ms == 2500.0
        assert span.input_tokens == 10
        assert span.output_tokens == 5
        assert span.total_tokens == 15
        assert span.cost_usd == 0.000105
        assert span.status == "ok"
        assert span.metadata == {"session": "abc"}
        assert span.tags == ["test", "v1"]

    def test_span_duration_calculation(self) -> None:
        """Calling finish() computes duration_ms from start and end times."""
        span = Span.new(name="test.op", kind="tool")
        # Override start_time to a known value
        span.start_time = 1000.0

        span.finish(end_time=1000.150)

        assert span.end_time == 1000.150
        assert span.duration_ms is not None
        assert abs(span.duration_ms - 150.0) < 0.01

    def test_span_set_tokens(self) -> None:
        """set_tokens() computes total from input + output."""
        span = Span.new(name="test.op")
        span.set_tokens(input_tokens=100, output_tokens=50)

        assert span.input_tokens == 100
        assert span.output_tokens == 50
        assert span.total_tokens == 150


class TestTraceAggregation:
    """Test Trace.from_spans aggregation logic."""

    def test_trace_aggregation(self) -> None:
        """Trace correctly aggregates cost and tokens from its spans."""
        spans = [
            Span(
                id="s1",
                trace_id="t1",
                parent_id=None,
                name="call-1",
                kind="llm",
                model="claude-sonnet-4-6",
                provider="anthropic",
                start_time=1000.0,
                end_time=1001.0,
                duration_ms=1000.0,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_usd=0.001050,
            ),
            Span(
                id="s2",
                trace_id="t1",
                parent_id="s1",
                name="call-2",
                kind="tool",
                model="gpt-4o-mini",
                provider="openai",
                start_time=1001.0,
                end_time=1002.0,
                duration_ms=1000.0,
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
                cost_usd=0.000090,
            ),
        ]

        trace = Trace.from_spans("t1", spans)

        assert trace.id == "t1"
        assert len(trace.spans) == 2
        assert trace.start_time == 1000.0
        assert trace.end_time == 1002.0
        assert trace.total_tokens == 450
        assert abs(trace.total_cost_usd - 0.001140) < 1e-9
        assert trace.root_span is not None
        assert trace.root_span.id == "s1"

    def test_empty_trace(self) -> None:
        """Trace.from_spans handles empty span list gracefully."""
        trace = Trace.from_spans("t-empty", [])

        assert trace.id == "t-empty"
        assert trace.spans == []
        assert trace.total_cost_usd == 0.0
        assert trace.total_tokens == 0
        assert trace.root_span is None

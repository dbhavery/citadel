"""Tests for TraceCollector SQLite storage."""

import os
import tempfile
import time

import pytest

from citadel_trace.collector import TraceCollector
from citadel_trace.span import Span


@pytest.fixture
def collector():
    """Create a TraceCollector with a temporary database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    c = TraceCollector(db_path=path)
    yield c
    c.close()
    os.unlink(path)


class TestRecordAndRetrieve:
    """Test basic span recording and retrieval."""

    def test_record_and_retrieve_span(self, collector: TraceCollector) -> None:
        """A recorded span can be retrieved by its trace_id."""
        span = Span(
            id="span-100",
            trace_id="trace-100",
            parent_id=None,
            name="chat.completion",
            kind="llm",
            model="claude-sonnet-4-6",
            provider="anthropic",
            start_time=time.time(),
            end_time=time.time() + 1.0,
            duration_ms=1000.0,
            input_messages=[{"role": "user", "content": "test"}],
            output_content="response text",
            input_tokens=50,
            output_tokens=20,
            total_tokens=70,
            cost_usd=0.000450,
            status="ok",
            metadata={"env": "test"},
            tags=["unit-test"],
        )
        collector.record_span(span)

        trace = collector.get_trace("trace-100")
        assert len(trace.spans) == 1

        retrieved = trace.spans[0]
        assert retrieved.id == "span-100"
        assert retrieved.model == "claude-sonnet-4-6"
        assert retrieved.input_tokens == 50
        assert retrieved.output_tokens == 20
        assert retrieved.cost_usd == 0.000450
        assert retrieved.metadata == {"env": "test"}
        assert retrieved.tags == ["unit-test"]

    def test_start_and_end_span(self, collector: TraceCollector) -> None:
        """start_span + end_span creates a complete span with auto-cost."""
        span = collector.start_span(
            name="completion",
            kind="llm",
            model="gpt-4o",
            provider="openai",
        )

        collector.end_span(
            span,
            output="Hello world",
            tokens={"input": 1000, "output": 500},
        )

        trace = collector.get_trace(span.trace_id)
        assert len(trace.spans) == 1

        s = trace.spans[0]
        assert s.output_content == "Hello world"
        assert s.input_tokens == 1000
        assert s.output_tokens == 500
        assert s.total_tokens == 1500
        assert s.duration_ms is not None
        assert s.duration_ms > 0
        # gpt-4o: 2.50/1M in + 10.0/1M out
        expected_cost = (1000 * 2.50 / 1_000_000) + (500 * 10.0 / 1_000_000)
        assert s.cost_usd is not None
        assert abs(s.cost_usd - expected_cost) < 1e-9


class TestListTraces:
    """Test trace listing functionality."""

    def test_list_traces_with_limit(self, collector: TraceCollector) -> None:
        """list_traces respects the limit parameter."""
        now = time.time()
        for i in range(5):
            span = Span(
                id=f"s-{i}",
                trace_id=f"t-{i}",
                parent_id=None,
                name="op",
                kind="llm",
                model="gpt-4o-mini",
                provider="openai",
                start_time=now + i,
                end_time=now + i + 0.5,
                duration_ms=500.0,
            )
            collector.record_span(span)

        traces = collector.list_traces(limit=3)
        assert len(traces) == 3


class TestSearchTraces:
    """Test trace search/filter functionality."""

    def test_search_by_model(self, collector: TraceCollector) -> None:
        """search_traces filters by model name."""
        now = time.time()

        span_a = Span(
            id="sa",
            trace_id="ta",
            parent_id=None,
            name="op",
            kind="llm",
            model="claude-sonnet-4-6",
            provider="anthropic",
            start_time=now,
        )
        span_b = Span(
            id="sb",
            trace_id="tb",
            parent_id=None,
            name="op",
            kind="llm",
            model="gpt-4o",
            provider="openai",
            start_time=now + 1,
        )
        collector.record_span(span_a)
        collector.record_span(span_b)

        results = collector.search_traces(model="claude-sonnet-4-6")
        assert len(results) == 1
        assert results[0].id == "ta"

    def test_search_by_status(self, collector: TraceCollector) -> None:
        """search_traces filters by span status."""
        now = time.time()

        span_ok = Span(
            id="sok",
            trace_id="tok",
            parent_id=None,
            name="op",
            kind="llm",
            model="gpt-4o",
            provider="openai",
            start_time=now,
            status="ok",
        )
        span_err = Span(
            id="serr",
            trace_id="terr",
            parent_id=None,
            name="op",
            kind="llm",
            model="gpt-4o",
            provider="openai",
            start_time=now + 1,
            status="error",
            error_message="timeout",
        )
        collector.record_span(span_ok)
        collector.record_span(span_err)

        errors = collector.search_traces(status="error")
        assert len(errors) == 1
        assert errors[0].id == "terr"

        ok_traces = collector.search_traces(status="ok")
        assert len(ok_traces) == 1
        assert ok_traces[0].id == "tok"

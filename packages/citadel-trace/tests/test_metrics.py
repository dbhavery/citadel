"""Tests for MetricsCalculator."""

import os
import tempfile
import time

import pytest

from citadel_trace.collector import TraceCollector
from citadel_trace.metrics import MetricsCalculator
from citadel_trace.span import Span


@pytest.fixture
def populated_collector():
    """Create a TraceCollector with pre-loaded test spans."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    collector = TraceCollector(db_path=path)

    now = time.time()

    # Span 1: claude-sonnet, ok, fast
    s1 = Span(
        id="m1",
        trace_id="mt1",
        parent_id=None,
        name="chat.completion",
        kind="llm",
        model="claude-sonnet-4-6",
        provider="anthropic",
        start_time=now - 100,
        end_time=now - 99.5,
        duration_ms=500.0,
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        cost_usd=0.010500,
        status="ok",
    )

    # Span 2: gpt-4o, ok, medium
    s2 = Span(
        id="m2",
        trace_id="mt2",
        parent_id=None,
        name="chat.completion",
        kind="llm",
        model="gpt-4o",
        provider="openai",
        start_time=now - 90,
        end_time=now - 89.0,
        duration_ms=1000.0,
        input_tokens=2000,
        output_tokens=800,
        total_tokens=2800,
        cost_usd=0.013000,
        status="ok",
    )

    # Span 3: gpt-4o, error, slow
    s3 = Span(
        id="m3",
        trace_id="mt3",
        parent_id=None,
        name="chat.completion",
        kind="llm",
        model="gpt-4o",
        provider="openai",
        start_time=now - 80,
        end_time=now - 77.0,
        duration_ms=3000.0,
        input_tokens=500,
        output_tokens=0,
        total_tokens=500,
        cost_usd=0.001250,
        status="error",
        error_message="rate_limit_exceeded",
    )

    # Span 4: ollama, ok, fast
    s4 = Span(
        id="m4",
        trace_id="mt4",
        parent_id=None,
        name="ollama.chat",
        kind="llm",
        model="ollama/qwen3:8b",
        provider="ollama",
        start_time=now - 70,
        end_time=now - 69.8,
        duration_ms=200.0,
        input_tokens=800,
        output_tokens=400,
        total_tokens=1200,
        cost_usd=0.0,
        status="ok",
    )

    for s in [s1, s2, s3, s4]:
        collector.record_span(s)

    yield collector

    collector.close()
    os.unlink(path)


class TestCostSummary:
    """Test cost summary calculations."""

    def test_cost_summary(self, populated_collector: TraceCollector) -> None:
        """cost_summary returns correct total and per-model breakdown."""
        metrics = MetricsCalculator(populated_collector)
        summary = metrics.cost_summary(days=1)

        # Total cost = 0.010500 + 0.013000 + 0.001250 + 0.0 = 0.024750
        assert abs(summary["total_cost_usd"] - 0.024750) < 1e-5
        assert summary["span_count"] == 4
        assert "claude-sonnet-4-6" in summary["cost_by_model"]
        assert "gpt-4o" in summary["cost_by_model"]
        assert summary["cost_by_model"]["ollama/qwen3:8b"] == 0.0


class TestLatencyPercentiles:
    """Test latency percentile calculations."""

    def test_latency_percentiles(self, populated_collector: TraceCollector) -> None:
        """latency_percentiles returns correct p50, p95, p99 values."""
        metrics = MetricsCalculator(populated_collector)
        latency = metrics.latency_percentiles()

        assert latency["count"] == 4
        # Sorted durations: [200, 500, 1000, 3000]
        # p50 = median, should be between 500 and 1000
        assert 500.0 <= latency["p50"] <= 1000.0
        # p99 should be close to 3000
        assert latency["p99"] >= 2000.0
        # Mean = (200 + 500 + 1000 + 3000) / 4 = 1175
        assert abs(latency["mean"] - 1175.0) < 0.01


class TestModelComparison:
    """Test model comparison."""

    def test_model_comparison(self, populated_collector: TraceCollector) -> None:
        """model_comparison returns one entry per model with correct stats."""
        metrics = MetricsCalculator(populated_collector)
        comparison = metrics.model_comparison()

        models = {entry["model"]: entry for entry in comparison}

        assert len(models) == 3  # claude-sonnet, gpt-4o, ollama/qwen3

        # claude-sonnet: 1 call, $0.010500
        claude = models["claude-sonnet-4-6"]
        assert claude["call_count"] == 1
        assert abs(claude["total_cost_usd"] - 0.010500) < 1e-5
        assert claude["error_count"] == 0

        # gpt-4o: 2 calls, 1 error
        gpt = models["gpt-4o"]
        assert gpt["call_count"] == 2
        assert gpt["error_count"] == 1
        assert gpt["error_rate"] == 0.5

        # ollama: free
        ollama = models["ollama/qwen3:8b"]
        assert ollama["total_cost_usd"] == 0.0
        assert ollama["error_count"] == 0

"""Tests for AlertManager."""

import os
import tempfile
import time

import pytest

from citadel_trace.alerts import AlertManager, AlertRule
from citadel_trace.collector import TraceCollector
from citadel_trace.span import Span


@pytest.fixture
def collector_with_data():
    """Create a collector with known cost data for alert testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    collector = TraceCollector(db_path=path)

    now = time.time()

    # Create spans with known cost totaling $0.50 today
    for i in range(10):
        span = Span(
            id=f"alert-s{i}",
            trace_id=f"alert-t{i}",
            parent_id=None,
            name="chat.completion",
            kind="llm",
            model="claude-sonnet-4-6",
            provider="anthropic",
            start_time=now - (i * 10),
            end_time=now - (i * 10) + 0.5,
            duration_ms=500.0,
            input_tokens=5000,
            output_tokens=2000,
            total_tokens=7000,
            # Each span costs (5000*3.0/1M) + (2000*15.0/1M) = 0.015 + 0.030 = 0.045
            cost_usd=0.045,
            status="ok" if i < 9 else "error",
            error_message="test error" if i >= 9 else None,
        )
        collector.record_span(span)

    yield collector

    collector.close()
    os.unlink(path)


class TestAlertTriggering:
    """Test alert rule evaluation."""

    def test_alert_triggers_when_threshold_exceeded(
        self, collector_with_data: TraceCollector
    ) -> None:
        """Alert fires when the metric exceeds the threshold."""
        manager = AlertManager(collector_with_data)

        # Total daily cost = 10 * $0.045 = $0.45
        # Set threshold below actual cost so it triggers
        manager.add_rule(
            AlertRule(name="Cost too high", metric="daily_cost", threshold=0.10)
        )

        triggered = manager.check_rules()
        assert len(triggered) == 1
        assert triggered[0]["rule_name"] == "Cost too high"
        assert triggered[0]["metric"] == "daily_cost"
        assert triggered[0]["current_value"] > 0.10
        assert "triggered_at" in triggered[0]

    def test_alert_does_not_trigger_when_under_threshold(
        self, collector_with_data: TraceCollector
    ) -> None:
        """Alert does not fire when the metric is below the threshold."""
        manager = AlertManager(collector_with_data)

        # Total daily cost = $0.45, so $100 threshold should NOT trigger
        manager.add_rule(
            AlertRule(name="Budget limit", metric="daily_cost", threshold=100.0)
        )

        triggered = manager.check_rules()
        assert len(triggered) == 0

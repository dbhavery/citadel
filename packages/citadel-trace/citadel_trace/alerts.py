"""Alerting system for LLM observability thresholds."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .collector import TraceCollector
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """A rule that triggers when a metric exceeds a threshold.

    Attributes:
        name: Human-readable name for this alert.
        metric: The metric to check - "daily_cost", "error_rate", or "p99_latency".
        threshold: The threshold value that triggers the alert.
        webhook_url: Optional URL to POST alert data to.
    """

    name: str
    metric: str  # "daily_cost", "error_rate", "p99_latency"
    threshold: float
    webhook_url: Optional[str] = None


class AlertManager:
    """Manages alert rules and checks them against current metrics."""

    def __init__(self, collector: TraceCollector) -> None:
        self.rules: list[AlertRule] = []
        self.collector = collector
        self._metrics = MetricsCalculator(collector)

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: The AlertRule to add.
        """
        self.rules.append(rule)

    def check_rules(self) -> list[dict]:
        """Check all rules and return triggered alerts.

        Returns:
            List of dicts describing each triggered alert, with keys:
            - rule_name, metric, threshold, current_value, triggered_at.
        """
        triggered: list[dict] = []

        for rule in self.rules:
            current_value = self._get_metric_value(rule.metric)
            if current_value is not None and current_value > rule.threshold:
                alert_data = {
                    "rule_name": rule.name,
                    "metric": rule.metric,
                    "threshold": rule.threshold,
                    "current_value": round(current_value, 6),
                    "triggered_at": time.time(),
                }
                triggered.append(alert_data)

                if rule.webhook_url:
                    self._send_webhook(rule.webhook_url, alert_data)

        return triggered

    def _get_metric_value(self, metric: str) -> Optional[float]:
        """Get the current value for a metric name.

        Args:
            metric: One of "daily_cost", "error_rate", "p99_latency".

        Returns:
            The current metric value, or None if unknown.
        """
        if metric == "daily_cost":
            return self._check_daily_cost()
        elif metric == "error_rate":
            return self._check_error_rate()
        elif metric == "p99_latency":
            return self._check_latency()
        else:
            logger.warning("Unknown alert metric: %s", metric)
            return None

    def _check_daily_cost(self) -> float:
        """Get today's total cost in USD."""
        summary = self._metrics.cost_summary(days=1)
        return summary["total_cost_usd"]

    def _check_error_rate(self) -> float:
        """Get the error rate over the last day."""
        errors = self._metrics.error_rate(days=1)
        return errors["error_rate"]

    def _check_latency(self) -> float:
        """Get the p99 latency in ms."""
        latency = self._metrics.latency_percentiles()
        return latency["p99"]

    def _send_webhook(self, url: str, data: dict) -> None:
        """POST alert data to a webhook URL.

        Args:
            url: The webhook URL.
            data: The alert data to send as JSON.
        """
        try:
            import httpx  # type: ignore[import-untyped]
            with httpx.Client(timeout=10.0) as client:
                client.post(url, json=data)
            logger.info("Alert webhook sent to %s", url)
        except ImportError:
            logger.warning("httpx not installed, cannot send webhook to %s", url)
        except Exception as exc:
            logger.error("Failed to send alert webhook to %s: %s", url, exc)

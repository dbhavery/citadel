"""Aggregated metrics computed from stored spans."""

from __future__ import annotations

import time
from typing import Optional

from .collector import TraceCollector


class MetricsCalculator:
    """Compute aggregate metrics from spans."""

    def __init__(self, collector: TraceCollector) -> None:
        self.collector = collector

    def cost_summary(self, days: int = 7) -> dict:
        """Total cost, cost per day, cost per model.

        Args:
            days: Number of days to look back.

        Returns:
            Dict with total_cost, daily_average, and per_model breakdown.
        """
        since = time.time() - (days * 86400)
        spans = self.collector.get_spans(since=since)

        total_cost = 0.0
        cost_by_model: dict[str, float] = {}
        cost_by_day: dict[str, float] = {}

        for span in spans:
            cost = span.cost_usd or 0.0
            total_cost += cost

            model = span.model or "unknown"
            cost_by_model[model] = cost_by_model.get(model, 0.0) + cost

            day_key = time.strftime("%Y-%m-%d", time.localtime(span.start_time))
            cost_by_day[day_key] = cost_by_day.get(day_key, 0.0) + cost

        daily_average = total_cost / max(days, 1)

        return {
            "total_cost_usd": round(total_cost, 6),
            "daily_average_usd": round(daily_average, 6),
            "cost_by_model": {k: round(v, 6) for k, v in cost_by_model.items()},
            "cost_by_day": {k: round(v, 6) for k, v in cost_by_day.items()},
            "days": days,
            "span_count": len(spans),
        }

    def latency_percentiles(self, model: Optional[str] = None) -> dict:
        """p50, p95, p99 latency in ms.

        Args:
            model: Optional model filter.

        Returns:
            Dict with p50, p95, p99, mean, and count.
        """
        spans = self.collector.get_spans(model=model)
        durations = sorted(
            s.duration_ms for s in spans if s.duration_ms is not None
        )

        if not durations:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}

        return {
            "p50": round(_percentile(durations, 50), 2),
            "p95": round(_percentile(durations, 95), 2),
            "p99": round(_percentile(durations, 99), 2),
            "mean": round(sum(durations) / len(durations), 2),
            "count": len(durations),
        }

    def token_usage(self, days: int = 7) -> dict:
        """Total tokens, tokens per model, input vs output ratio.

        Args:
            days: Number of days to look back.

        Returns:
            Dict with total, input, output tokens, and per-model breakdown.
        """
        since = time.time() - (days * 86400)
        spans = self.collector.get_spans(since=since)

        total_input = 0
        total_output = 0
        tokens_by_model: dict[str, dict[str, int]] = {}

        for span in spans:
            inp = span.input_tokens or 0
            out = span.output_tokens or 0
            total_input += inp
            total_output += out

            model = span.model or "unknown"
            if model not in tokens_by_model:
                tokens_by_model[model] = {"input": 0, "output": 0, "total": 0}
            tokens_by_model[model]["input"] += inp
            tokens_by_model[model]["output"] += out
            tokens_by_model[model]["total"] += inp + out

        total = total_input + total_output
        ratio = round(total_input / total_output, 2) if total_output > 0 else 0.0

        return {
            "total_tokens": total,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "input_output_ratio": ratio,
            "tokens_by_model": tokens_by_model,
            "days": days,
        }

    def error_rate(self, days: int = 7) -> dict:
        """Error count, error rate, errors by model.

        Args:
            days: Number of days to look back.

        Returns:
            Dict with error_count, total_count, error_rate, and per-model errors.
        """
        since = time.time() - (days * 86400)
        spans = self.collector.get_spans(since=since)

        total_count = len(spans)
        error_count = 0
        errors_by_model: dict[str, int] = {}

        for span in spans:
            if span.status == "error":
                error_count += 1
                model = span.model or "unknown"
                errors_by_model[model] = errors_by_model.get(model, 0) + 1

        error_rate_val = (error_count / total_count) if total_count > 0 else 0.0

        return {
            "error_count": error_count,
            "total_count": total_count,
            "error_rate": round(error_rate_val, 4),
            "errors_by_model": errors_by_model,
            "days": days,
        }

    def model_comparison(self) -> list[dict]:
        """Compare models by cost, latency, error rate.

        Returns:
            List of dicts, one per model, with cost, latency, token, and error stats.
        """
        spans = self.collector.get_spans()

        models: dict[str, list] = {}
        for span in spans:
            model = span.model or "unknown"
            if model not in models:
                models[model] = []
            models[model].append(span)

        results = []
        for model, model_spans in sorted(models.items()):
            durations = [s.duration_ms for s in model_spans if s.duration_ms is not None]
            costs = [s.cost_usd for s in model_spans if s.cost_usd is not None]
            errors = sum(1 for s in model_spans if s.status == "error")
            total_tokens = sum(s.total_tokens for s in model_spans if s.total_tokens is not None)

            results.append({
                "model": model,
                "call_count": len(model_spans),
                "total_cost_usd": round(sum(costs), 6),
                "avg_latency_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
                "p95_latency_ms": round(_percentile(sorted(durations), 95), 2) if durations else 0.0,
                "total_tokens": total_tokens,
                "error_count": errors,
                "error_rate": round(errors / len(model_spans), 4) if model_spans else 0.0,
            })

        return results


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute a percentile from sorted data.

    Args:
        sorted_data: A pre-sorted list of float values.
        pct: The percentile to compute (0-100).

    Returns:
        The value at the given percentile.
    """
    if not sorted_data:
        return 0.0
    k = (pct / 100.0) * (len(sorted_data) - 1)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])

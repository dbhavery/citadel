"""REST API server for citadel-trace observability data.

Provides endpoints for traces, metrics, and alerts.
Run with: citadel-trace serve
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from .alerts import AlertManager
from .collector import TraceCollector
from .metrics import MetricsCalculator

# Lazy-load FastAPI to keep it optional
_app = None
_collector: Optional[TraceCollector] = None
_metrics: Optional[MetricsCalculator] = None
_alert_manager: Optional[AlertManager] = None


def create_app(db_path: str = "./traces.db") -> "FastAPI":
    """Create and configure the FastAPI application.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Configured FastAPI app instance.
    """
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    global _collector, _metrics, _alert_manager

    app = FastAPI(
        title="citadel-trace",
        description="LLM observability API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _collector = TraceCollector(db_path=db_path)
    _metrics = MetricsCalculator(_collector)
    _alert_manager = AlertManager(_collector)

    @app.get("/health")
    def health() -> dict:
        """Health check endpoint."""
        return {"status": "ok", "service": "citadel-trace", "version": "0.1.0"}

    @app.get("/traces")
    def list_traces(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        """List recent traces."""
        if _collector is None:
            raise RuntimeError("TraceCollector not initialized — call create_app() first")
        traces = _collector.list_traces(limit=limit)
        return [_serialize_trace(t) for t in traces]

    @app.get("/traces/{trace_id}")
    def get_trace(trace_id: str) -> dict:
        """Get a full trace with all spans."""
        if _collector is None:
            raise RuntimeError("TraceCollector not initialized — call create_app() first")
        trace = _collector.get_trace(trace_id)
        return _serialize_trace(trace)

    @app.get("/metrics/cost")
    def cost_metrics(days: int = Query(7, ge=1, le=365)) -> dict:
        """Get cost summary."""
        if _metrics is None:
            raise RuntimeError("MetricsCalculator not initialized — call create_app() first")
        return _metrics.cost_summary(days=days)

    @app.get("/metrics/latency")
    def latency_metrics(model: Optional[str] = None) -> dict:
        """Get latency percentiles."""
        if _metrics is None:
            raise RuntimeError("MetricsCalculator not initialized — call create_app() first")
        return _metrics.latency_percentiles(model=model)

    @app.get("/metrics/tokens")
    def token_metrics(days: int = Query(7, ge=1, le=365)) -> dict:
        """Get token usage summary."""
        if _metrics is None:
            raise RuntimeError("MetricsCalculator not initialized — call create_app() first")
        return _metrics.token_usage(days=days)

    @app.get("/metrics/models")
    def model_metrics() -> list[dict]:
        """Compare models by cost, latency, error rate."""
        if _metrics is None:
            raise RuntimeError("MetricsCalculator not initialized — call create_app() first")
        return _metrics.model_comparison()

    @app.get("/alerts")
    def check_alerts() -> dict:
        """Check alert rules and return any triggered alerts."""
        if _alert_manager is None:
            raise RuntimeError("AlertManager not initialized — call create_app() first")
        triggered = _alert_manager.check_rules()
        return {
            "rules_count": len(_alert_manager.rules),
            "triggered": triggered,
        }

    return app


def _serialize_trace(trace) -> dict:
    """Convert a Trace object to a JSON-serializable dict."""
    return {
        "id": trace.id,
        "start_time": trace.start_time,
        "end_time": trace.end_time,
        "total_cost_usd": trace.total_cost_usd,
        "total_tokens": trace.total_tokens,
        "span_count": len(trace.spans),
        "spans": [_serialize_span(s) for s in trace.spans],
    }


def _serialize_span(span) -> dict:
    """Convert a Span object to a JSON-serializable dict."""
    return {
        "id": span.id,
        "trace_id": span.trace_id,
        "parent_id": span.parent_id,
        "name": span.name,
        "kind": span.kind,
        "model": span.model,
        "provider": span.provider,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "duration_ms": span.duration_ms,
        "input_tokens": span.input_tokens,
        "output_tokens": span.output_tokens,
        "total_tokens": span.total_tokens,
        "cost_usd": span.cost_usd,
        "status": span.status,
        "error_message": span.error_message,
        "metadata": span.metadata,
        "tags": span.tags,
    }


def main() -> None:
    """CLI entry point for running the trace server."""
    parser = argparse.ArgumentParser(
        prog="citadel-trace",
        description="LLM observability server",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8081, help="Port to listen on (default: 8081)"
    )
    serve_parser.add_argument(
        "--db", default="./traces.db", help="Path to SQLite database (default: ./traces.db)"
    )

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn  # type: ignore[import-untyped]

        app = create_app(db_path=args.db)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

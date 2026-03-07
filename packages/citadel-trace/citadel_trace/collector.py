"""Trace collector with SQLite backend for span storage and retrieval."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Optional

from .pricing import calculate_cost
from .span import Span, Trace


class TraceCollector:
    """Collects and stores spans in a SQLite database."""

    def __init__(self, db_path: str = "./traces.db") -> None:
        self._db_path = db_path
        self._conn = self._init_db(db_path)

    def _init_db(self, db_path: str) -> sqlite3.Connection:
        """Initialize the SQLite database and create tables."""
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_id TEXT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                model TEXT,
                provider TEXT,
                start_time REAL NOT NULL,
                end_time REAL,
                duration_ms REAL,
                input_messages TEXT,
                output_content TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                status TEXT NOT NULL DEFAULT 'ok',
                error_message TEXT,
                metadata TEXT,
                tags TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_spans_model ON spans(model)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_spans_status ON spans(status)"
        )
        conn.commit()
        return conn

    def start_span(
        self,
        name: str,
        kind: str = "llm",
        parent_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> Span:
        """Create and return a new in-progress span.

        Args:
            name: Descriptive name for the span (e.g., "chat.completion").
            kind: Span kind - "llm", "tool", "agent", or "chain".
            parent_id: ID of the parent span for nested operations.
            trace_id: ID to group related spans. Auto-generated if not provided.
            **kwargs: Additional Span fields (model, provider, metadata, tags, etc.).

        Returns:
            A new Span with start_time set to now.
        """
        return Span(
            id=uuid.uuid4().hex,
            trace_id=trace_id or uuid.uuid4().hex,
            parent_id=parent_id,
            name=name,
            kind=kind,
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
            start_time=time.time(),
            input_messages=kwargs.get("input_messages"),
            metadata=kwargs.get("metadata", {}),
            tags=kwargs.get("tags", []),
        )

    def end_span(
        self,
        span: Span,
        output: Optional[str] = None,
        tokens: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Finish a span and record it to the database.

        Args:
            span: The span to finish.
            output: The output content from the LLM call.
            tokens: Dict with "input", "output" keys for token counts.
            error: Error message if the span failed.
        """
        span.finish()

        if output is not None:
            span.output_content = output

        if tokens is not None:
            span.set_tokens(
                input_tokens=tokens.get("input"),
                output_tokens=tokens.get("output"),
            )

        if error is not None:
            span.status = "error"
            span.error_message = error

        # Auto-calculate cost
        if span.model and span.input_tokens is not None and span.output_tokens is not None:
            span.cost_usd = calculate_cost(
                span.model,
                span.input_tokens,
                span.output_tokens,
            )

        self.record_span(span)

    def record_span(self, span: Span) -> None:
        """Write a completed span to the database.

        Args:
            span: A fully populated span to persist.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO spans (
                id, trace_id, parent_id, name, kind, model, provider,
                start_time, end_time, duration_ms,
                input_messages, output_content,
                input_tokens, output_tokens, total_tokens,
                cost_usd, status, error_message, metadata, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                span.id,
                span.trace_id,
                span.parent_id,
                span.name,
                span.kind,
                span.model,
                span.provider,
                span.start_time,
                span.end_time,
                span.duration_ms,
                json.dumps(span.input_messages) if span.input_messages else None,
                span.output_content,
                span.input_tokens,
                span.output_tokens,
                span.total_tokens,
                span.cost_usd,
                span.status,
                span.error_message,
                json.dumps(span.metadata),
                json.dumps(span.tags),
            ),
        )
        self._conn.commit()

    def _row_to_span(self, row: sqlite3.Row) -> Span:
        """Convert a database row to a Span object."""
        input_messages = None
        if row["input_messages"]:
            input_messages = json.loads(row["input_messages"])

        metadata = {}
        if row["metadata"]:
            metadata = json.loads(row["metadata"])

        tags: list[str] = []
        if row["tags"]:
            tags = json.loads(row["tags"])

        return Span(
            id=row["id"],
            trace_id=row["trace_id"],
            parent_id=row["parent_id"],
            name=row["name"],
            kind=row["kind"],
            model=row["model"],
            provider=row["provider"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration_ms=row["duration_ms"],
            input_messages=input_messages,
            output_content=row["output_content"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            cost_usd=row["cost_usd"],
            status=row["status"],
            error_message=row["error_message"],
            metadata=metadata,
            tags=tags,
        )

    def get_trace(self, trace_id: str) -> Trace:
        """Retrieve a full trace by its ID.

        Args:
            trace_id: The trace ID to look up.

        Returns:
            A Trace object containing all spans for this trace_id.
        """
        cursor = self._conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        )
        spans = [self._row_to_span(row) for row in cursor.fetchall()]
        return Trace.from_spans(trace_id, spans)

    def list_traces(
        self,
        limit: int = 50,
        since: Optional[float] = None,
    ) -> list[Trace]:
        """List recent traces.

        Args:
            limit: Maximum number of traces to return.
            since: Only return traces started after this Unix timestamp.

        Returns:
            List of Trace objects, most recent first.
        """
        if since is not None:
            cursor = self._conn.execute(
                """
                SELECT DISTINCT trace_id FROM spans
                WHERE start_time >= ?
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (since, limit),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT DISTINCT trace_id FROM spans
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (limit,),
            )

        trace_ids = [row["trace_id"] for row in cursor.fetchall()]
        return [self.get_trace(tid) for tid in trace_ids]

    def search_traces(
        self,
        model: Optional[str] = None,
        min_cost: Optional[float] = None,
        status: Optional[str] = None,
    ) -> list[Trace]:
        """Search traces by various filters.

        Args:
            model: Filter by model name.
            min_cost: Filter for traces with total cost >= this value.
            status: Filter by span status ("ok" or "error").

        Returns:
            List of matching Trace objects.
        """
        conditions: list[str] = []
        params: list = []

        if model is not None:
            conditions.append("model = ?")
            params.append(model)

        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        cursor = self._conn.execute(
            f"SELECT DISTINCT trace_id FROM spans {where_clause} ORDER BY start_time DESC",
            params,
        )
        trace_ids = [row["trace_id"] for row in cursor.fetchall()]
        traces = [self.get_trace(tid) for tid in trace_ids]

        if min_cost is not None:
            traces = [t for t in traces if t.total_cost_usd >= min_cost]

        return traces

    def get_spans(
        self,
        since: Optional[float] = None,
        model: Optional[str] = None,
    ) -> list[Span]:
        """Retrieve raw spans with optional filters.

        Args:
            since: Only return spans started after this Unix timestamp.
            model: Filter by model name.

        Returns:
            List of Span objects.
        """
        conditions: list[str] = []
        params: list = []

        if since is not None:
            conditions.append("start_time >= ?")
            params.append(since)

        if model is not None:
            conditions.append("model = ?")
            params.append(model)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        cursor = self._conn.execute(
            f"SELECT * FROM spans {where_clause} ORDER BY start_time DESC",
            params,
        )
        return [self._row_to_span(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

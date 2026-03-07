"""Semantic response cache backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Optional


class ResponseCache:
    """SHA-256-keyed cache for LLM responses, stored in a SQLite database."""

    def __init__(self, db_path: str = "./gateway_cache.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key       TEXT PRIMARY KEY,
                response  TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl        INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(model: str, messages: list[dict[str, Any]]) -> str:
        """Create a deterministic cache key from model + messages.

        Messages are sorted-key serialized to ensure order-independence of
        dict keys (but list order is preserved).
        """
        payload = json.dumps(
            {"model": model, "messages": messages}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return cached response dict, or ``None`` on miss / expiry."""
        row = self._conn.execute(
            "SELECT response, created_at, ttl FROM cache WHERE key = ?", (key,)
        ).fetchone()

        if row is None:
            self._misses += 1
            return None

        response_json, created_at, ttl = row
        if time.time() - created_at > ttl:
            # Expired — delete and treat as miss
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            self._misses += 1
            return None

        self._hits += 1
        return json.loads(response_json)

    def put(self, key: str, response: dict[str, Any], ttl: int = 3600) -> None:
        """Store a response in the cache (upsert)."""
        self._conn.execute(
            """
            INSERT INTO cache (key, response, created_at, ttl)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                response   = excluded.response,
                created_at = excluded.created_at,
                ttl        = excluded.ttl
            """,
            (key, json.dumps(response, ensure_ascii=False), time.time(), ttl),
        )
        self._conn.commit()

    def evict(self, key: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        cursor = self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        """Remove all entries. Returns count deleted."""
        cursor = self._conn.execute("DELETE FROM cache")
        self._conn.commit()
        return cursor.rowcount

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        # Page count * page size gives approximate DB size in bytes
        page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]
        size_bytes = page_count * page_size

        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

        return {
            "total_entries": total,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "size_bytes": size_bytes,
        }

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

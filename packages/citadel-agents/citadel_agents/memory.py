"""Agent memory systems.

Provides short-term conversation memory and optional long-term vector memory.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


class ConversationMemory:
    """Short-term conversation history.

    Stores messages as role/content dicts with automatic truncation
    when max_turns is exceeded.
    """

    def __init__(self, max_turns: int = 50) -> None:
        """Initialize conversation memory.

        Args:
            max_turns: Maximum number of messages to retain.
        """
        self.max_turns = max_turns
        self._messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: Message role (e.g., "user", "assistant", "system").
            content: Message content.
        """
        self._messages.append({"role": role, "content": content})
        # Truncate if over limit
        if len(self._messages) > self.max_turns:
            self._messages = self._messages[-self.max_turns:]

    def get_messages(self) -> list[dict[str, str]]:
        """Return all stored messages."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear all stored messages."""
        self._messages.clear()

    def summarize(self, keep_last: int = 10) -> str:
        """Summarize older messages, keeping the most recent ones.

        This is a simple summarization that concatenates older messages
        into a summary string. For LLM-based summarization, the caller
        should use the LLM client directly.

        Args:
            keep_last: Number of recent messages to preserve intact.

        Returns:
            A summary string of the older messages.
        """
        if len(self._messages) <= keep_last:
            return ""

        older = self._messages[:-keep_last]
        summary_parts = []
        for msg in older:
            role = msg["role"]
            content = msg["content"]
            # Truncate long messages in summary
            if len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"[{role}]: {content}")

        summary = "Previous conversation summary:\n" + "\n".join(summary_parts)

        # Replace older messages with the summary
        self._messages = [
            {"role": "system", "content": summary},
            *self._messages[-keep_last:],
        ]

        return summary


class VectorMemory:
    """Long-term memory using vector search.

    When citadel_vector is available, text is embedded with a small
    deterministic hashing embedder and stored/queried through the HNSW
    VectorStore; otherwise it falls back to a zero-dependency TF-IDF-style
    keyword search. The embedder is a lightweight hashing bag-of-words
    (not a neural model) — enough to exercise the vector path end to end.
    """

    _EMBED_DIM = 256

    def __init__(self, path: str = "./agent_memory") -> None:
        """Initialize vector memory.

        Args:
            path: Storage path for the memory backend.
        """
        self.path = path
        self._backend: str = "keyword"
        self._store: list[dict[str, Any]] = []  # Fallback store
        self._next_id = 0

        # Try to use citadel_vector for real vector search. The store needs a
        # fixed dimension and a way to turn text into vectors (see _embed). If
        # the optional package is missing or fails to initialise, degrade to the
        # always-available keyword backend rather than crash.
        try:
            import citadel_vector  # type: ignore[import-not-found] — optional dependency
            self._vector_store = citadel_vector.VectorStore(
                path=path, dim=self._EMBED_DIM
            )
            self._backend = "vector"
        except Exception:
            self._backend = "keyword"

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a piece of text in memory.

        Args:
            text: The text to store.
            metadata: Optional metadata to associate with the text.
        """
        if self._backend == "vector":
            meta = {"text": text, **(metadata or {})}
            self._vector_store.add(self._embed(text), id=self._next_id, metadata=meta)
            self._next_id += 1
        else:
            self._store.append({
                "text": text,
                "metadata": metadata or {},
                "tokens": self._tokenize(text),
            })

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Recall stored texts similar to the query.

        Args:
            query: The search query.
            k: Maximum number of results to return.

        Returns:
            List of matching text strings, ordered by relevance.
        """
        if self._backend == "vector":
            results = self._vector_store.search(self._embed(query), k=k)
            # VectorStore.search returns (id, distance, metadata) tuples; the
            # original text is carried in metadata under "text".
            return [
                meta["text"]
                for _id, _distance, meta in results
                if meta and "text" in meta
            ]
        else:
            return self._keyword_search(query, k)

    def _tokenize(self, text: str) -> Counter:
        """Tokenize text into word frequency counts."""
        words = re.findall(r'\w+', text.lower())
        return Counter(words)

    def _embed(self, text: str) -> Any:
        """Embed text into a fixed-dimension vector via hashing bag-of-words.

        Deterministic and dependency-light: each token is hashed with BLAKE2b
        (stable across processes, unlike the builtin hash()) into one of
        _EMBED_DIM signed buckets weighted by term frequency, then the vector is
        L2-normalised. This is a lightweight signal — token overlap in vector
        space, not a neural embedding — enough to drive the HNSW vector store.
        Only called on the vector backend, where numpy (a citadel_vector
        dependency) is guaranteed to be importable.
        """
        import numpy as np

        vec = np.zeros(self._EMBED_DIM, dtype=np.float64)
        for token, count in self._tokenize(text).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            bucket = h % self._EMBED_DIM
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[bucket] += sign * float(count)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def _keyword_search(self, query: str, k: int) -> list[str]:
        """Simple TF-IDF-like keyword search fallback.

        Args:
            query: Search query.
            k: Number of results.

        Returns:
            List of matching texts ranked by relevance.
        """
        if not self._store:
            return []

        query_tokens = self._tokenize(query)

        # Calculate document frequencies for IDF
        doc_count = len(self._store)
        doc_freq: Counter = Counter()
        for item in self._store:
            for token in item["tokens"]:
                doc_freq[token] += 1

        # Score each document
        scored: list[tuple[float, str]] = []
        for item in self._store:
            score = 0.0
            for token, query_count in query_tokens.items():
                tf = item["tokens"].get(token, 0)
                if tf > 0 and doc_freq[token] > 0:
                    idf = math.log(doc_count / doc_freq[token]) + 1.0
                    score += tf * idf * query_count
            scored.append((score, item["text"]))

        # Sort by score descending, return top k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:k] if score > 0]

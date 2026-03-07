"""Content deduplication using SHA-256 hashes."""

import json
from pathlib import Path

from citadel_ingest.chunker import Chunk


class ContentDedup:
    """Track seen content hashes to detect and skip duplicate chunks.

    Can persist the set of seen hashes to a JSON file for reuse
    across pipeline runs.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, chunk: Chunk) -> bool:
        """Check if a chunk's content has already been seen.

        Args:
            chunk: The chunk to check.

        Returns:
            True if the chunk ID (content hash) was already seen.
        """
        return chunk.id in self._seen

    def mark_seen(self, chunk: Chunk) -> None:
        """Record a chunk's content hash as seen.

        Args:
            chunk: The chunk to mark.
        """
        self._seen.add(chunk.id)

    @property
    def count(self) -> int:
        """Number of unique hashes tracked."""
        return len(self._seen)

    def save(self, path: str) -> None:
        """Persist seen hashes to a JSON file.

        Args:
            path: File path to write the JSON list of hashes.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(sorted(self._seen), f)

    def load(self, path: str) -> None:
        """Load previously seen hashes from a JSON file.

        Args:
            path: File path to read the JSON list of hashes.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dedup hash file not found: {path}")

        with p.open("r", encoding="utf-8") as f:
            hashes = json.load(f)

        self._seen.update(hashes)

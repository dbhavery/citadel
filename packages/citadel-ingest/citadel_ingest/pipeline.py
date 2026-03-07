"""End-to-end ingestion pipeline: parse -> chunk -> embed -> store."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from citadel_ingest.chunker import Chunk, FixedSizeChunker
from citadel_ingest.config import IngestConfig
from citadel_ingest.dedup import ContentDedup
from citadel_ingest.embedder import MockEmbedder
from citadel_ingest.parser import DocumentParser


@dataclass
class IngestResult:
    """Result of an ingestion operation."""

    files_processed: int = 0
    chunks_created: int = 0
    duplicates_skipped: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    text: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class IngestPipeline:
    """End-to-end: parse -> chunk -> embed -> store.

    If no external vector store is provided, chunks and vectors are
    stored in memory and search uses brute-force cosine similarity.
    """

    def __init__(
        self,
        chunker: Any = None,
        embedder: Any = None,
        store: Any = None,
        config: IngestConfig | None = None,
    ) -> None:
        self.config = config or IngestConfig()
        self.chunker = chunker or FixedSizeChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.embedder = embedder or MockEmbedder(dim=self.config.embed_dim)
        self.store = store  # Optional: VectorStore from citadel-vector
        self.parser = DocumentParser()
        self.dedup = ContentDedup() if self.config.dedup_enabled else None

        # In-memory fallback storage
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    async def ingest_file(self, path: str) -> IngestResult:
        """Ingest a single file through the full pipeline.

        Args:
            path: Absolute path to the file to ingest.

        Returns:
            IngestResult with processing statistics.
        """
        result = IngestResult()

        try:
            doc = self.parser.parse(path)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            result.errors.append(f"{path}: {exc}")
            return result

        result.files_processed = 1

        # Chunk
        chunk_meta = {"source_file": path}
        if doc.metadata:
            chunk_meta.update(doc.metadata)

        chunks = self.chunker.chunk(doc.text, metadata=chunk_meta)

        # Dedup and embed
        new_chunks: list[Chunk] = []
        for chunk in chunks:
            if self.dedup is not None:
                if self.dedup.is_duplicate(chunk):
                    result.duplicates_skipped += 1
                    continue
                self.dedup.mark_seen(chunk)
            new_chunks.append(chunk)

        if not new_chunks:
            return result

        texts = [c.text for c in new_chunks]
        vectors = await self.embedder.embed_batch(texts)

        # Store
        if self.store is not None:
            # External vector store integration point
            for chunk, vector in zip(new_chunks, vectors):
                self.store.add(chunk, vector)
        else:
            self._chunks.extend(new_chunks)
            self._vectors.extend(vectors)

        result.chunks_created = len(new_chunks)
        return result

    async def ingest_directory(
        self,
        path: str,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> IngestResult:
        """Ingest all supported files in a directory.

        Args:
            path: Absolute path to the directory.
            extensions: List of file extensions to include (e.g. [".md", ".py"]).
                       Defaults to config.supported_extensions.
            recursive: Whether to recurse into subdirectories.

        Returns:
            IngestResult with aggregate processing statistics.
        """
        result = IngestResult()
        allowed = set(extensions or self.config.supported_extensions)
        dir_path = Path(path)

        if not dir_path.is_dir():
            result.errors.append(f"Not a directory: {path}")
            return result

        if recursive:
            files = list(dir_path.rglob("*"))
        else:
            files = list(dir_path.iterdir())

        for file_path in sorted(files):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed:
                continue

            file_result = await self.ingest_file(str(file_path))
            result.files_processed += file_result.files_processed
            result.chunks_created += file_result.chunks_created
            result.duplicates_skipped += file_result.duplicates_skipped
            result.errors.extend(file_result.errors)

        return result

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search indexed chunks by cosine similarity to the query.

        Args:
            query: The search query text.
            k: Number of top results to return.

        Returns:
            List of SearchResult objects sorted by descending similarity.
        """
        if not self._chunks:
            return []

        query_vector = await self.embedder.embed(query)
        query_arr = np.array(query_vector, dtype=np.float64)

        stored_arr = np.array(self._vectors, dtype=np.float64)

        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        dots = stored_arr @ query_arr
        norms = np.linalg.norm(stored_arr, axis=1) * np.linalg.norm(query_arr)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        similarities = dots / norms

        # Get top-k indices
        top_k = min(k, len(self._chunks))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    text=self._chunks[idx].text,
                    score=float(similarities[idx]),
                    metadata=self._chunks[idx].metadata,
                )
            )

        return results

    def stats(self) -> dict:
        """Return pipeline statistics.

        Returns:
            Dictionary with chunk count, vector count, and dedup info.
        """
        info: dict = {
            "chunks_stored": len(self._chunks),
            "vectors_stored": len(self._vectors),
        }
        if self.dedup is not None:
            info["unique_hashes"] = self.dedup.count
        return info

"""Chunking strategies for splitting documents into processable pieces."""

import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of text with metadata and a content-based ID."""

    text: str
    metadata: dict = field(default_factory=dict)  # source_file, chunk_index, start_char, end_char
    id: str = ""  # SHA-256 hash of text for dedup

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class FixedSizeChunker:
    """Chunk text into fixed-size windows with optional overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        if not text.strip():
            return []

        base_meta = metadata or {}
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunk_meta = {
                    **base_meta,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                }
                chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
                chunk_index += 1

            if end >= len(text):
                break
            start += step

        return chunks


class SentenceChunker:
    """Chunk text by grouping sentences together."""

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> None:
        if sentences_per_chunk <= 0:
            raise ValueError("sentences_per_chunk must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be non-negative")
        if overlap_sentences >= sentences_per_chunk:
            raise ValueError("overlap_sentences must be less than sentences_per_chunk")
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Split on sentence-ending punctuation followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into chunks of grouped sentences."""
        if not text.strip():
            return []

        base_meta = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks: list[Chunk] = []
        step = self.sentences_per_chunk - self.overlap_sentences
        chunk_index = 0

        for i in range(0, len(sentences), step):
            group = sentences[i:i + self.sentences_per_chunk]
            if not group:
                break

            chunk_text = " ".join(group)
            chunk_meta = {
                **base_meta,
                "chunk_index": chunk_index,
                "start_sentence": i,
                "end_sentence": i + len(group),
            }
            chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
            chunk_index += 1

            # Stop if we've consumed all sentences
            if i + self.sentences_per_chunk >= len(sentences):
                break

        return chunks


class SemanticChunker:
    """Chunk text on paragraph/section boundaries."""

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into chunks based on paragraph boundaries.

        Paragraphs are detected by double newlines. Consecutive short
        paragraphs are merged to avoid tiny chunks.
        """
        if not text.strip():
            return []

        base_meta = metadata or {}
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0

        for para in paragraphs:
            chunk_meta = {
                **base_meta,
                "chunk_index": chunk_index,
                "chunk_type": "paragraph",
            }
            chunks.append(Chunk(text=para, metadata=chunk_meta))
            chunk_index += 1

        return chunks


class CodeChunker:
    """Chunk code by function/class boundaries using simple regex."""

    # Language-specific patterns for function/class boundaries
    PATTERNS: dict[str, str] = {
        "python": r'^(?=(?:def |class |async def ))',
        "javascript": r'^(?=(?:function |class |const \w+ = (?:async )?\(|export ))',
        "typescript": r'^(?=(?:function |class |const \w+ = (?:async )?\(|export |interface ))',
        "go": r'^(?=(?:func |type ))',
        "rust": r'^(?=(?:fn |pub fn |struct |impl |enum |trait ))',
    }

    def chunk(
        self, text: str, metadata: dict | None = None, language: str = "python"
    ) -> list[Chunk]:
        """Split code into chunks by function/class boundaries.

        Args:
            text: Source code text.
            metadata: Additional metadata to attach to each chunk.
            language: Programming language for boundary detection.

        Returns:
            List of Chunk objects split on code boundaries.
        """
        if not text.strip():
            return []

        base_meta = metadata or {}
        base_meta["language"] = language

        pattern = self.PATTERNS.get(language, self.PATTERNS["python"])
        parts = re.split(pattern, text, flags=re.MULTILINE)
        parts = [p for p in parts if p.strip()]

        if not parts:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0

        for part in parts:
            chunk_meta = {
                **base_meta,
                "chunk_index": chunk_index,
                "chunk_type": "code_block",
            }
            chunks.append(Chunk(text=part.rstrip(), metadata=chunk_meta))
            chunk_index += 1

        return chunks

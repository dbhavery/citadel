"""Tests for chunking strategies."""

import pytest

from citadel_ingest.chunker import (
    Chunk,
    CodeChunker,
    FixedSizeChunker,
    SemanticChunker,
    SentenceChunker,
)


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_correct_chunk_count_and_size(self) -> None:
        """FixedSizeChunker produces the right number of chunks with correct sizes."""
        text = "a" * 1000
        chunker = FixedSizeChunker(chunk_size=300, overlap=0)
        chunks = chunker.chunk(text)

        # 1000 / 300 = 3 full + 1 partial = 4 chunks
        assert len(chunks) == 4
        assert len(chunks[0].text) == 300
        assert len(chunks[1].text) == 300
        assert len(chunks[2].text) == 300
        assert len(chunks[3].text) == 100

    def test_overlap_between_chunks(self) -> None:
        """FixedSizeChunker produces overlapping chunks correctly."""
        text = "0123456789" * 10  # 100 chars
        chunker = FixedSizeChunker(chunk_size=30, overlap=10)
        chunks = chunker.chunk(text)

        # With step=20, chunk_size=30 on 100 chars:
        # Chunk 0: [0:30], Chunk 1: [20:50], Chunk 2: [40:70], Chunk 3: [60:90], Chunk 4: [80:100]
        assert len(chunks) == 5

        # Verify overlap: end of chunk[0] should match start of chunk[1]
        overlap_from_first = chunks[0].text[20:]   # last 10 chars of chunk 0
        overlap_from_second = chunks[1].text[:10]   # first 10 chars of chunk 1
        assert overlap_from_first == overlap_from_second


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_respects_sentence_boundaries(self) -> None:
        """SentenceChunker groups the right number of sentences per chunk."""
        text = (
            "First sentence. Second sentence. Third sentence. "
            "Fourth sentence. Fifth sentence. Sixth sentence. "
            "Seventh sentence."
        )
        chunker = SentenceChunker(sentences_per_chunk=3, overlap_sentences=0)
        chunks = chunker.chunk(text)

        # 7 sentences / 3 per chunk = 3 chunks (3, 3, 1)
        assert len(chunks) == 3
        assert "First sentence." in chunks[0].text
        assert "Third sentence." in chunks[0].text
        assert "Fourth sentence." in chunks[1].text
        assert "Seventh sentence." in chunks[2].text


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_splits_on_paragraph_boundaries(self) -> None:
        """SemanticChunker splits text on double-newline paragraph boundaries."""
        text = (
            "This is paragraph one with some content.\n\n"
            "This is paragraph two with more content.\n\n"
            "This is paragraph three, the final one."
        )
        chunker = SemanticChunker()
        chunks = chunker.chunk(text)

        assert len(chunks) == 3
        assert chunks[0].text == "This is paragraph one with some content."
        assert chunks[1].text == "This is paragraph two with more content."
        assert chunks[2].text == "This is paragraph three, the final one."


class TestCodeChunker:
    """Tests for CodeChunker."""

    def test_splits_on_function_boundaries(self) -> None:
        """CodeChunker splits Python code at function/class definitions."""
        code = '''import os

def hello():
    print("hello")

def world():
    print("world")

class Foo:
    pass
'''
        chunker = CodeChunker()
        chunks = chunker.chunk(code, language="python")

        # Should split into: imports block, hello(), world(), Foo class
        assert len(chunks) >= 3

        # Verify each function/class is in its own chunk
        texts = [c.text for c in chunks]
        assert any("def hello()" in t for t in texts)
        assert any("def world()" in t for t in texts)
        assert any("class Foo:" in t for t in texts)


class TestChunkId:
    """Tests for chunk ID determinism."""

    def test_chunk_ids_are_deterministic(self) -> None:
        """Same text always produces the same chunk ID (SHA-256 hash)."""
        chunk_a = Chunk(text="Hello, world!")
        chunk_b = Chunk(text="Hello, world!")
        chunk_c = Chunk(text="Different text.")

        assert chunk_a.id == chunk_b.id
        assert chunk_a.id != chunk_c.id
        # SHA-256 produces 64 hex chars
        assert len(chunk_a.id) == 64

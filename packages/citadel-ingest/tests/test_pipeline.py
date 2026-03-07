"""Tests for the ingestion pipeline."""

import os
import tempfile
from pathlib import Path

import pytest

from citadel_ingest.chunker import FixedSizeChunker
from citadel_ingest.config import IngestConfig
from citadel_ingest.embedder import MockEmbedder
from citadel_ingest.pipeline import IngestPipeline


@pytest.fixture
def pipeline() -> IngestPipeline:
    """Create a pipeline with MockEmbedder and small chunks for testing."""
    config = IngestConfig(
        chunk_size=50,
        chunk_overlap=10,
        embed_provider="mock",
        embed_dim=64,
        dedup_enabled=True,
    )
    return IngestPipeline(
        chunker=FixedSizeChunker(chunk_size=50, overlap=10),
        embedder=MockEmbedder(dim=64),
        config=config,
    )


@pytest.fixture
def temp_dir() -> str:
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as d:
        # Create a markdown file
        md_path = Path(d) / "readme.md"
        md_path.write_text(
            "# Test Document\n\nThis is a test markdown file with enough "
            "content to produce multiple chunks when processed by the pipeline.",
            encoding="utf-8",
        )

        # Create a Python file
        py_path = Path(d) / "example.py"
        py_path.write_text(
            "def hello():\n    return 'hello world'\n\n"
            "def goodbye():\n    return 'goodbye world'\n",
            encoding="utf-8",
        )

        # Create a text file
        txt_path = Path(d) / "notes.txt"
        txt_path.write_text(
            "These are some notes about the project. "
            "They contain useful information for testing the pipeline.",
            encoding="utf-8",
        )

        # Create an unsupported file (should be skipped)
        jpg_path = Path(d) / "image.jpg"
        jpg_path.write_bytes(b"\xff\xd8\xff\xe0")

        yield d


@pytest.mark.asyncio
class TestIngestPipeline:
    """Tests for IngestPipeline."""

    async def test_ingest_single_file(self, pipeline: IngestPipeline) -> None:
        """Ingest a single file end-to-end with MockEmbedder."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "This is a test document with enough text to produce "
                "multiple chunks when processed through the pipeline system."
            )
            f.flush()
            path = f.name

        try:
            result = await pipeline.ingest_file(path)

            assert result.files_processed == 1
            assert result.chunks_created > 0
            assert result.errors == []
            assert len(pipeline._chunks) == result.chunks_created
            assert len(pipeline._vectors) == result.chunks_created
        finally:
            os.unlink(path)

    async def test_ingest_directory_with_extension_filter(
        self, pipeline: IngestPipeline, temp_dir: str
    ) -> None:
        """Ingest a directory filtering by extension."""
        result = await pipeline.ingest_directory(
            temp_dir, extensions=[".md", ".txt"]
        )

        # Should process readme.md and notes.txt but NOT example.py or image.jpg
        assert result.files_processed == 2
        assert result.chunks_created > 0
        assert result.errors == []

    async def test_dedup_skips_duplicate_chunks(
        self, pipeline: IngestPipeline
    ) -> None:
        """Deduplication skips chunks with identical content."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Identical content that repeats across files for testing.")
            f.flush()
            path1 = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            # Exact same content -> same chunk hashes
            f.write("Identical content that repeats across files for testing.")
            f.flush()
            path2 = f.name

        try:
            result1 = await pipeline.ingest_file(path1)
            result2 = await pipeline.ingest_file(path2)

            assert result1.chunks_created > 0
            assert result2.duplicates_skipped == result1.chunks_created
            assert result2.chunks_created == 0
        finally:
            os.unlink(path1)
            os.unlink(path2)

    async def test_search_returns_relevant_results(
        self, pipeline: IngestPipeline
    ) -> None:
        """Search returns results ordered by similarity score."""
        # Ingest several distinct documents
        texts = [
            "Python is a programming language used for web development.",
            "The weather today is sunny and warm with clear skies.",
            "Machine learning uses algorithms to find patterns in data.",
        ]
        paths: list[str] = []
        for i, text in enumerate(texts):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                f.flush()
                paths.append(f.name)

        try:
            for path in paths:
                await pipeline.ingest_file(path)

            results = await pipeline.search("programming language", k=3)

            assert len(results) > 0
            # Results should be sorted by descending score
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
            # Each result should have text and metadata
            for r in results:
                assert r.text != ""
                assert isinstance(r.score, float)
        finally:
            for p in paths:
                os.unlink(p)

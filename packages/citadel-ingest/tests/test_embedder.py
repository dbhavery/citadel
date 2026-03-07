"""Tests for embedding generation."""

import pytest

from citadel_ingest.embedder import MockEmbedder


@pytest.mark.asyncio
class TestMockEmbedder:
    """Tests for MockEmbedder."""

    async def test_returns_correct_dimension(self) -> None:
        """MockEmbedder returns vectors with the configured dimensionality."""
        embedder = MockEmbedder(dim=384)
        vector = await embedder.embed("test text")

        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)

    async def test_is_deterministic(self) -> None:
        """MockEmbedder returns the same vector for the same input text."""
        embedder = MockEmbedder(dim=128)

        vec_a = await embedder.embed("hello world")
        vec_b = await embedder.embed("hello world")
        vec_c = await embedder.embed("different text")

        # Same input -> same output
        assert vec_a == vec_b
        # Different input -> different output
        assert vec_a != vec_c

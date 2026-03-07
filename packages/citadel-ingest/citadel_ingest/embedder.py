"""Embedding generation for text chunks."""

import hashlib
import struct
from typing import Any


class Embedder:
    """Generate embeddings for text chunks via Ollama's /api/embed endpoint."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init httpx async client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "httpx is required for Ollama embeddings. "
                    "Install it with: pip install citadel-ingest[ollama]"
                ) from exc
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        client = self._get_client()
        response = await client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        # Ollama /api/embed returns {"embeddings": [[...]]}
        return data["embeddings"][0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        client = self._get_client()
        response = await client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class MockEmbedder:
    """Deterministic embedder for testing -- returns hash-based vectors.

    Produces the same vector for the same text every time, without
    needing any external service.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _text_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic vector via SHA-256 expansion.

        The hash is expanded by re-hashing iteratively to fill the
        required dimensionality, then normalized to unit length.
        """
        values: list[float] = []
        seed = text.encode("utf-8")
        counter = 0

        while len(values) < self.dim:
            h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            # Each SHA-256 gives 32 bytes = 8 floats (4 bytes each, via struct)
            for i in range(0, 32, 4):
                if len(values) >= self.dim:
                    break
                # Convert 4 bytes to a float in [-1, 1]
                raw = struct.unpack(">I", h[i:i + 4])[0]
                values.append((raw / (2**32 - 1)) * 2.0 - 1.0)
            counter += 1

        # Normalize to unit vector
        magnitude = sum(v * v for v in values) ** 0.5
        if magnitude > 0:
            values = [v / magnitude for v in values]

        return values

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A deterministic list of floats of length self.dim.
        """
        return self._text_to_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of deterministic embedding vectors.
        """
        return [self._text_to_vector(t) for t in texts]

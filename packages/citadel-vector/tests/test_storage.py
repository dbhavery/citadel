"""Tests for persistent vector storage."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from citadel_vector.storage import VectorStore


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test storage."""
    d = tempfile.mkdtemp(prefix="citadel_vector_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestVectorStore:
    """Tests for VectorStore persistence and retrieval."""

    def test_save_and_load_preserves_vectors(self, tmp_dir: str) -> None:
        """Vectors should survive save/load cycle."""
        store_path = str(Path(tmp_dir) / "test_store")

        # Create store, add vectors, save
        store = VectorStore(path=store_path, dim=4, max_elements=100)
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        store.add(v1, "vec1")
        store.add(v2, "vec2")
        store.save()

        # Load into new store
        loaded = VectorStore.load(store_path)
        assert len(loaded) == 2

        # Search should find vec1
        results = loaded.search(v1, k=1)
        assert len(results) == 1
        assert results[0][0] == "vec1"
        assert results[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_search_after_load_same_results(self, tmp_dir: str) -> None:
        """Search results should be identical before and after save/load."""
        store_path = str(Path(tmp_dir) / "test_store2")
        rng = np.random.default_rng(99)

        store = VectorStore(
            path=store_path, dim=16, max_elements=200, metric="euclidean"
        )

        # Add 50 random vectors
        vectors = rng.random((50, 16))
        for i in range(50):
            store.add(vectors[i], f"v{i}")

        query = rng.random(16)

        # Search before save
        results_before = store.search(query, k=5)
        store.save()

        # Search after load
        loaded = VectorStore.load(store_path)
        results_after = loaded.search(query, k=5)

        # IDs and distances should match
        ids_before = [r[0] for r in results_before]
        ids_after = [r[0] for r in results_after]
        assert ids_before == ids_after

        for rb, ra in zip(results_before, results_after):
            assert rb[1] == pytest.approx(ra[1], abs=1e-6)

    def test_metadata_stored_and_retrieved(self, tmp_dir: str) -> None:
        """Metadata should be stored in SQLite and retrievable after load."""
        store_path = str(Path(tmp_dir) / "test_store3")

        store = VectorStore(path=store_path, dim=3, max_elements=100)
        store.add(
            np.array([1.0, 0.0, 0.0]),
            "doc1",
            metadata={"title": "Hello", "page": 42},
        )
        store.add(
            np.array([0.0, 1.0, 0.0]),
            "doc2",
            metadata={"title": "World", "page": 7},
        )
        store.save()

        loaded = VectorStore.load(store_path)
        results = loaded.search(np.array([1.0, 0.0, 0.0]), k=1)
        assert results[0][0] == "doc1"
        assert results[0][2] == {"title": "Hello", "page": 42}

    def test_stats_return_correct_counts(self, tmp_dir: str) -> None:
        """stats() should return accurate counts."""
        store_path = str(Path(tmp_dir) / "test_store4")

        store = VectorStore(path=store_path, dim=8, max_elements=500)
        assert store.stats()["count"] == 0

        for i in range(25):
            vec = np.zeros(8)
            vec[i % 8] = 1.0
            store.add(vec, f"item_{i}")

        stats = store.stats()
        assert stats["count"] == 25
        assert stats["dim"] == 8
        assert stats["metric"] == "cosine"
        assert stats["max_elements"] == 500

"""Tests for HNSW index implementation."""

import numpy as np
import pytest

from citadel_vector.hnsw import HNSWIndex


class TestHNSWBasic:
    """Basic HNSW index operations."""

    def test_add_single_and_search(self) -> None:
        """Add a single vector and search should return it."""
        index = HNSWIndex(dim=4, max_elements=100)
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        index.add(vec, "a", metadata={"label": "first"})

        results = index.search(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(0.0, abs=1e-6)
        assert results[0][2] == {"label": "first"}

    def test_knn_returns_correct_nearest(self) -> None:
        """k-NN should return the closest vectors."""
        index = HNSWIndex(dim=3, max_elements=100, metric="euclidean")

        # Insert 5 vectors at known positions
        index.add(np.array([0.0, 0.0, 0.0]), "origin")
        index.add(np.array([1.0, 0.0, 0.0]), "x1")
        index.add(np.array([0.0, 1.0, 0.0]), "y1")
        index.add(np.array([10.0, 10.0, 10.0]), "far1")
        index.add(np.array([20.0, 20.0, 20.0]), "far2")

        # Search near origin
        query = np.array([0.1, 0.1, 0.0])
        results = index.search(query, k=3, ef_search=50)

        assert len(results) == 3
        result_ids = [r[0] for r in results]
        # The three closest should be origin, x1, y1 (not far1, far2)
        assert "origin" in result_ids
        assert "x1" in result_ids
        assert "y1" in result_ids

    def test_search_k_greater_than_num_vectors(self) -> None:
        """Search with k > num_vectors should return all vectors."""
        index = HNSWIndex(dim=2, max_elements=100)
        index.add(np.array([1.0, 0.0]), "a")
        index.add(np.array([0.0, 1.0]), "b")

        results = index.search(np.array([1.0, 0.0]), k=100)
        assert len(results) == 2

    def test_empty_index_search(self) -> None:
        """Searching an empty index should return empty list."""
        index = HNSWIndex(dim=4, max_elements=100)
        results = index.search(np.array([1.0, 0.0, 0.0, 0.0]), k=5)
        assert results == []

    def test_delete_removes_from_results(self) -> None:
        """Deleted vectors should not appear in search results."""
        index = HNSWIndex(dim=3, max_elements=100, metric="euclidean")
        index.add(np.array([0.0, 0.0, 0.0]), "a")
        index.add(np.array([1.0, 0.0, 0.0]), "b")
        index.add(np.array([2.0, 0.0, 0.0]), "c")

        # Before delete: searching near origin returns a
        results = index.search(np.array([0.0, 0.0, 0.0]), k=1)
        assert results[0][0] == "a"

        # Delete 'a'
        index.delete("a")
        assert len(index) == 2

        # After delete: 'a' should not appear
        results = index.search(np.array([0.0, 0.0, 0.0]), k=3)
        result_ids = [r[0] for r in results]
        assert "a" not in result_ids
        assert len(results) == 2

    def test_batch_add(self) -> None:
        """batch_add should work like multiple individual adds."""
        index = HNSWIndex(dim=3, max_elements=100)
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        ids = ["x", "y", "z"]
        metas = [{"axis": "x"}, {"axis": "y"}, {"axis": "z"}]
        index.batch_add(vectors, ids, metas)

        assert len(index) == 3
        results = index.search(np.array([1.0, 0.0, 0.0]), k=1)
        assert results[0][0] == "x"
        assert results[0][2] == {"axis": "x"}

    def test_filter_function(self) -> None:
        """filter_fn should exclude non-matching results."""
        index = HNSWIndex(dim=2, max_elements=100, metric="euclidean")
        index.add(np.array([0.0, 0.0]), "a", metadata={"color": "red"})
        index.add(np.array([0.1, 0.0]), "b", metadata={"color": "blue"})
        index.add(np.array([0.2, 0.0]), "c", metadata={"color": "red"})
        index.add(np.array([10.0, 10.0]), "d", metadata={"color": "red"})

        # Search near origin but only red items
        results = index.search(
            np.array([0.0, 0.0]),
            k=10,
            filter_fn=lambda m: m is not None and m.get("color") == "red",
        )
        result_ids = [r[0] for r in results]
        assert "b" not in result_ids  # blue, should be filtered
        assert "a" in result_ids
        assert "c" in result_ids


class TestHNSWRecall:
    """Recall benchmark: the key correctness test for the HNSW implementation."""

    def test_recall_at_10_above_90_percent(self) -> None:
        """Add 1000 random vectors and verify recall@10 > 0.9 against brute force.

        This is the gold-standard test: if HNSW doesn't achieve good recall
        against a brute-force search, the algorithm is broken.
        """
        rng = np.random.default_rng(12345)
        dim = 64
        n = 1000
        n_queries = 50
        k = 10

        # Generate random vectors
        vectors = rng.random((n, dim)).astype(np.float64)
        # Normalize to unit vectors (makes cosine distance more meaningful)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        # Build index with generous construction parameters
        index = HNSWIndex(
            dim=dim,
            max_elements=n + 100,
            M=16,
            ef_construction=200,
            metric="cosine",
        )
        for i in range(n):
            index.add(vectors[i], i)

        assert len(index) == n

        # Generate random queries
        queries = rng.random((n_queries, dim)).astype(np.float64)
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / query_norms

        total_recall = 0.0

        for q_idx in range(n_queries):
            query = queries[q_idx]

            # Brute force: compute distance to all vectors
            brute_force_dists = []
            for i in range(n):
                d = 1.0 - float(np.dot(query, vectors[i]))
                brute_force_dists.append((i, d))
            brute_force_dists.sort(key=lambda x: x[1])
            true_nn = set(x[0] for x in brute_force_dists[:k])

            # HNSW search with generous ef_search
            results = index.search(query, k=k, ef_search=200)
            hnsw_nn = set(r[0] for r in results)

            recall = len(true_nn & hnsw_nn) / k
            total_recall += recall

        avg_recall = total_recall / n_queries
        assert avg_recall > 0.9, (
            f"Average recall@{k} = {avg_recall:.4f}, expected > 0.9. "
            f"The HNSW implementation may have a correctness issue."
        )

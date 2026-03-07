"""Tests for distance functions."""

import numpy as np
import pytest

from citadel_vector.distance import (
    batch_cosine_distance,
    batch_euclidean_distance,
    cosine_distance,
    dot_product_distance,
    euclidean_distance,
)


class TestCosineDistance:
    """Tests for cosine_distance."""

    def test_identical_vectors_zero(self) -> None:
        """Identical vectors should have cosine distance of 0."""
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_vectors_one(self) -> None:
        """Orthogonal vectors should have cosine distance of 1."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-10)

    def test_opposite_vectors_two(self) -> None:
        """Opposite vectors should have cosine distance of approximately 2."""
        a = np.array([1.0, 2.0, 3.0])
        b = -a
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-10)


class TestEuclideanDistance:
    """Tests for euclidean_distance."""

    def test_known_values(self) -> None:
        """Euclidean distance for known vectors."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 4.0, 0.0])
        assert euclidean_distance(a, b) == pytest.approx(5.0, abs=1e-10)

        c = np.array([1.0, 1.0])
        d = np.array([4.0, 5.0])
        assert euclidean_distance(c, d) == pytest.approx(5.0, abs=1e-10)


class TestDotProductDistance:
    """Tests for dot_product_distance."""

    def test_known_values(self) -> None:
        """Dot product distance for known vectors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        # dot = 1*4 + 2*5 + 3*6 = 32
        # negative dot = -32
        assert dot_product_distance(a, b) == pytest.approx(-32.0, abs=1e-10)

        # More similar (larger dot product) should give more negative distance
        c = np.array([1.0, 0.0])
        d_aligned = np.array([2.0, 0.0])
        d_ortho = np.array([0.0, 2.0])
        assert dot_product_distance(c, d_aligned) < dot_product_distance(c, d_ortho)


class TestBatchOperations:
    """Tests for batch distance functions."""

    def test_batch_matches_individual(self) -> None:
        """Batch operations must match individual calls."""
        rng = np.random.default_rng(42)
        query = rng.random(64)
        vectors = rng.random((50, 64))

        # Cosine
        batch_cos = batch_cosine_distance(query, vectors)
        for i in range(len(vectors)):
            individual = cosine_distance(query, vectors[i])
            assert batch_cos[i] == pytest.approx(individual, abs=1e-10), (
                f"Cosine mismatch at index {i}: batch={batch_cos[i]}, individual={individual}"
            )

        # Euclidean
        batch_euc = batch_euclidean_distance(query, vectors)
        for i in range(len(vectors)):
            individual = euclidean_distance(query, vectors[i])
            assert batch_euc[i] == pytest.approx(individual, abs=1e-10), (
                f"Euclidean mismatch at index {i}: batch={batch_euc[i]}, individual={individual}"
            )

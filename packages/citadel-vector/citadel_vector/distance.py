"""Distance functions for vector similarity search.

All functions operate on NumPy arrays. Distances are non-negative where
smaller values indicate greater similarity.
"""

import numpy as np
import numpy.typing as npt


def cosine_distance(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> float:
    """Compute cosine distance between two vectors.

    cosine_distance = 1 - cosine_similarity.
    Returns 0 for identical directions, 1 for orthogonal, 2 for opposite.

    Args:
        a: First vector, shape (d,).
        b: Second vector, shape (d,).

    Returns:
        Cosine distance in [0, 2].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = np.dot(a, b) / (norm_a * norm_b)
    # Clamp to [-1, 1] to handle floating point errors
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(1.0 - similarity)


def euclidean_distance(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> float:
    """Compute Euclidean (L2) distance between two vectors.

    Args:
        a: First vector, shape (d,).
        b: Second vector, shape (d,).

    Returns:
        Non-negative Euclidean distance.
    """
    return float(np.linalg.norm(a - b))


def dot_product_distance(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> float:
    """Compute negative dot product distance.

    Uses negative dot product so that smaller values indicate greater
    similarity, consistent with the other distance functions.

    Args:
        a: First vector, shape (d,).
        b: Second vector, shape (d,).

    Returns:
        Negative dot product (more negative = vectors were more aligned).
    """
    return float(-np.dot(a, b))


def batch_cosine_distance(
    query: npt.NDArray[np.floating],
    vectors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute cosine distance from query to each row in vectors.

    Vectorized with NumPy for batch operations.

    Args:
        query: Query vector, shape (d,).
        vectors: Matrix of vectors, shape (n, d).

    Returns:
        Array of cosine distances, shape (n,).
    """
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        return np.ones(vectors.shape[0], dtype=np.float64)
    vector_norms = np.linalg.norm(vectors, axis=1)
    # Avoid division by zero
    safe_norms = np.where(vector_norms == 0.0, 1.0, vector_norms)
    similarities = vectors @ query / (safe_norms * query_norm)
    # Zero-norm vectors get distance 1.0
    similarities = np.where(vector_norms == 0.0, 0.0, similarities)
    similarities = np.clip(similarities, -1.0, 1.0)
    return 1.0 - similarities


def batch_euclidean_distance(
    query: npt.NDArray[np.floating],
    vectors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute Euclidean distance from query to each row in vectors.

    Vectorized with NumPy for batch operations.

    Args:
        query: Query vector, shape (d,).
        vectors: Matrix of vectors, shape (n, d).

    Returns:
        Array of Euclidean distances, shape (n,).
    """
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    diff = vectors - query[np.newaxis, :]
    return np.linalg.norm(diff, axis=1)

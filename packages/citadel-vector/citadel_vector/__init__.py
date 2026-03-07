"""
citadel-vector: HNSW vector search engine from scratch.

A standalone vector similarity search engine implementing the HNSW
(Hierarchical Navigable Small World) algorithm with NumPy. No ChromaDB,
no FAISS -- the actual algorithm implemented in Python.
"""

__version__ = "0.1.0"

from citadel_vector.distance import (
    batch_cosine_distance,
    batch_euclidean_distance,
    cosine_distance,
    dot_product_distance,
    euclidean_distance,
)
from citadel_vector.hnsw import HNSWIndex
from citadel_vector.storage import VectorStore
from citadel_vector.config import VectorConfig

__all__ = [
    "HNSWIndex",
    "VectorStore",
    "VectorConfig",
    "cosine_distance",
    "euclidean_distance",
    "dot_product_distance",
    "batch_cosine_distance",
    "batch_euclidean_distance",
]

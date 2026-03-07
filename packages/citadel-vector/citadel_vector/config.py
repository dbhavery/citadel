"""Configuration dataclass for citadel-vector."""

from dataclasses import dataclass


@dataclass
class VectorConfig:
    """Configuration for VectorStore and HNSWIndex defaults."""

    storage_path: str = "./vector_data"
    default_metric: str = "cosine"
    default_M: int = 16
    default_ef_construction: int = 200
    default_ef_search: int = 50

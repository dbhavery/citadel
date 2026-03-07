"""Configuration for citadel-ingest pipeline."""

from dataclasses import dataclass, field


@dataclass
class IngestConfig:
    """Configuration for the ingestion pipeline."""

    chunk_strategy: str = "fixed"  # fixed, sentence, semantic, code
    chunk_size: int = 500
    chunk_overlap: int = 50
    embed_provider: str = "mock"  # mock, ollama
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 384
    ollama_url: str = "http://localhost:11434"
    dedup_enabled: bool = True
    supported_extensions: list[str] = field(
        default_factory=lambda: [".md", ".txt", ".py", ".js", ".ts", ".pdf", ".docx"]
    )

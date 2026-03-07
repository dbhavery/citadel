# citadel-ingest

Document ingestion pipeline for RAG. Parses documents (PDF, Markdown, DOCX, plain text, code), splits them into chunks, generates embeddings, and stores them for retrieval.

## Install

```bash
pip install -e ".[dev]"

# With optional format support:
pip install -e ".[all,dev]"
```

## Quick Start

```python
import asyncio
from citadel_ingest.pipeline import IngestPipeline
from citadel_ingest.chunker import FixedSizeChunker
from citadel_ingest.embedder import MockEmbedder

async def main():
    pipeline = IngestPipeline(
        chunker=FixedSizeChunker(chunk_size=500, overlap=50),
        embedder=MockEmbedder(dim=384),
    )

    # Ingest a single file
    result = await pipeline.ingest_file("/path/to/document.md")
    print(f"Chunks created: {result.chunks_created}")

    # Ingest a directory
    result = await pipeline.ingest_directory(
        "/path/to/docs/",
        extensions=[".md", ".txt", ".py"],
    )

    # Search
    results = await pipeline.search("how does authentication work?", k=5)
    for r in results:
        print(f"[{r.score:.3f}] {r.text[:100]}...")

asyncio.run(main())
```

## Chunking Strategies

| Strategy | Class | Best For |
|---|---|---|
| Fixed-size | `FixedSizeChunker` | General-purpose, predictable chunk sizes |
| Sentence | `SentenceChunker` | Prose, articles, documentation |
| Semantic | `SemanticChunker` | Documents with clear paragraph structure |
| Code | `CodeChunker` | Source code (Python, JS, TS, Go, Rust) |

## Tests

```bash
python -m pytest tests/ -v
```

## Architecture

```
file -> DocumentParser -> plain text
     -> Chunker        -> list[Chunk]
     -> Embedder        -> list[vector]
     -> Store           -> in-memory or citadel-vector
```

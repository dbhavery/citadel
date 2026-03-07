"""REST API server for citadel-vector.

Provides a FastAPI application for managing vector collections over HTTP.
Requires the [server] optional dependency: pip install citadel-vector[server]

Entry point: citadel-vector serve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import fastapi
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    fastapi = None  # type: ignore[assignment]

from citadel_vector.storage import VectorStore

# --- Pydantic models (only defined if FastAPI is available) ---

if fastapi is not None:

    class CreateCollectionRequest(BaseModel):
        name: str
        dim: int
        metric: str = "cosine"
        max_elements: int = 100_000
        M: int = 16
        ef_construction: int = 200

    class AddVectorsRequest(BaseModel):
        vectors: list[list[float]]
        ids: list[str]
        metadatas: Optional[list[Optional[dict[str, Any]]]] = None

    class SearchRequest(BaseModel):
        query: list[float]
        k: int = 10
        ef_search: int = 50

    class SearchResult(BaseModel):
        id: str
        distance: float
        metadata: Optional[dict[str, Any]] = None


def create_app(storage_dir: str = "./vector_data") -> "FastAPI":
    """Create and configure the FastAPI application.

    Args:
        storage_dir: Base directory for collection storage.

    Returns:
        Configured FastAPI app.
    """
    if fastapi is None:
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install citadel-vector[server]"
        )

    app = FastAPI(
        title="citadel-vector",
        description="Vector search engine with HNSW indexing",
        version="0.1.0",
    )

    base_path = Path(storage_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # In-memory collection registry
    collections: dict[str, VectorStore] = {}

    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/collections")
    def create_collection(req: CreateCollectionRequest) -> dict[str, str]:
        """Create a new vector collection."""
        if req.name in collections:
            raise HTTPException(status_code=409, detail=f"Collection '{req.name}' already exists")

        col_path = base_path / req.name
        store = VectorStore(
            path=str(col_path),
            dim=req.dim,
            max_elements=req.max_elements,
            M=req.M,
            ef_construction=req.ef_construction,
            metric=req.metric,
        )
        collections[req.name] = store
        return {"status": "created", "name": req.name}

    @app.post("/collections/{name}/add")
    def add_vectors(name: str, req: AddVectorsRequest) -> dict[str, Any]:
        """Add vectors to a collection."""
        if name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

        store = collections[name]
        vectors = np.array(req.vectors, dtype=np.float64)
        metadatas = req.metadatas

        for i, vid in enumerate(req.ids):
            meta = metadatas[i] if metadatas is not None else None
            store.add(vectors[i], vid, meta)

        store.save()
        return {"status": "added", "count": len(req.ids)}

    @app.post("/collections/{name}/search")
    def search_vectors(name: str, req: SearchRequest) -> list[dict[str, Any]]:
        """Search for nearest neighbors in a collection."""
        if name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

        store = collections[name]
        query = np.array(req.query, dtype=np.float64)
        results = store.search(query, k=req.k, ef_search=req.ef_search)

        return [
            {"id": str(rid), "distance": float(dist), "metadata": meta}
            for rid, dist, meta in results
        ]

    @app.get("/collections/{name}/stats")
    def collection_stats(name: str) -> dict[str, Any]:
        """Get statistics for a collection."""
        if name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

        return collections[name].stats()

    return app


def main() -> None:
    """CLI entry point for `citadel-vector serve`."""
    if fastapi is None:
        print(
            "Error: FastAPI is required for the server.\n"
            "Install with: pip install citadel-vector[server]",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="citadel-vector",
        description="Vector search engine with HNSW indexing",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the REST API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    serve_parser.add_argument("--port", type=int, default=8082, help="Bind port")
    serve_parser.add_argument(
        "--storage-dir", default="./vector_data", help="Base storage directory"
    )

    args = parser.parse_args()

    if args.command == "serve":
        app = create_app(storage_dir=args.storage_dir)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

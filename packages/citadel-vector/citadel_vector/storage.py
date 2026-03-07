"""Persistent vector storage wrapping HNSWIndex.

Vectors are stored as a NumPy memory-mapped file, metadata in SQLite,
and the graph structure as a pickle file.
"""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from citadel_vector.hnsw import HNSWIndex


class VectorStore:
    """Persistent vector store backed by HNSWIndex.

    Data layout on disk::

        {path}/
            vectors.npy    -- NumPy array of all vectors
            graph.pkl      -- pickled graph structure + index parameters
            metadata.db    -- SQLite database for metadata

    Args:
        path: Directory to store data in. Created if it doesn't exist.
        dim: Vector dimensionality (required for new stores, ignored on load).
        **hnsw_kwargs: Keyword arguments forwarded to HNSWIndex
            (max_elements, M, ef_construction, metric).
    """

    def __init__(
        self,
        path: str,
        dim: int = 0,
        **hnsw_kwargs: Any,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._vectors_path = self.path / "vectors.npy"
        self._graph_path = self.path / "graph.pkl"
        self._metadata_path = self.path / "metadata.db"

        # Try to load existing store
        if self._graph_path.exists() and self._vectors_path.exists():
            self._load()
        else:
            if dim <= 0:
                raise ValueError("dim must be > 0 when creating a new VectorStore")
            max_elements = hnsw_kwargs.pop("max_elements", 100_000)
            self._index = HNSWIndex(dim=dim, max_elements=max_elements, **hnsw_kwargs)
            self._id_order: list[Any] = []  # Tracks insertion order for numpy file

        # Initialize SQLite metadata store
        self._init_metadata_db()

    def _init_metadata_db(self) -> None:
        """Create the metadata SQLite table if it doesn't exist."""
        conn = sqlite3.connect(str(self._metadata_path))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS metadata ("
                "  id TEXT PRIMARY KEY,"
                "  data TEXT"
                ")"
            )
            conn.commit()
        finally:
            conn.close()

    def _set_metadata(self, id: Any, metadata: Optional[dict]) -> None:
        """Store metadata for a vector ID in SQLite."""
        conn = sqlite3.connect(str(self._metadata_path))
        try:
            data_json = json.dumps(metadata) if metadata is not None else None
            conn.execute(
                "INSERT OR REPLACE INTO metadata (id, data) VALUES (?, ?)",
                (str(id), data_json),
            )
            conn.commit()
        finally:
            conn.close()

    def _get_metadata(self, id: Any) -> Optional[dict]:
        """Retrieve metadata for a vector ID from SQLite."""
        conn = sqlite3.connect(str(self._metadata_path))
        try:
            cursor = conn.execute(
                "SELECT data FROM metadata WHERE id = ?", (str(id),)
            )
            row = cursor.fetchone()
            if row is None or row[0] is None:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

    def _delete_metadata(self, id: Any) -> None:
        """Delete metadata for a vector ID from SQLite."""
        conn = sqlite3.connect(str(self._metadata_path))
        try:
            conn.execute("DELETE FROM metadata WHERE id = ?", (str(id),))
            conn.commit()
        finally:
            conn.close()

    def add(
        self,
        vector: npt.NDArray[np.floating],
        id: Any,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a vector with optional metadata.

        Args:
            vector: Vector to add, shape (dim,).
            id: Unique identifier.
            metadata: Optional metadata dict.
        """
        self._index.add(vector, id, metadata)
        if id not in self._id_order:
            self._id_order.append(id)
        self._set_metadata(id, metadata)

    def search(
        self,
        query: npt.NDArray[np.floating],
        k: int = 10,
        ef_search: int = 50,
        filter_fn: Optional[Callable[[Optional[dict]], bool]] = None,
    ) -> list[tuple[Any, float, Optional[dict]]]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector, shape (dim,).
            k: Number of results.
            ef_search: Beam width for search.
            filter_fn: Optional metadata filter.

        Returns:
            List of (id, distance, metadata) sorted by distance.
        """
        return self._index.search(query, k=k, ef_search=ef_search, filter_fn=filter_fn)

    def delete(self, id: Any) -> None:
        """Mark a vector as deleted.

        Args:
            id: The ID to delete.
        """
        self._index.delete(id)
        self._delete_metadata(id)

    def save(self) -> None:
        """Persist the index to disk.

        Writes three files:
        - vectors.npy: all vector data as a NumPy array
        - graph.pkl: graph structure, parameters, and bookkeeping
        - metadata.db: already persisted on each add() call
        """
        # Save vectors as numpy array
        if self._index._vectors:
            # Build array in insertion order
            active_ids = [vid for vid in self._id_order if vid in self._index._vectors]
            vec_array = np.array(
                [self._index._vectors[vid] for vid in active_ids],
                dtype=np.float64,
            )
            np.save(str(self._vectors_path), vec_array)
        else:
            np.save(str(self._vectors_path), np.array([], dtype=np.float64))

        # Save graph and index state
        state = {
            "dim": self._index.dim,
            "max_elements": self._index.max_elements,
            "M": self._index.M,
            "ef_construction": self._index.ef_construction,
            "metric": self._index.metric,
            "graph": self._index._graph,
            "node_level": self._index._node_level,
            "entry_point": self._index._entry_point,
            "max_level": self._index._max_level,
            "deleted": self._index._deleted,
            "metadata_map": self._index._metadata,
            "id_order": self._id_order,
        }
        with open(self._graph_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self) -> None:
        """Load index from disk."""
        # Load graph state
        with open(self._graph_path, "rb") as f:
            state = pickle.load(f)

        # Reconstruct index
        self._index = HNSWIndex(
            dim=state["dim"],
            max_elements=state["max_elements"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            metric=state["metric"],
        )

        self._index._graph = state["graph"]
        self._index._node_level = state["node_level"]
        self._index._entry_point = state["entry_point"]
        self._index._max_level = state["max_level"]
        self._index._deleted = state["deleted"]
        self._index._metadata = state["metadata_map"]
        self._id_order = state["id_order"]

        # Load vectors
        vec_array = np.load(str(self._vectors_path), allow_pickle=False)
        active_ids = [vid for vid in self._id_order if vid not in self._index._deleted]

        if vec_array.size > 0:
            for i, vid in enumerate(active_ids):
                if i < len(vec_array):
                    self._index._vectors[vid] = vec_array[i]

    def stats(self) -> dict[str, Any]:
        """Return statistics about the store.

        Returns:
            Dict with count, dim, metric, max_elements, layers.
        """
        return {
            "count": len(self._index),
            "dim": self._index.dim,
            "metric": self._index.metric,
            "max_elements": self._index.max_elements,
            "layers": len(self._index._graph),
            "total_inserted": len(self._index._vectors),
            "deleted": len(self._index._deleted),
        }

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load an existing VectorStore from disk.

        Args:
            path: Directory containing the saved store.

        Returns:
            Loaded VectorStore instance.

        Raises:
            FileNotFoundError: If the path doesn't contain a valid store.
        """
        p = Path(path)
        if not (p / "graph.pkl").exists():
            raise FileNotFoundError(f"No VectorStore found at {path}")
        # dim=0 is fine because _load() will read it from the pickle
        store = cls.__new__(cls)
        store.path = p
        store._vectors_path = p / "vectors.npy"
        store._graph_path = p / "graph.pkl"
        store._metadata_path = p / "metadata.db"
        store._load()
        store._init_metadata_db()
        return store

    def __len__(self) -> int:
        """Return number of active vectors."""
        return len(self._index)

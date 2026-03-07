"""HNSW (Hierarchical Navigable Small World) index implementation.

Based on the paper:
    "Efficient and robust approximate nearest neighbor using Hierarchical
     Navigable Small World graphs" (Malkov & Yashunin, 2018)

This is a from-scratch implementation -- no FAISS, no ChromaDB.
"""

from __future__ import annotations

import heapq
import math
import random
import threading
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from citadel_vector.distance import (
    cosine_distance,
    euclidean_distance,
    dot_product_distance,
)

# Map metric names to distance functions
_METRIC_FNS: dict[str, Callable] = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
    "dot": dot_product_distance,
}


class HNSWIndex:
    """Hierarchical Navigable Small World graph index for approximate nearest neighbor search.

    Args:
        dim: Dimensionality of vectors.
        max_elements: Maximum number of elements the index can hold.
        M: Maximum number of connections per node per layer.
        ef_construction: Beam width during index construction.
        metric: Distance metric -- "cosine", "euclidean", or "dot".
    """

    def __init__(
        self,
        dim: int,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        metric: str = "cosine",
    ) -> None:
        if metric not in _METRIC_FNS:
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(_METRIC_FNS.keys())}")

        self.dim: int = dim
        self.max_elements: int = max_elements
        self.M: int = M
        self.M_max0: int = 2 * M  # Max connections at layer 0 (denser)
        self.ef_construction: int = ef_construction
        self.metric: str = metric
        self._dist_fn: Callable = _METRIC_FNS[metric]

        # mL = 1 / ln(M), used for random level generation
        self._mL: float = 1.0 / math.log(M)

        # Storage
        self._vectors: dict[Any, npt.NDArray[np.floating]] = {}
        self._metadata: dict[Any, Optional[dict]] = {}
        self._deleted: set[Any] = set()

        # Graph: _graph[layer][node_id] = set of neighbor node_ids
        self._graph: list[dict[Any, set[Any]]] = []
        # Level assigned to each node
        self._node_level: dict[Any, int] = {}

        # Entry point: the node with the highest layer
        self._entry_point: Optional[Any] = None
        self._max_level: int = -1

        # Thread safety for concurrent adds
        self._lock = threading.Lock()

    def _random_level(self) -> int:
        """Assign a random layer using exponential distribution.

        level = floor(-ln(uniform()) * mL)
        """
        return int(-math.log(random.random()) * self._mL)

    def _distance(self, a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> float:
        """Compute distance between two vectors using the configured metric."""
        return self._dist_fn(a, b)

    def _search_layer(
        self,
        query: npt.NDArray[np.floating],
        entry_points: list[Any],
        ef: int,
        layer: int,
    ) -> list[tuple[float, Any]]:
        """Greedy beam search within a single layer.

        Args:
            query: Query vector.
            entry_points: Starting node IDs.
            ef: Beam width (number of candidates to track).
            layer: Which layer to search.

        Returns:
            List of (distance, node_id) tuples, sorted ascending by distance.
        """
        visited: set[Any] = set(entry_points)

        # candidates: min-heap of (distance, id) -- closest first for expansion
        # Deleted nodes ARE added to candidates (for graph traversal) but NOT to results.
        candidates: list[tuple[float, Any]] = []
        # results: max-heap of (-distance, id) -- furthest first for pruning
        results: list[tuple[float, Any]] = []

        for ep in entry_points:
            dist = self._distance(query, self._vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            if ep not in self._deleted:
                heapq.heappush(results, (-dist, ep))

        while candidates:
            # Get closest candidate
            c_dist, c_id = heapq.heappop(candidates)

            # Get furthest result distance (or infinity if no results yet)
            f_dist = -results[0][0] if results else float("inf")

            # If closest candidate is further than furthest result AND we have enough, stop
            if c_dist > f_dist and len(results) >= ef:
                break

            # Expand neighbors of c_id in this layer
            neighbors = self._graph[layer].get(c_id, set())
            for n_id in neighbors:
                if n_id in visited:
                    continue
                visited.add(n_id)

                n_dist = self._distance(query, self._vectors[n_id])

                # Always consider for traversal; only add to results if not deleted
                f_dist = -results[0][0] if results else float("inf")

                if n_dist < f_dist or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    if n_id not in self._deleted:
                        heapq.heappush(results, (-n_dist, n_id))
                        if len(results) > ef:
                            heapq.heappop(results)

        # Convert results from max-heap to sorted list
        output = [(-neg_dist, node_id) for neg_dist, node_id in results]
        output.sort(key=lambda x: x[0])
        return output

    def _select_neighbors(
        self,
        candidates: list[tuple[float, Any]],
        M: int,
    ) -> list[tuple[float, Any]]:
        """Select M nearest neighbors from candidates (simple selection).

        Args:
            candidates: List of (distance, node_id) tuples.
            M: Max number of neighbors to select.

        Returns:
            At most M nearest candidates, sorted by distance.
        """
        candidates_sorted = sorted(candidates, key=lambda x: x[0])
        return candidates_sorted[:M]

    def add(
        self,
        vector: npt.NDArray[np.floating],
        id: Any,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert a vector into the index.

        Args:
            vector: Vector to insert, shape (dim,).
            id: Unique identifier for this vector.
            metadata: Optional metadata dict associated with this vector.

        Raises:
            ValueError: If vector dimension doesn't match or index is full.
        """
        vector = np.asarray(vector, dtype=np.float64).ravel()
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} doesn't match index dimension {self.dim}"
            )

        with self._lock:
            if id in self._vectors and id not in self._deleted:
                raise ValueError(f"ID '{id}' already exists in the index")

            # If re-inserting a deleted ID, un-delete it
            self._deleted.discard(id)

            active_count = len(self._vectors) - len(self._deleted)
            if active_count >= self.max_elements:
                raise ValueError(
                    f"Index is full ({self.max_elements} elements). "
                    f"Cannot add more vectors."
                )

            # Store vector and metadata
            self._vectors[id] = vector
            self._metadata[id] = metadata

            # Assign random level
            level = self._random_level()
            self._node_level[id] = level

            # Ensure graph has enough layers
            while len(self._graph) <= level:
                self._graph.append({})

            # If this is the first node, make it the entry point
            if self._entry_point is None:
                self._entry_point = id
                self._max_level = level
                for lyr in range(level + 1):
                    self._graph[lyr][id] = set()
                return

            # Phase 1: Traverse from top layer down to level+1, greedy (ef=1)
            curr_entry = self._entry_point
            for lyr in range(self._max_level, level, -1):
                if lyr < len(self._graph):
                    results = self._search_layer(vector, [curr_entry], 1, lyr)
                    if results:
                        curr_entry = results[0][1]

            # Phase 2: From min(level, max_level) down to layer 0, insert with ef_construction
            entry_points_for_layer = [curr_entry]
            for lyr in range(min(level, self._max_level), -1, -1):
                # Search this layer
                candidates = self._search_layer(
                    vector, entry_points_for_layer, self.ef_construction, lyr
                )

                # Select neighbors
                M_lyr = self.M_max0 if lyr == 0 else self.M
                neighbors = self._select_neighbors(candidates, M_lyr)

                # Add the new node to this layer
                if lyr not in range(len(self._graph)):
                    continue
                self._graph[lyr][id] = set()

                # Connect new node to its neighbors (bidirectional)
                for _dist, neighbor_id in neighbors:
                    self._graph[lyr][id].add(neighbor_id)
                    if neighbor_id not in self._graph[lyr]:
                        self._graph[lyr][neighbor_id] = set()
                    self._graph[lyr][neighbor_id].add(id)

                    # Prune neighbor's connections if they exceed M
                    if len(self._graph[lyr][neighbor_id]) > M_lyr:
                        # Keep only M_lyr nearest neighbors
                        neighbor_vec = self._vectors[neighbor_id]
                        scored = []
                        for nn_id in self._graph[lyr][neighbor_id]:
                            if nn_id in self._deleted:
                                continue
                            nn_dist = self._distance(neighbor_vec, self._vectors[nn_id])
                            scored.append((nn_dist, nn_id))
                        pruned = self._select_neighbors(scored, M_lyr)
                        self._graph[lyr][neighbor_id] = {nid for _, nid in pruned}

                # Use the closest candidates as entry points for next layer
                entry_points_for_layer = [nid for _, nid in neighbors[:1]] if neighbors else entry_points_for_layer

            # Update entry point if new node has a higher level
            if level > self._max_level:
                self._entry_point = id
                self._max_level = level
                # Ensure the new node exists in all layers up to its level
                for lyr in range(len(self._graph)):
                    if lyr <= level and id not in self._graph[lyr]:
                        self._graph[lyr][id] = set()

    def search(
        self,
        query: npt.NDArray[np.floating],
        k: int = 10,
        ef_search: int = 50,
        filter_fn: Optional[Callable[[Optional[dict]], bool]] = None,
    ) -> list[tuple[Any, float, Optional[dict]]]:
        """Find k approximate nearest neighbors.

        Args:
            query: Query vector, shape (dim,).
            k: Number of nearest neighbors to return.
            ef_search: Beam width during search (higher = more accurate but slower).
            filter_fn: Optional callable (metadata) -> bool. Only results where
                       filter_fn returns True are included.

        Returns:
            List of (id, distance, metadata) tuples sorted by distance (ascending).
        """
        query = np.asarray(query, dtype=np.float64).ravel()
        if query.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {query.shape[0]} doesn't match index dimension {self.dim}"
            )

        if self._entry_point is None:
            return []

        # Use at least k for ef_search
        ef_search = max(ef_search, k)

        # Phase 1: Traverse from top layer down to layer 1 with ef=1
        curr_entry = self._entry_point
        for lyr in range(self._max_level, 0, -1):
            results = self._search_layer(query, [curr_entry], 1, lyr)
            if results:
                curr_entry = results[0][1]

        # Phase 2: Search layer 0 with full ef_search
        candidates = self._search_layer(query, [curr_entry], ef_search, 0)

        # Apply filter and collect top-k
        results: list[tuple[Any, float, Optional[dict]]] = []
        for dist, node_id in candidates:
            if node_id in self._deleted:
                continue
            meta = self._metadata.get(node_id)
            if filter_fn is not None and not filter_fn(meta):
                continue
            results.append((node_id, dist, meta))
            if len(results) >= k:
                break

        return results

    def batch_add(
        self,
        vectors: npt.NDArray[np.floating],
        ids: list[Any],
        metadatas: Optional[list[Optional[dict]]] = None,
    ) -> None:
        """Insert multiple vectors at once.

        Args:
            vectors: Matrix of vectors, shape (n, dim).
            ids: List of unique identifiers, length n.
            metadatas: Optional list of metadata dicts, length n.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of IDs ({len(ids)}) doesn't match number of vectors ({vectors.shape[0]})"
            )

        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError(
                f"Number of metadatas ({len(metadatas)}) doesn't match number of IDs ({len(ids)})"
            )

        for i, vid in enumerate(ids):
            meta = metadatas[i] if metadatas is not None else None
            self.add(vectors[i], vid, meta)

    def batch_search(
        self,
        queries: npt.NDArray[np.floating],
        k: int = 10,
        ef_search: int = 50,
    ) -> list[list[tuple[Any, float, Optional[dict]]]]:
        """Search for multiple queries at once.

        Args:
            queries: Matrix of query vectors, shape (n, dim).
            k: Number of nearest neighbors per query.
            ef_search: Beam width during search.

        Returns:
            List of result lists, one per query. Each inner list contains
            (id, distance, metadata) tuples sorted by distance.
        """
        queries = np.asarray(queries, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        return [self.search(q, k=k, ef_search=ef_search) for q in queries]

    def delete(self, id: Any) -> None:
        """Mark a vector as deleted (lazy deletion).

        The vector is excluded from search results but not physically removed
        from the graph. This preserves graph connectivity.

        Args:
            id: The ID of the vector to delete.

        Raises:
            KeyError: If the ID does not exist in the index.
        """
        if id not in self._vectors:
            raise KeyError(f"ID '{id}' not found in the index")
        self._deleted.add(id)

    def __len__(self) -> int:
        """Return the number of active (non-deleted) vectors in the index."""
        return len(self._vectors) - len(self._deleted)

    def __contains__(self, id: Any) -> bool:
        """Check if an ID exists and is not deleted."""
        return id in self._vectors and id not in self._deleted

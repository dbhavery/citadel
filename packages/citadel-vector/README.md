# citadel-vector

A from-scratch vector similarity search engine implementing the HNSW (Hierarchical Navigable Small World) algorithm. No ChromaDB, no FAISS -- the actual algorithm, implemented in Python with NumPy.

Based on: *"Efficient and robust approximate nearest neighbor using Hierarchical Navigable Small World graphs"* (Malkov & Yashunin, 2018).

## Quick Start

```python
import numpy as np
from citadel_vector import HNSWIndex

# Create an index
index = HNSWIndex(dim=128, max_elements=10_000, M=16, metric="cosine")

# Add vectors
index.add(np.random.rand(128), id="doc_1", metadata={"title": "Hello"})
index.add(np.random.rand(128), id="doc_2", metadata={"title": "World"})

# Search
results = index.search(np.random.rand(128), k=5)
for id, distance, metadata in results:
    print(f"{id}: {distance:.4f} -- {metadata}")

# Filtered search
results = index.search(
    query,
    k=5,
    filter_fn=lambda m: m and m.get("title") == "Hello"
)

# Batch operations
vectors = np.random.rand(100, 128)
ids = [f"doc_{i}" for i in range(100)]
index.batch_add(vectors, ids)
```

## Persistence

```python
from citadel_vector import VectorStore

# Create a persistent store
store = VectorStore(path="./my_vectors", dim=128)
store.add(np.random.rand(128), "doc_1", metadata={"source": "web"})
store.save()

# Load later
store = VectorStore.load("./my_vectors")
results = store.search(query, k=10)
```

Storage format:
- `vectors.npy` -- vector data (NumPy)
- `graph.pkl` -- HNSW graph structure
- `metadata.db` -- metadata (SQLite)

## Performance Characteristics

- **Build time:** O(n * log(n)) -- each insertion is O(log(n)) with beam search
- **Search time:** O(log(n)) -- hierarchical traversal from top layer to bottom
- **Memory:** O(n * M) for graph edges + O(n * dim) for vectors
- **Recall@10:** >90% with default parameters (M=16, ef_construction=200, ef_search=50)

Key tuning parameters:
- `M` -- connections per node. Higher = better recall, more memory. Default: 16.
- `ef_construction` -- beam width during build. Higher = better graph, slower build. Default: 200.
- `ef_search` -- beam width during search. Higher = better recall, slower search. Default: 50.

## REST API (Optional)

Install with server dependencies:

```bash
pip install citadel-vector[server]
```

Start the server:

```bash
citadel-vector serve --host 127.0.0.1 --port 8100
```

Endpoints:
- `POST /collections` -- create a collection
- `POST /collections/{name}/add` -- add vectors
- `POST /collections/{name}/search` -- search
- `GET /collections/{name}/stats` -- stats
- `GET /health` -- health check

## Distance Metrics

- `cosine` (default) -- 1 - cosine_similarity. Best for normalized embeddings.
- `euclidean` -- L2 distance. Best for raw feature vectors.
- `dot` -- negative dot product. Best when magnitude matters.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

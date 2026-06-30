# Vector Index

High-performance vector similarity search with HNSW, IVF, and product quantization.

## Features

- **HNSW**: Hierarchical Navigable Small World graphs
- **IVF**: Inverted file index with clustering
- **Product Quantization**: Compressed vector storage
- **Hybrid Search**: Combined dense + sparse retrieval
- **Filtering**: Metadata-based filtering

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from vecindex import HNSWIndex

# Create index
index = HNSWIndex(
    dim=384,
    max_elements=100000,
    ef_construction=200,
    M=16
)

# Add vectors
vectors = np.random.randn(10000, 384).astype(np.float32)
ids = list(range(10000))
index.add(vectors, ids)

# Search
query = np.random.randn(384).astype(np.float32)
results = index.search(query, k=10)
# Returns: [(id, distance), ...]
```

## Index Types

### HNSW (Recommended for most cases)

```python
from vecindex import HNSWIndex

index = HNSWIndex(
    dim=384,
    max_elements=1000000,
    ef_construction=200,  # Build-time accuracy
    M=16,                 # Connections per node
    ef_search=100         # Search-time accuracy
)
```

### IVF (For very large datasets)

```python
from vecindex import IVFIndex

index = IVFIndex(
    dim=384,
    n_lists=1000,    # Number of clusters
    n_probes=10      # Clusters to search
)
index.train(training_vectors)
index.add(vectors, ids)
```

### Product Quantization (Memory efficient)

```python
from vecindex import PQIndex

index = PQIndex(
    dim=384,
    n_subvectors=48,  # Must divide dim
    n_bits=8          # Bits per subvector
)
index.train(training_vectors)
index.add(vectors, ids)
```

## Benchmarks

| Index | Build Time | Query Time | Memory |
|-------|-----------|------------|--------|
| HNSW | 45s | 0.5ms | 100% |
| IVF | 30s | 2ms | 100% |
| PQ | 60s | 5ms | 25% |

## Testing

```bash
pytest tests/ -v  # 130 tests
```

# Vector Index

## Overview

Vector Index (`vecindex`) is a FAISS-lite approximate nearest neighbor (ANN) search
library implemented from scratch in Python over NumPy. It is a teaching-grade but fully
functional implementation of the core data structures that power modern vector databases
and retrieval systems: inverted-file partitioning (IVF), hierarchical navigable
small-world graphs (HNSW), and product quantization (PQ), plus the exact brute-force
baseline (Flat) that the approximate methods are measured against.

The library exists to make the algorithms legible. Every index, quantizer, and distance
kernel is written in plain NumPy (with an optional Numba fast path) rather than delegating
to a compiled backend, so the trade-offs between recall, speed, and memory are visible in
the source. The same library can be used to actually index and search synthetic datasets
of a few thousand to a few hundred thousand vectors in process.

Concretely, the project teaches:

- **The k-NN search abstraction.** A single `Index` interface — `train`, `add`, `search`,
  `ntotal` — unifies an exact method and three approximate ones, so they are
  interchangeable in benchmarks and rerank pipelines.
- **Coarse quantization with IVF.** How k-means centroids partition a vector space into
  cells, how `nprobe` trades recall for speed, and why a vector only ever competes against
  the contents of a few cells instead of the whole database.
- **Graph-based ANN with HNSW.** How a multi-layer proximity graph is constructed with
  randomized levels, greedy descent, and neighbor pruning, and how `ef_construction` and
  `ef_search` control build quality and query accuracy.
- **Compression with product quantization.** How splitting a vector into subspaces and
  quantizing each independently shrinks storage by an order of magnitude, and how
  asymmetric distance computation (ADC) with precomputed lookup tables keeps search fast
  on compressed codes.
- **Metric handling.** How L2, inner product, and cosine are reconciled under a single
  "lower is better" convention so the same top-k machinery serves all three.
- **The cost of approximation.** How recall@k is measured against brute-force ground
  truth, and how each knob (`nprobe`, `ef_search`, `nbits`, `M`) moves the recall/speed/memory
  frontier.

Scope is deliberately bounded. Everything is in-memory: there is no persistence layer, no
memory-mapped storage, and no GPU backend. Inverted lists are Python lists of tuples, which
is clear but not cache-optimal. The SIMD module accelerates distance and top-k kernels via
Numba when it is available and otherwise falls back to NumPy. These boundaries are stated
plainly in the README's "What's Real vs Simulated" section.

## Architecture

```mermaid
flowchart TD
    subgraph API (Public package vecindex)
        EXPORTS(Re-exported symbols)
    end
    subgraph INDEXES (index/indexes.py)
        ABC(Index ABC)
        FLAT(FlatIndex)
        IVF(IVFIndex)
        HNSW(HNSWIndex)
        IVFPQ(IVFPQIndex)
    end
    subgraph CORE (core/vectors.py)
        METRICS(compute_distance L2 IP cosine)
        TOPK(topk selection)
        KMEANS(kmeans clustering)
        STORE(VectorStore)
    end
    subgraph QUANT (quantize/pq.py)
        PQ(ProductQuantizer)
        OPQ(OPQ rotation)
        SQ(ScalarQuantizer)
        BQ(BinaryQuantizer)
    end
    subgraph SEARCHU (search/search.py)
        BATCH(BatchSearcher)
        HYBRID(HybridSearcher)
        RERANK(RerankSearcher)
        FACTORY(IndexFactory and build_index)
        BENCH(benchmark_index)
    end
    subgraph SIMD (simd/ops.py)
        KERNELS(Numba kernels)
        FALLBACK(NumPy fallback)
    end

    EXPORTS --> ABC
    EXPORTS --> PQ
    EXPORTS --> BATCH
    EXPORTS --> KERNELS
    ABC --> FLAT
    ABC --> IVF
    ABC --> HNSW
    ABC --> IVFPQ
    FLAT --> METRICS
    IVF --> KMEANS
    IVF --> METRICS
    HNSW --> METRICS
    IVFPQ --> KMEANS
    IVFPQ --> PQ
    METRICS --> TOPK
    FACTORY --> ABC
    BENCH --> METRICS
    KERNELS --> FALLBACK
```

The library is organized into five cooperating layers, each a Python subpackage under
`src/vecindex/`:

- **`core`** holds primitives every index needs: the `MetricType` enum, the distance
  functions, a unified `compute_distance` that normalizes all metrics to "lower is better",
  the `topk` selector, a `kmeans` clustering routine, a `pca` helper, the `SearchResult`
  dataclass, and a simple `VectorStore` for in-memory vector/ID management.
- **`index`** defines the `Index` abstract base class and the four concrete indexes. Each
  index composes core primitives — `FlatIndex` wraps a `VectorStore`, `IVFIndex` and
  `IVFPQIndex` call `kmeans` to learn centroids, and `HNSWIndex` builds its own adjacency
  structure but uses `compute_distance` for every comparison.
- **`quantize`** implements the compression schemes. `ProductQuantizer` is the workhorse;
  `OPQ` subclasses it to add a learned rotation; `ScalarQuantizer` and `BinaryQuantizer`
  are simpler per-dimension schemes. `compute_recall` lives here as the accuracy yardstick.
- **`search`** provides higher-level query orchestration on top of any `Index`: batching,
  ID filtering, hybrid score fusion, two-stage rerank, range search, plus the `build_index`
  / `IndexFactory` constructors and `benchmark_index`.
- **`simd`** is an optional acceleration layer. When Numba imports successfully,
  `SIMD_AVAILABLE` is `True` and the public functions dispatch to JIT-compiled kernels;
  otherwise they run equivalent NumPy code. `SIMDVectorOps` is a class-based brute-force
  searcher with cached norms for cosine.

The package's `__init__.py` re-exports the public surface so callers write
`from vecindex import HNSWIndex` regardless of which submodule a symbol lives in.

## Core Components

### Index abstract base class

`Index` (`index/indexes.py`) fixes the contract that every index honors:

```python
class Index(ABC):
    def __init__(self, dim: int, metric: MetricType = MetricType.L2):
        self.dim = dim
        self.metric = metric
        self.is_trained = False

    @abstractmethod
    def train(self, vectors: np.ndarray): ...

    @abstractmethod
    def add(self, vectors: np.ndarray): ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> SearchResult: ...

    @property
    @abstractmethod
    def ntotal(self) -> int: ...

    def __len__(self) -> int:
        return self.ntotal
```

`train` learns any data-dependent structure (centroids, codebooks); methods that need no
training (Flat, HNSW) implement it as a no-op and set `is_trained = True` in the
constructor. `add` ingests vectors, `search` returns a `SearchResult`, and `ntotal` is a
property so subclasses can compute it from whatever storage they use. This uniformity is
what lets `IndexFactory`, `build_index`, `benchmark_index`, and `RerankSearcher` treat all
indexes interchangeably.

### FlatIndex

`FlatIndex` is the exact baseline. It stores vectors in a `VectorStore` and answers every
query by computing all distances and taking the top k:

```python
def search(self, query: np.ndarray, k: int) -> SearchResult:
    query = np.asarray(query)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    distances = compute_distance(query, self.store.vectors, self.metric)
    if distances.ndim == 1:
        distances = distances.reshape(1, -1)
    indices, dists = topk(distances, k)
    ids = self.store.ids[indices]
    return SearchResult(ids.squeeze(), dists.squeeze())
```

It is O(n) per query and needs no training, so `is_trained` is `True` from construction and
`train` does nothing. FlatIndex is the source of ground truth in the test suite: the
approximate indexes are scored by how many of FlatIndex's results they recover. The
`test_search_exact_correctness` test asserts FlatIndex's IDs match a brute-force
`np.argsort` exactly.

### IVFIndex

`IVFIndex` implements the inverted-file method. Training runs k-means to learn `nlist`
centroids:

```python
def train(self, vectors: np.ndarray):
    vectors = np.asarray(vectors)
    self.centroids, _ = kmeans(vectors, self.nlist)
    self.is_trained = True
```

Adding a vector assigns it to its nearest centroid and appends `(id, vector)` to that
centroid's inverted list:

```python
def add(self, vectors: np.ndarray):
    if not self.is_trained:
        raise RuntimeError("Index must be trained before adding")
    ...
    distances = l2_distance(vectors, self.centroids)
    assignments = np.argmin(distances, axis=1)
    for i, vec in enumerate(vectors):
        list_id = assignments[i]
        self.inverted_lists[list_id].append((self._next_id, vec))
        self._next_id += 1
```

Search finds the `nprobe` nearest centroids to the query, gathers the candidate vectors
from those lists, and runs an exact top-k over just that subset:

```python
def search(self, query: np.ndarray, k: int) -> SearchResult:
    ...
    centroid_dists = l2_distance(query, self.centroids)
    partition_indices, _ = topk(centroid_dists, self.nprobe)
    ...
    for p in partitions:
        for vid, vec in self.inverted_lists[p]:
            candidates_ids.append(vid)
            candidates_vecs.append(vec)
    distances = compute_distance(query, np.array(candidates_vecs), self.metric)
    idx, dists = topk(distances[0], k)
```

The recall/speed trade-off is governed entirely by `nprobe`: probing one cell is fast but
may miss neighbors that fell into an adjacent cell; probing more cells raises recall toward
the exact answer at higher cost. `test_search_nprobe_affects_results` confirms recall is
monotonic in `nprobe`.

### HNSWIndex

`HNSWIndex` builds a multi-layer proximity graph. Each node is placed on a randomly chosen
top level drawn from an exponential distribution scaled by `ml = 1/ln(M)`:

```python
def _get_random_level(self) -> int:
    return int(-np.log(np.random.random()) * self.ml)
```

Higher levels are sparse and act as express lanes; level 0 contains every node. Insertion
(`_add_single`) proceeds in two phases. First it greedily descends from the current entry
point through the upper levels using a width-1 beam to reach the neighborhood of the new
node. Then, from the node's assigned level down to layer 0, it runs `_search_layer` with
width `ef_construction` to collect candidates, selects up to `M` (or `M0 = 2*M` at layer 0)
neighbors, and wires bidirectional edges:

```python
for layer in range(min(node_level, self.max_layer), -1, -1):
    results = self._search_layer(vector, ep, self.ef_construction, layer)
    M = self.M0 if layer == 0 else self.M
    neighbors = self._select_neighbors(node_id, results, M)
    self.graphs[layer][node_id] = neighbors
    for neighbor in neighbors:
        self.graphs[layer][neighbor].append(node_id)
        if len(self.graphs[layer][neighbor]) > M:
            # keep only the M closest neighbors
            dists = [(self._distance(neighbor, nn), nn)
                     for nn in self.graphs[layer][neighbor]]
            dists.sort()
            self.graphs[layer][neighbor] = [d[1] for d in dists[:M]]
    ep = [r[1] for r in results]
```

The pruning step keeps each node's degree bounded by re-selecting its closest neighbors
whenever a new edge pushes it over the limit — this is what stops the graph from degrading
into a dense, slow-to-traverse blob.

`_search_layer` is the heart of both insertion and query. It maintains a min-heap of
candidates to explore and a bounded max-heap of the best `ef` results found so far,
stopping when the nearest unexplored candidate is farther than the current worst result:

```python
def _search_layer(self, query, entry_points, ef, layer):
    visited = set(entry_points)
    candidates = []   # min-heap by distance
    results = []      # max-heap (negated) of size <= ef
    for ep in entry_points:
        dist = self._distance_to_query(query, ep)
        heapq.heappush(candidates, (dist, ep))
        heapq.heappush(results, (-dist, ep))
    while candidates:
        dist_c, c = heapq.heappop(candidates)
        if results and dist_c > -results[0][0]:
            break
        for neighbor in self.graphs[layer][c]:
            if neighbor not in visited:
                visited.add(neighbor)
                dist_n = self._distance_to_query(query, neighbor)
                if len(results) < ef or dist_n < -results[0][0]:
                    heapq.heappush(candidates, (dist_n, neighbor))
                    heapq.heappush(results, (-dist_n, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
    return [(-d, i) for d, i in results]
```

Query (`search`) mirrors insertion: descend the upper levels with a width-1 beam from the
entry point, then run `_search_layer` at level 0 with width `ef_search`, sort, and return
the top k. `ef_search` is the query-time recall knob; `test_ef_search_affects_recall`
asserts higher `ef_search` never lowers recall, and `test_search_approximate_recall`
asserts the index reaches at least 70% recall under its default test parameters.

### IVFPQIndex

`IVFPQIndex` combines coarse IVF partitioning with PQ compression of the residuals, which
is the standard recipe for memory-efficient billion-scale search. Training learns coarse
centroids, computes each training vector's residual against its assigned coarse centroid,
then trains one PQ codebook per subspace on those residuals:

```python
def train(self, vectors: np.ndarray):
    self.coarse_centroids, _ = kmeans(vectors, self.nlist)
    distances = l2_distance(vectors, self.coarse_centroids)
    assignments = np.argmin(distances, axis=1)
    residuals = vectors - self.coarse_centroids[assignments]
    self.pq_centroids = np.zeros((self.M, self.ksub, self.dsub))
    for m in range(self.M):
        sub_vectors = residuals[:, m * self.dsub:(m + 1) * self.dsub]
        self.pq_centroids[m], _ = kmeans(sub_vectors, self.ksub)
    self.is_trained = True
```

The constructor asserts `dim % M == 0` so each of the `M` subquantizers owns an equal-width
slice `dsub = dim // M`, with `ksub = 2**nbits` centroids per subquantizer. Adding a vector
re-assigns it to a coarse cell, computes its residual, encodes that residual to `M` byte
codes, and stores `(id, codes)` — never the original vector — in the inverted list.

Search precomputes a per-cell distance table mapping each subspace centroid to its squared
distance from the query residual, then scores every stored code by summing table lookups
(asymmetric distance computation):

```python
dist_table = np.zeros((self.M, self.ksub))
for m in range(self.M):
    sub_query = residual[m * self.dsub:(m + 1) * self.dsub]
    dist_table[m] = np.sum((self.pq_centroids[m] - sub_query) ** 2, axis=1)
for vid, codes in self.inverted_lists[p]:
    dist = sum(dist_table[m, codes[m]] for m in range(self.M))
    candidates.append((dist, vid))
```

This is the key efficiency: distances to encoded vectors are M table lookups and adds,
never a full vector subtraction. IVF-PQ is lossy by construction, so
`test_approximate_recall` only requires recall >= 0.3, while `test_compression_effect`
verifies each vector is stored as M bytes instead of `dim * 4`.

### Core distance and selection primitives

`core/vectors.py` provides the math every index leans on. `compute_distance` normalizes all
three metrics to a single "lower is better" convention by negating the two similarity
metrics:

```python
def compute_distance(a, b, metric):
    if metric == MetricType.L2:
        return l2_distance(a, b)
    elif metric == MetricType.IP:
        return -inner_product(a, b)
    elif metric == MetricType.COSINE:
        return -cosine_similarity(a, b)
```

`l2_distance` uses the expanded form `||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b`, which lets a
batch of distances be computed as two norm vectors plus one matrix product, then clamps to
zero (`np.maximum(dist, 0)`) before the square root to absorb the small negative values that
floating-point cancellation can produce when a query nearly coincides with a database
vector. `inner_product` is a plain `a @ b.T`, and `cosine_similarity` normalizes both sides
with an `1e-8` epsilon guard against zero vectors before the dot product. All three return a
squeezed array so a single query yields a 1-D result and a batch yields a 2-D matrix; the
indexes reshape defensively because of this.

`topk` performs an O(n) partial selection with `np.argpartition` and then sorts only the k
survivors, giving O(n + k log k) instead of a full O(n log n) sort. It handles both a 1-D
distance vector and a 2-D `(nq, n)` batch, and clamps k to the available count so requesting
more neighbors than exist never raises:

```python
def topk(distances, k):
    if distances.ndim == 1:
        k = min(k, len(distances))
        idx = np.argpartition(distances, k - 1)[:k]
        idx = idx[np.argsort(distances[idx])]
        return idx, distances[idx]
    ...
```

`kmeans` is a Lloyd's iteration: random initialization without replacement, assignment by
`argmin` over `l2_distance`, mean recomputation per cluster, an empty-cluster guard that
keeps the previous centroid when a cluster loses all its members, and a convergence test on
the centroid movement norm against `tol`:

```python
def kmeans(vectors, k, max_iter=100, tol=1e-4):
    idx = np.random.choice(n, k, replace=False)
    centroids = vectors[idx].copy()
    for _ in range(max_iter):
        assignments = np.argmin(l2_distance(vectors, centroids), axis=1)
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            new_centroids[i] = vectors[mask].mean(axis=0) if mask.sum() else centroids[i]
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return centroids, assignments
```

This one routine is reused three ways: IVF learns its `nlist` coarse centroids, IVF-PQ learns
both its coarse centroids and (per subspace) its PQ codebooks, and the standalone
`ProductQuantizer` trains every subquantizer with it. A `pca` helper performs
eigendecomposition of the covariance matrix and returns the top-`n_components` projection,
mean, and transformed vectors, available for dimensionality reduction before indexing.

`VectorStore` is an append-only in-memory store backing `FlatIndex`. It keeps a contiguous
`(ntotal, dim)` float array and a parallel int64 ID array, auto-increments IDs on `add`,
supports `add_with_ids` for caller-supplied IDs, `remove` by boolean mask, `get` by ID via
`searchsorted`, and `clear`. Because it is contiguous, FlatIndex's distance computation is a
single vectorized call rather than a Python loop.

### Quantizers

`ProductQuantizer` (`quantize/pq.py`) is the standalone PQ used inside `IVFPQIndex` and
exposed directly. `train` runs one k-means per subspace; `encode` returns `(n, M)` uint8
codes by nearest-centroid lookup per slice; `decode` reconstructs an approximate vector by
gathering the chosen centroids. It offers both `asymmetric_distance` (exact query against
encoded database, via a precomputed `(M, ksub)` table) and `symmetric_distance` (both sides
encoded), and reports `code_size` and `compression_ratio` properties.

The asymmetric path is the one that matters for search. `compute_distance_table` precomputes,
for a single query, the squared distance from each of its `M` subvectors to all `ksub`
centroids in the corresponding codebook, producing an `(M, ksub)` table. Scoring an encoded
vector is then `M` lookups and adds:

```python
def asymmetric_distance(self, query, codes):
    table = self.compute_distance_table(query)        # (M, ksub)
    distances = np.sum([table[m, codes[:, m]] for m in range(self.M)], axis=0)
    return np.sqrt(distances)
```

`symmetric_distance` instead sums centroid-to-centroid distances between two codes; it is
cheaper to precompute across a database but loses accuracy because the query is also
quantized. The `compression_ratio` property reports `(dim * 4) / (M * nbits / 8)` — for
`dim=64, M=8, nbits=8` that is 32x.

`OPQ` subclasses `ProductQuantizer` and learns a rotation matrix via alternating
optimization. On each iteration it trains PQ on the rotated data, encodes and decodes to get
a reconstruction, and updates the rotation through a Procrustes step that finds the optimal
orthogonal alignment between the original and reconstructed vectors:

```python
def train(self, vectors, n_iter=10):
    self.rotation = np.eye(d, dtype=np.float32)
    rotated = vectors.copy()
    for i in range(n_iter):
        super().train(rotated)
        reconstructed = self.decode(self.encode(rotated))
        U, _, Vt = np.linalg.svd(vectors.T @ reconstructed)
        self.rotation = (U @ Vt).T
        rotated = vectors @ self.rotation.T
    super().train(rotated)
```

`encode` rotates before delegating to the parent and `decode` inverse-rotates afterward, so
the rotation is transparent to callers. The intent is to align the data so each PQ subspace
carries comparable variance, reducing total quantization error.

`ScalarQuantizer` quantizes each dimension independently into `2**nbits` levels using learned
per-dimension min/max ranges, normalizing to `[0, 1]`, scaling to `[0, levels-1]`, and
clipping; it stores codes as uint8 (nbits <= 8) or uint16. `BinaryQuantizer` thresholds each
dimension at its training median, packs the resulting bits into bytes with explicit shift/OR,
exposes a static `hamming_distance` over packed codes, and offers an approximate `decode`
that maps each bit back to threshold ± 0.5. `compute_recall` computes recall@k as the average
fraction of true neighbors present in the retrieved set:

```python
def compute_recall(ground_truth, results, k):
    correct = sum(len(set(ground_truth[i, :k]) & set(results[i, :k]))
                  for i in range(nq))
    return correct / (nq * k)
```

### Search utilities

`search/search.py` layers query orchestration over any `Index`. `BatchSearcher` loops a
single index over many queries with optional chunking and offers `search_with_filter`,
which over-fetches and keeps only results whose IDs are in an allowed set. `HybridSearcher`
registers multiple named indexes with weights and fuses their results by weighted
reciprocal rank — the mechanism for blending, say, a dense and a sparse retriever.
`RerankSearcher` runs a fast first-stage index to fetch `rerank_k` candidates, then
re-scores them exactly against the original vectors and returns the top k, recovering
accuracy lost to approximation. `RangeSearcher` returns all neighbors within a radius.

`HybridSearcher`'s fusion is reciprocal-rank with per-index weights: each result at rank `i`
contributes `weight / (i + 1)` to its ID's score, scores accumulate across indexes, and the
top k by total score are returned. This rank-based blending sidesteps the problem that
different indexes (or dense vs sparse retrievers) produce distances on incomparable scales.
`RerankSearcher` is the canonical two-stage pattern: the first stage fetches `rerank_k`
candidates cheaply, then `compute_distance` re-scores those candidates against the original
full-precision vectors and returns the exact top k among them — recovering accuracy that a
compressed or graph-based first stage gave up.

`build_index` constructs, trains (where needed), and populates an index in one call from a
type string, defaulting parameters per type (`nlist=100`, `nprobe=10`, `M=16`, etc.).
`IndexFactory.create` parses FAISS-style description strings: `"Flat"` -> `FlatIndex`,
`"IVF100,Flat"` -> `IVFIndex(nlist=100)`, `"IVF100,PQ8"` -> `IVFPQIndex(nlist=100, M=8)`, and
`"HNSW32"` -> `HNSWIndex(M=32)`. `benchmark_index` times one query at a time, computes
`recall@k` against supplied ground truth, and returns a dict of `recall@k`, `qps`, and
`latency_ms`.

### SIMD acceleration

`simd/ops.py` provides Numba-JIT kernels for L2, inner-product, and cosine distance, top-k
selection, and fused distance+top-k batch search, each guarded so that when Numba is absent
the public functions (`simd_l2_distance`, `simd_topk`, `simd_l2_batch`, `batch_search`,
etc.) run equivalent NumPy. `SIMD_AVAILABLE` reflects whether the JIT path is active. Helper
functions ensure inputs are C-contiguous and aligned before entering kernels.
`SIMDVectorOps` is a class-based brute-force searcher that holds a database, precomputes and
caches normalized vectors for cosine, and exposes `search`, `range_search`, and `add`.

### End-to-end query lifecycle

The four indexes share a vocabulary but diverge in how a single `search(query, k)` call
spends its time. Reading them side by side clarifies the design:

- **Flat** does no pruning. It calls `compute_distance(query, all_vectors, metric)` once,
  hands the full distance row to `topk`, and maps the resulting positions back to stored
  IDs. The entire cost is one vectorized distance pass over `ntotal` vectors plus a partial
  sort — exact, simple, and linear.
- **IVF** prunes by cell. It computes distances to the `nlist` centroids, takes the
  `nprobe` nearest with `topk`, concatenates the raw vectors from those cells into a
  candidate set, and runs an exact `compute_distance` + `topk` over only that set. Work is
  proportional to the candidate count, not the database size, but the answer can miss
  neighbors that landed in an unprobed cell.
- **HNSW** prunes by graph navigation. It starts at the entry point on the top layer,
  greedily walks toward the query with a width-1 beam down to layer 1, then runs
  `_search_layer` at layer 0 with width `ef_search`. It never enumerates the database; it
  only touches nodes reachable through the graph, which is what delivers the logarithmic
  comparison count.
- **IVF-PQ** prunes by cell and then scores on compressed codes. After selecting `nprobe`
  cells it builds a per-cell `(M, ksub)` distance table from the query residual and scores
  each stored code by summing M table lookups, never decompressing a vector. The final
  distances are `sqrt` of the summed squared sub-distances.

All four converge on the same output type — a `SearchResult` of IDs and ascending distances
— so a caller (or a `RerankSearcher`, or `benchmark_index`) can swap one index for another
without changing the surrounding code. The differences are entirely in the recall/speed/memory
trade-off each one chooses.

## Data Structures

The metric enum and the result type are the two values that cross every API boundary:

```python
class MetricType(Enum):
    L2 = "l2"                 # Euclidean distance
    IP = "inner_product"      # inner-product similarity
    COSINE = "cosine"         # cosine similarity


@dataclass
class SearchResult:
    ids: np.ndarray           # vector IDs
    distances: np.ndarray     # distances/similarities (lower is better)

    @property
    def num_results(self) -> int:
        return len(self.ids)
```

`VectorStore` backs the exact index and any code that needs raw storage:

```python
class VectorStore:
    def __init__(self, dim: int, dtype: np.dtype = np.float32):
        self.dim = dim
        self.dtype = dtype
        self.vectors = np.empty((0, dim), dtype=dtype)
        self.ids = np.empty(0, dtype=np.int64)
        self._next_id = 0

    @property
    def ntotal(self) -> int:
        return len(self.ids)

    def add(self, vectors: np.ndarray) -> np.ndarray: ...
    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray): ...
    def remove(self, ids: np.ndarray): ...
    def get(self, ids: np.ndarray) -> np.ndarray: ...
    def clear(self): ...
```

The IVF family stores inverted lists as Python lists keyed by cell. IVF stores raw vectors,
IVF-PQ stores compressed codes:

```python
# IVFIndex
self.centroids = None                              # (nlist, dim) after train
self.inverted_lists = [[] for _ in range(nlist)]   # cell -> list of (id, vector)

# IVFPQIndex
self.coarse_centroids = None                       # (nlist, dim)
self.pq_centroids = None                           # (M, ksub, dsub)
self.inverted_lists = [[] for _ in range(nlist)]   # cell -> list of (id, codes uint8[M])
self.ksub = 2 ** nbits                             # centroids per subquantizer
self.dsub = dim // M                               # subvector width
```

HNSW keeps its graph as a list of adjacency lists, one per layer, plus a flat vector store
and entry-point bookkeeping:

```python
# HNSWIndex
self.M = M                       # max connections per node (upper layers)
self.M0 = 2 * M                  # max connections at layer 0
self.ef_construction = 200       # build-time beam width
self.ef_search = 50              # query-time beam width
self.ml = 1 / np.log(M)          # level-assignment scale
self.vectors = []                # node_id -> vector
self.graphs = []                 # graphs[layer][node_id] -> [neighbor ids]
self.entry_point = -1            # top-level entry node
self.max_layer = -1              # current highest layer
```

The product quantizer's codebooks are a single dense tensor:

```python
# ProductQuantizer
self.M = M                       # number of subquantizers
self.ksub = 2 ** nbits           # centroids per subquantizer
self.dsub = dim // M             # subvector dimension
self.centroids = None            # (M, ksub, dsub) after train
```

## API Design

The public API is the set of symbols re-exported from `vecindex/__init__.py`. Indexes share
the `Index` interface; constructors differ by what each algorithm needs.

```python
# Core
MetricType                          # L2 | IP | COSINE
SearchResult(ids, distances)        # dataclass, .num_results
VectorStore(dim, dtype=np.float32)
l2_distance(a, b); inner_product(a, b); cosine_similarity(a, b)
normalize_vectors(v); kmeans(vectors, k, max_iter=100, tol=1e-4)

# Indexes (all expose train / add / search / ntotal / __len__)
FlatIndex(dim, metric=MetricType.L2)
IVFIndex(dim, nlist, metric=MetricType.L2, nprobe=1)
HNSWIndex(dim, M=16, ef_construction=200, ef_search=50, metric=MetricType.L2)
IVFPQIndex(dim, nlist, M, nbits=8, metric=MetricType.L2, nprobe=1)

index.train(vectors)                # no-op for Flat and HNSW
index.add(vectors)                  # IVF / IVF-PQ require train() first
result = index.search(query, k)     # -> SearchResult

# Quantizers
ProductQuantizer(dim, M, nbits=8)
  .train(vectors, max_iter=100); .encode(v) -> codes; .decode(codes) -> v
  .compute_distance_table(query); .asymmetric_distance(query, codes)
  .symmetric_distance(codes1, codes2); .code_size; .compression_ratio
OPQ(dim, M, nbits=8).train(vectors, n_iter=10)        # subclass with rotation
ScalarQuantizer(dim, nbits=8); BinaryQuantizer(dim)
compute_recall(ground_truth, results, k) -> float

# Search utilities
BatchSearcher(index).search(queries, k, batch_size=1000)
BatchSearcher(index).search_with_filter(query, k, filter_ids)
HybridSearcher().add_index(name, index, weight); .search({name: query}, k)
RerankSearcher(first_stage, vectors, metric).search(query, k, rerank_k=100)
build_index(vectors, index_type="flat"|"ivf"|"hnsw"|"ivfpq", metric=..., **kwargs)
IndexFactory.create(index_string, dim)                # "Flat","IVF100,PQ8","HNSW32"
benchmark_index(index, queries, ground_truth, k=10) -> {recall@k, qps, latency_ms}

# SIMD
SIMD_AVAILABLE                                        # True if Numba present
simd_l2_distance(query, database); simd_inner_product(...); simd_cosine_similarity(...)
simd_topk(distances, k); simd_l2_batch(queries, database, k); simd_ip_batch(...)
batch_search(queries, database, k, metric="l2", batch_size=1024)
SIMDVectorOps(database, metric="l2").search(queries, k); .range_search(queries, radius)
```

Two conventions matter across the surface. First, every metric is reduced to "lower is
better" by `compute_distance`, so the same `topk` works for L2, IP, and cosine; callers read
`SearchResult.distances` ascending. Second, IVF and IVF-PQ raise `RuntimeError` if `add` is
called before `train`, while Flat and HNSW set `is_trained = True` at construction so their
`train` is a documented no-op.

## Performance

The implementation prioritizes legibility over raw throughput; there are no published
benchmark numbers in the source, so this section describes the algorithmic complexity and
design choices rather than measured figures.

**Search complexity by index.** FlatIndex is O(n*d) per query — it touches every vector,
which is exact but does not scale. IVFIndex restricts work to the vectors in `nprobe` of
`nlist` cells, roughly O((nprobe/nlist) * n * d) plus the O(nlist * d) centroid scan, so
larger `nlist` shrinks each cell and `nprobe` trades recall for speed. HNSWIndex targets
O(log n) comparisons by descending sparse upper layers before a bounded level-0 search;
actual cost scales with `M` (degree) and `ef_search` (beam width). IVFPQIndex matches IVF's
candidate-gathering cost but replaces each full distance with `M` table lookups, so per-cell
scoring is cheap and dominated by table construction.

**Memory.** FlatIndex and IVFIndex store full float32 vectors (`dim * 4` bytes each). IVFPQ
stores `M` bytes per vector after PQ encoding — at `dim=64, M=8, nbits=8` that is 8 bytes
versus 256, a 32x reduction, which `ProductQuantizer.compression_ratio` reports and
`test_compression_effect` verifies. HNSW adds graph overhead of roughly `M` integer edges
per node per layer on top of the stored vectors.

**Recall knobs.** Each approximate index exposes a single dominant accuracy lever, and the
tests assert each one is monotonic: IVF's `nprobe` (`test_search_nprobe_affects_results`),
HNSW's `ef_search` (`test_ef_search_affects_recall`), and for IVF-PQ the combination of
`nprobe` and `nbits` (more bits per subquantizer means finer codebooks and lower
quantization error). The tests encode realistic recall floors: 100% exact-match IDs for
Flat, >= 70% for HNSW under default parameters, and >= 30% for the lossy IVF-PQ path.

**SIMD path.** When Numba is installed, `simd/ops.py` JIT-compiles the inner distance and
top-k loops with `fastmath` and `parallel=True`, and `simd_l2_batch` / `simd_ip_batch` fuse
distance computation and selection into a single pass to avoid materializing the full
distance matrix. Inputs are forced C-contiguous and aligned before kernel entry. Without
Numba the same functions fall back to vectorized NumPy, so correctness is identical and only
speed differs.

**Complexity summary.** The table below states asymptotic per-query cost; `c` is the number
of candidate vectors gathered (for IVF roughly `(nprobe/nlist) * n`):

| Index | Per-query work | Storage per vector | Exact? |
|-------|----------------|--------------------|--------|
| Flat | O(n*d) | dim * 4 bytes | yes |
| IVF | O(nlist*d + c*d) | dim * 4 bytes | no |
| HNSW | ~O(log n) comparisons, each O(d) | dim*4 + ~M ints/layer | no |
| IVF-PQ | O(nlist*d + M*ksub*dsub + c*M) | M bytes | no |

The IVF-PQ scoring term `c*M` is what makes compressed search fast: once the `(M, ksub)`
table is built (the `M*ksub*dsub` term), each candidate costs only `M` lookups regardless of
the original dimension. This is why PQ remains attractive even when the database no longer
fits in memory as full vectors — though this implementation keeps everything resident.

**Why the recall floors are what they are.** The test thresholds are not arbitrary. Flat is
asserted to be bit-exact against `np.argsort`, because it is exact by construction. HNSW's
70% floor reflects modest default parameters (`M=8..16`, small `ef`) on small synthetic
datasets where graph connectivity is the limiting factor; raising `M` and `ef_search` pushes
real-world recall well above this. IVF-PQ's 30% floor reflects two stacked approximations —
cell pruning and lossy quantization — compounding on data with no cluster structure (i.i.d.
Gaussian), which is close to the worst case for both methods. The tests therefore assert that
each index clears a floor that its algorithm should always meet, and that each accuracy knob
moves recall in the right direction, rather than pinning exact recall numbers that would be
brittle across NumPy versions.

## Testing Strategy

The suite contains 163 tests across four files and runs entirely on synthetic NumPy data
with a fixed seed (`conftest.py` sets `np.random.seed(42)` autouse), so results are
deterministic and need no external services. Shared fixtures provide datasets at several
scales (100 / 1000 / 5000 vectors at dims 16 / 64 / 128), clustered data for IVF, normalized
data for cosine, and pre-built indexes and quantizers. `conftest.py` also supplies
brute-force ground-truth helpers (`compute_ground_truth_l2`, `_ip`, `_cosine`).

**Index tests (`test_index.py`, 41 tests).** Per-index classes cover construction and
defaults, the train-before-add contract (IVF and IVF-PQ raise `RuntimeError`; the
`assert dim % M == 0` is checked), add/search basics, and correctness. FlatIndex is verified
to be exact via `np.testing.assert_array_equal` against ground truth and to return itself
for a self-query. HNSW and IVF-PQ are verified against recall floors, and `nprobe` /
`ef_search` are asserted to be monotonic in recall. `TestIndexComparison` checks all four
indexes share the common interface and that approximate indexes return results close to
exact.

**Quantization tests (`test_quantization.py`, 52 tests).** Encode/decode round-trips and
reconstruction-error bounds for each quantizer, ADC vs SDC distance behavior for PQ, the
rotation step for OPQ, range learning for scalar quantization, bit packing and Hamming
distance for binary quantization, and `compute_recall` correctness.

**Search tests (`test_search.py`, 42 tests).** Distance-metric and `compute_distance`
correctness, single- and batch-`topk`, the search utilities (`BatchSearcher` filtering,
`HybridSearcher` fusion, `RerankSearcher` two-stage accuracy), `build_index`,
`IndexFactory` string parsing, `benchmark_index` outputs, and the `SearchResult` dataclass.

**SIMD tests (`test_simd.py`, 28 tests).** Parity between the SIMD and NumPy paths for every
distance, top-k correctness, fused batch search, `SIMDVectorOps`, the `SIMD_AVAILABLE` flag,
and edge cases (single vector, k larger than the database, empty inputs).

**Invariants and edge cases worth calling out.** The suite is built around a few invariants
that hold regardless of index type: a self-query returns the queried vector first with a
near-zero distance; `search` results are monotonically non-decreasing in distance; requesting
`k` larger than `ntotal` returns all available results rather than raising; and an empty
HNSW index returns empty arrays instead of failing. The recall-monotonicity tests are
written as comparisons (`recall_high >= recall_low`) rather than absolute equalities so they
stay robust to the randomized graph construction and k-means initialization. Because the seed
is fixed autouse, every run sees the same data and the same centroid initialization, which is
what makes even the approximate-recall assertions deterministic.

Run the full suite with:

```bash
pytest tests/ -v
```

## References

- Malkov, Y. A., & Yashunin, D. A. "Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs."
- Jegou, H., Douze, M., & Schmid, C. "Product Quantization for Nearest Neighbor Search."
- Ge, T., He, K., Ke, Q., & Sun, J. "Optimized Product Quantization." (OPQ)
- Johnson, J., Douze, M., & Jegou, H. "Billion-scale similarity search with GPUs." (FAISS)
- FAISS — https://github.com/facebookresearch/faiss

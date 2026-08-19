# TurboQuant

**Vector quantization for ANN search, built from scratch in NumPy — 4x to 64x memory compression with honest, measured recall.**

TurboQuant implements the compression stack that powers FAISS, Pinecone, Weaviate, and every serious vector database — Scalar Quantization (SQ), Product Quantization (PQ), **OPQ**, IVFPQ — plus its own ideas: a **variance-adaptive bit allocator** borrowed from transform coding, a **score-aware anisotropic quantizer** for inner-product search, and a **budget-driven two-stage cascade index** that turns "how many bytes per vector can you afford?" into a single knob.

It also does the unglamorous part that from-scratch ANN implementations usually skip: **deletes, updates, compaction, metadata-filtered search, and pickle-free persistence**, with an edge-case suite for the inputs a real ingest pipeline actually produces.

No FAISS, no C++, no hidden magic: every algorithm is a few hundred lines of documented NumPy, with tests that assert the recall claims and a benchmark that regenerates every number in this README.

```
pip install numpy
python benchmarks/bench.py         # reproduces the tables below
python -m pytest tests/            # 144 tests, ~45 s
```

## The problem

Embeddings are memory hogs:

| Scale | float32 RAM |
|---|---|
| 1M vectors × 1536 dims (OpenAI `text-embedding-3-small`) | ~6 GB |
| 10M vectors × 1536 dims | ~60 GB |
| 100M vectors × 768 dims | ~300 GB |

RAM is the cost driver of vector search: the whole index must be resident for low-latency queries. Quantization replaces each float32 vector with a tiny learned code — 64x smaller at the aggressive end — while keeping distances *approximately* right, which is all nearest-neighbor search needs.

## Results

50,000 synthetic clustered embeddings, 128 dims, recall@10 against exact float32 search, single CPU core (`python benchmarks/bench.py`):

### L2 nearest-neighbor search

| Method | Compression | Memory (MB) | Recall@10 | ms/query |
|---|---|---|---|---|
| Flat float32 (exact) | 1.0x | 25.6 | 1.000 | 1.21 |
| Flat SQ8 | 4.0x | 6.4 | 0.960 | 1.38 |
| Flat SQ4 | 8.0x | 3.2 | 0.568 | 2.11 |
| **Flat AdaptiveBits (avg 4b)** | 8.0x | 3.2 | **0.732** | 2.05 |
| Flat PQ M=8 | 64.0x | 0.4 | 0.043 | 3.62 |
| **Flat OPQ M=8** | 64.0x | 0.4 | **0.122** | 2.55 |
| Flat PQ M=16 | 32.0x | 0.8 | 0.073 | 3.80 |
| **Flat OPQ M=16** | 32.0x | 0.8 | **0.217** | 4.55 |
| IVFPQ M=8, nprobe=16 | 64.0x | 0.4 | 0.200 | 2.48 |
| IVFPQ M=16, nprobe=16 | 32.0x | 0.8 | 0.351 | 3.87 |
| IVFPQ M=32, nprobe=16 | 16.0x | 1.6 | 0.623 | 7.09 |
| TurboIndex 16B (PQ8+PQ8) | 32.0x | 0.8 | 0.339 | 3.24 |
| TurboIndex 24B (PQ8+PQ16) | 21.3x | 1.2 | 0.460 | 2.86 |
| TurboIndex 32B (PQ16+PQ16) | 16.0x | 1.6 | 0.579 | **4.42** |
| **TurboIndex 64B (PQ16+adaptive)** | 8.0x | 3.2 | **0.833** | 4.90 |

What the numbers show:

- **SQ8 is nearly free**: 4x compression for 4 points of recall.
- **AdaptiveBits beats uniform SQ4 by +16 recall points at identical 8x cost** — spending bits where the variance is pays.
- **OPQ is free recall**: 2.8–3.0x the recall of plain PQ at *byte-identical* storage (0.043→0.122 at M=8, 0.073→0.217 at M=16). The rotation is d×d shared floats — 64 KB total, amortized over the whole index — and costs one matvec per query.
- **Residual encoding is why IVFPQ exists**: the same M=8 codes score 0.200 inside IVF vs 0.043 flat, a 4.7x recall lift from encoding `x − centroid` instead of `x`.
- **The TurboIndex cascade dominates the high-quality end**: at 8x compression it reaches 0.833 recall — +10 points over the best single-stage 8x option — and at 16x it comes within ~4 points of IVFPQ M=32 while answering queries **~1.6x faster** (tier 1 scans with 16 table lookups/vector instead of 32).
- **Budgets are continuous**: plain PQ only exists at divisors of `dim` (16B, 32B, 64B...); TurboIndex fills the gaps (24B) with a principled split.

### Maximum inner-product search

| Method | Bytes/vector | Recall@10 | Reconstruction MSE |
|---|---|---|---|
| PQ M=8 | 8 | 0.043 | 2.82 |
| **AnisotropicPQ η=8, M=8** | 8 | **0.352** | 3.07 |
| PQ M=16 | 16 | 0.070 | 2.66 |
| **AnisotropicPQ η=8, M=16** | 16 | **0.449** | 2.89 |

6–8x the MIPS recall at identical storage — while *reconstruction gets worse*. That inversion is the point, not a side effect: the loss deliberately spends accuracy on directions that do not move inner-product scores. (On unit-normalized vectors the gain disappears entirely; see below.)

### Filtered search — "nearest neighbors WHERE ..."

nprobe=4, measured against exact *filtered* ground truth:

| Filter selectivity | Results returned | Recall@10 |
|---|---|---|
| 10%, post-filter only | 10.0 / 10 | 0.587 |
| 10%, + probe escalation | 10.0 / 10 | 0.587 |
| 1%, post-filter only | 9.7 / 10 | 0.800 |
| 1%, + probe escalation | 10.0 / 10 | 0.823 |
| 0.1%, post-filter only | 1.6 / 10 | 0.159 |
| **0.1%, + probe escalation** | **10.0 / 10** | **0.871** |

Post-filtering an ANN scan fails silently: at 0.1% selectivity a `k=10` query returns 1.6 results at 0.159 recall and raises nothing. Escalation — widen the scan until enough candidates *survive the filter*, scanning only newly added cells — restores it to 0.871. At 10% selectivity it never fires and costs nothing.

> These are *hard-mode* numbers: the synthetic dataset has 100 tight clusters, so the true top-10 are fine-grained within-cluster neighbors. On typical real embedding distributions absolute recalls are higher across the board; the *relative* ordering is what transfers.

## Quickstart

```python
import numpy as np
from turboquant import TurboIndex, FlatIndex, recall_at_k, train_base_query_split

# synthetic data mimicking real embedding structure (clusters + variance decay)
train, base, queries = train_base_query_split(
    n_train=20_000, n_base=100_000, n_query=100, dim=128
)

# one knob: bytes per vector. 32 bytes = 16x compression at dim=128.
index = TurboIndex(dim=128, budget_bytes=32, n_lists=256)
index.train(train)          # learn coarse centroids, PQ codebooks, refiner
index.add(base)             # store ~32 bytes/vector, floats are discarded
ids, dists = index.search(queries, k=10, n_probe=16, rerank_factor=8)

print(f"{index.bytes_per_vector:.0f} bytes/vector "
      f"({(4 * 128) / index.bytes_per_vector:.0f}x compression)")
```

Every layer is also usable on its own:

```python
from turboquant import (
    ScalarQuantizer,       # SQ8 / SQ4: per-dim uniform quantization
    AdaptiveBitQuantizer,  # variance-driven bit allocation (novel)
    ProductQuantizer,      # PQ + ADC lookup-table search
    OPQProductQuantizer,   # PQ behind a learned rotation
    AnisotropicPQ,         # score-aware PQ for inner-product search (novel)
    QuantizedFlatIndex,    # brute-force over any quantizer's codes
    IVFPQIndex,            # inverted lists + residual PQ
)

pq = ProductQuantizer(dim=128, n_subspaces=16).train(train)
codes = pq.encode(base)               # (n, 16) uint8 -- 32x smaller
lut = pq.compute_lut(queries)         # (nq, 16, 256) distance tables
dists = pq.adc_distances(lut, codes)  # search without decompressing
```

### A live index, not a frozen one

```python
index.remove_ids([17, 42])                   # tombstones; ids stay stable
index.update(np.array([3]), new_vector)      # re-encodes and re-files the cell
mapping = index.compact()                    # reclaims space, returns old->new ids

# "nearest neighbors WHERE tenant_id == 7"
mask = metadata["tenant_id"] == 7
ids, dists = index.search(queries, k=10, n_probe=16,
                          filter_mask=mask, max_probe=256)  # escalates if starved

from turboquant import save, load
load(save(index, "index.npz")).search(queries, k=10)        # no pickle involved
```

## What's inside

```
turboquant/
├── kmeans.py                  # k-means++ / Lloyd -- the training engine
├── metrics.py                 # L2 + inner-product kernels, top-k, recall@k
├── validation.py              # boundary guards: NaN/Inf, shapes, ids, k
├── io.py                      # pickle-free save / load
├── datasets.py                # synthetic clustered embeddings
├── quantizers/
│   ├── base.py                # train / encode / decode / bytes_per_vector
│   ├── scalar.py              # SQ with real bit-packing (SQ4 is 0.5 B/dim)
│   ├── adaptive.py            # ★ variance-driven bit allocation
│   ├── product.py             # PQ + asymmetric distance computation
│   ├── opq.py                 # learned rotation: eigen-balancing + Procrustes
│   └── anisotropic.py         # ★ score-aware loss for MIPS
├── index/
│   ├── flat.py                # exact baseline + quantized brute force
│   ├── ivfpq.py               # inverted lists + residual PQ, deletes, filters
│   └── turbo.py               # ★ budget-driven two-stage cascade
├── tests/                     # 144 tests: recall, edge cases, mutations, I/O
├── benchmarks/bench.py        # regenerates the results tables
└── docs/
    ├── CONCEPTS.md            # step-by-step theory, from SQ to the cascade
    └── CODE_WALKTHROUGH.md    # line-by-line reasoning for every module
```

## The novel pieces (★)

### 1. AdaptiveBitQuantizer — bits go where the variance is

Classical SQ gives every dimension the same bits, but embedding dimensions are wildly unequal (PCA-like variance decay is near-universal in learned embeddings). TurboQuant treats bit assignment as a **rate-allocation problem** from transform coding: with uniform quantization error `MSE(b) ∝ span² / 4^b`, greedily granting one bit at a time to the currently-worst dimension is the optimal discrete water-filling solution. Dimensions can earn 0 bits (dropped entirely, reconstructed as their mean) up to 8 bits, and codes are genuinely bit-packed — a 4-bit average really is 0.5 bytes/dim on disk and in RAM. Result: **+16 recall points over uniform SQ4 at identical storage**.

### 2. TurboIndex — a memory budget, not a config puzzle

You say `budget_bytes=32`; the index derives everything else:

```
tier 1 (≈⅓ budget): IVF + small PQ  -> cheap scan, candidate shortlist
tier 2 (≈⅔ budget): encodes  x − centroid − PQ₁(x)   (tier 1's own error)
re-rank: x̂ = centroid + PQ₁ + refine   -> every stored byte scores
```

Two design decisions matter, and both were driven by measurement (the failed alternatives are documented in `docs/CONCEPTS.md`):

- **Tier 2 encodes tier 1's error, not the raw vector.** An independent refine code throws tier 1's bytes away at re-rank time; we measured that variant *losing* to plain IVFPQ at equal budget. Encoding the second-stage residual makes the tiers additive — reconstruction uses all 32 bytes — and makes widening the shortlist strictly safe.
- **The tier-2 codec switches at a measured crossover (~3 bits/dim).** Below it, a second PQ (vector quantization is more bit-efficient at low rates); above it, the adaptive scalar quantizer (scalar codes approach lossless while 256-entry codebooks saturate). At an 80-byte budget: adaptive 0.925 vs second-PQ 0.914.

### 3. AnisotropicPQ — quantize for the score, not the reconstruction

Every other quantizer here minimizes `‖x − x̂‖²`, treating all error directions as equally bad. For **inner-product search** that is the wrong objective. Split the residual relative to the datapoint, `r = r∥ + r⊥`. Condition on the case that matters — `x` is a top result for `q`, so `q` points roughly along `x`. Then `⟨q, r⊥⟩` largely cancels, while `⟨q, r∥⟩` adds coherently to the score every time. Parallel error corrupts the ranking; perpendicular error mostly does not. So weight them:

```
loss = η‖r∥‖² + ‖r⊥‖² = ‖r‖² + (η−1)⟨r, x̄⟩²        (η = 1 is exactly PQ)
```

The penalty couples the subspaces, so PQ's independent per-subspace k-means no longer applies; training is **block coordinate descent** — a closed-form `(n, ks)` argmin to assign, and a `dsub × dsub` linear solve per cluster to update (which collapses back to the k-means mean at `η = 1`).

The result is a deliberate trade, and the trade *is* the evidence:

| η | MIPS recall@10 | reconstruction MSE |
|---|---|---|
| 1.0 (= PQ) | 0.126 | 2.67 |
| 8.0 | 0.566 | 2.90 |
| 16.0 | **0.605** | 3.00 |

**MIPS recall nearly 5x's while reconstruction error gets steadily worse** — the quantizer is spending accuracy where it does not move scores. A method that improved both columns would be evidence of a better-tuned PQ, not of the anisotropic argument.

**And where it stops working**: normalize the vectors and the gain inverts (0.079 → 0.029 at η=16). That is the premise expiring, not a bug — on the unit sphere every norm is 1, the score becomes pure angle, and MIPS collapses into cosine, which is just L2. Use it for MIPS over unnormalized vectors; for cosine, `normalize()` and use any L2 index here. Both results are pinned by tests.

## Documentation

- **[docs/CONCEPTS.md](docs/CONCEPTS.md)** — the theory, step by step: why quantization works, SQ → bit allocation → k-means → PQ → ADC → IVF → residuals → the cascade, including the negative results that shaped the design.
- **[docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md)** — line-by-line reasoning for every non-trivial line in the library: why `argpartition` instead of `argsort`, why the LUT gather is shaped the way it is, why percentile ranges instead of min/max, why `np.packbits(bitorder="little")`, and so on.

## Honest limitations

- Pure NumPy: per-query Python overhead makes absolute latencies ~10-100x slower than FAISS's SIMD kernels; the *relative* comparisons and all memory numbers are real.
- `memory_bytes` counts code storage (as FAISS's `code_size` does); IVF id lists add ~8 bytes/vector of bookkeeping in any implementation. Tombstoned vectors are counted until you `compact()` — deleting should not make a compression ratio look better.
- **IVF partitioning is L2-only.** Cosine is exact via `normalize()` (identical ordering), and MIPS is supported for *flat* search over `AnisotropicPQ` codes. IVF-based MIPS is deliberately absent: coarse quantization for inner product needs spherical clustering, and shipping the L2 partitioner with an IP scorer would be subtly wrong rather than merely approximate.
- Single-threaded, single-machine, in-memory. No sharding, no mmap, no concurrent writer safety — `save`/`load` is a snapshot, not a WAL.
- Searches loop over queries in Python; batching the ADC scan across queries would be the first real optimization.

## License

MIT — see [LICENSE](LICENSE).

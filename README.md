# TurboQuant

**Vector quantization for ANN search, built from scratch in NumPy — 4x to 64x memory compression with honest, measured recall.**

TurboQuant implements the compression stack that powers FAISS, Pinecone, Weaviate, and every serious vector database — Scalar Quantization (SQ), Product Quantization (PQ), and IVFPQ — plus two of its own ideas: a **variance-adaptive bit allocator** borrowed from transform coding, and a **budget-driven two-stage cascade index** that turns "how many bytes per vector can you afford?" into a single knob.

No FAISS, no C++, no hidden magic: every algorithm is a few hundred lines of documented NumPy, with tests that assert the recall claims and a benchmark that regenerates every number in this README.

```
pip install numpy
python benchmarks/bench.py         # reproduces the table below
python -m pytest tests/            # 31 tests, ~10 s
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

| Method | Compression | Memory (MB) | Recall@10 | ms/query |
|---|---|---|---|---|
| Flat float32 (exact) | 1.0x | 25.6 | 1.000 | 0.90 |
| Flat SQ8 | 4.0x | 6.4 | 0.960 | 0.93 |
| Flat SQ4 | 8.0x | 3.2 | 0.568 | 1.93 |
| **Flat AdaptiveBits (avg 4b)** | 8.0x | 3.2 | **0.732** | 2.58 |
| Flat PQ M=8 | 64.0x | 0.4 | 0.042 | 3.23 |
| Flat PQ M=16 | 32.0x | 0.8 | 0.073 | 4.40 |
| IVFPQ M=8, nprobe=16 | 64.0x | 0.4 | 0.199 | 2.46 |
| IVFPQ M=16, nprobe=16 | 32.0x | 0.8 | 0.352 | 3.88 |
| IVFPQ M=32, nprobe=16 | 16.0x | 1.6 | 0.620 | 7.57 |
| TurboIndex 16B (PQ8+PQ8) | 32.0x | 0.8 | 0.339 | 2.43 |
| TurboIndex 24B (PQ8+PQ16) | 21.3x | 1.2 | 0.463 | 2.53 |
| TurboIndex 32B (PQ16+PQ16) | 16.0x | 1.6 | 0.586 | **4.12** |
| **TurboIndex 64B (PQ16+adaptive)** | 8.0x | 3.2 | **0.830** | 4.39 |

What the numbers show:

- **SQ8 is nearly free**: 4x compression for 4 points of recall.
- **AdaptiveBits beats uniform SQ4 by +16 recall points at identical 8x cost** — spending bits where the variance is pays.
- **Residual encoding is why IVFPQ exists**: the same M=8 codes score 0.199 inside IVF vs 0.042 flat, a 4.7x recall lift from encoding `x − centroid` instead of `x`.
- **The TurboIndex cascade dominates the high-quality end**: at 8x compression it reaches 0.830 recall — +10 points over the best single-stage 8x option — and at 16x it comes within ~3 points of IVFPQ M=32 while answering queries **~2x faster** (tier 1 scans with 16 table lookups/vector instead of 32).
- **Budgets are continuous**: plain PQ only exists at divisors of `dim` (16B, 32B, 64B...); TurboIndex fills the gaps (24B) with a principled split.

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
    QuantizedFlatIndex,    # brute-force over any quantizer's codes
    IVFPQIndex,            # inverted lists + residual PQ
)

pq = ProductQuantizer(dim=128, n_subspaces=16).train(train)
codes = pq.encode(base)               # (n, 16) uint8 -- 32x smaller
lut = pq.compute_lut(queries)         # (nq, 16, 256) distance tables
dists = pq.adc_distances(lut, codes)  # search without decompressing
```

## What's inside

```
turboquant/
├── kmeans.py                  # k-means++ / Lloyd -- the training engine
├── metrics.py                 # squared-L2 kernels, top-k, recall@k
├── datasets.py                # synthetic clustered embeddings
├── quantizers/
│   ├── base.py                # train / encode / decode / bytes_per_vector
│   ├── scalar.py              # SQ with real bit-packing (SQ4 is 0.5 B/dim)
│   ├── adaptive.py            # ★ variance-driven bit allocation
│   └── product.py             # PQ + asymmetric distance computation
├── index/
│   ├── flat.py                # exact baseline + quantized brute force
│   ├── ivfpq.py               # inverted lists + residual PQ
│   └── turbo.py               # ★ budget-driven two-stage cascade
├── tests/                     # 31 tests incl. recall assertions
├── benchmarks/bench.py        # regenerates the results table
└── docs/
    ├── CONCEPTS.md            # step-by-step theory, from SQ to the cascade
    └── CODE_WALKTHROUGH.md    # line-by-line reasoning for every module
```

## The two novel pieces (★)

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

## Documentation

- **[docs/CONCEPTS.md](docs/CONCEPTS.md)** — the theory, step by step: why quantization works, SQ → bit allocation → k-means → PQ → ADC → IVF → residuals → the cascade, including the negative results that shaped the design.
- **[docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md)** — line-by-line reasoning for every non-trivial line in the library: why `argpartition` instead of `argsort`, why the LUT gather is shaped the way it is, why percentile ranges instead of min/max, why `np.packbits(bitorder="little")`, and so on.

## Honest limitations

- Pure NumPy: per-query Python overhead makes absolute latencies ~10-100x slower than FAISS's SIMD kernels; the *relative* comparisons and all memory numbers are real.
- `memory_bytes` counts code storage (as FAISS's `code_size` does); IVF id lists add ~8 bytes/vector of bookkeeping in any implementation.
- L2 distance only (inner-product / cosine reduce to L2 on normalized vectors).
- No deletes / updates; `add` is append-only.

## License

MIT — see [LICENSE](LICENSE).

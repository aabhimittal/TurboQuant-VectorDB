# TurboQuant Concepts — the theory, step by step

This document builds the whole library conceptually, in the order you'd
invent it. Each section ends with where the idea lives in the code.

---

## 0. Framing: what problem is quantization actually solving?

Nearest-neighbor search over `n` vectors of dimension `d` needs two resources:

- **Memory**: `4·d` bytes per vector in float32. At 10M × 1536-dim that is
  ~60 GB, and it must sit in RAM for millisecond latency.
- **Compute**: exact search compares the query to all `n` vectors: `O(n·d)`.

The key insight making compression possible: **search doesn't need the
vectors — it needs the *distances*, and only their *order*, and only
approximately.** If a lossy code preserves distance ordering well enough
that the true top-10 land in our reported top-10, the reconstruction error
is irrelevant. This is a much weaker requirement than reconstructing data,
which is why 64x compression can coexist with useful recall.

Two error sources will appear throughout, and it pays to keep them separate:

1. **Quantization error** — distances computed from codes are slightly wrong,
   so near-ties get mis-ordered.
2. **Partitioning error** — accelerating structures (IVF) skip parts of the
   space, so some true neighbors are never even scored.

`benchmarks/bench.py` isolates them: "Flat PQ" rows have only error 1;
"IVFPQ" rows add error 2 (and, surprisingly, *reduce* total error — see §6).

---

## 1. Scalar Quantization: the 80/20 of compression

Take each dimension independently. Learn its range `[min_d, max_d]` from a
training sample, chop the range into `2^b` equal cells, store the cell
index (`b` bits) instead of the float (32 bits).

```
value  x_d  ∈ [min_d, max_d]
code        = round((x_d − min_d) / step),   step = (max_d − min_d) / (2^b − 1)
reconstruct = min_d + code · step
```

- `b=8` (SQ8): 4x compression. The quantization step is ~0.4% of the range;
  distance ordering is almost untouched (recall 0.96 in our benchmark).
  This is why SQ8 is the "default on" compression in production systems.
- `b=4` (SQ4): 8x compression, but now only 16 levels per dimension and
  recall drops hard (0.568).

Two implementation details that matter more than they look:

- **Percentile ranges, not min/max.** One outlier would stretch the range
  and waste levels on empty space. We clip at the 0.1/99.9 percentiles;
  outliers saturate at the boundary code (`turboquant/quantizers/scalar.py`).
- **Real bit-packing.** A 4-bit code stored in a uint8 is a 4x compressor
  wearing an 8x costume. We pack the bitstream (`np.packbits`) so claimed
  bytes are actual bytes.

Code: `turboquant/quantizers/scalar.py`

---

## 2. Adaptive bit allocation: not all dimensions deserve the same bits

Look at per-dimension variance in any learned embedding space: it decays,
often by 1–2 orders of magnitude across dimensions (a PCA-like spectrum).
Uniform SQ4 spends 4 bits on a dimension whose entire range is noise and
4 bits on the dimension that decides who your neighbors are.

Treat it as a **rate-allocation problem** (the same one JPEG solves when
deciding how many bits each DCT coefficient gets). Uniform quantization of
a range of width `span` with `b` bits has expected squared error

```
MSE(b) ≈ span² / (12 · 4^b)          (each extra bit divides error by 4)
```

Minimize total error `Σ_d MSE_d(b_d)` subject to `Σ_d b_d = B`. Because each
MSE is convex and decreasing in `b_d`, the greedy algorithm — *give the next
bit to the dimension with the largest current error* — is exactly optimal
(discrete water-filling). Dimensions can get 0 bits (dropped; reconstructed
as their mean) or up to 8.

Result at identical 8x storage: **0.732 vs 0.568 recall@10** against
uniform SQ4. The negative space matters too: on *isotropic* data (all
variances equal) this collapses to uniform SQ — the win exists exactly
because real embeddings are anisotropic.

Code: `turboquant/quantizers/adaptive.py` (`_allocate_bits` is the
water-filling loop; `_build_layout`/`encode` implement mixed-width packing).

---

## 3. k-means: the engine everything else is built on

Both PQ codebooks and IVF partitions are k-means centroids. Ours is
textbook Lloyd's with three production details:

- **k-means++ seeding** — new seeds sampled proportional to squared
  distance from existing seeds. Prevents the arbitrarily-bad initializations
  uniform seeding permits (O(log k) approximation guarantee).
- **Empty-cluster repair** — a centroid that loses all its points is
  respawned on a random point, otherwise it is dead weight forever (with
  256-entry PQ codebooks, dead centroids directly waste code space).
- **Training-sample capping** — centroid estimates converge long before
  you've used millions of points; we cap at 100k (FAISS similarly
  subsamples).

Code: `turboquant/kmeans.py`

---

## 4. Product Quantization: the 30–64x workhorse

Scalar quantization can't go much below ~2 bits/dim — each dimension is
quantized *independently*, blind to correlations. Vector quantization
(one k-means over whole vectors) would be ideal but needs `2^128`-ish
centroids to be precise at d=128. PQ is the compromise that made
billion-scale search possible (Jégou, Douze & Schmid, 2011):

**Split each vector into M subvectors; run k-means with 256 centroids in
each subspace; store M one-byte centroid ids.**

```
d=128, M=16:  [--8 dims--][--8 dims--] ... [--8 dims--]     16 subspaces
codes:            id₀          id₁      ...     id₁₅        16 bytes total
```

The magic is in the *implicit* codebook: the full-vector codebook is the
cartesian product of the subspace codebooks — `256^16 ≈ 3·10^38` distinct
reconstructions from only `16 × 256 × 8` stored floats. Exponential
representational power for linear storage.

### ADC: searching without decompressing

For query `q`, precompute per subspace the distance to all 256 centroids:

```
LUT[m][c] = ‖q_m − codebook[m][c]‖²        (M × 256 floats, one-time)
dist(q, x) ≈ Σ_m LUT[m][ code_x[m] ]       (M lookups + adds per vector)
```

The database scan does **zero floating-point vector math** — only table
lookups. This is also why scan cost scales with M: 8-byte codes are
literally 2x fewer lookups than 16-byte codes (visible in our ms/query
column). "Asymmetric" = query stays exact, database is quantized; only one
side contributes quantization error.

Test `test_pq_adc_matches_decoded_distances` verifies ADC equals the
explicit `‖q − decode(code)‖²` — same quantity, computed two ways.

Code: `turboquant/quantizers/product.py`

---

## 5. Why flat PQ disappoints, and what it teaches

Our benchmark shows flat PQ M=8 at 0.042 recall. Two reasons:

1. At 64x compression, cells of the implicit codebook are simply large;
   many vectors share a code, and within-cell ordering is unrecoverable.
2. PQ quantizes the *global* distribution. Raw embedding space is wide
   (cluster structure spreads mass), so subspace centroids are spread
   thin over a huge region.

Reason 2 is fixable — you don't have to quantize the global distribution.
That's the door IVF opens.

---

## 6. IVF + residuals: partition first, then quantize what's left

**IVF (inverted file):** k-means the space into `nlist` coarse cells; store
each vector's id in its nearest cell's list. At query time, rank cells by
centroid distance and scan only the `nprobe` closest lists — searching
`~ n·nprobe/nlist` vectors instead of `n`.

**Residual encoding — the underrated half of IVFPQ:** within a cell, every
vector is near its centroid, so encode the *offset* `r = x − c` instead of
`x`. Residuals live in a small ball around the origin regardless of which
cell they came from; the PQ codebooks now cover a tight distribution
instead of the whole space, and the same M bytes buy far more precision.
The query is shifted the same way per probed cell (`q − c`) before building
the LUT, preserving the ADC trick.

The benchmark quantifies it: same M=8 codes, recall 0.042 flat vs 0.199 in
IVFPQ — **the partitioning structure *improved* accuracy while also making
search 30% faster.** Rare free lunch, bought by the residual trick.

Code: `turboquant/index/ivfpq.py`

---

## 7. The TurboIndex cascade — and the two designs that failed first

Goal: a single knob — bytes per vector — and the best recall/speed we can
engineer under it. The final design (tier 1 IVFPQ on ~⅓ of the budget for
candidates, tier 2 encoding tier 1's own error on the rest, re-rank on the
sum) is described in the README. What's more instructive is *how it got
there*; both intermediate designs are real measured failures on the
50k × 128-dim benchmark:

**Failure 1: independent refine codes over raw vectors.** Tier 2 as an
AdaptiveBitQuantizer over the raw vectors, re-rank on its reconstruction
alone. Result: recall *decreased* as the shortlist widened (0.432 at
rerank_factor=2 → 0.310 at 8). Diagnosis: tier 2 at ~2 bits/dim over the
*wide raw distribution* was a worse ranker than tier 1's residual PQ — and
re-ranking with a worse ranker un-sorts good candidates. Lesson: **a
cascade stage must score with more information than the stage before it,
otherwise it subtracts value.**

**Failure 2: independent refine codes over cell residuals.** Better (the
refine target shrinks), but at a fixed 24-byte total it still lost to
plain IVFPQ spending 16 bytes: 0.578 vs 0.640. Diagnosis: at re-rank time
the tier-1 bytes were *discarded* — the final score used only ⅔ of the
budget. Lesson: **at a fixed budget, every stored byte must contribute to
the final score.**

**The fix: encode the second-stage residual** `r₂ = x − c − PQ₁(x − c)`,
i.e. tier 1's exact error on that vector. Re-rank reconstruction sums all
stages (`c + PQ₁ + refine`), so tier 2 strictly adds information and both
failures disappear: 0.686 at 24B (beats plain IVFPQ's 0.640 at 16B, and no
plain config exists at 24B), monotone in shortlist width. This is the
residual/additive-quantization principle, driven by a byte budget.

**Choosing the tier-2 codec, empirically.** With the budget split fixed,
which codec encodes `r₂` best per byte?

| refine rate | second PQ | adaptive scalar | winner |
|---|---|---|---|
| 1.0 bit/dim (32B total) | 0.599 | 0.562 | PQ |
| 2.0 bits/dim (48B total) | 0.769 | 0.752 | PQ |
| 4.0 bits/dim (80B total) | 0.914 | **0.925** | adaptive |

Low rates favor vector quantization (a 256-centroid codebook over 8 dims
spends 1 bit/dim yet captures inter-dimension correlation); high rates
favor scalar codes (they approach lossless as bits grow, while a 256-entry
codebook's error floors out). TurboIndex's `refine="auto"` switches at the
measured ~3 bits/dim crossover.

**Why the cascade is also *faster*:** scan cost is M lookups per scanned
vector, and tier 1 runs a small M. TurboIndex-32B (tier-1 M=16) matches
IVFPQ M=32 within ~3 recall points at ~2x the query speed; the re-rank
step touches only `rerank_factor · k` vectors, a rounding error next to
the scan.

Code: `turboquant/index/turbo.py` (`_choose_split` is the budget policy).

---

## 8. Where each method belongs

| Situation | Reach for |
|---|---|
| Memory is tight but not desperate; recall is sacred | SQ8 (4x, ~free) |
| ~8x compression, anisotropic embeddings (i.e., real ones) | AdaptiveBits, or TurboIndex at `4·d/... ≈ d/2` bytes |
| Extreme compression, recall negotiable | IVFPQ, small M |
| A specific RAM budget to hit, best quality under it | TurboIndex with that budget |
| Ground truth / evaluation | FlatIndex |

## References

- Jégou, Douze, Schmid — *Product Quantization for Nearest Neighbor
  Search*, TPAMI 2011 (PQ, ADC, IVFADC).
- Arthur, Vassilvitskii — *k-means++: The Advantages of Careful Seeding*,
  SODA 2007.
- Gersho, Gray — *Vector Quantization and Signal Compression* (bit
  allocation / water-filling).
- Johnson, Douze, Jégou — *Billion-scale similarity search with GPUs*
  (FAISS), 2017.

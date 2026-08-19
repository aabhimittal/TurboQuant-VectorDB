# Code Walkthrough — line-by-line reasoning

The conceptual "why" lives in [CONCEPTS.md](CONCEPTS.md). This document
explains the *code*: for every non-trivial line or block in the library,
what it does, why it is shaped that way, and what breaks if you change it.

---

## `turboquant/metrics.py`

### `pairwise_l2_sq`

```python
q_sq = np.einsum("ij,ij->i", queries, queries)[:, None]
```
`einsum("ij,ij->i", A, A)` is the row-wise squared norm `Σ_j A[i,j]²`,
computed without materializing `A*A` as a temporary (einsum fuses the
multiply and the sum). `[:, None]` reshapes `(nq,)` to `(nq, 1)` so it
broadcasts down columns of the `(nq, n)` result.

```python
cross = queries @ database.T
dists = q_sq - 2.0 * cross + x_sq
```
This is the identity `‖q−x‖² = ‖q‖² − 2q·x + ‖x‖²`. The naive alternative
`((q[:,None,:] - x[None,:,:])**2).sum(-1)` materializes an `(nq, n, d)`
tensor — at 200 queries × 50k vectors × 128 dims that's 5 GB. The identity
form reduces the whole computation to one `(nq,d)×(d,n)` matmul, which
NumPy dispatches to a multithreaded BLAS — this one line is why the whole
library is fast enough to benchmark.

```python
np.maximum(dists, 0.0, out=dists)
```
The three-term expansion can produce `-1e-12` for identical vectors
(catastrophic cancellation between `‖q‖²+‖x‖²` and `2q·x`). Clamping
in-place (`out=dists`, no new allocation) protects any downstream `sqrt`
and keeps distances a valid ordering key.

### `top_k`

```python
part = np.argpartition(dists, k - 1, axis=1)[:, :k]
```
`argpartition` places the k smallest elements (unordered) in the first k
positions in O(n) per row; a full `argsort` would be O(n log n). At
n=50,000 and k=10 that's the difference between touching 50k elements once
and sorting all of them.

```python
part_d = np.take_along_axis(dists, part, axis=1)
order = np.argsort(part_d, axis=1)
```
Only the k survivors are fully sorted — O(k log k), negligible.
`take_along_axis` performs the row-wise fancy indexing `dists[i, part[i]]`
for all rows at once.

### `recall_at_k`

```python
hits += len(set(approx_row.tolist()) & set(exact_row.tolist()))
```
Set intersection because recall@k is *order-insensitive* membership: did
the true neighbors appear anywhere in our top-k? A per-query Python loop is
fine here — this is an evaluation metric, never a hot path.

---

## `turboquant/kmeans.py`

### `_kmeans_pp_init`

```python
closest_sq = pairwise_l2_sq(data, centroids[0:1]).ravel()
```
`centroids[0:1]` (not `centroids[0]`) keeps the array 2-D — the distance
kernel expects matrices. `closest_sq` maintains the running "distance to
nearest chosen seed" for every point; updating it incrementally with
`np.minimum(...)` after each new seed makes seeding O(n·k) total instead of
O(n·k²) (recomputing all pairs each round).

```python
probs = closest_sq / total
centroids[i] = data[rng.choice(n, p=probs)]
```
The k-means++ sampling rule: points far from all current seeds are more
likely to seed the next centroid. The `total <= 0` early-exit above it
handles the degenerate case where every remaining point coincides with a
seed (duplicate-heavy data) — `rng.choice(p=zeros)` would raise.

### `kmeans`

```python
assign = np.argmin(pairwise_l2_sq(data, centroids), axis=1)
```
The assignment step for all points in one shot — the entire Lloyd inner
loop is this matmul-backed line.

```python
counts = np.bincount(assign, minlength=k).astype(np.float32)
sums = np.zeros_like(centroids)
np.add.at(sums, assign, data)
```
The update step without a Python loop over clusters. `np.add.at` is an
*unbuffered* scatter-add: duplicate indices in `assign` accumulate
correctly (`sums[assign] += data` would silently drop duplicates — a
classic NumPy trap). `minlength=k` keeps `counts` length-k even when the
last clusters are empty this iteration.

```python
centroids[~nonempty] = data[rng.choice(n, size=n_dead, replace=False)]
```
Empty-cluster repair: a dead centroid can never re-acquire points (nothing
is assigned to it, so its mean never moves). Respawning it on a random
data point keeps all k codebook entries earning their storage.

---

## `turboquant/quantizers/scalar.py`

### `train`

```python
lo = np.percentile(data, 0.1, axis=0)
hi = np.percentile(data, 99.9, axis=0)
```
Ranges from robust percentiles, per dimension (`axis=0`). Raw min/max
would let a single outlier stretch the range: with 255 levels, doubling
the range doubles the step size — halving effective precision for 99.9% of
values, to represent one point exactly.

```python
span = np.maximum(hi - lo, 1e-12)
self.scales = (span / self.levels).astype(np.float32)
```
The floor guards constant dimensions (span 0) from producing a 0 divisor
in `encode`. Such dimensions quantize everything to code 0 and reconstruct
to `min_d` — exactly right for a constant.

### `encode`

```python
q = np.rint((data - self.mins) / self.scales)
codes = np.clip(q, 0, self.levels).astype(np.uint8)
```
Affine map to "level units", round to nearest level (`rint`, not
truncation — truncation would bias every reconstruction low by half a
step), then clip: values outside the trained percentile range saturate at
the boundary codes rather than wrapping around (which is what a bare
uint8 cast of 260 would do).

### `_pack_codes` / `_unpack_codes`

```python
shifts = np.arange(bits, dtype=np.uint8)
bit_planes = (codes[:, :, None] >> shifts) & 1          # (n, d, bits)
return np.packbits(bit_planes.reshape(n, d * bits), axis=1, bitorder="little")
```
Bit-packing without a Python loop: broadcasting `>> [0, 1, ..., b-1]`
explodes each code into its binary digits (LSB first), and `packbits`
compresses the flat digit stream 8-to-1 into real bytes. `bitorder=
"little"` must match between pack and unpack — it defines whether digit 0
lands in the low or high bit of each byte; mixing orders scrambles codes
silently. The unpacker is the exact mirror: `unpackbits` → reshape →
dot with `[1, 2, 4, ...]`. The slice `[:, : dim * bits]` drops the zero
padding `packbits` adds to fill the final byte. Roundtrip exactness is
asserted for every width 1–8 in `test_pack_unpack_roundtrip`.

---

## `turboquant/quantizers/adaptive.py`

### `_allocate_bits` — the water-filling core

```python
err = (span.astype(np.float64) ** 2) / 12.0
```
`span²/12` is the variance of a uniform distribution over the range —
i.e., the expected squared error of *storing nothing* (0 bits) and
reconstructing the midpoint. This is each dimension's starting "pressure"
in the water-filling picture. float64 because `err` is repeatedly divided
by 4 — at 8 allocations a float32 would be down 48 bits of exponent
headroom on tiny spans.

```python
for _ in range(self.total_bits):
    d = int(np.argmax(err))
    bits[d] += 1
    err[d] = err[d] / 4.0 if bits[d] < _MAX_BITS else -np.inf
```
One bit granted per iteration to the currently-worst dimension; `/4`
implements `MSE(b) ∝ 4^−b` (one more bit → half the step → quarter the
squared error). Setting a saturated dimension's error to `−inf` removes it
from all future argmax rounds. Greedy is optimal here because the
per-dimension error curves are convex and independent — each grant is the
globally best marginal move (this is why we may loop `total_bits` times
rather than solving anything fancier).

### `_build_layout`

```python
for b in range(1, _MAX_BITS + 1):
    dims = np.where(self.bits_per_dim == b)[0]
```
The packing problem: dimensions now have *different* widths, so a single
`(codes >> shifts)` broadcast no longer works. Solution: reorder the
bitstream so all dims of equal width are contiguous. Then each width group
packs with one vectorized pass — at most 8 groups total, independent of
`dim`. The `(bits, dims, offset)` tuples are the codec's file format:
group `g`'s digits occupy bitstream columns `[offset, offset + b·len(dims))`.
0-bit dims appear in no group: they cost nothing and are handled only at
decode time.

### `decode`

```python
out = np.tile(self.means, (n, 1))
for b, dims, offset in self._groups:
    ...
    out[:, dims] = self.mins[dims] + codes * self.scales[dims]
```
Initialize *every* dimension to its training mean, then overwrite the
quantized ones. The mean is the least-squares optimal reconstruction for a
dimension about which the code says nothing — this is what
`test_adaptive_roundtrip_zero_bit_dims` pins down.

```python
codes = (planes.astype(np.uint16) * weights).sum(axis=2)
```
uint16, not uint8: an 8-bit group's weights reach 128, and `planes * 128`
overflows uint8 for the high bit.

---

## `turboquant/quantizers/product.py`

### `train`

```python
self.codebooks = np.stack([
    kmeans(self._sub(data, m), self.ks, n_iters=n_iters, seed=seed + m)
    for m in range(self.M)
])
```
One *independent* k-means per subspace — the defining approximation of PQ
(it ignores cross-subspace correlation; that's the price of the product
structure). `seed + m` keeps runs reproducible while decorrelating the
random initializations across subspaces.

### `encode`

```python
dists = pairwise_l2_sq(self._sub(data, m), self.codebooks[m])
codes[:, m] = np.argmin(dists, axis=1)
```
Encoding *is* nearest-centroid assignment, per subspace. The `(n, 256)`
distance matrix per subspace never exists all-M-at-once — the loop keeps
peak memory at one subspace's worth.

### `compute_lut` / `adc_distances`

```python
lut[:, m, :] = pairwise_l2_sq(self._sub(queries, m), self.codebooks[m])
```
The whole query-side cost of a search: `M` small `(nq, dsub)×(dsub, 256)`
matmuls, independent of database size `n`.

```python
m_idx = np.arange(self.M)
for qi in range(nq):
    dists[qi] = lut[qi][m_idx, codes].sum(axis=1)
```
The scan. `lut[qi]` is `(M, 256)`; indexing it with `(m_idx, codes)` —
shapes `(M,)` and `(n, M)` — broadcasts to an `(n, M)` gather where
element `[i, m] = lut[qi][m, codes[i, m]]`; `.sum(axis=1)` adds the M
partial squared distances (subspaces are disjoint coordinates, so squared
L2 decomposes as an exact sum). Per query this is `n·M` memory lookups and
adds — zero float vector math, the ADC promise. The Python loop is over
*queries* only; a fully-batched 3-D gather would allocate `(nq, n, M)`.

---

## `turboquant/index/ivfpq.py`

### `train`

```python
assign = np.argmin(pairwise_l2_sq(data, self.coarse_centroids), axis=1)
residuals = data - self.coarse_centroids[assign]
self.pq.train(residuals, ...)
```
The residual trick, at training time: the PQ codebooks are fit to the
distribution they will actually encode — offsets from coarse centroids —
not raw vectors. Training PQ on raw data and encoding residuals with it
would mismatch codebooks to data and forfeit most of the residual win.

### `add`

```python
order = np.argsort(assign, kind="stable")
boundaries = np.searchsorted(assign[order], np.arange(self.nlist + 1))
```
Bucketing n new vectors into nlist lists via one sort + one binary search,
instead of `for v in vectors: lists[cell(v)].append(v)`. After sorting by
cell, each cell's members form a contiguous slice; `searchsorted` finds
all slice boundaries at once. `kind="stable"` preserves insertion order
within a cell — ids stay ascending, which keeps behavior deterministic.

Codes are stored in `self.codes` (one flat `(ntotal, M)` array indexed by
id) while lists hold only ids. The scan pays one gather (`self.codes[ids]`)
per probed cell, and in exchange *any* component can address any vector's
code by id — which is exactly what TurboIndex's re-ranker needs to reuse
tier-1 bytes without duplicating them.

### `search`

```python
coarse = pairwise_l2_sq(queries, self.coarse_centroids)
probe_cells = np.argpartition(coarse, n_probe - 1, axis=1)[:, :n_probe]
```
Cell ranking for all queries in one matmul. The probed cells don't need
sorting (we scan all of them anyway), so `argpartition` suffices.

```python
residual_q = (query - self.coarse_centroids[cell])[None, :]
lut = self.pq.compute_lut(residual_q)
```
Inside `_scan_cells`: the query is shifted into each probed cell's
residual frame before building the LUT. Distances computed against
residual codes are only valid in that frame; this per-cell LUT rebuild is
the (small) price of residual encoding — `nprobe · M · 256 · dsub`
multiply-adds, still independent of `n`.

```python
out_ids = np.full((nq, k), -1, dtype=np.int64)
out_dists = np.full((nq, k), np.inf, dtype=np.float32)
```
Explicit "no result" sentinels (-1 / +inf) rather than ragged returns:
with tiny nprobe on tiny datasets a query can see fewer than k candidates,
and fixed shapes keep the API composable (callers filter `ids >= 0`).

### `reconstruct`

```python
return self.coarse_centroids[self.cells[ids]] + self.pq.decode(self.codes[ids])
```
The additive inverse of the storage scheme: cell centroid plus decoded
residual. Both TurboIndex tiers build on this one-liner.

---

## `turboquant/index/turbo.py`

### `_choose_split`

```python
pq_share = max(4.0, budget_bytes / 3.0)
m1 = max((d for d in divisors if d <= pq_share), default=divisors[0])
```
The ⅓ policy with a floor: tier 1 only generates candidates, so past
"good enough shortlist" its marginal byte is worth less than tier 2's
(verified by the split sweep in CONCEPTS §7 — splits near ⅓ tied or won).
The floor of 4 bytes keeps tier 1 from degenerating (M<4 shortlists were
unusable). The generator-with-max picks the largest valid divisor of
`dim`, since PQ requires `M | dim`.

```python
refine_bits = (budget_bytes - m1) * 8.0 / dim
if refine_bits >= _ADAPTIVE_MIN_BITS:
    return m1, "adaptive", min(refine_bits, 8.0)
```
The measured codec crossover (~3 bits/dim; table in CONCEPTS §7). The
adaptive path returns a *float* bit rate — scalar allocation can absorb
any byte budget exactly, which is what makes TurboIndex budget-continuous.

```python
for cand_m1 in divisors:
    ...
    if used > best_used or (used == best_used and abs(cand_m1 - budget/3) < ...):
```
The low-rate path must pick *two* divisors (M1, M2). Objective order:
first maximize bytes actually used (an unused budget byte is pure waste),
then tie-break toward the ⅓ target. E.g. dim=128, budget=32: (8,16) uses
24B but (16,16) uses all 32 — the sweep showed them tied on recall per
byte, so utilization decides.

### `train`

```python
residual = data - self.ivfpq.coarse_centroids[assign]
stage2 = residual - self.ivfpq.pq.decode(self.ivfpq.pq.encode(residual))
self.refiner.train(stage2)
```
The refiner trains on tier 1's *actual encode-decode error* — the
`encode(decode(...))` roundtrip through tier 1 is deliberate, not
redundant: it produces exactly the distribution the refiner will see at
`add()` time. Training it on residuals (failure 2 in CONCEPTS §7) or raw
data (failure 1) is the difference between the cascade winning and losing.

### `add`

```python
prev = self.ivfpq.ntotal
self.ivfpq.add(vectors)
new_ids = np.arange(prev, self.ivfpq.ntotal)
stage2 = vectors - self.ivfpq.reconstruct(new_ids)
```
Order matters: tier 1 must ingest the vectors first so `reconstruct` can
compute what tier 1 *actually stored* for them; the refiner then encodes
the difference. `refine_codes` row i corresponds to vector id i by
construction (ids are assigned sequentially), so re-ranking can address
codes with plain fancy indexing.

### `search`

```python
shortlist = max(k, rerank_factor * k)
cand_ids, _ = self.ivfpq.search(queries, shortlist, n_probe=n_probe)
```
Tier 1 is asked for a *wider* top list than the caller wants — the cascade
gap. Tier-1 distances are discarded (`_`): they are strictly dominated by
the re-ranked scores, which are computed from a superset of the same bytes.

```python
recon = self.ivfpq.reconstruct(valid) + self.refiner.decode(self.refine_codes[valid])
dists = pairwise_l2_sq(queries[qi : qi + 1], recon)
```
The whole point of the architecture in two lines: reconstruction sums
*every* byte stored for the vector (centroid + tier-1 PQ + tier-2
correction), and the exact query is scored against it. Only
`rerank_factor · k` vectors are ever decoded per query — decode cost is
negligible next to the tier-1 scan, and the raw floats from `add()` are
never touched (they can be thrown away; that's the honest-compression
guarantee).

---

## `turboquant/datasets.py`

```python
scales = variance_decay ** np.arange(dim, dtype=np.float32)
centers = rng.standard_normal((n_clusters, dim)) * scales
points = centers[labels] + rng.standard_normal((n, dim)) * scales * noise
```
A Gaussian mixture with geometrically decaying per-dimension scale.
Both structural properties are load-bearing for honest evaluation:
cluster structure is what IVF exploits (isotropic noise would make every
partition equally bad), and variance decay is what adaptive bit
allocation exploits (uniform variance would make it collapse to plain
SQ). Generating data *without* these properties would fake out both
methods' wins — the generator documents this explicitly.

```python
total = clustered_embeddings(n_train + n_base + n_query, ...)
return total[:n_train], total[n_train:...], ...
```
One draw, three disjoint slices: train/base/query come from the same
distribution (as in real life) but share no vectors, so codebooks can't
memorize the exact vectors they'll encode.

---

## `turboquant/validation.py`

```python
with np.errstate(over="ignore", invalid="ignore"):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
...
if arr.size and not np.isfinite(arr).all():
```
The ordering is the whole trick. `1e308` is a perfectly finite float64
and becomes `inf` the moment it is narrowed to float32, so a finiteness
check *before* the cast passes and lets an infinity into the index.
Checking after catches both the values that arrived non-finite and the
ones the cast created. The `errstate` block exists because that overflow
is expected on this path — we are about to raise a real error about it,
and a `RuntimeWarning` in the logs would just be noise.

```python
bad = np.argwhere(~np.isfinite(arr))
row, col = int(bad[0, 0]), int(bad[0, 1])
```
The error names a coordinate. "Your data contains NaN" sends someone
hunting through ten million rows; "first at row 42, column 5" does not.

```python
if arr.min() < 0 or arr.max() >= ntotal:
    raise IndexError(...)
```
In `check_ids`, the lower bound matters more than the upper one. NumPy
fancy-indexing *accepts* negatives and wraps them to the end of the
array, so without this check `remove_ids([-1])` would silently delete the
most recently added vector instead of raising.

---

## `turboquant/quantizers/opq.py`

### `_eigenvalue_balanced_rotation`

```python
eigvals, eigvecs = np.linalg.eigh(cov)
```
`eigh`, not `eig`: a covariance matrix is symmetric, which guarantees
real eigenvalues and an orthonormal eigenvector basis. `eig` would return
complex dtypes and a non-orthogonal basis — and orthogonality is exactly
the property the whole method depends on.

```python
log_var = np.log(np.maximum(eigvals, 1e-12))
```
The greedy rule balances *summed log* variance because PQ's error tracks
`det(Σ_m)`, a product, and logs turn products into the additive quantity
a greedy load-balancer can equalize. The floor is not cosmetic: a
constant dimension has zero (or slightly negative, from round-off)
variance, and `log(0) = -inf` would make one subspace an infinitely
attractive dumping ground for every remaining direction.

```python
eligible = [m for m in range(n_subspaces) if len(buckets[m]) < dsub]
m = min(eligible, key=lambda m: loads[m])
```
Capacity first, then load. Every subspace must end up with exactly
`dsub` dimensions or the reshape into `(M, dsub)` blocks is invalid, so
the "lightest subspace" rule only ranges over subspaces with room left.

### `_procrustes`

```python
u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
return u @ vt
```
Three lines for the closed-form solution to `min_R ‖XR − Ŷ‖_F` subject to
`RᵀR = I`. Expanding the objective, `‖XR‖_F = ‖X‖_F` for orthogonal `R`,
so minimizing is equivalent to *maximizing* `tr(Rᵀ Xᵀ Ŷ)` — whose
maximizer over the orthogonal group is `U Vᵀ`. The singular values are
discarded (`_`) precisely because we want the rotation without the
scaling. Note the correlation matrix is only `(d, d)`, so this cost is
independent of how many training vectors produced it.

### `train`

```python
if rot_data.shape[0] > self.rotation_sample:
    pick = rng.choice(rot_data.shape[0], size=self.rotation_sample, replace=False)
...
self.pq.train(centered @ self.rotation, n_iters=n_iters, seed=seed)
```
Two different data budgets on purpose. `R` has d² parameters and pins
down from a few thousand vectors (measured: 4k/8 rounds beat 20k/8 rounds
at 8x the speed), so the alternating loop subsamples. The codebooks we
actually ship are then trained on *everything* under the final rotation —
the sample influences where the rotation points, never what gets stored.

### `compute_lut`

```python
return self.pq.compute_lut(self._rotate(queries))
```
The entire query-time cost of OPQ: one `(nq, d) @ (d, d)` matmul,
independent of database size. Because `R` is orthogonal, rotated-space
distances *equal* original-space distances — no approximation enters
here, which is why OPQ is free recall rather than a trade.

---

## `turboquant/quantizers/anisotropic.py`

### `_assign_block`

```python
sq = pairwise_l2_sq(x_m, codebook)
parallel = np.einsum("nd,nd->n", x_m, v_m)[:, None] - v_m @ codebook.T
loss = sq + (self.eta - 1.0) * np.square(other[:, None] + parallel)
```
The coupling between subspaces, made tractable. The anisotropic loss
needs `⟨r, x̄⟩` over the *whole* vector, but blocks are disjoint, so
choosing centroid `c` for block m changes that dot product only by
`(x_m − c)·v_m`. Everything the other blocks contribute is the scalar
`other`, computed once. Both terms are then `(n, ks)` matrices from one
BLAS call each — the loss stays as vectorized as plain PQ's, despite not
decomposing like plain PQ's.

### `_solve_centroids`

```python
lhs = counts[:, None, None] * eye + w * sum_vvt + _RIDGE * eye
rhs = sum_x + w * sum_cv
solved = np.linalg.solve(lhs, rhs[:, :, None])[:, :, 0]
```
The update step that is *not* a mean. Setting the gradient of the summed
loss to zero for cluster j gives
`[N_j I + (η−1) Σ vᵢvᵢᵀ] c = Σ xᵢ + (η−1) Σ (otherᵢ + xᵢ·vᵢ) vᵢ`, one
`dsub × dsub` system per cluster, batched by `np.linalg.solve` over the
leading axis. At `η = 1` the outer-product term vanishes and this
collapses back to `Σx / N` — the k-means mean — which is what makes
`eta=1.0` reproduce plain PQ exactly rather than approximately.

```python
out = np.where(counts[:, None] > 0, solved, previous)
```
The ridge keeps an empty cluster's system solvable, but its solution
would be the origin — dragging a dead centroid to zero, where it stays
dead. Freezing it where it was is the harmless option.

### `_unit_directions`

```python
return np.where(norms > eps, data / np.maximum(norms, eps), 0.0)
```
A zero datapoint has no direction, so there is no parallel component to
penalize; returning zeros makes `⟨r, x̄⟩` vanish and the loss degrades
gracefully to plain PQ error for that row. The `np.maximum` inside the
division is belt-and-braces: `np.where` evaluates *both* branches, so the
division happens even for rows the mask discards, and an unguarded `0/0`
would raise a warning and produce NaN before the selection ran.

---

## `turboquant/index/ivfpq.py` — mutation and filtering

### `remove_ids` / `compact`

```python
newly = int((~self.deleted[ids]).sum())
self.deleted[ids] = True
```
Tombstoning, not deletion. Physically removing row i would renumber every
row after it, invalidating ids the application stored. The return value
counts *newly* removed ids so that re-deleting is idempotent and
observably so.

```python
mapping = np.full(self.n_slots, -1, dtype=np.int64)
mapping[keep] = np.arange(int(keep.sum()), dtype=np.int64)
return mapping
```
`compact` returns the old→new map rather than applying it privately,
because it cannot know what else is keyed by id — TurboIndex's tier-2
codes, an application's metadata table. Handing the map back makes the
dependency explicit; hiding it would produce silently misaligned indexes.

### `update`

```python
moved = ids[changed]
for cell in np.unique(self.cells[moved]):
    leaving = moved[self.cells[moved] == cell]
    self.list_ids[cell] = self.list_ids[cell][~np.isin(self.list_ids[cell], leaving)]
```
Only ids whose *cell* changed are unfiled, and the loop runs over
affected cells rather than affected ids, so each inverted list is
rewritten at most once. Skipping this refiling is the subtle bug the
tests target: the code would be correct, the vector would be findable
only by queries probing where it used to live.

```python
for cell in np.unique(destinations):
    self.list_ids[cell] = np.sort(self.list_ids[cell])
```
This one is not about correctness in the usual sense, and it was added
after measuring. `add` emits ascending ids, so lists are naturally
sorted; refiling appends a moved id at the end and breaks that. The
consequence is invisible until you look for it — the mutated index had
byte-identical codes, cells and list *contents* to a freshly rebuilt one,
and identical distances for every query, but 62 of 2000 returned ids
differed. All of them were exact distance ties, broken differently
because the scan concatenates lists in storage order. ADC distances are
sums of table entries, so exact ties are common.

Sorting makes index state a function of *contents alone*, never of
mutation history. That turns "rebuild it and diff" into a usable
debugging tool, which it is not when a legitimate tie and a real
regression look the same.

### `_probe_until_enough`

```python
ids, dists = self._scan_cells(query, ranked_cells[scanned:target])
...
scanned = target
if kept >= k or scanned >= max_probe:
    break
target = min(max_probe, max(target * 2, target + 1))
```
Escalation, in four lines. The slice starts at `scanned`, so widening
scans only the *newly added* cells — no cell is ever scanned twice, and
the accumulated chunks carry forward. Doubling gives geometric growth so
a very selective filter reaches full coverage in `log₂(nlist)` rounds,
and `max(target*2, target+1)` keeps it moving when `target` is 1. The
loop always executes once, so the unfiltered path costs exactly what it
did before this feature existed.

```python
coarse = pairwise_l2_sq(queries, self.coarse_centroids)
ranked = np.argsort(coarse, axis=1)
```
A full sort where the pre-escalation code used `argpartition`. Escalation
needs cells in distance order *beyond* the initial `n_probe`, which a
partition does not provide. `nlist` is in the hundreds, so sorting it is
negligible beside the ADC scan it feeds — the one place in this library
where `argsort` beats `argpartition` on merit.

---

## `turboquant/io.py`

```python
with np.load(path, allow_pickle=False) as data:
```
The single most important argument in the module. `allow_pickle=True`
executes arbitrary code contained in the file, which turns "restore the
index from object storage" into remote code execution the moment that
bucket is writable by anyone else. Everything here is plain arrays plus a
JSON manifest specifically so this can stay `False`.

```python
else:
    raise TypeError(f"cannot serialize attribute {name}.{attr} ...")
```
The reflective walk refuses to guess. Hand-written `to_dict`/`from_dict`
pairs fail silently — someone adds a field, forgets the serializer, and
reloaded indexes are subtly wrong forever. Here an unrecognized attribute
raises at *save* time, so the mistake surfaces immediately.

```python
obj = cls.__new__(cls)
```
Construction without `__init__`, because every attribute is about to be
restored verbatim. Running the constructor would recompute derived
configuration — TurboIndex's budget split, for instance — and risk
disagreeing with the state actually stored alongside it.

```python
_DERIVED = {"IVFPQIndex": {"list_ids"}, ...}
```
Inverted lists are a permutation of the cell assignments, and they cost
*more* than the codes they index (8 bytes/vector of int64 ids against
8–32 bytes of code). Dropping and rebuilding them makes archives roughly
half the size, at the cost of one `argsort` on load.

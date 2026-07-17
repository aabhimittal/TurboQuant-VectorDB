"""TurboIndex: budget-driven two-stage residual cascade (the novel index).

Standard IVFPQ has one knob too few: you pick M (bytes/vector) and accept
whatever recall falls out -- and M must divide the dimension, so entire
regions of the memory/recall curve are simply unreachable. TurboIndex
instead takes a single *memory budget in bytes per vector* and splits it
across a two-tier cascade in which the second tier encodes what the first
tier got wrong:

    Tier 1 -- candidate generation (IVFPQ, ~1/3 of the budget):
        coarse centroid c + PQ code of the residual (x - c).
        Scans nprobe inverted lists with ADC and emits a shortlist of
        rerank_factor * k candidates. A small tier-1 M keeps the scan
        cheap: the per-vector scan cost is M table lookups, so a
        16-lookup scan + tiny re-rank beats a 32-lookup scan outright.

    Tier 2 -- error correction (~2/3 of the budget):
        encodes the SECOND-stage residual  r2 = x - c - PQ1_decode(code),
        i.e. exactly the error tier 1 makes on this vector. At re-rank
        time a candidate is reconstructed from ALL of its stored bytes:

            x_hat = c + PQ1_decode(code) + refine_decode(refine_code)

        and re-scored exactly against x_hat.

The tier-2 codec is chosen by a measured crossover ("auto" mode):

    refine budget < 3 bits/dim  ->  a second ProductQuantizer.
        Vector quantization is far more bit-efficient than scalar
        codes in the low-bit regime (a 256-centroid codebook over 8
        dims spends 1 bit/dim but captures inter-dim correlation).

    refine budget >= 3 bits/dim ->  AdaptiveBitQuantizer.
        At high rates scalar codes approach lossless while PQ codebooks
        saturate (256 centroids can only cut error so far), and the
        variance-driven bit allocation spends the budget where the
        stage-2 error actually lives. Measured on 128-dim data at an
        80-byte budget: adaptive 0.925 recall@10 vs second-PQ 0.914.

Why the cascade helps, in three properties:

* Shrinking targets. Each stage quantizes a distribution far tighter than
  the last (raw space -> cell residual -> post-PQ error), so every stored
  bit acts on a small range. This is the classic residual/additive
  quantization structure, here driven by a single byte budget.

* No information is discarded. Naive "refine" re-ranking scores
  candidates with an independent code and throws tier 1's bytes away --
  at a fixed total budget that can *lose* to plain IVFPQ (we measured
  exactly that during development; see docs/CONCEPTS.md). Summing the
  stages means every stored byte contributes to the final score, so
  widening the shortlist can never systematically hurt.

* Compression is honest end to end. FAISS's IndexRefineFlat re-ranks
  against raw float32, so RAM still holds 4*d bytes/vector. Here the
  floats are never consulted after add().
"""

from __future__ import annotations

import numpy as np

from ..metrics import pairwise_l2_sq, top_k
from ..quantizers.adaptive import AdaptiveBitQuantizer
from ..quantizers.product import ProductQuantizer
from .ivfpq import IVFPQIndex

# Measured crossover between the two tier-2 codecs (see module docstring).
_ADAPTIVE_MIN_BITS = 3.0


def _divisors(dim: int) -> list[int]:
    return [m for m in range(1, dim + 1) if dim % m == 0]


def _choose_split(dim: int, budget_bytes: float) -> tuple[int, str, float]:
    """Split a byte budget into (tier1_M, refine_kind, refine_param).

    Policy: aim tier 1 at ~1/3 of the budget (candidate generation only
    needs to land true neighbors in the shortlist; past that, a byte
    spent on tier 2 improves the final ranking more), then hand the rest
    to the refiner. If the refiner would run below the scalar-efficiency
    crossover, use a second PQ and pick the divisor pair (M1, M2) that
    wastes the fewest budget bytes; otherwise use adaptive scalar bits,
    which can absorb any byte budget exactly.

    Returns:
        (tier1_M, "pq", tier2_M) or (tier1_M, "adaptive", avg_bits).
    """
    if budget_bytes < 5:
        raise ValueError("budget_bytes must be at least 5 (4 for PQ + 1 refine)")
    divisors = _divisors(dim)
    pq_share = max(4.0, budget_bytes / 3.0)
    m1 = max((d for d in divisors if d <= pq_share), default=divisors[0])

    refine_bits = (budget_bytes - m1) * 8.0 / dim
    if refine_bits >= _ADAPTIVE_MIN_BITS:
        return m1, "adaptive", min(refine_bits, 8.0)

    # Low-rate regime: second PQ. Among divisor pairs within budget, take
    # the one using the most bytes; break ties toward the 1/3 target.
    best: tuple[int, int] | None = None
    for cand_m1 in divisors:
        if cand_m1 < 4:
            continue  # tier 1 too coarse to produce a usable shortlist
        rest = budget_bytes - cand_m1
        m2 = max((d for d in divisors if d <= rest), default=0)
        if m2 == 0:
            continue
        if best is None:
            best = (cand_m1, m2)
            continue
        used, best_used = cand_m1 + m2, best[0] + best[1]
        if used > best_used or (
            used == best_used
            and abs(cand_m1 - budget_bytes / 3) < abs(best[0] - budget_bytes / 3)
        ):
            best = (cand_m1, m2)
    if best is None:
        # Budget too small for two PQ stages; fall back to adaptive bits.
        return m1, "adaptive", max(0.25, refine_bits)
    return best[0], "pq", float(best[1])


class TurboIndex:
    def __init__(
        self,
        dim: int,
        budget_bytes: float = 32.0,
        n_lists: int = 256,
        refine: str = "auto",
    ):
        """Args:
        dim: vector dimensionality.
        budget_bytes: total storage budget per vector, both tiers included.
        n_lists: IVF cells for tier 1.
        refine: tier-2 codec -- "auto" (measured crossover), "pq", or
            "adaptive".
        """
        if refine not in ("auto", "pq", "adaptive"):
            raise ValueError('refine must be "auto", "pq", or "adaptive"')
        self.dim = dim
        self.budget_bytes = budget_bytes

        m1, kind, param = _choose_split(dim, budget_bytes)
        if refine != "auto" and kind != refine:
            # Explicit override: recompute the refiner for the forced kind.
            m1 = max((d for d in _divisors(dim) if d <= max(4.0, budget_bytes / 3.0)))
            rest = budget_bytes - m1
            if refine == "adaptive":
                kind, param = "adaptive", min(max(0.25, rest * 8.0 / dim), 8.0)
            else:
                m2 = max((d for d in _divisors(dim) if d <= rest), default=0)
                if m2 == 0:
                    raise ValueError(
                        f"budget {budget_bytes}B leaves no room for a PQ refiner"
                    )
                kind, param = "pq", float(m2)

        self.ivfpq = IVFPQIndex(dim, n_lists=n_lists, n_subspaces=m1)
        self.refine_kind = kind
        if kind == "pq":
            self.refiner: ProductQuantizer | AdaptiveBitQuantizer = ProductQuantizer(
                dim, n_subspaces=int(param)
            )
        else:
            self.refiner = AdaptiveBitQuantizer(dim, avg_bits=param)
        # Tier-2 codes indexed by insertion id (row i = vector id i).
        self.refine_codes: np.ndarray | None = None
        self.is_trained = False

    def train(self, data: np.ndarray, n_iters: int = 25, seed: int = 0) -> "TurboIndex":
        data = np.asarray(data, dtype=np.float32)
        self.ivfpq.train(data, n_iters=n_iters, seed=seed)
        # Train the refiner on the tier-1 *error* of the training set --
        # the exact distribution it will encode at add() time.
        assign = np.argmin(
            pairwise_l2_sq(data, self.ivfpq.coarse_centroids), axis=1
        )
        residual = data - self.ivfpq.coarse_centroids[assign]
        stage2 = residual - self.ivfpq.pq.decode(self.ivfpq.pq.encode(residual))
        self.refiner.train(stage2)
        self.is_trained = True
        return self

    def add(self, vectors: np.ndarray) -> None:
        if not self.is_trained:
            raise RuntimeError("TurboIndex must be trained before add()")
        vectors = np.asarray(vectors, dtype=np.float32)
        prev = self.ivfpq.ntotal
        self.ivfpq.add(vectors)
        # Tier-1 reconstruction of the vectors just added; the refiner
        # stores what tier 1 could not represent.
        new_ids = np.arange(prev, self.ivfpq.ntotal)
        stage2 = vectors - self.ivfpq.reconstruct(new_ids)
        codes = self.refiner.encode(stage2)
        if self.refine_codes is None:
            self.refine_codes = codes
        else:
            self.refine_codes = np.vstack([self.refine_codes, codes])

    def search(
        self,
        queries: np.ndarray,
        k: int,
        n_probe: int = 8,
        rerank_factor: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two-tier cascade: IVFPQ shortlist, then full-budget re-rank.

        rerank_factor controls the cascade width: tier 1 hands
        rerank_factor * k candidates to tier 2. Because tier 2 scores
        with strictly more information than tier 1 (its own bytes PLUS
        tier 1's), widening the shortlist cannot systematically hurt.
        """
        queries = np.asarray(queries, dtype=np.float32)
        nq = queries.shape[0]
        shortlist = max(k, rerank_factor * k)
        cand_ids, _ = self.ivfpq.search(queries, shortlist, n_probe=n_probe)

        out_ids = np.full((nq, k), -1, dtype=np.int64)
        out_dists = np.full((nq, k), np.inf, dtype=np.float32)
        for qi in range(nq):
            valid = cand_ids[qi][cand_ids[qi] >= 0]
            if len(valid) == 0:
                continue
            # Full reconstruction: tier 1 (centroid + PQ) plus tier 2's
            # stored correction. Floats from add() are never consulted.
            recon = self.ivfpq.reconstruct(valid) + self.refiner.decode(
                self.refine_codes[valid]
            )
            dists = pairwise_l2_sq(queries[qi : qi + 1], recon)
            idx, d = top_k(dists, min(k, len(valid)))
            out_ids[qi, : idx.shape[1]] = valid[idx[0]]
            out_dists[qi, : idx.shape[1]] = d[0]
        return out_ids, out_dists

    @property
    def memory_bytes(self) -> int:
        """Both tiers' codes -- everything search touches."""
        refine = 0 if self.refine_codes is None else self.refine_codes.nbytes
        return self.ivfpq.memory_bytes + refine

    @property
    def bytes_per_vector(self) -> float:
        n = self.ivfpq.ntotal
        return self.memory_bytes / n if n else 0.0

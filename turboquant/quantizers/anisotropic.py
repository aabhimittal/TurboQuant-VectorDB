"""Anisotropic PQ: quantize for the *score*, not for the reconstruction.

Every quantizer so far minimizes reconstruction error ||x - x_hat||^2,
which treats all error directions as equally bad. For maximum
inner-product search (MIPS) that is measurably the wrong objective, and
the reason is worth spelling out.

What search actually cares about is the error in the score:

    <q, x> - <q, x_hat> = <q, r>,     r = x - x_hat

Split the residual into the part along the datapoint and the part across
it:  r = r_par + r_perp, with r_par = (<r, x_bar>) x_bar and x_bar the
unit vector along x. Now condition on the case that matters: x is a top
result for q, which means q points roughly *along* x. Then <q, r_perp>
largely cancels -- q has little component in those directions and their
signs are unrelated to x -- while <q, r_par> adds coherently to the score
every time. Error parallel to the datapoint corrupts the ranking; error
perpendicular to it mostly does not.

So we weight them differently (Guo et al., "Accelerating Large-Scale
Inference with Anisotropic Vector Quantization", ICML 2020):

    loss(x, x_hat) = eta * ||r_par||^2 + ||r_perp||^2
                   = ||r||^2 + (eta - 1) * <r, x_bar>^2

with eta > 1. Note what the second form says: this is ordinary PQ loss
plus a penalty on the parallel component. eta = 1 recovers plain PQ
exactly, which makes the knob easy to reason about and easy to test.

The catch is that the penalty term couples the subspaces -- <r, x_bar>
is a full-vector quantity, so subspace m's best code now depends on what
the other subspaces chose. Plain PQ has no such coupling, which is why it
can train each subspace with an independent k-means. We therefore
optimize by block coordinate descent, sweeping subspace by subspace:

    assign step: with all other blocks frozen, the loss restricted to
        block m is a closed-form quadratic in the candidate centroid, so
        the best code is one (n, ks) argmin -- still fully vectorized.

    update step: with the assignment frozen, the optimal centroid solves
        a small (dsub x dsub) linear system per cluster (`_solve_centroids`)
        rather than being a plain mean. The mean is what you get when
        eta = 1; the anisotropic weighting tilts it along the datapoint
        directions of the cluster's members.

Storage and search are completely unchanged from PQ -- same codebooks,
same M bytes per vector, same lookup tables. All of the difference is in
where the centroids sit.

**When this does nothing -- and when it backfires.** Measured on 128-dim
data, M=16, 1000 queries, MIPS recall@10 against plain PQ:

                  PQ     eta=2   eta=4   eta=8   eta=16
    raw          0.141   0.240   0.420   0.553   0.579
    unit-norm    0.079   0.078   0.070   0.057   0.029

On the unit sphere the method does not merely stop helping, it degrades
monotonically. That is the premise expiring, not a bug. The asymmetry
buys something because a datapoint's norm scales its score, so error
along x moves the score in proportion to ||x||. Normalize and every norm
is 1, the score becomes pure angle, MIPS collapses into cosine -- which
is L2 on unit vectors (see `metrics.normalize`) -- and the parallel
direction is no longer special. Keep weighting it anyway and you are just
spending accuracy for nothing, which is exactly the shape of that bottom
row. Use AnisotropicPQ for MIPS over unnormalized vectors; for cosine,
normalize and use any L2 index in this library.
"""

from __future__ import annotations

import numpy as np

from ..metrics import pairwise_l2_sq
from ..validation import as_2d_float32, warn_small_training_set
from .product import ProductQuantizer

# Ridge added to the normal equations so empty/degenerate clusters give a
# solvable system instead of a LinAlgError.
_RIDGE = 1e-6


class AnisotropicPQ(ProductQuantizer):
    """PQ trained under the score-aware loss. Same code size, better MIPS.

    Args:
        dim, n_subspaces, n_centroids: as ProductQuantizer.
        eta: weight on parallel residual error. 1.0 reproduces plain PQ
            exactly. Measured on 128-dim clustered data, M=16, recall@10:

                eta    MIPS    L2     recon MSE
                1.0    0.126   0.117  2.67
                4.0    0.438   0.125  2.81
                8.0    0.566   0.127  2.90
                16.0   0.605   0.135  3.00
                32.0   0.608   0.119  3.15

            Read the columns together, because they are the whole point:
            MIPS recall nearly 5x's while reconstruction error gets
            steadily *worse*. The quantizer is deliberately spending
            accuracy where it does not affect scores. Gains saturate
            around 16 while MSE keeps climbing, so 8.0 is the default.
        n_sweeps: block coordinate descent passes during training.
    """

    def __init__(
        self,
        dim: int,
        n_subspaces: int = 8,
        n_centroids: int = 256,
        eta: float = 8.0,
        n_sweeps: int = 6,
    ):
        super().__init__(dim, n_subspaces, n_centroids)
        eta = float(eta)
        if not np.isfinite(eta) or eta < 1.0:
            raise ValueError("eta must be finite and >= 1 (1.0 == plain PQ)")
        if int(n_sweeps) < 0:
            raise ValueError("n_sweeps must be >= 0")
        self.eta = eta
        self.n_sweeps = int(n_sweeps)

    # ------------------------------------------------------------------ train
    def train(
        self, data: np.ndarray, n_iters: int = 25, seed: int = 0
    ) -> "AnisotropicPQ":
        data = as_2d_float32(data, name="training data", dim=self.dim)
        warn_small_training_set(data.shape[0], self.ks, "AnisotropicPQ")

        # Start from the plain-PQ solution: it is the eta = 1 optimum, so
        # coordinate descent begins somewhere sensible and only has to
        # migrate the centroids toward the anisotropic optimum.
        super().train(data, n_iters=n_iters, seed=seed)
        if self.eta == 1.0 or self.n_sweeps == 0:
            return self

        directions = _unit_directions(data)
        codes = super().encode(data)
        for _ in range(self.n_sweeps):
            codes = self._sweep(data, directions, codes, update=True)
        self.is_trained = True
        return self

    def _sweep(
        self,
        data: np.ndarray,
        directions: np.ndarray,
        codes: np.ndarray,
        *,
        update: bool,
    ) -> np.ndarray:
        """One block coordinate descent pass over the subspaces.

        Maintains the full residual `r` incrementally so each subspace can
        read off the other blocks' contribution to <r, x_bar> in O(n)
        instead of recomputing the whole thing.
        """
        recon = self.decode(codes)
        residual = data - recon
        # <r, x_bar> for the whole vector; the per-block correction below
        # subtracts just the current block's share.
        r_dot_dir = np.einsum("nd,nd->n", residual, directions)

        for m in range(self.M):
            sl = slice(m * self.dsub, (m + 1) * self.dsub)
            x_m = data[:, sl]
            v_m = directions[:, sl]
            # Contribution to <r, x_bar> from every block except this one.
            other = r_dot_dir - np.einsum("nd,nd->n", residual[:, sl], v_m)

            new_codes = self._assign_block(x_m, v_m, other, self.codebooks[m])
            if update:
                self.codebooks[m] = self._solve_centroids(
                    x_m, v_m, other, new_codes, self.codebooks[m]
                )
                # Re-assign against the moved centroids so the residual we
                # carry into the next subspace is the true one.
                new_codes = self._assign_block(x_m, v_m, other, self.codebooks[m])

            codes[:, m] = new_codes
            # Refresh this block's residual and the running dot product.
            residual[:, sl] = x_m - self.codebooks[m][new_codes]
            r_dot_dir = other + np.einsum("nd,nd->n", residual[:, sl], v_m)
        return codes

    def _assign_block(
        self,
        x_m: np.ndarray,
        v_m: np.ndarray,
        other: np.ndarray,
        codebook: np.ndarray,
    ) -> np.ndarray:
        """Best centroid per point for one subspace, under the full loss.

        Dropping the terms that do not depend on the candidate c, the loss
        of choosing c for point i is

            ||x_i - c||^2 + (eta - 1) * (other_i + (x_i - c).v_i)^2

        Both pieces are (n, ks) matrices built from one BLAS call each.
        """
        sq = pairwise_l2_sq(x_m, codebook)  # (n, ks)
        # (x_i - c).v_i  =  x_i.v_i  -  c.v_i
        parallel = np.einsum("nd,nd->n", x_m, v_m)[:, None] - v_m @ codebook.T
        loss = sq + (self.eta - 1.0) * np.square(other[:, None] + parallel)
        return np.argmin(loss, axis=1).astype(np.uint8)

    def _solve_centroids(
        self,
        x_m: np.ndarray,
        v_m: np.ndarray,
        other: np.ndarray,
        codes: np.ndarray,
        previous: np.ndarray,
    ) -> np.ndarray:
        """Optimal centroids for one subspace given fixed assignments.

        Setting the gradient of the summed loss to zero for cluster j:

            [N_j I + (eta-1) sum_i v_i v_i^T] c
                = sum_i x_i + (eta-1) sum_i (other_i + x_i.v_i) v_i

        i.e. a dsub x dsub solve per cluster instead of a mean. Every
        accumulation is a segment sum over the assignment, done with
        `np.bincount` per column.
        """
        ks, dsub = self.ks, self.dsub
        w = self.eta - 1.0
        counts = np.bincount(codes, minlength=ks).astype(np.float64)

        sum_x = _segment_sum(codes, x_m, ks)  # (ks, dsub)
        coef = other + np.einsum("nd,nd->n", x_m, v_m)  # (n,)
        sum_cv = _segment_sum(codes, v_m * coef[:, None], ks)  # (ks, dsub)
        outer = (v_m[:, :, None] * v_m[:, None, :]).reshape(-1, dsub * dsub)
        sum_vvt = _segment_sum(codes, outer, ks).reshape(ks, dsub, dsub)

        eye = np.eye(dsub)
        lhs = counts[:, None, None] * eye + w * sum_vvt + _RIDGE * eye
        rhs = sum_x + w * sum_cv
        solved = np.linalg.solve(lhs, rhs[:, :, None])[:, :, 0]

        # Clusters that lost all their members have no evidence to move
        # on; the ridge would drag them to the origin, so freeze them.
        out = np.where(counts[:, None] > 0, solved, previous)
        return np.ascontiguousarray(out, dtype=np.float32)

    # ----------------------------------------------------------------- encode
    def encode(self, data: np.ndarray, n_sweeps: int = 2) -> np.ndarray:
        """Encode under the same anisotropic loss used for training.

        Encoding with plain nearest-centroid assignment would throw away
        part of the benefit: the codebooks sit at anisotropic optima, so
        the codes should be chosen anisotropically too. Two coordinate
        descent sweeps from the plain-PQ assignment recover nearly all of
        it; the codebooks are held fixed here, only the codes move.
        """
        self._check_trained()
        data = as_2d_float32(data, name="data", dim=self.dim, allow_empty=True)
        codes = super().encode(data)
        if self.eta == 1.0 or data.shape[0] == 0:
            return codes
        directions = _unit_directions(data)
        for _ in range(max(0, int(n_sweeps))):
            codes = self._sweep(data, directions, codes, update=False)
        return codes


# --------------------------------------------------------------------- helpers
def _unit_directions(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Unit vectors along each row; the zero vector maps to zero.

    A zero datapoint has no direction, so there is no parallel component
    to penalize. Returning zeros makes <r, x_bar> vanish for that row and
    the loss collapses to plain PQ error -- the right limit, and it avoids
    a 0/0 that would put NaN into every centroid the row touches.
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return np.where(norms > eps, data / np.maximum(norms, eps), 0.0).astype(np.float32)


def _segment_sum(codes: np.ndarray, values: np.ndarray, ks: int) -> np.ndarray:
    """Sum `values` rows into `ks` buckets given by `codes`. Returns (ks, c)."""
    return np.stack(
        [
            np.bincount(codes, weights=values[:, j], minlength=ks)
            for j in range(values.shape[1])
        ],
        axis=1,
    )

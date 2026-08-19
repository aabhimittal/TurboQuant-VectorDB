"""OPQ: Optimized Product Quantization -- learn a rotation before splitting.

Plain PQ carves the vector into contiguous slices: dims 0-15 go to
subspace 0, dims 16-31 to subspace 1, and so on. That split is an
accident of how the embedding model happened to order its output
neurons, and it costs real recall for two reasons:

1. **Unbalanced energy.** Embedding variance is heavily skewed (our own
   generator decays it geometrically, matching real embedding spectra).
   Contiguous slicing hands one subspace most of the variance and another
   almost none. Every subspace gets the same 256-entry codebook, so the
   high-variance subspace is badly under-resourced while the low-variance
   one wastes its 8 bits describing noise. Total error is dominated by
   the worst subspace.

2. **Ignored correlation.** PQ assumes subspaces are independent -- its
   reconstruction is a concatenation, so it can represent no correlation
   *between* subspaces at all. Whatever mutual information the split
   leaves straddling a boundary is simply lost.

OPQ (Ge, He, Ke & Sun, CVPR 2013) fixes both by inserting a learned
orthogonal rotation R before the split:

    encode(x) = PQ.encode((x - mu) @ R)
    decode(c) = PQ.decode(c) @ R.T + mu

A rotation is exactly the right tool because it is *free at search time*
and *lossless in distance*: R orthogonal means ||q - x|| == ||qR - xR||,
so we can rotate the query once (a d x d matvec, independent of database
size) and then do ordinary ADC in the rotated space. The per-vector
storage is unchanged -- still M bytes. R itself is d x d floats of
*shared* overhead, amortized across the whole index.

How R is learned, in two stages:

**Stage 1 -- parametric init (eigenvalue balancing).** Diagonalize the
covariance, then deal the eigen-directions out to subspaces greedily:
each direction, in descending eigenvalue order, goes to whichever
subspace currently has the smallest log-variance sum. This equalizes
`det(Sigma_m)` across subspaces, which is the quantity PQ's error
actually depends on. Decorrelation comes free from using eigenvectors.

**Stage 2 -- non-parametric refinement (alternating optimization).**
Repeat: (a) freeze R, train PQ on the rotated data; (b) freeze the PQ
reconstruction Y_hat, and solve for the R that best maps X onto it,

    min_R ||X R - Y_hat||_F^2   subject to   R^T R = I

which is the orthogonal Procrustes problem with the closed form
R = U V^T from the SVD  X^T Y_hat = U S V^T. Each half is a global
optimum for its variable, so the objective decreases monotonically.
"""

from __future__ import annotations

import numpy as np

from ..validation import as_2d_float32, check_dim
from .base import BaseQuantizer
from .product import ProductQuantizer


class OPQProductQuantizer(BaseQuantizer):
    """PQ preceded by a learned orthogonal rotation. Same bytes, better recall.

    Args:
        dim: vector dimensionality; must be divisible by n_subspaces.
        n_subspaces: M, as in ProductQuantizer. Bytes per vector at ks=256.
        n_centroids: ks, codebook size per subspace.
        n_rotation_iters: alternating optimization rounds. 0 keeps the
            parametric (eigenvalue-balanced) rotation, which is already
            most of the win and costs one eigendecomposition; each extra
            round costs one PQ training on the rotation sample.
        rotation_sample: cap on training points used for the alternating
            phase. R has only d^2 parameters, so a few thousand vectors
            pin it down as well as the full set: measured on 128-dim data
            at M=16, a 4k sample for 8 rounds scored 0.251 recall@10 in
            5.5s where the full 20k set scored 0.240 in 42.7s. The final
            codebooks are always trained on all the data.
    """

    def __init__(
        self,
        dim: int,
        n_subspaces: int = 8,
        n_centroids: int = 256,
        n_rotation_iters: int = 8,
        rotation_sample: int = 4096,
    ):
        self.dim = check_dim(dim)
        if int(n_rotation_iters) < 0:
            raise ValueError("n_rotation_iters must be >= 0")
        if int(rotation_sample) < 1:
            raise ValueError("rotation_sample must be >= 1")
        self.n_rotation_iters = int(n_rotation_iters)
        self.rotation_sample = int(rotation_sample)
        # Constructor validates dim % M == 0 and the ks range for us.
        self.pq = ProductQuantizer(self.dim, n_subspaces, n_centroids)
        self.M = self.pq.M
        self.ks = self.pq.ks
        self.rotation: np.ndarray | None = None  # (d, d) orthogonal
        self.mean: np.ndarray | None = None  # (d,) pre-rotation translation

    # ------------------------------------------------------------------ train
    def train(
        self, data: np.ndarray, n_iters: int = 25, seed: int = 0
    ) -> "OPQProductQuantizer":
        data = as_2d_float32(data, name="training data", dim=self.dim)
        self.mean = data.mean(axis=0)
        centered = data - self.mean

        self.rotation = _eigenvalue_balanced_rotation(centered, self.M)

        # Alternating optimization runs on a sample: R is a d x d object and
        # converges long before the codebooks would. Cheap inner k-means
        # too -- the codebooks here only have to point R in the right
        # direction, not be the ones we ship.
        rot_data = centered
        if rot_data.shape[0] > self.rotation_sample:
            rng = np.random.default_rng(seed)
            pick = rng.choice(rot_data.shape[0], size=self.rotation_sample, replace=False)
            rot_data = np.ascontiguousarray(rot_data[pick])

        if rot_data.shape[0] >= self.ks:
            for _ in range(self.n_rotation_iters):
                rotated = rot_data @ self.rotation
                self.pq.train(rotated, n_iters=max(4, n_iters // 3), seed=seed)
                approx = self.pq.decode(self.pq.encode(rotated))
                self.rotation = _procrustes(rot_data, approx)

        # Ship codebooks trained on everything, under the final rotation.
        self.pq.train(centered @ self.rotation, n_iters=n_iters, seed=seed)
        self.is_trained = True
        return self

    # ----------------------------------------------------------- encode/decode
    def _rotate(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) @ self.rotation

    def encode(self, data: np.ndarray) -> np.ndarray:
        self._check_trained()
        data = as_2d_float32(data, name="data", dim=self.dim, allow_empty=True)
        return self.pq.encode(self._rotate(data))

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Codes -> original-space vectors (rotate back, then un-center)."""
        self._check_trained()
        return self.pq.decode(codes) @ self.rotation.T + self.mean

    # ---------------------------------------------------------------- search
    def compute_lut(self, queries: np.ndarray) -> np.ndarray:
        """ADC tables in the rotated space.

        Rotating the query is the entire query-time cost of OPQ: one
        (nq, d) @ (d, d) matmul, independent of database size. Because R
        is orthogonal the rotated-space distances equal the original-space
        distances exactly -- no approximation is introduced here.
        """
        self._check_trained()
        queries = as_2d_float32(queries, name="queries", dim=self.dim)
        return self.pq.compute_lut(self._rotate(queries))

    def adc_distances(self, lut: np.ndarray, codes: np.ndarray) -> np.ndarray:
        return self.pq.adc_distances(lut, codes)

    @property
    def bytes_per_vector(self) -> float:
        return self.pq.bytes_per_vector  # rotation is shared, not per-vector

    @property
    def shared_bytes(self) -> int:
        """One-off overhead: the rotation matrix and the mean.

        Reported separately from `bytes_per_vector` because it does not
        scale with the database. At d=128 it is 64 KB total -- irrelevant
        next to a million 16-byte codes, but worth stating honestly.
        """
        self._check_trained()
        return int(self.rotation.nbytes + self.mean.nbytes)


# --------------------------------------------------------------------- helpers
def _procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orthogonal R minimizing ||source @ R - target||_F.

    Closed form: with source^T @ target = U S V^T, the maximizer of
    tr(R^T source^T target) over the orthogonal group is R = U V^T. The
    correlation matrix is only (d, d), so the SVD cost is independent of
    how many training vectors were used.
    """
    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return np.ascontiguousarray(u @ vt, dtype=np.float32)


def _eigenvalue_balanced_rotation(centered: np.ndarray, n_subspaces: int) -> np.ndarray:
    """PCA rotation with eigen-directions dealt out to balance subspaces.

    PQ's expected error per subspace grows with that subspace's
    generalized variance, so the best contiguous split is the one where
    every subspace has a comparable one. We sort the eigen-directions by
    variance and greedily give each to the currently "lightest" subspace,
    measured in summed log-variance (log turns the product `det(Sigma_m)`
    into an additive quantity the greedy rule can balance).

    Returns a (d, d) float32 matrix whose columns are eigenvectors,
    ordered so that columns [m*dsub, (m+1)*dsub) are subspace m's block.
    """
    d = centered.shape[1]
    dsub = d // n_subspaces
    # eigh (not eig) because a covariance matrix is symmetric: real
    # eigenvalues, guaranteed-orthonormal eigenvectors, and no complex dtype.
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # descending variance
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Floor at a tiny positive value: constant dimensions have zero (or
    # slightly negative, from round-off) variance and log(0) = -inf would
    # make one subspace an infinitely attractive dumping ground.
    log_var = np.log(np.maximum(eigvals, 1e-12))

    buckets: list[list[int]] = [[] for _ in range(n_subspaces)]
    loads = np.zeros(n_subspaces)
    for i in range(d):
        # Only subspaces with room left are eligible.
        eligible = [m for m in range(n_subspaces) if len(buckets[m]) < dsub]
        m = min(eligible, key=lambda m: loads[m])
        buckets[m].append(i)
        loads[m] += log_var[i]

    column_order = np.concatenate([np.array(b, dtype=np.int64) for b in buckets])
    return np.ascontiguousarray(eigvecs[:, column_order], dtype=np.float32)

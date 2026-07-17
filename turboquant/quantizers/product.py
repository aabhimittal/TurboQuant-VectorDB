"""Product Quantization (PQ) with Asymmetric Distance Computation (ADC).

The workhorse of billion-scale vector search (Jegou, Douze & Schmid, 2011)
and the "PQ" inside FAISS's IVFPQ, ScaNN, and most managed vector DBs.

Core idea: split each d-dim vector into M subvectors of d/M dims and
quantize each subvector independently against its own 256-entry codebook.
One vector then costs M bytes (one codebook id per subspace), but the
*effective* codebook over the full vector is the cartesian product:
256^M distinct reconstructions. For d=128, M=8: 16 bytes/vector (32x
compression) with 256^8 ~ 1.8e19 virtual centroids -- a codebook that
large could never be trained or stored explicitly; the product structure
is what makes it representable.

Search never decompresses the database. For a query q, precompute a
lookup table LUT[m][c] = ||q_m - codebook[m][c]||^2 (M x 256 floats, one
small pairwise-distance call). Then the distance to any encoded vector is
just M table lookups + adds:

    dist(q, x) ~ sum_m LUT[m][code_x[m]]

This is "asymmetric" because q stays exact while x is quantized -- more
accurate than quantizing both sides.
"""

from __future__ import annotations

import numpy as np

from ..kmeans import kmeans
from ..metrics import pairwise_l2_sq
from .base import BaseQuantizer


class ProductQuantizer(BaseQuantizer):
    """PQ with M subspaces and ks centroids per subspace (default 256 = 1 byte).

    Args:
        dim: full vector dimensionality; must be divisible by n_subspaces.
        n_subspaces: M. Bytes per vector (at ks=256). Higher M = better
            recall, more memory.
        n_centroids: ks, codebook size per subspace. 256 keeps codes in a
            uint8 and is the standard choice.
    """

    def __init__(self, dim: int, n_subspaces: int = 8, n_centroids: int = 256):
        if dim % n_subspaces != 0:
            raise ValueError(
                f"dim={dim} must be divisible by n_subspaces={n_subspaces}"
            )
        if n_centroids > 256:
            raise ValueError("n_centroids > 256 would not fit in uint8 codes")
        self.dim = dim
        self.M = n_subspaces
        self.ks = n_centroids
        self.dsub = dim // n_subspaces
        # codebooks[m] has shape (ks, dsub)
        self.codebooks: np.ndarray | None = None  # (M, ks, dsub)

    def train(self, data: np.ndarray, n_iters: int = 25, seed: int = 0) -> "ProductQuantizer":
        """Run an independent k-means per subspace to learn the codebooks."""
        data = np.asarray(data, dtype=np.float32)
        self.codebooks = np.stack(
            [
                kmeans(self._sub(data, m), self.ks, n_iters=n_iters, seed=seed + m)
                for m in range(self.M)
            ]
        )
        self.is_trained = True
        return self

    def _sub(self, data: np.ndarray, m: int) -> np.ndarray:
        """Slice out subspace m: columns [m*dsub, (m+1)*dsub)."""
        return data[:, m * self.dsub : (m + 1) * self.dsub]

    def encode(self, data: np.ndarray) -> np.ndarray:
        """(n, d) float32 -> (n, M) uint8: nearest codebook entry per subspace."""
        self._check_trained()
        data = np.asarray(data, dtype=np.float32)
        codes = np.empty((data.shape[0], self.M), dtype=np.uint8)
        for m in range(self.M):
            dists = pairwise_l2_sq(self._sub(data, m), self.codebooks[m])
            codes[:, m] = np.argmin(dists, axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """(n, M) uint8 -> (n, d): concatenate the selected centroids."""
        self._check_trained()
        n = codes.shape[0]
        out = np.empty((n, self.dim), dtype=np.float32)
        for m in range(self.M):
            out[:, m * self.dsub : (m + 1) * self.dsub] = self.codebooks[m][codes[:, m]]
        return out

    # ---------------------------------------------------------------- search
    def compute_lut(self, queries: np.ndarray) -> np.ndarray:
        """ADC lookup tables: (nq, M, ks) squared distances to every centroid.

        Cost is nq * M * ks * dsub multiply-adds -- independent of database
        size. This one small computation replaces all query-side float math
        during the scan.
        """
        self._check_trained()
        queries = np.asarray(queries, dtype=np.float32)
        nq = queries.shape[0]
        lut = np.empty((nq, self.M, self.ks), dtype=np.float32)
        for m in range(self.M):
            lut[:, m, :] = pairwise_l2_sq(self._sub(queries, m), self.codebooks[m])
        return lut

    def adc_distances(self, lut: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Scan encoded database with the LUT: (nq, n) approximate sq-distances.

        For each query row, gathers LUT[m][codes[:, m]] for all m and sums.
        The gather `lut_q[np.arange(M), codes]` broadcasts codes (n, M)
        against the table (M, ks) -> (n, M), then reduces over M.
        """
        nq = lut.shape[0]
        n = codes.shape[0]
        dists = np.empty((nq, n), dtype=np.float32)
        m_idx = np.arange(self.M)
        for qi in range(nq):
            dists[qi] = lut[qi][m_idx, codes].sum(axis=1)
        return dists

    @property
    def bytes_per_vector(self) -> float:
        return float(self.M)  # one uint8 id per subspace

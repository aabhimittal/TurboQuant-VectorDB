"""IVFPQ: Inverted File index + Product Quantization on residuals.

This is the architecture behind FAISS's `IndexIVFPQ` and, in various
guises, most managed vector databases. It combines two orthogonal wins:

1. IVF (inverted file) attacks *search time*: coarse k-means partitions
   the space into `nlist` Voronoi cells; each database vector is filed
   under its nearest cell. A query only scans the `nprobe` closest cells,
   so the scan touches roughly n * nprobe / nlist vectors instead of n.

2. PQ attacks *memory*: vectors inside the lists are stored as M-byte PQ
   codes, not floats.

The crucial refinement is *residual encoding*: instead of PQ-encoding the
raw vector x, we encode r = x - centroid(cell(x)). Residuals are centered
near zero and occupy a much smaller, denser region than the raw data, so
the same 256-entry codebooks cover them with far less error. At search
time the query is likewise shifted per-probed-cell (q - centroid) before
building the ADC lookup table.

Storage layout: PQ codes live in ONE flat array indexed by insertion id;
the inverted lists hold only ids. This costs one gather per scanned cell
but lets other components (TurboIndex's re-ranking cascade) address any
vector's code by id without duplicating it.
"""

from __future__ import annotations

import numpy as np

from ..kmeans import kmeans
from ..metrics import pairwise_l2_sq
from ..quantizers.product import ProductQuantizer


class IVFPQIndex:
    def __init__(
        self,
        dim: int,
        n_lists: int = 256,
        n_subspaces: int = 8,
        n_centroids: int = 256,
    ):
        self.dim = dim
        self.nlist = n_lists
        self.pq = ProductQuantizer(dim, n_subspaces, n_centroids)
        self.coarse_centroids: np.ndarray | None = None  # (nlist, dim)
        self.codes: np.ndarray | None = None  # (ntotal, M) uint8, by id
        self.cells: np.ndarray | None = None  # (ntotal,) int32, cell of each id
        self.list_ids: list[np.ndarray] = []  # ids filed under each cell
        self.ntotal = 0
        self.is_trained = False

    # ----------------------------------------------------------------- train
    def train(self, data: np.ndarray, n_iters: int = 25, seed: int = 0) -> "IVFPQIndex":
        data = np.asarray(data, dtype=np.float32)
        # Stage 1: coarse partition of the raw space.
        self.coarse_centroids = kmeans(data, self.nlist, n_iters=n_iters, seed=seed)
        # Stage 2: PQ is trained on *residuals*, the distribution it will
        # actually encode, not on raw vectors.
        assign = np.argmin(pairwise_l2_sq(data, self.coarse_centroids), axis=1)
        residuals = data - self.coarse_centroids[assign]
        self.pq.train(residuals, n_iters=n_iters, seed=seed)
        self.codes = np.empty((0, self.pq.M), dtype=np.uint8)
        self.cells = np.empty(0, dtype=np.int32)
        self.list_ids = [np.empty(0, dtype=np.int64) for _ in range(self.nlist)]
        self.is_trained = True
        return self

    # ------------------------------------------------------------------- add
    def add(self, vectors: np.ndarray) -> None:
        if not self.is_trained:
            raise RuntimeError("IVFPQIndex must be trained before add()")
        vectors = np.asarray(vectors, dtype=np.float32)
        n = vectors.shape[0]
        ids = np.arange(self.ntotal, self.ntotal + n, dtype=np.int64)
        assign = np.argmin(pairwise_l2_sq(vectors, self.coarse_centroids), axis=1)
        codes = self.pq.encode(vectors - self.coarse_centroids[assign])
        self.codes = np.vstack([self.codes, codes])
        self.cells = np.concatenate([self.cells, assign.astype(np.int32)])
        # File ids under their cells with one argsort instead of n appends.
        order = np.argsort(assign, kind="stable")
        boundaries = np.searchsorted(assign[order], np.arange(self.nlist + 1))
        for cell in range(self.nlist):
            lo, hi = boundaries[cell], boundaries[cell + 1]
            if lo < hi:
                self.list_ids[cell] = np.concatenate(
                    [self.list_ids[cell], ids[order[lo:hi]]]
                )
        self.ntotal += n

    # ---------------------------------------------------------------- search
    def search(
        self, queries: np.ndarray, k: int, n_probe: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scan the n_probe nearest cells per query with residual-ADC.

        Returns (ids, sq_distances) of shape (nq, k); rows are padded with
        id -1 / dist +inf if fewer than k candidates were found (tiny
        nprobe on tiny datasets).
        """
        if not self.is_trained:
            raise RuntimeError("IVFPQIndex must be trained before search()")
        queries = np.asarray(queries, dtype=np.float32)
        nq = queries.shape[0]
        n_probe = min(n_probe, self.nlist)
        # Rank cells once for all queries: (nq, nlist) coarse distances.
        coarse = pairwise_l2_sq(queries, self.coarse_centroids)
        probe_cells = np.argpartition(coarse, n_probe - 1, axis=1)[:, :n_probe]

        out_ids = np.full((nq, k), -1, dtype=np.int64)
        out_dists = np.full((nq, k), np.inf, dtype=np.float32)
        for qi in range(nq):
            cand_ids, cand_dists = self._scan_cells(queries[qi], probe_cells[qi])
            if len(cand_ids) == 0:
                continue
            kk = min(k, len(cand_ids))
            sel = np.argpartition(cand_dists, kk - 1)[:kk]
            order = np.argsort(cand_dists[sel])
            out_ids[qi, :kk] = cand_ids[sel][order]
            out_dists[qi, :kk] = cand_dists[sel][order]
        return out_ids, out_dists

    def _scan_cells(
        self, query: np.ndarray, cells: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ADC over every code stored in `cells` for one query.

        The query residual differs per cell (q - centroid_cell), so the LUT
        is built per (query, cell) pair -- this is exactly why IVFPQ keeps
        nprobe small: LUT cost is nprobe * M * ks * dsub, still independent
        of database size.
        """
        id_chunks: list[np.ndarray] = []
        dist_chunks: list[np.ndarray] = []
        for cell in cells:
            ids = self.list_ids[cell]
            if ids.shape[0] == 0:
                continue
            residual_q = (query - self.coarse_centroids[cell])[None, :]
            lut = self.pq.compute_lut(residual_q)  # (1, M, ks)
            dist_chunks.append(self.pq.adc_distances(lut, self.codes[ids])[0])
            id_chunks.append(ids)
        if not id_chunks:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.concatenate(id_chunks), np.concatenate(dist_chunks)

    def reconstruct(self, ids: np.ndarray) -> np.ndarray:
        """Approximate vectors for the given ids: centroid + decoded residual."""
        return self.coarse_centroids[self.cells[ids]] + self.pq.decode(self.codes[ids])

    @property
    def memory_bytes(self) -> int:
        """Code storage only -- the quantity the compression ratio measures.

        The id lists and cell array are bookkeeping shared by every IVF
        implementation and excluded here, matching how FAISS reports code_size.
        """
        return 0 if self.codes is None else self.codes.nbytes

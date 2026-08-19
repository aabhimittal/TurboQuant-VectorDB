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

Beyond the textbook index, this implementation carries the machinery a
real deployment needs and most from-scratch versions omit:

* **Deletion and update** via tombstones, with `compact()` to reclaim the
  space. Ids stay stable until you compact, which is what lets external
  systems hold references to them.
* **Filtered search** against a boolean mask, for the ubiquitous "nearest
  neighbors *where tenant_id = 7*" query.
* **Probe escalation**, which keeps filtered search honest -- see
  `search` for why a selective filter otherwise collapses recall
  silently.
"""

from __future__ import annotations

import numpy as np

from ..kmeans import kmeans
from ..metrics import pairwise_l2_sq
from ..quantizers.product import ProductQuantizer
from ..validation import as_2d_float32, check_dim, check_ids, check_k


class IVFPQIndex:
    def __init__(
        self,
        dim: int,
        n_lists: int = 256,
        n_subspaces: int = 8,
        n_centroids: int = 256,
    ):
        self.dim = check_dim(dim)
        if int(n_lists) < 1:
            raise ValueError(f"n_lists must be >= 1, got {n_lists}")
        self.nlist = int(n_lists)
        self.pq = ProductQuantizer(self.dim, n_subspaces, n_centroids)
        self.coarse_centroids: np.ndarray | None = None  # (nlist, dim)
        self.codes: np.ndarray | None = None  # (n_slots, M) uint8, by id
        self.cells: np.ndarray | None = None  # (n_slots,) int32, cell of each id
        self.list_ids: list[np.ndarray] = []  # ids filed under each cell
        # Tombstones. `n_slots` is how many ids have ever been handed out;
        # `ntotal` is how many are still live. Ids are never reused while
        # tombstoned, so any id an application stored stays valid (or stays
        # deleted) until an explicit compact().
        self.n_slots = 0
        self.deleted: np.ndarray | None = None  # (n_slots,) bool
        self.is_trained = False

    # ----------------------------------------------------------------- train
    def train(self, data: np.ndarray, n_iters: int = 25, seed: int = 0) -> "IVFPQIndex":
        data = as_2d_float32(data, name="training data", dim=self.dim)
        if data.shape[0] < self.nlist:
            raise ValueError(
                f"training set has {data.shape[0]} vectors but n_lists="
                f"{self.nlist} coarse centroids were requested; supply more "
                "training data or reduce n_lists"
            )
        # Stage 1: coarse partition of the raw space.
        self.coarse_centroids = kmeans(data, self.nlist, n_iters=n_iters, seed=seed)
        # Stage 2: PQ is trained on *residuals*, the distribution it will
        # actually encode, not on raw vectors.
        assign = np.argmin(pairwise_l2_sq(data, self.coarse_centroids), axis=1)
        residuals = data - self.coarse_centroids[assign]
        self.pq.train(residuals, n_iters=n_iters, seed=seed)
        self.codes = np.empty((0, self.pq.M), dtype=np.uint8)
        self.cells = np.empty(0, dtype=np.int32)
        self.deleted = np.empty(0, dtype=bool)
        self.list_ids = [np.empty(0, dtype=np.int64) for _ in range(self.nlist)]
        self.n_slots = 0
        self.is_trained = True
        return self

    # ------------------------------------------------------------------- add
    def add(self, vectors: np.ndarray) -> None:
        self._check_trained()
        vectors = as_2d_float32(
            vectors, name="vectors", dim=self.dim, allow_empty=True
        )
        n = vectors.shape[0]
        if n == 0:
            return
        ids = np.arange(self.n_slots, self.n_slots + n, dtype=np.int64)
        assign = np.argmin(pairwise_l2_sq(vectors, self.coarse_centroids), axis=1)
        codes = self.pq.encode(vectors - self.coarse_centroids[assign])
        self.codes = np.vstack([self.codes, codes])
        self.cells = np.concatenate([self.cells, assign.astype(np.int32)])
        self.deleted = np.concatenate([self.deleted, np.zeros(n, dtype=bool)])
        self._file_ids(ids, assign)
        self.n_slots += n

    def _file_ids(self, ids: np.ndarray, assign: np.ndarray) -> None:
        """Append ids to their cells' inverted lists with one argsort."""
        order = np.argsort(assign, kind="stable")
        boundaries = np.searchsorted(assign[order], np.arange(self.nlist + 1))
        for cell in range(self.nlist):
            lo, hi = boundaries[cell], boundaries[cell + 1]
            if lo < hi:
                self.list_ids[cell] = np.concatenate(
                    [self.list_ids[cell], ids[order[lo:hi]]]
                )

    # -------------------------------------------------------- delete / update
    def remove_ids(self, ids: np.ndarray) -> int:
        """Tombstone the given ids. Returns how many were newly removed.

        Tombstoning rather than physically deleting keeps every other id
        stable, which matters because ids are the handle applications
        store in their own database. The codes stay resident until
        `compact()`, so this is O(len(ids)) and safe to call on a live
        index; `search` skips tombstoned ids.
        """
        self._check_trained()
        ids = check_ids(ids, self.n_slots)
        if ids.size == 0:
            return 0
        newly = int((~self.deleted[ids]).sum())
        self.deleted[ids] = True
        return newly

    def update(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Replace the vectors behind existing ids, in place.

        Re-encoding is not enough on its own: a changed vector may belong
        to a different coarse cell, and leaving it filed under the old one
        would make it invisible to queries that probe its true
        neighborhood. So the id is unfiled from its old list and refiled
        under the new one. Updating a tombstoned id revives it.
        """
        self._check_trained()
        ids = check_ids(ids, self.n_slots)
        vectors = as_2d_float32(vectors, name="vectors", dim=self.dim)
        if ids.shape[0] != vectors.shape[0]:
            raise ValueError(
                f"got {ids.shape[0]} ids but {vectors.shape[0]} vectors"
            )
        if ids.size == 0:
            return
        if np.unique(ids).size != ids.size:
            raise ValueError("ids passed to update() must be unique")

        new_cells = np.argmin(
            pairwise_l2_sq(vectors, self.coarse_centroids), axis=1
        ).astype(np.int32)
        self.codes[ids] = self.pq.encode(
            vectors - self.coarse_centroids[new_cells]
        )

        # Unfile only from the cells that actually change, and do it one
        # cell at a time so each list is rewritten at most once.
        changed = self.cells[ids] != new_cells
        moved = ids[changed]
        if moved.size:
            for cell in np.unique(self.cells[moved]):
                leaving = moved[self.cells[moved] == cell]
                keep = ~np.isin(self.list_ids[cell], leaving)
                self.list_ids[cell] = self.list_ids[cell][keep]
            destinations = new_cells[changed]
            self._file_ids(moved, destinations)
            # Re-sort the lists we appended to. `add` produces ascending
            # ids naturally, so without this an updated index would hold
            # the same vectors in a different list order than a rebuilt
            # one -- identical distances, but ties broken differently.
            # Keeping lists ordered makes index state a function of
            # contents alone, not of mutation history, which is what makes
            # "rebuild and compare" a usable debugging tool.
            for cell in np.unique(destinations):
                self.list_ids[cell] = np.sort(self.list_ids[cell])
        self.cells[ids] = new_cells
        self.deleted[ids] = False

    def compact(self) -> np.ndarray:
        """Physically drop tombstoned vectors and renumber the survivors.

        Returns an (n_slots,) int64 map from old id to new id, with -1 for
        ids that were deleted. Callers holding parallel per-id arrays
        (TurboIndex's tier-2 codes, an application's metadata table) must
        apply this map -- which is exactly why it is returned rather than
        being a silent internal detail.
        """
        self._check_trained()
        keep = ~self.deleted
        mapping = np.full(self.n_slots, -1, dtype=np.int64)
        mapping[keep] = np.arange(int(keep.sum()), dtype=np.int64)

        self.codes = np.ascontiguousarray(self.codes[keep])
        new_cells = np.ascontiguousarray(self.cells[keep])
        self.cells = new_cells
        self.deleted = np.zeros(new_cells.shape[0], dtype=bool)
        self.n_slots = int(new_cells.shape[0])
        self.list_ids = [np.empty(0, dtype=np.int64) for _ in range(self.nlist)]
        if self.n_slots:
            self._file_ids(np.arange(self.n_slots, dtype=np.int64), new_cells)
        return mapping

    # ---------------------------------------------------------------- search
    def search(
        self,
        queries: np.ndarray,
        k: int,
        n_probe: int = 8,
        filter_mask: np.ndarray | None = None,
        max_probe: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scan the n_probe nearest cells per query with residual-ADC.

        Args:
            queries: (nq, d) float array.
            k: neighbors to return.
            n_probe: cells to scan initially.
            filter_mask: optional (n_slots,) boolean array; only ids where
                the mask is True are eligible. This is the "nearest
                neighbors WHERE ..." query that every production vector
                store has to answer.
            max_probe: ceiling for probe escalation (see below). Defaults
                to n_probe, i.e. escalation off. Pass `nlist` to let a
                query widen as far as it needs.

        **Probe escalation.** Filtering *after* the ANN scan is the easy
        implementation and it fails in a specific, silent way: if the
        filter keeps 1% of the corpus, a 16-cell scan yielding 600
        candidates leaves ~6 survivors, so a k=10 query quietly returns 6
        results -- or 10 bad ones -- and reports no error. Recall
        collapses exactly when the filter is most selective, which is
        precisely when users notice.

        Escalation fixes this by making the scan width respond to what
        survives instead of being fixed in advance: probe, count the
        survivors, and if there are fewer than k, double the number of
        cells and scan only the newly added ones. Work is proportional to
        how selective the filter turns out to be, and no cell is ever
        scanned twice.

        Returns:
            (ids, sq_distances) of shape (nq, k); rows are padded with id
            -1 / distance +inf when fewer than k candidates exist.
        """
        self._check_trained()
        k = check_k(k)
        queries = as_2d_float32(queries, name="queries", dim=self.dim)
        n_probe = max(1, min(int(n_probe), self.nlist))
        max_probe = n_probe if max_probe is None else min(int(max_probe), self.nlist)
        max_probe = max(max_probe, n_probe)
        eligible = self._eligibility(filter_mask)

        nq = queries.shape[0]
        # Full ranking, not argpartition: escalation needs cells in order
        # beyond the initial n_probe. nlist is small (hundreds), so the
        # sort is negligible next to the ADC scan it feeds.
        coarse = pairwise_l2_sq(queries, self.coarse_centroids)
        ranked = np.argsort(coarse, axis=1)

        out_ids = np.full((nq, k), -1, dtype=np.int64)
        out_dists = np.full((nq, k), np.inf, dtype=np.float32)
        for qi in range(nq):
            cand_ids, cand_dists = self._probe_until_enough(
                queries[qi], ranked[qi], k, n_probe, max_probe, eligible
            )
            if cand_ids.size == 0:
                continue
            kk = min(k, cand_ids.shape[0])
            sel = np.argpartition(cand_dists, kk - 1)[:kk]
            order = np.argsort(cand_dists[sel])
            out_ids[qi, :kk] = cand_ids[sel][order]
            out_dists[qi, :kk] = cand_dists[sel][order]
        return out_ids, out_dists

    def _probe_until_enough(
        self,
        query: np.ndarray,
        ranked_cells: np.ndarray,
        k: int,
        n_probe: int,
        max_probe: int,
        eligible: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scan cells in distance order, widening until k candidates survive."""
        id_chunks: list[np.ndarray] = []
        dist_chunks: list[np.ndarray] = []
        kept = 0
        scanned, target = 0, n_probe
        while True:
            ids, dists = self._scan_cells(query, ranked_cells[scanned:target])
            if eligible is not None and ids.size:
                keep = eligible[ids]
                ids, dists = ids[keep], dists[keep]
            if ids.size:
                id_chunks.append(ids)
                dist_chunks.append(dists)
                kept += ids.shape[0]
            scanned = target
            if kept >= k or scanned >= max_probe:
                break
            target = min(max_probe, max(target * 2, target + 1))
        if not id_chunks:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.concatenate(id_chunks), np.concatenate(dist_chunks)

    def _eligibility(self, filter_mask: np.ndarray | None) -> np.ndarray | None:
        """Combine tombstones and the caller's filter into one lookup array.

        Returns None when everything is eligible, which lets the scan skip
        the mask gather entirely on the common unfiltered path.
        """
        if filter_mask is None:
            if self.deleted is None or not self.deleted.any():
                return None
            return ~self.deleted
        mask = np.asarray(filter_mask)
        if mask.dtype != bool:
            raise TypeError(
                f"filter_mask must be a boolean array, got dtype {mask.dtype!r}"
            )
        if mask.shape != (self.n_slots,):
            raise ValueError(
                f"filter_mask must have shape ({self.n_slots},) -- one entry "
                f"per indexed id -- got {mask.shape}"
            )
        return mask & ~self.deleted

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
        self._check_trained()
        return self.coarse_centroids[self.cells[ids]] + self.pq.decode(self.codes[ids])

    # ------------------------------------------------------------ bookkeeping
    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("IVFPQIndex must be trained before use; call .train()")

    @property
    def ntotal(self) -> int:
        """Live vectors, excluding tombstones."""
        if self.deleted is None:
            return 0
        return int(self.n_slots - self.deleted.sum())

    @property
    def n_deleted(self) -> int:
        return 0 if self.deleted is None else int(self.deleted.sum())

    @property
    def memory_bytes(self) -> int:
        """Code storage only -- the quantity the compression ratio measures.

        Counts tombstoned rows too, because they really are still resident;
        `compact()` is what makes that memory go away. The id lists and
        cell array are bookkeeping shared by every IVF implementation and
        excluded here, matching how FAISS reports code_size.
        """
        return 0 if self.codes is None else self.codes.nbytes

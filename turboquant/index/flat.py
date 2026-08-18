"""Flat (brute-force) indexes.

`FlatIndex` is the exact-search ground truth every approximate method is
measured against. `QuantizedFlatIndex` runs the same exhaustive scan but
over quantized codes -- it isolates the recall cost of *compression alone*,
with no partitioning error mixed in.
"""

from __future__ import annotations

import numpy as np

from ..metrics import pairwise_l2_sq, top_k
from ..quantizers.base import BaseQuantizer
from ..validation import as_2d_float32, check_dim, check_k


class FlatIndex:
    """Exact exhaustive search over raw float32 vectors."""

    def __init__(self, dim: int):
        self.dim = check_dim(dim)
        self.vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = as_2d_float32(
            vectors, name="vectors", dim=self.dim, allow_empty=True
        )
        if vectors.shape[0] == 0:
            return
        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ids, sq_distances), each (nq, min(k, ntotal)), ascending.

        Asking for more neighbors than the index holds returns everything it
        holds rather than raising: `k` is a request, and a 3-vector index
        answering `k=10` with 3 results is the useful behavior.
        """
        k = check_k(k)
        queries = as_2d_float32(queries, name="queries", dim=self.dim)
        if self.ntotal == 0:
            raise RuntimeError("FlatIndex is empty; call add() before search()")
        dists = pairwise_l2_sq(queries, self.vectors)
        return top_k(dists, k)

    @property
    def ntotal(self) -> int:
        return 0 if self.vectors is None else int(self.vectors.shape[0])

    @property
    def memory_bytes(self) -> int:
        return 0 if self.vectors is None else self.vectors.nbytes


class QuantizedFlatIndex:
    """Exhaustive scan over quantized codes.

    With a ProductQuantizer the scan uses ADC lookup tables (no
    decompression). With any other quantizer it decodes in blocks and
    computes distances against the reconstructions -- same recall, more
    compute, kept simple on purpose.
    """

    def __init__(self, quantizer: BaseQuantizer):
        if not isinstance(quantizer, BaseQuantizer):
            raise TypeError(
                "quantizer must be a BaseQuantizer instance, got "
                f"{type(quantizer).__name__}"
            )
        self.quantizer = quantizer
        self.dim = quantizer.dim
        self.codes: np.ndarray | None = None
        self.ntotal = 0

    def add(self, vectors: np.ndarray) -> None:
        vectors = as_2d_float32(
            vectors, name="vectors", dim=self.dim, allow_empty=True
        )
        if vectors.shape[0] == 0:
            return
        codes = self.quantizer.encode(vectors)
        if self.codes is None:
            self.codes = codes
        else:
            self.codes = np.vstack([self.codes, codes])
        self.ntotal += vectors.shape[0]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = check_k(k)
        queries = as_2d_float32(queries, name="queries", dim=self.dim)
        if self.ntotal == 0:
            raise RuntimeError("QuantizedFlatIndex is empty; call add() first")
        # Duck-typed rather than `isinstance(ProductQuantizer)`: OPQ wraps a
        # PQ instead of subclassing it, and any future codec that can build
        # a lookup table should get the fast path too.
        if hasattr(self.quantizer, "compute_lut"):
            lut = self.quantizer.compute_lut(queries)
            dists = self.quantizer.adc_distances(lut, self.codes)
        else:
            dists = pairwise_l2_sq(queries, self.quantizer.decode(self.codes))
        return top_k(dists, k)

    @property
    def memory_bytes(self) -> int:
        return 0 if self.codes is None else self.codes.nbytes

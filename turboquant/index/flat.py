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
from ..quantizers.product import ProductQuantizer


class FlatIndex:
    """Exact exhaustive search over raw float32 vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ids, sq_distances), each (nq, k), ascending by distance."""
        dists = pairwise_l2_sq(np.asarray(queries, dtype=np.float32), self.vectors)
        return top_k(dists, k)

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
        self.quantizer = quantizer
        self.codes: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        codes = self.quantizer.encode(vectors)
        if self.codes is None:
            self.codes = codes
        else:
            self.codes = np.vstack([self.codes, codes])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if isinstance(self.quantizer, ProductQuantizer):
            lut = self.quantizer.compute_lut(queries)
            dists = self.quantizer.adc_distances(lut, self.codes)
        else:
            dists = pairwise_l2_sq(queries, self.quantizer.decode(self.codes))
        return top_k(dists, k)

    @property
    def memory_bytes(self) -> int:
        return 0 if self.codes is None else self.codes.nbytes

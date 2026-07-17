"""AdaptiveBitQuantizer: variance-driven per-dimension bit allocation.

TurboQuant's novel twist on scalar quantization. Classical SQ spends the
same number of bits on every dimension, but embedding dimensions are far
from equal: after training, some dimensions carry 10-100x the variance of
others. Uniform allocation wastes precision on near-constant dimensions
and starves the informative ones.

This quantizer borrows the *bit allocation* idea from transform coding
(the math behind JPEG/MP3 rate allocation) and applies it to vector
search: given a total bit budget B = avg_bits * dim, greedily hand out
bits one at a time to whichever dimension currently has the largest
expected quantization error. For a uniform quantizer with b bits over a
range of width `span`, the expected squared error is

    MSE(b) = span^2 / (12 * 4^b)

so each extra bit cuts a dimension's error by 4x, and the greedy
allocation is provably optimal for this convex cost (it is the discrete
water-filling solution).

Dimensions can receive 0 bits: they are dropped from storage entirely and
reconstructed as their training mean. High-variance dimensions can receive
up to 8 bits. The resulting variable-width codes are genuinely bit-packed,
so a 4-bit average budget really costs 0.5 bytes/dim.
"""

from __future__ import annotations

import numpy as np

from .base import BaseQuantizer

_MAX_BITS = 8


class AdaptiveBitQuantizer(BaseQuantizer):
    def __init__(self, dim: int, avg_bits: float = 4.0):
        if not 0 < avg_bits <= _MAX_BITS:
            raise ValueError(f"avg_bits must be in (0, {_MAX_BITS}]")
        self.dim = dim
        self.total_bits = int(round(avg_bits * dim))
        self.bits_per_dim: np.ndarray | None = None
        self.mins: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.means: np.ndarray | None = None  # reconstruction for 0-bit dims
        # Layout metadata for packing: dims sorted by bit width so each
        # width group occupies one contiguous block of the bitstream.
        self._dim_order: np.ndarray | None = None
        self._groups: list[tuple[int, np.ndarray, int]] = []  # (bits, dims, bit_offset)

    # ------------------------------------------------------------------ train
    def train(self, data: np.ndarray) -> "AdaptiveBitQuantizer":
        data = np.asarray(data, dtype=np.float32)
        lo = np.percentile(data, 0.1, axis=0).astype(np.float32)
        hi = np.percentile(data, 99.9, axis=0).astype(np.float32)
        span = np.maximum(hi - lo, 1e-12)
        self.mins = lo
        self.means = data.mean(axis=0)

        self.bits_per_dim = self._allocate_bits(span)

        # Per-dim step size given its allocated levels; dims with 0 bits
        # keep scale 1 (unused) to avoid divide-by-zero in encode.
        levels = (1 << self.bits_per_dim) - 1
        self.scales = np.where(
            levels > 0, span / np.maximum(levels, 1), 1.0
        ).astype(np.float32)

        self._build_layout()
        self.is_trained = True
        return self

    def _allocate_bits(self, span: np.ndarray) -> np.ndarray:
        """Greedy discrete water-filling over the per-dimension MSE.

        Loop invariant: err[d] is the expected squared quantization error of
        dimension d under its current bit count. Each of the `total_bits`
        rounds gives one bit to the worst dimension, dividing its error by 4.
        Dimensions capped at 8 bits get their error forced to -inf so they
        are never picked again.
        """
        bits = np.zeros(self.dim, dtype=np.int64)
        err = (span.astype(np.float64) ** 2) / 12.0  # MSE at 0 bits
        for _ in range(self.total_bits):
            d = int(np.argmax(err))
            if err[d] == -np.inf:
                break  # every dimension is saturated at _MAX_BITS
            bits[d] += 1
            err[d] = err[d] / 4.0 if bits[d] < _MAX_BITS else -np.inf
        return bits

    def _build_layout(self) -> None:
        """Sort dimensions by bit width so packing is vectorized per group.

        The bitstream layout is: all 1-bit dims' codes, then all 2-bit
        dims', ... then all 8-bit dims'. Within a group every code has the
        same width, so a whole group packs/unpacks with one reshape.
        """
        self._groups = []
        offset = 0
        for b in range(1, _MAX_BITS + 1):
            dims = np.where(self.bits_per_dim == b)[0]
            if len(dims):
                self._groups.append((b, dims, offset))
                offset += b * len(dims)
        self._stream_bits = offset

    # ----------------------------------------------------------- encode/decode
    def encode(self, data: np.ndarray) -> np.ndarray:
        self._check_trained()
        data = np.asarray(data, dtype=np.float32)
        n = data.shape[0]
        bit_matrix = np.zeros((n, self._stream_bits), dtype=np.uint8)
        for b, dims, offset in self._groups:
            levels = (1 << b) - 1
            q = np.rint((data[:, dims] - self.mins[dims]) / self.scales[dims])
            codes = np.clip(q, 0, levels).astype(np.uint8)  # (n, nd)
            shifts = np.arange(b, dtype=np.uint8)
            planes = (codes[:, :, None] >> shifts) & 1  # (n, nd, b), LSB first
            width = b * len(dims)
            bit_matrix[:, offset : offset + width] = planes.reshape(n, width)
        return np.packbits(bit_matrix, axis=1, bitorder="little")

    def decode(self, packed: np.ndarray) -> np.ndarray:
        self._check_trained()
        n = packed.shape[0]
        flat = np.unpackbits(packed, axis=1, bitorder="little")[:, : self._stream_bits]
        # 0-bit dims fall back to the training mean; quantized dims overwrite.
        out = np.tile(self.means, (n, 1))
        for b, dims, offset in self._groups:
            width = b * len(dims)
            planes = flat[:, offset : offset + width].reshape(n, len(dims), b)
            weights = (1 << np.arange(b)).astype(np.uint16)
            codes = (planes.astype(np.uint16) * weights).sum(axis=2)
            out[:, dims] = self.mins[dims] + codes.astype(np.float32) * self.scales[dims]
        return out

    @property
    def bytes_per_vector(self) -> float:
        self._check_trained()
        return float(np.ceil(self._stream_bits / 8))

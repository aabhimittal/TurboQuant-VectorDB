"""Scalar Quantization (SQ): per-dimension uniform quantization.

The simplest compression in the family. Each dimension d gets a learned
range [min_d, max_d]; a float value is snapped to one of 2^bits evenly
spaced levels inside that range and stored as a small integer.

    float32 (4 bytes/dim)  ->  8 bits/dim = 4x   |  4 bits/dim = 8x

Why per-dimension ranges instead of one global range: embedding dimensions
have wildly different scales (some carry 10x the variance of others). A
global range would waste most levels on the widest dimension and crush the
narrow ones into a single level.
"""

from __future__ import annotations

import numpy as np

from .base import BaseQuantizer


class ScalarQuantizer(BaseQuantizer):
    """Uniform per-dimension scalar quantizer with `bits` in {1..8}.

    Codes are stored one uint8 per dimension; for bits < 8 the codes are
    bit-packed (see `_pack_codes`) so 4-bit SQ genuinely costs 0.5
    bytes/dim, not 1.
    """

    def __init__(self, dim: int, bits: int = 8):
        if not 1 <= bits <= 8:
            raise ValueError("bits must be in [1, 8]")
        self.dim = dim
        self.bits = bits
        self.levels = (1 << bits) - 1  # max integer code value
        self.mins: np.ndarray | None = None
        self.scales: np.ndarray | None = None

    def train(self, data: np.ndarray) -> "ScalarQuantizer":
        """Learn per-dimension ranges from data.

        Uses the 0.1 / 99.9 percentile instead of raw min/max: a single
        outlier would otherwise stretch the range and starve the bulk of
        the distribution of quantization levels.
        """
        data = np.asarray(data, dtype=np.float32)
        lo = np.percentile(data, 0.1, axis=0).astype(np.float32)
        hi = np.percentile(data, 99.9, axis=0).astype(np.float32)
        self.mins = lo
        # Guard degenerate (constant) dimensions against divide-by-zero.
        span = np.maximum(hi - lo, 1e-12)
        self.scales = (span / self.levels).astype(np.float32)
        self.is_trained = True
        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        """float32 (n, d) -> packed uint8 codes (n, ceil(d*bits/8))."""
        self._check_trained()
        data = np.asarray(data, dtype=np.float32)
        # Affine map to [0, levels], round to nearest level, clip outliers
        # (values beyond the trained percentile range saturate at 0/levels).
        q = np.rint((data - self.mins) / self.scales)
        codes = np.clip(q, 0, self.levels).astype(np.uint8)
        return _pack_codes(codes, self.bits)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Packed codes -> float32 reconstruction (center of each cell)."""
        self._check_trained()
        unpacked = _unpack_codes(codes, self.bits, self.dim)
        return self.mins + unpacked.astype(np.float32) * self.scales

    @property
    def bytes_per_vector(self) -> float:
        return np.ceil(self.dim * self.bits / 8)


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Bit-pack (n, d) integer codes (< 2^bits) into (n, ceil(d*bits/8)) uint8.

    Expands each code into its `bits` binary digits (LSB first), then packs
    the flat bitstream with np.packbits. Fully vectorized; the roundtrip
    with `_unpack_codes` is exact.
    """
    if bits == 8:
        return codes  # already one byte per code, nothing to do
    n, d = codes.shape
    shifts = np.arange(bits, dtype=np.uint8)
    bit_planes = (codes[:, :, None] >> shifts) & 1  # (n, d, bits)
    return np.packbits(bit_planes.reshape(n, d * bits), axis=1, bitorder="little")


def _unpack_codes(packed: np.ndarray, bits: int, dim: int) -> np.ndarray:
    """Inverse of `_pack_codes`: (n, bytes) uint8 -> (n, dim) integer codes."""
    if bits == 8:
        return packed
    n = packed.shape[0]
    flat_bits = np.unpackbits(packed, axis=1, bitorder="little")[:, : dim * bits]
    bit_planes = flat_bits.reshape(n, dim, bits)
    weights = (1 << np.arange(bits)).astype(np.uint8)
    return (bit_planes * weights).sum(axis=2).astype(np.uint8)

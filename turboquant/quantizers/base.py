"""Common interface for all quantizers.

A quantizer maps float32 vectors to compact codes and back:

    train(X)  -> learn parameters (ranges, codebooks, bit budgets)
    encode(X) -> codes        (the compressed representation)
    decode(C) -> float32      (lossy reconstruction)

`bytes_per_vector` is the honest storage cost of one encoded vector and is
what every compression-ratio claim in this repo is computed from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseQuantizer(ABC):
    dim: int
    is_trained: bool = False

    @abstractmethod
    def train(self, data: np.ndarray) -> "BaseQuantizer":
        """Learn quantization parameters from a sample of the data."""

    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Compress float32 vectors into codes."""

    @abstractmethod
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate float32 vectors from codes."""

    @property
    @abstractmethod
    def bytes_per_vector(self) -> float:
        """Storage cost of one encoded vector, in bytes."""

    def compression_ratio(self) -> float:
        """Ratio versus raw float32 storage (4 bytes per dimension)."""
        return (4.0 * self.dim) / self.bytes_per_vector

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                f"{type(self).__name__} must be trained before use; call .train()"
            )

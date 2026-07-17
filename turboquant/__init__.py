"""TurboQuant: vector quantization for ANN search, from scratch in NumPy.

Quantizers (compress vectors):
    ScalarQuantizer      -- uniform per-dimension SQ (SQ8/SQ4/...)
    AdaptiveBitQuantizer -- variance-driven per-dimension bit allocation
    ProductQuantizer     -- PQ with ADC lookup-table search

Indexes (search compressed vectors):
    FlatIndex            -- exact brute force (ground truth)
    QuantizedFlatIndex   -- brute force over codes
    IVFPQIndex           -- inverted lists + residual PQ
    TurboIndex           -- budget-split IVFPQ + adaptive-bit re-rank cascade
"""

from .datasets import clustered_embeddings, train_base_query_split
from .index.flat import FlatIndex, QuantizedFlatIndex
from .index.ivfpq import IVFPQIndex
from .index.turbo import TurboIndex
from .kmeans import kmeans
from .metrics import pairwise_l2_sq, recall_at_k, top_k
from .quantizers.adaptive import AdaptiveBitQuantizer
from .quantizers.base import BaseQuantizer
from .quantizers.product import ProductQuantizer
from .quantizers.scalar import ScalarQuantizer

__version__ = "0.1.0"

__all__ = [
    "AdaptiveBitQuantizer",
    "BaseQuantizer",
    "FlatIndex",
    "IVFPQIndex",
    "ProductQuantizer",
    "QuantizedFlatIndex",
    "ScalarQuantizer",
    "TurboIndex",
    "clustered_embeddings",
    "kmeans",
    "pairwise_l2_sq",
    "recall_at_k",
    "top_k",
    "train_base_query_split",
]

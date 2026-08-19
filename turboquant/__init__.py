"""TurboQuant: vector quantization for ANN search, from scratch in NumPy.

Quantizers (compress vectors):
    ScalarQuantizer      -- uniform per-dimension SQ (SQ8/SQ4/...)
    AdaptiveBitQuantizer -- variance-driven per-dimension bit allocation
    ProductQuantizer     -- PQ with ADC lookup-table search
    OPQProductQuantizer  -- PQ behind a learned rotation (same bytes, more recall)
    AnisotropicPQ        -- score-aware PQ for maximum inner-product search

Indexes (search compressed vectors):
    FlatIndex            -- exact brute force (ground truth)
    QuantizedFlatIndex   -- brute force over codes
    IVFPQIndex           -- inverted lists + residual PQ, with deletes and filters
    TurboIndex           -- budget-split IVFPQ + adaptive-bit re-rank cascade

Persistence:
    save / load          -- pickle-free .npz archives of any of the above
"""

from .datasets import clustered_embeddings, train_base_query_split
from .index.flat import FlatIndex, QuantizedFlatIndex
from .index.ivfpq import IVFPQIndex
from .index.turbo import TurboIndex
from .io import load, save
from .kmeans import kmeans
from .metrics import (
    inner_product,
    normalize,
    pairwise_l2_sq,
    recall_at_k,
    top_k,
    top_k_max,
)
from .quantizers.adaptive import AdaptiveBitQuantizer
from .quantizers.anisotropic import AnisotropicPQ
from .quantizers.base import BaseQuantizer
from .quantizers.opq import OPQProductQuantizer
from .quantizers.product import ProductQuantizer
from .quantizers.scalar import ScalarQuantizer

__version__ = "0.2.0"

__all__ = [
    "AdaptiveBitQuantizer",
    "AnisotropicPQ",
    "BaseQuantizer",
    "FlatIndex",
    "IVFPQIndex",
    "OPQProductQuantizer",
    "ProductQuantizer",
    "QuantizedFlatIndex",
    "ScalarQuantizer",
    "TurboIndex",
    "clustered_embeddings",
    "inner_product",
    "kmeans",
    "load",
    "normalize",
    "pairwise_l2_sq",
    "recall_at_k",
    "save",
    "top_k",
    "top_k_max",
    "train_base_query_split",
]

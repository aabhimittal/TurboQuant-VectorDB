"""Synthetic datasets that mimic the structure of real embedding spaces.

Real embeddings (OpenAI, sentence-transformers, ...) are not isotropic
Gaussian noise: they form clusters (topics), and per-dimension variance is
highly non-uniform. Both properties matter -- clustering is what makes IVF
work, and variance skew is what AdaptiveBitQuantizer exploits -- so the
generator reproduces them. Pure white noise would make every ANN method
look artificially bad and adaptive bits look useless.
"""

from __future__ import annotations

import numpy as np


def clustered_embeddings(
    n: int,
    dim: int,
    n_clusters: int = 64,
    variance_decay: float = 0.985,
    noise: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Gaussian-mixture vectors with geometrically decaying per-dim variance.

    Args:
        n: number of vectors.
        dim: dimensionality.
        n_clusters: mixture components ("topics").
        variance_decay: per-dimension std multiplier; 0.985^128 ~ 0.14, so
            the last dims carry ~2% of the first dims' variance -- similar
            in spirit to the eigenvalue decay of real embedding matrices.
        noise: within-cluster std relative to between-cluster spread.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    scales = variance_decay ** np.arange(dim, dtype=np.float32)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * scales
    labels = rng.integers(n_clusters, size=n)
    points = centers[labels] + (
        rng.standard_normal((n, dim)).astype(np.float32) * scales * noise
    )
    return points.astype(np.float32)


def train_base_query_split(
    n_train: int,
    n_base: int,
    n_query: int,
    dim: int,
    seed: int = 0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint train/base/query sets drawn from the same distribution.

    Training on the base set itself would let quantizers overfit their
    codebooks to the exact vectors they later encode -- real systems train
    on a sample and index a stream, so we keep the sets disjoint.
    """
    total = clustered_embeddings(n_train + n_base + n_query, dim, seed=seed, **kwargs)
    return (
        total[:n_train],
        total[n_train : n_train + n_base],
        total[n_train + n_base :],
    )

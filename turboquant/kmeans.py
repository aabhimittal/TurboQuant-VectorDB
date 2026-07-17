"""From-scratch k-means with k-means++ seeding.

This is the training engine behind both Product Quantization (256-centroid
codebooks per subspace) and IVF (coarse partitioning of the vector space).
Implemented fully vectorized in NumPy: the inner loop is one BLAS matmul
per Lloyd iteration.
"""

from __future__ import annotations

import numpy as np

from .metrics import pairwise_l2_sq


def _kmeans_pp_init(
    data: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ seeding: spread initial centroids apart.

    Each new centroid is sampled with probability proportional to the
    squared distance from the nearest already-chosen centroid. This gives
    an O(log k) approximation guarantee versus arbitrarily bad uniform
    seeding (Arthur & Vassilvitskii, 2007).
    """
    n = data.shape[0]
    centroids = np.empty((k, data.shape[1]), dtype=data.dtype)
    centroids[0] = data[rng.integers(n)]
    # Running minimum of squared distance to the closest chosen centroid.
    closest_sq = pairwise_l2_sq(data, centroids[0:1]).ravel()
    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            # All remaining points coincide with chosen centroids: any pick works.
            centroids[i:] = data[rng.integers(n, size=k - i)]
            break
        probs = closest_sq / total
        centroids[i] = data[rng.choice(n, p=probs)]
        new_sq = pairwise_l2_sq(data, centroids[i : i + 1]).ravel()
        np.minimum(closest_sq, new_sq, out=closest_sq)
    return centroids


def kmeans(
    data: np.ndarray,
    k: int,
    n_iters: int = 25,
    seed: int = 0,
    sample_size: int | None = 100_000,
) -> np.ndarray:
    """Lloyd's algorithm. Returns (k, d) centroids.

    Args:
        data: (n, d) training vectors.
        k: number of centroids.
        n_iters: Lloyd iterations; quantization codebooks converge to
            within noise in ~25 iterations on typical embedding data.
        seed: RNG seed for reproducible codebooks.
        sample_size: cap on training points. k-means cost is O(n*k*d) per
            iteration; a 100k sample estimates centroids nearly as well as
            the full set (FAISS defaults to 256 points per centroid).
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    n = data.shape[0]
    if k > n:
        raise ValueError(f"k={k} exceeds number of training points n={n}")
    rng = np.random.default_rng(seed)
    if sample_size is not None and n > sample_size:
        data = data[rng.choice(n, size=sample_size, replace=False)]
        n = sample_size

    centroids = _kmeans_pp_init(data, k, rng)
    for _ in range(n_iters):
        # Assignment step: nearest centroid for every point (one matmul).
        assign = np.argmin(pairwise_l2_sq(data, centroids), axis=1)
        # Update step: mean of each cluster, computed with bincount so it
        # stays O(n*d) with no Python-level loop over clusters.
        counts = np.bincount(assign, minlength=k).astype(np.float32)
        sums = np.zeros_like(centroids)
        np.add.at(sums, assign, data)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        # Empty-cluster repair: respawn dead centroids on random points,
        # otherwise they stay dead forever and waste codebook capacity.
        n_dead = int((~nonempty).sum())
        if n_dead:
            centroids[~nonempty] = data[rng.choice(n, size=n_dead, replace=False)]
    return centroids

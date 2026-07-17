"""Distance kernels and evaluation metrics.

Everything here is expressed as squared L2 distance. Squared L2 preserves
nearest-neighbor ordering (sqrt is monotonic), so we never pay for the sqrt.
"""

from __future__ import annotations

import numpy as np


def pairwise_l2_sq(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Squared L2 distance between every query and every database vector.

    Uses the expansion ||q - x||^2 = ||q||^2 - 2 q.x + ||x||^2 so the
    dominant cost is a single BLAS matmul instead of an O(nq * n * d)
    broadcasted subtraction that would materialize a huge intermediate.

    Args:
        queries:  (nq, d) float array.
        database: (n, d) float array.

    Returns:
        (nq, n) array where out[i, j] = ||queries[i] - database[j]||^2.
    """
    q_sq = np.einsum("ij,ij->i", queries, queries)[:, None]  # (nq, 1)
    x_sq = np.einsum("ij,ij->i", database, database)[None, :]  # (1, n)
    cross = queries @ database.T  # (nq, n) -- the BLAS-heavy term
    dists = q_sq - 2.0 * cross + x_sq
    # Floating-point cancellation can produce tiny negatives; clamp so
    # downstream sqrt/argsort callers never see -1e-12.
    np.maximum(dists, 0.0, out=dists)
    return dists


def top_k(dists: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the k smallest entries per row.

    np.argpartition is O(n) per row versus O(n log n) for a full sort;
    we only fully sort the k survivors.

    Returns:
        (indices, distances), each of shape (nq, k), sorted ascending.
    """
    k = min(k, dists.shape[1])
    part = np.argpartition(dists, k - 1, axis=1)[:, :k]  # unordered top-k
    part_d = np.take_along_axis(dists, part, axis=1)
    order = np.argsort(part_d, axis=1)  # sort only k elements
    idx = np.take_along_axis(part, order, axis=1)
    return idx, np.take_along_axis(part_d, order, axis=1)


def recall_at_k(approx_ids: np.ndarray, exact_ids: np.ndarray, k: int) -> float:
    """Fraction of true top-k neighbors recovered by the approximate search.

    recall@k = |approx_topk ∩ exact_topk| / k, averaged over queries.
    This is the standard ANN benchmark metric (ann-benchmarks.com).
    """
    hits = 0
    for approx_row, exact_row in zip(approx_ids[:, :k], exact_ids[:, :k]):
        hits += len(set(approx_row.tolist()) & set(exact_row.tolist()))
    return hits / (len(exact_ids) * k)

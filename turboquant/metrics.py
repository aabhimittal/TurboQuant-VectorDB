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


def inner_product(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Inner-product scores: out[i, j] = queries[i] . database[j].

    Maximum-inner-product search (MIPS) is the native metric for
    recommendation and retrieval models whose scores are dot products.
    Unlike L2 it is *not* a distance -- higher is better, the triangle
    inequality does not hold, and a vector's norm can make it beat
    better-aligned rivals. Callers therefore take the k largest.
    """
    return queries @ database.T


def normalize(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale rows to unit L2 norm, for cosine similarity search.

    Cosine search needs no separate code path: on unit vectors,

        ||q - x||^2 = 2 - 2 * cos(q, x)

    so L2 ordering and cosine ordering are identical. Normalizing at
    ingest turns every L2 index in this library into an exact cosine
    index. Zero rows have no direction; they are left at zero rather than
    dividing by zero, which places them equidistant from everything.
    """
    data = np.asarray(data, dtype=np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return data / np.maximum(norms, eps)


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


def top_k_max(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and scores of the k *largest* entries per row (for MIPS).

    Implemented by negating and reusing `top_k` so the two ranking paths
    cannot drift apart in tie-breaking or padding behavior.
    """
    idx, neg = top_k(-scores, k)
    return idx, -neg


def recall_at_k(approx_ids: np.ndarray, exact_ids: np.ndarray, k: int) -> float:
    """Fraction of true top-k neighbors recovered by the approximate search.

    recall@k = |approx_topk ∩ exact_topk| / k, averaged over queries.
    This is the standard ANN benchmark metric (ann-benchmarks.com).
    """
    hits = 0
    for approx_row, exact_row in zip(approx_ids[:, :k], exact_ids[:, :k]):
        hits += len(set(approx_row.tolist()) & set(exact_row.tolist()))
    return hits / (len(exact_ids) * k)

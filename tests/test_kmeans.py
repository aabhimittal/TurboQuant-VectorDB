"""Tests for the k-means engine."""

import numpy as np
import pytest

from turboquant import kmeans, pairwise_l2_sq


def test_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(0)
    true_centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    data = np.vstack(
        [c + rng.standard_normal((200, 2)).astype(np.float32) * 0.3 for c in true_centers]
    )
    centroids = kmeans(data, 4, n_iters=20, seed=1)
    # Every true center must have a learned centroid within noise distance.
    d = pairwise_l2_sq(true_centers, centroids)
    assert (d.min(axis=1) < 0.5).all()


def test_kmeans_k_exceeds_n_raises():
    with pytest.raises(ValueError):
        kmeans(np.zeros((3, 2), dtype=np.float32), 5)


def test_kmeans_deterministic_with_seed():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((500, 8)).astype(np.float32)
    c1 = kmeans(data, 16, seed=42)
    c2 = kmeans(data, 16, seed=42)
    np.testing.assert_array_equal(c1, c2)


def test_kmeans_no_empty_cluster_output():
    """Even with duplicate-heavy data, all k centroids must be finite."""
    data = np.repeat(np.eye(4, dtype=np.float32), 50, axis=0)
    centroids = kmeans(data, 8, n_iters=10, seed=0)
    assert np.isfinite(centroids).all()
    assert centroids.shape == (8, 4)

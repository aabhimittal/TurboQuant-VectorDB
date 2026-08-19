"""OPQ: the rotation must be a real rotation, and must earn its keep."""

import numpy as np
import pytest

from turboquant import (
    FlatIndex,
    OPQProductQuantizer,
    ProductQuantizer,
    QuantizedFlatIndex,
    recall_at_k,
    train_base_query_split,
)

DIM = 64
K = 10


@pytest.fixture(scope="module")
def split():
    return train_base_query_split(4000, 6000, 50, DIM, n_clusters=32, seed=7)


@pytest.fixture(scope="module")
def ground_truth(split):
    _, base, queries = split
    flat = FlatIndex(DIM)
    flat.add(base)
    ids, _ = flat.search(queries, K)
    return ids


@pytest.fixture(scope="module")
def trained(split):
    train, _, _ = split
    return OPQProductQuantizer(DIM, n_subspaces=8, n_rotation_iters=6).train(train)


def test_rotation_is_orthogonal(trained):
    """R^T R = I is what makes rotating the query lossless.

    If this drifts, distances in the rotated space stop equalling
    distances in the original one and every recall number is a lie.
    """
    r = trained.rotation
    assert r.shape == (DIM, DIM)
    assert np.abs(r.T @ r - np.eye(DIM)).max() < 1e-4
    assert abs(abs(np.linalg.det(r.astype(np.float64))) - 1.0) < 1e-3


def test_rotation_preserves_distances(trained, split):
    _, base, queries = split
    direct = ((queries[:5, None, :] - base[None, :20, :]) ** 2).sum(axis=2)
    rq = (queries[:5] - trained.mean) @ trained.rotation
    rb = (base[:20] - trained.mean) @ trained.rotation
    rotated = ((rq[:, None, :] - rb[None, :, :]) ** 2).sum(axis=2)
    assert np.allclose(direct, rotated, rtol=1e-3, atol=1e-3)


def test_opq_beats_plain_pq_at_identical_storage(split, ground_truth, trained):
    """The whole point: same bytes per vector, materially better recall."""
    train, base, queries = split
    pq = ProductQuantizer(DIM, n_subspaces=8).train(train)

    plain = QuantizedFlatIndex(pq)
    plain.add(base)
    opq = QuantizedFlatIndex(trained)
    opq.add(base)

    r_plain = recall_at_k(plain.search(queries, K)[0], ground_truth, K)
    r_opq = recall_at_k(opq.search(queries, K)[0], ground_truth, K)

    assert trained.bytes_per_vector == pq.bytes_per_vector
    assert plain.memory_bytes == opq.memory_bytes
    # Measured ~2.3-2.8x on 128-dim data; a conservative floor here.
    assert r_opq > r_plain * 1.3


def test_eigenvalue_balancing_alone_already_helps(split, ground_truth):
    """Even with zero refinement rounds, the parametric init should win.

    Isolates the two halves of the algorithm: if this fails, the greedy
    variance-balancing deal is broken independently of Procrustes.
    """
    train, base, queries = split
    pq = ProductQuantizer(DIM, n_subspaces=8).train(train)
    init_only = OPQProductQuantizer(DIM, n_subspaces=8, n_rotation_iters=0).train(train)

    a, b = QuantizedFlatIndex(pq), QuantizedFlatIndex(init_only)
    a.add(base)
    b.add(base)
    assert recall_at_k(b.search(queries, K)[0], ground_truth, K) > recall_at_k(
        a.search(queries, K)[0], ground_truth, K
    )


def test_balanced_allocation_equalizes_subspace_variance(split):
    """Check the mechanism, not just the outcome.

    After rotation, the per-subspace total variance should be far more
    even than under a naive contiguous split of the raw dimensions.
    """
    train, _, _ = split
    opq = OPQProductQuantizer(DIM, n_subspaces=8, n_rotation_iters=0).train(train)
    rotated = (train - opq.mean) @ opq.rotation

    def spread(x):
        per_sub = x.var(axis=0).reshape(8, -1).sum(axis=1)
        return per_sub.max() / max(per_sub.min(), 1e-12)

    assert spread(rotated) < spread(train)


def test_adc_matches_decoded_distance(trained, split):
    """The lookup table must agree with actually decoding the vector."""
    _, base, queries = split
    codes = trained.encode(base[:200])
    lut = trained.compute_lut(queries[:4])
    approx = trained.adc_distances(lut, codes)
    exact = ((queries[:4, None, :] - trained.decode(codes)[None, :, :]) ** 2).sum(axis=2)
    assert np.allclose(approx, exact, rtol=1e-2, atol=1e-2)


def test_shared_overhead_is_reported_and_does_not_scale(trained):
    """The rotation is real memory; it just is not per-vector memory."""
    assert trained.shared_bytes == DIM * DIM * 4 + DIM * 4
    assert trained.bytes_per_vector == 8


def test_rotation_sample_cap_is_respected(split):
    """A tiny sample must still train, not crash on the subsample path."""
    train, _, _ = split
    q = OPQProductQuantizer(
        DIM, n_subspaces=8, n_centroids=64, n_rotation_iters=2, rotation_sample=300
    ).train(train)
    assert np.abs(q.rotation.T @ q.rotation - np.eye(DIM)).max() < 1e-4


def test_rejects_bad_construction():
    with pytest.raises(ValueError):
        OPQProductQuantizer(DIM, n_rotation_iters=-1)
    with pytest.raises(ValueError):
        OPQProductQuantizer(DIM, rotation_sample=0)

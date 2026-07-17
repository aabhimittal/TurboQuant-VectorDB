"""Integration tests for indexes: recall against exact ground truth."""

import numpy as np
import pytest

from turboquant import (
    FlatIndex,
    IVFPQIndex,
    ProductQuantizer,
    QuantizedFlatIndex,
    ScalarQuantizer,
    TurboIndex,
    recall_at_k,
    train_base_query_split,
)

DIM = 64
K = 10


@pytest.fixture(scope="module")
def split():
    return train_base_query_split(4000, 8000, 50, DIM, n_clusters=32, seed=7)


@pytest.fixture(scope="module")
def ground_truth(split):
    _, base, queries = split
    flat = FlatIndex(DIM)
    flat.add(base)
    ids, _ = flat.search(queries, K)
    return ids


def test_flat_search_is_exact(split):
    _, base, queries = split
    flat = FlatIndex(DIM)
    flat.add(base)
    ids, dists = flat.search(queries, K)
    # Distances must be sorted ascending and match a direct computation.
    assert (np.diff(dists, axis=1) >= 0).all()
    d0 = ((queries[0] - base[ids[0, 0]]) ** 2).sum()
    assert dists[0, 0] == pytest.approx(d0, rel=1e-4)


def test_sq8_flat_recall(split, ground_truth):
    train, base, queries = split
    idx = QuantizedFlatIndex(ScalarQuantizer(DIM, bits=8).train(train))
    idx.add(base)
    ids, _ = idx.search(queries, K)
    # 8-bit SQ is nearly lossless for ranking.
    assert recall_at_k(ids, ground_truth, K) > 0.95


def test_pq_flat_recall(split, ground_truth):
    train, base, queries = split
    idx = QuantizedFlatIndex(ProductQuantizer(DIM, n_subspaces=8).train(train))
    idx.add(base)
    # 32x compression with no partitioning and no re-ranking is the
    # weakest configuration in the library; ~0.16 measured on this data.
    ids, _ = idx.search(queries, K)
    assert recall_at_k(ids, ground_truth, K) > 0.10


def test_ivfpq_recall_and_memory(split, ground_truth):
    train, base, queries = split
    idx = IVFPQIndex(DIM, n_lists=64, n_subspaces=8).train(train)
    idx.add(base)
    ids, _ = idx.search(queries, K, n_probe=16)
    # Residual encoding lifts IVFPQ well above flat PQ at the same M
    # (~0.41 measured vs ~0.16 flat).
    assert recall_at_k(ids, ground_truth, K) > 0.30
    assert idx.memory_bytes == base.shape[0] * 8  # M bytes per vector
    assert idx.ntotal == base.shape[0]


def test_ivfpq_recall_rises_with_nprobe(split, ground_truth):
    train, base, queries = split
    idx = IVFPQIndex(DIM, n_lists=64, n_subspaces=8).train(train)
    idx.add(base)
    recalls = []
    for n_probe in (1, 8, 64):
        ids, _ = idx.search(queries, K, n_probe=n_probe)
        recalls.append(recall_at_k(ids, ground_truth, K))
    assert recalls[0] < recalls[2]
    # At n_probe == nlist, IVF prunes nothing; only PQ error remains.
    assert recalls[2] > 0.30


def test_turbo_beats_plain_ivfpq_at_same_budget(split, ground_truth):
    """The cascade's reason to exist: more recall per byte than pure IVFPQ."""
    train, base, queries = split
    budget = 24.0

    turbo = TurboIndex(DIM, budget_bytes=budget, n_lists=64).train(train)
    turbo.add(base)
    t_ids, _ = turbo.search(queries, K, n_probe=16, rerank_factor=4)
    t_recall = recall_at_k(t_ids, ground_truth, K)

    # Plain IVFPQ spending as much of the budget as it can on PQ codes:
    # M must divide dim=64, so 16 bytes is the largest config under 24.
    plain = IVFPQIndex(DIM, n_lists=64, n_subspaces=16).train(train)
    plain.add(base)
    p_ids, _ = plain.search(queries, K, n_probe=16)
    p_recall = recall_at_k(p_ids, ground_truth, K)

    assert turbo.bytes_per_vector <= budget + 0.5
    # Measured: turbo ~0.69 at 24 B vs plain ~0.64 at 16 B.
    assert t_recall > p_recall
    assert t_recall > 0.6


def test_turbo_ids_valid(split):
    train, base, queries = split
    turbo = TurboIndex(DIM, budget_bytes=16.0, n_lists=32).train(train)
    turbo.add(base)
    ids, dists = turbo.search(queries, K, n_probe=8)
    assert ids.shape == (len(queries), K)
    valid = ids[ids >= 0]
    assert valid.max() < base.shape[0]
    assert (np.diff(dists, axis=1) >= 0).all()

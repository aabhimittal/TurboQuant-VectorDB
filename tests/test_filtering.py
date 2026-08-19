"""Filtered search and probe escalation.

"Nearest neighbors WHERE tenant_id = 7" is the query every production
vector store has to answer, and post-filtering an ANN scan is where it
quietly goes wrong: the more selective the filter, the fewer of the
scanned candidates survive, until a k=10 query is returning three results
and nobody notices because nothing raised.

These tests pin both halves: correctness (never return an excluded id)
and sufficiency (escalation must actually fill the result set, and must
recover the recall that post-filtering loses).
"""

import numpy as np
import pytest

from turboquant import (
    FlatIndex,
    IVFPQIndex,
    TurboIndex,
    recall_at_k,
    train_base_query_split,
)

DIM = 32
K = 10
N_BASE = 6000


@pytest.fixture(scope="module")
def split():
    return train_base_query_split(3000, N_BASE, 40, DIM, n_clusters=16, seed=21)


@pytest.fixture(scope="module")
def index(split):
    train, base, _ = split
    idx = IVFPQIndex(DIM, n_lists=64, n_subspaces=8).train(train)
    idx.add(base)
    return idx


def _mask(keep_every: int) -> np.ndarray:
    mask = np.zeros(N_BASE, dtype=bool)
    mask[::keep_every] = True
    return mask


def test_filtered_results_always_satisfy_the_filter(index, split):
    _, _, queries = split
    mask = _mask(7)
    ids, _ = index.search(queries, K, n_probe=16, filter_mask=mask, max_probe=64)
    returned = ids[ids >= 0]
    assert returned.size > 0
    assert mask[returned].all()


def test_selective_filter_starves_the_result_set_without_escalation(index, split):
    """The failure this feature exists to prevent, demonstrated.

    A 1%-selective filter over a 16-cell scan leaves far fewer than k
    survivors. Without escalation the index returns short rows; with it,
    full ones.
    """
    _, _, queries = split
    mask = _mask(100)  # keep 1%

    starved, _ = index.search(queries, K, n_probe=4, filter_mask=mask)
    filled, _ = index.search(
        queries, K, n_probe=4, filter_mask=mask, max_probe=index.nlist
    )

    assert (starved >= 0).sum(axis=1).mean() < K
    assert (filled >= 0).sum(axis=1).mean() > (starved >= 0).sum(axis=1).mean()
    assert (filled >= 0).all()


def test_escalation_recovers_recall_against_filtered_ground_truth(index, split):
    """Escalated filtered search should approach exact filtered search."""
    _, base, queries = split
    mask = _mask(50)
    eligible = np.flatnonzero(mask)

    flat = FlatIndex(DIM)
    flat.add(base[eligible])
    local_truth, _ = flat.search(queries, K)
    truth = eligible[local_truth]

    starved, _ = index.search(queries, K, n_probe=4, filter_mask=mask)
    filled, _ = index.search(
        queries, K, n_probe=4, filter_mask=mask, max_probe=index.nlist
    )
    r_starved = recall_at_k(starved, truth, K)
    r_filled = recall_at_k(filled, truth, K)
    assert r_filled > r_starved
    assert r_filled > 0.7


def test_escalation_costs_nothing_when_the_filter_is_permissive(index, split):
    """No filter, no extra work: escalation must not fire when unneeded.

    Identical results with and without a generous max_probe means the
    widening loop stopped at the first round, as it should.
    """
    _, _, queries = split
    plain, _ = index.search(queries, K, n_probe=16)
    with_headroom, _ = index.search(queries, K, n_probe=16, max_probe=index.nlist)
    assert np.array_equal(plain, with_headroom)


def test_all_false_filter_returns_empty_rows(index, split):
    _, _, queries = split
    ids, dists = index.search(
        queries,
        K,
        n_probe=4,
        filter_mask=np.zeros(N_BASE, dtype=bool),
        max_probe=index.nlist,
    )
    assert (ids == -1).all() and np.isinf(dists).all()


def test_all_true_filter_matches_unfiltered_search(index, split):
    _, _, queries = split
    a, _ = index.search(queries, K, n_probe=8)
    b, _ = index.search(queries, K, n_probe=8, filter_mask=np.ones(N_BASE, dtype=bool))
    assert np.array_equal(a, b)


def test_filter_and_tombstones_compose(split):
    """A deleted id must stay excluded even when the filter admits it."""
    train, base, queries = split
    idx = IVFPQIndex(DIM, n_lists=64, n_subspaces=8).train(train)
    idx.add(base)
    victims = np.arange(0, N_BASE, 3)
    idx.remove_ids(victims)

    ids, _ = idx.search(
        queries,
        K,
        n_probe=8,
        filter_mask=np.ones(N_BASE, dtype=bool),
        max_probe=idx.nlist,
    )
    returned = set(ids[ids >= 0].tolist())
    assert returned.isdisjoint(set(victims.tolist()))


def test_malformed_filter_masks_are_rejected(index, split):
    _, _, queries = split
    with pytest.raises(ValueError, match="shape"):
        index.search(queries, K, filter_mask=np.ones(5, dtype=bool))
    with pytest.raises(TypeError, match="boolean"):
        index.search(queries, K, filter_mask=np.ones(N_BASE, dtype=np.int8))


def test_turbo_cascade_honors_filters(split):
    """The cascade must filter in tier 1, or tier 2 re-ranks the wrong set."""
    train, base, queries = split
    idx = TurboIndex(DIM, budget_bytes=16.0, n_lists=64).train(train)
    idx.add(base)
    mask = _mask(20)

    ids, dists = idx.search(
        queries, K, n_probe=8, filter_mask=mask, max_probe=idx.ivfpq.nlist
    )
    returned = ids[ids >= 0]
    assert mask[returned].all()
    assert (np.diff(dists, axis=1) >= 0).all()


def test_escalation_is_bounded_by_max_probe(index, split):
    """max_probe is a ceiling, not a suggestion -- it caps worst-case latency."""
    _, _, queries = split
    mask = np.zeros(N_BASE, dtype=bool)
    mask[-1] = True  # exactly one eligible vector, deep in some cell
    ids, _ = index.search(queries, K, n_probe=1, filter_mask=mask, max_probe=2)
    # With only two cells scanned, most queries cannot reach the single
    # eligible vector; the call must still return cleanly rather than
    # widening forever.
    assert ids.shape == (len(queries), K)
    assert set(ids[ids >= 0].tolist()) <= {N_BASE - 1}

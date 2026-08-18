"""Deletion, update and compaction -- the lifecycle a live index actually has.

A corpus is not loaded once and frozen. Documents are removed, re-embedded
after a model upgrade, and re-indexed continuously. The invariants that
matter are: a deleted vector never comes back in results, an updated
vector is findable at its *new* location and not at its old one, ids stay
stable until an explicit compaction, and compaction hands back the id map
so parallel per-id state can follow.
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


@pytest.fixture(scope="module")
def split():
    return train_base_query_split(3000, 4000, 40, DIM, n_clusters=16, seed=13)


def _ivfpq(train):
    return IVFPQIndex(DIM, n_lists=32, n_subspaces=8).train(train)


def _turbo(train):
    return TurboIndex(DIM, budget_bytes=16.0, n_lists=32).train(train)


# ------------------------------------------------------------------ deletion
def test_deleted_ids_never_appear_in_results(split):
    train, base, queries = split
    idx = _ivfpq(train)
    idx.add(base)
    victims = np.arange(0, len(base), 3)
    idx.remove_ids(victims)

    ids, _ = idx.search(queries, K, n_probe=32)
    returned = set(ids[ids >= 0].tolist())
    assert returned.isdisjoint(set(victims.tolist()))


def test_delete_accounting(split):
    train, base, _ = split
    idx = _ivfpq(train)
    idx.add(base)
    assert idx.ntotal == len(base) and idx.n_deleted == 0

    assert idx.remove_ids([1, 2, 3]) == 3
    assert idx.ntotal == len(base) - 3 and idx.n_deleted == 3
    # Re-deleting is idempotent and reports zero *newly* removed.
    assert idx.remove_ids([1, 2, 3]) == 0
    assert idx.n_deleted == 3


def test_delete_does_not_renumber_survivors(split):
    """Ids are external handles; deleting one must not move another."""
    train, base, queries = split
    idx = _ivfpq(train)
    idx.add(base)
    before, _ = idx.search(queries, K, n_probe=32)

    idx.remove_ids([int(before[0, 0])])
    after, _ = idx.search(queries, K, n_probe=32)
    # Every id that survived and was returned before is still that id.
    survivors = set(after[after >= 0].tolist())
    assert survivors.issubset(set(range(len(base))))
    assert int(before[0, 0]) not in survivors


def test_deleting_everything_returns_empty_rows(split):
    train, base, queries = split
    idx = _ivfpq(train)
    idx.add(base)
    idx.remove_ids(np.arange(len(base)))
    ids, dists = idx.search(queries, K, n_probe=32)
    assert (ids == -1).all() and np.isinf(dists).all()
    assert idx.ntotal == 0


def test_out_of_range_delete_is_rejected(split):
    train, base, _ = split
    idx = _ivfpq(train)
    idx.add(base)
    with pytest.raises(IndexError):
        idx.remove_ids([len(base)])
    # Negative ids would silently wrap to the end of the array under
    # NumPy fancy-indexing -- deleting the newest vector, not erroring.
    with pytest.raises(IndexError):
        idx.remove_ids([-1])


# -------------------------------------------------------------------- update
def test_update_moves_a_vector_to_its_new_neighborhood(split):
    """An updated vector must be findable where it now is.

    This is the check that catches forgetting to re-file the id into its
    new coarse cell: re-encoding alone leaves the vector filed under its
    old centroid, invisible to queries near its new position.
    """
    train, base, _ = split
    idx = _ivfpq(train)
    idx.add(base)

    target = 7
    new_vector = base[-1] * 1.0  # somewhere else entirely in the space
    idx.update(np.array([target]), new_vector[None, :])

    ids, _ = idx.search(new_vector[None, :], K, n_probe=32)
    assert target in ids[0].tolist()


def test_update_revives_a_deleted_id(split):
    train, base, _ = split
    idx = _ivfpq(train)
    idx.add(base)
    idx.remove_ids([4])
    assert idx.n_deleted == 1
    idx.update(np.array([4]), base[4][None, :])
    assert idx.n_deleted == 0
    ids, _ = idx.search(base[4][None, :], K, n_probe=32)
    assert 4 in ids[0].tolist()


def test_updated_index_is_indistinguishable_from_a_rebuilt_one(split):
    """The strongest statement available: state depends on contents alone.

    An index that was mutated into a given set of vectors must be
    byte-identical to one built from those vectors directly -- same codes,
    same cells, same inverted-list *order*, same search output. Without
    the re-sort in `update`, everything matches except list order, and
    equal-distance candidates then tie-break differently: identical
    distances, different ids, and no way to tell a real regression from
    mutation history.
    """
    train, base, queries = split
    ids = np.arange(0, len(base), 37)

    mutated = _ivfpq(train)
    mutated.add(base)
    replacement = train[: len(ids)]
    mutated.update(ids, replacement)

    changed = base.copy()
    changed[ids] = replacement
    rebuilt = _ivfpq(train)
    rebuilt.add(changed)

    assert np.array_equal(mutated.codes, rebuilt.codes)
    assert np.array_equal(mutated.cells, rebuilt.cells)
    for a, b in zip(mutated.list_ids, rebuilt.list_ids):
        assert np.array_equal(a, b)
    m_ids, m_d = mutated.search(queries, K, n_probe=16)
    r_ids, r_d = rebuilt.search(queries, K, n_probe=16)
    assert np.array_equal(m_ids, r_ids)
    assert np.allclose(m_d, r_d)


def test_update_rejects_mismatched_lengths_and_duplicates(split):
    train, base, _ = split
    idx = _ivfpq(train)
    idx.add(base)
    with pytest.raises(ValueError, match="ids but"):
        idx.update(np.array([1, 2]), base[:1])
    with pytest.raises(ValueError, match="unique"):
        idx.update(np.array([1, 1]), base[:2])


def test_turbo_update_refreshes_the_second_tier(split):
    """Tier 2 encodes tier 1's error on *this* vector.

    If update() refreshed only tier 1, the stale correction would describe
    the old vector and re-ranking would trust it -- actively worse than
    having no correction. Compare against a freshly built index holding
    the same final contents.
    """
    train, base, _ = split
    idx = _turbo(train)
    idx.add(base)

    # Deliberately a vector from the disjoint training split, not a copy of
    # another base row: an exact duplicate would tie for nearest neighbor
    # and the comparison would test tie-breaking order, not correctness.
    changed = base.copy()
    changed[3] = train[0]
    idx.update(np.array([3]), changed[3][None, :])

    reference = _turbo(train)
    reference.add(changed)

    query = changed[3][None, :]
    got, _ = idx.search(query, K, n_probe=32)
    want, _ = reference.search(query, K, n_probe=32)
    assert got[0, 0] == want[0, 0] == 3


# ----------------------------------------------------------------- compaction
def test_compact_reclaims_memory_and_remaps_ids(split):
    train, base, _ = split
    idx = _ivfpq(train)
    idx.add(base)
    before_bytes = idx.memory_bytes

    victims = np.arange(0, len(base), 2)
    idx.remove_ids(victims)
    # Tombstoned rows are still resident until compaction; say so honestly.
    assert idx.memory_bytes == before_bytes

    mapping = idx.compact()
    assert idx.memory_bytes == pytest.approx(before_bytes / 2, rel=0.01)
    assert idx.n_deleted == 0
    assert idx.ntotal == idx.n_slots == len(base) - len(victims)
    assert (mapping[victims] == -1).all()
    survivors = np.setdiff1d(np.arange(len(base)), victims)
    assert np.array_equal(mapping[survivors], np.arange(len(survivors)))


def test_compact_preserves_search_semantics(split):
    """The same query must find the same *vectors*, under their new ids."""
    train, base, queries = split
    idx = _ivfpq(train)
    idx.add(base)
    victims = np.arange(0, len(base), 5)
    idx.remove_ids(victims)

    before, _ = idx.search(queries, K, n_probe=32)
    mapping = idx.compact()
    after, _ = idx.search(queries, K, n_probe=32)

    expected = np.where(before >= 0, mapping[np.maximum(before, 0)], -1)
    assert np.array_equal(after, expected)


def test_turbo_compact_keeps_both_tiers_aligned(split):
    """Tier-2 codes are addressed by id; compaction must renumber them too.

    A misalignment here is silent and disastrous: every re-rank would add
    some *other* vector's correction term.
    """
    train, base, queries = split
    idx = _turbo(train)
    idx.add(base)
    idx.remove_ids(np.arange(0, len(base), 4))

    before, _ = idx.search(queries, K, n_probe=16)
    mapping = idx.compact()
    after, dists = idx.search(queries, K, n_probe=16)

    assert idx.refine_codes.shape[0] == idx.ivfpq.n_slots
    expected = np.where(before >= 0, mapping[np.maximum(before, 0)], -1)
    assert np.array_equal(after, expected)
    assert (np.diff(dists, axis=1) >= 0).all()


def test_compact_on_a_clean_index_is_identity(split):
    train, base, queries = split
    idx = _ivfpq(train)
    idx.add(base)
    before, _ = idx.search(queries, K, n_probe=16)
    mapping = idx.compact()
    after, _ = idx.search(queries, K, n_probe=16)
    assert np.array_equal(mapping, np.arange(len(base)))
    assert np.array_equal(before, after)


def test_add_after_compact_continues_numbering(split):
    train, base, _ = split
    idx = _ivfpq(train)
    idx.add(base[:100])
    idx.remove_ids([0, 1])
    idx.compact()
    idx.add(base[100:110])
    assert idx.n_slots == 108 and idx.ntotal == 108


def test_recall_survives_a_delete_compact_cycle(split):
    """End-to-end: quality must not quietly degrade across mutations."""
    train, base, queries = split
    keep = np.arange(len(base))[np.arange(len(base)) % 3 != 0]

    flat = FlatIndex(DIM)
    flat.add(base[keep])
    truth, _ = flat.search(queries, K)

    idx = _ivfpq(train)
    idx.add(base)
    idx.remove_ids(np.arange(0, len(base), 3))
    idx.compact()

    ids, _ = idx.search(queries, K, n_probe=32)
    assert recall_at_k(ids, truth, K) > 0.5

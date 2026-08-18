"""Persistence: a reloaded index must be indistinguishable from the original.

"Close enough" is not a passing grade here. Training is the expensive
part of building an index, so save/load sits on the deploy path, and a
reload that shifts results by one neighbor is a silent quality
regression that no monitoring will attribute to serialization. Every test
below demands bit-identical codes or identical result ids.

The suite also pins two properties that are easy to lose later: that
loading never unpickles, and that a new attribute added to a serializable
class cannot silently escape the archive.
"""

import numpy as np
import pytest

from turboquant import (
    AdaptiveBitQuantizer,
    AnisotropicPQ,
    FlatIndex,
    IVFPQIndex,
    OPQProductQuantizer,
    ProductQuantizer,
    QuantizedFlatIndex,
    ScalarQuantizer,
    TurboIndex,
    load,
    save,
    train_base_query_split,
)
from turboquant.io import FORMAT_VERSION

DIM = 32
K = 10


@pytest.fixture(scope="module")
def split():
    return train_base_query_split(2000, 3000, 30, DIM, n_clusters=16, seed=17)


def _quantizers(train):
    return [
        ScalarQuantizer(DIM, bits=8).train(train),
        ScalarQuantizer(DIM, bits=3).train(train),
        AdaptiveBitQuantizer(DIM, avg_bits=3.5).train(train),
        ProductQuantizer(DIM, n_subspaces=8, n_centroids=64).train(train),
        OPQProductQuantizer(
            DIM, n_subspaces=8, n_centroids=64, n_rotation_iters=2
        ).train(train),
        AnisotropicPQ(DIM, n_subspaces=8, n_centroids=64, n_sweeps=2).train(train),
    ]


def test_quantizer_roundtrip_is_bit_identical(split, tmp_path):
    train, base, _ = split
    for q in _quantizers(train):
        name = type(q).__name__
        restored = load(save(q, tmp_path / f"{name}_{q.bytes_per_vector}"))
        assert type(restored) is type(q)
        assert np.array_equal(restored.encode(base), q.encode(base)), name
        assert np.array_equal(
            restored.decode(q.encode(base)), q.decode(q.encode(base))
        ), name
        assert restored.bytes_per_vector == q.bytes_per_vector, name


def test_adaptive_layout_is_rebuilt_not_stored(split, tmp_path):
    """Derived state must be recomputed correctly, not just present.

    The bit-packing layout is dropped on save; if the rebuild hook were
    wrong, decode would read the bitstream at the wrong offsets and
    produce garbage that still has the right shape.
    """
    train, base, _ = split
    q = AdaptiveBitQuantizer(DIM, avg_bits=2.5).train(train)
    restored = load(save(q, tmp_path / "adaptive"))
    assert restored._groups is not q._groups
    assert restored._stream_bits == q._stream_bits
    assert np.array_equal(restored.decode(q.encode(base)), q.decode(q.encode(base)))


def test_index_roundtrip_preserves_search_exactly(split, tmp_path):
    train, base, queries = split
    indexes = {
        "flat": FlatIndex(DIM),
        "qflat": QuantizedFlatIndex(
            ProductQuantizer(DIM, n_subspaces=8, n_centroids=64).train(train)
        ),
        "ivfpq": IVFPQIndex(DIM, n_lists=32, n_subspaces=8).train(train),
        "turbo": TurboIndex(DIM, budget_bytes=16.0, n_lists=32).train(train),
    }
    for name, idx in indexes.items():
        idx.add(base)
        restored = load(save(idx, tmp_path / name))
        kwargs = {} if name in ("flat", "qflat") else {"n_probe": 8}
        a_ids, a_d = idx.search(queries, K, **kwargs)
        b_ids, b_d = restored.search(queries, K, **kwargs)
        assert np.array_equal(a_ids, b_ids), name
        assert np.allclose(a_d, b_d), name


def test_inverted_lists_are_rebuilt_from_cells(split, tmp_path):
    """The lists cost more than the codes; they must not be in the file."""
    train, base, _ = split
    idx = IVFPQIndex(DIM, n_lists=32, n_subspaces=8).train(train)
    idx.add(base)
    path = save(idx, tmp_path / "ivfpq")
    restored = load(path)

    for original, rebuilt in zip(idx.list_ids, restored.list_ids):
        assert np.array_equal(np.sort(original), np.sort(rebuilt))
    # id lists would be 8 bytes/vector of int64 on top of 8 bytes of codes.
    assert path.stat().st_size < base.nbytes / 2


def test_tombstones_survive_the_roundtrip(split, tmp_path):
    train, base, queries = split
    idx = IVFPQIndex(DIM, n_lists=32, n_subspaces=8).train(train)
    idx.add(base)
    idx.remove_ids(np.arange(0, 300, 2))

    restored = load(save(idx, tmp_path / "deleted"))
    assert restored.ntotal == idx.ntotal
    assert restored.n_deleted == idx.n_deleted
    assert np.array_equal(
        restored.search(queries, K, n_probe=8)[0],
        idx.search(queries, K, n_probe=8)[0],
    )


def test_turbo_budget_split_is_restored_not_recomputed(split, tmp_path):
    """Loading bypasses __init__, so the saved split must come back verbatim."""
    train, base, _ = split
    idx = TurboIndex(DIM, budget_bytes=20.0, n_lists=32).train(train)
    idx.add(base)
    restored = load(save(idx, tmp_path / "turbo"))
    assert restored.refine_kind == idx.refine_kind
    assert restored.ivfpq.pq.M == idx.ivfpq.pq.M
    assert restored.bytes_per_vector == idx.bytes_per_vector


def test_untrained_objects_roundtrip(tmp_path):
    """Saving before training should not be a special case that explodes."""
    q = AdaptiveBitQuantizer(DIM, avg_bits=4.0)
    restored = load(save(q, tmp_path / "untrained"))
    assert restored.is_trained is False
    assert restored.total_bits == q.total_bits


def test_index_can_be_mutated_after_loading(split, tmp_path):
    """A reloaded index is a live index, not a read-only snapshot."""
    train, base, queries = split
    idx = IVFPQIndex(DIM, n_lists=32, n_subspaces=8).train(train)
    idx.add(base[:1000])
    restored = load(save(idx, tmp_path / "mutable"))

    restored.add(base[1000:1500])
    restored.remove_ids([0, 1, 2])
    assert restored.ntotal == 1497 and restored.n_slots == 1500
    ids, _ = restored.search(queries, K, n_probe=8)
    assert ids[ids >= 0].max() < 1500


def test_load_refuses_a_foreign_npz(tmp_path):
    path = tmp_path / "foreign.npz"
    np.savez(path, x=np.arange(4))
    with pytest.raises(ValueError, match="not a TurboQuant archive"):
        load(path)


def test_load_refuses_a_future_format_version(split, tmp_path):
    """A newer writer must fail loudly, not be misread as the current format."""
    import json

    train, _, _ = split
    path = save(ScalarQuantizer(DIM).train(train), tmp_path / "v")
    with np.load(path, allow_pickle=False) as data:
        contents = dict(data)
    manifest = json.loads(bytes(contents.pop("__manifest__")).decode())
    manifest["version"] = FORMAT_VERSION + 1
    np.savez(
        path,
        __manifest__=np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8),
        **contents,
    )
    with pytest.raises(ValueError, match="format version"):
        load(path)


def test_save_refuses_an_unserializable_attribute(split, tmp_path):
    """A new attribute must not slip out of the archive unnoticed.

    This is the guard that keeps the reflective serializer honest: adding
    a field to a class and forgetting about persistence fails at save
    time, instead of producing archives that silently lose state.
    """
    train, _, _ = split
    q = ScalarQuantizer(DIM).train(train)
    q.new_thing = {"not": "serializable"}
    with pytest.raises(TypeError, match="cannot serialize attribute"):
        save(q, tmp_path / "bad")


def test_save_refuses_an_unregistered_class(tmp_path):
    class Sneaky:
        pass

    with pytest.raises(TypeError, match="not serializable"):
        save(Sneaky(), tmp_path / "sneaky")


def test_loading_does_not_unpickle(split, tmp_path, monkeypatch):
    """Archives must be readable with allow_pickle=False, always.

    An index restored from object storage is attacker-controlled input if
    the bucket is writable; `allow_pickle=True` would turn that into
    arbitrary code execution.
    """
    train, base, _ = split
    idx = TurboIndex(DIM, budget_bytes=16.0, n_lists=32).train(train)
    idx.add(base)
    path = save(idx, tmp_path / "safe")

    real_load = np.load

    def no_pickle(*args, **kwargs):
        assert kwargs.get("allow_pickle") is False, "np.load must forbid pickle"
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", no_pickle)
    assert load(path).ntotal == idx.ntotal

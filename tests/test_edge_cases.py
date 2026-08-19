"""Industrial edge cases: the inputs that reach a vector index in production.

The recall tests elsewhere check that the algorithms are good. These
check that the library is *safe* -- that the ugly, degenerate, and
hostile inputs a real ingest pipeline produces either work correctly or
fail with an error that names the problem, and never silently return
wrong neighbors.

Grouped by the failure mode each class of input causes:

* NaN/Inf -- poison the statistics, then every query, invisibly.
* Degenerate geometry (constant dims, duplicate rows, zero vectors) --
  division by zero and empty k-means clusters.
* Extreme magnitudes -- float32 overflow and catastrophic cancellation in
  the distance kernel.
* Boundary sizes (k > n, n == 1, dim == 1, empty batches) -- off-by-one
  and shape bugs.
* Caller mistakes (wrong dim, wrong order of operations, bad k) -- should
  be loud, not creative.
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
    clustered_embeddings,
    kmeans,
    pairwise_l2_sq,
)

DIM = 16


@pytest.fixture(scope="module")
def data():
    return clustered_embeddings(1500, DIM, n_clusters=8, seed=11)


def _quantizers(dim=DIM):
    """One trained-capable instance of every quantizer, small and fast."""
    return [
        ScalarQuantizer(dim, bits=8),
        ScalarQuantizer(dim, bits=3),
        AdaptiveBitQuantizer(dim, avg_bits=4.0),
        ProductQuantizer(dim, n_subspaces=4, n_centroids=16),
        OPQProductQuantizer(dim, n_subspaces=4, n_centroids=16, n_rotation_iters=2),
        AnisotropicPQ(dim, n_subspaces=4, n_centroids=16, n_sweeps=2),
    ]


# --------------------------------------------------------------- non-finite
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_training_data_is_rejected(data, bad):
    """A single NaN must not be allowed to reach the statistics.

    This is the highest-value check in the file: NaN propagates through
    percentiles and k-means into every parameter, and the index then
    answers every query wrongly while raising nothing.
    """
    poisoned = data.copy()
    poisoned[7, 3] = bad
    for q in _quantizers():
        with pytest.raises(ValueError, match="non-finite"):
            q.train(poisoned)


def test_non_finite_error_names_the_offending_position(data):
    poisoned = data.copy()
    poisoned[42, 5] = np.nan
    with pytest.raises(ValueError, match=r"row 42, column 5"):
        ScalarQuantizer(DIM).train(poisoned)


def test_non_finite_query_is_rejected(data):
    idx = FlatIndex(DIM)
    idx.add(data)
    with pytest.raises(ValueError, match="non-finite"):
        idx.search(np.full((1, DIM), np.nan, dtype=np.float32), 5)


def test_float64_overflow_on_narrowing_is_caught(data):
    """1e308 is finite in float64 and infinite in float32.

    The check has to run *after* the cast, or this input sails through
    validation and becomes an Inf inside the index.
    """
    huge = np.full((20, DIM), 1e308, dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        ScalarQuantizer(DIM).train(huge)


# ------------------------------------------------------------ dtype / layout
def test_float64_input_is_accepted_and_matches_float32(data):
    q = ScalarQuantizer(DIM, bits=8).train(data)
    assert np.array_equal(q.encode(data.astype(np.float64)), q.encode(data))


def test_non_contiguous_and_fortran_order_inputs_work(data):
    q = ScalarQuantizer(DIM, bits=8).train(data)
    expected = q.encode(data[:50])
    assert np.array_equal(q.encode(np.asfortranarray(data[:50])), expected)
    # A strided view: every other row of a doubled array.
    strided = np.repeat(data[:50], 2, axis=0)[::2]
    assert np.array_equal(q.encode(strided), expected)


def test_python_lists_are_accepted(data):
    q = ScalarQuantizer(DIM, bits=8).train(data)
    assert q.encode(data[:3].tolist()).shape == q.encode(data[:3]).shape


def test_ragged_input_raises_type_error():
    with pytest.raises(TypeError):
        ScalarQuantizer(3).train([[1.0, 2.0, 3.0], [1.0, 2.0]])


def test_string_input_raises_type_error():
    with pytest.raises(TypeError, match="numeric"):
        ScalarQuantizer(2).train(np.array([["a", "b"]]))


def test_single_query_is_promoted_to_2d(data):
    """`search(q, k)` with one 1-D query is an unambiguous, common call."""
    idx = FlatIndex(DIM)
    idx.add(data)
    ids, _ = idx.search(data[0], 5)
    assert ids.shape == (1, 5)


def test_3d_input_is_rejected(data):
    with pytest.raises(ValueError, match="1-D or 2-D"):
        FlatIndex(DIM).add(data[:8].reshape(2, 4, DIM))


def test_dimension_mismatch_is_rejected(data):
    idx = FlatIndex(DIM)
    idx.add(data)
    with pytest.raises(ValueError, match="dimension"):
        idx.search(np.zeros((1, DIM + 1), dtype=np.float32), 3)


# ------------------------------------------------------- degenerate geometry
def test_constant_dimension_does_not_divide_by_zero(data):
    """A dead feature column (all rows identical) has zero range.

    Zero span is a literal divide-by-zero in every scalar codec and a
    zero-variance direction in every k-means. Nothing may produce NaN,
    and nothing may blow the column up.
    """
    flat = data.copy()
    flat[:, 2] = 0.5
    scale = float(np.abs(data).mean())
    for q in _quantizers():
        q.train(flat)
        out = q.decode(q.encode(flat))
        assert np.isfinite(out).all()
        assert np.abs(out[:, 2] - 0.5).max() < scale


def test_constant_dimension_is_exact_for_axis_aligned_codecs(data):
    """The codecs that quantize per-dimension must reproduce it exactly.

    SQ, adaptive-bit SQ and PQ all keep the coordinate axes: a column that
    never varies costs them nothing to store perfectly (SQ collapses its
    range, PQ's centroid coordinate is the column's mean).

    OPQ and AnisotropicPQ are deliberately excluded. OPQ rotates before
    splitting, so the constant column is smeared across the rotated basis
    and its error is shared with every other dimension. AnisotropicPQ
    places centroids by weighted normal equations rather than means, so it
    will happily perturb a constant coordinate if that buys score
    accuracy. Both are the documented behavior of those codecs, not
    defects -- which is exactly why this test names the split instead of
    weakening the bound for everyone.
    """
    flat = data.copy()
    flat[:, 2] = 0.5
    axis_aligned = [
        ScalarQuantizer(DIM, bits=8),
        AdaptiveBitQuantizer(DIM, avg_bits=4.0),
        ProductQuantizer(DIM, n_subspaces=4, n_centroids=16),
    ]
    for q in axis_aligned:
        q.train(flat)
        out = q.decode(q.encode(flat))
        assert np.abs(out[:, 2] - 0.5).max() < 1e-3, type(q).__name__


def test_all_identical_vectors():
    """Every k-means cluster but one is empty; codebooks must survive it."""
    same = np.full((300, DIM), 2.5, dtype=np.float32)
    for q in _quantizers():
        q.train(same)
        out = q.decode(q.encode(same))
        assert np.isfinite(out).all()
        assert np.abs(out - 2.5).max() < 1e-3


def test_all_zero_vectors():
    """Zero rows have no direction -- the anisotropic loss must not 0/0."""
    zeros = np.zeros((200, DIM), dtype=np.float32)
    for q in _quantizers():
        q.train(zeros)
        out = q.decode(q.encode(zeros))
        assert np.isfinite(out).all()
        assert np.abs(out).max() < 1e-3


def test_duplicate_rows_beyond_codebook_size():
    """Fewer distinct points than centroids: k-means++ must not divide by 0."""
    data = np.repeat(np.arange(4, dtype=np.float32)[:, None], DIM, axis=1)
    data = np.repeat(data, 50, axis=0)  # 200 rows, 4 distinct
    centroids = kmeans(data, k=32, n_iters=5, seed=0)
    assert centroids.shape == (32, DIM)
    assert np.isfinite(centroids).all()


def test_kmeans_rejects_k_above_population():
    with pytest.raises(ValueError, match="exceeds number of training points"):
        kmeans(np.zeros((5, DIM), dtype=np.float32), k=10)


def test_single_vector_index():
    one = np.ones((1, DIM), dtype=np.float32)
    idx = FlatIndex(DIM)
    idx.add(one)
    ids, dists = idx.search(one, 5)
    assert ids.shape == (1, 1) and dists[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_dim_one_end_to_end():
    """d=1 makes every subspace one-dimensional; nothing may assume d > 1."""
    x = np.linspace(-3, 3, 400, dtype=np.float32)[:, None]
    pq = ProductQuantizer(1, n_subspaces=1, n_centroids=16).train(x)
    idx = QuantizedFlatIndex(pq)
    idx.add(x)
    ids, _ = idx.search(x[:5], 3)
    assert ids.shape == (5, 3) and (ids >= 0).all()


# ---------------------------------------------------------- extreme numbers
def test_extreme_magnitudes_stay_finite_and_non_negative():
    """||q||^2 at 1e18 is 1e36 -- within float32 range, but only just.

    The expansion ||q-x||^2 = ||q||^2 - 2q.x + ||x||^2 also cancels
    catastrophically here, which is why the kernel clamps at zero.
    """
    rng = np.random.default_rng(0)
    big = rng.standard_normal((100, DIM)).astype(np.float32) * 1e18
    d = pairwise_l2_sq(big[:5], big)
    assert np.isfinite(d).all()
    assert (d >= 0).all()


def test_tiny_magnitudes_do_not_underflow_to_garbage():
    rng = np.random.default_rng(0)
    tiny = rng.standard_normal((100, DIM)).astype(np.float32) * 1e-20
    d = pairwise_l2_sq(tiny[:5], tiny)
    assert np.isfinite(d).all() and (d >= 0).all()
    assert np.diag(d[:, :5]) == pytest.approx(np.zeros(5), abs=1e-30)


def test_mixed_scale_dimensions(data):
    """One dimension a billion times wider than the rest.

    This is the case that motivates per-dimension ranges; a global range
    would collapse every other dimension to a single level.
    """
    skewed = data.copy()
    skewed[:, 0] *= 1e9
    q = ScalarQuantizer(DIM, bits=8).train(skewed)
    out = q.decode(q.encode(skewed))
    # The narrow dimensions must still be resolved, not flattened.
    assert out[:, 1].std() > 0.1 * skewed[:, 1].std()


# -------------------------------------------------------------- boundaries
def test_k_larger_than_index_returns_what_exists(data):
    idx = FlatIndex(DIM)
    idx.add(data[:4])
    ids, dists = idx.search(data[:2], 100)
    assert ids.shape == (2, 4) and dists.shape == (2, 4)


def test_ivfpq_pads_short_rows_with_sentinels(data):
    """Too few candidates must be padded with -1 / +inf, never left stale."""
    idx = IVFPQIndex(DIM, n_lists=16, n_subspaces=4, n_centroids=16).train(data)
    idx.add(data[:20])
    ids, dists = idx.search(data[:3], 50, n_probe=1)
    assert ((ids == -1) == np.isinf(dists)).all()


def test_empty_add_is_a_noop(data):
    idx = IVFPQIndex(DIM, n_lists=8, n_subspaces=4, n_centroids=16).train(data)
    idx.add(data[:50])
    idx.add(np.empty((0, DIM), dtype=np.float32))
    assert idx.ntotal == 50


def test_empty_training_set_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        ScalarQuantizer(DIM).train(np.empty((0, DIM), dtype=np.float32))


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_rejected(data, k):
    idx = FlatIndex(DIM)
    idx.add(data)
    with pytest.raises(ValueError, match=">= 1"):
        idx.search(data[:1], k)


def test_float_k_is_rejected(data):
    idx = FlatIndex(DIM)
    idx.add(data)
    with pytest.raises(TypeError, match="integer"):
        idx.search(data[:1], 2.5)


# ------------------------------------------------------- order-of-operations
def test_untrained_quantizer_refuses_to_encode(data):
    for q in _quantizers():
        with pytest.raises(RuntimeError, match="trained"):
            q.encode(data[:2])


def test_untrained_index_refuses_add_and_search(data):
    idx = IVFPQIndex(DIM, n_lists=8, n_subspaces=4, n_centroids=16)
    with pytest.raises(RuntimeError, match="trained"):
        idx.add(data)
    with pytest.raises(RuntimeError, match="trained"):
        idx.search(data[:1], 3)
    turbo = TurboIndex(DIM, budget_bytes=8.0, n_lists=8)
    with pytest.raises(RuntimeError, match="trained"):
        turbo.add(data)


def test_empty_index_search_is_a_clear_error(data):
    with pytest.raises(RuntimeError, match="empty"):
        FlatIndex(DIM).search(data[:1], 3)


def test_more_lists_than_training_vectors_is_a_clear_error(data):
    with pytest.raises(ValueError, match="n_lists"):
        IVFPQIndex(DIM, n_lists=500, n_subspaces=4).train(data[:100])


def test_indivisible_subspace_count_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ProductQuantizer(10, n_subspaces=3)


def test_codebook_larger_than_uint8_is_rejected():
    with pytest.raises(ValueError, match="uint8"):
        ProductQuantizer(DIM, n_subspaces=4, n_centroids=512)


def test_too_few_training_points_for_codebook_is_rejected():
    with pytest.raises(ValueError, match="one per centroid"):
        ProductQuantizer(DIM, n_subspaces=2, n_centroids=64).train(
            np.zeros((10, DIM), dtype=np.float32)
        )


# ------------------------------------------------------------- determinism
def test_training_is_reproducible_under_a_fixed_seed(data):
    a = ProductQuantizer(DIM, n_subspaces=4, n_centroids=16).train(data, seed=5)
    b = ProductQuantizer(DIM, n_subspaces=4, n_centroids=16).train(data, seed=5)
    assert np.array_equal(a.codebooks, b.codebooks)


def test_different_seeds_give_different_codebooks(data):
    a = ProductQuantizer(DIM, n_subspaces=4, n_centroids=16).train(data, seed=1)
    b = ProductQuantizer(DIM, n_subspaces=4, n_centroids=16).train(data, seed=2)
    assert not np.array_equal(a.codebooks, b.codebooks)


def test_search_results_are_ordered_and_in_range(data):
    idx = TurboIndex(DIM, budget_bytes=12.0, n_lists=8).train(data)
    idx.add(data)
    ids, dists = idx.search(data[:10], 5, n_probe=4)
    assert (np.diff(dists, axis=1) >= 0).all()
    assert ids[ids >= 0].max() < len(data)


# ------------------------------------------------------- quantizer contracts
@pytest.mark.parametrize("bits", list(range(1, 9)))
def test_bit_packing_roundtrip_is_exact(bits):
    """Packing is lossless for every width; only quantization loses data."""
    from turboquant.quantizers.scalar import _pack_codes, _unpack_codes

    rng = np.random.default_rng(bits)
    codes = rng.integers(0, 1 << bits, size=(37, 23), dtype=np.uint8)
    assert np.array_equal(_unpack_codes(_pack_codes(codes, bits), bits, 23), codes)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_scalar_quantizer_storage_claim_is_honest(data, bits):
    q = ScalarQuantizer(DIM, bits=bits).train(data)
    codes = q.encode(data)
    assert codes.shape[1] == q.bytes_per_vector
    assert q.compression_ratio() == pytest.approx(4 * DIM / q.bytes_per_vector)


def test_adaptive_quantizer_storage_claim_is_honest(data):
    q = AdaptiveBitQuantizer(DIM, avg_bits=3.0).train(data)
    codes = q.encode(data)
    assert codes.shape[1] == q.bytes_per_vector
    assert q.bits_per_dim.sum() <= q.total_bits


def test_out_of_range_values_saturate_rather_than_wrap(data):
    """Beyond the trained percentile range, codes must clip to the ends.

    Wrapping would be catastrophic and silent: an outlier at +1e6 would
    reconstruct near the *bottom* of the range and land in the wrong
    neighborhood entirely.
    """
    q = ScalarQuantizer(DIM, bits=4).train(data)
    hi = q.decode(q.encode(np.full((1, DIM), 1e6, dtype=np.float32)))
    lo = q.decode(q.encode(np.full((1, DIM), -1e6, dtype=np.float32)))
    assert np.isfinite(hi).all() and np.isfinite(lo).all()
    assert (hi > lo).all()
    assert (hi <= q.mins + q.scales * q.levels + 1e-3).all()
    assert (lo >= q.mins - 1e-3).all()


def test_quantization_error_is_bounded_by_step_size(data):
    """Inside the trained range, error cannot exceed half a quantization step."""
    q = ScalarQuantizer(DIM, bits=6).train(data)
    inside = np.clip(data, q.mins, q.mins + q.scales * q.levels)
    err = np.abs(q.decode(q.encode(inside)) - inside)
    assert (err <= q.scales / 2 + 1e-5).all()


def test_encode_of_empty_batch_returns_empty_codes(data):
    empty = np.empty((0, DIM), dtype=np.float32)
    for q in _quantizers():
        q.train(data)
        assert q.encode(empty).shape[0] == 0

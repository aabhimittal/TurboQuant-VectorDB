"""AnisotropicPQ: the score-aware loss must trade the right things away.

The claim under test is deliberately counter-intuitive -- this quantizer
should get *worse* at reconstruction while getting much better at
inner-product ranking. A test suite that only checked "recall goes up"
would pass for a quantizer that was simply better at everything, which is
not what was built.
"""

import numpy as np
import pytest

from turboquant import (
    AnisotropicPQ,
    ProductQuantizer,
    normalize,
    recall_at_k,
    top_k_max,
    train_base_query_split,
)

DIM = 64
K = 10


@pytest.fixture(scope="module")
def split():
    return train_base_query_split(3000, 5000, 50, DIM, n_clusters=32, seed=5)


@pytest.fixture(scope="module")
def mips_ground_truth(split):
    _, base, queries = split
    ids, _ = top_k_max(queries @ base.T, K)
    return ids


def _mips_recall(quantizer, base, queries, truth):
    codes = quantizer.encode(base)
    scores = quantizer.adc_distances(quantizer.compute_ip_lut(queries), codes)
    return recall_at_k(top_k_max(scores, K)[0], truth, K)


def test_eta_one_reproduces_plain_pq(split):
    """eta = 1 must be *exactly* PQ -- it is the identity of the knob."""
    train, base, _ = split
    a = AnisotropicPQ(DIM, n_subspaces=8, eta=1.0).train(train, seed=3)
    p = ProductQuantizer(DIM, n_subspaces=8).train(train, seed=3)
    assert np.array_equal(a.codebooks, p.codebooks)
    assert np.array_equal(a.encode(base), p.encode(base))


def test_beats_plain_pq_on_inner_product_search(split, mips_ground_truth):
    train, base, queries = split
    plain = ProductQuantizer(DIM, n_subspaces=8).train(train)
    aniso = AnisotropicPQ(DIM, n_subspaces=8, eta=8.0).train(train)

    r_plain = _mips_recall(plain, base, queries, mips_ground_truth)
    r_aniso = _mips_recall(aniso, base, queries, mips_ground_truth)

    assert aniso.bytes_per_vector == plain.bytes_per_vector
    # Measured ~4.5x at 128 dims; a conservative floor for the smaller case.
    assert r_aniso > r_plain * 1.5


def test_mips_recall_rises_with_eta(split, mips_ground_truth):
    train, base, queries = split
    recalls = [
        _mips_recall(
            AnisotropicPQ(DIM, n_subspaces=8, eta=eta).train(train),
            base,
            queries,
            mips_ground_truth,
        )
        for eta in (1.0, 4.0, 16.0)
    ]
    assert recalls[0] < recalls[1] < recalls[2]


def test_reconstruction_gets_worse_as_scores_get_better(split, mips_ground_truth):
    """The trade-off is the thesis; assert it explicitly.

    If reconstruction error ever *improved* alongside MIPS recall, the
    implementation would not be doing what the docstring says -- it would
    just be a better-tuned PQ, and the anisotropic weighting would be
    unproven.
    """
    train, base, queries = split

    def mse(q):
        return float(((base - q.decode(q.encode(base))) ** 2).sum(axis=1).mean())

    low = AnisotropicPQ(DIM, n_subspaces=8, eta=1.0).train(train)
    high = AnisotropicPQ(DIM, n_subspaces=8, eta=16.0).train(train)

    assert mse(high) > mse(low)
    assert _mips_recall(high, base, queries, mips_ground_truth) > _mips_recall(
        low, base, queries, mips_ground_truth
    )


def test_gain_disappears_on_unit_normalized_data(split):
    """The documented null result, pinned so it cannot silently change.

    On the unit sphere every norm is 1, MIPS collapses to cosine, and the
    parallel direction stops being privileged -- so the large gain seen on
    raw vectors should evaporate. The comparison is between *ratios* on
    the two datasets rather than an absolute bound on the normalized one,
    because a small-sample absolute bound would be measuring noise: the
    claim is "the effect is specific to varying norms", and that is what
    this asserts.
    """
    train, base, queries = split
    raw_truth, _ = top_k_max(queries @ base.T, K)

    def ratio(tr, bs, qs, truth):
        plain = ProductQuantizer(DIM, n_subspaces=8).train(tr)
        aniso = AnisotropicPQ(DIM, n_subspaces=8, eta=8.0).train(tr)
        return _mips_recall(aniso, bs, qs, truth) / _mips_recall(
            plain, bs, qs, truth
        )

    raw_ratio = ratio(train, base, queries, raw_truth)
    n_train, n_base, n_queries = normalize(train), normalize(base), normalize(queries)
    norm_truth, _ = top_k_max(n_queries @ n_base.T, K)
    norm_ratio = ratio(n_train, n_base, n_queries, norm_truth)

    assert raw_ratio > 1.5
    assert norm_ratio < 1.25
    assert raw_ratio > norm_ratio * 1.5


def test_ip_lut_matches_direct_inner_product(split):
    """The MIPS lookup table must agree with decoding and dotting."""
    train, base, queries = split
    q = AnisotropicPQ(DIM, n_subspaces=8).train(train)
    codes = q.encode(base[:200])
    approx = q.adc_distances(q.compute_ip_lut(queries[:4]), codes)
    exact = queries[:4] @ q.decode(codes).T
    assert np.allclose(approx, exact, rtol=1e-3, atol=1e-3)


def test_anisotropic_encode_beats_nearest_centroid_encode(split, mips_ground_truth):
    """Codes should be chosen under the same loss the codebooks were.

    `encode` runs coordinate descent; `ProductQuantizer.encode` on the
    same codebooks is plain nearest-centroid. The former must not be worse.
    """
    train, base, queries = split
    q = AnisotropicPQ(DIM, n_subspaces=8, eta=8.0).train(train)

    smart = q.encode(base)
    naive = ProductQuantizer.encode(q, base)
    truth = mips_ground_truth

    def recall(codes):
        scores = q.adc_distances(q.compute_ip_lut(queries), codes)
        return recall_at_k(top_k_max(scores, K)[0], truth, K)

    assert not np.array_equal(smart, naive)
    assert recall(smart) >= recall(naive)


def test_rejects_invalid_eta():
    with pytest.raises(ValueError, match="eta"):
        AnisotropicPQ(DIM, eta=0.5)
    with pytest.raises(ValueError, match="eta"):
        AnisotropicPQ(DIM, eta=np.nan)

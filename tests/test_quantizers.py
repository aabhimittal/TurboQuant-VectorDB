"""Unit tests for quantizers: roundtrip fidelity, packing, memory accounting."""

import numpy as np
import pytest

from turboquant import (
    AdaptiveBitQuantizer,
    ProductQuantizer,
    ScalarQuantizer,
    clustered_embeddings,
)
from turboquant.quantizers.scalar import _pack_codes, _unpack_codes

DIM = 32
RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def data():
    return clustered_embeddings(2000, DIM, n_clusters=8, seed=1)


# --------------------------------------------------------------- bit packing
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_pack_unpack_roundtrip(bits):
    codes = RNG.integers(0, 1 << bits, size=(50, DIM)).astype(np.uint8)
    packed = _pack_codes(codes, bits)
    assert packed.shape[1] == int(np.ceil(DIM * bits / 8))
    np.testing.assert_array_equal(_unpack_codes(packed, bits, DIM), codes)


# ------------------------------------------------------------------- scalar
def test_scalar_untrained_raises():
    with pytest.raises(RuntimeError):
        ScalarQuantizer(DIM).encode(np.zeros((1, DIM), dtype=np.float32))


@pytest.mark.parametrize("bits,max_ratio_err", [(8, 0.01), (4, 0.15)])
def test_scalar_roundtrip_error(data, bits, max_ratio_err):
    sq = ScalarQuantizer(DIM, bits=bits).train(data)
    recon = sq.decode(sq.encode(data))
    # Relative reconstruction error should shrink with more bits.
    rel = np.linalg.norm(recon - data) / np.linalg.norm(data)
    assert rel < max_ratio_err


def test_scalar_compression_ratio(data):
    sq8 = ScalarQuantizer(DIM, bits=8).train(data)
    sq4 = ScalarQuantizer(DIM, bits=4).train(data)
    assert sq8.compression_ratio() == pytest.approx(4.0)
    assert sq4.compression_ratio() == pytest.approx(8.0)
    # Encoded arrays must actually be that small.
    codes = sq4.encode(data)
    assert codes.nbytes == data.shape[0] * DIM // 2


# ----------------------------------------------------------------- adaptive
def test_adaptive_bit_budget_respected(data):
    aq = AdaptiveBitQuantizer(DIM, avg_bits=4.0).train(data)
    assert aq.bits_per_dim.sum() == aq.total_bits == 4 * DIM
    assert aq.bits_per_dim.max() <= 8
    codes = aq.encode(data)
    assert codes.shape[1] == int(np.ceil(4 * DIM / 8))


def test_adaptive_allocates_by_variance(data):
    aq = AdaptiveBitQuantizer(DIM, avg_bits=2.0).train(data)
    spans = np.percentile(data, 99.9, axis=0) - np.percentile(data, 0.1, axis=0)
    # The widest dimension must get at least as many bits as the narrowest.
    assert aq.bits_per_dim[np.argmax(spans)] >= aq.bits_per_dim[np.argmin(spans)]
    # With decaying variance, allocation must be non-uniform.
    assert aq.bits_per_dim.max() > aq.bits_per_dim.min()


def test_adaptive_beats_uniform_at_same_budget(data):
    """The headline property: adaptive bits reconstruct better than uniform
    SQ at the exact same storage cost."""
    aq = AdaptiveBitQuantizer(DIM, avg_bits=4.0).train(data)
    sq = ScalarQuantizer(DIM, bits=4).train(data)
    assert aq.bytes_per_vector == sq.bytes_per_vector
    err_a = np.linalg.norm(aq.decode(aq.encode(data)) - data)
    err_s = np.linalg.norm(sq.decode(sq.encode(data)) - data)
    assert err_a < err_s


def test_adaptive_roundtrip_zero_bit_dims():
    """Dims that get 0 bits must reconstruct to the mean, not garbage."""
    rng = np.random.default_rng(0)
    x = np.hstack(
        [
            rng.standard_normal((500, 4)).astype(np.float32) * 10,  # wide dims
            np.full((500, 4), 3.14, dtype=np.float32),  # constant dims
        ]
    )
    aq = AdaptiveBitQuantizer(8, avg_bits=2.0).train(x)
    assert aq.bits_per_dim[4:].sum() == 0  # constant dims get nothing
    recon = aq.decode(aq.encode(x))
    np.testing.assert_allclose(recon[:, 4:], 3.14, atol=1e-3)


# ------------------------------------------------------------------ product
def test_pq_requires_divisible_dim():
    with pytest.raises(ValueError):
        ProductQuantizer(30, n_subspaces=8)


def test_pq_encode_decode_shapes(data):
    pq = ProductQuantizer(DIM, n_subspaces=4).train(data, n_iters=10)
    codes = pq.encode(data)
    assert codes.shape == (data.shape[0], 4)
    assert codes.dtype == np.uint8
    assert pq.decode(codes).shape == data.shape
    assert pq.compression_ratio() == pytest.approx(DIM * 4 / 4)


def test_pq_adc_matches_decoded_distances(data):
    """ADC via LUT must equal the distance to the decoded reconstruction --
    they are algebraically the same quantity computed two ways."""
    pq = ProductQuantizer(DIM, n_subspaces=4).train(data, n_iters=10)
    codes = pq.encode(data[:200])
    queries = data[500:505]
    lut = pq.compute_lut(queries)
    adc = pq.adc_distances(lut, codes)
    recon = pq.decode(codes)
    exact = ((queries[:, None, :] - recon[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(adc, exact, rtol=1e-4, atol=1e-3)


def test_pq_reconstruction_improves_with_m(data):
    errs = []
    for m in (2, 4, 8):
        pq = ProductQuantizer(DIM, n_subspaces=m).train(data, n_iters=10)
        recon = pq.decode(pq.encode(data))
        errs.append(np.linalg.norm(recon - data))
    assert errs[0] > errs[1] > errs[2]

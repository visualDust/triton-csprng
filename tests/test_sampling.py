import pytest
import torch

from triton_csprng import (
    ChaCha20Rng,
    bounded_uint64,
    discrete_gaussian,
    stochastic_round,
)
from triton_csprng.sampling import (
    _half_plane_cdt_threshold_words,
    bounded_uint64_two_streams,
    discrete_gaussian_two_streams,
)


def _bounded_multiply_high_cpu(words, bounds):
    rows = words.cpu().reshape(-1, 4).tolist()
    out = []
    for row, bound in zip(rows, bounds, strict=True):
        x_low = (int(row[0]) << 32) | int(row[1])
        x_high = (int(row[2]) << 32) | int(row[3])
        u128 = x_low | (x_high << 64)
        out.append((int(bound) * u128) >> 128)
    return torch.tensor(out, dtype=torch.int64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bounded_uint64_scalar_and_channel_bounds():
    rng = ChaCha20Rng(key=list(range(8)), nonce=[1, 2], device="cuda:0")
    scalar = bounded_uint64(rng, 17, (4, 64))
    assert scalar.shape == (4, 64)
    assert scalar.dtype is torch.int64
    assert int(scalar.min()) >= 0
    assert int(scalar.max()) < 17

    bounds = torch.tensor([3, 5, 97], dtype=torch.uint64, device="cuda:0")
    per_channel = bounded_uint64(rng, bounds, (3, 128))
    assert per_channel.shape == (3, 128)
    for row, bound in enumerate([3, 5, 97]):
        assert int(per_channel[row].min()) >= 0
        assert int(per_channel[row].max()) < bound


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bounded_uint64_matches_multiply_high_mapping():
    bounds = [2, 3, 17, (1 << 31) - 1, (1 << 60) + 93]
    rng_words = ChaCha20Rng(key=list(range(8)), nonce=[11, 12], device="cuda:0")
    words = rng_words.uint32(len(bounds) * 4)
    expected = _bounded_multiply_high_cpu(words, bounds)

    rng_sample = ChaCha20Rng(
        key=list(range(8)), nonce=[11, 12], device="cuda:0"
    )
    got = bounded_uint64(rng_sample, bounds, (len(bounds),)).cpu()
    assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bounded_uint64_two_streams_matches_separate_streams():
    bounds = [3, 17, 5, 257]
    first_expected = ChaCha20Rng(
        key=list(range(8)), nonce=[21, 22], device="cuda:0"
    )
    second_expected = ChaCha20Rng(
        key=list(range(8)), nonce=[23, 24], device="cuda:0"
    )
    expected = torch.cat(
        [
            bounded_uint64(first_expected, bounds[:2], (2, 64)),
            bounded_uint64(second_expected, bounds[2:], (2, 64)),
        ],
        dim=0,
    )

    first = ChaCha20Rng(key=list(range(8)), nonce=[21, 22], device="cuda:0")
    second = ChaCha20Rng(key=list(range(8)), nonce=[23, 24], device="cuda:0")
    got = bounded_uint64_two_streams(
        first,
        second,
        bounds,
        (4, 64),
        first_channels=2,
    )
    assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bounded_uint64_state_restore_and_distribution_sanity():
    rng = ChaCha20Rng(key=list(range(8)), nonce=[13, 14], device="cuda:0")
    first = rng.randint(3, (30_000,))
    state = rng.state_dict()
    second = rng.randint(3, (1024,))
    restored = ChaCha20Rng.from_state_dict(state)
    assert torch.equal(second, restored.randint(3, (1024,)))

    counts = torch.bincount(first.cpu(), minlength=3).float()
    expected = first.numel() / 3.0
    assert torch.max(torch.abs(counts - expected)).item() < expected * 0.04


def test_half_plane_cdt_threshold_shape_and_range():
    lows, highs, depth = _half_plane_cdt_threshold_words(3.2)
    assert depth == 5
    assert len(lows) == 31
    assert len(highs) == 31
    assert all(0 <= v < 2**64 for v in lows)
    assert all(0 <= v < 2**64 for v in highs)
    assert highs[-1] < 2**63
    full = [(hi << 64) | lo for lo, hi in zip(lows, highs, strict=True)]
    assert full == sorted(full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_discrete_gaussian_shape_and_rough_moments():
    rng = ChaCha20Rng(key=list(range(8)), nonce=[3, 4], device="cuda:0")
    samples = discrete_gaussian(rng, (16384,), sigma=3.2)
    assert samples.shape == (16384,)
    assert samples.dtype is torch.int64
    mean = float(samples.float().mean().cpu())
    std = float(samples.float().std().cpu())
    assert abs(mean) < 0.25
    assert 2.5 < std < 3.8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_discrete_gaussian_symmetry_and_tail_bucket():
    rng = ChaCha20Rng(key=list(range(8)), nonce=[15, 16], device="cuda:0")
    samples = rng.discrete_gaussian((50_000,), sigma=3.2).cpu()
    assert int(samples.abs().max()) <= 31
    positive = int((samples > 0).sum())
    negative = int((samples < 0).sum())
    nonzero = positive + negative
    assert nonzero > 0
    assert abs(positive - negative) / nonzero < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_discrete_gaussian_two_streams_matches_separate_streams():
    first_expected = ChaCha20Rng(
        key=list(range(8)), nonce=[31, 32], device="cuda:0"
    )
    second_expected = ChaCha20Rng(
        key=list(range(8)), nonce=[33, 34], device="cuda:0"
    )
    expected = torch.cat(
        [
            discrete_gaussian(first_expected, (2, 64)),
            discrete_gaussian(second_expected, (3, 64)),
        ],
        dim=0,
    )

    first = ChaCha20Rng(key=list(range(8)), nonce=[31, 32], device="cuda:0")
    second = ChaCha20Rng(key=list(range(8)), nonce=[33, 34], device="cuda:0")
    got = discrete_gaussian_two_streams(
        first,
        second,
        (5, 64),
        first_channels=2,
    )
    assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stochastic_round_integral_values_and_determinism():
    values = torch.tensor(
        [-2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0],
        dtype=torch.float64,
        device="cuda:0",
    )
    rng1 = ChaCha20Rng(key=list(range(8)), nonce=[5, 6], device="cuda:0")
    rng2 = ChaCha20Rng(key=list(range(8)), nonce=[5, 6], device="cuda:0")
    rounded1 = stochastic_round(rng1, values)
    rounded2 = stochastic_round(rng2, values)
    assert torch.equal(rounded1, rounded2)
    assert rounded1[0].item() == -2
    assert rounded1[3].item() == 0
    assert rounded1[-1].item() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stream_distribution_methods():
    rng = ChaCha20Rng(key=list(range(8)), nonce=[7, 8], device="cuda:0")
    assert rng.randint([11, 13], (2, 16)).shape == (2, 16)
    assert rng.discrete_gaussian((2, 16)).shape == (2, 16)
    values = torch.linspace(-1.0, 1.0, 16, device="cuda:0")
    assert rng.stochastic_round(values).shape == (16,)

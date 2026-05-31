import pytest
import torch

from triton_csprng import RnsRandomStreams


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two CUDA devices"
)
def test_rns_random_streams_repeated_uint32_channels_match_across_devices():
    streams = RnsRandomStreams(
        num_coeffs=32,
        channel_counts=[2, 3],
        repeated_channels=1,
        devices=["cuda:0", "cuda:1"],
        key=list(range(8)),
        nonce=[10, 20],
    )
    out = streams.uint32_channels()
    assert out[0].shape == (3, 32)
    assert out[1].shape == (4, 32)
    assert torch.equal(out[0][-1].cpu(), out[1][-1].cpu())
    assert not torch.equal(out[0][0].cpu(), out[1][0].cpu())


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two CUDA devices"
)
def test_rns_random_streams_repeated_distribution_channels_match():
    streams = RnsRandomStreams(
        num_coeffs=32,
        channel_counts=[1, 1],
        repeated_channels=1,
        devices=["cuda:0", "cuda:1"],
        key=list(range(8)),
        nonce=[30, 40],
    )
    ints = streams.randint_channels([[17, 257], [19, 257]])
    assert ints[0].shape == (2, 32)
    assert ints[1].shape == (2, 32)
    assert int(ints[0][0].max()) < 17
    assert int(ints[1][0].max()) < 19
    assert torch.equal(ints[0][-1].cpu(), ints[1][-1].cpu())

    gauss = streams.discrete_gaussian_channels()
    assert gauss[0].shape == (2, 32)
    assert gauss[1].shape == (2, 32)
    assert torch.equal(gauss[0][-1].cpu(), gauss[1][-1].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rns_random_streams_state_roundtrip_single_device():
    streams = RnsRandomStreams(
        num_coeffs=16,
        channel_counts=[2],
        repeated_channels=1,
        devices=["cuda:0"],
        key=list(range(8)),
        nonce=[50, 60],
    )
    _ = streams.uint32_channels()
    restored = RnsRandomStreams.from_state_dict(streams.state_dict())
    got = streams.uint32_channels()
    expected = restored.uint32_channels()
    assert torch.equal(got[0], expected[0])

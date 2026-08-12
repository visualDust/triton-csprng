import pytest
import torch

from triton_csprng import ChaCha20Rng


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stream_is_deterministic_from_same_key_nonce_counter():
    key = list(range(8))
    nonce = [100, 200]
    rng1 = ChaCha20Rng(key=key, nonce=nonce, counter=9, device="cuda:0")
    rng2 = ChaCha20Rng(key=key, nonce=nonce, counter=9, device="cuda:0")

    assert torch.equal(rng1.uint32((3, 7)), rng2.uint32((3, 7)))
    assert rng1.counter == rng2.counter == 11


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stream_chunking_matches_block_order():
    key = list(range(8))
    nonce = [5, 6]
    one_shot = ChaCha20Rng(key=key, nonce=nonce, counter=0, device="cuda:0")
    chunked = ChaCha20Rng(key=key, nonce=nonce, counter=0, device="cuda:0")

    expected = one_shot.uint32(40)
    got = torch.cat([chunked.uint32(17), chunked.uint32(23)])
    assert torch.equal(got, expected)
    assert one_shot.counter == chunked.counter == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stream_bytes_shape_and_state_roundtrip():
    key = bytes(range(32))
    nonce = bytes(range(8))
    rng = ChaCha20Rng(key=key, nonce=nonce, counter=3, device="cuda:0")
    first = rng.bytes((5, 13))
    assert first.shape == (5, 13)
    assert first.dtype is torch.uint8

    restored = ChaCha20Rng.from_state_dict(rng.state_dict())
    assert torch.equal(rng.bytes(70), restored.bytes(70))


def test_cpu_stream_matches_cuda_stream_when_available():
    key = list(range(8))
    nonce = [100, 200]
    cpu = ChaCha20Rng(key=key, nonce=nonce, counter=9, device="cpu")
    expected = cpu.uint32((3, 7))
    if torch.cuda.is_available():
        cuda = ChaCha20Rng(key=key, nonce=nonce, counter=9, device="cuda:0")
        assert torch.equal(expected, cuda.uint32((3, 7)).cpu())


def test_cpu_stream_chunking_and_state_roundtrip():
    key = list(range(8))
    nonce = [5, 6]
    one_shot = ChaCha20Rng(key=key, nonce=nonce, device="cpu")
    chunked = ChaCha20Rng(key=key, nonce=nonce, device="cpu")
    expected = one_shot.uint32(40)
    got = torch.cat([chunked.uint32(17), chunked.uint32(23)])
    assert torch.equal(got, expected)
    restored = ChaCha20Rng.from_state_dict(chunked.state_dict())
    assert torch.equal(chunked.bytes(70), restored.bytes(70))


def test_stream_rejects_counter_wraparound():
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        ChaCha20Rng(counter=-1, device="cpu")
    rng = ChaCha20Rng(counter=(1 << 64) - 1, device="cpu")
    assert rng.blocks(1).shape == (1, 16)
    with pytest.raises(OverflowError, match="counter space"):
        rng.blocks(1)

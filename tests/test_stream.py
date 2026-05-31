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

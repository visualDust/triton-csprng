import pytest
import torch

from triton_csprng import chacha20_blocks, make_chacha20_state


def _rotl32(v: int, n: int) -> int:
    return ((v << n) | (v >> (32 - n))) & 0xFFFFFFFF


def _quarter_round(state, a, b, c, d):
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] = _rotl32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotl32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] = _rotl32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotl32(state[b] ^ state[c], 7)


def _chacha20_reference(block):
    x = [int(v) for v in block]
    s = list(x)
    for _ in range(10):
        _quarter_round(x, 0, 4, 8, 12)
        _quarter_round(x, 1, 5, 9, 13)
        _quarter_round(x, 2, 6, 10, 14)
        _quarter_round(x, 3, 7, 11, 15)
        _quarter_round(x, 0, 5, 10, 15)
        _quarter_round(x, 1, 6, 11, 12)
        _quarter_round(x, 2, 7, 8, 13)
        _quarter_round(x, 3, 4, 9, 14)
    return [(a + b) & 0xFFFFFFFF for a, b in zip(x, s)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chacha20_blocks_matches_reference():
    key = [
        0x03020100,
        0x07060504,
        0x0B0A0908,
        0x0F0E0D0C,
        0x13121110,
        0x17161514,
        0x1B1A1918,
        0x1F1E1D1C,
    ]
    nonce = [0x09000000, 0x4A000000]
    state = make_chacha20_state(
        num_blocks=3, key=key, nonce=nonce, counter=1, device="cuda:0"
    )
    original = state.clone()
    out = chacha20_blocks(state, step=7)
    got = out.cpu().tolist()
    expected = [_chacha20_reference(row) for row in original.cpu().tolist()]
    assert got == expected
    assert state[:, 12].cpu().tolist() == [8, 9, 10]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chacha20_blocks_supports_non_multiple_block_counts():
    key = list(range(8))
    nonce = [123, 456]
    state = make_chacha20_state(
        num_blocks=130, key=key, nonce=nonce, counter=0, device="cuda:0"
    )
    out = chacha20_blocks(state, step=0, block_size=128)
    assert out.shape == (130, 16)
    assert out.dtype is torch.uint32
